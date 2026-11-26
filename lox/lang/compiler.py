"""
Policy & Skill Compiler.
Compiles `policy.py` + `skills/` with `PolicyInstrumentTransformer` into an
executable `CompiledPolicy` factory that instantiates fast per-episode `PolicyRunner` objects.
"""

from __future__ import annotations
import __future__ as _future_mod

import ast
import bisect
import collections
import copy
import dataclasses
import enum
import functools
import heapq
import importlib
import inspect
import itertools
import math
import operator
import random
import re
import sys
import threading
import types
import typing
from typing import Any

import numpy as np

from lox.env.actions import (
    ACTION_BY_CHAR,
    ACTION_BY_NAME,
    DIR_DELTAS,
    Action,
    ActionSequence,
    RawAction,
    extcmd,
    key,
    repeat,
    seq,
)
from lox.env.state import (
    FloorMemory,
    GameState,
    InventoryItem,
    MonsterInfo,
    ObjectInfo,
    VisibleMonster,
    VisibleObject,
    WorldMemory,
    astar_next_step,
    bfs_distance_grid,
)
from lox.lang.contracts import (
    CapabilityScorer,
    GoalIntent,
    SafetyReport,
    ThreatEvaluator,
    WorldTracker,
)
from lox.lang.contracts_check import check_undefined_symbols
from lox.lang.instrument import (
    ExecutionTracer,
    PolicyInstrumentTransformer,
    PolicyLoopError,
    _TracerProxy,
)
from lox.lang.sandbox import (
    _STRICT_FORBIDDEN_ATTRS,
    ALLOWED_DUNDERS,
    ALLOWED_LOX_SUBMODULES,
    FORBIDDEN_DUNDERS,
    FORBIDDEN_IMPORT_NAMES,
    FORBIDDEN_MODULE_ATTRS,
    SAFE_STDLIB_SUBMODULES,
    PolicyValidationError,
    normalize_skill_key,
    skill_key_to_relpath,
    validate_python_code,
)

_COMPILE_EXEC_LOCK = threading.RLock()


def _is_forbidden_attr(obj: Any, name: Any) -> bool:
    if not isinstance(name, str):
        return False
    if name in FORBIDDEN_DUNDERS or (
        name.startswith("__") and name not in ALLOWED_DUNDERS
    ):
        return True
    if name in _STRICT_FORBIDDEN_ATTRS:
        return True
    if isinstance(obj, types.ModuleType):
        mod_root = str(getattr(obj, "__name__", "") or "").split(".")[0]
        if (
            name.startswith("_")
            or name in FORBIDDEN_IMPORT_NAMES
            or name in FORBIDDEN_MODULE_ATTRS.get(mod_root, frozenset())
        ):
            return True
    return False


def _safe_getattr(obj: Any, name: str, *default: Any) -> Any:
    if _is_forbidden_attr(obj, name):
        raise AttributeError(
            f"Access to attribute '{name}' is forbidden in policy sandbox."
        )
    if (
        isinstance(obj, (_TracerProxy, ExecutionTracer))
        and name not in _TracerProxy.__slots__
    ):
        raise AttributeError(
            f"Access to internal tracer attribute '{name}' is forbidden in policy sandbox."
        )
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    if _is_forbidden_attr(obj, name) or isinstance(
        obj, (types.ModuleType, _TracerProxy, ExecutionTracer)
    ):
        raise AttributeError(
            f"Setting attribute '{name}' is forbidden in policy sandbox."
        )
    setattr(obj, name, value)


def _safe_hasattr(obj: Any, name: str) -> bool:
    if _is_forbidden_attr(obj, name):
        return False
    if (
        isinstance(obj, (_TracerProxy, ExecutionTracer))
        and name not in _TracerProxy.__slots__
    ):
        return False
    return hasattr(obj, name)


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "classmethod": classmethod,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": _safe_getattr,
    "hasattr": _safe_hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": _safe_setattr,
    "slice": slice,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "LookupError": LookupError,
    "True": True,
    "False": False,
    "None": None,
}


def _build_safe_lox_modules() -> dict[str, types.ModuleType]:
    """Constructs sanitized, minimal proxy modules for the explicit `lox.*` allowlist."""
    lox_root = types.ModuleType("lox")
    lox_env = types.ModuleType("lox.env")
    lox_lang = types.ModuleType("lox.lang")

    actions_mod = types.ModuleType("lox.env.actions")
    for k, v in (
        ("Action", Action),
        ("RawAction", RawAction),
        ("ActionSequence", ActionSequence),
        ("DIR_DELTAS", DIR_DELTAS),
        ("ACTION_BY_NAME", ACTION_BY_NAME),
        ("ACTION_BY_CHAR", ACTION_BY_CHAR),
        ("key", key),
        ("seq", seq),
        ("repeat", repeat),
        ("extcmd", extcmd),
    ):
        setattr(actions_mod, k, v)
    # Include optional safe direction helpers if present on lox.env.actions
    from lox.env import actions as _real_actions

    for opt_k in (
        "OPPOSITE_DIR",
        "CARDINAL_DELTAS",
        "menu_pick",
        "delta_to_dir",
        "dir_to_action",
    ):
        if hasattr(_real_actions, opt_k):
            setattr(actions_mod, opt_k, getattr(_real_actions, opt_k))

    state_mod = types.ModuleType("lox.env.state")
    for k, v in (
        ("GameState", GameState),
        ("WorldMemory", WorldMemory),
        ("FloorMemory", FloorMemory),
        ("InventoryItem", InventoryItem),
        ("MonsterInfo", MonsterInfo),
        ("ObjectInfo", ObjectInfo),
        ("VisibleMonster", VisibleMonster),
        ("VisibleObject", VisibleObject),
        ("bfs_distance_grid", bfs_distance_grid),
        ("astar_next_step", astar_next_step),
    ):
        setattr(state_mod, k, v)

    contracts_mod = types.ModuleType("lox.lang.contracts")
    for k, v in (
        ("GoalIntent", GoalIntent),
        ("SafetyReport", SafetyReport),
        ("WorldTracker", WorldTracker),
        ("CapabilityScorer", CapabilityScorer),
        ("ThreatEvaluator", ThreatEvaluator),
    ):
        setattr(contracts_mod, k, v)

    setattr(lox_env, "actions", actions_mod)
    setattr(lox_env, "state", state_mod)
    setattr(lox_lang, "contracts", contracts_mod)
    setattr(lox_root, "env", lox_env)
    setattr(lox_root, "lang", lox_lang)

    return {
        "lox": lox_root,
        "lox.env": lox_env,
        "lox.lang": lox_lang,
        "lox.env.actions": actions_mod,
        "lox.env.state": state_mod,
        "lox.lang.contracts": contracts_mod,
    }


_SAFE_LOX_MODULES = _build_safe_lox_modules()


def _make_safe_module_proxy(
    mod_name: str,
    raw_mod: types.ModuleType,
    safe_submods: tuple[str, ...] = (),
) -> types.ModuleType:
    """Builds a sanitized module proxy stripping all private (`_*`), forbidden, and submodule attributes."""
    proxy = types.ModuleType(mod_name)
    root_name = mod_name.split(".")[0]
    mod_denied = FORBIDDEN_MODULE_ATTRS.get(root_name, frozenset())
    for attr_name in dir(raw_mod):
        if (
            attr_name.startswith("_")
            or attr_name in FORBIDDEN_IMPORT_NAMES
            or attr_name in mod_denied
        ):
            continue
        try:
            val = getattr(raw_mod, attr_name)
        except Exception:
            continue
        if isinstance(val, types.ModuleType):
            continue
        setattr(proxy, attr_name, val)
    for sub_name in safe_submods:
        sub_val = getattr(raw_mod, sub_name, None)
        if not isinstance(sub_val, types.ModuleType):
            try:
                sub_val = importlib.import_module(f"{mod_name}.{sub_name}")
            except ImportError:
                continue
        setattr(
            proxy,
            sub_name,
            _make_safe_module_proxy(f"{mod_name}.{sub_name}", sub_val),
        )
    return proxy


def _build_safe_stdlib_modules() -> dict[str, types.ModuleType]:
    raw_modules: dict[str, types.ModuleType] = {
        "math": math,
        "collections": collections,
        "heapq": heapq,
        "itertools": itertools,
        "functools": functools,
        "re": re,
        "enum": enum,
        "dataclasses": dataclasses,
        "typing": typing,
        "numpy": np,
        "random": random,
        "operator": operator,
        "bisect": bisect,
        "copy": copy,
    }
    proxies: dict[str, types.ModuleType] = {}
    for m_name, raw_mod in raw_modules.items():
        proxies[m_name] = _make_safe_module_proxy(
            m_name, raw_mod, SAFE_STDLIB_SUBMODULES.get(m_name, ())
        )
    proxies["np"] = proxies["numpy"]
    return proxies


_SAFE_STDLIB_MODULES: dict[str, types.ModuleType] = _build_safe_stdlib_modules()


def _make_scoped_import(scoped_modules: dict[str, types.ModuleType] | None = None):
    """Returns an isolated `__import__` bound strictly to safe stdlib, safe `lox.*` proxies, and this runner's `skills.*` dict."""
    scoped = scoped_modules if scoped_modules is not None else {}

    def _safe_import(
        name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0
    ):
        if level != 0:
            raise ImportError("Relative imports are not permitted in policy sandbox.")
        if name == "__future__":
            fl = tuple(fromlist or ())
            if any(x != "annotations" for x in fl):
                raise ImportError(f"Unsupported __future__ import: {fl}")
            return _future_mod

        root = name.split(".")[0]

        if root in _SAFE_STDLIB_MODULES:
            mod_denied = FORBIDDEN_MODULE_ATTRS.get(root, frozenset())
            for item in fromlist or ():
                s_item = str(item)
                if (
                    s_item.startswith("_")
                    or s_item in FORBIDDEN_IMPORT_NAMES
                    or s_item in mod_denied
                ):
                    raise ImportError(
                        f"Import of private/re-exported symbol '{item}' from '{name}' is forbidden."
                    )
            root_proxy = _SAFE_STDLIB_MODULES[root]
            if name == root:
                return root_proxy
            parts = name.split(".")
            if len(parts) != 2 or parts[1] not in SAFE_STDLIB_SUBMODULES.get(root, ()):
                raise ImportError(
                    f"Submodule import '{name}' is not permitted in policy sandbox."
                )
            return root_proxy if not fromlist else getattr(root_proxy, parts[1])

        if root == "lox":
            if name not in ALLOWED_LOX_SUBMODULES:
                raise ImportError(
                    f"Import of internal module '{name}' is not permitted in policy sandbox."
                )
            for item in fromlist or ():
                if str(item).startswith("_") or str(item) in FORBIDDEN_IMPORT_NAMES:
                    raise ImportError(
                        f"Import of private symbol '{item}' from '{name}' is forbidden."
                    )
            if not fromlist:
                return _SAFE_LOX_MODULES["lox"]
            return _SAFE_LOX_MODULES[name]

        if root == "skills":
            if name != "skills" and name not in scoped:
                avail_skills = sorted(k for k in scoped if k.startswith("skills."))
                raise ImportError(
                    f"Skill module '{name}' not found. Available skills: {avail_skills or 'none'}"
                )
            if not fromlist:
                if root in scoped:
                    return scoped[root]
                if name in scoped:
                    return scoped[name]
            else:
                if name in scoped:
                    return scoped[name]
                if root in scoped:
                    return scoped[root]
            raise ImportError(f"Skill module '{name}' not found.")

        raise ImportError(f"Import of '{name}' is not permitted in policy sandbox.")

    return _safe_import


_safe_import = _make_scoped_import(None)
SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS["__build_class__"] = __build_class__


def _build_base_namespace(
    tracer: ExecutionTracer,
    scoped_modules: dict[str, types.ModuleType] | None = None,
    module_name: str = "lox_policy",
) -> dict[str, Any]:
    """Builds the sandboxed execution globals with 1-to-1 actions, state utilities, read-only tracer proxy, and stdlib."""
    builtins_ns = dict(SAFE_BUILTINS)
    if scoped_modules is not None:
        builtins_ns["__import__"] = _make_scoped_import(scoped_modules)
    tracer_proxy = tracer.make_proxy()
    ns: dict[str, Any] = {
        "__builtins__": builtins_ns,
        "__name__": module_name,
        "_lox_tracer": tracer_proxy,
        # Sanitized standard library module proxies pre-loaded for convenience
        "math": _SAFE_STDLIB_MODULES["math"],
        "collections": _SAFE_STDLIB_MODULES["collections"],
        "deque": collections.deque,
        "defaultdict": collections.defaultdict,
        "Counter": collections.Counter,
        "heapq": _SAFE_STDLIB_MODULES["heapq"],
        "itertools": _SAFE_STDLIB_MODULES["itertools"],
        "functools": _SAFE_STDLIB_MODULES["functools"],
        "re": _SAFE_STDLIB_MODULES["re"],
        "enum": _SAFE_STDLIB_MODULES["enum"],
        "dataclasses": _SAFE_STDLIB_MODULES["dataclasses"],
        "dataclass": dataclasses.dataclass,
        "typing": _SAFE_STDLIB_MODULES["typing"],
        "Any": Any,
        "np": _SAFE_STDLIB_MODULES["np"],
        "numpy": _SAFE_STDLIB_MODULES["numpy"],
        "random": _SAFE_STDLIB_MODULES["random"],
        "operator": _SAFE_STDLIB_MODULES["operator"],
        "bisect": _SAFE_STDLIB_MODULES["bisect"],
        "copy": _SAFE_STDLIB_MODULES["copy"],
        # 1-to-1 Action & State primitives
        "Action": Action,
        "RawAction": RawAction,
        "ActionSequence": ActionSequence,
        "DIR_DELTAS": DIR_DELTAS,
        "ACTION_BY_NAME": ACTION_BY_NAME,
        "ACTION_BY_CHAR": ACTION_BY_CHAR,
        "key": key,
        "seq": seq,
        "repeat": repeat,
        "extcmd": extcmd,
        "GameState": GameState,
        "WorldMemory": WorldMemory,
        "FloorMemory": FloorMemory,
        "InventoryItem": InventoryItem,
        "MonsterInfo": MonsterInfo,
        "ObjectInfo": ObjectInfo,
        "VisibleMonster": VisibleMonster,
        "VisibleObject": VisibleObject,
        "bfs_distance_grid": bfs_distance_grid,
        "astar_next_step": astar_next_step,
        # Contracts
        "GoalIntent": GoalIntent,
        "SafetyReport": SafetyReport,
        "WorldTracker": WorldTracker,
        "CapabilityScorer": CapabilityScorer,
        "ThreatEvaluator": ThreatEvaluator,
        # Global parameter helper `param("name", default, min_val, max_val)`
        "param": tracer.get_param,
    }
    return ns


def _exec_skills_into_namespace(
    ordered_skill_objs: dict[str, Any],
    ns: dict[str, Any],
    scoped_modules: dict[str, types.ModuleType],
) -> None:
    """Executes compiled skills (1-level and 2-level `domain/mod`) strictly in per-runner `scoped_modules` without mutating global `sys.modules`."""
    skills_pkg = types.ModuleType("skills")
    skills_pkg.__path__ = []
    scoped_modules["skills"] = skills_pkg
    ns["skills"] = skills_pkg

    for s_name, s_code_obj in ordered_skill_objs.items():
        parts = [p for p in str(s_name).split("/") if p]
        if not parts:
            continue
        mod_full = f"skills.{'.'.join(parts)}"
        if len(parts) == 2:
            pkg_full = f"skills.{parts[0]}"
            pkg_mod = scoped_modules.get(pkg_full)
            if pkg_mod is None:
                pkg_mod = types.ModuleType(pkg_full)
                pkg_mod.__path__ = []
                scoped_modules[pkg_full] = pkg_mod
                setattr(skills_pkg, parts[0], pkg_mod)
        s_mod = types.ModuleType(mod_full)
        s_mod.__dict__.update(ns)
        s_mod.__name__ = mod_full
        s_mod.__package__ = f"skills.{parts[0]}" if len(parts) == 2 else "skills"
        scoped_modules[mod_full] = s_mod
        # Temporarily bind in sys.modules ONLY during exec so `@dataclass` / `from __future__ import annotations`
        # resolves cls.__module__ cleanly, then restore immediately so global sys.modules stays unpolluted.
        prev_mod = sys.modules.get(mod_full)
        sys.modules[mod_full] = s_mod
        try:
            exec(s_code_obj, s_mod.__dict__)
        finally:
            if prev_mod is None:
                sys.modules.pop(mod_full, None)
            else:
                sys.modules[mod_full] = prev_mod
        if len(parts) == 2:
            setattr(scoped_modules[f"skills.{parts[0]}"], parts[1], s_mod)
        else:
            setattr(skills_pkg, parts[0], s_mod)


class PolicyRunner:
    """
    Per-episode driver for a compiled policy instance.
    Supports both:
    - Generator policies: `def run(self, state): while True: state = yield action`
    - Step policies: `def step(self, state) -> action` (including step methods returning subgenerators)
    """

    def __init__(self, policy_instance: Any, tracer: ExecutionTracer):
        self.instance = policy_instance
        self.tracer = tracer
        self._gen: Any = None
        self._active_subgen: Any = None
        self._started: bool = False
        self._mode: str = "step"
        self._text_prompt_ticks: int = 0
        self.auto_esc_count: int = 0
        self.subgen_preemptions: int = 0
        self.policy_exceptions: int = 0
        self.last_exception_msg: str = ""

        # Inject `self.param(...)` onto the policy instance if not already defined
        if not hasattr(self.instance, "param"):
            try:
                setattr(self.instance, "param", self.tracer.get_param)
            except Exception:
                pass

        if hasattr(self.instance, "run") and callable(self.instance.run):
            self._mode = "run"
        elif hasattr(self.instance, "step") and callable(self.instance.step):
            self._mode = "step"
        elif hasattr(self.instance, "act") and callable(self.instance.act):
            self._mode = "act"
        elif callable(self.instance):
            self._mode = "callable"
        else:
            raise PolicyValidationError(
                "Policy class must define `def run(self, state):` (generator) or `def step(self, state):`."
            )
        self._goal_holders_cache: tuple[Any, ...] | None = None

    def _iter_goal_holders(self) -> tuple[Any, ...]:
        if self._goal_holders_cache:
            return self._goal_holders_cache
        holders: list[Any] = []
        for obj in (
            self.instance,
            getattr(self.instance, "ws", None),
            getattr(self.instance, "world_state", None),
        ):
            if (
                obj is not None
                and (
                    hasattr(obj, "last_active_goal") or hasattr(obj, "last_active_leaf")
                )
                and obj not in holders
            ):
                holders.append(obj)
        # Only cache once non-empty so lazily initialized `self.ws` created during turn 1 is picked up
        if holders:
            self._goal_holders_cache = tuple(holders)
        return tuple(holders)

    def _clear_last_active_goal(self) -> None:
        for obj in self._iter_goal_holders():
            if hasattr(obj, "last_active_goal"):
                obj.last_active_goal = ""
            if hasattr(obj, "last_active_leaf"):
                obj.last_active_leaf = ""

    def _get_last_active_goal(self) -> str:
        for obj in self._iter_goal_holders():
            goal = str(getattr(obj, "last_active_goal", "") or "").strip()
            if goal:
                return goal
            leaf = str(getattr(obj, "last_active_leaf", "") or "").strip()
            if leaf:
                return leaf
        return ""

    def _close_subgen(self) -> None:
        sg = self._active_subgen
        self._active_subgen = None
        if sg is not None and hasattr(sg, "close"):
            try:
                sg.close()
            except Exception:
                pass

    def _unwind_nested_subgen(self, gen: Any) -> tuple[Any, bool]:
        """Advances a newly returned generator (unwrapping nested generators). Returns `(action, yielded_ok)`."""
        cur = gen
        depth = 0
        while inspect.isgenerator(cur) and depth < 8:
            depth += 1
            self._active_subgen = cur
            try:
                nxt = next(cur)
            except StopIteration as stop_exc:
                self._active_subgen = None
                if stop_exc.value is not None:
                    return stop_exc.value, True
                return None, False
            if inspect.isgenerator(nxt):
                cur = nxt
                continue
            return nxt, True
        return cur, True

    def next_action(self, state: GameState) -> tuple[Any, str]:
        """
        Advances the policy by one turn given `state`.
        Returns `(raw_yielded_action, active_function_name)`.
        """
        self.tracer.loop_ticks = 0
        self._clear_last_active_goal()
        try:
            setattr(self.instance, "state", state)
        except Exception:
            pass

        policy_preempt = False
        if hasattr(self.instance, "should_preempt_subgenerator"):
            try:
                policy_preempt = bool(self.instance.should_preempt_subgenerator(state))
            except Exception:
                policy_preempt = False

        if self._active_subgen is not None and policy_preempt:
            self._close_subgen()
            self.subgen_preemptions += 1

        if self._active_subgen is not None:
            try:
                try:
                    act = (
                        self._active_subgen.send(state)
                        if hasattr(self._active_subgen, "send")
                        else next(self._active_subgen)
                    )
                except TypeError:
                    act = next(self._active_subgen)
                if inspect.isgenerator(act):
                    act, got_act = self._unwind_nested_subgen(act)
                    if not got_act:
                        act = None
                if act is not None:
                    active_fn = self.tracer.record_turn_yield(
                        preferred_fn=self._get_last_active_goal()
                    )
                    return act, active_fn
            except StopIteration as stop_exc:
                self._active_subgen = None
                if stop_exc.value is not None:
                    active_fn = self.tracer.record_turn_yield(
                        preferred_fn=self._get_last_active_goal()
                    )
                    return stop_exc.value, active_fn

        try:
            if self._mode == "run":
                if not self._started:
                    self._started = True
                    res = self.instance.run(state)
                    if inspect.isgenerator(res):
                        self._gen = res
                        act = next(self._gen)
                    else:
                        self._mode = "run_direct"
                        act = res
                else:
                    act = self._gen.send(state)
            elif self._mode == "run_direct":
                act = self.instance.run(state)
            elif self._mode == "step":
                res = self.instance.step(state)
                if inspect.isgenerator(res):
                    # Keep self._mode == "step" and track generator in self._active_subgen
                    # so preemption and post-exhaustion re-entry into step() both work cleanly.
                    act, got_act = self._unwind_nested_subgen(res)
                    if not got_act:
                        act = Action.WAIT
                else:
                    act = res
            elif self._mode == "act":
                act = self.instance.act(state)
            else:
                act = self.instance(state)
        except StopIteration as stop_exc:
            if stop_exc.value is not None:
                act = stop_exc.value
                self._gen = None
                self._started = False
            elif self._mode == "run" and hasattr(self.instance, "run"):
                try:
                    res = self.instance.run(state)
                    if inspect.isgenerator(res):
                        self._gen = res
                        act = next(self._gen)
                    else:
                        self._mode = "run_direct"
                        act = res
                except StopIteration as stop_exc2:
                    act = (
                        stop_exc2.value if stop_exc2.value is not None else Action.WAIT
                    )
            else:
                act = Action.WAIT
        except PolicyLoopError:
            raise

        if inspect.isgenerator(act):
            act, got_act = self._unwind_nested_subgen(act)
            if not got_act:
                act = Action.WAIT

        active_fn = self.tracer.record_turn_yield(
            preferred_fn=self._get_last_active_goal()
        )
        return act, active_fn


def _extract_skill_dependencies(tree: ast.AST, known_skills: set[str]) -> set[str]:
    """Extracts intra-package `skills.<name>` or `skills.<domain>.<name>` dependencies via bundle SSOT."""
    from lox.lang.bundle import extract_skill_imports

    return extract_skill_imports(tree, known_skills).referenced_keys


def _topological_skill_order(skill_trees: dict[str, Any]) -> list[str]:
    """
    Orders skill modules so dependencies (e.g. `skills.world`) execute BEFORE
    importers (e.g. `skills.combat.core`), regardless of alphabetical name.
    """
    if len(skill_trees) <= 1:
        return list(skill_trees.keys())

    known = set(skill_trees.keys())
    deps_of: dict[str, set[str]] = {}
    dependents: dict[str, list[str]] = {k: [] for k in known}
    in_degree: dict[str, int] = {}

    for name, val in skill_trees.items():
        if isinstance(val, set):
            deps = (val & known) - {name}
        else:
            deps = _extract_skill_dependencies(val, known) - {name}
        deps_of[name] = deps
        in_degree[name] = len(deps)
        for d in deps:
            dependents[d].append(name)

    def _sort_key(name: str) -> tuple[int, str]:
        return (0 if name.startswith(("shared", "world")) else 1, name)

    ready: list[tuple[tuple[int, str], str]] = [
        (_sort_key(k), k) for k, deg in in_degree.items() if deg == 0
    ]
    heapq.heapify(ready)

    order: list[str] = []
    seen: set[str] = set()
    while ready:
        _, cur = heapq.heappop(ready)
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        for child in dependents[cur]:
            in_degree[child] -= 1
            if in_degree[child] == 0 and child not in seen:
                heapq.heappush(ready, (_sort_key(child), child))

    if len(order) < len(skill_trees):
        remaining = sorted((k for k in known if k not in seen), key=_sort_key)
        order.extend(remaining)
    return order


_SKILL_CACHE_LOCK = threading.RLock()
_POLICY_CACHE_LOCK = threading.RLock()


@functools.lru_cache(maxsize=128)
def _compile_skill_unit_cached(
    skill_name: str,
    skill_code: str,
    is_macro_module: bool,
) -> tuple[Any, list[str], frozenset[str]]:
    norm_name = normalize_skill_key(skill_name)
    rel_fn = skill_key_to_relpath(norm_name)
    skill_tree = validate_python_code(skill_code, filename=rel_fn)
    skill_tree = PolicyInstrumentTransformer(is_macro_module=is_macro_module).visit(
        skill_tree
    )
    ast.fix_missing_locations(skill_tree)
    code_obj = compile(skill_tree, filename=rel_fn, mode="exec")
    fn_names = [
        n.name
        for n in skill_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name != "__init__"
    ]
    raw_deps: set[str] = set()
    for node in ast.walk(skill_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "")
                if name.startswith("skills."):
                    tail = name[len("skills.") :].replace(".", "/")
                    raw_deps.add(tail)
                    raw_deps.add(tail.split("/", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            mod = str(node.module or "")
            if mod == "skills":
                for alias in node.names:
                    if alias.name != "*":
                        raw_deps.add(str(alias.name or "").split(".", 1)[0])
            elif mod.startswith("skills."):
                tail = mod[len("skills.") :].replace(".", "/")
                raw_deps.add(tail)
                raw_deps.add(tail.split("/", 1)[0])
                for alias in node.names:
                    if alias.name != "*":
                        raw_deps.add(f"{tail}/{alias.name}")
    return code_obj, fn_names, frozenset(raw_deps)


def _compile_skill_unit(
    skill_name: str,
    skill_code: str,
    is_macro_module: bool = True,
) -> tuple[Any, list[str], frozenset[str]]:
    norm_name = normalize_skill_key(skill_name)
    with _SKILL_CACHE_LOCK:
        return _compile_skill_unit_cached(norm_name, skill_code, is_macro_module)


def _expand_skill_deps(
    raw_deps: frozenset[str], known_skill_keys: set[str]
) -> set[str]:
    """Expands both exact keys and package-level dependencies (e.g. `combat` -> `combat/core`, `combat/ranged`)."""
    expanded: set[str] = set()
    for dep in raw_deps:
        if dep in known_skill_keys:
            expanded.add(dep)
        prefix = f"{dep}/"
        for k in known_skill_keys:
            if k.startswith(prefix):
                expanded.add(k)
    return expanded


class CompiledPolicy:
    """Compiled bytecode bundle for `policy.py` + `skills/` package."""

    def __init__(
        self,
        policy_code: str,
        macros_code: str = "",
        skills: dict[str, str] | None = None,
        max_loop_ticks: int = 250000,
    ):
        self.policy_code = policy_code
        self.macros_code = macros_code
        self.max_loop_ticks = max_loop_ticks
        norm_skills: dict[str, str] = {}
        for rk, rv in (skills or {}).items():
            nk = normalize_skill_key(rk)
            if nk:
                norm_skills[nk] = rv
        self.skills = norm_skills
        self._skill_code_objs: dict[str, Any] = {}
        self.skill_function_names: dict[str, list[str]] = {}
        self.macro_function_names: list[str] = []

        # Parse, validate, and instrument skills package in topological dependency order
        raw_skill_deps: dict[str, set[str]] = {}
        raw_skill_code_objs: dict[str, Any] = {}
        known_skill_keys = {k for k, v in self.skills.items() if v and v.strip()}
        for skill_name, skill_code in self.skills.items():
            if not skill_code or not skill_code.strip():
                continue
            c_obj, fn_names, deps_fs = _compile_skill_unit(
                skill_name, skill_code, is_macro_module=True
            )
            raw_skill_deps[skill_name] = _expand_skill_deps(
                deps_fs, known_skill_keys
            ) - {skill_name}
            raw_skill_code_objs[skill_name] = c_obj
            self.skill_function_names[skill_name] = list(fn_names)
        self._skill_order: list[str] = _topological_skill_order(raw_skill_deps)
        for skill_name in self._skill_order:
            self._skill_code_objs[skill_name] = raw_skill_code_objs[skill_name]

        # Fast (<2ms) CPython symtable undefined-symbol and cross-skill export gate
        allowed_globals = frozenset(_build_base_namespace(ExecutionTracer()).keys())
        check_undefined_symbols(
            policy_code=policy_code,
            skills=self.skills,
            allowed_globals=allowed_globals,
        )

        policy_tree = validate_python_code(policy_code, filename="policy.py")
        policy_tree = PolicyInstrumentTransformer(is_macro_module=False).visit(
            policy_tree
        )
        ast.fix_missing_locations(policy_tree)
        self._policy_code_obj = compile(policy_tree, filename="policy.py", mode="exec")

        self.policy_function_names: list[str] = [
            n.name
            for n in ast.walk(policy_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name != "__init__"
        ]
        for skill_fns in self.skill_function_names.values():
            for sfn in skill_fns:
                if sfn not in self.policy_function_names:
                    self.policy_function_names.append(sfn)

    def create_runner(
        self,
        param_overrides: dict[str, float] | None = None,
    ) -> PolicyRunner:
        with _COMPILE_EXEC_LOCK:
            tracer = ExecutionTracer(
                max_loop_ticks=self.max_loop_ticks,
                param_overrides=param_overrides,
            )
            scoped_modules: dict[str, types.ModuleType] = {}
            mod_name = "lox_policy"
            ns = _build_base_namespace(
                tracer, scoped_modules=scoped_modules, module_name=mod_name
            )
            mod_obj = types.ModuleType(mod_name)
            mod_obj.__dict__.update(ns)

            prev_pol_mod = sys.modules.get(mod_name)
            sys.modules[mod_name] = mod_obj
            try:
                _exec_skills_into_namespace(
                    self._skill_code_objs, ns, scoped_modules=scoped_modules
                )
                exec(self._policy_code_obj, ns)
                mod_obj.__dict__.update(ns)

                policy_cls = ns.get("Policy") or ns.get("Agent")
                if policy_cls is None:
                    for val in ns.values():
                        if (
                            isinstance(val, type)
                            and val.__module__ == mod_name
                            and (
                                hasattr(val, "run")
                                or hasattr(val, "step")
                                or hasattr(val, "act")
                            )
                        ):
                            policy_cls = val
                            break

                if policy_cls is not None:
                    if not hasattr(policy_cls, "param"):
                        try:
                            setattr(
                                policy_cls,
                                "param",
                                lambda self, *a, **kw: tracer.get_param(*a, **kw),
                            )
                        except Exception:
                            pass
                    instance = policy_cls()
                elif "policy" in ns and callable(ns["policy"]):
                    instance = ns["policy"]
                elif "run" in ns and callable(ns["run"]):
                    instance = ns["run"]
                else:
                    raise PolicyValidationError(
                        "Policy code must define `class Policy:` (or `class Agent:`) with `run(self, state)` or `step(self, state)`."
                    )
            finally:
                if prev_pol_mod is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = prev_pol_mod

            return PolicyRunner(instance, tracer)


@functools.lru_cache(maxsize=64)
def _compile_policy_and_macros_cached(
    policy_code: str,
    macros_code: str = "",
    skills_tuple: tuple[tuple[str, str], ...] = (),
    max_loop_ticks: int = 250000,
) -> CompiledPolicy:
    skills_dict = dict(skills_tuple) if skills_tuple else None
    return CompiledPolicy(
        policy_code=policy_code,
        macros_code=macros_code,
        skills=skills_dict,
        max_loop_ticks=max_loop_ticks,
    )


def compile_policy_and_macros(
    policy_code: str,
    macros_code: str = "",
    skills_tuple: tuple[tuple[str, str], ...] = (),
    max_loop_ticks: int = 250000,
) -> CompiledPolicy:
    """Validates and compiles `policy_code` and skills into a reusable `CompiledPolicy`."""
    with _POLICY_CACHE_LOCK:
        return _compile_policy_and_macros_cached(
            policy_code, macros_code, skills_tuple, max_loop_ticks
        )


def compile_policy_package(
    policy_code: str,
    macros_code: str = "",
    skills: dict[str, str] | None = None,
    max_loop_ticks: int = 250000,
) -> CompiledPolicy:
    """Validates and compiles a full hierarchical policy package with skills dictionary."""
    skills_tuple = tuple(sorted(skills.items())) if skills else ()
    return compile_policy_and_macros(
        policy_code=policy_code,
        macros_code=macros_code,
        skills_tuple=skills_tuple,
        max_loop_ticks=max_loop_ticks,
    )
