"""
Policy & Macro Compiler.
Compiles `macros.py` + `policy.py` with `PolicyInstrumentTransformer` into an
executable `CompiledPolicy` factory that instantiates fast per-episode `PolicyRunner` objects.
"""

from __future__ import annotations

import ast
import bisect
import collections
import copy
import dataclasses
import enum
import functools
import heapq
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
from lox.lang.bt import (
    ActionLeaf,
    BehaviorNode,
    BehaviorTree,
    Blackboard,
    ConditionLeaf,
    Inverter,
    MemorySequence,
    NodeStatus,
    PrioritySelector,
    Sequence,
    bt_leaf,
)
from lox.lang.contracts import (
    CapabilityScorer,
    GoalIntent,
    SafetyReport,
    ThreatEvaluator,
    WorldTracker,
)
from lox.lang.instrument import (
    ExecutionTracer,
    PolicyInstrumentTransformer,
    PolicyLoopError,
)
from lox.lang.strategy import (
    StrategyFrame,
    StrategyStack,
    preempt_if,
    until,
    with_cleanup,
)
from lox.lang.validator import (
    PolicyValidationError,
    check_undefined_symbols,
    validate_python_code,
)

_COMPILE_EXEC_LOCK = threading.RLock()

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
    "getattr": getattr,
    "hasattr": hasattr,
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
    "setattr": setattr,
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
    "True": True,
    "False": False,
    "None": None,
}


def _safe_import(name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0):
    root = name.split(".")[0]
    allowed_map = {
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
        "np": np,
        "random": random,
        "operator": operator,
        "bisect": bisect,
        "copy": copy,
    }
    if root in allowed_map and root == name:
        return allowed_map[root]
    if root in allowed_map or root in ("skills", "lox"):
        if root == "skills" and name != "skills" and name not in sys.modules:
            avail_skills = sorted(k for k in sys.modules if k.startswith("skills."))
            raise ImportError(
                f"Skill module '{name}' not found. Available skills: {avail_skills or 'none'}"
            )
        if not fromlist:
            if root in sys.modules:
                return sys.modules[root]
            if name in sys.modules:
                return sys.modules[name]
        else:
            if name in sys.modules:
                return sys.modules[name]
            if root in sys.modules:
                return sys.modules[root]
    raise ImportError(f"Import of '{name}' is not permitted in policy sandbox.")


SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS["__build_class__"] = __build_class__


def _build_base_namespace(tracer: ExecutionTracer) -> dict[str, Any]:
    """Builds the sandboxed execution globals with 1-to-1 actions, state utilities, and stdlib."""
    ns: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "lox_policy",
        "_lox_tracer": tracer,
        # Standard library modules pre-loaded for convenience
        "math": math,
        "collections": collections,
        "deque": collections.deque,
        "defaultdict": collections.defaultdict,
        "Counter": collections.Counter,
        "heapq": heapq,
        "itertools": itertools,
        "functools": functools,
        "re": re,
        "enum": enum,
        "dataclasses": dataclasses,
        "dataclass": dataclasses.dataclass,
        "typing": typing,
        "Any": Any,
        "np": np,
        "numpy": np,
        "random": random,
        "operator": operator,
        "bisect": bisect,
        "copy": copy,
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
        # Behavior Tree Components
        "BehaviorNode": BehaviorNode,
        "BehaviorTree": BehaviorTree,
        "PrioritySelector": PrioritySelector,
        "Sequence": Sequence,
        "MemorySequence": MemorySequence,
        "ActionLeaf": ActionLeaf,
        "ConditionLeaf": ConditionLeaf,
        "Inverter": Inverter,
        "Blackboard": Blackboard,
        "NodeStatus": NodeStatus,
        "bt_leaf": bt_leaf,
        # StrategyStack & Contracts
        "StrategyStack": StrategyStack,
        "StrategyFrame": StrategyFrame,
        "preempt_if": preempt_if,
        "until": until,
        "with_cleanup": with_cleanup,
        "GoalIntent": GoalIntent,
        "SafetyReport": SafetyReport,
        "WorldTracker": WorldTracker,
        "CapabilityScorer": CapabilityScorer,
        "ThreatEvaluator": ThreatEvaluator,
        # Decorator no-op for `@macro`
        "macro": lambda fn: fn,
        # Global parameter helper `param("name", default, min_val, max_val)`
        "param": tracer.get_param,
    }
    return ns


class PolicyRunner:
    """
    Per-episode driver for a compiled policy instance.
    Supports both:
    - Generator policies: `def run(self, state): while True: state = yield action`
    - Step policies: `def step(self, state) -> action` (or `act(self, state)`)
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
        elif hasattr(self.instance, "tree") and hasattr(self.instance.tree, "tick"):
            self._mode = "bt"
        elif callable(self.instance):
            self._mode = "callable"
        else:
            raise PolicyValidationError(
                "Policy class must define `def run(self, state):` (generator) or `def step(self, state):`."
            )
        self._blackboards_cache: tuple[Any, ...] | None = None

    def _iter_instance_blackboards(self) -> tuple[Any, ...]:
        if self._blackboards_cache is not None:
            return self._blackboards_cache
        bbs: list[Any] = []
        for obj in (
            self.instance,
            getattr(self.instance, "ws", None),
            getattr(self.instance, "world_state", None),
            getattr(self.instance, "blackboard", None),
            getattr(self.instance, "bb", None),
            getattr(getattr(self.instance, "tree", None), "blackboard", None),
        ):
            if (
                obj is not None
                and (
                    hasattr(obj, "last_active_leaf") or hasattr(obj, "last_active_goal")
                )
                and obj not in bbs
            ):
                bbs.append(obj)
        self._blackboards_cache = tuple(bbs)
        return self._blackboards_cache

    def _clear_bt_last_active_leaf(self) -> None:
        for bb in self._iter_instance_blackboards():
            if hasattr(bb, "last_active_leaf"):
                bb.last_active_leaf = ""
            if hasattr(bb, "last_active_goal"):
                bb.last_active_goal = ""

    def _get_bt_last_active_leaf(self) -> str:
        for bb in self._iter_instance_blackboards():
            goal = str(getattr(bb, "last_active_goal", "") or "").strip()
            if goal:
                return goal
            leaf = str(getattr(bb, "last_active_leaf", "") or "").strip()
            if leaf:
                return leaf
        return ""

    def next_action(self, state: GameState) -> tuple[Any, str]:
        """
        Advances the policy by one turn given `state`.
        Returns `(raw_yielded_action, active_function_name)`.
        """
        self.tracer.loop_ticks = 0
        self._clear_bt_last_active_leaf()
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
            self._active_subgen = None
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
                    self._active_subgen = act
                    act = next(self._active_subgen)
                active_fn = self.tracer.record_turn_yield(
                    preferred_fn=self._get_bt_last_active_leaf()
                )
                return act, active_fn
            except StopIteration:
                self._active_subgen = None

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
                    self._gen = res
                    self._mode = "run"
                    self._started = True
                    act = next(self._gen)
                else:
                    act = res
            elif self._mode == "act":
                act = self.instance.act(state)
            elif self._mode == "bt":
                act = self.instance.tree.tick(state)
            else:
                act = self.instance(state)
        except StopIteration:
            if self._mode == "run":
                try:
                    res = self.instance.run(state)
                    if inspect.isgenerator(res):
                        self._gen = res
                        act = next(self._gen)
                    else:
                        self._mode = "run_direct"
                        act = res
                except StopIteration:
                    act = Action.WAIT
            else:
                act = Action.WAIT
        except PolicyLoopError:
            raise

        if inspect.isgenerator(act):
            self._active_subgen = act
            try:
                act = next(self._active_subgen)
            except StopIteration:
                self._active_subgen = None
                act = Action.WAIT

        active_fn = self.tracer.record_turn_yield(
            preferred_fn=self._get_bt_last_active_leaf()
        )
        return act, active_fn


def _extract_skill_dependencies(tree: ast.AST, known_skills: set[str]) -> set[str]:
    """Extracts intra-package `skills.<name>` or `skills.<domain>.<name>` dependencies."""
    from lox.lang.validator import resolve_skill_import

    deps: set[str] = set()
    pkg_names = {k.split("/", 1)[0] for k in known_skills if "/" in k}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "")
                if name.startswith("skills."):
                    dep = resolve_skill_import(name, known_skills)
                    if dep in known_skills:
                        deps.add(dep)
        elif isinstance(node, ast.ImportFrom):
            mod = str(node.module or "")
            if mod == "skills":
                for alias in node.names:
                    dep = str(alias.name or "").split(".", 1)[0]
                    if dep in known_skills:
                        deps.add(dep)
            elif mod.startswith("skills."):
                tail = mod[len("skills.") :]
                if tail in pkg_names and tail not in known_skills:
                    for alias in node.names:
                        sub_k = f"{tail}/{alias.name}"
                        if sub_k in known_skills:
                            deps.add(sub_k)
                dep = resolve_skill_import(mod, known_skills)
                if dep in known_skills:
                    deps.add(dep)
    return deps


def _topological_skill_order(skill_trees: dict[str, ast.AST]) -> list[str]:
    """
    Orders skill modules so dependencies (e.g. `skills.shared`) execute BEFORE
    importers (e.g. `skills.equip`, `skills.combat`), regardless of alphabetical name.
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
        return (0 if name.startswith("shared") else 1, name)

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
    from lox.lang.validator import normalize_skill_key, skill_key_to_relpath

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
        for n in ast.walk(skill_tree)
        if isinstance(n, ast.FunctionDef) and n.name != "__init__"
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
    with _SKILL_CACHE_LOCK:
        return _compile_skill_unit_cached(skill_name, skill_code, is_macro_module)


class CompiledPolicy:
    """Compiled bytecode bundle for `macros.py` + `policy.py` + optional `skills/` package."""

    def __init__(
        self,
        policy_code: str,
        macros_code: str = "",
        skills: dict[str, str] | None = None,
        max_loop_ticks: int = 250000,
    ):
        from lox.lang.validator import normalize_skill_key

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

        # Parse, validate, and instrument skills package in dependency order (per-skill LRU cached)
        raw_skill_deps: dict[str, set[str]] = {}
        raw_skill_code_objs: dict[str, Any] = {}
        known_skill_keys = {k for k, v in self.skills.items() if v and v.strip()}
        for skill_name, skill_code in self.skills.items():
            if not skill_code or not skill_code.strip():
                continue
            c_obj, fn_names, deps_fs = _compile_skill_unit(
                skill_name, skill_code, is_macro_module=True
            )
            raw_skill_deps[skill_name] = {d for d in deps_fs if d in known_skill_keys}
            raw_skill_code_objs[skill_name] = c_obj
            self.skill_function_names[skill_name] = list(fn_names)
        self._skill_order: list[str] = _topological_skill_order(raw_skill_deps)
        for skill_name in self._skill_order:
            self._skill_code_objs[skill_name] = raw_skill_code_objs[skill_name]

        # Legacy macros support (compiled if provided for backwards compatibility)
        if macros_code and macros_code.strip():
            macro_tree = validate_python_code(macros_code, filename="macros.py")
            macro_tree = PolicyInstrumentTransformer(is_macro_module=True).visit(
                macro_tree
            )
            ast.fix_missing_locations(macro_tree)
            self._macro_code_obj = compile(
                macro_tree, filename="macros.py", mode="exec"
            )
            for n in ast.walk(macro_tree):
                if isinstance(n, ast.FunctionDef) and n.name != "__init__":
                    if n.name not in self.macro_function_names:
                        self.macro_function_names.append(n.name)
        else:
            self._macro_code_obj = None

        # Fast (<2ms) CPython symtable undefined-symbol and cross-skill export gate
        allowed_globals = frozenset(
            _build_base_namespace(ExecutionTracer()).keys()
        ) | frozenset(self.macro_function_names)
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

        # Extract all function names defined in policy, macros, and skills
        self.policy_function_names: list[str] = [
            n.name
            for n in ast.walk(policy_tree)
            if isinstance(n, ast.FunctionDef) and n.name != "__init__"
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
            ns = _build_base_namespace(tracer)
            # Register `lox_policy` in sys.modules so `@dataclass` with type annotations works inside policy code
            mod_obj = sys.modules.get("lox_policy")
            if mod_obj is None:
                mod_obj = types.ModuleType("lox_policy")
                sys.modules["lox_policy"] = mod_obj
            mod_obj.__dict__.update(ns)

            if self._macro_code_obj is not None:
                exec(self._macro_code_obj, ns)
                mod_obj.__dict__.update(ns)

            # Purge any stale `skills.*` entries from sys.modules so worker processes
            # never retain an old skill module bound to a previous episode's ExecutionTracer.
            for mod_key in list(sys.modules.keys()):
                if mod_key == "skills" or mod_key.startswith("skills."):
                    sys.modules.pop(mod_key, None)

            # Register and execute skills modules (supports both `skills.<mod>` and 2-level `skills.<domain>.<mod>`)
            skills_pkg = types.ModuleType("skills")
            skills_pkg.__path__ = []
            sys.modules["skills"] = skills_pkg
            ns["skills"] = skills_pkg

            for s_name, s_code_obj in self._skill_code_objs.items():
                parts = [p for p in str(s_name).split("/") if p]
                mod_full = f"skills.{'.'.join(parts)}"
                if len(parts) == 2:
                    pkg_full = f"skills.{parts[0]}"
                    pkg_mod = sys.modules.get(pkg_full)
                    if pkg_mod is None:
                        pkg_mod = types.ModuleType(pkg_full)
                        pkg_mod.__path__ = []
                        sys.modules[pkg_full] = pkg_mod
                        setattr(skills_pkg, parts[0], pkg_mod)
                s_mod = types.ModuleType(mod_full)
                s_mod.__dict__.update(ns)
                s_mod.__name__ = mod_full
                s_mod.__package__ = (
                    f"skills.{parts[0]}" if len(parts) == 2 else "skills"
                )
                sys.modules[mod_full] = s_mod
                exec(s_code_obj, s_mod.__dict__)
                if len(parts) == 2:
                    setattr(sys.modules[f"skills.{parts[0]}"], parts[1], s_mod)
                    ns[f"{parts[0]}_{parts[1]}"] = s_mod
                else:
                    setattr(skills_pkg, parts[0], s_mod)
                    ns[parts[0]] = s_mod

            exec(self._policy_code_obj, ns)
            mod_obj.__dict__.update(ns)

            # Find policy class (`Policy` or `Agent` or any user class defining `run`/`step`)
            policy_cls = ns.get("Policy") or ns.get("Agent")
            if policy_cls is None:
                for val in ns.values():
                    if (
                        isinstance(val, type)
                        and val.__module__ == "lox_policy"
                        and (
                            hasattr(val, "run")
                            or hasattr(val, "step")
                            or hasattr(val, "act")
                        )
                    ):
                        policy_cls = val
                        break

            if policy_cls is not None:
                # Attach `self.param(...)` onto `policy_cls` BEFORE calling `__init__()`
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
    """Validates and compiles `policy_code`, `macros_code`, and skills into a reusable `CompiledPolicy`."""
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


def compile_bt_subtree(
    code: str,
    filename: str = "subtree.py",
    skills: dict[str, str] | None = None,
) -> Any:
    """Validates and executes a standalone Behavior Tree subtree, returning the root node or instance."""
    with _COMPILE_EXEC_LOCK:
        tree = validate_python_code(code, filename=filename)
        tree = PolicyInstrumentTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        tracer = ExecutionTracer()
        ns = _build_base_namespace(tracer)
        mod_obj = sys.modules.get("lox_policy")
        if mod_obj is None:
            mod_obj = types.ModuleType("lox_policy")
            sys.modules["lox_policy"] = mod_obj
        mod_obj.__dict__.update(ns)

        if skills:
            skills_pkg = sys.modules.get("skills")
            if skills_pkg is None:
                skills_pkg = types.ModuleType("skills")
                sys.modules["skills"] = skills_pkg
            ns["skills"] = skills_pkg
            bt_skill_deps: dict[str, set[str]] = {}
            bt_skill_objs: dict[str, Any] = {}
            for s_name, s_code in skills.items():
                if not s_code or not s_code.strip():
                    continue
                c_obj, _fns, deps_fs = _compile_skill_unit(
                    s_name, s_code, is_macro_module=False
                )
                bt_skill_deps[s_name] = set(deps_fs)
                bt_skill_objs[s_name] = c_obj
            for s_name in _topological_skill_order(bt_skill_deps):
                s_mod = types.ModuleType(f"skills.{s_name}")
                s_mod.__dict__.update(ns)
                s_mod.__name__ = f"skills.{s_name}"
                exec(bt_skill_objs[s_name], s_mod.__dict__)
                sys.modules[f"skills.{s_name}"] = s_mod
                setattr(skills_pkg, s_name, s_mod)
                ns[s_name] = s_mod

        exec(compile(tree, filename=filename, mode="exec"), ns)
        mod_obj.__dict__.update(ns)

        def _attach_tracer(obj: Any) -> Any:
            try:
                setattr(obj, "_lox_tracer", tracer)
            except Exception:
                pass
            return obj

        for preferred in ("subtree", "root", "tree", "node", "policy"):
            if preferred in ns:
                cand = ns[preferred]
                if isinstance(cand, (BehaviorTree, BehaviorNode)):
                    return _attach_tracer(cand)
                if not inspect.isclass(cand) and (
                    hasattr(cand, "tick") or hasattr(cand, "step")
                ):
                    return _attach_tracer(cand)

        bt_trees = [v for v in ns.values() if isinstance(v, BehaviorTree)]
        if bt_trees:
            return _attach_tracer(bt_trees[-1])

        composites = [
            v
            for v in ns.values()
            if isinstance(v, (PrioritySelector, Sequence, MemorySequence, Inverter))
        ]
        if composites:
            return _attach_tracer(composites[-1])

        leaves = [v for v in ns.values() if isinstance(v, BehaviorNode)]
        if leaves:
            return _attach_tracer(leaves[-1])

        user_classes = [
            v
            for v in ns.values()
            if inspect.isclass(v)
            and getattr(v, "__module__", None) == "lox_policy"
            and (hasattr(v, "tick") or hasattr(v, "step") or hasattr(v, "run"))
        ]
        for cls in reversed(user_classes):
            try:
                return _attach_tracer(cls())
            except Exception:
                pass

        user_fns = [
            v
            for k, v in ns.items()
            if not k.startswith("_")
            and callable(v)
            and not inspect.isclass(v)
            and getattr(v, "__module__", None) == "lox_policy"
        ]
        if user_fns:
            return _attach_tracer(user_fns[-1])

        raise PolicyValidationError(
            "No BehaviorTree, PrioritySelector, Sequence, ActionLeaf, ConditionLeaf, or Policy class found in subtree code."
        )
