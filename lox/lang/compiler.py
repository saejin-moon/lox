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
    key,
    seq,
)
from lox.env.state import (
    GameState,
    InventoryItem,
    MonsterInfo,
    ObjectInfo,
    VisibleMonster,
    VisibleObject,
    astar_next_step,
    bfs_distance_grid,
)
from lox.lang.instrument import (
    ExecutionTracer,
    PolicyInstrumentTransformer,
    PolicyLoopError,
)
from lox.lang.validator import PolicyValidationError, validate_python_code


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
    if root in allowed_map:
        return allowed_map[root]
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
        "GameState": GameState,
        "InventoryItem": InventoryItem,
        "MonsterInfo": MonsterInfo,
        "ObjectInfo": ObjectInfo,
        "VisibleMonster": VisibleMonster,
        "VisibleObject": VisibleObject,
        "bfs_distance_grid": bfs_distance_grid,
        "astar_next_step": astar_next_step,
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
        self._started: bool = False
        self._mode: str = "step"

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

    def next_action(self, state: GameState) -> tuple[Any, str]:
        """
        Advances the policy by one turn given `state`.
        Returns `(raw_yielded_action, active_function_name)`.
        """
        self.tracer.loop_ticks = 0
        try:
            setattr(self.instance, "state", state)
        except Exception:
            pass
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
            else:
                act = self.instance(state)
        except StopIteration:
            act = Action.WAIT
        except PolicyLoopError:
            raise

        active_fn = self.tracer.record_turn_yield()
        return act, active_fn


class CompiledPolicy:
    """Compiled bytecode bundle for `macros.py` + `policy.py`."""

    def __init__(
        self,
        policy_code: str,
        macros_code: str = "",
        max_loop_ticks: int = 5000,
    ):
        self.policy_code = policy_code
        self.macros_code = macros_code
        self.max_loop_ticks = max_loop_ticks

        # Parse, validate, and instrument both ASTs once
        if macros_code and macros_code.strip():
            macro_tree = validate_python_code(macros_code, filename="macros.py")
            macro_tree = PolicyInstrumentTransformer(is_macro_module=True).visit(macro_tree)
            ast.fix_missing_locations(macro_tree)
            self._macro_code_obj = compile(macro_tree, filename="macros.py", mode="exec")
        else:
            self._macro_code_obj = None

        policy_tree = validate_python_code(policy_code, filename="policy.py")
        policy_tree = PolicyInstrumentTransformer(is_macro_module=False).visit(policy_tree)
        ast.fix_missing_locations(policy_tree)
        self._policy_code_obj = compile(policy_tree, filename="policy.py", mode="exec")

        # Extract all function names defined in policy and macros
        self.policy_function_names: list[str] = [
            n.name
            for n in ast.walk(policy_tree)
            if isinstance(n, ast.FunctionDef) and n.name != "__init__"
        ]

    def create_runner(
        self,
        param_overrides: dict[str, float] | None = None,
    ) -> PolicyRunner:
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
        exec(self._policy_code_obj, ns)
        mod_obj.__dict__.update(ns)

        # Find policy class (`Policy` or `Agent` or any user class defining `run`/`step`)
        policy_cls = ns.get("Policy") or ns.get("Agent")
        if policy_cls is None:
            for val in ns.values():
                if (
                    isinstance(val, type)
                    and val.__module__ == "lox_policy"
                    and (hasattr(val, "run") or hasattr(val, "step") or hasattr(val, "act"))
                ):
                    policy_cls = val
                    break

        if policy_cls is not None:
            # Attach `self.param(...)` onto `policy_cls` BEFORE calling `__init__()`
            if not hasattr(policy_cls, "param"):
                try:
                    setattr(policy_cls, "param", lambda self, *a, **kw: tracer.get_param(*a, **kw))
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


def compile_policy_and_macros(
    policy_code: str,
    macros_code: str = "",
    max_loop_ticks: int = 5000,
) -> CompiledPolicy:
    """Validates and compiles `policy_code` and `macros_code` into a reusable `CompiledPolicy`."""
    return CompiledPolicy(
        policy_code=policy_code,
        macros_code=macros_code,
        max_loop_ticks=max_loop_ticks,
    )
