"""
AST Compile-Time Instrumentation:
1. ExecutionTracer: Tracks per-episode function call counts, active turns per function,
   and loop-iteration watchdogs with <1 us overhead per step.
2. _TracerProxy: Non-writable read-only proxy exposed to sandboxed policy code so policies
   cannot tamper with `max_loop_ticks` or override `tick_loop`.
3. TunableParam: Tracks `self.param(name, default, min_val, max_val)` declarations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


class PolicyLoopError(RuntimeError):
    """Raised when a policy loop iterates past `max_loop_ticks` without yielding an action."""


@dataclass(slots=True)
class TunableParam:
    """Metadata for a tunable numerical parameter registered via `self.param(...)`."""

    name: str
    default: float
    min_val: float
    max_val: float
    is_int: bool = False
    value: float = 0.0

    def __getitem__(self, key: str):
        return getattr(self, key)


class _TracerProxy:
    """Read-only proxy exposing ONLY tracer hook callables to policy code (blocks attribute mutation)."""

    __slots__ = (
        "enter_fn",
        "exit_fn",
        "on_return",
        "on_yield",
        "tick_loop",
        "guarded_range",
        "guarded_iter",
        "get_param",
    )

    def __init__(self, tracer: ExecutionTracer) -> None:
        object.__setattr__(self, "enter_fn", tracer.enter_fn)
        object.__setattr__(self, "exit_fn", tracer.exit_fn)
        object.__setattr__(self, "on_return", tracer.on_return)
        object.__setattr__(self, "on_yield", tracer.on_yield)
        object.__setattr__(self, "tick_loop", tracer.tick_loop)
        object.__setattr__(self, "guarded_range", tracer.guarded_range)
        object.__setattr__(self, "guarded_iter", tracer.guarded_iter)
        object.__setattr__(self, "get_param", tracer.get_param)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Cannot modify read-only tracer attribute '{name}' in policy sandbox."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Cannot delete read-only tracer attribute '{name}' in policy sandbox."
        )


def _is_valid_delta_tuple(val: tuple[Any, ...]) -> bool:
    if len(val) != 2:
        return False
    a, b = val[0], val[1]
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not (
        (type(a) is int or type(a).__module__ == "numpy")
        and (type(b) is int or type(b).__module__ == "numpy")
    ):
        return False
    try:
        ia, ib = int(a), int(b)
    except Exception:
        return False
    return -1 <= ia <= 1 and -1 <= ib <= 1 and not (ia == 0 and ib == 0)


class ExecutionTracer:
    """
    Lightweight per-episode runtime tracer injected into compiled policies and skills.
    Uses unique invocation frame IDs so closing/GCing suspended generators never corrupts
    the active call stack.
    """

    __slots__ = (
        "max_loop_ticks",
        "loop_ticks",
        "total_turns",
        "fn_calls",
        "fn_turns",
        "call_stack",
        "_next_frame_id",
        "active_fn",
        "_leaf_action_fn",
        "_turn_fn_locked",
        "registered_params",
        "param_overrides",
    )

    _ROOT_FNS = frozenset({"run", "step", "act", "root", "dispatch", "step_dispatch"})

    def __init__(
        self,
        max_loop_ticks: int = 250000,
        param_overrides: dict[str, float] | None = None,
    ):
        self.max_loop_ticks: int = max_loop_ticks
        self.loop_ticks: int = 0
        self.total_turns: int = 0
        self.fn_calls: dict[str, int] = {}
        self.fn_turns: dict[str, int] = {}
        # Stack frame: [frame_id, fn_name, enter_turn, returned_action_like, is_gen]
        self.call_stack: list[list[Any]] = []
        self._next_frame_id: int = 0
        self.active_fn: str = "root"
        self._leaf_action_fn: str = ""
        self._turn_fn_locked: bool = False
        self.registered_params: dict[str, TunableParam] = {}
        self.param_overrides: dict[str, float] = (
            dict(param_overrides) if param_overrides else {}
        )

    def make_proxy(self) -> _TracerProxy:
        return _TracerProxy(self)

    def reset_for_episode(self) -> None:
        self.loop_ticks = 0
        self.total_turns = 0
        self.fn_calls.clear()
        self.fn_turns.clear()
        self.call_stack.clear()
        self._next_frame_id = 0
        self.active_fn = "root"
        self._leaf_action_fn = ""
        self._turn_fn_locked = False

    def enter_fn(
        self, fn_name: str, has_value_return: bool = False, is_gen: bool = False
    ) -> int:
        self._next_frame_id += 1
        fid = self._next_frame_id
        self.call_stack.append([fid, fn_name, self.total_turns, False, bool(is_gen)])
        if not has_value_return:
            self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
            if is_gen:
                self.active_fn = fn_name
                if fn_name not in self._ROOT_FNS:
                    self._turn_fn_locked = True
            elif fn_name in self._ROOT_FNS and not self._turn_fn_locked:
                self.active_fn = fn_name
        return fid

    def on_return(self, fn_name: str, val: Any, is_gen: bool = False) -> Any:
        if val is None:
            return val
        val_type = type(val)
        if not is_gen:
            if val_type is bool or val_type is int or val_type is float:
                self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
                return val
            if val_type is tuple and len(val) == 2 and val == (0, 0):
                return val
            self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1

        is_action_like = False
        is_direct_action = False
        val_type_name = val_type.__name__
        if val_type_name in ("Action", "RawAction", "ActionSequence"):
            is_action_like = True
            is_direct_action = True
        elif val_type is str and len(val) == 1:
            is_action_like = True
            is_direct_action = True
        elif val_type is tuple and _is_valid_delta_tuple(val):
            is_action_like = True
        if is_action_like:
            if self.call_stack:
                self.call_stack[-1][3] = True
            if (
                is_direct_action
                and not self._leaf_action_fn
                and fn_name not in self._ROOT_FNS
            ):
                self._leaf_action_fn = fn_name
            if not self._turn_fn_locked:
                self.active_fn = fn_name
        return val

    def on_yield(self, fn_name: str, val: Any) -> Any:
        if not (self._turn_fn_locked and fn_name in self._ROOT_FNS):
            self.active_fn = fn_name
        if fn_name not in self._ROOT_FNS:
            self._turn_fn_locked = True
        return val

    def exit_fn(self, frame_or_name: int | str, fn_name: str | None = None) -> None:
        returned_non_noop = False
        resolved_name: str
        if isinstance(frame_or_name, int):
            fid = frame_or_name
            resolved_name = fn_name or ""
            # Search from top for this exact frame_id; never wipe unrelated frames if already popped/suspended.
            match_idx = -1
            for i in range(len(self.call_stack) - 1, -1, -1):
                if self.call_stack[i][0] == fid:
                    match_idx = i
                    break
            if match_idx < 0:
                return
            frame = self.call_stack.pop(match_idx)
            resolved_name = resolved_name or str(frame[1])
            returned_non_noop = bool(frame[3])
            # If the popped frame was NOT the top frame (e.g. a suspended generator closing beneath an active call),
            # leave the active call stack and active_fn undisturbed.
            if match_idx != len(self.call_stack):
                return
        else:
            resolved_name = str(frame_or_name)
            if self.call_stack:
                match_idx = -1
                for i in range(len(self.call_stack) - 1, -1, -1):
                    if self.call_stack[i][1] == resolved_name:
                        match_idx = i
                        break
                if match_idx < 0:
                    return
                frame = self.call_stack.pop(match_idx)
                returned_non_noop = bool(frame[3])
                if match_idx != len(self.call_stack):
                    return

        if not self.call_stack:
            return

        caller_fn = self.call_stack[-1][1]
        if returned_non_noop:
            if caller_fn in self._ROOT_FNS and resolved_name not in self._ROOT_FNS:
                self.active_fn = resolved_name
                self._turn_fn_locked = True
            elif not self._turn_fn_locked and caller_fn not in self._ROOT_FNS:
                self.active_fn = caller_fn
        else:
            if not self._turn_fn_locked:
                self.active_fn = caller_fn

    def record_turn_yield(self, preferred_fn: str | None = None) -> str:
        """Called by the runner each time the policy yields an action to the harness."""
        self.loop_ticks = 0
        self.total_turns += 1
        fn = self.active_fn
        leaf = self._leaf_action_fn
        self._leaf_action_fn = ""
        if isinstance(preferred_fn, str):
            pref = preferred_fn.strip()
            if pref and pref != "root":
                if pref != fn:
                    self.fn_calls[pref] = self.fn_calls.get(pref, 0) + 1
                if (
                    leaf
                    and leaf != pref
                    and leaf.lstrip("_") != pref.lstrip("_")
                    and leaf not in ("tick", "step", "_step", "run", "act")
                    and not leaf.startswith(pref + ".")
                ):
                    fn = f"{pref}.{leaf}"
                    self.fn_calls[fn] = self.fn_calls.get(fn, 0) + 1
                else:
                    fn = pref
        elif leaf and fn in self._ROOT_FNS:
            fn = leaf
        self.fn_turns[fn] = self.fn_turns.get(fn, 0) + 1
        self._turn_fn_locked = False
        if self.call_stack:
            self.active_fn = self.call_stack[-1][1]
        else:
            self.active_fn = "root"
        return fn

    def tick_loop(self) -> None:
        self.loop_ticks += 1
        if self.loop_ticks > self.max_loop_ticks:
            raise PolicyLoopError(
                f"Infinite zero-yield loop detected in '{self.active_fn}' "
                f"(>{self.max_loop_ticks} iterations without yielding an action)."
            )

    def guarded_range(self, r: range) -> Any:
        """Watchdog guard for `range(...)` loops: fast path for small ranges (<=64), chunked charging otherwise."""
        if isinstance(r, range):
            n = len(r)
            if n <= 64:
                self.loop_ticks += n
                if self.loop_ticks > self.max_loop_ticks:
                    raise PolicyLoopError(
                        f"Infinite zero-yield loop detected in '{self.active_fn}' "
                        f"(>{self.max_loop_ticks} iterations without yielding an action)."
                    )
                return r
        return self.guarded_iter(r)

    def guarded_iter(self, r: Any) -> Any:
        """Incremental tick-counting wrapper for arbitrary iterables (ranges, comprehensions, generators)."""
        ticks = 0
        for item in r:
            ticks += 1
            if (ticks & 63) == 0:
                self.loop_ticks += 64
                if self.loop_ticks > self.max_loop_ticks:
                    raise PolicyLoopError(
                        f"Infinite zero-yield loop detected in '{self.active_fn}' "
                        f"(>{self.max_loop_ticks} iterations without yielding an action)."
                    )
            yield item
        rem = ticks & 63
        if rem:
            self.loop_ticks += rem
            if self.loop_ticks > self.max_loop_ticks:
                raise PolicyLoopError(
                    f"Infinite zero-yield loop detected in '{self.active_fn}' "
                    f"(>{self.max_loop_ticks} iterations without yielding an action)."
                )

    def get_param(
        self,
        name: str,
        default: float | int,
        min_val: float | int | None = None,
        max_val: float | int | None = None,
    ) -> float | int:
        is_int = isinstance(default, int) and not isinstance(default, bool)
        f_def = float(default)
        f_min = (
            float(min_val)
            if min_val is not None
            else (f_def * 0.25 if f_def > 0 else f_def - 10.0)
        )
        f_max = (
            float(max_val)
            if max_val is not None
            else (f_def * 2.5 if f_def > 0 else f_def + 10.0)
        )
        if f_min > f_max:
            f_min, f_max = f_max, f_min

        val = self.param_overrides.get(name, f_def)
        if name not in self.registered_params:
            self.registered_params[name] = TunableParam(
                name=name,
                default=f_def,
                min_val=f_min,
                max_val=f_max,
                is_int=is_int,
                value=float(int(round(val)) if is_int else val),
            )
        else:
            self.registered_params[name].value = float(
                int(round(val)) if is_int else val
            )

        if is_int:
            return int(round(val))
        return float(val)

    @property
    def call_counts(self) -> dict[str, int]:
        return self.fn_calls

    @property
    def active_turns(self) -> dict[str, int]:
        return self.fn_turns

    def triggered_any(self, target_fns: list[str] | set[str]) -> bool:
        if not target_fns:
            return True
        return any(self.fn_calls.get(fn, 0) > 0 for fn in target_fns)


def _node_binds_range(stmts: list[ast.stmt]) -> bool:
    """Returns True if any binding in `stmts` (shallow over nested defs) shadows `range`."""
    stack: list[ast.AST] = list(stmts)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if cur.name == "range":
                return True
            continue
        if (
            isinstance(cur, ast.Name)
            and isinstance(cur.ctx, ast.Store)
            and cur.id == "range"
        ):
            return True
        if isinstance(cur, (ast.Import, ast.ImportFrom)):
            for al in cur.names:
                if (al.asname or al.name.split(".")[0]) == "range":
                    return True
        if isinstance(cur, ast.MatchAs) and cur.name == "range":
            return True
        if isinstance(cur, ast.MatchStar) and cur.name == "range":
            return True
        if isinstance(cur, ast.MatchMapping) and cur.rest == "range":
            return True
        for ch in ast.iter_child_nodes(cur):
            stack.append(ch)
    return False


class PolicyInstrumentTransformer(ast.NodeTransformer):
    """
    AST Transformer that injects:
    - `_lox_fid = _lox_tracer.enter_fn("<fn_name>", ...)` + `finally: _lox_tracer.exit_fn(_lox_fid, "<fn_name>")`
    - `_lox_tracer.tick_loop()` / `_lox_tracer.guarded_range(...)` / `_lox_tracer.guarded_iter(...)`
      on `while`, `for`, `async for`, and comprehension/generator-expression loops.
    """

    def __init__(self, is_macro_module: bool = False):
        super().__init__()
        self.is_macro_module = is_macro_module
        self._current_fn_stack: list[tuple[str, bool]] = []
        self._range_shadowed: bool = False

    def visit_Module(self, node: ast.Module) -> ast.Module:
        prev_shadow = self._range_shadowed
        self._range_shadowed = prev_shadow or _node_binds_range(node.body)
        self.generic_visit(node)
        self._range_shadowed = prev_shadow
        return node

    def _make_enter_stmt(
        self,
        fid_var: str,
        fn_name: str,
        has_val_ret: bool = False,
        is_gen: bool = False,
    ) -> ast.stmt:
        return ast.Assign(
            targets=[ast.Name(id=fid_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="enter_fn",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Constant(value=fn_name),
                    ast.Constant(value=has_val_ret),
                    ast.Constant(value=is_gen),
                ],
                keywords=[],
            ),
        )

    def _make_exit_stmt(self, fid_var: str, fn_name: str) -> ast.stmt:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="exit_fn",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id=fid_var, ctx=ast.Load()),
                    ast.Constant(value=fn_name),
                ],
                keywords=[],
            )
        )

    def _make_tick_stmt(self) -> ast.stmt:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="tick_loop",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

    def visit_Return(self, node: ast.Return) -> ast.Return:
        self.generic_visit(node)
        if self._current_fn_stack and self._current_fn_stack[-1][0] != "__init__":
            fn_name, is_gen = self._current_fn_stack[-1]
            ret_val = node.value if node.value is not None else ast.Constant(value=None)
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="on_return",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Constant(value=fn_name),
                    ret_val,
                    ast.Constant(value=is_gen),
                ],
                keywords=[],
            )
        return node

    def visit_Yield(self, node: ast.Yield) -> ast.AST:
        self.generic_visit(node)
        if self._current_fn_stack and self._current_fn_stack[-1][0] != "__init__":
            fn_name = self._current_fn_stack[-1][0]
            inner_val = (
                node.value if node.value is not None else ast.Constant(value=None)
            )
            marked = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="on_yield",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=fn_name), inner_val],
                keywords=[],
            )
            return ast.copy_location(ast.Yield(value=marked), node)
        return node

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.AST:
        self.generic_visit(node)
        if self._current_fn_stack and self._current_fn_stack[-1][0] != "__init__":
            fn_name = self._current_fn_stack[-1][0]
            marked = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="on_yield",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=fn_name), node.value],
                keywords=[],
            )
            return ast.copy_location(ast.YieldFrom(value=marked), node)
        return node

    @staticmethod
    def _scan_fn_flags(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[bool, bool, bool]:
        """Single shallow scan for yields, valued returns, and local `range` shadowing."""
        is_gen = False
        has_val_ret = False
        shadows_range = any(
            arg.arg == "range"
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
                + ([node.args.vararg] if node.args.vararg else [])
                + ([node.args.kwarg] if node.args.kwarg else [])
            )
        ) or _node_binds_range(node.body)
        stack: list[ast.AST] = list(node.body)
        while stack:
            cur = stack.pop()
            if isinstance(cur, (ast.Yield, ast.YieldFrom)):
                is_gen = True
            elif (
                isinstance(cur, ast.Return)
                and cur.value is not None
                and not (
                    isinstance(cur.value, ast.Constant) and cur.value.value is None
                )
            ):
                has_val_ret = True
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.iter_child_nodes(cur):
                stack.append(child)
        return is_gen, (not is_gen) and has_val_ret, shadows_range

    def _visit_fn_like(self, node: Any) -> Any:
        fn_name = node.name
        is_gen, has_val_ret, shadows_range = self._scan_fn_flags(node)
        self._current_fn_stack.append((fn_name, is_gen))
        prev_shadow = self._range_shadowed
        self._range_shadowed = prev_shadow or shadows_range
        self.generic_visit(node)
        self._range_shadowed = prev_shadow
        self._current_fn_stack.pop()

        if fn_name != "__init__":
            fid_var = f"_lox_fid_{len(self._current_fn_stack)}"
            enter_stmt = self._make_enter_stmt(fid_var, fn_name, has_val_ret, is_gen)
            exit_stmt = self._make_exit_stmt(fid_var, fn_name)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_stmt = node.body[0]
                rest_body: list[ast.stmt] = list(node.body[1:]) or [ast.Pass()]
                try_stmt = ast.Try(
                    body=rest_body,
                    handlers=[],
                    orelse=[],
                    finalbody=[exit_stmt],
                )
                node.body = [doc_stmt, enter_stmt, try_stmt]
            else:
                rest_body_all: list[ast.stmt] = list(node.body) or [ast.Pass()]
                try_stmt = ast.Try(
                    body=rest_body_all,
                    handlers=[],
                    orelse=[],
                    finalbody=[exit_stmt],
                )
                node.body = [enter_stmt, try_stmt]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._visit_fn_like(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return self._visit_fn_like(node)

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt())
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        if (
            not self._range_shadowed
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            node.iter = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="guarded_range",
                    ctx=ast.Load(),
                ),
                args=[node.iter],
                keywords=[],
            )
            return node
        node.body.insert(0, self._make_tick_stmt())
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt())
        return node

    def _guard_comprehension_generators(
        self, generators: list[ast.comprehension]
    ) -> None:
        for gen in generators:
            if gen.is_async:
                continue
            if (
                not self._range_shadowed
                and isinstance(gen.iter, ast.Call)
                and isinstance(gen.iter.func, ast.Name)
                and gen.iter.func.id == "range"
            ):
                gen.iter = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                        attr="guarded_range",
                        ctx=ast.Load(),
                    ),
                    args=[gen.iter],
                    keywords=[],
                )
            else:
                gen.iter = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                        attr="guarded_iter",
                        ctx=ast.Load(),
                    ),
                    args=[gen.iter],
                    keywords=[],
                )

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        self.generic_visit(node)
        self._guard_comprehension_generators(node.generators)
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        self.generic_visit(node)
        self._guard_comprehension_generators(node.generators)
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        self.generic_visit(node)
        self._guard_comprehension_generators(node.generators)
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        self.generic_visit(node)
        self._guard_comprehension_generators(node.generators)
        return node
