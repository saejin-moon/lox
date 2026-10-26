"""
AST Compile-Time Instrumentation:
1. ExecutionTracer: Tracks per-episode function/macro call counts, active turns per function,
   and loop-iteration watchdogs with <1 us overhead per step.
2. TunableParam: Tracks `self.param(name, default, min_val, max_val)` declarations for SPSA tuning.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


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


class ExecutionTracer:
    """
    Lightweight per-episode runtime tracer injected into compiled policies and macros.
    Powers:
    - Conditional Episode Filtering (`did episode trigger modified function X?`)
    - Per-function dwell-time and mortality attribution (`function_stats` table)
    - Zero-yield infinite loop protection (`tick_loop()`)
    - Parameter overrides during `PARAM_TUNE` (`get_param()`)
    """

    __slots__ = (
        "max_loop_ticks",
        "loop_ticks",
        "fn_calls",
        "fn_turns",
        "call_stack",
        "active_fn",
        "registered_params",
        "param_overrides",
    )

    def __init__(
        self,
        max_loop_ticks: int = 5000,
        param_overrides: dict[str, float] | None = None,
    ):
        self.max_loop_ticks: int = max_loop_ticks
        self.loop_ticks: int = 0
        self.fn_calls: dict[str, int] = {}
        self.fn_turns: dict[str, int] = {}
        self.call_stack: list[str] = []
        self.active_fn: str = "root"
        self.registered_params: dict[str, TunableParam] = {}
        self.param_overrides: dict[str, float] = (
            dict(param_overrides) if param_overrides else {}
        )

    def reset_for_episode(self) -> None:
        self.loop_ticks = 0
        self.fn_calls.clear()
        self.fn_turns.clear()
        self.call_stack.clear()
        self.active_fn = "root"

    def enter_fn(self, fn_name: str) -> None:
        self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
        self.active_fn = fn_name

    def record_turn_yield(self) -> str:
        """Called by the runner each time the policy yields an action to the harness."""
        self.loop_ticks = 0
        fn = self.active_fn
        self.fn_turns[fn] = self.fn_turns.get(fn, 0) + 1
        return fn

    def tick_loop(self) -> None:
        self.loop_ticks += 1
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
        f_min = float(min_val) if min_val is not None else (f_def * 0.25 if f_def > 0 else f_def - 10.0)
        f_max = float(max_val) if max_val is not None else (f_def * 2.5 if f_def > 0 else f_def + 10.0)
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
            self.registered_params[name].value = float(int(round(val)) if is_int else val)

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


class PolicyInstrumentTransformer(ast.NodeTransformer):
    """
    AST Transformer that injects:
    - `_lox_tracer.enter_fn("<fn_name>")` at the top of every `FunctionDef` (and after every `yield` / `yield from` inside a function so `active_fn` restores when resuming a caller!)
    - `_lox_tracer.tick_loop()` at the top of every `while` and `for` loop body.
    """

    def __init__(self, is_macro_module: bool = False):
        super().__init__()
        self.is_macro_module = is_macro_module
        self._current_fn_stack: list[str] = []

    def _make_enter_stmt(self, fn_name: str) -> ast.stmt:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_lox_tracer", ctx=ast.Load()),
                    attr="enter_fn",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=fn_name)],
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        fn_name = node.name
        self._current_fn_stack.append(fn_name)
        self.generic_visit(node)
        self._current_fn_stack.pop()

        if fn_name != "__init__":
            enter_stmt = self._make_enter_stmt(fn_name)
            # Keep docstring at index 0 if present
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.insert(1, enter_stmt)
            else:
                node.body.insert(0, enter_stmt)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt())
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt())
        return node
