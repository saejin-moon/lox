"""
Preemptive StrategyStack Control Abstraction (`lox.lang.strategy`).
Lets policies push prioritized generator/callable frames with declarative
`preempt_if`, `until`, and `with_cleanup` guards so interrupted multi-step routines
always restore invariants cleanly without leaving orphan generator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

PredicateFn = Callable[[Any, Any], bool]
CleanupFn = Callable[[Any, Any], Any]
RunnerFn = Callable[[Any, Any], Any]


def preempt_if(predicate: PredicateFn) -> PredicateFn:
    """Identity combinator marking a preemption condition on a StrategyFrame."""
    return predicate


def until(predicate: PredicateFn) -> PredicateFn:
    """Identity combinator marking a completion condition on a StrategyFrame."""
    return predicate


def with_cleanup(cleanup: CleanupFn) -> CleanupFn:
    """Identity combinator marking an unwind handler on a StrategyFrame."""
    return cleanup


@dataclass
class StrategyFrame:
    """A prioritized, interruptible execution frame on `StrategyStack`."""

    name: str
    priority: int
    factory: RunnerFn
    preempt_if: PredicateFn | None = None
    until: PredicateFn | None = None
    cleanup: CleanupFn | None = None
    max_ticks: int = 0
    ticks_elapsed: int = 0
    _gen: Any = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        self._gen = None
        self._started = False
        self.ticks_elapsed = 0


class StrategyStack:
    """
    Preemptive priority stack for LOX policies.
    - Registered background/interrupt frames with `preempt_if(state, bb) -> True`
      automatically preempt lower-priority active frames.
    - When preempted or timed out (`max_ticks`), `cleanup(state, bb)` runs before popping.
    - Returns a native Action / RawAction / ActionSequence on `tick(state, bb)`, or `None`
      if no registered/pushed frame can act.
    """

    def __init__(self) -> None:
        self._stack: list[StrategyFrame] = []
        self._registered: list[StrategyFrame] = []
        self.preemptions: int = 0

    def register(self, frame: StrategyFrame) -> StrategyFrame:
        """Registers a candidate reactive/preemptible strategy frame."""
        for i, existing in enumerate(self._registered):
            if existing.name == frame.name:
                self._registered[i] = frame
                return frame
        self._registered.append(frame)
        self._registered.sort(key=lambda f: f.priority, reverse=True)
        return frame

    def push(self, frame: StrategyFrame) -> StrategyFrame:
        """Explicitly pushes an imperative strategy frame onto the active stack."""
        frame.reset()
        self._stack.append(frame)
        return frame

    def active_name(self) -> str:
        return self._stack[-1].name if self._stack else "idle"

    def _pop_top(self, state: Any, bb: Any, *, run_cleanup: bool = True) -> None:
        if not self._stack:
            return
        top = self._stack.pop()
        if top._gen is not None:
            try:
                top._gen.close()
            except Exception:
                pass
        if run_cleanup and top.cleanup is not None:
            try:
                top.cleanup(state, bb)
            except Exception:
                pass
        top.reset()

    def clear(self, state: Any = None, bb: Any = None) -> None:
        while self._stack:
            self._pop_top(state, bb, run_cleanup=state is not None)

    def tick(self, state: Any, bb: Any = None) -> Any:
        """Advances the highest-priority active or preempting frame and yields 1 action."""
        # 1. Check registered higher-priority preemptors
        cur_prio = self._stack[-1].priority if self._stack else -(10**9)
        for cand in self._registered:
            if cand.priority <= cur_prio and self._stack:
                break
            should_fire = False
            if cand.preempt_if is not None:
                try:
                    should_fire = bool(cand.preempt_if(state, bb))
                except Exception:
                    should_fire = False
            elif not self._stack:
                should_fire = True
            if should_fire:
                if self._stack and self._stack[-1].name != cand.name:
                    self._pop_top(state, bb, run_cleanup=True)
                    self.preemptions += 1
                if not self._stack or self._stack[-1].name != cand.name:
                    cand.reset()
                    self._stack.append(cand)
                break

        # 2. Unwind completed / expired frames and step the top frame
        while self._stack:
            top = self._stack[-1]
            if top.until is not None:
                try:
                    if top.until(state, bb):
                        self._pop_top(state, bb, run_cleanup=True)
                        continue
                except Exception:
                    pass
            if top.max_ticks > 0 and top.ticks_elapsed >= top.max_ticks:
                self._pop_top(state, bb, run_cleanup=True)
                continue

            top.ticks_elapsed += 1
            if not top._started:
                out = top.factory(state, bb)
                if hasattr(out, "__next__"):
                    top._gen = out
                    top._started = True
                    try:
                        act = next(top._gen)
                    except StopIteration as st_exc:
                        act = st_exc.value
                        self._pop_top(state, bb, run_cleanup=True)
                        if act is not None:
                            return act
                        continue
                    return act
                else:
                    # Single-step function frame
                    if out is not None:
                        return out
                    self._pop_top(state, bb, run_cleanup=False)
                    continue
            else:
                try:
                    act = top._gen.send(state)
                    return act
                except StopIteration as st_exc:
                    act = st_exc.value
                    self._pop_top(state, bb, run_cleanup=True)
                    if act is not None:
                        return act
                    continue
        return None


__all__ = [
    "StrategyFrame",
    "StrategyStack",
    "preempt_if",
    "until",
    "with_cleanup",
]
