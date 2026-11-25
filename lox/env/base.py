"""
Abstract Benchmark Adapter & Environment Harness Protocol ("Hard Code Nothing").
Defines the clean, pluggable contract for running synthesis on NetHack / MiniHack scenarios.
"""

from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvHarness(Protocol):
    """Protocol for a deterministic benchmark simulation harness."""

    step_count: int
    max_steps: int
    allow_menu_linger: bool
    done: bool
    action_bytes: bytearray
    current_seed: int
    mem: Any

    def reset(self, seed: int = 42) -> Any:
        """Resets the environment with the given deterministic seed."""
        ...

    def step(self, action: Any, /) -> tuple[Any, bool, dict[str, Any]]:
        """Advances the simulation by 1 high-level action, returning (state, done, info)."""
        ...

    def replay_bytes(
        self,
        seed: int,
        recorded_bytes: bytes | bytearray | None = None,
        stop_at_byte_idx: int | None = None,
        action_bytes: bytes | bytearray | None = None,
    ) -> Any:
        """Fast-forwards simulation directly in C/compiled code from recorded byte prefix."""
        ...

    def close(self) -> None:
        """Cleans up underlying simulation handles."""
        ...


class BenchmarkAdapter(abc.ABC):
    """
    Pluggable benchmark contract consumed by orchestrator, runner pool,
    summarizer, hypothesis tester, and author agent.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier of the benchmark (e.g. 'nethack', 'nethack_scenarios')."""
        ...

    @abc.abstractmethod
    def create_harness(self, **kwargs: Any) -> EnvHarness:
        """Instantiates a deterministic environment harness."""
        ...

    @abc.abstractmethod
    def progression_metric(self, run: dict[str, Any]) -> float:
        """
        Computes a continuous composite progression score for paired CRN evaluation.
        """
        ...

    def is_horizon_waste(self, run: dict[str, Any]) -> bool:
        """
        Domain-agnostic horizon/stall waste check: True if a non-shard episode
        exhausted step horizon without solving/ascending, or hit a stall termination.
        """
        if run.get("is_shard"):
            return False
        er = str(run.get("end_reason") or "").strip().lower()
        if "stalled" in er:
            return True
        if er == "max_steps" and not bool(
            run.get("is_ascended") or run.get("is_solved")
        ):
            return True
        return False

    def smoke_check(
        self,
        compiled: Any,
        seed: int = 42,
        max_steps: int = 60,
        policy_code: str = "",
        skills: dict[str, str] | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Runs a short single-episode hygiene smoke check using `self.create_harness(...)`.
        Returns `(error_message_or_None, metrics_dict)`.
        """
        from collections import Counter

        runner = compiled.create_runner()
        h_kwargs: dict[str, Any] = {"max_steps": max_steps}
        if self.name == "nethack" and policy_code:
            from lox.env.harness import policy_handles_menus

            h_kwargs["allow_menu_linger"] = policy_handles_menus(policy_code, skills)
        harness = self.create_harness(**h_kwargs)
        try:
            st = harness.reset(seed=seed)
            acts: list[str] = []
            fn_turns: Counter[str] = Counter()
            for _ in range(max_steps):
                act, active_fn = runner.next_action(st)
                fn_turns[str(active_fn or "root")] += 1
                st, done, info = harness.step(act)
                acts.append(str(info.get("action_name", "")))
                if done:
                    break
            exc_cnt = int(getattr(runner, "policy_exceptions", 0) or 0)
            if exc_cnt > 0:
                last_err = str(
                    getattr(runner, "last_exception_msg", "") or "policy exception"
                )
                return (
                    f"Runtime policy exception(s) ({exc_cnt}): {last_err}",
                    {"fn_turns": dict(fn_turns)},
                )
            non_wait = [a for a in acts if a != "WAIT"]
            if len(acts) >= 20 and (
                not non_wait or len(non_wait) <= max(1, len(acts) // 10)
            ):
                return (
                    f"WAIT-dominant deadlock on smoke check ({len(non_wait)}/{len(acts)} non-WAIT actions).",
                    {"fn_turns": dict(fn_turns)},
                )
            total_t = sum(fn_turns.values())
            wrapper_t = sum(
                fn_turns.get(w, 0) for w in ("root", "wrapped", "_osc_gate")
            )
            return None, {
                "fn_turns": dict(fn_turns),
                "wrapper_ratio": float(wrapper_t / max(1, total_t)),
            }
        except Exception as exc:
            return f"Smoke runtime error: {type(exc).__name__}: {exc}", {}
        finally:
            harness.close()

    @abc.abstractmethod
    def extract_milestones_and_stalls(
        self,
        ticks: list[dict[str, Any]],
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Domain-agnostic timeline and stall analysis.
        Returns (summary_digest, game_log, stats_dict).
        """
        ...

    @abc.abstractmethod
    def get_api_docs(self) -> str:
        """Returns the pure 1-to-1 GameState and Action API specification (zero heuristics)."""
        ...

    @abc.abstractmethod
    def search_knowledge_base(self, query: str, top_k: int = 3) -> str:
        """Searches the benchmark's offline wiki, manual, or achievement tech-tree index."""
        ...
