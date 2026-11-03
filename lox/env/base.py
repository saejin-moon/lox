"""
Abstract Benchmark Adapter & Environment Harness Protocol ("Hard Code Nothing").
Defines the clean, pluggable contract for running synthesis across NetHack, Craftax,
and external reinforcement learning / synthesis benchmarks.
"""

from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvHarness(Protocol):
    """Protocol for a deterministic benchmark simulation harness."""

    step_count: int
    done: bool
    action_bytes: bytearray
    current_seed: int
    mem: Any

    def reset(self, seed: int = 42) -> Any:
        """Resets the environment with the given deterministic seed."""
        ...

    def step(self, action: Any) -> tuple[Any, bool, dict[str, Any]]:
        """Advances the simulation by 1 high-level action, returning (state, done, info)."""
        ...

    def replay_bytes(
        self,
        seed: int,
        recorded_bytes: bytes | bytearray | None = None,
        stop_at_byte_idx: int | None = None,
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
        """Unique identifier of the benchmark (e.g. 'nethack', 'craftax')."""
        ...

    @abc.abstractmethod
    def create_harness(self, **kwargs: Any) -> EnvHarness:
        """Instantiates a deterministic environment harness."""
        ...

    @abc.abstractmethod
    def progression_metric(self, run: dict[str, Any]) -> float:
        """
        Computes a continuous composite progression score for paired confidence sequences
        and Wilcoxon signed-rank tests.
        """
        ...

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
