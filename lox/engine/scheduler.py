"""
Mode Scheduler (`ModeScheduler`).
Schedules the 3 core policy synthesis modes (`CREATE`, `WRITE`, `REFACTOR`):
1. Generation 0 (or empty policy): `CREATE` (formerly `FULL_WRITE`)
2. Focused synthesis / modification: `WRITE` (formerly `FUNC_MOD_ADD`, max 2 functions)
3. Modular skill extraction when `policy.py` reaches >= 450 lines: `REFACTOR` (formerly `STRUCTURAL_REFACTOR`)
"""

from __future__ import annotations

from collections import deque
from typing import Any

MODE_ALIASES: dict[str, str] = {
    "CREATE": "CREATE",
    "FULL_WRITE": "CREATE",
    "WRITE": "WRITE",
    "FUNC_MOD_ADD": "WRITE",
    "REFACTOR": "REFACTOR",
    "STRUCTURAL_REFACTOR": "REFACTOR",
    "STRUCTURAL_REWRITE": "REFACTOR",
}


def normalize_mode(mode: str | None) -> str:
    """Normalizes legacy or lowercase mode names to canonical (`CREATE`, `WRITE`, `REFACTOR`)."""
    if not mode:
        return "WRITE"
    u = str(mode).strip().upper()
    return MODE_ALIASES.get(u, u)


class ModeScheduler:
    """Determines which Modification Mode to trigger at each generation (CREATE, WRITE, REFACTOR)."""

    def __init__(
        self,
        use_bandit_after_gen: int = 50,
        refactor_line_threshold: int = 450,
        refactor_min_interval: int = 4,
        **kwargs: Any,
    ):
        self.use_bandit_after_gen = int(use_bandit_after_gen)
        self.refactor_line_threshold = int(refactor_line_threshold)
        self.refactor_min_interval = int(refactor_min_interval)
        self.history: dict[str, deque[float]] = {
            "WRITE": deque(maxlen=10),
            "REFACTOR": deque(maxlen=10),
        }
        self.counts: dict[str, int] = {
            "WRITE": 0,
            "REFACTOR": 0,
        }
        self.gens_since_refactor = 999

    def select_mode(
        self,
        generation: int,
        has_policy: bool,
        has_tunable_params: bool = False,
        forced_mode: str | None = None,
        consecutive_failures: int = 0,
        policy_line_count: int = 0,
    ) -> str:
        if forced_mode:
            return normalize_mode(forced_mode)

        if generation == 0 or not has_policy:
            self.gens_since_refactor = 0
            return "CREATE"

        self.gens_since_refactor += 1

        if (
            policy_line_count >= self.refactor_line_threshold
            and self.gens_since_refactor >= self.refactor_min_interval
        ):
            self.gens_since_refactor = 0
            self.counts["REFACTOR"] += 1
            return "REFACTOR"

        self.counts["WRITE"] += 1
        return "WRITE"

    def record_promotion(self, mode: str = "WRITE") -> None:
        """Records that a candidate policy was promoted."""
        pass

    def record_outcome(
        self,
        mode: str,
        mean_delta: float,
        wall_sec: float,
        passed: bool = False,
    ) -> None:
        canon = normalize_mode(mode)
        if canon in self.history:
            minutes = max(0.05, wall_sec / 60.0)
            credit = max(0.0, float(mean_delta)) / minutes
            self.history[canon].append(credit)
