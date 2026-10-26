"""
Hybrid Mode Scheduler (`ModeScheduler`).
Schedules the 4 policy modification modes (`FULL_WRITE`, `FUNC_MOD_ADD`, `PARAM_TUNE`, `MACRO_WRITE`):
1. Generation 0 (or empty policy): `FULL_WRITE`
2. Configurable cycle (`func_mod_interval=3` -> `PARAM_TUNE`, `macro_interval=5` -> `MACRO_WRITE`)
3. Also tracks Extreme-Value UCB bandit credit (`record_outcome`) so mode selection can
   dynamically adapt based on empirical progression yield per compute minute.
"""

from __future__ import annotations

import math
from collections import deque


class ModeScheduler:
    """Determines which Modification Mode to trigger at each generation."""

    def __init__(
        self,
        param_tune_every: int = 3,
        macro_every: int = 5,
        use_bandit_after_gen: int = 15,
    ):
        self.param_tune_every = max(1, int(param_tune_every))
        self.macro_every = max(2, int(macro_every))
        self.use_bandit_after_gen = int(use_bandit_after_gen)
        self.history: dict[str, deque[float]] = {
            "FUNC_MOD_ADD": deque(maxlen=10),
            "PARAM_TUNE": deque(maxlen=10),
            "MACRO_WRITE": deque(maxlen=10),
        }
        self.counts: dict[str, int] = {
            "FUNC_MOD_ADD": 0,
            "PARAM_TUNE": 0,
            "MACRO_WRITE": 0,
        }
        self._func_mods_since_tune = 0
        self._gens_since_macro = 0

    def select_mode(
        self,
        generation: int,
        has_policy: bool,
        has_tunable_params: bool = True,
        forced_mode: str | None = None,
    ) -> str:
        if forced_mode:
            return forced_mode.upper()

        if generation == 0 or not has_policy:
            return "FULL_WRITE"

        self._gens_since_macro += 1

        # Periodic Macro Synthesis so the Author builds up `macros.py` early and regularly
        if (
            generation > 0 and generation % self.macro_every == 0
        ) or self._gens_since_macro >= self.macro_every:
            self._gens_since_macro = 0
            self.counts["MACRO_WRITE"] += 1
            return "MACRO_WRITE"

        # Periodic Parameter Finetuning every `param_tune_every` function modifications
        if has_tunable_params and (
            (generation > 0 and generation % self.param_tune_every == 0)
            or self._func_mods_since_tune >= self.param_tune_every
        ):
            self._func_mods_since_tune = 0
            self.counts["PARAM_TUNE"] += 1
            return "PARAM_TUNE"

        # Optional Extreme-Value UCB bandit override after sufficient warmup generations
        total_pulls = sum(self.counts.values())
        if generation >= self.use_bandit_after_gen and total_pulls >= 6:
            candidates = ["FUNC_MOD_ADD", "MACRO_WRITE"]
            if has_tunable_params:
                candidates.append("PARAM_TUNE")
            max_credit = max(
                (max(self.history[m]) if self.history[m] else 0.0) for m in candidates
            )
            best_m = "FUNC_MOD_ADD"
            best_ucb = -1e9
            for m in candidates:
                q_m = max(self.history[m]) if self.history[m] else 0.1
                norm_q = q_m / (max_credit + 1e-6) if max_credit > 0 else 0.5
                n_m = self.counts[m]
                ucb = norm_q + 0.6 * math.sqrt(
                    2.0 * math.log(total_pulls + 1.0) / (n_m + 1.0)
                )
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_m = m
            self.counts[best_m] += 1
            if best_m == "FUNC_MOD_ADD":
                self._func_mods_since_tune += 1
            elif best_m == "PARAM_TUNE":
                self._func_mods_since_tune = 0
            elif best_m == "MACRO_WRITE":
                self._gens_since_macro = 0
            return best_m

        self._func_mods_since_tune += 1
        self.counts["FUNC_MOD_ADD"] += 1
        return "FUNC_MOD_ADD"

    def record_outcome(self, mode: str, mean_delta: float, wall_sec: float) -> None:
        if mode in self.history:
            minutes = max(0.05, wall_sec / 60.0)
            credit = max(0.0, float(mean_delta)) / minutes
            self.history[mode].append(credit)
