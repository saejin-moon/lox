"""
Hybrid Mode Scheduler (`ModeScheduler`).
Schedules the 3 core policy synthesis modes (`CREATE`, `WRITE`, `REFACTOR`) plus optional `PARAM_TUNE` / `MACRO_WRITE`:
1. Generation 0 (or empty policy): `CREATE` (formerly `FULL_WRITE`)
2. Focused synthesis / modification: `WRITE` (formerly `FUNC_MOD_ADD`)
3. Modular skill extraction when `policy.py` grows large: `REFACTOR` (formerly `STRUCTURAL_REFACTOR`)
4. Optional `PARAM_TUNE` / `MACRO_WRITE` at configured intervals.
"""

from __future__ import annotations

import math
from collections import deque

MODE_ALIASES: dict[str, str] = {
    "CREATE": "CREATE",
    "FULL_WRITE": "CREATE",
    "WRITE": "WRITE",
    "FUNC_MOD_ADD": "WRITE",
    "REFACTOR": "REFACTOR",
    "STRUCTURAL_REFACTOR": "REFACTOR",
    "STRUCTURAL_REWRITE": "REFACTOR",
    "PARAM_TUNE": "PARAM_TUNE",
    "MACRO_WRITE": "MACRO_WRITE",
}


def normalize_mode(mode: str | None) -> str:
    """Normalizes legacy or lowercase mode names to canonical (`CREATE`, `WRITE`, `REFACTOR`, `PARAM_TUNE`, `MACRO_WRITE`)."""
    if not mode:
        return "WRITE"
    u = str(mode).strip().upper()
    return MODE_ALIASES.get(u, u)


class ModeScheduler:
    """Determines which Modification Mode to trigger at each generation."""

    def __init__(
        self,
        param_tune_every: int = 100,
        macro_every: int = 0,
        use_bandit_after_gen: int = 50,
    ):
        self.param_tune_every = max(1, int(param_tune_every))
        self.macro_every = int(macro_every)
        self.use_bandit_after_gen = int(use_bandit_after_gen)
        self.history: dict[str, deque[float]] = {
            "WRITE": deque(maxlen=10),
            "PARAM_TUNE": deque(maxlen=10),
            "MACRO_WRITE": deque(maxlen=10),
            "REFACTOR": deque(maxlen=10),
        }
        self.counts: dict[str, int] = {
            "WRITE": 0,
            "PARAM_TUNE": 0,
            "MACRO_WRITE": 0,
            "REFACTOR": 0,
        }
        self._func_mods_since_tune = 0
        self._promoted_since_tune = 0
        self._gens_since_macro = 0
        self._last_tune_gen = 0
        self._last_macro_gen = 0
        self.gens_since_refactor = 999

    def select_mode(
        self,
        generation: int,
        has_policy: bool,
        has_tunable_params: bool = True,
        forced_mode: str | None = None,
        consecutive_failures: int = 0,
        policy_line_count: int = 0,
    ) -> str:
        if forced_mode:
            return normalize_mode(forced_mode)

        if generation == 0 or not has_policy:
            self._promoted_since_tune = 1
            self.gens_since_refactor = 0
            return "CREATE"

        self.gens_since_refactor += 1

        if policy_line_count >= 450 and self.gens_since_refactor >= 4:
            self.gens_since_refactor = 0
            self.counts["REFACTOR"] += 1
            return "REFACTOR"

        self._gens_since_macro += 1

        # Periodic Macro Synthesis (only if enabled > 0)
        if self.macro_every > 0 and (
            (
                generation > 0
                and generation % self.macro_every == 0
                and (generation - self._last_macro_gen) >= self.macro_every
            )
            or self._gens_since_macro >= self.macro_every
        ):
            self._gens_since_macro = 0
            self._last_macro_gen = generation
            self.counts["MACRO_WRITE"] += 1
            return "MACRO_WRITE"

        can_tune = (
            has_tunable_params
            and self.param_tune_every > 0
            and self._promoted_since_tune >= 1
        )

        # Periodic Parameter Finetuning every `param_tune_every` function modifications (only if promoted changes occurred)
        if can_tune and (
            (
                generation > 0
                and generation % self.param_tune_every == 0
                and (generation - self._last_tune_gen) >= self.param_tune_every
            )
            or self._func_mods_since_tune >= self.param_tune_every
        ):
            self._func_mods_since_tune = 0
            self._promoted_since_tune = 0
            self._last_tune_gen = generation
            self.counts["PARAM_TUNE"] += 1
            return "PARAM_TUNE"

        # Optional Extreme-Value UCB bandit override after sufficient warmup generations
        total_pulls = sum(self.counts.values())
        if generation >= self.use_bandit_after_gen and total_pulls >= 6:
            candidates = ["WRITE"]
            if (
                self.macro_every > 0
                and self._gens_since_macro >= max(6, self.macro_every)
                and (generation - self._last_macro_gen) >= 6
            ):
                candidates.append("MACRO_WRITE")
            if can_tune and (generation - self._last_tune_gen) >= self.param_tune_every:
                candidates.append("PARAM_TUNE")
            max_credit = max(
                (max(self.history[m]) if self.history[m] else 0.0) for m in candidates
            )
            best_m = "WRITE"
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
            if best_m == "WRITE":
                self._func_mods_since_tune += 1
            elif best_m == "PARAM_TUNE":
                self._func_mods_since_tune = 0
                self._promoted_since_tune = 0
                self._last_tune_gen = generation
            elif best_m == "MACRO_WRITE":
                self._gens_since_macro = 0
                self._last_macro_gen = generation
            return best_m

        self._func_mods_since_tune += 1
        self.counts["WRITE"] += 1
        return "WRITE"

    def record_promotion(self, mode: str = "WRITE") -> None:
        """Records that a candidate policy was promoted, enabling PARAM_TUNE."""
        self._promoted_since_tune += 1

    def record_outcome(
        self,
        mode: str,
        mean_delta: float,
        wall_sec: float,
        passed: bool = False,
    ) -> None:
        canon = normalize_mode(mode)
        if passed and canon in ("WRITE", "REFACTOR", "CREATE"):
            self._promoted_since_tune += 1
        if canon in self.history:
            minutes = max(0.05, wall_sec / 60.0)
            credit = max(0.0, float(mean_delta)) / minutes
            self.history[canon].append(credit)
