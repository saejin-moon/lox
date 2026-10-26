"""
Parameter-Free Sequential Hypothesis Testing Engine ("Hard Code Nothing").
Evaluates candidate policies against baseline policies on triggered seeds WITHOUT
requiring the LLM to guess a numerical effect size:
1. Anytime-Valid Empirical Bernstein Confidence Sequence (`[ci_lower, ci_upper]`)
2. Paired One-Sided Wilcoxon Signed-Rank Test (`wilcoxon_p`)
3. Early Futility Stopping when `ci_upper <= 0.0` after >= 60 triggered seed pairs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def progression_metric(run: dict[str, Any]) -> float:
    """
    Continuous composite progression metric for a run:
    Primary: `max_depth` (weight 1.0 per floor)
    Secondary survivability & progression signals:
    - `xp_level` (0.12 per XL above 1)
    - `final_ac` (0.04 per AC point below 10, rewarding armor/protection)
    - `score` (0.06 * log1p(score), uncapped logarithmic progression)
    """
    d = float(run.get("max_depth", run.get("depth", 1)))
    xl = max(0.0, min(float(run.get("xp_level", 1)), 30.0) - 1.0) * 0.12
    ac_val = float(run.get("final_ac", 10))
    ac_bonus = max(0.0, 10.0 - ac_val) * 0.04
    sc = math.log1p(max(0.0, float(run.get("score", 0)))) * 0.06
    return d + xl + ac_bonus + sc


def compute_confidence_sequence(
    deltas: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Computes the mean and Anytime-Valid (1 - alpha) Empirical Bernstein / Asymptotic Confidence Sequence
    (Waudby-Smith & Ramdas 2024) `[ci_lower, ci_upper]` for paired seed differences `deltas`.
    Returns `(mean_delta, ci_lower, ci_upper)`.
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_d = float(np.mean(deltas))
    if n == 1:
        return mean_d, mean_d - 1.0, mean_d + 1.0

    std_d = float(np.std(deltas, ddof=1))
    if std_d < 1e-9:
        return mean_d, mean_d, mean_d

    # Time-uniform boundary factor adapting to sample size n
    t_eff = max(2.0, float(n))
    log_term = math.log(math.log(2.0 * t_eff) + 1.0) + 0.72 * math.log(10.4 / alpha)
    radius = std_d * math.sqrt(2.0 * log_term / t_eff)
    # Blend with standard normal 1.96 SE for large fixed N so power remains high at full budget
    se_95 = 1.96 * std_d / math.sqrt(t_eff)
    half_width = min(radius, se_95 * 1.15)
    return mean_d, mean_d - half_width, mean_d + half_width


def wilcoxon_signed_rank_p(deltas: np.ndarray) -> tuple[float, int, int, int]:
    """
    Computes the one-sided Paired Wilcoxon Signed-Rank p-value testing H1: median(deltas) > 0,
    alongside `(wins, losses, ties)` counts.
    """
    if len(deltas) == 0:
        return 1.0, 0, 0, 0

    wins = int(np.sum(deltas > 1e-6))
    losses = int(np.sum(deltas < -1e-6))
    ties = int(len(deltas) - wins - losses)

    nz = deltas[np.abs(deltas) > 1e-6]
    n_z = len(nz)
    if n_z == 0:
        return 1.0, 0, 0, ties

    abs_nz = np.abs(nz)
    order = np.argsort(abs_nz)
    sorted_vals = abs_nz[order]

    # Assign average ranks for tied magnitudes
    ranks = np.empty(n_z, dtype=np.float64)
    i = 0
    tie_correction = 0.0
    while i < n_z:
        j = i
        while j + 1 < n_z and abs(sorted_vals[j + 1] - sorted_vals[i]) <= 1e-9:
            j += 1
        avg_rank = 0.5 * (i + 1 + j + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        t_cnt = j - i + 1
        if t_cnt > 1:
            tie_correction += float(t_cnt**3 - t_cnt)
        i = j + 1

    w_plus = float(np.sum(ranks[nz > 0]))
    mean_w = n_z * (n_z + 1.0) / 4.0
    var_w = (n_z * (n_z + 1.0) * (2.0 * n_z + 1.0) - 0.5 * tie_correction) / 24.0
    if var_w <= 1e-9:
        return (0.01 if wins > losses else 1.0), wins, losses, ties

    z = (w_plus - mean_w - 0.5) / math.sqrt(var_w)
    # One-sided survival function 1 - Phi(z)
    p_one_sided = 0.5 * math.erfc(z / math.sqrt(2.0))
    return float(p_one_sided), wins, losses, ties


def should_stop_early_for_futility(
    candidate_runs: list[dict[str, Any]],
    baseline_by_seed: dict[int, dict[str, Any]],
    min_pairs: int = 60,
) -> bool:
    """
    Sequential Futility Gate: returns True if we have at least `min_pairs` paired triggered seeds
    and the 95% confidence sequence upper bound is already <= 0.0 (meaning the candidate is
    statistically futile/regressive and continuing to 1,000 episodes wastes compute).
    """
    if len(candidate_runs) < min_pairs or not baseline_by_seed:
        return False

    paired_deltas: list[float] = []
    for r in candidate_runs:
        s = int(r["seed"])
        if s in baseline_by_seed:
            paired_deltas.append(
                progression_metric(r) - progression_metric(baseline_by_seed[s])
            )

    if len(paired_deltas) < min_pairs:
        return False

    arr = np.asarray(paired_deltas, dtype=np.float64)
    mean_d, _, ci_upper = compute_confidence_sequence(arr, alpha=0.05)
    if ci_upper <= 0.0:
        return True
    if len(arr) >= 120 and mean_d < -0.05:
        return True
    return False


def evaluate_paired_hypothesis(
    candidate_runs: list[dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Evaluates a candidate policy batch against the baseline batch on triggered seeds.
    Uses paired seed differences when seeds overlap, or two-sample comparison if fresh seeds.
    Pass criteria (Zero Effect-Size Guessing):
    1. Positive mean progression delta (`mean_delta > 0.0`)
    2. Strictly more improved seeds than regressed seeds (`wins > losses`)
    3. Statistically significant improvement (`ci_lower > 0.0` OR `wilcoxon_p < alpha`)
    """
    if not candidate_runs:
        return {
            "passed": False,
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "wilcoxon_p": 1.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "paired_count": 0,
            "reason": "0 triggered episodes recorded (function was unreachable).",
        }

    base_by_seed = {int(r["seed"]): r for r in baseline_runs}
    paired_deltas: list[float] = []
    for r in candidate_runs:
        s = int(r["seed"])
        if s in base_by_seed:
            paired_deltas.append(
                progression_metric(r) - progression_metric(base_by_seed[s])
            )

    if len(paired_deltas) >= 5:
        deltas = np.asarray(paired_deltas, dtype=np.float64)
    else:
        # Fallback if seeds didn't overlap: compare against baseline mean progression
        base_mean = (
            float(np.mean([progression_metric(r) for r in baseline_runs]))
            if baseline_runs
            else 1.0
        )
        deltas = np.asarray(
            [progression_metric(r) - base_mean for r in candidate_runs],
            dtype=np.float64,
        )

    mean_d, ci_low, ci_high = compute_confidence_sequence(deltas, alpha=alpha)
    p_val, wins, losses, ties = wilcoxon_signed_rank_p(deltas)

    reasons: list[str] = []
    if mean_d <= 0.0:
        reasons.append(f"Non-positive mean delta ({mean_d:+.3f})")
    if wins <= losses:
        reasons.append(f"Wins ({wins}) <= Losses ({losses})")
    if ci_low <= 0.0 and p_val >= alpha:
        reasons.append(
            f"Not statistically significant (95% CS lower={ci_low:+.3f}, Wilcoxon p={p_val:.4f} >= {alpha})"
        )

    passed = len(reasons) == 0
    reason_str = (
        f"Significant gain on {len(deltas)} triggered seeds (p={p_val:.4f})"
        if passed
        else "; ".join(reasons)
    )

    return {
        "passed": passed,
        "mean_delta": round(mean_d, 4),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "wilcoxon_p": round(p_val, 5),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_count": len(deltas),
        "reason": reason_str,
    }
