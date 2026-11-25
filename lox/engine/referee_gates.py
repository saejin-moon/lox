"""
Statistical Promotion Gates, Safety Invariants, Stepping-Stone Selection,
and Package Fingerprinting helpers for `SwarmReferee`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lox.author.swarm_peer import PeerCandidate
from lox.core.contracts import BatchEvaluation, split_full_and_shard_runs
from lox.engine.hypothesis import (
    cvar_bottom_quantile,
    interquartile_mean,
    progression_metric,
)
from lox.lang.validator import is_ast_equivalent


def opt_float(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Explicit `is None` float extraction — NEVER uses `or default` (safe when value == 0.0)."""
    val = d.get(key)
    return float(default) if val is None else float(val)


def opt_int(d: dict[str, Any], key: str, default: int = 0) -> int:
    """Explicit `is None` int extraction — NEVER uses `or default` (safe when value == 0)."""
    val = d.get(key)
    return int(default) if val is None else int(val)


@dataclass
class TournamentVerdict:
    """Result of the head-to-head CRN evaluation among incubated peer candidates."""

    promoted: bool
    winner_id: str
    winner_policy_code: str
    winner_skills: dict[str, str]
    winner_batch_eval: BatchEvaluation
    incumbent_avg_depth: float
    incumbent_avg_score: float
    winner_avg_depth: float
    winner_avg_score: float
    target_functions: list[str] = field(default_factory=list)
    scoreboard: list[dict[str, Any]] = field(default_factory=list)
    all_runner_up_batches: list[dict[str, Any]] = field(default_factory=list)
    incumbent_batch_eval: BatchEvaluation | None = None
    attempts_summary: list[dict[str, Any]] = field(default_factory=list)
    holdout_avg_depth: float | None = None
    holdout_iqm_delta: float | None = None
    holdout_rejected: bool = False
    best_pair: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    select_batches: dict[str, BatchEvaluation] = field(default_factory=dict)
    select_pair_by_peer: dict[str, dict[str, Any]] = field(default_factory=dict)
    select_stats_by_peer: dict[str, dict[str, Any]] = field(default_factory=dict)
    select_inc_avg_depth: float = 0.0
    select_inc_auto_esc: int = 0
    confirm_challenger_eval: tuple[str, BatchEvaluation] | None = None
    qualifying_stepping_stone: dict[str, Any] | None = None


_FIXABLE_GUARD_REASONS = frozenset(
    {
        "guard_failed",
        "cvar25_tail_regression",
        "auto_esc_regression",
        "horizon_waste_regression",
    }
)


def _binomial_sign_p(wins: int, losses: int) -> float:
    """Exact one-sided binomial sign-test p-value P(Bin(W+L, 0.5) >= W) using big-int division."""
    w = max(0, int(wins))
    los = max(0, int(losses))
    n = w + los
    if n <= 0 or w <= 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(w, n + 1)) / (1 << n))


def select_qualifying_stepping_stone(
    generation: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Empirically calibrated Hybrid Stepping-Stone selector:
    - Hard prerequisites: `policy_exceptions == 0`, `wins >= 3`, `wins > losses`,
      and `max(winsor_delta, iqm_delta) > 0.0`.
    - Signal gate (A or B):
      A. Statistical signal: `min(sign_perm_p, wilcoxon_p, binomial_sign_p(wins, losses)) < 0.25`.
      B. Fixable guard-blocked upside: `reject_reason` in `_FIXABLE_GUARD_REASONS`,
         `wins - losses >= 2`, and `max(winsor_delta, iqm_delta) >= 0.05`.
    """
    qualified: list[tuple[tuple[int, float, float, int], dict[str, Any]]] = []
    for item in entries:
        cand: PeerCandidate | None = item.get("candidate")
        if cand is None or not str(cand.policy_code or "").strip():
            continue
        if (
            str(getattr(cand, "peer_id", "")) == "proactive_2"
            or str(getattr(cand, "track", "")) == "proactive_branch"
        ):
            continue
        exc = opt_int(item, "policy_exceptions", 0)
        if exc > 0:
            continue
        wins = opt_int(item, "wins", 0)
        losses = opt_int(item, "losses", 0)
        if wins < 3 or wins <= losses:
            continue
        iqm_d = opt_float(item, "iqm_delta", 0.0)
        win_d = opt_float(item, "winsor_delta", iqm_d)
        best_d = max(win_d, iqm_d)
        if best_d <= 0.0:
            continue
        wp = opt_float(item, "wilcoxon_p", 1.0)
        spp = opt_float(item, "sign_perm_p", wp)
        sp = _binomial_sign_p(wins, losses)
        min_p = min(spp, wp, sp)
        rej = str(item.get("reject_reason", "") or "").strip()
        src_kind = str(item.get("source_run_kind", "select") or "select")

        stat_ok = min_p < 0.25
        guard_upside_ok = (
            rej in _FIXABLE_GUARD_REASONS and (wins - losses) >= 2 and best_d >= 0.05
        )
        if not (stat_ok or guard_upside_ok):
            continue

        is_guard_blocked = rej in _FIXABLE_GUARD_REASONS
        if src_kind == "confirm_challenger" and not is_guard_blocked and best_d > 0.05:
            tier = 0
        elif not is_guard_blocked:
            tier = 1
        else:
            tier = 2

        mech = str(
            (cand.hypothesis_yaml or {}).get("mechanism", "")
            or cand.claimed_label
            or ""
        )[:1000]
        tfs = (
            sorted(cand.touched_branches)
            or sorted(cand.touched_skills)
            or [cand.claimed_deficit_id]
        )
        stone = {
            "stone_id": f"stone_g{int(generation)}_{cand.peer_id}",
            "generation": int(generation),
            "source_peer_id": str(cand.peer_id),
            "source_run_kind": src_kind,
            "source_batch_id": str(item.get("source_batch_id", "") or ""),
            "revision_count": 0,
            "status": "active",
            "policy_code": str(cand.policy_code),
            "skills": dict(cand.skills or {}),
            "mechanism": mech,
            "target_functions": tfs,
            "wins": wins,
            "losses": losses,
            "iqm_delta": iqm_d,
            "winsor_delta": win_d,
            "wilcoxon_p": wp,
            "sign_p": min(spp, sp),
            "reject_reason": rej,
            "seeds_gained": list(item.get("seeds_gained") or []),
            "seeds_lost": list(item.get("seeds_lost") or []),
        }
        rank_key = (tier, min_p, -best_d, -(wins - losses))
        qualified.append((rank_key, stone))
    if not qualified:
        return None
    qualified.sort(key=lambda pair: pair[0])
    return qualified[0][1]


def _package_fingerprint(policy_code: str, skills: dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(policy_code.strip().encode("utf-8", errors="replace"))
    for k in sorted(skills.keys()):
        h.update(f"\n##{k}\n".encode("utf-8"))
        h.update(skills[k].strip().encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _is_package_ast_equivalent(
    pol_a: str,
    skills_a: dict[str, str],
    pol_b: str,
    skills_b: dict[str, str],
) -> bool:
    if not pol_a.strip() or not pol_b.strip():
        return False
    if set(skills_a.keys()) != set(skills_b.keys()):
        return False
    if not is_ast_equivalent(pol_a, pol_b):
        return False
    return all(is_ast_equivalent(skills_a[k], skills_b[k]) for k in skills_a)


def _clone_incumbent_eval(
    inc_batch_eval: BatchEvaluation, generation: int
) -> BatchEvaluation:
    """Clones incumbent runs under a fresh generation-scoped `batch_id` so prior `batches`/`runs` rows are never overwritten."""
    new_bid = f"batch_swarm_g{generation}_inc_retained_{int(time.time() * 1000)}"
    cloned = dict(inc_batch_eval)
    cloned["batch_id"] = new_bid
    cloned_runs: list[dict[str, Any]] = []
    for idx, r in enumerate(inc_batch_eval.get("runs", []) or []):
        rc = dict(r)
        new_rid = f"{new_bid}_r{idx:05d}"
        rc["batch_id"] = new_bid
        rc["run_id"] = new_rid
        orig_ticks = rc.get("ticks")
        if isinstance(orig_ticks, list) and orig_ticks:
            rc["ticks"] = [
                {**t, "run_id": new_rid} if isinstance(t, dict) else t
                for t in orig_ticks
            ]
        cloned_runs.append(rc)
    cloned["runs"] = cloned_runs
    full_runs, shard_runs = split_full_and_shard_runs(cloned_runs)
    cloned["full_runs"] = full_runs
    cloned["shard_runs"] = shard_runs
    return BatchEvaluation(cloned)


def _count_ast_nodes(policy_code: str, skills: dict[str, str]) -> int:
    """Counts total AST nodes across policy.py + skills (un-gameable complexity metric)."""
    total = 0
    for src in (policy_code, *(skills or {}).values()):
        s = str(src or "")
        if not s.strip():
            continue
        try:
            total += sum(1 for _ in ast.walk(ast.parse(s)))
        except SyntaxError:
            total += len(s.splitlines()) * 4
    return total


def _assemble_batch_eval(
    batch_id: str,
    runs: list[dict[str, Any]],
) -> BatchEvaluation:
    full_runs, shard_runs = split_full_and_shard_runs(runs)
    stat_runs = full_runs if full_runs else runs
    depths = [int(r.get("max_depth", r.get("depth", 1)) or 1) for r in stat_runs]
    scores = [float(r.get("score", 0.0) or 0.0) for r in stat_runs]
    turns = [float(r.get("turns", 0.0) or 0.0) for r in stat_runs]
    return BatchEvaluation(
        {
            "batch_id": batch_id,
            "runs": runs,
            "full_runs": full_runs,
            "shard_runs": shard_runs,
            "episodes_recorded": len(runs),
            "episodes_discarded": 0,
            "raw_episodes_run": len(runs),
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "p90_depth": float(np.percentile(depths, 90)) if depths else 0.0,
            "max_depth": int(max(depths)) if depths else 0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": int(max(scores)) if scores else 0,
            "avg_turns": float(np.mean(turns)) if turns else 0.0,
            "early_stopped": False,
            "unreachable_abort": False,
        }
    )


def _extract_eval_stats(
    batch_eval: BatchEvaluation,
    loc_total: int,
    benchmark: str = "nethack",
    ast_nodes: int = 0,
) -> dict[str, Any]:
    runs = batch_eval.get("runs", []) or []
    if not runs:
        return {
            "avg_depth": opt_float(batch_eval, "avg_depth", 1.0),
            "max_depth": opt_int(batch_eval, "max_depth", 1),
            "avg_score": opt_float(batch_eval, "avg_score", 0.0),
            "avg_ac": 10.0,
            "avg_xp_level": 1.0,
            "avg_turns": 0.0,
            "iqm_prog": 1.0,
            "mean_prog": 1.0,
            "cvar25_prog": 1.0,
            "stall_count": 0,
            "hard_stall_count": 0,
            "horizon_waste_count": 0,
            "horizon_waste_rate": 0.0,
            "stall_rate": 0.0,
            "auto_esc_total": 0,
            "policy_exceptions": 0,
            "exc_runs": 0,
            "run_count": 0,
            "top_tracebacks": [],
            "loc_total": loc_total,
            "ast_nodes": ast_nodes or loc_total * 4,
        }

    from lox.env import get_adapter

    adapter = get_adapter(benchmark or "nethack")
    depths = [
        float(max(int(r.get("depth", 1) or 1), int(r.get("max_depth", 1) or 1)))
        for r in runs
    ]
    scores = [float(r.get("score", 0.0) or 0.0) for r in runs]
    acs = [
        float(r.get("final_ac", 10) if r.get("final_ac") is not None else 10)
        for r in runs
    ]
    xls = [float(r.get("xp_level", 1) or 1) for r in runs]
    turns = [float(r.get("turns", 0) or 0) for r in runs]
    progs = [progression_metric(r, benchmark=benchmark) for r in runs]

    confinement_stalls = sum(
        1 for r in runs if "CONFINEMENT_STALL" in str(r.get("summary_digest", ""))
    )
    horizon_waste_cnt = sum(1 for r in runs if adapter.is_horizon_waste(r))
    hard_stalls = sum(
        1 for r in runs if "stalled" in str(r.get("end_reason", "")).lower()
    )
    auto_esc_total = sum(int(r.get("auto_esc_count", 0) or 0) for r in runs)
    exc_total = sum(int(r.get("policy_exceptions", 0) or 0) for r in runs)
    exc_runs = sum(1 for r in runs if int(r.get("policy_exceptions", 0) or 0) > 0)
    tbs: list[str] = []
    for r in runs:
        if int(r.get("policy_exceptions", 0) or 0) <= 0:
            continue
        mj = r.get("milestones_json")
        if isinstance(mj, str) and mj:
            try:
                mo = json.loads(mj)
                tbs.extend(
                    mo.get("exception_tracebacks") or mo.get("leaf_errors") or []
                )
            except Exception:
                pass
        elif isinstance(r.get("milestones"), dict):
            mo = r["milestones"]
            tbs.extend(mo.get("exception_tracebacks") or mo.get("leaf_errors") or [])
        er = str(r.get("end_reason", ""))
        dm = str(r.get("death_message", ""))
        if er or dm:
            tbs.append(f"{er}: {dm}"[:140])

    n = float(len(runs))
    return {
        "avg_depth": sum(depths) / n,
        "max_depth": int(max(depths)),
        "avg_score": sum(scores) / n,
        "avg_ac": sum(acs) / n,
        "avg_xp_level": sum(xls) / n,
        "avg_turns": sum(turns) / n,
        "iqm_prog": interquartile_mean(progs),
        "mean_prog": sum(progs) / n,
        "cvar25_prog": cvar_bottom_quantile(progs, q=0.25),
        "stall_count": confinement_stalls,
        "hard_stall_count": hard_stalls,
        "horizon_waste_count": horizon_waste_cnt,
        "horizon_waste_rate": horizon_waste_cnt / n,
        "stall_rate": horizon_waste_cnt / n,
        "auto_esc_total": auto_esc_total,
        "policy_exceptions": exc_total,
        "exc_runs": exc_runs,
        "run_count": len(runs),
        "top_tracebacks": list(dict.fromkeys(tbs))[:5],
        "loc_total": loc_total,
        "ast_nodes": ast_nodes or loc_total * 4,
    }


def _passes_safety_invariants(
    cand_stats: dict[str, Any],
    inc_stats: dict[str, Any],
    *,
    has_base: bool,
    pair_stats: dict[str, Any] | None = None,
    stage: str = "confirm",
) -> tuple[bool, str]:
    """
    4 benchmark-agnostic safety invariants (paired & statistically calibrated when `pair_stats` is supplied):
      1. Runtime exceptions: cand_exc <= inc_exc (0 when inc_exc == 0).
      2. Harness prompt dismissals: cand_auto_esc <= max(inc_auto_esc, 2).
      3. Paired horizon waste (max_steps / stalled*): McNemar binomial discordance check.
      4. Paired bottom-quartile tail risk (CVaR-25%): requires BOTH unpaired CVaR drop beyond stage
         tolerance AND paired incumbent-tail drop exceeding 1.5*SE.
    """
    cand_exc = opt_int(cand_stats, "policy_exceptions", 0)
    if not has_base:
        if cand_exc > 0:
            return False, "policy_exceptions_regression"
        return True, ""

    inc_exc = opt_int(inc_stats, "policy_exceptions", 0)
    if cand_exc > inc_exc:
        return False, "policy_exceptions_regression"

    cand_auto_esc = opt_int(cand_stats, "auto_esc_total", 0)
    inc_auto_esc = opt_int(inc_stats, "auto_esc_total", 0)
    if cand_auto_esc > max(inc_auto_esc, 2):
        return False, "auto_esc_regression"

    cand_n = max(1, opt_int(cand_stats, "run_count", 1))
    is_sel = str(stage or "confirm").lower() == "select"

    if pair_stats is not None and "hw_mcnemar_p" in pair_stats:
        hw_w = opt_int(pair_stats, "hw_worsened", 0)
        hw_i = opt_int(pair_stats, "hw_improved", 0)
        hw_p = opt_float(pair_stats, "hw_mcnemar_p", 1.0)
        net_hw = hw_w - hw_i
        min_net = max(3, int((0.10 if is_sel else 0.06) * cand_n))
        p_cut = 0.10 if is_sel else 0.05
        if net_hw >= min_net and hw_p < p_cut:
            return False, "horizon_waste_regression"
    else:
        cand_hw = opt_float(cand_stats, "horizon_waste_rate", 0.0)
        inc_hw = opt_float(inc_stats, "horizon_waste_rate", 0.0)
        hw_tol = max(0.08 if is_sel else 0.05, 1.0 / float(cand_n))
        if cand_hw > inc_hw + hw_tol + 1e-9:
            return False, "horizon_waste_regression"

    cand_cvar = opt_float(cand_stats, "cvar25_prog", 0.0)
    inc_cvar = opt_float(inc_stats, "cvar25_prog", 0.0)
    cvar_tol = (
        max(0.35, 0.08 * abs(inc_cvar)) if is_sel else max(0.25, 0.06 * abs(inc_cvar))
    )
    if cand_cvar < inc_cvar - cvar_tol - 1e-9:
        if pair_stats is not None and "paired_cvar25_delta" in pair_stats:
            p_cvar_d = opt_float(pair_stats, "paired_cvar25_delta", 0.0)
            p_cvar_se = opt_float(pair_stats, "paired_cvar25_se", 0.0)
            if p_cvar_d < -max(0.25, 1.5 * p_cvar_se):
                return False, "cvar25_tail_regression"
        else:
            return False, "cvar25_tail_regression"

    return True, ""


def _has_positive_dominance(pair_stats: dict[str, Any]) -> bool:
    """True iff candidate has net wins, positive mean delta, and non-outlier-driven winsorized delta."""
    wins = opt_int(pair_stats, "wins", 0)
    losses = opt_int(pair_stats, "losses", 0)
    mean_d = opt_float(pair_stats, "mean_delta", 0.0)
    winsor_d = opt_float(pair_stats, "winsor_delta", mean_d)
    return (
        wins > losses
        and mean_d > 0.0
        and (winsor_d > 0.0 or (losses == 0 and winsor_d >= 0.0))
    )


def _meets_select_admission_floor(
    pair_stats: dict[str, Any], *, has_base: bool
) -> bool:
    """Prevents trivial 1W/0L (+0.0002) near-no-ops from hijacking S_confirm."""
    if not has_base:
        return True
    if not _has_positive_dominance(pair_stats):
        return False
    wins = opt_int(pair_stats, "wins", 0)
    losses = opt_int(pair_stats, "losses", 0)
    mean_d = opt_float(pair_stats, "mean_delta", 0.0)
    winsor_d = opt_float(pair_stats, "winsor_delta", mean_d)
    return wins >= 2 or (wins == 1 and losses == 0 and winsor_d >= 0.05)


def _evaluate_confirmation(
    cand_stats: dict[str, Any],
    pair_stats: dict[str, Any],
    inc_stats: dict[str, Any],
    *,
    has_base: bool,
    alpha_cutoff: float = 0.05,
) -> tuple[bool, bool, str, float]:
    """
    Unified v3 confirmation gate:
      1. 4 paired, statistically calibrated safety invariants.
      2. Positive dominance: wins > losses, mean_delta > 0, winsor_delta > 0.
      3. Exact Paired Sign-Permutation p < eff_alpha (with 0.5x alpha bar on >25% AST bloat).
    Returns `(beats_incumbent, safety_ok, reject_reason, eff_alpha)`.
    """
    safety_ok, safety_reason = _passes_safety_invariants(
        cand_stats,
        inc_stats,
        has_base=has_base,
        pair_stats=pair_stats,
        stage="confirm",
    )
    if not safety_ok:
        return False, False, safety_reason, alpha_cutoff
    if not has_base:
        return True, True, "promoted", alpha_cutoff

    cand_ast = opt_int(cand_stats, "ast_nodes", 0)
    inc_ast = opt_int(inc_stats, "ast_nodes", 0)
    bloated = inc_ast >= 50 and cand_ast > int(inc_ast * 1.25)
    eff_alpha = alpha_cutoff * 0.5 if bloated else alpha_cutoff

    n_paired = opt_int(
        pair_stats,
        "paired_count",
        opt_int(pair_stats, "n_paired", 0),
    )
    sign_p = opt_float(pair_stats, "sign_perm_p", 1.0)

    if not _has_positive_dominance(pair_stats):
        return False, True, "non_positive_dominance", eff_alpha

    # For tiny bootstrap smoke batches (N < 5), exact 2^n cannot drop below 0.05.
    if n_paired < 5:
        return True, True, "promoted", eff_alpha

    if sign_p < eff_alpha:
        return True, True, "promoted", eff_alpha
    if bloated and sign_p < alpha_cutoff:
        return False, True, "ast_bloat_below_tightened_alpha", eff_alpha
    return False, True, "below_sign_perm_alpha", eff_alpha


def _selection_rank_key(
    cand_stats: dict[str, Any], pair_stats: dict[str, Any]
) -> tuple[int, int, float, float, float, int]:
    """
    Precision-aware v3 selection ranker:
      1. 3-level confidence tier (`2` for `p < 0.05`, `1` for `p < 0.15`, `0` otherwise).
      2. Inside `cred_tier >= 1`: lower-confidence Winsorized delta (`lcb_score`), `-sign_p`, `winsor_d`.
      3. Inside `cred_tier == 0`: multi-seed margin (`wins - losses >= 2`), then `winsor_delta`,
         then `-sign_p`.
      4. Simplicity tie-breaker (`-ast_nodes`).
    """
    wins = opt_int(pair_stats, "wins", 0)
    losses = opt_int(pair_stats, "losses", 0)
    sign_p = opt_float(pair_stats, "sign_perm_p", 1.0)
    mean_d = opt_float(pair_stats, "mean_delta", 0.0)
    winsor_d = opt_float(pair_stats, "winsor_delta", mean_d)
    lcb = opt_float(pair_stats, "lcb_score", winsor_d)
    if wins > losses and sign_p < 0.05:
        cred_tier = 2
    elif wins > losses and sign_p < 0.15:
        cred_tier = 1
    else:
        cred_tier = 0
    ast_n = opt_int(cand_stats, "ast_nodes", 0)
    if cred_tier >= 1:
        return (
            cred_tier,
            0,
            round(lcb, 4),
            round(-sign_p, 4),
            round(winsor_d, 4),
            -ast_n,
        )
    return (
        0,
        1 if (wins - losses) >= 2 else 0,
        round(winsor_d, 4),
        round(-sign_p, 4),
        round(lcb, 4),
        -ast_n,
    )
