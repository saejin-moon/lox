"""
Deterministic CRN Tournament Referee & AST-Disjoint Orthogonal Merger.
("Hard Code Nothing": zero hardcoded NetHack heuristics, zero LLM review pass.)

Replaces hierarchical `SwarmCriticAgent` + `SwarmMetaAgent.assemble_policy`:
1. Filters incubated `PeerCandidate`s on deterministic conflict & attribution checks.
2. Optionally constructs 1 AST/skill-disjoint orthogonal pair from top-2 survivors
   and verifies compilation before tournament entry.
3. Evaluates Incumbent vs Surviving Peer Candidates head-to-head on paired CRN seeds.
4. Promotes a candidate ONLY if it meets `delta >= 0` vs the Incumbent champion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_peer import PeerCandidate
from lox.core.contracts import BatchEvaluation, split_full_and_shard_runs
from lox.engine.hypothesis import (
    cvar_bottom_quantile,
    evaluate_tournament_pair,
    interquartile_mean,
    progression_metric,
)
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package
from lox.lang.validator import is_ast_equivalent, three_way_ast_merge
from lox.ui.console import ui


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
    import numpy as np

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
            "max_depth": int(max(r.get("max_depth", 1) for r in runs)) if runs else 0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": int(max(r.get("score", 0) for r in runs)) if runs else 0,
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
            "avg_depth": float(batch_eval.get("avg_depth", 1.0) or 1.0),
            "max_depth": int(batch_eval.get("max_depth", 1) or 1),
            "avg_score": float(batch_eval.get("avg_score", 0.0) or 0.0),
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
) -> tuple[bool, str]:
    """
    4 benchmark-agnostic safety invariants (zero hardcoded NetHack fields):
      1. Runtime exceptions: cand_exc <= inc_exc (0 when inc_exc == 0).
      2. Harness prompt dismissals: cand_auto_esc <= max(inc_auto_esc, 2).
      3. All-depth horizon waste (max_steps / stalled*): cand_hw <= inc_hw + max(0.05, 1/N).
      4. Bottom-quartile tail risk (CVaR-25%): cand_cvar25 >= inc_cvar25 - max(0.15, 0.05*|inc_cvar25|).
    """
    cand_exc = int(cand_stats.get("policy_exceptions", 0) or 0)
    if not has_base:
        if cand_exc > 0:
            return False, "policy_exceptions_regression"
        return True, ""

    inc_exc = int(inc_stats.get("policy_exceptions", 0) or 0)
    if cand_exc > inc_exc:
        return False, "policy_exceptions_regression"

    cand_auto_esc = int(cand_stats.get("auto_esc_total", 0) or 0)
    inc_auto_esc = int(inc_stats.get("auto_esc_total", 0) or 0)
    if cand_auto_esc > max(inc_auto_esc, 2):
        return False, "auto_esc_regression"

    cand_n = max(1, int(cand_stats.get("run_count", 0) or 1))
    cand_hw = float(cand_stats.get("horizon_waste_rate", 0.0) or 0.0)
    inc_hw = float(inc_stats.get("horizon_waste_rate", 0.0) or 0.0)
    hw_tol = max(0.05, 1.0 / float(cand_n))
    if cand_hw > inc_hw + hw_tol + 1e-9:
        return False, "horizon_waste_regression"

    cand_cvar = float(cand_stats.get("cvar25_prog", 0.0) or 0.0)
    inc_cvar = float(inc_stats.get("cvar25_prog", 0.0) or 0.0)
    cvar_tol = max(0.15, 0.05 * abs(inc_cvar))
    if cand_cvar < inc_cvar - cvar_tol - 1e-9:
        return False, "cvar25_tail_regression"

    return True, ""


def _has_positive_dominance(pair_stats: dict[str, Any]) -> bool:
    """True iff candidate has net wins, positive mean delta, and non-outlier-driven winsorized delta."""
    wins = int(pair_stats.get("wins", 0) or 0)
    losses = int(pair_stats.get("losses", 0) or 0)
    mean_d = float(pair_stats.get("mean_delta", 0.0) or 0.0)
    winsor_d = float(pair_stats.get("winsor_delta", mean_d) or 0.0)
    return (
        wins > losses
        and mean_d > 0.0
        and (winsor_d > 0.0 or (losses == 0 and winsor_d >= 0.0))
    )


def _evaluate_confirmation(
    cand_stats: dict[str, Any],
    pair_stats: dict[str, Any],
    inc_stats: dict[str, Any],
    *,
    has_base: bool,
    alpha_cutoff: float = 0.05,
) -> tuple[bool, bool, str, float]:
    """
    Unified v2 confirmation gate:
      1. 4 benchmark-agnostic safety invariants.
      2. Positive dominance: wins > losses, mean_delta > 0, winsor_delta > 0.
      3. Exact Paired Sign-Permutation p < eff_alpha (with 0.5x alpha bar on >25% AST bloat).
    Returns `(beats_incumbent, safety_ok, reject_reason, eff_alpha)`.
    """
    safety_ok, safety_reason = _passes_safety_invariants(
        cand_stats, inc_stats, has_base=has_base
    )
    if not safety_ok:
        return False, False, safety_reason, alpha_cutoff
    if not has_base:
        return True, True, "promoted", alpha_cutoff

    cand_ast = int(cand_stats.get("ast_nodes", 0) or 0)
    inc_ast = int(inc_stats.get("ast_nodes", 0) or 0)
    bloated = inc_ast >= 50 and cand_ast > int(inc_ast * 1.25)
    eff_alpha = alpha_cutoff * 0.5 if bloated else alpha_cutoff

    n_paired = int(pair_stats.get("paired_count", pair_stats.get("n_paired", 0)) or 0)
    sign_p = float(pair_stats.get("sign_perm_p", 1.0) or 1.0)

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
) -> tuple[int, float, float, float, int]:
    """
    Precision-aware v2.1 selection ranker:
      1. Credibility tier (`sign_perm_p < 0.15` and `wins > losses`) so high-precision surgical
         winners outrank high-variance coin-flip edits.
      2. Lower-confidence-adjusted Winsorized delta (`lcb_score = winsor_delta - 0.5 * winsor_se`).
      3. Statistical confidence (`-sign_perm_p`).
      4. Point Winsorized delta (`winsor_delta`).
      5. Simplicity tie-breaker (`-ast_nodes`).
    """
    wins = int(pair_stats.get("wins", 0) or 0)
    losses = int(pair_stats.get("losses", 0) or 0)
    sign_p = float(pair_stats.get("sign_perm_p", 1.0) or 1.0)
    winsor_d = float(
        pair_stats.get("winsor_delta", pair_stats.get("mean_delta", 0.0)) or 0.0
    )
    lcb = float(pair_stats.get("lcb_score", winsor_d) or 0.0)
    is_credible = 1 if (sign_p < 0.15 and wins > losses) else 0
    ast_n = int(cand_stats.get("ast_nodes", 0) or 0)
    return (
        is_credible,
        round(lcb, 4),
        round(-sign_p, 4),
        round(winsor_d, 4),
        -ast_n,
    )


class SwarmReferee:
    """Deterministic CRN tournament selector & AST-disjoint orthogonal pair builder."""

    def __init__(self, max_tournament_peers: int = 3) -> None:
        self.max_tournament_peers = max(1, int(max_tournament_peers))
        self._incumbent_cache: dict[
            tuple[int, str, tuple[int, ...], int],
            tuple[BatchEvaluation, dict[str, Any]],
        ] = {}
        self._rejected_fingerprints: dict[str, set[str]] = {}

    def select_finalists(
        self,
        candidates: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
    ) -> list[PeerCandidate]:
        """Filters to valid novel incubated candidates, removes no-ops/repeats, and ranks top-K."""
        survivors = [
            c for c in candidates if c.sandbox_passed and c.policy_code.strip()
        ]
        if not survivors:
            return []

        # Prefer candidates that did NOT clobber protected D_succ branches unless they proved +delta on guards
        safe_survivors = [
            c
            for c in survivors
            if not c.clobbers_protected
            or float(c.sandbox_metrics.get("guard_depth_delta", 0.0)) > 0.0
        ]
        pool = safe_survivors if safe_survivors else survivors

        base_fp = (
            _package_fingerprint(base_policy, base_skills)
            if base_policy.strip()
            else ""
        )
        rejected_fps = (
            self._rejected_fingerprints.get(base_fp, set()) if base_fp else set()
        )
        unique: dict[str, PeerCandidate] = {}
        for c in pool:
            fp = _package_fingerprint(c.policy_code, c.skills)
            if base_fp and (
                fp == base_fp
                or _is_package_ast_equivalent(
                    c.policy_code, c.skills, base_policy, base_skills
                )
            ):
                continue
            if fp in rejected_fps:
                continue
            if fp not in unique:
                unique[fp] = c

        ranked = sorted(
            unique.values(),
            key=lambda c: (
                float(c.sandbox_metrics.get("depth_delta", 0.0)),
                float(c.sandbox_metrics.get("frontier_depth_delta", 0.0)),
                int(c.sandbox_metrics.get("frontier_survived", 0)),
                float(c.sandbox_metrics.get("score_delta", 0.0)),
                int(c.sandbox_metrics.get("probe_diverged", 0)),
                -float(c.sandbox_metrics.get("wrapper_ratio", 0.0)),
                -int(c.loc_total),
            ),
            reverse=True,
        )
        return ranked[: self.max_tournament_peers]

    @staticmethod
    def _compose_two_packages(
        base_policy: str,
        base_skills: dict[str, str],
        a: PeerCandidate,
        b: PeerCandidate,
        *,
        peer_id: str | None = None,
        claimed_label: str | None = None,
        check_symbol_disjoint: bool = True,
    ) -> PeerCandidate | None:
        """3-way AST-merges two non-conflicting `PeerCandidate` diffs atop `base_policy`/`base_skills`."""
        if check_symbol_disjoint:
            shared_sk = a.touched_skills & b.touched_skills
            for sk in shared_sk:
                sa = a.touched_skill_symbols.get(sk, set())
                sb = b.touched_skill_symbols.get(sk, set())
                if not sa or not sb or not sa.isdisjoint(sb):
                    return None

        a_changed_pol = not is_ast_equivalent(a.policy_code, base_policy)
        b_changed_pol = not is_ast_equivalent(b.policy_code, base_policy)
        if a_changed_pol and b_changed_pol:
            chosen_pol_maybe = three_way_ast_merge(
                base_policy, a.policy_code, b.policy_code
            )
            if not chosen_pol_maybe:
                return None
            chosen_pol = chosen_pol_maybe
        else:
            chosen_pol = (
                a.policy_code
                if a_changed_pol
                else (b.policy_code if b_changed_pol else base_policy)
            )

        merged_skills = dict(base_skills)
        for sk in a.touched_skills | b.touched_skills:
            in_a = sk in a.touched_skills
            in_b = sk in b.touched_skills
            if in_a and in_b:
                m_sk = three_way_ast_merge(
                    base_skills.get(sk, ""),
                    a.skills.get(sk, ""),
                    b.skills.get(sk, ""),
                )
                if m_sk is None:
                    return None
                merged_skills[sk] = m_sk
            elif in_a:
                if sk in a.skills:
                    merged_skills[sk] = a.skills[sk]
                else:
                    merged_skills.pop(sk, None)
            else:
                if sk in b.skills:
                    merged_skills[sk] = b.skills[sk]
                else:
                    merged_skills.pop(sk, None)

        kept_skills = SwarmMetaAgent.prune_unused_skills(chosen_pol, merged_skills)
        try:
            with _COMPILE_EXEC_LOCK:
                compiled = compile_policy_package(chosen_pol, "", skills=kept_skills)
                compiled.create_runner()
        except Exception:
            return None

        merged_touched_syms: dict[str, set[str]] = {}
        for sk in a.touched_skills | b.touched_skills:
            s_union = set(a.touched_skill_symbols.get(sk, set())) | set(
                b.touched_skill_symbols.get(sk, set())
            )
            if s_union and sk in kept_skills:
                merged_touched_syms[sk] = s_union

        loc_total = len(chosen_pol.splitlines()) + sum(
            len(v.splitlines()) for v in kept_skills.values()
        )
        return PeerCandidate(
            peer_id=peer_id or f"ortho_{a.peer_id}_{b.peer_id}",
            claimed_deficit_id=f"{a.claimed_deficit_id}+{b.claimed_deficit_id}",
            claimed_label=claimed_label
            or f"Orthogonal merge: {a.peer_id} + {b.peer_id}",
            hypothesis_yaml={
                "mechanism": f"AST-merged disjoint combination of {a.peer_id} and {b.peer_id}"
            },
            policy_code=chosen_pol,
            skills=kept_skills,
            touched_skills=a.touched_skills | b.touched_skills,
            touched_skill_symbols=merged_touched_syms,
            touched_branches=a.touched_branches | b.touched_branches,
            sandbox_passed=True,
            loc_total=loc_total,
        )

    @staticmethod
    def build_orthogonal_candidates(
        finalists: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
        max_combos: int = 3,
    ) -> list[PeerCandidate]:
        """
        Scans pairs `(i, j)` in `finalists` and constructs up to `max_combos` verified AST-disjoint
        compositions via 3-way AST merge (supporting both disjoint skill files, disjoint symbols
        within the same skill module, and disjoint `policy.py` methods / `PrioritySelector` additions).
        """
        if len(finalists) < 2 or not base_policy.strip():
            return []

        combos: list[PeerCandidate] = []
        seen_fps: set[str] = {_package_fingerprint(base_policy, base_skills)}
        for f in finalists:
            seen_fps.add(_package_fingerprint(f.policy_code, f.skills))

        scan = finalists[:5]
        for idx_i in range(len(scan)):
            for idx_j in range(idx_i + 1, len(scan)):
                if len(combos) >= max_combos:
                    return combos
                a, b = scan[idx_i], scan[idx_j]
                merged = SwarmReferee._compose_two_packages(
                    base_policy, base_skills, a, b, check_symbol_disjoint=True
                )
                if merged is None:
                    continue
                fp = _package_fingerprint(merged.policy_code, merged.skills)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                combos.append(merged)
        return combos

    @staticmethod
    def build_orthogonal_pair(
        finalists: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
    ) -> PeerCandidate | None:
        combos = SwarmReferee.build_orthogonal_candidates(
            finalists, base_policy, base_skills, max_combos=1
        )
        return combos[0] if combos else None

    def evaluate_tournament(
        self,
        generation: int,
        candidates: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
        eval_pool: Any,
        seeds: list[int],
        max_steps: int | None = None,
        holdout_seeds: list[int] | None = None,
        race_seeds: int | None = None,
        select_seeds: list[int] | None = None,
        scenario_gate: str = "off",
        frontier_shards: list[Any] | None = None,
    ) -> TournamentVerdict:
        """
        Minimalist 3-Stage Split-Sample Promotion Gate (v2):
          - Stage 1 (already run in `incubate_candidate` + `select_finalists`):
            deterministic hygiene filter + up to 2 AST-disjoint ortho merges.
          - Stage 2 (In-Sample Selection on `S_select`, `record_ticks=False`):
            evaluates Incumbent + finalists + optional in-sample 2-way greedy stack;
            ranks safe positive candidates by 5%-Winsorized paired mean delta (tie: -ast_nodes);
            advances ONLY the single #1 winner `C*`.
          - Stage 3 (Out-of-Sample Confirmation on `S_confirm`, `record_ticks=True`):
            evaluates ONLY Incumbent + `C*` on strictly unseen seeds (`S_confirm`),
            with optional O'Brien-Fleming `+64` extension (`α₁=0.030`, `α₂=0.032` -> FWER <= 0.05)
            when `len(S_confirm) >= 32`. Zero fallback loop -> zero winner's-curse leakage.
          - Mode B (Single-Stage Direct Fallback): when neither `select_seeds` nor `race_seeds`
            is provided, evaluates directly on `seeds` with the identical unified v2 rule.
        """
        _ = frontier_shards  # Retained in signature for backward compatibility; subsumed by S_confirm.
        bm_name = str(getattr(eval_pool, "benchmark", "nethack") or "nethack")
        has_base = bool(base_policy.strip())

        finalists = self.select_finalists(candidates, base_policy, base_skills)

        base_loc = len(base_policy.splitlines()) + sum(
            len(v.splitlines()) for v in base_skills.values()
        )
        inc_fp = _package_fingerprint(base_policy, base_skills) if has_base else ""
        steps_key = int(max_steps or 0)
        fin_eval_kwargs: dict[str, Any] = (
            {"max_steps": int(max_steps)} if max_steps is not None else {}
        )

        # Resolve Split-Sample (Mode A) vs Single-Stage Direct Fallback (Mode B)
        use_split_sample = bool(select_seeds) or (
            race_seeds is not None and 4 <= int(race_seeds) < len(seeds)
        )
        # In Mode A (Split-Sample), NEVER pre-merge unverified parents before S_select safety
        # checks — stacks are composed ONLY from individually safe positive S_select winners.
        # In Mode B (single-stage direct), preserve pre-merge orthogonal compositions.
        if not use_split_sample:
            orthos = self.build_orthogonal_candidates(
                finalists, base_policy, base_skills, max_combos=2
            )
            if orthos:
                finalists.extend(orthos)
        if use_split_sample:
            if select_seeds:
                s_select = [int(s) for s in select_seeds]
                sel_set = set(s_select)
                s_confirm = [int(s) for s in seeds if int(s) not in sel_set]
                if not s_confirm:
                    s_confirm = [int(s) for s in seeds]
            else:
                r_cnt = int(race_seeds or 0)
                s_select = [int(s) for s in seeds[:r_cnt]]
                s_confirm = [int(s) for s in seeds[r_cnt:]]
        else:
            s_select = []
            s_confirm = [int(s) for s in seeds]

        can_parallel_dispatch = (
            hasattr(eval_pool, "evaluate_tasks")
            and hasattr(eval_pool, "role")
            and callable(eval_pool.evaluate_tasks)
        )

        def _extract_stats(
            be: BatchEvaluation,
            pol: str,
            skl: dict[str, str],
            loc_override: int | None = None,
        ) -> dict[str, Any]:
            loc_val = (
                int(loc_override)
                if loc_override is not None
                else (
                    len(pol.splitlines())
                    + sum(len(v.splitlines()) for v in skl.values())
                )
            )
            return _extract_eval_stats(
                be,
                loc_total=loc_val,
                benchmark=bm_name,
                ast_nodes=_count_ast_nodes(pol, skl),
            )

        def _get_or_eval_incumbent(
            eval_seeds: list[int], *, record_ticks: bool, stage_tag: str
        ) -> tuple[BatchEvaluation | None, dict[str, Any]]:
            if not has_base:
                return None, _extract_eval_stats(
                    BatchEvaluation({"runs": []}),
                    loc_total=0,
                    benchmark=bm_name,
                    ast_nodes=0,
                )
            s_key = tuple(int(s) for s in eval_seeds)
            c_key = (id(eval_pool), inc_fp, s_key, steps_key)
            cached = self._incumbent_cache.get(c_key)
            if cached is not None:
                return cached
            be = eval_pool.evaluate_batch(
                batch_id=f"batch_swarm_g{generation}_{stage_tag}inc_{int(time.time())}",
                policy_code=base_policy,
                macros_code="",
                skills=base_skills,
                target_episodes=len(eval_seeds),
                priority_seeds=list(eval_seeds),
                record_ticks=record_ticks,
                **fin_eval_kwargs,
            )
            st = _extract_stats(be, base_policy, base_skills, loc_override=base_loc)
            if len(self._incumbent_cache) >= 4:
                self._incumbent_cache.clear()
            self._incumbent_cache[c_key] = (be, st)
            return be, st

        def _run_batch_slice(
            pkg_items: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ],
            eval_seeds: list[int],
            stage_tag: str,
        ) -> dict[str, BatchEvaluation]:
            """Evaluates `[(tag, pol, sk, cand_or_None, rec_ticks), ...]` on `eval_seeds`."""
            ts_now = int(time.time())
            seed_rank = {int(s): i for i, s in enumerate(eval_seeds)}
            out_batches: dict[str, BatchEvaluation] = {}
            if not pkg_items:
                return out_batches
            if can_parallel_dispatch:
                eff_steps = int(
                    max_steps
                    if max_steps is not None
                    else getattr(eval_pool, "max_steps", 25000)
                )
                pool_role = str(getattr(eval_pool, "role", "val-hum-law-fem"))
                all_tasks: list[dict[str, Any]] = []
                batch_groups: list[tuple[str, str]] = []
                for tag, pol, skl, _cand, rec_ticks in pkg_items:
                    bid = f"batch_swarm_g{generation}_{stage_tag}{tag}_{ts_now}"
                    batch_groups.append((tag, bid))
                    for idx, sd in enumerate(eval_seeds, start=1):
                        all_tasks.append(
                            {
                                "seed": int(sd),
                                "run_id": f"{bid}_r{idx:05d}",
                                "role": pool_role,
                                "max_steps": eff_steps,
                                "policy_code": pol,
                                "macros_code": "",
                                "target_functions": None,
                                "param_overrides": None,
                                "record_ticks": rec_ticks,
                                "is_priority_seed": True,
                                "enforce_trigger": False,
                                "skills": skl,
                                "benchmark": bm_name,
                            }
                        )
                raw_results = eval_pool.evaluate_tasks(all_tasks)
                by_bid: dict[str, list[dict[str, Any]]] = {
                    bid: [] for _t, bid in batch_groups
                }
                for res in raw_results:
                    rid = str(res.get("run_id", ""))
                    prefix = rid.rsplit("_r", 1)[0] if "_r" in rid else ""
                    if prefix in by_bid:
                        by_bid[prefix].append(res)
                for tag, bid in batch_groups:
                    runs_sorted = sorted(
                        by_bid.get(bid, []),
                        key=lambda r: seed_rank.get(int(r.get("seed", 0)), 999999),
                    )
                    out_batches[tag] = _assemble_batch_eval(bid, runs_sorted)
            else:
                for tag, pol, skl, _cand, rec_ticks in pkg_items:
                    bid = f"batch_swarm_g{generation}_{stage_tag}{tag}_{ts_now}"
                    out_batches[tag] = eval_pool.evaluate_batch(
                        batch_id=bid,
                        policy_code=pol,
                        macros_code="",
                        skills=skl,
                        target_episodes=len(eval_seeds),
                        priority_seeds=list(eval_seeds),
                        record_ticks=rec_ticks,
                        **fin_eval_kwargs,
                    )
            return out_batches

        def _make_attempt_row(
            c: PeerCandidate,
            st: dict[str, Any],
            pr: dict[str, Any],
            *,
            promoted: bool,
            reject_reason: str,
            h_avg: float | None = None,
            h_iqm: float | None = None,
            h_rej: bool = False,
        ) -> dict[str, Any]:
            return {
                "attempt_id": f"att_g{generation}_{c.peer_id}_{int(time.time() * 1000)}",
                "generation": generation,
                "peer_id": c.peer_id,
                "deficit_id": c.claimed_deficit_id,
                "candidate_fp": _package_fingerprint(c.policy_code, c.skills)
                if c.policy_code
                else "",
                "promoted": promoted,
                "sandbox_passed": bool(c.sandbox_passed),
                "sandbox_error": str(c.error or ""),
                "avg_depth": float(st.get("avg_depth", 1.0)),
                "max_depth": int(st.get("max_depth", 1)),
                "avg_score": float(st.get("avg_score", 0.0)),
                "avg_ac": float(st.get("avg_ac", 10.0)),
                "avg_xp_level": float(st.get("avg_xp_level", 1.0)),
                "policy_exceptions": int(st.get("policy_exceptions", 0)),
                "seeds_gained": list(pr.get("seeds_gained", [])),
                "seeds_lost": list(pr.get("seeds_lost", [])),
                "top_tracebacks": list(st.get("top_tracebacks", [])),
                "holdout_avg_depth": h_avg,
                "holdout_iqm_delta": h_iqm,
                "holdout_rejected": bool(h_rej),
                "iqm_delta": float(pr.get("iqm_delta", 0.0)),
                "wilcoxon_p": float(pr.get("wilcoxon_p", 1.0)),
                "sign_perm_p": float(pr.get("sign_perm_p", 1.0)),
                "winsor_delta": float(pr.get("winsor_delta", 0.0)),
                "wins": int(pr.get("wins", 0) or 0),
                "losses": int(pr.get("losses", 0) or 0),
                "reject_reason": reject_reason,
            }

        def _append_untested_candidates(
            attempts_out: list[dict[str, Any]], seen_ids: set[str]
        ) -> None:
            for c in candidates:
                if c.peer_id in seen_ids:
                    continue
                attempts_out.append(
                    {
                        "attempt_id": f"att_g{generation}_{c.peer_id}_{int(time.time() * 1000)}",
                        "generation": generation,
                        "peer_id": c.peer_id,
                        "deficit_id": c.claimed_deficit_id,
                        "candidate_fp": _package_fingerprint(c.policy_code, c.skills)
                        if c.policy_code
                        else "",
                        "promoted": False,
                        "sandbox_passed": bool(c.sandbox_passed),
                        "sandbox_error": str(c.error or ""),
                        "avg_depth": float(c.sandbox_metrics.get("avg_depth", 1.0)),
                        "max_depth": int(c.sandbox_metrics.get("max_depth", 1)),
                        "avg_score": float(c.sandbox_metrics.get("avg_score", 0.0)),
                        "avg_ac": 10.0,
                        "avg_xp_level": 1.0,
                        "policy_exceptions": 1
                        if "exception" in str(c.error).lower()
                        else 0,
                        "seeds_gained": [],
                        "seeds_lost": [],
                        "top_tracebacks": [c.error[:180]] if c.error else [],
                        "reject_reason": "incubation_or_filter_failed",
                    }
                )

        def _run_scenario_gate(cand_obj: PeerCandidate) -> bool:
            if str(scenario_gate or "off").lower() not in ("advisory", "strict"):
                return True
            try:
                from lox.env.benchmarks.nethack_scenarios import run_scenario_gate

                sg = run_scenario_gate(
                    cand_obj.policy_code,
                    cand_obj.skills,
                    seed=int(s_confirm[0]) if s_confirm else 42,
                    max_steps=40,
                )
                ui.info(
                    f"Gen {generation} Scenario Gate ({scenario_gate}): "
                    f"{sg['passed_count']}/{sg['total_count']} passed "
                    f"(exc={sg['exceptions']}, livelocks={sg['prompt_livelocks']})."
                )
                if str(scenario_gate).lower() == "strict" and (
                    sg.get("exceptions", 0) > 0 or sg.get("prompt_livelocks", 0) > 0
                ):
                    return False
            except Exception:
                if str(scenario_gate).lower() == "strict":
                    return False
            return True

        # Early exit: zero novel incubated survivors
        if not finalists:
            if has_base:
                inc_batch_eval, inc_stats = _get_or_eval_incumbent(
                    s_confirm, record_ticks=True, stage_tag=""
                )
                assert inc_batch_eval is not None
                retained_eval = _clone_incumbent_eval(inc_batch_eval, generation)
                return TournamentVerdict(
                    promoted=False,
                    winner_id="incumbent",
                    winner_policy_code=base_policy,
                    winner_skills=dict(base_skills),
                    winner_batch_eval=retained_eval,
                    incumbent_avg_depth=inc_stats["avg_depth"],
                    incumbent_avg_score=inc_stats["avg_score"],
                    winner_avg_depth=inc_stats["avg_depth"],
                    winner_avg_score=inc_stats["avg_score"],
                    scoreboard=[],
                    all_runner_up_batches=[],
                    incumbent_batch_eval=inc_batch_eval,
                    reason="No novel non-rejected peer passed isolated sandbox incubation; retained incumbent.",
                )
            raise RuntimeError(
                f"Gen {generation}: Zero peer candidates passed sandbox incubation on bootstrap."
            )

        # =====================================================================
        # MODE A: Strict Split-Sample (Stage 2 Select -> Stage 3 Confirm + OBF)
        # =====================================================================
        if use_split_sample:
            sel_pkgs: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ] = []
            sel_inc_key = (
                id(eval_pool),
                inc_fp,
                tuple(int(s) for s in s_select),
                steps_key,
            )
            cached_sel_inc = (
                self._incumbent_cache.get(sel_inc_key) if has_base else None
            )
            if has_base and cached_sel_inc is None:
                sel_pkgs.append(("inc", base_policy, base_skills, None, False))
            for cand in finalists:
                sel_pkgs.append(
                    (cand.peer_id, cand.policy_code, cand.skills, cand, False)
                )
            sel_batches = _run_batch_slice(sel_pkgs, s_select, "sel_")

            if cached_sel_inc is not None:
                sel_inc_eval, sel_inc_stats = cached_sel_inc
            elif has_base and "inc" in sel_batches:
                sel_inc_eval = sel_batches["inc"]
                sel_inc_stats = _extract_stats(
                    sel_inc_eval, base_policy, base_skills, loc_override=base_loc
                )
            else:
                sel_inc_eval = None
                sel_inc_stats = _extract_eval_stats(
                    BatchEvaluation({"runs": []}),
                    loc_total=0,
                    benchmark=bm_name,
                    ast_nodes=0,
                )
            sel_inc_runs = (
                list(sel_inc_eval.get("runs", []) or []) if sel_inc_eval else []
            )

            # Score every finalist on S_select
            sel_entries: list[
                tuple[
                    PeerCandidate,
                    BatchEvaluation,
                    dict[str, Any],
                    dict[str, Any],
                    bool,
                    str,
                ]
            ] = []
            sel_seen_fps: set[str] = {inc_fp} if inc_fp else set()
            for cand in finalists:
                cb = sel_batches.get(cand.peer_id)
                if cb is None:
                    continue
                sel_seen_fps.add(_package_fingerprint(cand.policy_code, cand.skills))
                st = _extract_stats(
                    cb, cand.policy_code, cand.skills, loc_override=cand.loc_total
                )
                pr = (
                    evaluate_tournament_pair(
                        cb.get("runs", []) or [], sel_inc_runs, benchmark=bm_name
                    )
                    if sel_inc_runs
                    else {
                        "mean_delta": float(st.get("mean_prog", 1.0)),
                        "winsor_delta": float(st.get("mean_prog", 1.0)),
                        "iqm_delta": float(st.get("iqm_prog", 1.0)),
                        "wins": len(s_select),
                        "losses": 0,
                        "ties": 0,
                        "n_paired": len(s_select),
                        "sign_perm_p": 0.0,
                        "wilcoxon_p": 0.0,
                        "seeds_gained": [],
                        "seeds_lost": [],
                    }
                )
                s_ok, s_reason = _passes_safety_invariants(
                    st, sel_inc_stats, has_base=has_base
                )
                sel_entries.append((cand, cb, st, pr, s_ok, s_reason))

            # Safe In-Sample 2-Way Stacks on S_select (ONLY from individually safe positive winners!)
            safe_indiv_winners = [
                (c, st, pr)
                for (c, _cb, st, pr, s_ok, _sr) in sel_entries
                if s_ok
                and not c.peer_id.startswith("ortho_")
                and not c.peer_id.startswith("stack_")
                and (not has_base or _has_positive_dominance(pr))
            ]
            safe_indiv_winners.sort(
                key=lambda item: _selection_rank_key(item[1], item[2]), reverse=True
            )
            stack_parent_p: dict[str, float] = {}
            if has_base and len(safe_indiv_winners) >= 2:
                stack_cands: list[PeerCandidate] = []
                scan_safe = safe_indiv_winners[:4]
                for idx_i in range(len(scan_safe)):
                    if len(stack_cands) >= 2:
                        break
                    for idx_j in range(idx_i + 1, len(scan_safe)):
                        if len(stack_cands) >= 2:
                            break
                        top_a, _st_a, pr_a = scan_safe[idx_i]
                        top_b, _st_b, pr_b = scan_safe[idx_j]
                        cand_stack = self._compose_two_packages(
                            base_policy,
                            base_skills,
                            top_a,
                            top_b,
                            peer_id=f"stack_{top_a.peer_id}_{top_b.peer_id}",
                            claimed_label=f"In-sample stack: {top_a.peer_id} + {top_b.peer_id}",
                            check_symbol_disjoint=True,
                        )
                        if cand_stack is None and idx_i == 0 and idx_j == 1:
                            cand_stack = self._compose_two_packages(
                                base_policy,
                                base_skills,
                                top_a,
                                top_b,
                                peer_id=f"stack_{top_a.peer_id}_{top_b.peer_id}",
                                claimed_label=f"In-sample stack: {top_a.peer_id} + {top_b.peer_id}",
                                check_symbol_disjoint=False,
                            )
                        if cand_stack is None:
                            continue
                        stk_fp = _package_fingerprint(
                            cand_stack.policy_code, cand_stack.skills
                        )
                        if stk_fp in sel_seen_fps:
                            continue
                        sel_seen_fps.add(stk_fp)
                        stack_cands.append(cand_stack)
                        stack_parent_p[cand_stack.peer_id] = min(
                            float(pr_a.get("sign_perm_p", 1.0) or 1.0),
                            float(pr_b.get("sign_perm_p", 1.0) or 1.0),
                        )
                if stack_cands:
                    stk_batches = _run_batch_slice(
                        [
                            (sc.peer_id, sc.policy_code, sc.skills, sc, False)
                            for sc in stack_cands
                        ],
                        s_select,
                        "sel_stk_",
                    )
                    for cand_stack in stack_cands:
                        stk_be = stk_batches.get(cand_stack.peer_id)
                        if stk_be is None:
                            continue
                        stk_st = _extract_stats(
                            stk_be,
                            cand_stack.policy_code,
                            cand_stack.skills,
                            loc_override=cand_stack.loc_total,
                        )
                        stk_pr = evaluate_tournament_pair(
                            stk_be.get("runs", []) or [],
                            sel_inc_runs,
                            benchmark=bm_name,
                        )
                        stk_ok, stk_reason = _passes_safety_invariants(
                            stk_st, sel_inc_stats, has_base=True
                        )
                        # Guard against noisy stacks diluting a cleaner individual parent
                        stk_p = float(stk_pr.get("sign_perm_p", 1.0) or 1.0)
                        parent_p = stack_parent_p.get(cand_stack.peer_id, 1.0)
                        if stk_ok and stk_p > max(parent_p * 1.5, 0.10):
                            stk_ok = False
                            stk_reason = "stack_precision_degraded"
                        sel_entries.append(
                            (cand_stack, stk_be, stk_st, stk_pr, stk_ok, stk_reason)
                        )

            # Choose single #1 winner C* from S_select
            eligible_sel = [
                entry
                for entry in sel_entries
                if entry[4] and (not has_base or _has_positive_dominance(entry[3]))
            ]
            eligible_sel.sort(
                key=lambda entry: _selection_rank_key(entry[2], entry[3]), reverse=True
            )

            # Build Stage-2 scoreboard rows for diagnostics
            scoreboard: list[dict[str, Any]] = []
            for c, _be, st, pr, _s_ok, _sr in sel_entries:
                scoreboard.append(
                    {
                        "peer_id": c.peer_id,
                        "deficit": c.claimed_deficit_id,
                        "stage": "select",
                        "avg_depth": round(float(st["avg_depth"]), 3),
                        "avg_score": round(float(st["avg_score"]), 1),
                        "winsor_delta": round(float(pr.get("winsor_delta", 0.0)), 4),
                        "iqm_delta": round(float(pr.get("iqm_delta", 0.0)), 4),
                        "wins": int(pr.get("wins", 0) or 0),
                        "losses": int(pr.get("losses", 0) or 0),
                        "loc": int(c.loc_total),
                        "ast_nodes": int(st.get("ast_nodes", 0)),
                    }
                )

            # If ZERO candidates passed Stage 2 Select, evaluate ONLY Incumbent on S_confirm and retain
            if not eligible_sel:
                inc_batch_eval, inc_stats = _get_or_eval_incumbent(
                    s_confirm, record_ticks=True, stage_tag="conf_"
                )
                if inc_fp:
                    rej_set = self._rejected_fingerprints.setdefault(inc_fp, set())
                    for c, _be, _st, _pr, _s_ok, _sr in sel_entries:
                        rej_set.add(_package_fingerprint(c.policy_code, c.skills))

                attempts_summary: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for c, _be, st, pr, s_ok, s_reason in sel_entries:
                    seen_ids.add(c.peer_id)
                    attempts_summary.append(
                        _make_attempt_row(
                            c,
                            st,
                            pr,
                            promoted=False,
                            reject_reason=s_reason
                            if not s_ok
                            else "failed_select_stage",
                        )
                    )
                _append_untested_candidates(attempts_summary, seen_ids)

                runner_ups = [
                    {
                        "peer_id": c.peer_id,
                        "policy_code": c.policy_code,
                        "skills": dict(c.skills),
                        "metrics": st,
                        "batch_id": str(be.get("batch_id", "")),
                    }
                    for c, be, st, _pr, _s_ok, _sr in sel_entries
                ]
                if inc_batch_eval is not None:
                    retained_eval = _clone_incumbent_eval(inc_batch_eval, generation)
                    ui.warn(
                        f"Gen {generation}: Zero finalists beat incumbent safely on S_select (N={len(s_select)}). Retaining Incumbent."
                    )
                    return TournamentVerdict(
                        promoted=False,
                        winner_id="incumbent",
                        winner_policy_code=base_policy,
                        winner_skills=dict(base_skills),
                        winner_batch_eval=retained_eval,
                        incumbent_avg_depth=inc_stats["avg_depth"],
                        incumbent_avg_score=inc_stats["avg_score"],
                        winner_avg_depth=inc_stats["avg_depth"],
                        winner_avg_score=inc_stats["avg_score"],
                        scoreboard=scoreboard,
                        all_runner_up_batches=runner_ups,
                        incumbent_batch_eval=inc_batch_eval,
                        attempts_summary=attempts_summary,
                        reason=f"Zero candidates beat incumbent safely on S_select (N={len(s_select)}); retained incumbent.",
                    )
                raise RuntimeError(
                    f"Gen {generation}: Zero safe bootstrap candidates on S_select."
                )

            # Stage 3: Pre-Committed Out-of-Sample Confirmation on S_confirm (ONLY Incumbent + C*)
            best_cand = eligible_sel[0][0]
            conf_inc_key = (
                id(eval_pool),
                inc_fp,
                tuple(int(s) for s in s_confirm),
                steps_key,
            )
            cached_conf_inc = (
                self._incumbent_cache.get(conf_inc_key) if has_base else None
            )
            conf_pkgs: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ] = []
            if has_base and cached_conf_inc is None:
                conf_pkgs.append(("inc", base_policy, base_skills, None, True))
            conf_pkgs.append(
                (
                    best_cand.peer_id,
                    best_cand.policy_code,
                    best_cand.skills,
                    best_cand,
                    True,
                )
            )
            conf_batches = _run_batch_slice(conf_pkgs, s_confirm, "conf_")

            if cached_conf_inc is not None:
                inc_batch_eval, inc_stats = cached_conf_inc
            elif has_base and "inc" in conf_batches:
                inc_batch_eval = conf_batches["inc"]
                inc_stats = _extract_stats(
                    inc_batch_eval, base_policy, base_skills, loc_override=base_loc
                )
            else:
                inc_batch_eval = None
                inc_stats = _extract_eval_stats(
                    BatchEvaluation({"runs": []}),
                    loc_total=0,
                    benchmark=bm_name,
                    ast_nodes=0,
                )

            best_eval = conf_batches[best_cand.peer_id]
            best_stats = _extract_stats(
                best_eval,
                best_cand.policy_code,
                best_cand.skills,
                loc_override=best_cand.loc_total,
            )
            inc_runs_conf = (
                list(inc_batch_eval.get("runs", []) or []) if inc_batch_eval else []
            )
            cand_runs_conf = list(best_eval.get("runs", []) or [])
            best_pair: dict[str, Any] = (
                evaluate_tournament_pair(
                    cand_runs_conf, inc_runs_conf, benchmark=bm_name
                )
                if inc_runs_conf
                else {
                    "mean_delta": float(best_stats.get("mean_prog", 1.0)),
                    "winsor_delta": float(best_stats.get("mean_prog", 1.0)),
                    "iqm_delta": float(best_stats.get("iqm_prog", 1.0)),
                    "wins": len(s_confirm),
                    "losses": 0,
                    "ties": 0,
                    "n_paired": len(s_confirm),
                    "sign_perm_p": 0.0,
                    "wilcoxon_p": 0.0,
                    "seeds_gained": [],
                    "seeds_lost": [],
                }
            )

            # O'Brien-Fleming 2-Look Confirmation when N_confirm >= 32 (α₁=0.030, α₂=0.032 -> FWER <= 0.05)
            if has_base and len(s_confirm) >= 32:
                beats_incumbent, guard_ok, best_rej_reason, eff_a1 = (
                    _evaluate_confirmation(
                        best_stats,
                        best_pair,
                        inc_stats,
                        has_base=True,
                        alpha_cutoff=0.030,
                    )
                )
                look1_p = float(best_pair.get("sign_perm_p", 1.0) or 1.0)
                if (
                    not beats_incumbent
                    and guard_ok
                    and _has_positive_dominance(best_pair)
                    and eff_a1 <= look1_p < 0.16
                ):
                    ext_n = min(64, len(s_confirm))
                    used_seeds = set(s_select) | set(s_confirm)
                    s_obf: list[int] = []
                    cand_sd = 20_000_000 + generation * 10_000
                    while len(s_obf) < ext_n:
                        if cand_sd not in used_seeds:
                            s_obf.append(cand_sd)
                            used_seeds.add(cand_sd)
                        cand_sd += 1
                    ui.info(
                        f"Gen {generation}: OBF Stage-3b extension on `{best_cand.peer_id}` "
                        f"(look1 p={look1_p:.4f} in [{eff_a1:.3f}, 0.160), +{len(s_obf)} unseen seeds)..."
                    )
                    obf_batches = _run_batch_slice(
                        [
                            ("inc", base_policy, base_skills, None, False),
                            (
                                best_cand.peer_id,
                                best_cand.policy_code,
                                best_cand.skills,
                                best_cand,
                                False,
                            ),
                        ],
                        s_obf,
                        "obf_",
                    )
                    pooled_inc_runs = inc_runs_conf + list(
                        obf_batches["inc"].get("runs", []) or []
                    )
                    pooled_cand_runs = cand_runs_conf + list(
                        obf_batches[best_cand.peer_id].get("runs", []) or []
                    )
                    pooled_inc_eval = _assemble_batch_eval(
                        f"obf_pool_inc_g{generation}", pooled_inc_runs
                    )
                    pooled_cand_eval = _assemble_batch_eval(
                        f"obf_pool_{best_cand.peer_id}_g{generation}", pooled_cand_runs
                    )
                    pooled_inc_stats = _extract_stats(
                        pooled_inc_eval, base_policy, base_skills, loc_override=base_loc
                    )
                    pooled_cand_stats = _extract_stats(
                        pooled_cand_eval,
                        best_cand.policy_code,
                        best_cand.skills,
                        loc_override=best_cand.loc_total,
                    )
                    pooled_pair = evaluate_tournament_pair(
                        pooled_cand_runs, pooled_inc_runs, benchmark=bm_name
                    )
                    beats_incumbent, guard_ok, best_rej_reason, eff_a2 = (
                        _evaluate_confirmation(
                            pooled_cand_stats,
                            pooled_pair,
                            pooled_inc_stats,
                            has_base=True,
                            alpha_cutoff=0.032,
                        )
                    )
                    best_pair = dict(pooled_pair)
                    best_pair["obf_extended"] = True
                    best_pair["obf_look1_p"] = look1_p
                    best_pair["obf_alpha2"] = eff_a2
            else:
                beats_incumbent, guard_ok, best_rej_reason, _ = _evaluate_confirmation(
                    best_stats,
                    best_pair,
                    inc_stats,
                    has_base=has_base,
                    alpha_cutoff=0.05,
                )

            if beats_incumbent and not _run_scenario_gate(best_cand):
                beats_incumbent = False
                guard_ok = False
                best_rej_reason = "scenario_gate_failed"

            holdout_avg_depth: float | None = float(best_stats["avg_depth"])
            holdout_iqm_delta: float | None = float(best_pair.get("iqm_delta", 0.0))
            holdout_rejected = bool(not beats_incumbent and guard_ok and has_base)

            d_delta = float(best_stats["avg_depth"]) - float(inc_stats["avg_depth"])
            s_delta = float(best_stats["avg_score"]) - float(inc_stats["avg_score"])

            # Update scoreboard entry for C* with its out-of-sample S_confirm stats
            for row in scoreboard:
                if row["peer_id"] == best_cand.peer_id:
                    row["stage"] = "confirm"
                    row["avg_depth"] = round(float(best_stats["avg_depth"]), 3)
                    row["avg_score"] = round(float(best_stats["avg_score"]), 1)
                    row["winsor_delta"] = round(
                        float(best_pair.get("winsor_delta", 0.0)), 4
                    )
                    row["iqm_delta"] = round(float(best_pair.get("iqm_delta", 0.0)), 4)
                    row["sign_perm_p"] = round(
                        float(best_pair.get("sign_perm_p", 1.0)), 4
                    )
                    row["wins"] = int(best_pair.get("wins", 0) or 0)
                    row["losses"] = int(best_pair.get("losses", 0) or 0)

            attempts_summary = []
            seen_ids = set()
            for c, _be, st, pr, s_ok, s_reason in sel_entries:
                seen_ids.add(c.peer_id)
                if c.peer_id == best_cand.peer_id:
                    attempts_summary.append(
                        _make_attempt_row(
                            c,
                            best_stats,
                            best_pair,
                            promoted=beats_incumbent,
                            reject_reason="promoted"
                            if beats_incumbent
                            else best_rej_reason,
                            h_avg=holdout_avg_depth,
                            h_iqm=holdout_iqm_delta,
                            h_rej=holdout_rejected,
                        )
                    )
                else:
                    attempts_summary.append(
                        _make_attempt_row(
                            c,
                            st,
                            pr,
                            promoted=False,
                            reject_reason=s_reason
                            if not s_ok
                            else f"cut_in_select_stage_by_{best_cand.peer_id}",
                        )
                    )
            _append_untested_candidates(attempts_summary, seen_ids)

            runner_ups = [
                {
                    "peer_id": c.peer_id,
                    "policy_code": c.policy_code,
                    "skills": dict(c.skills),
                    "metrics": best_stats if c.peer_id == best_cand.peer_id else st,
                    "batch_id": str(
                        (best_eval if c.peer_id == best_cand.peer_id else be).get(
                            "batch_id", ""
                        )
                    ),
                }
                for c, be, st, _pr, _s_ok, _sr in sel_entries
                if c.peer_id != best_cand.peer_id or not beats_incumbent
            ]

            scoreboard_str = " | ".join(
                f"{row['peer_id']}[{row['stage']}]: D={row['avg_depth']:.2f}/WΔ={row['winsor_delta']:+.3f}/AST={row['ast_nodes']}"
                for row in scoreboard
            )
            ui.info(
                f"Gen {generation} Split-Sample Tournament (Sel={len(s_select)} -> Conf={len(s_confirm)}): "
                f"Inc D={inc_stats['avg_depth']:.2f} | {scoreboard_str}"
            )

            if not beats_incumbent and inc_batch_eval is not None:
                if inc_fp:
                    rej_set = self._rejected_fingerprints.setdefault(inc_fp, set())
                    for c, _be, _st, _pr, _s_ok, _sr in sel_entries:
                        rej_set.add(_package_fingerprint(c.policy_code, c.skills))
                reject_note = (
                    f"{best_rej_reason} on S_confirm "
                    f"(W/L={best_pair.get('wins', 0)}/{best_pair.get('losses', 0)}, "
                    f"winsorΔ={best_pair.get('winsor_delta', 0.0):+.3f}, "
                    f"p={best_pair.get('sign_perm_p', 1.0):.4f})"
                )
                ui.warn(
                    f"Gen {generation}: Selected `{best_cand.peer_id}` rejected ({reject_note}). Retaining Incumbent."
                )
                retained_eval = _clone_incumbent_eval(inc_batch_eval, generation)
                return TournamentVerdict(
                    promoted=False,
                    winner_id="incumbent",
                    winner_policy_code=base_policy,
                    winner_skills=dict(base_skills),
                    winner_batch_eval=retained_eval,
                    incumbent_avg_depth=inc_stats["avg_depth"],
                    incumbent_avg_score=inc_stats["avg_score"],
                    winner_avg_depth=inc_stats["avg_depth"],
                    winner_avg_score=inc_stats["avg_score"],
                    target_functions=sorted(best_cand.touched_branches)
                    or [best_cand.claimed_deficit_id],
                    scoreboard=scoreboard,
                    all_runner_up_batches=runner_ups,
                    incumbent_batch_eval=inc_batch_eval,
                    attempts_summary=attempts_summary,
                    holdout_avg_depth=holdout_avg_depth,
                    holdout_iqm_delta=holdout_iqm_delta,
                    holdout_rejected=holdout_rejected,
                    best_pair=dict(best_pair),
                    reason=f"Rejected `{best_cand.peer_id}` ({reject_note}); kept incumbent.",
                )

            self.cache_incumbent(
                eval_pool=eval_pool,
                policy_code=best_cand.policy_code,
                skills=best_cand.skills,
                seeds=s_confirm,
                max_steps=max_steps,
                batch_eval=best_eval,
                stats=best_stats,
                benchmark=bm_name,
            )
            return TournamentVerdict(
                promoted=True,
                winner_id=best_cand.peer_id,
                winner_policy_code=best_cand.policy_code,
                winner_skills=dict(best_cand.skills),
                winner_batch_eval=best_eval,
                incumbent_avg_depth=inc_stats["avg_depth"],
                incumbent_avg_score=inc_stats["avg_score"],
                winner_avg_depth=best_stats["avg_depth"],
                winner_avg_score=best_stats["avg_score"],
                target_functions=sorted(best_cand.touched_branches)
                or [best_cand.claimed_deficit_id],
                scoreboard=scoreboard,
                all_runner_up_batches=runner_ups,
                incumbent_batch_eval=inc_batch_eval,
                attempts_summary=attempts_summary,
                holdout_avg_depth=holdout_avg_depth,
                holdout_iqm_delta=holdout_iqm_delta,
                holdout_rejected=False,
                best_pair=dict(best_pair),
                reason=(
                    f"Promoted `{best_cand.peer_id}` on S_confirm targeting `{best_cand.claimed_deficit_id}` "
                    f"(Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}, "
                    f"winsorΔ={best_pair.get('winsor_delta', 0.0):+.3f}, "
                    f"p={best_pair.get('sign_perm_p', 1.0):.4f})."
                ),
            )

        # =====================================================================
        # MODE B: Single-Stage Direct Fallback (when select_seeds & race_seeds omitted)
        # =====================================================================
        inc_key_b = (id(eval_pool), inc_fp, tuple(int(s) for s in s_confirm), steps_key)
        cached_inc_b = self._incumbent_cache.get(inc_key_b) if has_base else None
        b_pkgs: list[tuple[str, str, dict[str, str], PeerCandidate | None, bool]] = []
        if has_base and cached_inc_b is None:
            b_pkgs.append(("inc", base_policy, base_skills, None, True))
        for cand in finalists:
            b_pkgs.append((cand.peer_id, cand.policy_code, cand.skills, cand, True))
        b_batches = _run_batch_slice(b_pkgs, s_confirm, "")

        if cached_inc_b is not None:
            inc_batch_eval, inc_stats = cached_inc_b
        elif has_base and "inc" in b_batches:
            inc_batch_eval = b_batches["inc"]
            inc_stats = _extract_stats(
                inc_batch_eval, base_policy, base_skills, loc_override=base_loc
            )
            if len(self._incumbent_cache) >= 4:
                self._incumbent_cache.clear()
            self._incumbent_cache[inc_key_b] = (inc_batch_eval, inc_stats)
        else:
            inc_batch_eval = None
            inc_stats = _extract_eval_stats(
                BatchEvaluation({"runs": []}),
                loc_total=0,
                benchmark=bm_name,
                ast_nodes=0,
            )

        inc_runs_b = (
            list(inc_batch_eval.get("runs", []) or []) if inc_batch_eval else []
        )

        eval_rows_b: list[
            tuple[
                PeerCandidate,
                BatchEvaluation,
                dict[str, Any],
                dict[str, Any],
                bool,
                bool,
                str,
            ]
        ] = []
        scoreboard = []
        for cand in finalists:
            cb = b_batches.get(cand.peer_id)
            if cb is None:
                continue
            st = _extract_stats(
                cb, cand.policy_code, cand.skills, loc_override=cand.loc_total
            )
            pr = (
                evaluate_tournament_pair(
                    cb.get("runs", []) or [], inc_runs_b, benchmark=bm_name
                )
                if inc_runs_b
                else {
                    "mean_delta": float(st.get("mean_prog", 1.0)),
                    "winsor_delta": float(st.get("mean_prog", 1.0)),
                    "iqm_delta": float(st.get("iqm_prog", 1.0)),
                    "wins": len(s_confirm),
                    "losses": 0,
                    "ties": 0,
                    "n_paired": len(s_confirm),
                    "sign_perm_p": 0.0,
                    "wilcoxon_p": 0.0,
                    "seeds_gained": [],
                    "seeds_lost": [],
                }
            )
            c_beats, c_safe, c_rej, _ = _evaluate_confirmation(
                st, pr, inc_stats, has_base=has_base, alpha_cutoff=0.05
            )
            eval_rows_b.append((cand, cb, st, pr, c_beats, c_safe, c_rej))
            scoreboard.append(
                {
                    "peer_id": cand.peer_id,
                    "deficit": cand.claimed_deficit_id,
                    "stage": "direct",
                    "avg_depth": round(float(st["avg_depth"]), 3),
                    "avg_score": round(float(st["avg_score"]), 1),
                    "winsor_delta": round(float(pr.get("winsor_delta", 0.0)), 4),
                    "iqm_delta": round(float(pr.get("iqm_delta", 0.0)), 4),
                    "sign_perm_p": round(float(pr.get("sign_perm_p", 1.0)), 4),
                    "wins": int(pr.get("wins", 0) or 0),
                    "losses": int(pr.get("losses", 0) or 0),
                    "loc": int(cand.loc_total),
                    "ast_nodes": int(st.get("ast_nodes", 0)),
                }
            )

        ranked_b = sorted(
            range(len(eval_rows_b)),
            key=lambda i: (
                1 if eval_rows_b[i][4] else 0,
                1 if eval_rows_b[i][5] else 0,
                *_selection_rank_key(eval_rows_b[i][2], eval_rows_b[i][3]),
            ),
            reverse=True,
        )
        best_idx = ranked_b[0]
        (
            best_cand,
            best_eval,
            best_stats,
            best_pair,
            beats_incumbent,
            guard_ok,
            best_rej_reason,
        ) = eval_rows_b[best_idx]

        # Optional Scenario Gate + Explicit Holdout Check (no fallback loop)
        holdout_avg_depth = None
        holdout_iqm_delta = None
        holdout_rejected = False
        if beats_incumbent and not _run_scenario_gate(best_cand):
            beats_incumbent = False
            guard_ok = False
            best_rej_reason = "scenario_gate_failed"

        if beats_incumbent and holdout_seeds and has_base:
            h_seeds = [int(s) for s in holdout_seeds]
            try:
                h_inc_eval, h_inc_stats = _get_or_eval_incumbent(
                    h_seeds, record_ticks=False, stage_tag="holdout_"
                )
                h_cand_eval = eval_pool.evaluate_batch(
                    batch_id=f"holdout_g{generation}_{best_cand.peer_id}_{int(time.time())}",
                    policy_code=best_cand.policy_code,
                    macros_code="",
                    skills=best_cand.skills,
                    target_episodes=len(h_seeds),
                    priority_seeds=h_seeds,
                    record_ticks=False,
                    **fin_eval_kwargs,
                )
                h_cand_stats = _extract_stats(
                    h_cand_eval,
                    best_cand.policy_code,
                    best_cand.skills,
                    loc_override=best_cand.loc_total,
                )
                h_inc_runs = (
                    list(h_inc_eval.get("runs", []) or []) if h_inc_eval else []
                )
                h_cand_runs = list(h_cand_eval.get("runs", []) or [])
                h_pair = evaluate_tournament_pair(
                    h_cand_runs, h_inc_runs, benchmark=bm_name
                )
                holdout_avg_depth = float(h_cand_stats["avg_depth"])
                holdout_iqm_delta = float(h_pair.get("iqm_delta", 0.0))
                h_winsor_d = float(h_pair.get("winsor_delta", holdout_iqm_delta))
                h_safe, h_sreason = _passes_safety_invariants(
                    h_cand_stats, h_inc_stats, has_base=True
                )
                if not h_safe or holdout_iqm_delta < -0.25 or h_winsor_d < -0.25:
                    beats_incumbent = False
                    holdout_rejected = True
                    best_rej_reason = (
                        "holdout_rejected" if h_safe else f"holdout_{h_sreason}"
                    )
                elif holdout_iqm_delta < -0.05 or h_winsor_d < -0.05:
                    # Mild holdout dip: pool main + holdout and re-check unified gate
                    p_cand_runs = list(best_eval.get("runs", []) or []) + h_cand_runs
                    p_inc_runs = inc_runs_b + h_inc_runs
                    p_cand_st = _extract_stats(
                        _assemble_batch_eval("h_pool_c", p_cand_runs),
                        best_cand.policy_code,
                        best_cand.skills,
                        loc_override=best_cand.loc_total,
                    )
                    p_inc_st = _extract_stats(
                        _assemble_batch_eval("h_pool_i", p_inc_runs),
                        base_policy,
                        base_skills,
                        loc_override=base_loc,
                    )
                    p_pair = evaluate_tournament_pair(
                        p_cand_runs, p_inc_runs, benchmark=bm_name
                    )
                    p_ok, _, p_rej, _ = _evaluate_confirmation(
                        p_cand_st, p_pair, p_inc_st, has_base=True, alpha_cutoff=0.05
                    )
                    if not p_ok:
                        beats_incumbent = False
                        holdout_rejected = True
                        best_rej_reason = f"holdout_pooled_{p_rej}"
            except Exception as exc:
                ui.warn(
                    f"Gen {generation}: Holdout evaluation error for `{best_cand.peer_id}` ({exc}); failing closed."
                )
                beats_incumbent = False
                best_rej_reason = "holdout_error"

        d_delta = float(best_stats["avg_depth"]) - float(inc_stats["avg_depth"])
        s_delta = float(best_stats["avg_score"]) - float(inc_stats["avg_score"])

        attempts_summary = []
        seen_ids = set()
        for c, _be, st, pr, c_beats, c_safe, c_rej in eval_rows_b:
            seen_ids.add(c.peer_id)
            is_best = c.peer_id == best_cand.peer_id
            is_prom = bool(beats_incumbent and is_best)
            if is_prom:
                r_txt = "promoted"
            elif is_best:
                r_txt = best_rej_reason
            elif not c_safe:
                r_txt = c_rej
            elif not c_beats:
                r_txt = c_rej
            elif beats_incumbent:
                r_txt = f"outcompeted_by_{best_cand.peer_id}"
            else:
                r_txt = "rejected"
            attempts_summary.append(
                _make_attempt_row(
                    c,
                    st,
                    pr,
                    promoted=is_prom,
                    reject_reason=r_txt,
                    h_avg=holdout_avg_depth if is_best else None,
                    h_iqm=holdout_iqm_delta if is_best else None,
                    h_rej=bool(holdout_rejected and is_best),
                )
            )
        _append_untested_candidates(attempts_summary, seen_ids)

        runner_ups = [
            {
                "peer_id": c.peer_id,
                "policy_code": c.policy_code,
                "skills": dict(c.skills),
                "metrics": st,
                "batch_id": str(be.get("batch_id", "")),
            }
            for c, be, st, _pr, _cb, _cs, _cr in eval_rows_b
            if c.peer_id != best_cand.peer_id or not beats_incumbent
        ]

        scoreboard_str = " | ".join(
            f"{row['peer_id']}: D={row['avg_depth']:.2f}/WΔ={row['winsor_delta']:+.3f}/p={row['sign_perm_p']:.3f}"
            for row in scoreboard
        )
        ui.info(
            f"Gen {generation} CRN Tournament: Incumbent D={inc_stats['avg_depth']:.2f} | {scoreboard_str}"
        )

        if not beats_incumbent and inc_batch_eval is not None:
            if inc_fp:
                rej_set = self._rejected_fingerprints.setdefault(inc_fp, set())
                for c, _be, _st, _pr, _cb, _cs, _cr in eval_rows_b:
                    rej_set.add(_package_fingerprint(c.policy_code, c.skills))
            reject_note = (
                f"{best_rej_reason} "
                f"(Δdepth={d_delta:+.2f}, winsorΔ={best_pair.get('winsor_delta', 0.0):+.3f}, "
                f"p={best_pair.get('sign_perm_p', 1.0):.4f})"
            )
            ui.warn(
                f"Gen {generation}: Best peer `{best_cand.peer_id}` rejected ({reject_note}). Retaining Incumbent."
            )
            retained_eval = _clone_incumbent_eval(inc_batch_eval, generation)
            return TournamentVerdict(
                promoted=False,
                winner_id="incumbent",
                winner_policy_code=base_policy,
                winner_skills=dict(base_skills),
                winner_batch_eval=retained_eval,
                incumbent_avg_depth=inc_stats["avg_depth"],
                incumbent_avg_score=inc_stats["avg_score"],
                winner_avg_depth=inc_stats["avg_depth"],
                winner_avg_score=inc_stats["avg_score"],
                target_functions=sorted(best_cand.touched_branches)
                or [best_cand.claimed_deficit_id],
                scoreboard=scoreboard,
                all_runner_up_batches=runner_ups,
                incumbent_batch_eval=inc_batch_eval,
                attempts_summary=attempts_summary,
                holdout_avg_depth=holdout_avg_depth,
                holdout_iqm_delta=holdout_iqm_delta,
                holdout_rejected=holdout_rejected,
                best_pair=dict(best_pair),
                reason=f"Rejected `{best_cand.peer_id}` ({reject_note}); kept incumbent.",
            )

        self.cache_incumbent(
            eval_pool=eval_pool,
            policy_code=best_cand.policy_code,
            skills=best_cand.skills,
            seeds=s_confirm,
            max_steps=max_steps,
            batch_eval=best_eval,
            stats=best_stats,
            benchmark=bm_name,
        )
        return TournamentVerdict(
            promoted=True,
            winner_id=best_cand.peer_id,
            winner_policy_code=best_cand.policy_code,
            winner_skills=dict(best_cand.skills),
            winner_batch_eval=best_eval,
            incumbent_avg_depth=inc_stats["avg_depth"],
            incumbent_avg_score=inc_stats["avg_score"],
            winner_avg_depth=best_stats["avg_depth"],
            winner_avg_score=best_stats["avg_score"],
            target_functions=sorted(best_cand.touched_branches)
            or [best_cand.claimed_deficit_id],
            scoreboard=scoreboard,
            all_runner_up_batches=runner_ups,
            incumbent_batch_eval=inc_batch_eval,
            attempts_summary=attempts_summary,
            holdout_avg_depth=holdout_avg_depth,
            holdout_iqm_delta=holdout_iqm_delta,
            holdout_rejected=False,
            best_pair=dict(best_pair),
            reason=(
                f"Promoted `{best_cand.peer_id}` targeting `{best_cand.claimed_deficit_id}` "
                f"(Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}, "
                f"winsorΔ={best_pair.get('winsor_delta', 0.0):+.3f}, "
                f"p={best_pair.get('sign_perm_p', 1.0):.4f})."
            ),
        )

    def cache_incumbent(
        self,
        *,
        eval_pool: Any,
        policy_code: str,
        skills: dict[str, str],
        seeds: list[int],
        max_steps: int | None,
        batch_eval: BatchEvaluation,
        stats: dict[str, Any] | None = None,
        benchmark: str = "nethack",
    ) -> None:
        """Stores a verified incumbent evaluation in the per-seed CRN cache."""
        seeds_key = tuple(int(s) for s in seeds)
        steps_key = int(max_steps or 0)
        fp = _package_fingerprint(policy_code, skills)
        loc_total = len(policy_code.splitlines()) + sum(
            len(v.splitlines()) for v in skills.values()
        )
        resolved_stats = stats or _extract_eval_stats(
            batch_eval,
            loc_total=loc_total,
            benchmark=benchmark,
            ast_nodes=_count_ast_nodes(policy_code, skills),
        )
        if len(self._incumbent_cache) >= 4:
            self._incumbent_cache.clear()
        self._incumbent_cache[(id(eval_pool), fp, seeds_key, steps_key)] = (
            batch_eval,
            resolved_stats,
        )
