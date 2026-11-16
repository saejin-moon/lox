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

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_peer import PeerCandidate
from lox.core.contracts import BatchEvaluation
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package
from lox.lang.validator import is_ast_equivalent
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
    reason: str = ""


def _package_fingerprint(policy_code: str, skills: dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(policy_code.strip().encode("utf-8", errors="replace"))
    for k in sorted(skills.keys()):
        h.update(f"\n##{k}\n".encode("utf-8"))
        h.update(skills[k].strip().encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _extract_eval_stats(batch_eval: BatchEvaluation, loc_total: int) -> dict[str, Any]:
    runs = batch_eval.get("runs", []) or []
    if not runs:
        return {
            "avg_depth": float(batch_eval.get("avg_depth", 1.0) or 1.0),
            "max_depth": int(batch_eval.get("max_depth", 1) or 1),
            "avg_score": float(batch_eval.get("avg_score", 0.0) or 0.0),
            "stall_count": 0,
            "loc_total": loc_total,
        }
    depths = [
        float(max(int(r.get("depth", 1) or 1), int(r.get("max_depth", 1) or 1)))
        for r in runs
    ]
    scores = [float(r.get("score", 0.0) or 0.0) for r in runs]
    confinement_stalls = sum(
        1 for r in runs if "CONFINEMENT_STALL" in str(r.get("summary_digest", ""))
    )
    return {
        "avg_depth": sum(depths) / len(depths),
        "max_depth": int(max(depths)),
        "avg_score": sum(scores) / len(scores),
        "stall_count": confinement_stalls,
        "loc_total": loc_total,
    }


class SwarmReferee:
    """Deterministic CRN tournament selector & AST-disjoint orthogonal pair builder."""

    def __init__(self, max_tournament_peers: int = 3) -> None:
        self.max_tournament_peers = max(1, int(max_tournament_peers))
        self._incumbent_cache: dict[
            tuple[int, str, tuple[int, ...], int],
            tuple[BatchEvaluation, dict[str, Any]],
        ] = {}

    def select_finalists(
        self,
        candidates: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
    ) -> list[PeerCandidate]:
        """Filters to valid incubated candidates, removes duplicates, and ranks top-K."""
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

        # Deduplicate identical policy+skills bundles (including identical-to-incumbent no-ops when alternatives exist)
        base_fp = (
            _package_fingerprint(base_policy, base_skills)
            if base_policy.strip()
            else ""
        )
        unique: dict[str, PeerCandidate] = {}
        for c in pool:
            fp = _package_fingerprint(c.policy_code, c.skills)
            if fp == base_fp and len(pool) > 1:
                continue
            if fp not in unique:
                unique[fp] = c

        ranked = sorted(
            unique.values(),
            key=lambda c: (
                float(c.sandbox_metrics.get("depth_delta", 0.0)),
                float(c.sandbox_metrics.get("score_delta", 0.0)),
                -float(c.sandbox_metrics.get("wrapper_ratio", 0.0)),
                -int(c.loc_total),
            ),
            reverse=True,
        )
        return ranked[: self.max_tournament_peers]

    @staticmethod
    def build_orthogonal_pair(
        finalists: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
    ) -> PeerCandidate | None:
        """
        Attempts to compose 1 zero-conflict orthogonal candidate from the top-2 survivors
        iff they touch disjoint `skills/` modules & disjoint branches and at most one modified `policy.py`.
        """
        if len(finalists) < 2 or not base_policy.strip():
            return None

        a, b = finalists[0], finalists[1]
        if not a.touched_skills.isdisjoint(b.touched_skills):
            return None
        if not a.touched_branches.isdisjoint(b.touched_branches):
            return None

        a_changed_pol = not is_ast_equivalent(a.policy_code, base_policy)
        b_changed_pol = not is_ast_equivalent(b.policy_code, base_policy)

        if a_changed_pol and b_changed_pol:
            return None

        chosen_pol = (
            a.policy_code
            if a_changed_pol
            else (b.policy_code if b_changed_pol else base_policy)
        )
        merged_skills = dict(base_skills)
        for sk in a.touched_skills:
            if sk in a.skills:
                merged_skills[sk] = a.skills[sk]
            else:
                merged_skills.pop(sk, None)
        for sk in b.touched_skills:
            if sk in b.skills:
                merged_skills[sk] = b.skills[sk]
            else:
                merged_skills.pop(sk, None)

        kept_skills = SwarmMetaAgent.prune_unused_skills(chosen_pol, merged_skills)
        try:
            with _COMPILE_EXEC_LOCK:
                compile_policy_package(chosen_pol, "", skills=kept_skills)
        except Exception:
            return None

        loc_total = len(chosen_pol.splitlines()) + sum(
            len(v.splitlines()) for v in kept_skills.values()
        )
        return PeerCandidate(
            peer_id=f"ortho_{a.peer_id}_{b.peer_id}",
            claimed_deficit_id=f"{a.claimed_deficit_id}+{b.claimed_deficit_id}",
            claimed_label=f"Orthogonal merge: {a.peer_id} + {b.peer_id}",
            hypothesis_yaml={
                "mechanism": f"Verified disjoint combination of {a.peer_id} and {b.peer_id}"
            },
            policy_code=chosen_pol,
            skills=kept_skills,
            touched_skills=a.touched_skills | b.touched_skills,
            touched_branches=a.touched_branches | b.touched_branches,
            sandbox_passed=True,
            loc_total=loc_total,
        )

    def evaluate_tournament(
        self,
        generation: int,
        candidates: list[PeerCandidate],
        base_policy: str,
        base_skills: dict[str, str],
        eval_pool: Any,
        seeds: list[int],
        max_steps: int | None = None,
    ) -> TournamentVerdict:
        """Runs paired CRN evaluation across Incumbent + Top Survivors (+ Orthogonal Pair)."""
        finalists = self.select_finalists(candidates, base_policy, base_skills)
        ortho = self.build_orthogonal_pair(finalists, base_policy, base_skills)
        if ortho is not None:
            finalists.append(ortho)

        base_loc = len(base_policy.splitlines()) + sum(
            len(v.splitlines()) for v in base_skills.values()
        )

        # Evaluate Incumbent baseline on identical paired seeds (cached across gens when unchanged)
        inc_batch_eval: BatchEvaluation | None = None
        inc_stats = {
            "avg_depth": 1.0,
            "max_depth": 1,
            "avg_score": -1e9,
            "stall_count": 999,
            "loc_total": base_loc,
        }
        seeds_key = tuple(int(s) for s in seeds)
        steps_key = int(max_steps or 0)
        inc_fp = (
            _package_fingerprint(base_policy, base_skills)
            if base_policy.strip()
            else ""
        )
        inc_cache_key = (id(eval_pool), inc_fp, seeds_key, steps_key)
        if base_policy.strip():
            cached_inc = self._incumbent_cache.get(inc_cache_key)
            if cached_inc is not None:
                inc_batch_eval, inc_stats = cached_inc
            else:
                eval_kwargs: dict[str, Any] = (
                    {"max_steps": int(max_steps)} if max_steps is not None else {}
                )
                inc_batch_eval = eval_pool.evaluate_batch(
                    batch_id=f"batch_swarm_g{generation}_inc_{int(time.time())}",
                    policy_code=base_policy,
                    macros_code="",
                    skills=base_skills,
                    target_episodes=len(seeds),
                    priority_seeds=list(seeds),
                    record_ticks=False,
                    **eval_kwargs,
                )
                inc_stats = _extract_eval_stats(inc_batch_eval, loc_total=base_loc)
                if len(self._incumbent_cache) >= 4:
                    self._incumbent_cache.clear()
                self._incumbent_cache[inc_cache_key] = (inc_batch_eval, inc_stats)

        # If no incubated survivor exists, fall back cleanly to incumbent (or fail clearly if Gen 1 empty)
        if not finalists:
            if inc_batch_eval is not None:
                return TournamentVerdict(
                    promoted=False,
                    winner_id="incumbent",
                    winner_policy_code=base_policy,
                    winner_skills=dict(base_skills),
                    winner_batch_eval=inc_batch_eval,
                    incumbent_avg_depth=inc_stats["avg_depth"],
                    incumbent_avg_score=inc_stats["avg_score"],
                    winner_avg_depth=inc_stats["avg_depth"],
                    winner_avg_score=inc_stats["avg_score"],
                    reason="No peer passed isolated sandbox incubation; retained incumbent.",
                )
            # Gen 1 bootstrap safety: pick first raw candidate if any compiled
            raise RuntimeError(
                f"Gen {generation}: Zero peer candidates passed sandbox incubation on bootstrap."
            )

        scoreboard: list[dict[str, Any]] = []
        evaluated_entries: list[
            tuple[PeerCandidate, BatchEvaluation, dict[str, Any]]
        ] = []
        fin_eval_kwargs: dict[str, Any] = (
            {"max_steps": int(max_steps)} if max_steps is not None else {}
        )
        for cand in finalists:
            b_eval = eval_pool.evaluate_batch(
                batch_id=f"batch_swarm_g{generation}_{cand.peer_id}_{int(time.time())}",
                policy_code=cand.policy_code,
                macros_code="",
                skills=cand.skills,
                target_episodes=len(seeds),
                priority_seeds=list(seeds),
                record_ticks=True,
                **fin_eval_kwargs,
            )
            stats = _extract_eval_stats(b_eval, loc_total=cand.loc_total)
            evaluated_entries.append((cand, b_eval, stats))
            scoreboard.append(
                {
                    "peer_id": cand.peer_id,
                    "deficit": cand.claimed_deficit_id,
                    "avg_depth": round(stats["avg_depth"], 3),
                    "max_depth": stats["max_depth"],
                    "avg_score": round(stats["avg_score"], 1),
                    "stalls": stats["stall_count"],
                    "loc": stats["loc_total"],
                }
            )

        def _passes_score_guard(cand_stats: dict[str, Any]) -> bool:
            """Prevents marginal depth ticks (<=0.35) from overriding severe score collapses (>30%)."""
            if not base_policy.strip():
                return True
            inc_s = float(inc_stats["avg_score"])
            cand_s = float(cand_stats["avg_score"])
            d_gain = float(cand_stats["avg_depth"]) - float(inc_stats["avg_depth"])
            if inc_s >= 50.0 and cand_s < inc_s * 0.70 and d_gain < 0.50:
                return False
            return True

        def _rank_key(
            item: tuple[PeerCandidate, BatchEvaluation, dict[str, Any]],
        ) -> tuple[int, float, float, int, int]:
            _c, _be, st = item
            return (
                1 if _passes_score_guard(st) else 0,
                round(float(st["avg_depth"]), 4),
                round(float(st["avg_score"]), 2),
                -int(st["stall_count"]),
                -int(st["loc_total"]),
            )

        best_cand, best_eval, best_stats = max(evaluated_entries, key=_rank_key)

        # Strict non-regression + score-collapse guard vs Incumbent
        if base_policy.strip():
            d_delta = best_stats["avg_depth"] - inc_stats["avg_depth"]
            s_delta = best_stats["avg_score"] - inc_stats["avg_score"]
            loc_improved = best_stats["loc_total"] <= inc_stats["loc_total"]
            guard_ok = _passes_score_guard(best_stats)
            beats_incumbent = guard_ok and (
                (d_delta > 1e-4)
                or (abs(d_delta) <= 1e-4 and s_delta > 1e-2)
                or (abs(d_delta) <= 1e-4 and abs(s_delta) <= 1e-2 and loc_improved)
            )
        else:
            d_delta = best_stats["avg_depth"] - 1.0
            s_delta = best_stats["avg_score"]
            guard_ok = True
            beats_incumbent = True

        scoreboard_str = " | ".join(
            f"{row['peer_id']}: D={row['avg_depth']:.2f}/S={row['avg_score']:.0f}/LOC={row['loc']}"
            for row in scoreboard
        )
        ui.info(
            f"Gen {generation} CRN Tournament: Incumbent D={inc_stats['avg_depth']:.2f} | {scoreboard_str}"
        )

        runner_ups = [
            {
                "peer_id": c.peer_id,
                "policy_code": c.policy_code,
                "metrics": st,
                "batch_id": str(be.get("batch_id", "")),
            }
            for c, be, st in evaluated_entries
            if c.peer_id != best_cand.peer_id
        ]

        if not beats_incumbent and inc_batch_eval is not None:
            reject_note = (
                "score-collapse guard triggered (>30% score drop for <0.50 depth gain)"
                if not guard_ok
                else f"regressed vs Incumbent (Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f})"
            )
            ui.warn(
                f"Gen {generation}: Best peer `{best_cand.peer_id}` rejected ({reject_note}). Retaining Incumbent."
            )
            return TournamentVerdict(
                promoted=False,
                winner_id="incumbent",
                winner_policy_code=base_policy,
                winner_skills=dict(base_skills),
                winner_batch_eval=best_eval,  # Log empirical batch telemetry cleanly with is_promoted=False
                incumbent_avg_depth=inc_stats["avg_depth"],
                incumbent_avg_score=inc_stats["avg_score"],
                winner_avg_depth=best_stats["avg_depth"],
                winner_avg_score=best_stats["avg_score"],
                target_functions=sorted(best_cand.touched_branches)
                or [best_cand.claimed_deficit_id],
                scoreboard=scoreboard,
                all_runner_up_batches=runner_ups,
                reason=f"Rejected regression from `{best_cand.peer_id}` (Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}); kept incumbent.",
            )

        win_fp = _package_fingerprint(best_cand.policy_code, best_cand.skills)
        if len(self._incumbent_cache) >= 4:
            self._incumbent_cache.clear()
        self._incumbent_cache[(id(eval_pool), win_fp, seeds_key, steps_key)] = (
            best_eval,
            best_stats,
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
            reason=f"Promoted `{best_cand.peer_id}` targeting `{best_cand.claimed_deficit_id}` (Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}).",
        )
