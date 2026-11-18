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
import json
import time
from dataclasses import dataclass, field
from typing import Any

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_peer import PeerCandidate
from lox.core.contracts import BatchEvaluation, split_full_and_shard_runs
from lox.engine.hypothesis import (
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


def _is_wait_zombie_run(run: dict[str, Any]) -> bool:
    """Returns True when an episode spent >=50 turns heavily stalled on WAIT or stuck on <=1 tile."""
    turns = int(run.get("turns", 0) or 0)
    if turns < 50:
        return False
    if float(run.get("pct_wait", 0.0) or 0.0) >= 65.0:
        return True
    raw_tiles = run.get("unique_tiles")
    return raw_tiles is not None and int(raw_tiles) <= 1 and turns >= 100


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


def _extract_eval_stats(batch_eval: BatchEvaluation, loc_total: int) -> dict[str, Any]:
    runs = batch_eval.get("runs", []) or []
    if not runs:
        return {
            "avg_depth": float(batch_eval.get("avg_depth", 1.0) or 1.0),
            "max_depth": int(batch_eval.get("max_depth", 1) or 1),
            "p75_depth": float(batch_eval.get("avg_depth", 1.0) or 1.0),
            "avg_score": float(batch_eval.get("avg_score", 0.0) or 0.0),
            "avg_ac": 10.0,
            "avg_xp_level": 1.0,
            "avg_oprop": 0.0,
            "avg_branches": 1.0,
            "avg_unique_tiles": 0.0,
            "avg_turns": 100.0,
            "pct_wait": 0.0,
            "pct_search": 0.0,
            "iqm_prog": 1.0,
            "stall_count": 0,
            "hard_stall_count": 0,
            "stall_rate": 0.0,
            "early_blunder_rate": 0.0,
            "policy_exceptions": 0,
            "exc_runs": 0,
            "run_count": 0,
            "top_tracebacks": [],
            "loc_total": loc_total,
        }
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
    oprops = [float(r.get("worn_oprop_count", 0) or 0) for r in runs]
    branches = [float(max(1, int(r.get("dungeons_visited", 1) or 1))) for r in runs]
    tiles = [float(r.get("unique_tiles", 0) or 0) for r in runs]
    turns = [float(r.get("turns", 0) or 0) for r in runs]
    p_wait = [float(r.get("pct_wait", 0.0) or 0.0) for r in runs]
    p_search = [float(r.get("pct_search", 0.0) or 0.0) for r in runs]
    progs = [progression_metric(r) for r in runs]

    confinement_stalls = sum(
        1 for r in runs if "CONFINEMENT_STALL" in str(r.get("summary_digest", ""))
    )
    hard_stalls = sum(
        1
        for r in runs
        if "stalled" in str(r.get("end_reason", "")).lower() or _is_wait_zombie_run(r)
    )
    early_blunders = sum(
        1
        for r in runs
        if int(r.get("turns", 0) or 0) <= 200
        and str(r.get("end_reason", "")).strip().lower()
        not in ("max_steps", "ascended")
    )
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

    import numpy as _np

    n = float(len(runs))
    return {
        "avg_depth": sum(depths) / n,
        "max_depth": int(max(depths)),
        "p75_depth": float(_np.percentile(depths, 75)) if depths else 1.0,
        "avg_score": sum(scores) / n,
        "avg_ac": sum(acs) / n,
        "avg_xp_level": sum(xls) / n,
        "avg_oprop": sum(oprops) / n,
        "avg_branches": sum(branches) / n,
        "avg_unique_tiles": sum(tiles) / n,
        "avg_turns": sum(turns) / n,
        "pct_wait": sum(p_wait) / n,
        "pct_search": sum(p_search) / n,
        "iqm_prog": interquartile_mean(progs),
        "stall_count": confinement_stalls,
        "hard_stall_count": hard_stalls,
        "stall_rate": (confinement_stalls + hard_stalls) / n,
        "early_blunder_rate": early_blunders / n,
        "policy_exceptions": exc_total,
        "exc_runs": exc_runs,
        "run_count": len(runs),
        "top_tracebacks": list(dict.fromkeys(tbs))[:5],
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
        scenario_gate: str = "off",
        frontier_shards: list[Any] | None = None,
    ) -> TournamentVerdict:
        """Runs paired CRN evaluation across Incumbent + Top Survivors (+ Orthogonal Merges, Seed Racing, N-Way Stacking, Holdout)."""
        finalists = self.select_finalists(candidates, base_policy, base_skills)
        orthos = self.build_orthogonal_candidates(
            finalists, base_policy, base_skills, max_combos=3
        )
        if orthos:
            finalists.extend(orthos)

        base_loc = len(base_policy.splitlines()) + sum(
            len(v.splitlines()) for v in base_skills.values()
        )

        # Evaluate Incumbent baseline on identical paired seeds (cached across gens when unchanged)
        inc_batch_eval: BatchEvaluation | None = None
        inc_stats = {
            "avg_depth": 1.0,
            "max_depth": 1,
            "p75_depth": 1.0,
            "avg_score": -1e9,
            "avg_ac": 10.0,
            "avg_oprop": 0.0,
            "avg_branches": 1.0,
            "stall_count": 999,
            "hard_stall_count": 999,
            "stall_rate": 1.0,
            "early_blunder_rate": 1.0,
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
        inc_cached_hit = False
        if base_policy.strip():
            cached_inc = self._incumbent_cache.get(inc_cache_key)
            if cached_inc is not None:
                inc_batch_eval, inc_stats = cached_inc
                inc_cached_hit = True

        can_parallel_dispatch = (
            hasattr(eval_pool, "evaluate_tasks")
            and hasattr(eval_pool, "role")
            and callable(eval_pool.evaluate_tasks)
        )

        # If no novel incubated survivor exists, fall back cleanly to cloned incumbent eval
        if not finalists:
            if base_policy.strip() and not inc_cached_hit:
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
                    record_ticks=True,
                    **eval_kwargs,
                )
                inc_stats = _extract_eval_stats(inc_batch_eval, loc_total=base_loc)
                if len(self._incumbent_cache) >= 4:
                    self._incumbent_cache.clear()
                self._incumbent_cache[inc_cache_key] = (inc_batch_eval, inc_stats)
            if inc_batch_eval is not None:
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

        fin_eval_kwargs: dict[str, Any] = (
            {"max_steps": int(max_steps)} if max_steps is not None else {}
        )

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
            if can_parallel_dispatch:
                eff_steps = int(
                    max_steps
                    if max_steps is not None
                    else getattr(eval_pool, "max_steps", 25000)
                )
                pool_role = str(getattr(eval_pool, "role", "val-hum-law-fem"))
                pool_bench = str(getattr(eval_pool, "benchmark", "nethack"))
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
                                "benchmark": pool_bench,
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

        scoreboard: list[dict[str, Any]] = []
        evaluated_entries: list[
            tuple[PeerCandidate, BatchEvaluation, dict[str, Any]]
        ] = []
        stage1_stats_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

        use_two_stage = (
            race_seeds is not None
            and 4 <= int(race_seeds) < len(seeds)
            and len(finalists) >= 3
        )
        if use_two_stage:
            r_cnt = int(race_seeds)
            stage1_seeds = list(seeds[:r_cnt])
            stage2_seeds = list(seeds[r_cnt:])
            s1_pkgs: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ] = []
            if base_policy.strip() and not inc_cached_hit:
                s1_pkgs.append(("inc", base_policy, base_skills, None, True))
            for cand in finalists:
                s1_pkgs.append(
                    (cand.peer_id, cand.policy_code, cand.skills, cand, True)
                )
            s1_batches = _run_batch_slice(s1_pkgs, stage1_seeds, "s1_")
            if inc_batch_eval is not None:
                s1_seed_set = {int(s) for s in stage1_seeds}
                s1_inc_runs = [
                    r
                    for r in (inc_batch_eval.get("runs", []) or [])
                    if int(r.get("seed", 0)) in s1_seed_set
                ]
            else:
                s1_inc_batch = s1_batches.get("inc")
                s1_inc_runs = (
                    list(s1_inc_batch.get("runs", []) or []) if s1_inc_batch else []
                )

            for cand in finalists:
                cb = s1_batches.get(cand.peer_id)
                st = _extract_eval_stats(cb, loc_total=cand.loc_total) if cb else {}
                pr = (
                    evaluate_tournament_pair(cb.get("runs", []) or [], s1_inc_runs)
                    if (cb and s1_inc_runs)
                    else {"iqm_delta": float(st.get("iqm_prog", 1.0))}
                )
                if cb is not None:
                    stage1_stats_by_id[cand.peer_id] = (st, pr)

            def _s1_score(cand: PeerCandidate) -> tuple[int, int, float, float]:
                if cand.peer_id in stage1_stats_by_id:
                    st, pr = stage1_stats_by_id[cand.peer_id]
                else:
                    st, pr = {}, {"iqm_delta": 0.0, "wins": 0, "losses": 0}
                exc = 1 if int(st.get("policy_exceptions", 0) or 0) == 0 else 0
                non_noop = (
                    1
                    if (
                        not base_policy.strip()
                        or int(pr.get("wins", 0) or 0) > 0
                        or int(pr.get("losses", 0) or 0) > 0
                    )
                    else 0
                )
                return (
                    exc,
                    non_noop,
                    float(pr.get("iqm_delta", 0.0)),
                    float(st.get("avg_depth", 1.0)),
                )

            indiv_fins = sorted(
                [c for c in finalists if not c.peer_id.startswith("ortho_")],
                key=_s1_score,
                reverse=True,
            )
            ortho_fins = sorted(
                [c for c in finalists if c.peer_id.startswith("ortho_")],
                key=_s1_score,
                reverse=True,
            )
            survivors = indiv_fins[:2] + ortho_fins[:1]
            s2_pkgs: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ] = []
            if base_policy.strip() and not inc_cached_hit:
                s2_pkgs.append(("inc", base_policy, base_skills, None, True))
            for cand in survivors:
                s2_pkgs.append(
                    (cand.peer_id, cand.policy_code, cand.skills, cand, True)
                )
            s2_batches = _run_batch_slice(s2_pkgs, stage2_seeds, "s2_")
            seed_rank_full = {int(s): i for i, s in enumerate(seeds)}
            if base_policy.strip() and not inc_cached_hit:
                comb_inc_runs = sorted(
                    list(s1_inc_runs)
                    + list((s2_batches.get("inc") or {}).get("runs", []) or []),
                    key=lambda r: seed_rank_full.get(int(r.get("seed", 0)), 999999),
                )
                inc_batch_eval = _assemble_batch_eval(
                    f"batch_swarm_g{generation}_inc_{int(time.time())}", comb_inc_runs
                )
                inc_stats = _extract_eval_stats(inc_batch_eval, loc_total=base_loc)
                if len(self._incumbent_cache) >= 4:
                    self._incumbent_cache.clear()
                self._incumbent_cache[inc_cache_key] = (inc_batch_eval, inc_stats)
            for cand in survivors:
                r1 = list((s1_batches.get(cand.peer_id) or {}).get("runs", []) or [])
                r2 = list((s2_batches.get(cand.peer_id) or {}).get("runs", []) or [])
                comb_runs = sorted(
                    r1 + r2,
                    key=lambda r: seed_rank_full.get(int(r.get("seed", 0)), 999999),
                )
                b_eval = _assemble_batch_eval(
                    f"batch_swarm_g{generation}_{cand.peer_id}_{int(time.time())}",
                    comb_runs,
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
        else:
            all_pkgs: list[
                tuple[str, str, dict[str, str], PeerCandidate | None, bool]
            ] = []
            if base_policy.strip() and not inc_cached_hit:
                all_pkgs.append(("inc", base_policy, base_skills, None, True))
            for cand in finalists:
                all_pkgs.append(
                    (cand.peer_id, cand.policy_code, cand.skills, cand, True)
                )
            full_batches = _run_batch_slice(all_pkgs, list(seeds), "")
            if base_policy.strip() and not inc_cached_hit and "inc" in full_batches:
                inc_batch_eval = full_batches["inc"]
                inc_stats = _extract_eval_stats(inc_batch_eval, loc_total=base_loc)
                if len(self._incumbent_cache) >= 4:
                    self._incumbent_cache.clear()
                self._incumbent_cache[inc_cache_key] = (inc_batch_eval, inc_stats)
            for cand in finalists:
                b_eval = full_batches[cand.peer_id]
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
            """Prevents runtime exception regressions, score collapses, or new hard stalls."""
            inc_exc = int(inc_stats.get("policy_exceptions", 0) or 0)
            cand_exc = int(cand_stats.get("policy_exceptions", 0) or 0)
            if cand_exc > 0 and (
                not base_policy.strip() or inc_exc == 0 or cand_exc > inc_exc
            ):
                return False
            if not base_policy.strip():
                return True
            inc_s = float(inc_stats["avg_score"])
            cand_s = float(cand_stats["avg_score"])
            d_gain = float(cand_stats["avg_depth"]) - float(inc_stats["avg_depth"])
            if inc_s >= 50.0 and cand_s < inc_s * 0.70 and d_gain < 0.50:
                return False
            inc_hard = int(inc_stats.get("hard_stall_count", 0))
            cand_hard = int(cand_stats.get("hard_stall_count", 0))
            cand_n = max(1, int(cand_stats.get("run_count", 0) or 0))
            stall_tol = cand_n // 16
            inc_zombie = float(inc_stats.get("pct_wait", 0.0) or 0.0) >= 65.0
            cand_active = float(cand_stats.get("pct_wait", 0.0) or 0.0) <= 60.0
            inc_ac = float(inc_stats.get("avg_ac", 10.0) or 10.0)
            cand_ac = float(cand_stats.get("avg_ac", 10.0) or 10.0)
            multi_axis_win = d_gain >= 0.05 and cand_s > inc_s and cand_ac <= inc_ac
            if cand_hard > inc_hard and not (
                inc_zombie and cand_active and d_gain > 0.0
            ):
                if d_gain < 0.05 and cand_s <= inc_s:
                    return False
                hard_cap = inc_hard + (
                    max(2, cand_n // 8) if multi_axis_win else stall_tol
                )
                if cand_hard > hard_cap and d_gain < 0.25:
                    return False
            return True

        # Compute paired hypothesis stats (IQM, CS, Wilcoxon) for each evaluated finalist
        bm_name = str(getattr(eval_pool, "benchmark", "nethack") or "nethack")
        inc_runs_list = (
            (inc_batch_eval.get("runs", []) or []) if inc_batch_eval is not None else []
        )
        paired_by_idx: list[dict[str, Any]] = []
        for _c, b_eval, _st in evaluated_entries:
            p_res = (
                evaluate_tournament_pair(
                    b_eval.get("runs", []) or [],
                    inc_runs_list,
                    benchmark=bm_name,
                )
                if inc_runs_list
                else {
                    "iqm_cand": float(_st.get("iqm_prog", 1.0)),
                    "iqm_inc": 1.0,
                    "iqm_delta": float(_st.get("iqm_prog", 1.0)) - 1.0,
                    "mean_delta": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "wilcoxon_p": 1.0,
                    "wins": len(b_eval.get("runs", []) or []),
                    "losses": 0,
                    "ties": 0,
                    "paired_count": len(b_eval.get("runs", []) or []),
                    "seeds_gained": [],
                    "seeds_lost": [],
                }
            )
            paired_by_idx.append(p_res)

        def _evaluate_promotion_eligibility(
            cand_stats: dict[str, Any],
            pair_stats: dict[str, Any],
        ) -> tuple[bool, float, float, bool]:
            """Returns `(beats_incumbent, d_delta, s_delta, guard_ok)` against incumbent."""
            guard_ok = _passes_score_guard(cand_stats)
            if not base_policy.strip():
                return (
                    guard_ok,
                    float(cand_stats["avg_depth"]) - 1.0,
                    float(cand_stats["avg_score"]),
                    guard_ok,
                )
            d_delta = float(cand_stats["avg_depth"]) - float(inc_stats["avg_depth"])
            s_delta = float(cand_stats["avg_score"]) - float(inc_stats["avg_score"])
            iqm_delta = float(pair_stats.get("iqm_delta", 0.0))
            mean_p_delta = float(pair_stats.get("mean_delta", 0.0))
            wins = int(pair_stats.get("wins", 0))
            losses = int(pair_stats.get("losses", 0))
            n_pairs = int(pair_stats.get("paired_count", 0))
            p_val = float(pair_stats.get("wilcoxon_p", 1.0))
            ci_low = float(pair_stats.get("ci_lower", 0.0))
            ac_improved = (
                float(cand_stats.get("avg_ac", 10.0))
                < float(inc_stats.get("avg_ac", 10.0)) - 0.1
            )
            ac_non_worse = (
                float(cand_stats.get("avg_ac", 10.0))
                <= float(inc_stats.get("avg_ac", 10.0)) + 0.5
            )
            pareto_non_regressing = (
                d_delta >= -0.05
                and iqm_delta >= -0.02
                and ac_non_worse
                and float(cand_stats.get("stall_rate", 0.0))
                <= float(inc_stats.get("stall_rate", 0.0)) + 0.04
                and float(cand_stats.get("early_blunder_rate", 0.0))
                <= float(inc_stats.get("early_blunder_rate", 0.0)) + 0.04
            )
            pareto_axis_gain = (
                float(cand_stats.get("p75_depth", 1.0))
                > float(inc_stats.get("p75_depth", 1.0)) + 0.25
                or float(cand_stats.get("avg_oprop", 0.0))
                > float(inc_stats.get("avg_oprop", 0.0)) + 0.15
                or float(cand_stats.get("avg_branches", 1.0))
                > float(inc_stats.get("avg_branches", 1.0)) + 0.08
                or (ac_improved and iqm_delta >= 0.0)
            )
            pareto_beats = (
                pareto_non_regressing
                and pareto_axis_gain
                and (wins > losses)
                and d_delta >= 0.02
                and s_delta >= 0.0
            )
            inc_zombie_rescue = (
                float(inc_stats.get("pct_wait", 0.0) or 0.0) >= 65.0
                and float(cand_stats.get("pct_wait", 0.0) or 0.0) <= 60.0
                and d_delta > 0.05
                and wins > losses
            )
            stat_sig_ok = (
                (n_pairs < 8)
                or (p_val < 0.05)
                or (ci_low > 0.0)
                or (wins > losses and losses == 0 and d_delta > 0.05 and s_delta >= 0.0)
            )
            is_noop = wins == 0 and losses == 0 and n_pairs >= 4
            beats = (
                guard_ok
                and not is_noop
                and wins > losses
                and (
                    pareto_beats
                    or inc_zombie_rescue
                    or (
                        losses == 0
                        and wins >= max(2, n_pairs // 16)
                        and d_delta > 0.05
                        and s_delta >= 0.0
                        and ac_non_worse
                    )
                    or (
                        (p_val < 0.05 or ci_low > 0.0 or n_pairs < 8)
                        and iqm_delta >= 0.0
                        and d_delta >= -0.02
                        and mean_p_delta >= 0.0
                        and ac_non_worse
                        and stat_sig_ok
                    )
                )
            )
            return (beats, d_delta, s_delta, guard_ok)

        eligibility_by_idx = [
            _evaluate_promotion_eligibility(st, paired_by_idx[idx_e])
            for idx_e, (_c, _be, st) in enumerate(evaluated_entries)
        ]

        def _rank_key(
            idx_item: tuple[int, tuple[PeerCandidate, BatchEvaluation, dict[str, Any]]],
        ) -> tuple[int, int, int, int, float, float, float, int, float, int]:
            idx_e, (_c, _be, st) = idx_item
            pr = paired_by_idx[idx_e]
            cand_beats, _dd, _sd, cand_guard_ok = eligibility_by_idx[idx_e]
            non_noop = (
                1
                if (
                    not base_policy.strip()
                    or int(pr.get("wins", 0) or 0) > 0
                    or int(pr.get("losses", 0) or 0) > 0
                )
                else 0
            )
            return (
                1 if int(st.get("policy_exceptions", 0) or 0) == 0 else 0,
                non_noop,
                1 if cand_guard_ok else 0,
                1 if cand_beats else 0,
                round(float(pr.get("iqm_delta", 0.0)), 4),
                round(float(st["avg_depth"]), 4),
                round(-float(st.get("avg_ac", 10.0)), 2),
                -int(st.get("hard_stall_count", 0)),
                round(float(st["avg_score"]), 2),
                -int(st["loc_total"]),
            )

        ranked_indices = [
            idx_e
            for idx_e, _item in sorted(
                enumerate(evaluated_entries), key=_rank_key, reverse=True
            )
        ]
        best_idx = ranked_indices[0]
        best_cand, best_eval, best_stats = evaluated_entries[best_idx]
        best_pair = paired_by_idx[best_idx]
        beats_incumbent, d_delta, s_delta, guard_ok = eligibility_by_idx[best_idx]

        # N-Way Sequential Greedy Stacking across non-conflicting winning peers
        if beats_incumbent and base_policy.strip():
            winning_indivs = sorted(
                [
                    (c, st, paired_by_idx[idx_w])
                    for idx_w, (c, _be, st) in enumerate(evaluated_entries)
                    if not c.peer_id.startswith("ortho_")
                    and c.peer_id != best_cand.peer_id
                    and _passes_score_guard(st)
                    and (
                        float(paired_by_idx[idx_w].get("iqm_delta", 0.0)) > 0.0
                        or float(st.get("avg_depth", 1.0))
                        > float(inc_stats.get("avg_depth", 1.0)) + 0.02
                    )
                ],
                key=lambda item: float(item[2].get("iqm_delta", 0.0)),
                reverse=True,
            )
            cur_stack_fp = _package_fingerprint(best_cand.policy_code, best_cand.skills)
            for stk_step_i, (next_c, _nxt_st, _nxt_pr) in enumerate(
                winning_indivs[:3], start=1
            ):
                stack_pid = (
                    f"stk{stk_step_i}_{best_cand.peer_id[:16]}_{next_c.peer_id[:16]}"[
                        :48
                    ]
                )
                cand_stack = self._compose_two_packages(
                    base_policy,
                    base_skills,
                    best_cand,
                    next_c,
                    peer_id=stack_pid,
                    claimed_label=f"Greedy N-way stack (+{next_c.peer_id})",
                    check_symbol_disjoint=True,
                )
                if cand_stack is None:
                    continue
                stk_fp = _package_fingerprint(cand_stack.policy_code, cand_stack.skills)
                if stk_fp == cur_stack_fp:
                    continue
                stk_batches = _run_batch_slice(
                    [
                        (
                            cand_stack.peer_id,
                            cand_stack.policy_code,
                            cand_stack.skills,
                            cand_stack,
                            True,
                        )
                    ],
                    list(seeds),
                    "stk_",
                )
                stk_eval = stk_batches.get(cand_stack.peer_id)
                if stk_eval is None:
                    continue
                stk_stats = _extract_eval_stats(
                    stk_eval, loc_total=cand_stack.loc_total
                )
                if not _passes_score_guard(stk_stats):
                    continue
                vs_best = evaluate_tournament_pair(
                    stk_eval.get("runs", []) or [],
                    best_eval.get("runs", []) or [],
                )
                if float(vs_best.get("iqm_delta", 0.0)) > 0.01 and int(
                    vs_best.get("wins", 0)
                ) >= int(vs_best.get("losses", 0)):
                    best_cand = cand_stack
                    best_eval = stk_eval
                    best_stats = stk_stats
                    best_pair = (
                        evaluate_tournament_pair(
                            stk_eval.get("runs", []) or [], inc_runs_list
                        )
                        if inc_runs_list
                        else vs_best
                    )
                    cur_stack_fp = stk_fp
                    d_delta = float(best_stats["avg_depth"]) - float(
                        inc_stats["avg_depth"]
                    )
                    s_delta = float(best_stats["avg_score"]) - float(
                        inc_stats["avg_score"]
                    )

        def _run_post_crn_gates(
            cand_obj: PeerCandidate,
        ) -> tuple[bool, float | None, float | None, bool]:
            """Runs Scenario Gate + hidden Holdout Gate; returns `(ok, holdout_d, holdout_iqm, holdout_rej)`."""
            if str(scenario_gate or "off").lower() in ("advisory", "strict"):
                try:
                    from lox.env.benchmarks.nethack_scenarios import run_scenario_gate

                    sg = run_scenario_gate(
                        cand_obj.policy_code,
                        cand_obj.skills,
                        seed=int(seeds[0]) if seeds else 42,
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
                        return False, None, None, False
                except Exception:
                    pass

            if holdout_seeds and base_policy.strip():
                try:
                    h_seeds = [int(s) for s in holdout_seeds]
                    h_inc_key = (id(eval_pool), inc_fp, tuple(h_seeds), steps_key)
                    cached_h_inc = self._incumbent_cache.get(h_inc_key)
                    if cached_h_inc is not None:
                        h_inc_eval = cached_h_inc[0]
                    else:
                        h_inc_eval = eval_pool.evaluate_batch(
                            batch_id=f"holdout_g{generation}_inc_{int(time.time())}",
                            policy_code=base_policy,
                            macros_code="",
                            skills=base_skills,
                            target_episodes=len(h_seeds),
                            priority_seeds=h_seeds,
                            record_ticks=False,
                            **fin_eval_kwargs,
                        )
                        h_inc_stats = _extract_eval_stats(
                            h_inc_eval, loc_total=base_loc
                        )
                        if len(self._incumbent_cache) >= 4:
                            self._incumbent_cache.pop(next(iter(self._incumbent_cache)))
                        self._incumbent_cache[h_inc_key] = (h_inc_eval, h_inc_stats)
                    h_cand_eval = eval_pool.evaluate_batch(
                        batch_id=f"holdout_g{generation}_{cand_obj.peer_id}_{int(time.time())}",
                        policy_code=cand_obj.policy_code,
                        macros_code="",
                        skills=cand_obj.skills,
                        target_episodes=len(h_seeds),
                        priority_seeds=h_seeds,
                        record_ticks=False,
                        **fin_eval_kwargs,
                    )
                    h_cand_stats = _extract_eval_stats(
                        h_cand_eval, loc_total=cand_obj.loc_total
                    )
                    h_pair = evaluate_tournament_pair(
                        h_cand_eval.get("runs", []) or [],
                        h_inc_eval.get("runs", []) or [],
                        benchmark=bm_name,
                    )
                    h_avg_d = float(h_cand_stats.get("avg_depth", 1.0))
                    h_iqm_d = float(h_pair.get("iqm_delta", 0.0))
                    h_exc = int(h_cand_stats.get("policy_exceptions", 0) or 0)
                    if h_exc > 0 or h_iqm_d < -0.05:
                        ui.warn(
                            f"Gen {generation}: Holdout gate cancelled promotion of `{cand_obj.peer_id}` "
                            f"(holdout_iqm_delta={h_iqm_d:+.3f}, exc={h_exc})."
                        )
                        return False, h_avg_d, h_iqm_d, True
                    # Optional Organic Tier-2 Frontier Checkpoint Evaluation (strictly campaign-saved Dlvl >= 3 states)
                    if frontier_shards:
                        f_inc_eval = eval_pool.evaluate_batch(
                            batch_id=f"tier2_g{generation}_inc_{int(time.time())}",
                            policy_code=base_policy,
                            macros_code="",
                            skills=base_skills,
                            target_episodes=len(frontier_shards),
                            shards=list(frontier_shards),
                            record_ticks=False,
                            **fin_eval_kwargs,
                        )
                        f_cand_eval = eval_pool.evaluate_batch(
                            batch_id=f"tier2_g{generation}_{cand_obj.peer_id}_{int(time.time())}",
                            policy_code=cand_obj.policy_code,
                            macros_code="",
                            skills=cand_obj.skills,
                            target_episodes=len(frontier_shards),
                            shards=list(frontier_shards),
                            record_ticks=False,
                            **fin_eval_kwargs,
                        )
                        f_cand_stats = _extract_eval_stats(
                            f_cand_eval, loc_total=cand_obj.loc_total
                        )
                        f_pair = evaluate_tournament_pair(
                            f_cand_eval.get("runs", []) or [],
                            f_inc_eval.get("runs", []) or [],
                            benchmark=bm_name,
                        )
                        f_exc = int(f_cand_stats.get("policy_exceptions", 0) or 0)
                        f_iqm_d = float(f_pair.get("iqm_delta", 0.0))
                        if f_exc > 0 or f_iqm_d < -0.10:
                            ui.warn(
                                f"Gen {generation}: Organic Tier-2 Frontier gate cancelled `{cand_obj.peer_id}` "
                                f"(frontier_iqm_delta={f_iqm_d:+.3f}, exc={f_exc})."
                            )
                            return False, h_avg_d, h_iqm_d, True
                    return True, h_avg_d, h_iqm_d, False
                except Exception as exc:
                    ui.warn(
                        f"Gen {generation}: Post-CRN gate error for `{cand_obj.peer_id}` ({exc}); failing closed."
                    )
                    return False, None, None, False
            return True, None, None, False

        # Optional Scenario + Hidden Holdout Gate (with fallback to next eligible finalist on rejection)
        holdout_avg_depth: float | None = None
        holdout_iqm_delta: float | None = None
        holdout_rejected = False
        if beats_incumbent:
            gate_ok, holdout_avg_depth, holdout_iqm_delta, holdout_rejected = (
                _run_post_crn_gates(best_cand)
            )
            if not gate_ok:
                beats_incumbent = False
                for fb_idx in ranked_indices:
                    fb_beats, fb_dd, fb_sd, fb_guard = eligibility_by_idx[fb_idx]
                    if not fb_beats:
                        continue
                    fb_cand, fb_eval, fb_stats = evaluated_entries[fb_idx]
                    if fb_cand.peer_id == best_cand.peer_id:
                        continue
                    fb_ok, fb_hd, fb_hiqm, fb_hrej = _run_post_crn_gates(fb_cand)
                    if fb_ok:
                        best_cand = fb_cand
                        best_eval = fb_eval
                        best_stats = fb_stats
                        best_pair = paired_by_idx[fb_idx]
                        d_delta, s_delta, guard_ok = fb_dd, fb_sd, fb_guard
                        holdout_avg_depth, holdout_iqm_delta, holdout_rejected = (
                            fb_hd,
                            fb_hiqm,
                            fb_hrej,
                        )
                        beats_incumbent = True
                        break

        # Build complete attempts_summary for DuckDB `tournament_attempts`
        evaluated_ids = {c.peer_id for c, _be, _st in evaluated_entries}
        attempts_summary: list[dict[str, Any]] = []
        for idx_e, (c, _be, st) in enumerate(evaluated_entries):
            pr = paired_by_idx[idx_e]
            is_best = c.peer_id == best_cand.peer_id
            attempts_summary.append(
                {
                    "attempt_id": f"att_g{generation}_{c.peer_id}_{int(time.time() * 1000)}",
                    "generation": generation,
                    "peer_id": c.peer_id,
                    "deficit_id": c.claimed_deficit_id,
                    "candidate_fp": _package_fingerprint(c.policy_code, c.skills),
                    "promoted": bool(beats_incumbent and is_best),
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
                    "holdout_avg_depth": holdout_avg_depth if is_best else None,
                    "holdout_iqm_delta": holdout_iqm_delta if is_best else None,
                    "holdout_rejected": bool(holdout_rejected and is_best),
                }
            )
        if best_cand.peer_id not in evaluated_ids and best_cand.peer_id != "incumbent":
            evaluated_ids.add(best_cand.peer_id)
            attempts_summary.append(
                {
                    "attempt_id": f"att_g{generation}_{best_cand.peer_id}_{int(time.time() * 1000)}",
                    "generation": generation,
                    "peer_id": best_cand.peer_id,
                    "deficit_id": best_cand.claimed_deficit_id,
                    "candidate_fp": _package_fingerprint(
                        best_cand.policy_code, best_cand.skills
                    ),
                    "promoted": bool(beats_incumbent),
                    "sandbox_passed": bool(best_cand.sandbox_passed),
                    "sandbox_error": str(best_cand.error or ""),
                    "avg_depth": float(best_stats.get("avg_depth", 1.0)),
                    "max_depth": int(best_stats.get("max_depth", 1)),
                    "avg_score": float(best_stats.get("avg_score", 0.0)),
                    "avg_ac": float(best_stats.get("avg_ac", 10.0)),
                    "avg_xp_level": float(best_stats.get("avg_xp_level", 1.0)),
                    "policy_exceptions": int(best_stats.get("policy_exceptions", 0)),
                    "seeds_gained": list(best_pair.get("seeds_gained", [])),
                    "seeds_lost": list(best_pair.get("seeds_lost", [])),
                    "top_tracebacks": list(best_stats.get("top_tracebacks", [])),
                    "holdout_avg_depth": holdout_avg_depth,
                    "holdout_iqm_delta": holdout_iqm_delta,
                    "holdout_rejected": bool(holdout_rejected),
                }
            )
        for c in candidates:
            if c.peer_id in evaluated_ids:
                continue
            if c.peer_id in stage1_stats_by_id:
                s1_st, s1_pr = stage1_stats_by_id[c.peer_id]
                attempts_summary.append(
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
                        "avg_depth": float(s1_st.get("avg_depth", 1.0)),
                        "max_depth": int(s1_st.get("max_depth", 1)),
                        "avg_score": float(s1_st.get("avg_score", 0.0)),
                        "avg_ac": float(s1_st.get("avg_ac", 10.0)),
                        "avg_xp_level": float(s1_st.get("avg_xp_level", 1.0)),
                        "policy_exceptions": int(s1_st.get("policy_exceptions", 0)),
                        "seeds_gained": list(s1_pr.get("seeds_gained", [])),
                        "seeds_lost": list(s1_pr.get("seeds_lost", [])),
                        "top_tracebacks": list(s1_st.get("top_tracebacks", [])),
                    }
                )
                continue
            attempts_summary.append(
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
                }
            )

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
                "skills": dict(c.skills),
                "metrics": st,
                "batch_id": str(be.get("batch_id", "")),
            }
            for c, be, st in evaluated_entries
            if c.peer_id != best_cand.peer_id or not beats_incumbent
        ]

        if not beats_incumbent and inc_batch_eval is not None:
            if inc_fp:
                rej_set = self._rejected_fingerprints.setdefault(inc_fp, set())
                for c, _be, _st in evaluated_entries:
                    rej_set.add(_package_fingerprint(c.policy_code, c.skills))
            if not guard_ok:
                reject_note = "guard triggered (exceptions, score collapse, or hard-stall regression)"
            elif holdout_rejected:
                reject_note = (
                    f"holdout gate rejected (holdout_iqm_delta={holdout_iqm_delta or 0.0:+.3f}, "
                    f"CRN Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f})"
                )
            else:
                reject_note = (
                    f"regressed vs incumbent (Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}, "
                    f"IQMΔ={best_pair.get('iqm_delta', 0.0):+.3f})"
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
            seeds=seeds,
            max_steps=max_steps,
            batch_eval=best_eval,
            stats=best_stats,
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
                f"(Δdepth={d_delta:+.2f}, Δscore={s_delta:+.1f}, IQMΔ={best_pair.get('iqm_delta', 0.0):+.3f})."
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
    ) -> None:
        """Stores a verified incumbent evaluation in the cross-generation CRN cache."""
        seeds_key = tuple(int(s) for s in seeds)
        steps_key = int(max_steps or 0)
        fp = _package_fingerprint(policy_code, skills)
        loc_total = len(policy_code.splitlines()) + sum(
            len(v.splitlines()) for v in skills.values()
        )
        resolved_stats = stats or _extract_eval_stats(batch_eval, loc_total=loc_total)
        if len(self._incumbent_cache) >= 4:
            self._incumbent_cache.clear()
        self._incumbent_cache[(id(eval_pool), fp, seeds_key, steps_key)] = (
            batch_eval,
            resolved_stats,
        )
