"""
Dedicated Extraneous Modular Refactor Agent & CRN Non-Regression Gate.

Keeps `policy.py` and `skills/*.py` clean, readable, and modular for the peer swarm
without diverting any main gameplay Peer Agents from empirical depth/score gains.
Every refactor is strictly gated by a paired CRN non-regression check.
"""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lox.author.agent import AuthorAgent
from lox.author.swarm_meta import SwarmMetaAgent
from lox.core.contracts import BatchEvaluation
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package
from lox.ui.console import ui


@dataclass
class ModularizeResult:
    """Outcome of a single extraneous modular-refactoring pass."""

    attempted: bool
    accepted: bool
    reason: str
    policy_code: str
    skills: dict[str, str]
    batch_eval: BatchEvaluation | None = None
    incumbent_avg_depth: float = 0.0
    refactored_avg_depth: float = 0.0
    incumbent_avg_score: float = 0.0
    refactored_avg_score: float = 0.0
    wilcoxon_p: float = 1.0
    loc_before: int = 0
    loc_after: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


def _paired_less_p_value(diffs: list[float]) -> float:
    """One-sided paired p-value testing H1: median(diffs) < 0 (regression)."""
    nz = [float(d) for d in diffs if abs(float(d)) > 1e-9]
    if len(nz) < 3:
        return 1.0
    if all(d >= 0.0 for d in nz):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        _stat, p_val = wilcoxon(nz, alternative="less", zero_method="wilcox")
        p = float(p_val)
        return p if math.isfinite(p) else 1.0
    except Exception:
        # Exact binomial sign-test fallback when scipy cannot run (e.g. ties)
        neg = sum(1 for d in nz if d < 0.0)
        n = len(nz)
        prob = sum(math.comb(n, k) for k in range(neg, n + 1)) / (2**n)
        return float(min(1.0, max(0.0, prob)))


def _run_depth_val(r: dict[str, Any]) -> float:
    return float(max(int(r.get("depth", 1) or 1), int(r.get("max_depth", 1) or 1)))


def verify_modular_non_regression(
    base_eval: BatchEvaluation,
    ref_eval: BatchEvaluation,
    seeds: list[int],
    *,
    max_depth_drop: float = 0.10,
    min_score_ratio: float = 0.90,
    alpha: float = 0.10,
) -> tuple[bool, dict[str, Any]]:
    """
    Paired CRN non-regression check ensuring modularization preserves gameplay quality.

    Rejects if the refactored bundle introduces runtime exceptions, drops mean depth
    beyond `max_depth_drop`, drops mean score below `min_score_ratio`, increases stall
    counts noticeably, or exhibits a statistically detectable paired regression (`p < alpha`).
    """
    base_by_seed = {int(r.get("seed", 0)): r for r in (base_eval.get("runs", []) or [])}
    ref_by_seed = {int(r.get("seed", 0)): r for r in (ref_eval.get("runs", []) or [])}

    common_seeds = [
        int(s) for s in seeds if int(s) in base_by_seed and int(s) in ref_by_seed
    ]
    if not common_seeds:
        common_seeds = sorted(set(base_by_seed.keys()) & set(ref_by_seed.keys()))
    if not common_seeds:
        return False, {"reason": "No overlapping CRN seeds to verify non-regression."}

    base_depths = [_run_depth_val(base_by_seed[s]) for s in common_seeds]
    ref_depths = [_run_depth_val(ref_by_seed[s]) for s in common_seeds]
    base_scores = [
        float(base_by_seed[s].get("score", 0.0) or 0.0) for s in common_seeds
    ]
    ref_scores = [float(ref_by_seed[s].get("score", 0.0) or 0.0) for s in common_seeds]

    base_avg_d = float(np.mean(base_depths))
    ref_avg_d = float(np.mean(ref_depths))
    base_avg_s = float(np.mean(base_scores))
    ref_avg_s = float(np.mean(ref_scores))
    d_delta = ref_avg_d - base_avg_d
    s_delta = ref_avg_s - base_avg_s

    base_exc = sum(
        int(base_by_seed[s].get("policy_exceptions", 0) or 0) for s in common_seeds
    )
    ref_exc = sum(
        int(ref_by_seed[s].get("policy_exceptions", 0) or 0) for s in common_seeds
    )
    base_stalls = sum(
        1
        for s in common_seeds
        if str(base_by_seed[s].get("end_reason", ""))
        in ("max_steps", "stalled_zero_turn")
    )
    ref_stalls = sum(
        1
        for s in common_seeds
        if str(ref_by_seed[s].get("end_reason", ""))
        in ("max_steps", "stalled_zero_turn")
    )

    depth_diffs = [r - b for b, r in zip(base_depths, ref_depths, strict=True)]
    score_diffs = [r - b for b, r in zip(base_scores, ref_scores, strict=True)]
    p_depth = _paired_less_p_value(depth_diffs)
    p_score = _paired_less_p_value(score_diffs)
    min_p = min(p_depth, p_score)

    stall_margin = max(1, int(0.10 * len(common_seeds)))
    reasons: list[str] = []
    if ref_exc > base_exc:
        reasons.append(f"new policy_exceptions ({base_exc}->{ref_exc})")
    if d_delta < -max_depth_drop:
        reasons.append(f"depth drop Δ={d_delta:+.2f} < -{max_depth_drop:.2f}")
    if base_avg_s >= 50.0 and ref_avg_s < base_avg_s * min_score_ratio:
        reasons.append(
            f"score drop {base_avg_s:.1f}->{ref_avg_s:.1f} (<{min_score_ratio * 100:.0f}%)"
        )
    if ref_stalls > base_stalls + stall_margin:
        reasons.append(f"stalls rose {base_stalls}->{ref_stalls}")
    if d_delta < -1e-4 and p_depth < alpha:
        reasons.append(f"paired depth regression (p={p_depth:.3f} < {alpha:.2f})")
    if s_delta < -1e-2 and p_score < alpha:
        reasons.append(f"paired score regression (p={p_score:.3f} < {alpha:.2f})")

    passed = len(reasons) == 0
    metrics = {
        "n_paired": len(common_seeds),
        "base_avg_depth": base_avg_d,
        "ref_avg_depth": ref_avg_d,
        "depth_delta": d_delta,
        "base_avg_score": base_avg_s,
        "ref_avg_score": ref_avg_s,
        "score_delta": s_delta,
        "base_stalls": base_stalls,
        "ref_stalls": ref_stalls,
        "base_exceptions": base_exc,
        "ref_exceptions": ref_exc,
        "p_depth_reg": p_depth,
        "p_score_reg": p_score,
        "wilcoxon_p": min_p,
        "reason": "Passed paired CRN non-regression check."
        if passed
        else "Rejected: " + "; ".join(reasons),
    }
    return passed, metrics


class SwarmModularizerAgent:
    """
    Single extraneous house-keeping agent that extracts bloated `policy.py` subsystems
    into cohesive `skills/*.py` modules after gameplay evaluation, gated by CRN parity.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        db_path: str = "data/lox.duckdb",
        theories_store: Any = None,
        max_tool_turns: int = 6,
    ) -> None:
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = theories_store
        self.max_tool_turns = max_tool_turns
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role="modularizer",
        )
        if theories_store is not None:
            self.author.tools.theories_store = theories_store

    @staticmethod
    def should_trigger(
        policy_code: str,
        skills: dict[str, str],
        *,
        loc_threshold: int = 350,
        monolith_threshold: int = 300,
        max_skill_modules: int = 6,
    ) -> bool:
        """Triggers modularization when `policy.py` grows bulky or `skills/` sprawls."""
        if not policy_code.strip():
            return False
        policy_loc = len(policy_code.splitlines())
        skill_count = len(skills or {})
        if policy_loc >= loc_threshold:
            return True
        if skill_count == 0 and policy_loc >= monolith_threshold:
            return True
        if skill_count >= max_skill_modules and policy_loc >= 180:
            return True
        return False

    @staticmethod
    def _mock_modularize(
        policy_code: str, skills: dict[str, str]
    ) -> tuple[str, dict[str, str]]:
        """Deterministic helper extraction for mock-mode tests."""
        tree = ast.parse(policy_code)
        lines = policy_code.splitlines()
        helper_blocks: list[str] = []
        keep_ranges: list[tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in (
                "build_tree",
                "step",
                "run",
            ):
                if getattr(node, "end_lineno", None):
                    seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    helper_blocks.append(seg)
                    keep_ranges.append((node.lineno - 1, node.end_lineno))
        if not helper_blocks:
            return policy_code, dict(skills)
        removed: set[int] = set()
        for a, b in keep_ranges:
            removed.update(range(a, b))
        kept = [ln for idx, ln in enumerate(lines) if idx not in removed]
        mod_name = "extracted_helpers"
        mod_header = (
            "from lox.env.actions import Action, DIR_DELTAS, key, seq\n"
            "from lox.lang.bt import ActionLeaf, Blackboard, PrioritySelector, Sequence\n\n"
        )
        mod_src = mod_header + "\n\n".join(helper_blocks) + "\n"
        fn_names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name not in ("build_tree", "step", "run")
        ]
        new_pol = f"from skills.{mod_name} import {', '.join(fn_names)}\n" + "\n".join(
            kept
        )
        new_skills = dict(skills)
        new_skills[mod_name] = mod_src
        return new_pol, new_skills

    def run_refactor(
        self,
        *,
        generation: int,
        policy_code: str,
        skills: dict[str, str],
        baseline_eval: BatchEvaluation,
        eval_pool: Any,
        seeds: list[int],
        max_steps: int | None = None,
        dormant_branches: list[str] | None = None,
        frontier: Any = None,
    ) -> ModularizeResult:
        """Runs a single behavior-preserving modularization pass + paired CRN check."""
        base_loc = len(policy_code.splitlines())
        base_skills = dict(skills or {})
        base_avg_d = float(baseline_eval.get("avg_depth", 1.0) or 1.0)
        base_avg_s = float(baseline_eval.get("avg_score", 0.0) or 0.0)

        if self.provider == "mock":
            ref_policy, updated_skills = self._mock_modularize(policy_code, base_skills)
        else:
            self.author.tools.configure_generation(
                pool=None,
                baseline_runs=[],
                skills=base_skills,
                generation=generation,
                mode="SWARM_PEER",
                swarm_frontier=frontier,
            )
            self.author.tools.current_policy_code = str(policy_code)

            dormant_note = (
                f"- Dormant zero-turn BT branches (safe to inline/prune only if dead): {', '.join(dormant_branches[:8])}\n"
                if dormant_branches
                else ""
            )
            task_digest = (
                f"### Extraneous Modular Refactoring Pass (Generation {generation})\n"
                f"**ROLE**: You are the dedicated `modularizer` agent. Your SOLE job is structural hygiene—NOT gameplay optimization.\n"
                f"- Current footprint: `policy.py` = {base_loc} LOC, `skills/` = {len(base_skills)} module(s).\n"
                f"{dormant_note}"
                "### Strict Rules (Behavioral Parity Gate):\n"
                "1. Extract cohesive helper groups / BT leaf subtrees out of `policy.py` into 2–5 clean `skills/<domain>.py` modules "
                "(e.g., `skills/combat.py`, `skills/nutrition.py`, `skills/navigation.py`, `skills/prompts.py`).\n"
                "2. Keep `policy.py` concise (target <= 250 LOC) as the clean composition root importing from `skills.<module>` and wiring `build_tree` + `class Policy`.\n"
                "3. **DO NOT** alter thresholds, heuristics, priority order, or Blackboard variable semantics! Preserve behavior 1-to-1.\n"
                "4. Verify via `dry_run_policy(...)` before outputting `<hypothesis>`, `<skill_code module='...'>`, and `<policy_code>`.\n"
                "5. Your refactored bundle MUST pass a paired CRN non-regression check against the current incumbent or it will be rejected.\n"
            )
            proposal = self.author.synthesize(
                mode="SWARM_PEER",
                generation=generation,
                current_policy=policy_code,
                batch_digest=task_digest,
                skills=base_skills,
                max_tool_turns=self.max_tool_turns,
            )
            if getattr(proposal, "aborted", False):
                return ModularizeResult(
                    attempted=True,
                    accepted=False,
                    reason=f"Modularizer aborted: {getattr(proposal, 'abort_reason', 'aborted')}",
                    policy_code=policy_code,
                    skills=base_skills,
                    incumbent_avg_depth=base_avg_d,
                    incumbent_avg_score=base_avg_s,
                    loc_before=base_loc,
                    loc_after=base_loc,
                )

            ref_policy = str(getattr(proposal, "full_policy_code", "") or "").strip()
            updated_skills = dict(base_skills)
            for s_name, s_src in (getattr(proposal, "skill_modules", {}) or {}).items():
                clean_s = str(s_name or "").strip().removesuffix(".py")
                if clean_s and str(s_src or "").strip():
                    updated_skills[clean_s] = str(s_src).strip()

        if not ref_policy.strip():
            return ModularizeResult(
                attempted=True,
                accepted=False,
                reason="Modularizer returned empty policy_code.",
                policy_code=policy_code,
                skills=base_skills,
                incumbent_avg_depth=base_avg_d,
                incumbent_avg_score=base_avg_s,
                loc_before=base_loc,
                loc_after=base_loc,
            )

        kept_skills = SwarmMetaAgent.prune_unused_skills(ref_policy, updated_skills)
        after_loc = len(ref_policy.splitlines())

        # Structural sanity check before spending CRN eval compute
        shrank_policy = after_loc <= base_loc - 20
        reduced_sprawl = (
            len(base_skills) >= 6 and len(kept_skills) <= 5 and after_loc <= base_loc
        )
        added_skills = (
            len(base_skills) == 0 and len(kept_skills) >= 1 and after_loc < base_loc
        )
        if not (shrank_policy or reduced_sprawl or added_skills):
            return ModularizeResult(
                attempted=True,
                accepted=False,
                reason=f"No meaningful modularization gain (policy LOC {base_loc}->{after_loc}, skills {len(base_skills)}->{len(kept_skills)}).",
                policy_code=policy_code,
                skills=base_skills,
                incumbent_avg_depth=base_avg_d,
                incumbent_avg_score=base_avg_s,
                loc_before=base_loc,
                loc_after=after_loc,
            )

        try:
            with _COMPILE_EXEC_LOCK:
                compile_policy_package(ref_policy, "", skills=kept_skills)
        except Exception as exc:
            return ModularizeResult(
                attempted=True,
                accepted=False,
                reason=f"Modularized bundle failed compile check: {exc}",
                policy_code=policy_code,
                skills=base_skills,
                incumbent_avg_depth=base_avg_d,
                incumbent_avg_score=base_avg_s,
                loc_before=base_loc,
                loc_after=after_loc,
            )

        eval_kwargs: dict[str, Any] = (
            {"max_steps": int(max_steps)} if max_steps is not None else {}
        )
        ref_eval = eval_pool.evaluate_batch(
            batch_id=f"batch_swarm_g{generation}_modularizer_{int(time.time())}",
            policy_code=ref_policy,
            macros_code="",
            skills=kept_skills,
            target_episodes=len(seeds),
            priority_seeds=list(seeds),
            record_ticks=True,
            **eval_kwargs,
        )

        passed, metrics = verify_modular_non_regression(baseline_eval, ref_eval, seeds)
        if not passed:
            ui.warn(
                f"Gen {generation} Modularizer rejected ({metrics['reason']}); keeping un-refactored policy."
            )
            return ModularizeResult(
                attempted=True,
                accepted=False,
                reason=str(metrics["reason"]),
                policy_code=policy_code,
                skills=base_skills,
                incumbent_avg_depth=metrics["base_avg_depth"],
                refactored_avg_depth=metrics["ref_avg_depth"],
                incumbent_avg_score=metrics["base_avg_score"],
                refactored_avg_score=metrics["ref_avg_score"],
                wilcoxon_p=metrics["wilcoxon_p"],
                loc_before=base_loc,
                loc_after=after_loc,
                metrics=metrics,
            )

        ui.info(
            f"Gen {generation} Modularizer ACCEPTED: policy.py {base_loc}->{after_loc} LOC, "
            f"{len(kept_skills)} skill(s) (D={metrics['ref_avg_depth']:.2f}, S={metrics['ref_avg_score']:.0f}, p={metrics['wilcoxon_p']:.2f})."
        )
        return ModularizeResult(
            attempted=True,
            accepted=True,
            reason=str(metrics["reason"]),
            policy_code=ref_policy,
            skills=kept_skills,
            batch_eval=ref_eval,
            incumbent_avg_depth=metrics["base_avg_depth"],
            refactored_avg_depth=metrics["ref_avg_depth"],
            incumbent_avg_score=metrics["base_avg_score"],
            refactored_avg_score=metrics["ref_avg_score"],
            wilcoxon_p=metrics["wilcoxon_p"],
            loc_before=base_loc,
            loc_after=after_loc,
            metrics=metrics,
        )
