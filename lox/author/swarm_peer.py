"""
Symmetric Data-Driven Peer Agent & Isolated Sandbox Incubation (`tau_iso`).

Replaces hierarchical `SwarmCriticAgent` / `SwarmMetaAgent` / thematic `SwarmWorker` silos:
- Every Peer Agent has identical full-stack access to `policy.py`, `skills/*.py`, DuckDB, Wiki, and dry-run tools.
- Peers autonomously claim distinct empirical `DeficitCluster`s from `DeficitReport` (`D_fail`) while
  protecting high-Ochiai `D_succ` branches.
- Each peer develops inside an isolated copy of the workspace and must survive `incubate_candidate()`
  on a paired micro-shard before entering the deterministic CRN Tournament.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.swarm_critic import extract_priority_selector_branches
from lox.author.swarm_deficits import DeficitCluster, DeficitReport
from lox.author.swarm_meta import SwarmMetaAgent
from lox.env.harness import NetHackHarness
from lox.lang.bt import extract_runner_leaf_errors
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package
from lox.ui.console import ui


@dataclass
class PeerCandidate:
    """Self-contained candidate bundle produced and incubated by a SwarmPeerAgent."""

    peer_id: str
    claimed_deficit_id: str
    claimed_label: str
    hypothesis_yaml: dict[str, Any] = field(default_factory=dict)
    policy_code: str = ""
    skills: dict[str, str] = field(default_factory=dict)
    pruned_skills: list[str] = field(default_factory=list)
    touched_skills: set[str] = field(default_factory=set)
    touched_branches: set[str] = field(default_factory=set)
    sandbox_passed: bool = False
    sandbox_metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    loc_total: int = 0
    clobbers_protected: bool = False


def parse_peer_output(
    raw_text: str,
    base_policy: str,
    base_skills: dict[str, str],
) -> tuple[dict[str, Any], str, dict[str, str], list[str]]:
    """
    Extracts `(hypothesis_yaml, policy_code, updated_skills, pruned_skills)` from a Peer response.
    Supports:
    - ```yaml ... ``` with optional `skills_to_prune: [...]`
    - Optional `### skill: <name>` followed by ```python ... ```
    - `<policy_code> ... </policy_code>` or the final ```python ... ``` defining `class Policy`
    """
    hyp, extracted_code = AuthorAgent.extract_yaml_and_python(raw_text or "")

    updated_skills = dict(base_skills or {})
    explicit_prunes: list[str] = []
    raw_prunes = hyp.get("skills_to_prune")
    if isinstance(raw_prunes, list):
        for s in raw_prunes:
            s_name = str(s or "").strip().removesuffix(".py")
            if s_name and s_name in updated_skills:
                updated_skills.pop(s_name, None)
                explicit_prunes.append(s_name)

    skill_blocks = re.findall(
        r"###\s*skill:\s*([A-Za-z0-9_]+)\s*\n+```(?:python)?\s*\n(.*?)```",
        raw_text or "",
        flags=re.DOTALL,
    )
    for s_name, s_src in skill_blocks:
        clean_name = s_name.strip().removesuffix(".py")
        if clean_name and s_src.strip():
            updated_skills[clean_name] = s_src.strip()

    # If <policy_code> wasn't explicit and the trailing python block was actually a ### skill block,
    # inspect any python block containing `class Policy`.
    policy_code = extracted_code.strip()
    if "<policy_code>" not in (raw_text or "") and "class Policy" not in policy_code:
        py_blocks = re.findall(
            r"```(?:python)?\s*\n(.*?)```", raw_text or "", flags=re.DOTALL
        )
        for blk in reversed(py_blocks):
            if "class Policy" in blk:
                policy_code = blk.strip()
                break

    if not policy_code:
        policy_code = str(base_policy or "").strip()

    # Transitive prune so unused skills aren't carried into evaluation
    kept_skills = SwarmMetaAgent.prune_unused_skills(policy_code, updated_skills)
    auto_pruned = [k for k in updated_skills if k not in kept_skills]
    all_pruned = list(dict.fromkeys(explicit_prunes + auto_pruned))
    return hyp, policy_code, kept_skills, all_pruned


def _extract_branch_names(policy_code: str) -> list[str]:
    if not policy_code or not policy_code.strip():
        return []
    raw = extract_priority_selector_branches(policy_code)
    names: list[str] = []
    for b in raw:
        if isinstance(b, dict):
            nm = str(b.get("leaf_name") or b.get("fn_name") or "").strip()
            if nm:
                names.append(nm)
        else:
            s = str(b or "").strip()
            if s:
                names.append(s)
    return names


def compute_candidate_footprint(
    policy_code: str,
    skills: dict[str, str],
    base_policy: str,
    base_skills: dict[str, str],
    protected_branches: list[dict[str, Any]] | None = None,
) -> tuple[set[str], set[str], bool]:
    """Computes `(touched_skills, touched_branches, clobbers_protected)` relative to incumbent."""
    touched_skills: set[str] = set()
    all_skill_names = set(skills.keys()) | set(base_skills.keys())
    for sk in all_skill_names:
        if skills.get(sk, "").strip() != base_skills.get(sk, "").strip():
            touched_skills.add(sk)

    cur_branches = _extract_branch_names(policy_code)
    base_branches = _extract_branch_names(base_policy)
    touched_branches = set(cur_branches) ^ set(base_branches)
    if policy_code.strip() != base_policy.strip() and not touched_branches:
        # Changed internal logic on existing policy branches
        touched_branches.update(cur_branches or ["step"])

    clobbers_protected = False
    if protected_branches and base_policy.strip():
        cur_set = set(cur_branches)
        for pb in protected_branches:
            fn = str(pb.get("fn", ""))
            if fn and fn in base_branches and fn not in cur_set:
                clobbers_protected = True
                break

    return touched_skills, touched_branches, clobbers_protected


def _run_replay_probe(
    compiled: Any,
    seed: int,
    action_bytes: bytes,
    stop_at_byte_idx: int | None = None,
    live_steps: int = 80,
) -> tuple[bool, int, int, str]:
    """
    Fast-forwards a candidate policy into a recorded mid/late-game trajectory byte stream
    and runs `live_steps` turns, returning `(survived, start_depth, end_depth, error_msg)`.
    """
    buf = bytes(action_bytes or b"")
    if not buf:
        return True, 1, 1, ""
    rh = NetHackHarness(max_steps=live_steps + 20)
    try:
        with _COMPILE_EXEC_LOCK:
            r_runner = compiled.create_runner()
        r_st = rh.replay_bytes(
            seed=int(seed),
            recorded_bytes=buf,
            stop_at_byte_idx=stop_at_byte_idx,
        )
        if rh.done and stop_at_byte_idx is None and len(buf) > 1:
            r_st = rh.replay_bytes(
                seed=int(seed),
                recorded_bytes=buf,
                stop_at_byte_idx=max(1, len(buf) - 30),
            )
        start_depth = int(getattr(r_st, "depth", 1) or 1)
        max_seen_depth = start_depth
        rh.step_count = 0
        for _ in range(live_steps):
            ra, _ = r_runner.next_action(r_st)
            r_st, r_done, _ = rh.step(ra)
            cur_d = int(getattr(r_st, "depth", 1) or 1)
            if cur_d > max_seen_depth:
                max_seen_depth = cur_d
            if r_done:
                break
        r_errs = extract_runner_leaf_errors(r_runner)
        if r_errs:
            return (
                False,
                start_depth,
                max_seen_depth,
                f"leaf error: {r_errs[0][0]} -> {r_errs[0][1]}",
            )
        return not rh.done, start_depth, max_seen_depth, ""
    except Exception as exc:
        return False, 1, 1, f"runtime error: {type(exc).__name__}: {exc}"
    finally:
        rh.close()


def incubate_candidate(
    candidate: PeerCandidate,
    deficit_report: DeficitReport,
    claimed_deficit: DeficitCluster,
    eval_pool: Any = None,
    incumbent_shard_runs: dict[int, dict[str, Any]] | None = None,
    shard_seeds: list[int] | None = None,
    frontier_checkpoints: list[Any] | None = None,
) -> PeerCandidate:
    """
    Isolated Sandbox Incubation (`tau_iso`):
    1. Compiles candidate `(policy_code, skills)` cleanly under `_COMPILE_EXEC_LOCK`.
    2. Smoke-tests execution, checks `WAIT`-only deadlock and `extract_runner_leaf_errors`.
    3. Runs paired micro-shard (`D_succ` guards + `D_fail` cluster seeds + baseline) when `eval_pool` is present.
    4. Checks wrapper attribution ratio and guard regressions before admitting candidate to tournament.
    """
    code = candidate.policy_code
    skills = candidate.skills
    candidate.loc_total = len(code.splitlines()) + sum(
        len(v.splitlines()) for v in skills.values()
    )

    if not code.strip():
        candidate.sandbox_passed = False
        candidate.error = "Empty policy_code produced."
        return candidate

    try:
        with _COMPILE_EXEC_LOCK:
            compiled = compile_policy_package(code, "", skills=skills)
            runner = compiled.create_runner()
    except Exception as exc:
        candidate.sandbox_passed = False
        candidate.error = f"Compile/validation error: {type(exc).__name__}: {exc}"
        return candidate

    # Fast local smoke + failure-fixture check (always runs, <50ms)
    smoke_harness = NetHackHarness(max_steps=60)
    try:
        st = smoke_harness.reset(seed=42)
        acts: list[str] = []
        fn_turns: Counter[str] = Counter()
        for _ in range(60):
            act, active_fn = runner.next_action(st)
            fn_turns[str(active_fn or "root")] += 1
            st, done, info = smoke_harness.step(act)
            acts.append(str(info.get("action_name", "")))
            if done:
                break

        leaf_errs = extract_runner_leaf_errors(runner)
        if leaf_errs:
            candidate.sandbox_passed = False
            candidate.error = (
                f"Swallowed leaf error(s): {leaf_errs[0][0]} -> {leaf_errs[0][1]}"
            )
            return candidate

        non_wait = [a for a in acts if a != "WAIT"]
        if len(acts) >= 20 and not non_wait:
            candidate.sandbox_passed = False
            candidate.error = "WAIT-only deadlock on smoke check (0 non-WAIT actions)."
            return candidate

        total_t = sum(fn_turns.values())
        wrapper_t = sum(fn_turns.get(w, 0) for w in ("root", "wrapped", "_osc_gate"))
        wrapper_ratio = float(wrapper_t / max(1, total_t))
    except Exception as exc:
        candidate.sandbox_passed = False
        candidate.error = f"Smoke runtime error: {type(exc).__name__}: {exc}"
        return candidate
    finally:
        smoke_harness.close()

    # Check failure-replay fixtures from claimed deficit
    for fix in claimed_deficit.replay_fixtures[:2]:
        r_bytes = bytes(fix.get("action_bytes") or b"")
        if not r_bytes:
            continue
        stop_idx = int(fix.get("stop_at_byte_idx", max(1, len(r_bytes) - 25)))
        _, _, _, err_msg = _run_replay_probe(
            compiled,
            seed=int(fix["seed"]),
            action_bytes=r_bytes,
            stop_at_byte_idx=stop_idx,
            live_steps=80,
        )
        if err_msg:
            candidate.sandbox_passed = False
            candidate.error = f"Failure-replay `{fix['run_id']}` {err_msg}"
            return candidate

    # Check high-depth Go-Explore FrontierCheckpoints (~15ms fast-forward each)
    frontier_survived = 0
    frontier_depth_deltas: list[int] = []
    for cp in (frontier_checkpoints or [])[:2]:
        cp_bytes = bytes(getattr(cp, "action_bytes", b"") or b"")
        if not cp_bytes:
            continue
        f_surv, start_d, max_d, f_err = _run_replay_probe(
            compiled,
            seed=int(getattr(cp, "seed", 0)),
            action_bytes=cp_bytes,
            stop_at_byte_idx=None,
            live_steps=100,
        )
        if f_err:
            cp_id = getattr(cp, "checkpoint_id", "frontier_cp")
            candidate.sandbox_passed = False
            candidate.error = f"Frontier `{cp_id}` {f_err}"
            return candidate
        if f_surv:
            frontier_survived += 1
        frontier_depth_deltas.append(max_d - start_d)

    metrics: dict[str, Any] = {
        "wrapper_ratio": round(wrapper_ratio, 4),
        "smoke_fn_turns": dict(fn_turns),
        "avg_depth": 1.0,
        "avg_score": 0.0,
        "depth_delta": 0.0,
        "score_delta": 0.0,
        "guard_depth_delta": 0.0,
        "frontier_survived": frontier_survived,
        "frontier_depth_delta": (
            round(sum(frontier_depth_deltas) / len(frontier_depth_deltas), 2)
            if frontier_depth_deltas
            else 0.0
        ),
    }

    # Multi-core paired incubation shard if eval_pool is attached
    if eval_pool is not None and shard_seeds:
        try:
            cand_eval = eval_pool.evaluate_batch(
                batch_id=f"incubate_{candidate.peer_id}_g{deficit_report.generation}_{int(time.time())}",
                policy_code=code,
                macros_code="",
                skills=skills,
                target_episodes=len(shard_seeds),
                priority_seeds=list(shard_seeds),
                record_ticks=False,
            )
            c_runs = cand_eval.get("runs", [])
            if c_runs:
                c_depths = [
                    float(max(int(r.get("depth", 1)), int(r.get("max_depth", 1))))
                    for r in c_runs
                ]
                c_scores = [float(r.get("score", 0.0)) for r in c_runs]
                c_avg_d = sum(c_depths) / len(c_depths)
                c_avg_s = sum(c_scores) / len(c_scores)
                metrics["avg_depth"] = round(c_avg_d, 4)
                metrics["avg_score"] = round(c_avg_s, 2)

                # Compare against cached incumbent shard runs
                if incumbent_shard_runs:
                    paired_d_diffs: list[float] = []
                    paired_s_diffs: list[float] = []
                    guard_d_diffs: list[float] = []
                    d_succ_set = set(deficit_report.d_succ_seeds)
                    for r in c_runs:
                        sd = int(r.get("seed", 0))
                        base_r = incumbent_shard_runs.get(sd)
                        if not base_r:
                            continue
                        cd = float(
                            max(int(r.get("depth", 1)), int(r.get("max_depth", 1)))
                        )
                        bd = float(
                            max(
                                int(base_r.get("depth", 1)),
                                int(base_r.get("max_depth", 1)),
                            )
                        )
                        cs = float(r.get("score", 0.0))
                        bs = float(base_r.get("score", 0.0))
                        paired_d_diffs.append(cd - bd)
                        paired_s_diffs.append(cs - bs)
                        if sd in d_succ_set:
                            guard_d_diffs.append(cd - bd)

                    if paired_d_diffs:
                        metrics["depth_delta"] = round(
                            sum(paired_d_diffs) / len(paired_d_diffs), 4
                        )
                    if paired_s_diffs:
                        metrics["score_delta"] = round(
                            sum(paired_s_diffs) / len(paired_s_diffs), 2
                        )
                    if guard_d_diffs:
                        g_delta = sum(guard_d_diffs) / len(guard_d_diffs)
                        metrics["guard_depth_delta"] = round(g_delta, 4)
                        # Reject if clobbering D_succ guards AND overall shard regressed
                        if g_delta < -0.5 and metrics["depth_delta"] < 0.0:
                            candidate.sandbox_metrics = metrics
                            candidate.sandbox_passed = False
                            candidate.error = (
                                f"Incubation guard regression: D_succ guard_delta={g_delta:+.2f}, "
                                f"shard depth_delta={metrics['depth_delta']:+.2f}"
                            )
                            return candidate
        except Exception as exc:
            candidate.sandbox_passed = False
            candidate.error = f"Incubation shard failed: {type(exc).__name__}: {exc}"
            return candidate

    candidate.sandbox_metrics = metrics
    candidate.sandbox_passed = True
    return candidate


class SwarmPeerAgent:
    """Symmetric full-privilege Peer Agent operating in an isolated sandbox."""

    def __init__(
        self,
        peer_id: str,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        db_path: str = "data/lox.duckdb",
        theories_store: Any = None,
        max_tool_turns: int = 8,
    ) -> None:
        self.peer_id = str(peer_id)
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = theories_store
        self.max_tool_turns = max_tool_turns
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role=self.peer_id,
        )
        if theories_store is not None:
            self.author.tools.theories_store = theories_store

    def run_cycle(
        self,
        generation: int,
        deficit_report: DeficitReport,
        claimed_deficit: DeficitCluster,
        current_policy: str = "",
        current_skills: dict[str, str] | None = None,
        eval_pool: Any = None,
        incumbent_shard_runs: dict[int, dict[str, Any]] | None = None,
        shard_seeds: list[int] | None = None,
        frontier: Any = None,
    ) -> PeerCandidate:
        skills_copy = dict(current_skills or {})
        sampled_frontier: list[Any] = []
        if frontier is not None and getattr(frontier, "checkpoints", None):
            min_f_d = max(2, int(deficit_report.incumbent_max_depth or 2) - 1)
            seen_cp_ids: set[str] = set()
            for target_d in (min_f_d, 2):
                for _ in range(2):
                    cp = frontier.sample_checkpoint(min_depth=target_d)
                    if cp is not None and cp.checkpoint_id not in seen_cp_ids:
                        seen_cp_ids.add(cp.checkpoint_id)
                        sampled_frontier.append(cp)
                if len(sampled_frontier) >= 2:
                    break
        ui.info(
            f"Peer [{self.peer_id}]: Claimed `{claimed_deficit.cluster_id}` "
            f"({claimed_deficit.label[:60]}) for Gen {generation}..."
        )

        if self.provider == "mock":
            mock_code = self._build_mock_peer_policy(current_policy)
            touched_skills, touched_branches, clobbers = compute_candidate_footprint(
                mock_code,
                skills_copy,
                current_policy,
                skills_copy,
                protected_branches=deficit_report.protected_branches,
            )
            cand = PeerCandidate(
                peer_id=self.peer_id,
                claimed_deficit_id=claimed_deficit.cluster_id,
                claimed_label=claimed_deficit.label,
                hypothesis_yaml={
                    "observation_from_db": claimed_deficit.evidence or "bootstrap",
                    "mechanism": f"Mock peer improvement targeting {claimed_deficit.cluster_id}",
                },
                policy_code=mock_code,
                skills=SwarmMetaAgent.prune_unused_skills(mock_code, skills_copy),
                touched_skills=touched_skills,
                touched_branches=touched_branches,
                clobbers_protected=clobbers,
            )
            cand = incubate_candidate(
                cand,
                deficit_report=deficit_report,
                claimed_deficit=claimed_deficit,
                eval_pool=eval_pool,
                incumbent_shard_runs=incumbent_shard_runs,
                shard_seeds=shard_seeds,
                frontier_checkpoints=sampled_frontier,
            )
            self._record_peer_theory(generation, claimed_deficit, cand)
            return cand

        # Configure isolated AuthorToolRegistry context (no live pool mutation, check-only dry_run_policy)
        self.author.tools.configure_generation(
            pool=None,
            baseline_runs=[],
            skills=skills_copy,
            generation=generation,
            mode="SWARM_PEER",
            swarm_frontier=frontier,
            swarm_checkpoint=sampled_frontier[0] if sampled_frontier else None,
        )
        self.author.tools.current_policy_code = str(current_policy or "")

        frontier_block = ""
        if sampled_frontier:
            f_lines = []
            for cp in sampled_frontier[:2]:
                tail_log = " | ".join(
                    (getattr(cp, "game_log", "") or "").splitlines()[-2:]
                )[:200]
                f_lines.append(
                    f"  * `{cp.checkpoint_id}` (seed={cp.seed}, Dlvl {cp.depth}, turns={cp.turns}): {tail_log}"
                )
            frontier_block = (
                "\n- High-Depth Frontier Checkpoints (auto-replayed in `dry_run_policy` & incubation):\n"
                + "\n".join(f_lines)
            )

        extra_context = (
            f"{deficit_report.format_brief()}\n\n"
            f"### Your Claimed Target Deficit (`{self.peer_id}`):\n"
            f"- Cluster ID: `{claimed_deficit.cluster_id}` ({claimed_deficit.kind})\n"
            f"- Description: {claimed_deficit.label}\n"
            f"- Evidence: {claimed_deficit.evidence}\n"
            f"- Target Sample Seeds: {claimed_deficit.sample_seeds}\n"
            f"- Dominant Functions Observed: {claimed_deficit.dominant_fns}"
            f"{frontier_block}\n"
        )

        proposal = self.author.synthesize(
            mode="SWARM_PEER",
            generation=generation,
            current_policy=current_policy,
            batch_digest=extra_context,
            skills=skills_copy,
            max_tool_turns=self.max_tool_turns,
        )

        if getattr(proposal, "aborted", False):
            aborted_cand = PeerCandidate(
                peer_id=self.peer_id,
                claimed_deficit_id=claimed_deficit.cluster_id,
                claimed_label=claimed_deficit.label,
                hypothesis_yaml=dict(getattr(proposal, "hypothesis", {}) or {}),
                policy_code=str(
                    getattr(proposal, "full_policy_code", "") or current_policy
                ),
                skills=skills_copy,
                sandbox_passed=False,
                error=str(getattr(proposal, "abort_reason", "") or "Synthesis aborted"),
            )
            self._record_peer_theory(generation, claimed_deficit, aborted_cand)
            return aborted_cand

        hyp = dict(getattr(proposal, "hypothesis", {}) or {})
        pol_code = str(getattr(proposal, "full_policy_code", "") or "").strip()
        if not pol_code:
            pol_code = str(getattr(proposal, "raw_code_patch", "") or "").strip()

        updated_skills = dict(skills_copy)
        for s_name, s_src in (getattr(proposal, "skill_modules", {}) or {}).items():
            clean_s = str(s_name or "").strip().removesuffix(".py")
            if clean_s and str(s_src or "").strip():
                updated_skills[clean_s] = str(s_src).strip()

        explicit_prunes: list[str] = []
        raw_prunes = hyp.get("skills_to_prune")
        if isinstance(raw_prunes, list):
            for s in raw_prunes:
                s_name = str(s or "").strip().removesuffix(".py")
                if s_name and s_name in updated_skills:
                    updated_skills.pop(s_name, None)
                    explicit_prunes.append(s_name)

        kept_skills = SwarmMetaAgent.prune_unused_skills(pol_code, updated_skills)
        auto_pruned = [k for k in updated_skills if k not in kept_skills]
        pruned = list(dict.fromkeys(explicit_prunes + auto_pruned))

        touched_skills, touched_branches, clobbers = compute_candidate_footprint(
            pol_code,
            kept_skills,
            current_policy,
            skills_copy,
            protected_branches=deficit_report.protected_branches,
        )

        cand = PeerCandidate(
            peer_id=self.peer_id,
            claimed_deficit_id=claimed_deficit.cluster_id,
            claimed_label=claimed_deficit.label,
            hypothesis_yaml=hyp,
            policy_code=pol_code,
            skills=kept_skills,
            pruned_skills=pruned,
            touched_skills=touched_skills,
            touched_branches=touched_branches,
            clobbers_protected=clobbers,
        )

        cand = incubate_candidate(
            cand,
            deficit_report=deficit_report,
            claimed_deficit=claimed_deficit,
            eval_pool=eval_pool,
            incumbent_shard_runs=incumbent_shard_runs,
            shard_seeds=shard_seeds,
            frontier_checkpoints=sampled_frontier,
        )

        status_lbl = (
            f"PASSED (depth={cand.sandbox_metrics.get('avg_depth', 1.0):.2f}, Δ={cand.sandbox_metrics.get('depth_delta', 0.0):+.2f})"
            if cand.sandbox_passed
            else f"FAILED ({cand.error[:80]})"
        )
        ui.info(f"Peer [{self.peer_id}] Incubation: {status_lbl}")
        self._record_peer_theory(generation, claimed_deficit, cand)
        return cand

    def _record_peer_theory(
        self,
        generation: int,
        claimed_deficit: DeficitCluster,
        cand: PeerCandidate,
    ) -> None:
        """Persists peer hypothesis and empirical incubation outcome into LanceDB `theories`."""
        if self.theories is None or not hasattr(self.theories, "save_document"):
            return
        hyp = dict(cand.hypothesis_yaml or {})
        obs = str(
            hyp.get("observation_from_db")
            or hyp.get("binding_constraint")
            or claimed_deficit.evidence
            or ""
        )
        mech = str(hyp.get("mechanism") or cand.claimed_label or "")
        c_rule = str(hyp.get("verified_c_rule") or "")
        outcome = "PASSED" if cand.sandbox_passed else f"FAILED: {cand.error}"
        tags = list(
            dict.fromkeys(
                [claimed_deficit.cluster_id, claimed_deficit.kind]
                + sorted(cand.touched_branches)[:4]
                + sorted(cand.touched_skills)[:4]
            )
        )
        content = (
            f"Peer {self.peer_id} | Gen {generation} | Target `{claimed_deficit.cluster_id}` "
            f"({claimed_deficit.label})\n"
            f"Observation: {obs}\n"
            f"Mechanism: {mech}\n"
            f"C/Wiki Rule: {c_rule}\n"
            f"Touched Branches: {sorted(cand.touched_branches)} | Skills: {sorted(cand.touched_skills)}\n"
            f"Incubation Outcome: {outcome}\n"
            f"Sandbox Metrics: {json.dumps(cand.sandbox_metrics or {})}"
        )
        try:
            self.theories.save_document(
                generation=generation,
                agent_role=self.peer_id,
                doc_type="THEORY",
                title=f"Gen {generation} {self.peer_id} ({claimed_deficit.cluster_id}): {outcome[:60]}",
                content=content,
                tags=tags,
                verified_c_rule=c_rule,
                metrics=dict(cand.sandbox_metrics or {}),
            )
        except Exception:
            pass

    @staticmethod
    def _build_mock_peer_policy(current_policy: str) -> str:
        if current_policy.strip() and "class Policy" in current_policy:
            return current_policy
        return (
            "from lox.env.actions import Action\n"
            "from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector\n\n"
            "def _explore_step(state, bb):\n"
            "    return Action.SEARCH\n\n"
            "class Policy:\n"
            "    def __init__(self):\n"
            "        self.blackboard = Blackboard()\n"
            "        self.tree = BehaviorTree(\n"
            "            root=PrioritySelector([\n"
            "                ActionLeaf(_explore_step, name='explore_step', min_commit=2),\n"
            "            ]),\n"
            "            blackboard=self.blackboard,\n"
            "        )\n\n"
            "    def step(self, state):\n"
            "        return self.tree.tick(state) or Action.WAIT\n"
        )
