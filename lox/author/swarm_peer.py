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
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.directives import extract_policy_branches, prune_unused_skills
from lox.author.parsing import safe_normalize_skill_key
from lox.author.swarm_deficits import DeficitCluster, DeficitReport
from lox.env.harness import NetHackHarness, policy_handles_menus
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
    touched_skill_symbols: dict[str, set[str]] = field(default_factory=dict)
    sandbox_passed: bool = False
    sandbox_metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    loc_total: int = 0
    clobbers_protected: bool = False
    track: str = "reactive"


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
    from lox.author.tools import merge_skill_module_update
    from lox.lang.bundle import (
        extract_policy_fallback_from_markdown,
        extract_skill_modules,
    )

    hyp, extracted_code = AuthorAgent.extract_yaml_and_python(raw_text or "")

    updated_skills = dict(base_skills or {})
    explicit_prunes: list[str] = []
    raw_prunes = hyp.get("skills_to_prune")
    if isinstance(raw_prunes, list):
        for s in raw_prunes:
            s_name = safe_normalize_skill_key(s)
            if s_name and s_name in updated_skills:
                updated_skills.pop(s_name, None)
                explicit_prunes.append(s_name)

    extracted_skills = extract_skill_modules(
        raw_text or "",
        merge_fn=merge_skill_module_update,
        include_markdown=True,
    )
    for k, v in extracted_skills.items():
        updated_skills[k] = (
            merge_skill_module_update(updated_skills[k], v)
            if k in updated_skills
            else v
        )

    # If <policy_code> wasn't explicit and the trailing python block was actually a skill block,
    # inspect any remaining python block containing `class Policy`.
    policy_code = extracted_code.strip()
    if "<policy_code>" not in (raw_text or "") and "class Policy" not in policy_code:
        fallback = extract_policy_fallback_from_markdown(raw_text or "")
        if fallback:
            policy_code = fallback.strip()

    if not policy_code:
        policy_code = str(base_policy or "").strip()

    # Transitive prune so unused skills aren't carried into evaluation
    kept_skills = prune_unused_skills(policy_code, updated_skills)
    auto_pruned = [k for k in updated_skills if k not in kept_skills]
    all_pruned = list(dict.fromkeys(explicit_prunes + auto_pruned))
    return hyp, policy_code, kept_skills, all_pruned


def _extract_branch_names(
    policy_code: str, skills: dict[str, str] | None = None
) -> list[str]:
    if not (policy_code and policy_code.strip()) and not skills:
        return []
    raw = extract_policy_branches(policy_code, skills)
    names: list[str] = []
    seen: set[str] = set()
    for b in raw:
        for k in ("leaf_name", "fn_name"):
            nm = str(b.get(k) or "").strip()
            if nm and nm not in seen:
                seen.add(nm)
                names.append(nm)
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

    cur_branches = _extract_branch_names(policy_code, skills)
    base_branches = _extract_branch_names(base_policy, base_skills)
    touched_branches = set(cur_branches) ^ set(base_branches)
    if policy_code.strip() != base_policy.strip() and not touched_branches:
        # Changed internal logic on existing policy branches
        touched_branches.update(cur_branches or ["step"])

    clobbers_protected = False
    if protected_branches and (base_policy.strip() or base_skills):
        cur_set = set(cur_branches)
        base_set = set(base_branches)
        for pb in protected_branches:
            fn = str(pb.get("fn", ""))
            if fn and fn in base_set and fn not in cur_set:
                clobbers_protected = True
                break

    return touched_skills, touched_branches, clobbers_protected


def compute_candidate_skill_symbols(
    skills: dict[str, str],
    base_skills: dict[str, str],
) -> dict[str, set[str]]:
    """
    Computes per-skill symbol-level diff (`{skill_name: {modified_fn_or_const_names}}`)
    relative to `base_skills` so `SwarmReferee` can compose disjoint edits inside the
    same skill module via 3-way AST merge.
    """
    import ast

    def _index_syms(src: str) -> dict[str, str]:
        if not src or not src.strip():
            return {}
        try:
            tr = ast.parse(src)
        except Exception:
            return {"__raw__": src.strip()}
        mp: dict[str, str] = {}
        for nd in tr.body:
            if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mp[nd.name] = ast.dump(nd, include_attributes=False)
            elif (
                isinstance(nd, ast.Assign)
                and len(nd.targets) == 1
                and isinstance(nd.targets[0], ast.Name)
            ):
                mp[nd.targets[0].id] = ast.dump(nd, include_attributes=False)
            elif isinstance(nd, ast.AnnAssign) and isinstance(nd.target, ast.Name):
                mp[nd.target.id] = ast.dump(nd, include_attributes=False)
        return mp

    changed: dict[str, set[str]] = {}
    for sk in set(skills.keys()) | set(base_skills.keys()):
        s_new = skills.get(sk, "").strip()
        s_old = base_skills.get(sk, "").strip()
        if s_new == s_old:
            continue
        m_new = _index_syms(s_new)
        m_old = _index_syms(s_old)
        diff_syms: set[str] = set()
        for nm in set(m_new.keys()) | set(m_old.keys()):
            if m_new.get(nm) != m_old.get(nm):
                diff_syms.add(nm)
        if not diff_syms:
            diff_syms.add("__module__")
        changed[sk] = diff_syms
    return changed


def _run_replay_probe(
    compiled: Any,
    seed: int,
    action_bytes: bytes,
    stop_at_byte_idx: int | None = None,
    live_steps: int = 80,
    harness: Any = None,
) -> tuple[bool, int, int, str]:
    """
    Fast-forwards a candidate policy into a recorded mid/late-game trajectory byte stream
    and runs `live_steps` turns, returning `(survived, start_depth, end_depth, error_msg)`.
    """
    buf = bytes(action_bytes or b"")
    if not buf:
        return True, 1, 1, ""
    own_harness = harness is None
    rh = harness if harness is not None else NetHackHarness(max_steps=live_steps + 20)
    rh.max_steps = live_steps + 20
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
        return not rh.done, start_depth, max_seen_depth, ""
    except Exception as exc:
        return False, 1, 1, f"runtime error: {type(exc).__name__}: {exc}"
    finally:
        if own_harness:
            rh.close()


def incubate_candidate(
    candidate: PeerCandidate,
    deficit_report: DeficitReport,
    claimed_deficit: DeficitCluster,
    eval_pool: Any = None,
    incumbent_shard_runs: dict[int, dict[str, Any]] | None = None,
    shard_seeds: list[int] | None = None,
    frontier_checkpoints: list[Any] | None = None,
    probe_seeds: list[int] | None = None,
    db_path: str = "",
) -> PeerCandidate:
    """
    Isolated Sandbox Incubation (`tau_iso`):
    1. Compiles candidate `(policy_code, skills)` cleanly under `_COMPILE_EXEC_LOCK`.
    2. Smoke-tests execution and checks `WAIT`-only deadlock.
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
            _ = compiled.create_runner()
    except Exception as exc:
        candidate.sandbox_passed = False
        candidate.error = f"Compile/validation error: {type(exc).__name__}: {exc}"
        return candidate

    # Benchmark-agnostic smoke check + failure-fixture + frontier probes
    bm_name = str(getattr(eval_pool, "benchmark", "nethack") or "nethack")
    from lox.env import get_adapter

    adapter = get_adapter(bm_name)
    smoke_err, smoke_m = adapter.smoke_check(
        compiled,
        seed=42,
        max_steps=60,
        policy_code=code,
        skills=skills,
    )
    if smoke_err is not None:
        candidate.sandbox_passed = False
        candidate.error = smoke_err
        return candidate
    wrapper_ratio = float(smoke_m.get("wrapper_ratio", 0.0))
    fn_turns = Counter(smoke_m.get("fn_turns") or {})

    frontier_survived = 0
    frontier_depth_deltas: list[int] = []
    has_probes = bool(
        any(
            bytes(fix.get("action_bytes") or b"")
            for fix in claimed_deficit.replay_fixtures[:2]
        )
        or any(
            bytes(getattr(cp, "action_bytes", b"") or b"")
            for cp in (frontier_checkpoints or [])[:2]
        )
    )
    if has_probes:
        h_kwargs: dict[str, Any] = {"max_steps": 100}
        if bm_name == "nethack":
            h_kwargs["allow_menu_linger"] = policy_handles_menus(code, skills)
        probe_harness = adapter.create_harness(**h_kwargs)
        try:
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
                    harness=probe_harness,
                )
                if err_msg:
                    candidate.sandbox_passed = False
                    candidate.error = f"Failure-replay `{fix['run_id']}` {err_msg}"
                    return candidate

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
                    harness=probe_harness,
                )
                if f_err:
                    cp_id = getattr(cp, "checkpoint_id", "frontier_cp")
                    candidate.sandbox_passed = False
                    candidate.error = f"Frontier `{cp_id}` {f_err}"
                    return candidate
                if f_surv:
                    frontier_survived += 1
                frontier_depth_deltas.append(max_d - start_d)
        finally:
            probe_harness.close()

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

    # Multi-core paired incubation shard + organic mined scenario stress tests
    organic_shards = list(deficit_report.__dict__.get("organic_shards", []) or [])
    if eval_pool is not None and (shard_seeds or organic_shards):
        try:
            target_ep_cnt = len(shard_seeds or []) + len(organic_shards)
            inc_batch_id = f"incubate_{candidate.peer_id}_g{deficit_report.generation}_{int(time.time())}"
            cand_eval = eval_pool.evaluate_batch(
                batch_id=inc_batch_id,
                policy_code=code,
                macros_code="",
                skills=skills,
                target_episodes=max(1, target_ep_cnt),
                priority_seeds=list(shard_seeds) if shard_seeds else None,
                shards=organic_shards if organic_shards else None,
                record_ticks=True,
            )
            c_runs = cand_eval.get("runs", [])
            if db_path and c_runs:
                try:
                    from lox.db.logger import BatchDatabaseLogger

                    BatchDatabaseLogger(db_path).save_runs_and_ticks(
                        batch_id=inc_batch_id,
                        generation=int(deficit_report.generation),
                        run_kind="incubate",
                        peer_id=str(candidate.peer_id),
                        runs=c_runs,
                        run_id_prefix=f"g{deficit_report.generation}_{candidate.peer_id}_inc",
                    )
                except Exception as log_exc:
                    ui.warn(
                        f"Peer `{candidate.peer_id}`: failed persisting incubation runs ({log_exc})"
                    )
            # Hard exception gate across incubation episodes & organic scenarios
            exc_total = sum(int(r.get("policy_exceptions", 0) or 0) for r in c_runs)
            if exc_total > 0:
                exc_tbs: list[str] = []
                for r in c_runs:
                    if int(r.get("policy_exceptions", 0) or 0) > 0:
                        mj = r.get("milestones_json")
                        if isinstance(mj, str) and mj:
                            try:
                                mo = json.loads(mj)
                                exc_tbs.extend(
                                    mo.get("exception_tracebacks")
                                    or mo.get("leaf_errors")
                                    or []
                                )
                            except Exception:
                                pass
                        elif isinstance(r.get("milestones"), dict):
                            mo = r["milestones"]
                            exc_tbs.extend(
                                mo.get("exception_tracebacks")
                                or mo.get("leaf_errors")
                                or []
                            )
                        er = str(r.get("end_reason", ""))
                        dm = str(r.get("death_message", ""))
                        if er or dm:
                            exc_tbs.append(f"{er}: {dm}"[:140])
                candidate.sandbox_passed = False
                candidate.error = f"Incubation runtime exceptions ({exc_total}x): " + (
                    "; ".join(list(dict.fromkeys(exc_tbs))[:2]) or "policy exception"
                )
                return candidate
            if c_runs:
                stat_runs = [r for r in c_runs if not r.get("is_shard")] or c_runs
                c_depths = [
                    float(max(int(r.get("depth", 1)), int(r.get("max_depth", 1))))
                    for r in stat_runs
                ]
                c_scores = [float(r.get("score", 0.0)) for r in stat_runs]
                c_avg_d = sum(c_depths) / len(c_depths)
                c_avg_s = sum(c_scores) / len(c_scores)
                metrics["avg_depth"] = round(c_avg_d, 4)
                metrics["max_depth"] = max((int(d) for d in c_depths), default=1)
                metrics["avg_score"] = round(c_avg_s, 2)

                # Compare against cached incumbent shard runs
                if incumbent_shard_runs:
                    paired_d_diffs: list[float] = []
                    paired_s_diffs: list[float] = []
                    guard_d_diffs: list[float] = []
                    d_succ_set = set(deficit_report.d_succ_seeds)
                    target_seed_set = {
                        int(s)
                        for s in (getattr(claimed_deficit, "sample_seeds", []) or [])
                    }
                    probe_seed_set = {int(s) for s in (probe_seeds or [])}
                    target_bytes_checked = 0
                    total_bytes_checked = 0
                    total_bytes_diverged = 0
                    probe_diverged = 0
                    for r in c_runs:
                        if r.get("is_shard"):
                            continue
                        sd = int(r.get("seed", 0))
                        base_r = incumbent_shard_runs.get(sd)
                        if not base_r:
                            continue
                        c_ab = bytes(r.get("action_bytes") or b"")
                        b_ab = bytes(base_r.get("action_bytes") or b"")
                        if c_ab and b_ab:
                            total_bytes_checked += 1
                            if sd in target_seed_set:
                                target_bytes_checked += 1
                            if c_ab != b_ab:
                                total_bytes_diverged += 1
                                if sd in probe_seed_set:
                                    probe_diverged += 1
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

                    if (
                        target_bytes_checked >= 2
                        and total_bytes_checked >= 2
                        and total_bytes_diverged == 0
                    ):
                        metrics["zero_shard_divergence"] = True
                        candidate.sandbox_metrics = metrics
                        candidate.sandbox_passed = False
                        candidate.error = (
                            f"Zero behavioral divergence across all {total_bytes_checked} evaluated "
                            f"guard/target shard seeds ({target_bytes_checked} target seeds unchanged; dead patch)"
                        )
                        return candidate

                    metrics["probe_diverged"] = probe_diverged
                    if paired_d_diffs:
                        avg_d_delta = sum(paired_d_diffs) / len(paired_d_diffs)
                        metrics["depth_delta"] = round(avg_d_delta, 4)
                    else:
                        avg_d_delta = 0.0
                    if paired_s_diffs:
                        avg_s_delta = sum(paired_s_diffs) / len(paired_s_diffs)
                        metrics["score_delta"] = round(avg_s_delta, 2)
                    else:
                        avg_s_delta = 0.0
                    if guard_d_diffs:
                        g_delta = sum(guard_d_diffs) / len(guard_d_diffs)
                        metrics["guard_depth_delta"] = round(g_delta, 4)

                    # Catastrophe-only depth regression guard (avoid noisy -0.15 vetoes on 12-14 seeds)
                    reg_thresh = -0.50 if len(paired_d_diffs) >= 6 else -0.75
                    if len(paired_d_diffs) >= 4 and avg_d_delta <= reg_thresh:
                        candidate.sandbox_metrics = metrics
                        candidate.sandbox_passed = False
                        candidate.error = (
                            f"Incubation paired depth regression across {len(paired_d_diffs)} shards "
                            f"(depth_delta={avg_d_delta:+.2f}, score_delta={avg_s_delta:+.1f})"
                        )
                        return candidate

                    # Per-step latency hygiene guard (catches O(21*79) Python per-step loops early)
                    cand_ms_tot = sum(
                        float(r.get("wall_ms", 0.0) or 0.0) for r in c_runs
                    )
                    cand_steps_tot = sum(int(r.get("steps", 0) or 0) for r in c_runs)
                    inc_ms_tot = sum(
                        float(r.get("wall_ms", 0.0) or 0.0)
                        for r in incumbent_shard_runs.values()
                    )
                    inc_steps_tot = sum(
                        int(r.get("steps", 0) or 0)
                        for r in incumbent_shard_runs.values()
                    )
                    if cand_steps_tot >= 200 and cand_ms_tot > 0.0:
                        cand_ms_per_step = cand_ms_tot / float(cand_steps_tot)
                        inc_ms_per_step = (
                            inc_ms_tot / float(inc_steps_tot)
                            if inc_steps_tot >= 200 and inc_ms_tot > 0.0
                            else 2.0
                        )
                        metrics["ms_per_step"] = round(cand_ms_per_step, 3)
                        if cand_ms_per_step > max(
                            15.0, 3.0 * max(0.5, inc_ms_per_step)
                        ):
                            candidate.sandbox_metrics = metrics
                            candidate.sandbox_passed = False
                            candidate.error = (
                                f"Incubation latency guard: {cand_ms_per_step:.2f} ms/step "
                                f"(> 3x incumbent {inc_ms_per_step:.2f} ms/step and > 15 ms/step; "
                                "use vectorized NumPy masks like `state.passable_mask` / `state.trap_mask`)"
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
        reasoning_effort: str = "medium",
        routing: str = "nitro",
        scenario_gate: str = "off",
    ) -> None:
        self.peer_id = str(peer_id)
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = theories_store
        self.max_tool_turns = max_tool_turns
        self.reasoning_effort = reasoning_effort
        self.routing = routing
        self.scenario_gate = str(scenario_gate or "off").strip().lower()
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role=self.peer_id,
            reasoning_effort=reasoning_effort,
            routing=routing,
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
        probe_seeds: list[int] | None = None,
        directive: dict[str, Any] | None = None,
        stepping_stone: dict[str, Any] | None = None,
        incumbent_policy: str = "",
        incumbent_skills: dict[str, str] | None = None,
        track: str = "reactive",
        proactive_branch: dict[str, Any] | None = None,
    ) -> PeerCandidate:
        peer_track = str(track or "reactive").strip().lower()
        skills_copy = dict(current_skills or {})
        base_inc_policy = str(incumbent_policy or current_policy or "")
        base_inc_skills = dict(
            incumbent_skills if incumbent_skills is not None else skills_copy
        )
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
            if not current_policy.strip() and not skills_copy:
                from lox.lang.starter import build_clean_option_a_starter

                mock_code, skills_copy = build_clean_option_a_starter()
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
                skills=prune_unused_skills(mock_code, skills_copy),
                touched_skills=touched_skills,
                touched_branches=touched_branches,
                touched_skill_symbols=compute_candidate_skill_symbols(
                    skills_copy, skills_copy
                ),
                clobbers_protected=clobbers,
                track=peer_track,
            )
            cand = incubate_candidate(
                cand,
                deficit_report=deficit_report,
                claimed_deficit=claimed_deficit,
                eval_pool=eval_pool,
                incumbent_shard_runs=incumbent_shard_runs,
                shard_seeds=shard_seeds,
                frontier_checkpoints=sampled_frontier,
                probe_seeds=probe_seeds,
                db_path=self.db_path,
            )
            self._record_peer_theory(generation, claimed_deficit, cand)
            return cand

        # Configure isolated AuthorToolRegistry context with paired CRN micro-pilot seeds
        self.author.tools.configure_generation(
            pool=eval_pool,
            baseline_runs=list((incumbent_shard_runs or {}).values()),
            skills=skills_copy,
            generation=generation,
            mode="SWARM_PEER",
            swarm_frontier=frontier,
            swarm_checkpoint=sampled_frontier[0] if sampled_frontier else None,
            claimed_deficit=claimed_deficit,
            scenario_gate=getattr(self, "scenario_gate", "off"),
            peer_guard_seeds=list(deficit_report.d_succ_seeds[:2]),
            peer_target_seeds=list(claimed_deficit.sample_seeds[:2]),
            peer_probe_seeds=list((probe_seeds or [])[:6]),
            max_peer_micro_pilots=4,
            peer_id=self.peer_id,
        )
        self.author.tools.enforce_peer_research_gate = self.provider != "mock"
        self.author.tools.current_policy_code = str(current_policy or "")
        self.author.tools.baseline_policy_code = str(
            base_inc_policy or current_policy or ""
        )
        self.author.tools.baseline_skills = dict(base_inc_skills)

        arch_block = ""
        try:
            arch_txt, _ = self.author.tools.architecture_map()
            if arch_txt and arch_txt.strip():
                if len(arch_txt) > 2500:
                    arch_txt = (
                        arch_txt[:2500]
                        + "\n... [truncated; call `architecture_map()` for full listing]"
                    )
                arch_block = f"\n\n{arch_txt}"
        except Exception:
            arch_block = ""

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

        plateau_directive = ""
        if str(deficit_report.regime or "").upper() == "PLATEAU":
            streak_n = int(getattr(deficit_report, "plateau_streak", 0) or 0)
            streak_txt = f" (streak={streak_n} gens)" if streak_n > 0 else ""
            plateau_directive = (
                f"\n### Plateau Regime Directive{streak_txt}:\n"
                "- Progression has plateaued; avoid narrow single-seed band-aids that trigger 0W/0L on unseen seeds.\n"
                "- Inspect Floor Residence & Action Concentration Telemetry and `champion_ticks` to target upstream "
                "efficiency, loop-termination, or capability bottlenecks across the owning subsystem.\n"
                "- Non-regressing drift (`OK`) on `GUARD` seeds is expected and healthy for systemic improvements.\n"
            )

        directive_block = ""
        if isinstance(directive, dict) and directive:
            focus_sub = str(directive.get("focus_subsystem") or "").strip()
            inv_q = str(directive.get("investigative_question") or "").strip()
            excl = str(directive.get("exclude_overlap") or "").strip()
            directive_block = (
                f"\n### Orthogonal Peer Directive (`{self.peer_id}`):\n"
                + (f"- Focus Subsystem: `{focus_sub}`\n" if focus_sub else "")
                + (f"- Investigative Question: {inv_q}\n" if inv_q else "")
                + (f"- Non-Overlap Boundary: {excl}\n" if excl else "")
            )

        revision_block = ""
        if isinstance(stepping_stone, dict) and stepping_stone:
            st_gen = int(stepping_stone.get("generation", 0) or 0)
            st_pid = str(stepping_stone.get("source_peer_id", "") or "")
            st_kind = str(stepping_stone.get("source_run_kind", "select") or "select")
            st_gained = list(stepping_stone.get("seeds_gained") or [])
            st_lost = list(stepping_stone.get("seeds_lost") or [])
            st_rej = str(stepping_stone.get("reject_reason", "") or "")
            st_rev = int(stepping_stone.get("revision_count", 0) or 0) + 1
            revision_block = (
                f"\n### Track A — Revision Track (`{self.peer_id}`, Revision Attempt #{st_rev}):\n"
                f"- Starting Codebase: You are starting DIRECTLY from high-signal unpromoted stepping stone "
                f"`{st_pid}` (Gen {st_gen}, `run_kind='{st_kind}'`, W/L={stepping_stone.get('wins', 0)}/{stepping_stone.get('losses', 0)}, "
                f"winsorΔ={float(stepping_stone.get('winsor_delta', 0.0)):+.3f}, iqmΔ={float(stepping_stone.get('iqm_delta', 0.0)):+.3f}, "
                f"reject_reason=`{st_rej}`).\n"
                f"- Seeds Gained vs Incumbent: {st_gained[:10]}\n"
                f"- Seeds Lost vs Incumbent: {st_lost[:10]}\n"
                f"- Unbiased Diagnosis Query: Compare the stepping stone's ticks vs incumbent ticks directly in `all_ticks` / `peer_ticks` via `query_db` "
                f"(`WHERE generation = {st_gen} AND peer_id = '{st_pid}' AND run_kind = '{st_kind}' AND seed IN (...)`) to discover why `seeds_lost` "
                f"(or `{st_rej}`) regressed, then refine the stepping stone while preserving `seeds_gained`.\n"
            )

        proactive_block = ""
        expert_gaps_txt = str(
            getattr(deficit_report, "expert_parity_gaps", "") or ""
        ).strip()
        if peer_track == "proactive_capability":
            proactive_block = (
                f"\n### Proactive Capability Track (`{self.peer_id}` — Expert-Driven & Future-Oriented):\n"
                "- **Mission**: Unlike reactive peers fixing recent death clusters, YOU start from the current champion "
                "and mine `expert_ticks` / `expert_runs` + the NetHack Wiki across `Dlvl 1 .. p90+2` to design ONE clean, "
                "self-contained missing capability or proactive efficiency upgrade (as a modular 2-level skill under `skills/<domain>/<module>.py`).\n"
                "- **Empirical Grounding**: Query `expert_ticks` directly via `query_db` (e.g., actions, item/consumable usage, "
                "floor residence, or branch behavior at your target depths) and verify exact C rules with `query_wiki` / `query_item` / `query_monster`.\n"
                "- **Triggered Confirmation Ready**: Your candidate competes directly in the tournament and benefits from "
                "Triggered Divergent-Seed Confirmation (`max N=128`) — so clean capabilities that trigger selectively without regressing early floors can promote cleanly!\n"
                + (f"\n{expert_gaps_txt}\n" if expert_gaps_txt else "")
            )
        elif peer_track == "proactive_branch":
            br = proactive_branch if isinstance(proactive_branch, dict) else {}
            br_id = str(br.get("branch_id", "") or f"proactive_g{generation}")
            br_step = int(br.get("steps_taken", 0) or 0) + 1
            br_max = int(br.get("max_steps", 4) or 4)
            br_goal = str(br.get("goal_summary", "") or "").strip()
            br_caps = list(br.get("capabilities_added") or [])
            br_prev_d = float(br.get("last_avg_depth", 0.0) or 0.0)
            br_prev_wd = float(br.get("last_winsor_delta", 0.0) or 0.0)
            br_state_line = (
                f"- **Continuing Branch `{br_id}` (Step {br_step}/{br_max})**: "
                f"goal=`{br_goal}`, capabilities_added={br_caps}, last S_select avg_depth={br_prev_d:.2f} (winsorΔ={br_prev_wd:+.3f}).\n"
                if br
                else f"- **New Sheltered Proactive Branch `{br_id}` (Step 1/{br_max})**: "
                "Starting from current champion; establish a clean multi-step architectural foundation.\n"
            )
            proactive_block = (
                f"\n### Sheltered Proactive Branch Track (`{self.peer_id}` — Multi-Step Architecture Horizon):\n"
                f"{br_state_line}"
                "- **Two-Tier Gate Advantage**: Your branch advances safely on a non-catastrophic branch-internal `S_select` gate "
                "(0 exceptions, non-catastrophic depth) without polluting the Champion, and **graduates to Champion** as soon as it "
                f"beats the Champion on strict `S_confirm` (resets to Champion after {br_max} steps if ungraduated).\n"
                "- **Mission**: Mine `expert_ticks` + Wiki + codebase architecture (`architecture_map`) to build multi-step proactive "
                "capabilities, structural efficiency upgrades (e.g., vectorized `state.trap_mask`, cleaner goal planning, inventory/resource/branch prep) "
                "that single-step reactive patches cannot reach in one generation.\n"
                + (f"\n{expert_gaps_txt}\n" if expert_gaps_txt else "")
            )

        situational_expert_note = (
            "\n- Optional Situational Expert Check: When inspecting a specific encounter, floor, or stuck state in `champion_ticks`, "
            "you may optionally check how human ascensions acted in `expert_ticks` (`WHERE depth = <D>` or `message LIKE '%...%'` / `screen_ascii LIKE '%...%'`)."
            if peer_track in ("reactive", "revision")
            else ""
        )

        extra_context = (
            f"{deficit_report.format_brief()}{arch_block}{plateau_directive}{directive_block}{revision_block}{proactive_block}\n\n"
            f"### Your Claimed Target Scope (`{self.peer_id}`, track=`{peer_track}`):\n"
            f"- Cluster ID: `{claimed_deficit.cluster_id}` ({claimed_deficit.kind})\n"
            f"- Description: {claimed_deficit.label}\n"
            f"- Evidence: {claimed_deficit.evidence}\n"
            f"- Target Sample Seeds: {claimed_deficit.sample_seeds}\n"
            f"- Guard Seeds (preserve non-regression; do NOT over-tighten on neutral drift): {list(deficit_report.d_succ_seeds[:2])}\n"
            f"- Unseen Probe Seeds (generalization check in `dry_run_policy`): {list((probe_seeds or [])[:6])}\n"
            f"- Diagnosis Note: Inspect `champion_ticks`, `peer_ticks`, `all_ticks`, and workspace `skills/` directly via `query_db` to locate the true root cause across module boundaries."
            f"{situational_expert_note}"
            f"{frontier_block}\n"
        )

        try:
            proposal = self.author.synthesize(
                mode="SWARM_PEER",
                generation=generation,
                current_policy=current_policy,
                batch_digest=extra_context,
                skills=skills_copy,
                max_tool_turns=self.max_tool_turns,
            )
        except RuntimeError as exc:
            failed_cand = PeerCandidate(
                peer_id=self.peer_id,
                claimed_deficit_id=claimed_deficit.cluster_id,
                claimed_label=claimed_deficit.label,
                hypothesis_yaml={},
                policy_code=current_policy,
                skills=skills_copy,
                sandbox_passed=False,
                error=str(exc),
                track=peer_track,
            )
            self._record_peer_theory(generation, claimed_deficit, failed_cand)
            return failed_cand
        finally:
            self.author.tools.close()

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
                track=peer_track,
            )
            self._record_peer_theory(generation, claimed_deficit, aborted_cand)
            return aborted_cand

        hyp = dict(getattr(proposal, "hypothesis", {}) or {})
        pol_code = str(getattr(proposal, "full_policy_code", "") or "").strip()
        if not pol_code:
            pol_code = str(getattr(proposal, "raw_code_patch", "") or "").strip()

        updated_skills = dict(skills_copy)
        for s_name, s_src in (getattr(proposal, "skill_modules", {}) or {}).items():
            clean_s = safe_normalize_skill_key(s_name)
            if clean_s and str(s_src or "").strip():
                # `AuthorAgent.synthesize` already ran `merge_skill_module_update` on edited skills.
                updated_skills[clean_s] = str(s_src)

        explicit_prunes: list[str] = []
        raw_prunes = hyp.get("skills_to_prune")
        if isinstance(raw_prunes, list):
            for s in raw_prunes:
                s_name = safe_normalize_skill_key(s)
                if s_name and s_name in updated_skills:
                    updated_skills.pop(s_name, None)
                    explicit_prunes.append(s_name)

        kept_skills = prune_unused_skills(pol_code, updated_skills)
        auto_pruned = [k for k in updated_skills if k not in kept_skills]
        pruned = list(dict.fromkeys(explicit_prunes + auto_pruned))

        # Soft micro-pilot submission hygiene: only applies when the final submitted
        # package matches the last micro-piloted draft (never clobbering a peer that
        # edited/fixed its code in the final turn after its last dry_run_policy call).
        last_mp = getattr(self.author.tools, "last_peer_micro_pilot", None)
        best_mp = getattr(self.author.tools, "best_peer_micro_pilot", None)
        effective_mp: dict[str, Any] | None = None
        if isinstance(last_mp, dict):
            last_code = str(last_mp.get("code", "") or "").strip()
            last_skills_pruned = prune_unused_skills(
                last_code, dict(last_mp.get("skills") or {})
            )
            final_matches_last_mp = pol_code.strip() == last_code and {
                k: v.strip() for k, v in kept_skills.items()
            } == {k: v.strip() for k, v in last_skills_pruned.items()}
            if final_matches_last_mp:
                effective_mp = last_mp
                last_div = int(last_mp.get("diverged", 0) or 0)
                last_dd = float(last_mp.get("avg_depth_delta", 0.0) or 0.0)
                last_w = int(last_mp.get("wins", 0) or 0)
                last_exc = int(last_mp.get("policy_exceptions", 0) or 0)
                last_bad = (
                    last_div == 0 or last_exc > 0 or (last_dd <= -0.75 and last_w == 0)
                )
                if last_bad and isinstance(best_mp, dict) and best_mp is not last_mp:
                    best_div = int(best_mp.get("diverged", 0) or 0)
                    best_dd = float(best_mp.get("avg_depth_delta", -999.0) or -999.0)
                    best_exc = int(best_mp.get("policy_exceptions", 0) or 0)
                    best_code = str(best_mp.get("code", "") or "").strip()
                    strictly_better = last_exc > 0 or last_div == 0 or best_dd > last_dd
                    if best_div > 0 and best_exc == 0 and best_code and strictly_better:
                        pol_code = best_code
                        rec_skills = dict(best_mp.get("skills") or {})
                        kept_skills = prune_unused_skills(pol_code, rec_skills)
                        effective_mp = best_mp
                        ui.info(
                            f"Peer [{self.peer_id}]: Auto-recovered earlier best micro-pilot "
                            f"(attempt #{best_mp.get('attempt', '?')}, Δdepth={best_dd:+.2f}, "
                            f"W/L={best_mp.get('wins', 0)}/{best_mp.get('losses', 0)}) over degraded final draft."
                        )

        if isinstance(effective_mp, dict):
            eff_div = int(effective_mp.get("diverged", 0) or 0)
            eff_dd = float(effective_mp.get("avg_depth_delta", 0.0) or 0.0)
            eff_w = int(effective_mp.get("wins", 0) or 0)
            if eff_div == 0:
                dead_cand = PeerCandidate(
                    peer_id=self.peer_id,
                    claimed_deficit_id=claimed_deficit.cluster_id,
                    claimed_label=claimed_deficit.label,
                    hypothesis_yaml=hyp,
                    policy_code=pol_code,
                    skills=kept_skills,
                    pruned_skills=pruned,
                    sandbox_passed=False,
                    error="Micro-pilot hygiene: 0-byte behavioral divergence across all micro-pilot seeds (dead patch).",
                    track=peer_track,
                )
                self._record_peer_theory(generation, claimed_deficit, dead_cand)
                return dead_cand
            if peer_track != "proactive_branch" and eff_dd <= -0.75 and eff_w == 0:
                cat_cand = PeerCandidate(
                    peer_id=self.peer_id,
                    claimed_deficit_id=claimed_deficit.cluster_id,
                    claimed_label=claimed_deficit.label,
                    hypothesis_yaml=hyp,
                    policy_code=pol_code,
                    skills=kept_skills,
                    pruned_skills=pruned,
                    sandbox_passed=False,
                    error=(
                        f"Micro-pilot hygiene: catastrophic regression (avg_Δdepth={eff_dd:+.2f}, "
                        f"wins=0, losses={effective_mp.get('losses', 0)})."
                    ),
                    track=peer_track,
                )
                self._record_peer_theory(generation, claimed_deficit, cat_cand)
                return cat_cand

        touched_skills, touched_branches, clobbers = compute_candidate_footprint(
            pol_code,
            kept_skills,
            base_inc_policy,
            base_inc_skills,
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
            touched_skill_symbols=compute_candidate_skill_symbols(
                kept_skills, base_inc_skills
            ),
            clobbers_protected=clobbers,
            track=peer_track,
        )

        cand = incubate_candidate(
            cand,
            deficit_report=deficit_report,
            claimed_deficit=claimed_deficit,
            eval_pool=eval_pool,
            incumbent_shard_runs=incumbent_shard_runs,
            shard_seeds=shard_seeds,
            frontier_checkpoints=sampled_frontier,
            probe_seeds=probe_seeds,
            db_path=self.db_path,
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
        """Persists peer hypothesis and empirical incubation outcome into DuckDB `theories`."""
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
        except Exception as th_exc:
            ui.warn(
                f"Peer `{self.peer_id}` Gen {generation}: failed saving theory note ({th_exc})"
            )

    @staticmethod
    def _build_mock_peer_policy(current_policy: str) -> str:
        if current_policy.strip() and "class Policy" in current_policy:
            return current_policy
        from lox.lang.starter import build_clean_option_a_starter

        pol, _ = build_clean_option_a_starter()
        return pol
