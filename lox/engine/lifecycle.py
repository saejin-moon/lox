"""
Lifecycle managers for Reactive Stepping Stones (`reactive_1`),
Sheltered Proactive Branches (`proactive_2`), Near-Miss Semantic Crossover buffer,
and Layout A (`2R+2P`) peer assignment.
"""

from __future__ import annotations

import copy as _copy
import difflib as _difflib
from typing import Any

from lox.author.swarm_deficits import DeficitCluster
from lox.author.swarm_peer import PeerCandidate
from lox.engine.referee_gates import TournamentVerdict, opt_float, opt_int
from lox.ui.console import ui


def update_near_misses(
    near_misses: list[dict[str, Any]],
    *,
    generation: int,
    verdict: TournamentVerdict,
    peer_candidates: list[PeerCandidate],
    current_policy: str,
    initial_skills: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Clears stale near-misses immediately upon champion promotion (`M-9`);
    otherwise records positive, net-winning, guard-passing near-misses for Semantic Crossover.
    """
    if verdict.promoted:
        near_misses.clear()
        return near_misses

    cand_by_id = {c.peer_id: c for c in peer_candidates}
    for att in verdict.attempts_summary or []:
        pid = str(att.get("peer_id", ""))
        c_obj = cand_by_id.get(pid)
        if c_obj is None or opt_int(att, "policy_exceptions", 0) > 0:
            continue
        rej_r = str(att.get("reject_reason", "") or "")
        if rej_r in (
            "guard_failed",
            "policy_exceptions_regression",
            "auto_esc_regression",
            "horizon_waste_regression",
            "cvar25_tail_regression",
        ):
            continue
        att_d = opt_float(att, "avg_depth", 1.0)
        att_md = opt_int(att, "max_depth", 1)
        att_s = opt_float(att, "avg_score", 0.0)
        att_ac = opt_float(att, "avg_ac", 10.0)
        att_w = opt_int(att, "wins", 0)
        att_l = opt_int(att, "losses", 0)
        att_iqm = opt_float(att, "iqm_delta", 0.0)
        att_win = opt_float(att, "winsor_delta", att_iqm)
        gained = list(att.get("seeds_gained") or [])
        lost = list(att.get("seeds_lost") or [])
        if (
            gained
            and (att_w >= 3 or (att_w + att_l >= 5 and att_w > att_l))
            and att_w > att_l
            and (att_win > 0.0 or att_iqm > 0.0)
            and att_d >= verdict.incumbent_avg_depth
        ):
            diff_chunks: list[str] = []
            cand_sks = dict(c_obj.skills or {})
            for sk_k in sorted(set(initial_skills) | set(cand_sks)):
                old_s = str(initial_skills.get(sk_k, "") or "")
                new_s = str(cand_sks.get(sk_k, "") or "")
                if old_s != new_s:
                    u_diff = list(
                        _difflib.unified_diff(
                            old_s.splitlines(),
                            new_s.splitlines(),
                            fromfile=f"inc/{sk_k}.py",
                            tofile=f"{pid}/{sk_k}.py",
                            lineterm="",
                            n=1,
                        )
                    )
                    if u_diff:
                        diff_chunks.append("\n".join(u_diff[:30]))
            if not diff_chunks and c_obj.policy_code != current_policy:
                u_pol = list(
                    _difflib.unified_diff(
                        current_policy.splitlines(),
                        c_obj.policy_code.splitlines(),
                        fromfile="inc/policy.py",
                        tofile=f"{pid}/policy.py",
                        lineterm="",
                        n=1,
                    )
                )
                if u_pol:
                    diff_chunks.append("\n".join(u_pol[:30]))
            diff_preview = "\n".join(diff_chunks)[:1200]
            hyp = getattr(c_obj, "hypothesis_yaml", {}) or {}
            near_misses.append(
                {
                    "peer_id": pid,
                    "generation": generation,
                    "avg_depth": att_d,
                    "max_depth": att_md,
                    "avg_score": att_s,
                    "avg_ac": att_ac,
                    "wins": att_w,
                    "losses": att_l,
                    "iqm_delta": att_iqm,
                    "seeds_gained": gained,
                    "seeds_lost": lost,
                    "touched_skills": set(
                        getattr(c_obj, "touched_skills", set()) or set()
                    ),
                    "mechanism": str(hyp.get("mechanism", "") or c_obj.claimed_label),
                    "diff_preview": diff_preview,
                    "policy_code": c_obj.policy_code,
                    "skills": cand_sks,
                }
            )
    near_misses.sort(
        key=lambda nm: (
            opt_float(nm, "iqm_delta", 0.0),
            opt_float(nm, "avg_depth", 0.0),
            opt_float(nm, "avg_score", 0.0),
        ),
        reverse=True,
    )
    del near_misses[4:]
    return near_misses


def update_stepping_stone_lifecycle(
    *,
    db_logger: Any,
    generation: int,
    cur_fp: str,
    active_stone: dict[str, Any] | None,
    verdict: TournamentVerdict,
    max_revisions: int = 2,
) -> None:
    """
    Updates `stepping_stones` lifecycle for Reactive Revision (`reactive_1`).
    Fix M-8: When `reactive_1` refines `active_stone` and produces another qualifying
    near-miss, carry forward `revision_count = active_stone['revision_count'] + 1`
    and refuse to re-save once `>= max_revisions` so stones retire cleanly without
    infinite respawn loops.
    """
    if verdict.promoted:
        if active_stone is not None and (
            "reactive_1" in str(verdict.winner_id)
            or "peer_rev" in str(verdict.winner_id)
        ):
            db_logger.update_stepping_stone_status(active_stone["stone_id"], "promoted")
        db_logger.retire_stepping_stones_for_parent(cur_fp, new_status="superseded")
        return

    if active_stone is not None:
        db_logger.increment_stepping_stone_revision(
            active_stone["stone_id"], max_revisions=max_revisions
        )

    q_stone = verdict.qualifying_stepping_stone
    if not (
        isinstance(q_stone, dict)
        and q_stone.get("policy_code")
        and str(q_stone.get("source_peer_id", "")) != "proactive_2"
    ):
        return

    q_stone = dict(q_stone)
    q_stone["parent_policy_id"] = cur_fp

    # Carry forward lineage revision count when the new qualifying stone came from
    # revising `active_stone` (`reactive_1` or a stack built on `reactive_1`).
    src_pid = str(q_stone.get("source_peer_id", "") or "")
    if active_stone is not None and (
        src_pid == "reactive_1" or "reactive_1" in src_pid
    ):
        carried_rev = opt_int(active_stone, "revision_count", 0) + 1
        if carried_rev >= max_revisions:
            ui.info(
                f"Gen {generation}: Reactive stepping stone `{active_stone['stone_id']}` "
                f"exhausted {carried_rev}/{max_revisions} revision(s); retiring cleanly."
            )
            return
        q_stone["revision_count"] = carried_rev

    still_active = db_logger.load_active_stepping_stone(
        parent_policy_id=cur_fp, max_revisions=max_revisions
    )
    should_save = still_active is None
    if still_active is not None:
        new_p = min(
            opt_float(q_stone, "wilcoxon_p", 1.0),
            opt_float(q_stone, "sign_p", 1.0),
        )
        old_p = min(
            opt_float(still_active, "wilcoxon_p", 1.0),
            opt_float(still_active, "sign_p", 1.0),
        )
        new_d = max(
            opt_float(q_stone, "winsor_delta", 0.0),
            opt_float(q_stone, "iqm_delta", 0.0),
        )
        old_d = max(
            opt_float(still_active, "winsor_delta", 0.0),
            opt_float(still_active, "iqm_delta", 0.0),
        )
        if (new_p < old_p) or (new_p == old_p and new_d > old_d):
            db_logger.update_stepping_stone_status(
                still_active["stone_id"], "superseded"
            )
            should_save = True
    if should_save:
        db_logger.save_stepping_stone(q_stone)
        ui.info(
            f"Gen {generation}: Saved stepping stone `{q_stone['stone_id']}` "
            f"(rev {opt_int(q_stone, 'revision_count', 0)}/{max_revisions}, "
            f"W/L={q_stone['wins']}/{q_stone['losses']}, "
            f"winsorΔ={opt_float(q_stone, 'winsor_delta', 0.0):+.3f}, "
            f"min_p={min(opt_float(q_stone, 'wilcoxon_p', 1.0), opt_float(q_stone, 'sign_p', 1.0)):.4f}) "
            f"for next-gen `reactive_1`."
        )


def update_proactive_branch_lifecycle(
    *,
    db_logger: Any,
    generation: int,
    cur_fp: str,
    active_branch: dict[str, Any] | None,
    verdict: TournamentVerdict,
    peer_candidates: list[PeerCandidate],
) -> None:
    """Updates `proactive_branches` lifecycle for Sheltered Proactive Branch (`proactive_2`, max_steps=4)."""
    if verdict.promoted:
        if active_branch is not None and "proactive_2" in str(verdict.winner_id):
            db_logger.update_proactive_branch_status(
                active_branch["branch_id"], "graduated"
            )
        db_logger.retire_proactive_branches_for_champion(
            cur_fp, new_status="superseded"
        )
        return

    p2_cand = next((c for c in peer_candidates if c.peer_id == "proactive_2"), None)
    if p2_cand is None:
        return

    p2_pair = dict((verdict.select_pair_by_peer or {}).get("proactive_2") or {})
    p2_st = dict((verdict.select_stats_by_peer or {}).get("proactive_2") or {})
    p2_exc = opt_int(p2_st, "policy_exceptions", 0)
    p2_esc = opt_int(p2_st, "auto_esc_total", opt_int(p2_st, "auto_esc", 0))
    p2_w = opt_int(p2_pair, "wins", 0)
    p2_l = opt_int(p2_pair, "losses", 0)
    p2_wd = opt_float(p2_pair, "winsor_delta", 0.0)
    sb_depth = opt_float(p2_cand.sandbox_metrics or {}, "avg_depth", 1.0)
    p2_avg_d = opt_float(p2_st, "avg_depth", sb_depth)
    inc_sel_d = float(getattr(verdict, "select_inc_avg_depth", 1.0) or 1.0)
    inc_sel_esc = int(getattr(verdict, "select_inc_auto_esc", 0) or 0)

    # Guard: if proactive_2 advanced to S_confirm and crashed or failed safety/holdout gates,
    # do NOT advance the broken/unsafe code onto the multi-generation sheltered branch.
    confirm_safety_ok = True
    conf_pair = getattr(verdict, "confirm_challenger_eval", None)
    if (
        isinstance(conf_pair, tuple)
        and len(conf_pair) == 2
        and str(conf_pair[0] or "") == "proactive_2"
    ):
        conf_eval = conf_pair[1] if isinstance(conf_pair[1], dict) else {}
        conf_runs = list(conf_eval.get("runs", []) or [])
        conf_exc = sum(opt_int(r, "policy_exceptions", 0) for r in conf_runs)
        reason_low = str(getattr(verdict, "reason", "") or "").lower()
        safety_fail_tokens = (
            "policy_exceptions",
            "exception",
            "auto_esc",
            "scenario_gate",
        )
        if (
            conf_exc > 0
            or bool(getattr(verdict, "holdout_rejected", False))
            or any(tok in reason_low for tok in safety_fail_tokens)
        ):
            confirm_safety_ok = False

    branch_gate_ok = bool(
        p2_cand.sandbox_passed
        and str(p2_cand.policy_code or "").strip()
        and confirm_safety_ok
        and p2_exc == 0
        and p2_esc <= max(inc_sel_esc, 2)
        and (p2_w + p2_l) > 0
        and p2_wd >= -0.25
        and p2_avg_d >= inc_sel_d - 0.35
    )
    max_steps_b = opt_int(active_branch or {}, "max_steps", 4)
    next_steps = opt_int(active_branch or {}, "steps_taken", 0) + 1
    next_status = "reset" if next_steps >= max_steps_b else "active"
    if branch_gate_ok:
        prev_caps = list((active_branch or {}).get("capabilities_added") or [])
        mech_s = str(
            (p2_cand.hypothesis_yaml or {}).get("mechanism", "")
            or p2_cand.claimed_label
            or ""
        )[:180]
        if mech_s and mech_s not in prev_caps:
            prev_caps.append(mech_s)
        br_id = str(
            (active_branch or {}).get("branch_id")
            or f"pbranch_g{generation}_{cur_fp[:8]}"
        )
        br_goal = str(
            (active_branch or {}).get("goal_summary")
            or mech_s
            or "Proactive multi-step architecture upgrade"
        )[:300]
        db_logger.save_proactive_branch(
            {
                "branch_id": br_id,
                "base_champion_fp": cur_fp,
                "generation_started": opt_int(
                    active_branch or {}, "generation_started", generation
                ),
                "last_generation": int(generation),
                "steps_taken": next_steps,
                "max_steps": max_steps_b,
                "status": next_status,
                "policy_code": str(p2_cand.policy_code),
                "skills": dict(p2_cand.skills or {}),
                "goal_summary": br_goal,
                "capabilities_added": prev_caps[-6:],
                "last_avg_depth": p2_avg_d,
                "last_winsor_delta": p2_wd,
            }
        )
        ui.info(
            f"Gen {generation}: Sheltered proactive branch `{br_id}` "
            f"{'reached max horizon and reset' if next_status == 'reset' else 'advanced'} "
            f"(step {next_steps}/{max_steps_b}, D={p2_avg_d:.2f}, winsorΔ={p2_wd:+.3f}, W/L={p2_w}/{p2_l})."
        )
    elif active_branch is not None:
        upd_br = dict(active_branch)
        upd_br["last_generation"] = int(generation)
        upd_br["steps_taken"] = next_steps
        upd_br["status"] = next_status
        db_logger.save_proactive_branch(upd_br)
        if next_status == "reset":
            ui.info(
                f"Gen {generation}: Sheltered proactive branch `{active_branch['branch_id']}` "
                f"expired at {next_steps}/{max_steps_b} steps without graduating; resetting to Champion."
            )


def build_layout_a_assignments(
    *,
    generation: int,
    roster_specs: list[tuple[str, str]],
    clusters: list[Any],
    orthogonal_clusters: list[Any],
    deficit_report: Any,
    current_policy: str,
    initial_skills: dict[str, str],
    probe_seeds: list[int],
    planned_directives: dict[str, dict[str, Any]],
    active_stone: dict[str, Any] | None,
    active_branch: dict[str, Any] | None,
    plateau_streak: int,
    near_misses: list[dict[str, Any]],
) -> list[
    tuple[
        str,
        Any,
        str,
        dict[str, str],
        dict[str, Any] | None,
        dict[str, Any] | None,
        str,
        dict[str, Any] | None,
    ]
]:
    """Constructs per-peer `(peer_id, cluster, start_policy, start_skills, directive, stone, track, branch)` tuples."""
    _high_leverage_kinds = {
        "stalled_trajectory",
        "recurring_terminal_state",
        "confinement_horizon",
        "search_or_horizon_loop",
    }
    cooled_down = dict(getattr(deficit_report, "cooled_down_deficits", {}) or {})
    reactive_slots = [pid for pid, trk in roster_specs if trk == "reactive"]
    assigned_reactive_clusters: list[Any] = []
    n_react = max(1, len(reactive_slots))
    if (
        str(deficit_report.regime or "").upper() == "PLATEAU"
        and len(orthogonal_clusters) >= 2
        and n_react >= 2
    ):
        ranked_cl = sorted(
            orthogonal_clusters,
            key=lambda c: (
                (
                    2.0
                    if str(getattr(c, "kind", "")) in _high_leverage_kinds
                    else (1.2 if int(getattr(c, "run_count", 0) or 0) >= 8 else 1.0)
                )
                * (0.25 if str(getattr(c, "cluster_id", "")) in cooled_down else 1.0)
                * max(1, int(getattr(c, "run_count", 0) or 1))
            ),
            reverse=True,
        )
        for idx_r in range(n_react):
            assigned_reactive_clusters.append(ranked_cl[idx_r % len(ranked_cl)])
    else:
        substantial = [
            c
            for c in orthogonal_clusters
            if int(getattr(c, "run_count", 0) or 0) >= 5
            and str(getattr(c, "cluster_id", "")) not in cooled_down
        ]
        if len(substantial) < 2:
            substantial = [
                c
                for c in orthogonal_clusters
                if int(getattr(c, "run_count", 0) or 0) >= 5
            ]
        climb_pool = substantial if len(substantial) >= 2 else orthogonal_clusters
        for idx_r in range(n_react):
            assigned_reactive_clusters.append(
                climb_pool[(generation + idx_r) % len(climb_pool)]
            )

    use_crossover = (
        active_stone is None
        and plateau_streak >= 4
        and bool(near_misses)
        and bool(current_policy.strip())
    )
    cluster_by_id = {str(getattr(c, "cluster_id", "") or ""): c for c in clusters}
    _angles = (
        "Hypothesis Angle A (Surgical Failure Fix): Isolate the exact failure precondition in the owning module so broken runs are rescued with zero regression on healthy GUARD seeds.",
        "Hypothesis Angle B (Upstream Resource/Survival/Capability): Fix upstream resource, encumbrance, survival, or capability bottlenecks causing `top_ends` terminations without touching unrelated exploration loops.",
        "Hypothesis Angle C (Stall/Trap Exit Guard): Add clean exit preconditions so stalled or repeated-failure states recover cleanly without perturbing healthy play.",
    )

    assignments: list[
        tuple[
            str,
            Any,
            str,
            dict[str, str],
            dict[str, Any] | None,
            dict[str, Any] | None,
            str,
            dict[str, Any] | None,
        ]
    ] = []
    cluster_seen_counts: dict[str, int] = {}
    fresh_r_cursor = 0

    for pid, rtrack in roster_specs:
        p_dir = planned_directives.get(pid)
        if (
            rtrack == "revision"
            and active_stone is not None
            and str(active_stone.get("policy_code", "") or "").strip()
        ):
            rev_cl = _copy.copy(clusters[0])
            rev_lost = [int(s) for s in (active_stone.get("seeds_lost") or [])[:4]]
            if rev_lost:
                rev_cl.sample_seeds = rev_lost
            rev_cl.cluster_id = f"revision_{active_stone['source_peer_id']}"
            rev_cl.label = (
                f"[Revision Track] Refine Gen {active_stone['generation']} "
                f"`{active_stone['source_peer_id']}` (W/L={active_stone['wins']}/{active_stone['losses']}, "
                f"reject={active_stone['reject_reason']})"
            )
            rev_cl.evidence = (
                f"Active stepping stone `{active_stone['stone_id']}` "
                f"(rev {opt_int(active_stone, 'revision_count', 0) + 1}/2, "
                f"winsorΔ={opt_float(active_stone, 'winsor_delta', 0.0):+.3f}, "
                f"iqmΔ={opt_float(active_stone, 'iqm_delta', 0.0):+.3f})."
            )
            assignments.append(
                (
                    pid,
                    rev_cl,
                    str(active_stone["policy_code"]),
                    dict(active_stone.get("skills") or {}),
                    p_dir,
                    active_stone,
                    "revision",
                    None,
                )
            )
            ui.info(
                f"Gen {generation}: `{pid}` picked up active reactive stepping stone "
                f"`{active_stone['stone_id']}` (Gen {active_stone['generation']} "
                f"W/L={active_stone['wins']}/{active_stone['losses']}, "
                f"rev {opt_int(active_stone, 'revision_count', 0) + 1}/2)."
            )
        elif rtrack == "proactive_capability":
            p1_seeds = list(
                dict.fromkeys(
                    (probe_seeds[:2] if probe_seeds else [])
                    + list(deficit_report.d_succ_seeds[:2])
                    + list(clusters[0].sample_seeds[:2] if clusters else [])
                )
            )[:4]
            p1_cl = DeficitCluster(
                cluster_id="proactive_expert_capability",
                kind="proactive_capability",
                label="[Proactive Capability] Expert-driven capability/efficiency gap (Dlvl 1..p90+2)",
                evidence=(
                    str(getattr(deficit_report, "expert_parity_gaps", "") or "")[:500]
                    or "Mine `expert_ticks` + Wiki to add one clean missing capability."
                ),
                run_count=len(p1_seeds),
                sample_seeds=p1_seeds or [1, 2],
            )
            assignments.append(
                (
                    pid,
                    p1_cl,
                    current_policy,
                    initial_skills,
                    p_dir,
                    None,
                    "proactive_capability",
                    None,
                )
            )
        elif rtrack == "proactive_branch":
            has_br_pol = isinstance(active_branch, dict) and bool(
                str(active_branch.get("policy_code", "") or "").strip()
            )
            br_start_pol = (
                str(active_branch["policy_code"])
                if has_br_pol and active_branch is not None
                else current_policy
            )
            br_start_skills = (
                dict(active_branch.get("skills") or {})
                if has_br_pol and active_branch is not None
                else initial_skills
            )
            br_step_idx = opt_int(active_branch or {}, "steps_taken", 0) + 1
            br_max_steps = opt_int(active_branch or {}, "max_steps", 4)
            p2_seeds = list(
                dict.fromkeys(
                    (probe_seeds[2:4] if len(probe_seeds) >= 4 else probe_seeds[:2])
                    + list(deficit_report.d_succ_seeds[:2])
                    + list(clusters[0].sample_seeds[:2] if clusters else [])
                )
            )[:4]
            p2_cl = DeficitCluster(
                cluster_id="proactive_sheltered_branch",
                kind="proactive_branch",
                label=(
                    f"[Sheltered Proactive Branch step {br_step_idx}/{br_max_steps}] "
                    + (
                        str(active_branch.get("goal_summary", "") or "")[:80]
                        if isinstance(active_branch, dict)
                        and active_branch.get("goal_summary")
                        else "Multi-step proactive architecture & efficiency upgrade"
                    )
                ),
                evidence=(
                    f"Branch `{active_branch['branch_id']}` (step {br_step_idx}/{br_max_steps}, "
                    f"last_avg_depth={opt_float(active_branch, 'last_avg_depth', 0.0):.2f}, "
                    f"last_winsorΔ={opt_float(active_branch, 'last_winsor_delta', 0.0):+.3f})."
                    if isinstance(active_branch, dict) and active_branch
                    else "Fresh 4-gen sheltered proactive branch starting from Champion."
                ),
                run_count=len(p2_seeds),
                sample_seeds=p2_seeds or [1, 2],
            )
            assignments.append(
                (
                    pid,
                    p2_cl,
                    br_start_pol,
                    br_start_skills,
                    p_dir,
                    None,
                    "proactive_branch",
                    active_branch,
                )
            )
            if isinstance(active_branch, dict) and active_branch:
                ui.info(
                    f"Gen {generation}: `{pid}` continuing sheltered proactive branch "
                    f"`{active_branch['branch_id']}` (step {br_step_idx}/{br_max_steps}, "
                    f"last_D={opt_float(active_branch, 'last_avg_depth', 0.0):.2f})."
                )
        else:
            dir_cid = str((p_dir or {}).get("target_cluster_id") or "")
            base_cl = cluster_by_id.get(
                dir_cid,
                assigned_reactive_clusters[
                    fresh_r_cursor % len(assigned_reactive_clusters)
                ],
            )
            cl = _copy.copy(base_cl)
            cid_key = str(getattr(cl, "cluster_id", "") or "")
            occ_idx = cluster_seen_counts.get(cid_key, 0)
            cluster_seen_counts[cid_key] = occ_idx + 1
            if (
                not p_dir
                and str(deficit_report.regime or "").upper() == "PLATEAU"
                and occ_idx > 0
            ):
                angle_note = _angles[occ_idx % len(_angles)]
                cl.evidence = (
                    f"{cl.evidence} | {angle_note}" if cl.evidence else angle_note
                )
            if use_crossover and fresh_r_cursor == 0:
                nm = near_misses[0]
                diff_snip = str(nm.get("diff_preview", "") or "").strip()
                diff_part = (
                    f"\nNear-Miss Unified Diff Preview:\n```diff\n{diff_snip}\n```"
                    if diff_snip
                    else ""
                )
                nm_note = (
                    f"SEMANTIC CROSSOVER / ARCHITECT PASS: Combine current incumbent's stability "
                    f"with positive near-miss `{nm['peer_id']}` from Gen {nm['generation']} "
                    f"(D={nm['avg_depth']:.2f}, maxD={nm['max_depth']}, S={nm['avg_score']:.0f}, "
                    f"W/L={nm.get('wins', 0)}/{nm.get('losses', 0)}, ΔIQM={nm.get('iqm_delta', 0.0):+.4f}, "
                    f"gained={nm['seeds_gained'][:5]}, lost={nm['seeds_lost'][:5]}, "
                    f"touched={sorted(nm['touched_skills'])[:4]}, mechanism=`{nm['mechanism'][:140]}`). "
                    f"Preserve incumbent guard stability while capturing gained seeds.{diff_part}"
                )
                cl.evidence = f"{cl.evidence} | {nm_note}" if cl.evidence else nm_note
                cl.label = f"[Semantic Crossover] {cl.label}"
            fresh_r_cursor += 1
            assignments.append(
                (
                    pid,
                    cl,
                    current_policy,
                    initial_skills,
                    p_dir,
                    None,
                    "reactive",
                    None,
                )
            )
    return assignments
