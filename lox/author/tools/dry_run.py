"""
Peer-focused live NetHack `dry_run_policy` verification:
1. AST + sandbox validation, cross-module signature checks, Pyright type check
2. Single-seed smoke check + behavioral critic (with preserved `smoke_steps`)
3. Mid-game failure & deep-frontier replays + claimed-deficit TRIGGER_MISS check
4. Optional `prompt_torture` scenario gate & Paired CRN Micro-Pilot
"""

from __future__ import annotations

import ast
import os
import traceback
from collections import Counter
from typing import Any

from lox.author.tools.code_tools import merge_skill_module_update, unpack_xml_bundle
from lox.author.tools.micro_pilot import run_peer_paired_micro_pilot
from lox.core.contracts import CandidatePackage, opt_int
from lox.db.schema import connect_db
from lox.env.harness import policy_handles_menus
from lox.lang.bundle import strip_code_fences
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    check_evolve_block_bounds,
    is_ast_equivalent,
    normalize_skill_key,
    run_pyright_on_package,
    splice_functions_into_policy,
    validate_cross_module_contracts,
)


def has_policy_class(src: str) -> bool:
    if not src or not src.strip():
        return False
    try:
        tree = ast.parse(src)
        return any(
            isinstance(n, ast.ClassDef) and n.name in ("Policy", "Agent")
            for n in tree.body
        )
    except Exception:
        return False


def _run_replay_and_frontier_checks(
    registry: Any,
    compiled: Any,
    *,
    seed: int,
    replay_run_id: str,
) -> tuple[str, int]:
    """Runs mid-game failure & deep-frontier replays; returns `(formatted_block, last_active_seed)`."""
    if (
        registry.mode != "SWARM_PEER"
        or replay_run_id
        or not os.path.exists(registry.db_path)
    ):
        return "", int(seed)
    active_seed = int(seed)
    replay_summaries: list[str] = []
    try:
        con_rep = connect_db(registry.db_path, read_only=True)
        try:
            fail_rows = con_rep.execute(
                """
                SELECT r.run_id, r.seed, r.end_reason, r.action_bytes
                FROM runs r
                WHERE octet_length(r.action_bytes) > 20
                  AND COALESCE(r.is_shard, FALSE) = FALSE
                  AND COALESCE(r.run_id, '') NOT LIKE 'shard_%'
                  AND COALESCE(r.run_kind, 'champion') = 'champion'
                  AND r.end_reason NOT IN ('ascended', 'max_steps')
                  AND (
                      NOT EXISTS (SELECT 1 FROM batches WHERE hypothesis_passed = TRUE OR batch_id LIKE '%_inc_%')
                      OR r.batch_id LIKE '%_inc_%'
                      OR EXISTS (SELECT 1 FROM batches b WHERE b.batch_id = r.batch_id AND b.hypothesis_passed = TRUE)
                  )
                ORDER BY r.generation DESC, r.turns DESC
                LIMIT 2
                """
            ).fetchall()
        finally:
            con_rep.close()
        for f_rid, f_seed, f_end, f_bytes in fail_rows:
            r_buf = bytes(f_bytes or b"")
            if not r_buf:
                continue
            cut_pt = (
                max(1, len(r_buf) - 50) if len(r_buf) > 60 else max(1, len(r_buf) - 10)
            )
            try:
                rh = registry._get_harness(120)
                r_runner = compiled.create_runner()
                r_st = rh.replay_bytes(
                    seed=int(f_seed),
                    recorded_bytes=r_buf,
                    stop_at_byte_idx=cut_pt,
                )
            except Exception:
                continue
            active_seed = int(f_seed)
            rh.step_count = 0
            r_acts: list[str] = []
            r_tiles: set[tuple[int, int, int]] = {
                (
                    max(1, int(getattr(r_st, "depth", 1) or 1)),
                    int(r_st.y),
                    int(r_st.x),
                )
            }
            for _ in range(120):
                ra, _ = r_runner.next_action(r_st)
                r_st, r_done, r_info = rh.step(ra)
                r_acts.append(str(r_info.get("action_name", "")))
                if not r_done and int(getattr(r_st, "depth", 0) or 0) > 0:
                    r_tiles.add(
                        (
                            int(getattr(r_st, "depth", 1) or 1),
                            int(r_st.y),
                            int(r_st.x),
                        )
                    )
                if r_done:
                    break
            top_r_acts = ", ".join(
                f"{k}:{v}" for k, v in Counter(r_acts).most_common(4)
            )
            r_end_s = (
                str(getattr(r_st, "end_reason", "") or "")
                or str(getattr(r_st, "message", "") or "").split(".")[0][:60]
                or "ended"
            )
            r_status = f"ENDED({r_end_s})" if rh.done else "SURVIVED(120 steps)"
            replay_summaries.append(
                f"  * `{f_rid}` (prior `{f_end}`, rewound -50B): {r_status} | "
                f"Dlvl={r_st.depth} | HP={r_st.hp}/{r_st.max_hp} | Tiles={len(r_tiles)} | Acts=[{top_r_acts}]"
            )
        active_seed = int(seed)
    except Exception as exc:
        if isinstance(
            exc,
            (
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                IndexError,
                ZeroDivisionError,
                NameError,
            ),
        ):
            raise
    if registry.swarm_frontier is not None and getattr(
        registry.swarm_frontier, "checkpoints", None
    ):
        try:
            f_cp = registry.swarm_frontier.sample_checkpoint(min_depth=2)
        except Exception:
            f_cp = None
        if f_cp is not None and getattr(f_cp, "action_bytes", b""):
            fh: Any = None
            f_runner: Any = None
            try:
                fh = registry._get_harness(len(f_cp.action_bytes) + 150)
                f_runner = compiled.create_runner()
                f_st = registry.swarm_frontier.replay_into(fh, f_cp)
            except Exception:
                f_st = None
            if f_st is not None and fh is not None and f_runner is not None:
                active_seed = int(f_cp.seed)
                fh.step_count = 0
                fh.max_steps = 120
                f_acts: list[str] = []
                f_tiles: set[tuple[int, int, int]] = {
                    (
                        max(1, int(getattr(f_st, "depth", 1) or 1)),
                        int(f_st.y),
                        int(f_st.x),
                    )
                }
                for _ in range(120):
                    fa, _ = f_runner.next_action(f_st)
                    f_st, f_done, f_info = fh.step(fa)
                    f_acts.append(str(f_info.get("action_name", "")))
                    if not f_done and int(getattr(f_st, "depth", 0) or 0) > 0:
                        f_tiles.add(
                            (
                                int(getattr(f_st, "depth", 1) or 1),
                                int(f_st.y),
                                int(f_st.x),
                            )
                        )
                    if f_done:
                        break
                top_f_acts = ", ".join(
                    f"{k}:{v}" for k, v in Counter(f_acts).most_common(4)
                )
                f_end_s = (
                    str(getattr(f_st, "end_reason", "") or "")
                    or str(getattr(f_st, "message", "") or "").split(".")[0][:60]
                    or "ended"
                )
                f_status = f"ENDED({f_end_s})" if fh.done else "SURVIVED(120 steps)"
                replay_summaries.append(
                    f"  * Frontier `{f_cp.checkpoint_id}` (Dlvl {f_cp.depth}, seed={f_cp.seed}): {f_status} | "
                    f"EndDlvl={f_st.depth} | HP={f_st.hp}/{f_st.max_hp} | Tiles={len(f_tiles)} | Acts=[{top_f_acts}]"
                )
                active_seed = int(seed)
    if not replay_summaries:
        return "", active_seed
    return (
        "\n- Mid-Game Failure & Frontier-Replay Checks:\n"
        + "\n".join(replay_summaries),
        active_seed,
    )


def _check_claimed_deficit_trigger_miss(
    registry: Any,
    compiled: Any,
    runner: Any,
    changed_fns: list[str],
    cand_skills: dict[str, str],
) -> str:
    if (
        registry.mode != "SWARM_PEER"
        or registry.claimed_deficit is None
        or not getattr(registry.claimed_deficit, "replay_fixtures", None)
    ):
        return ""
    edited_fns: set[str] = {
        str(fn).strip() for fn in (changed_fns or []) if str(fn).strip()
    }
    for sk_k, sk_src in cand_skills.items():
        base_sk = str(registry.skills.get(sk_k, "") or "")
        if sk_src.strip() == base_sk.strip():
            continue
        try:
            c_ast = ast.parse(sk_src)
            b_ast = (
                ast.parse(base_sk)
                if base_sk.strip()
                else ast.Module(body=[], type_ignores=[])
            )
            b_dumps = {
                n.name: ast.dump(n)
                for n in ast.walk(b_ast)
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")
            }
            for n in ast.walk(c_ast):
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("__"):
                    if b_dumps.get(n.name) != ast.dump(n):
                        edited_fns.add(n.name)
        except SyntaxError:
            pass
    edited_fns.discard("run")
    edited_fns.discard("__init__")
    if not edited_fns:
        return ""
    fixture_calls: Counter[str] = Counter(runner.tracer.fn_calls)
    for fix in (registry.claimed_deficit.replay_fixtures or [])[:2]:
        f_buf = bytes(fix.get("action_bytes") or b"")
        if not f_buf:
            continue
        try:
            fh = registry._get_harness(100)
            f_runner = compiled.create_runner()
            f_st = fh.replay_bytes(
                seed=int(fix.get("seed", 1)),
                recorded_bytes=f_buf,
                stop_at_byte_idx=int(
                    fix.get("stop_at_byte_idx", max(1, len(f_buf) - 35))
                ),
            )
        except Exception:
            continue
        fh.step_count = 0
        for _ in range(100):
            fa, _ = f_runner.next_action(f_st)
            f_st, f_done, _ = fh.step(fa)
            if f_done:
                break
        fixture_calls.update(f_runner.tracer.fn_calls)
    if sum(fixture_calls.get(fn, 0) for fn in edited_fns) == 0:
        return (
            f"\n- TRIGGER_MISS WARNING: Modified function(s) {sorted(edited_fns)} executed 0 times "
            "during smoke + claimed-deficit replay! Your change may be dead/unreachable on target states."
        )
    return ""


def execute_dry_run_policy(
    registry: Any,
    code: str = "",
    seed: int = 42,
    max_turns: int = 150,
    replay_run_id: str = "",
    replay_steps: int = 0,
    smoke_only: bool = False,
    skill_name: str | None = None,
    skill_code: str | None = None,
    skills: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Compiles and validates a candidate policy/skill bundle in a live NetHack sandbox."""
    policy_code = registry.current_policy_code
    registry.last_dry_run_mode = "check"
    registry.last_dry_run_passed = False
    registry.last_dry_run_diag = ""

    try:
        code, bundle_skills = unpack_xml_bundle(code)

        cand_skills = dict(registry.skills)
        if registry.mode != "SWARM_PEER":
            for prev_k, prev_v in registry.last_passed_skills.items():
                if prev_v:
                    cand_skills[prev_k] = prev_v
        if isinstance(skills, dict):
            for k, v in skills.items():
                clean_k = normalize_skill_key(str(k or ""))
                if clean_k and str(v or "").strip():
                    cleaned_v = strip_code_fences(str(v))
                    cand_skills[clean_k] = merge_skill_module_update(
                        cand_skills.get(clean_k, ""), cleaned_v
                    )
        for k, v in bundle_skills.items():
            clean_bk = normalize_skill_key(str(k or ""))
            if clean_bk:
                cand_skills[clean_bk] = merge_skill_module_update(
                    cand_skills.get(clean_bk, ""), v
                )
        if skill_name and skill_code:
            clean_sn = normalize_skill_key(str(skill_name))
            if clean_sn:
                cleaned_sc = strip_code_fences(str(skill_code))
                cand_skills[clean_sn] = merge_skill_module_update(
                    cand_skills.get(clean_sn, ""), cleaned_sc
                )
    except Exception as exc:
        registry.last_dry_run_diag = f"{type(exc).__name__}: {exc}"
        return (
            f"Compilation / Validation Error:\n{type(exc).__name__}: {exc}",
            "COMPILE_ERROR",
        )

    has_replay = bool(str(replay_run_id or "").strip())

    if (
        registry.mode == "SWARM_PEER"
        and registry.enforce_peer_research_gate
        and not smoke_only
    ):
        total_empirical = (
            int(registry.db_queries_run)
            + int(registry.wiki_queries_run)
            + int(registry.theories_queries_run)
        )
        if total_empirical == 0:
            return (
                "RESEARCH_REQUIRED: Before calling `dry_run_policy`, run at least 1 empirical grounding lookup "
                "(`query_db(...)` on `champion_ticks`/`champion_runs`, `query_wiki(...)`/`query_monster(...)`/`query_item(...)`, "
                "or `query_theories(...)`) to verify the exact failure trace or rule before patching.",
                "RESEARCH_REQUIRED",
            )

    try:
        use_surgical_splice = bool(policy_code.strip()) and (
            registry.mode != "SWARM_PEER" or not has_policy_class(code)
        )
        if use_surgical_splice:
            if code.strip():
                spliced_policy, changed_fns = splice_functions_into_policy(
                    policy_code, code
                )
            else:
                spliced_policy = policy_code
                changed_fns = []
        else:
            spliced_policy = code
            changed_fns = []

        if (
            registry.mode == "SWARM_PEER"
            and not smoke_only
            and not has_replay
            and registry.current_policy_code.strip()
            and is_ast_equivalent(spliced_policy, registry.current_policy_code)
            and set(cand_skills.keys()) == set(registry.skills.keys())
            and all(
                is_ast_equivalent(cand_skills[k], registry.skills[k])
                for k in cand_skills
            )
        ):
            return (
                "NOOP_DIFF: Your candidate code is AST-identical to the current incumbent policy and skills. "
                "Make a substantive behavioral modification before calling `dry_run_policy`.",
                "NOOP_DIFF",
            )

        scaffold_ref = registry.swarm_scaffold_code or registry.current_policy_code
        if (
            registry.mode == "SWARM_PEER"
            and scaffold_ref.strip()
            and "#EVOLVE-BLOCK" in scaffold_ref.upper()
        ):
            eb_err = check_evolve_block_bounds(scaffold_ref, spliced_policy)
            if eb_err:
                return (f"Evolve-Block Guard Error:\n{eb_err}", "COMPILE_ERROR")

        compiled = compile_policy_package(spliced_policy, "", skills=cand_skills)
        validate_cross_module_contracts(spliced_policy, cand_skills)
        pyright_errs = run_pyright_on_package(spliced_policy, cand_skills)
        if pyright_errs:
            registry.last_dry_run_diag = "PYRIGHT: " + " | ".join(pyright_errs[:4])
            return (
                "Type / Contract Validation Error (pyright):\n"
                + "\n".join(f"- {e}" for e in pyright_errs),
                "COMPILE_ERROR",
            )
    except Exception as exc:
        registry.last_dry_run_diag = f"{type(exc).__name__}: {exc}"
        return (
            f"Compilation / Validation Error:\n{type(exc).__name__}: {exc}",
            "COMPILE_ERROR",
        )

    # Step 1: Single-Episode Smoke Check + Behavioral Critic
    active_seed = int(seed)
    harness = registry._get_harness(max_turns)
    harness.allow_menu_linger = policy_handles_menus(spliced_policy, cand_skills)
    try:
        runner = compiled.create_runner()
        actions_taken: list[str] = []
        raw_act_names: list[str] = []
        unique_positions: set[tuple[int, int, int, int]] = set()
        positions_history: list[tuple[int, int, int]] = []
        oscillation_detected = False
        smoke_msg_counts: Counter[str] = Counter()
        prompt_repeat_count = 0
        last_prompt_text = ""
        prompt_livelock_detected = False
        livelock_prompt_msg = ""

        if replay_run_id and os.path.exists(registry.db_path):
            con = connect_db(registry.db_path, read_only=True)
            try:
                row = con.execute(
                    "SELECT seed, action_bytes FROM runs WHERE run_id = ?",
                    [replay_run_id],
                ).fetchone()
            finally:
                con.close()
            if row:
                r_seed, r_bytes = int(row[0]), bytes(row[1] or b"")
                cut = (
                    min(len(r_bytes), int(replay_steps))
                    if int(replay_steps) > 0
                    else max(0, len(r_bytes) - 25)
                )
                state = harness.replay_bytes(
                    seed=r_seed,
                    recorded_bytes=r_bytes,
                    stop_at_byte_idx=cut,
                )
                if harness.done and cut > 1:
                    for back in (25, 60, max(100, cut // 2)):
                        state = harness.replay_bytes(
                            seed=r_seed,
                            recorded_bytes=r_bytes,
                            stop_at_byte_idx=max(1, cut - back),
                        )
                        if not harness.done:
                            break
                harness.step_count = 0
                harness.done = False
            else:
                state = harness.reset(seed=int(seed))
        else:
            state = harness.reset(seed=int(seed))

        smoke_cap = max(10, min(2000, int(max_turns)))
        last_live_state = state if int(getattr(state, "depth", 0) or 0) > 0 else None
        smoke_ended = False
        smoke_end_reason = ""
        for _ in range(smoke_cap):
            raw_act, active_fn = runner.next_action(state)
            state, done, info = harness.step(raw_act)
            if not done and int(getattr(state, "depth", 0) or 0) > 0:
                last_live_state = state
            msg_l = (state.message or "").lower()
            is_prompt = (
                getattr(state, "in_yn_prompt", False)
                or getattr(state, "in_text_prompt", False)
                or getattr(state, "in_getdir_prompt", False)
                or getattr(state, "in_getobj_prompt", False)
                or getattr(state, "in_menu", False)
            )
            if is_prompt:
                if msg_l == last_prompt_text:
                    prompt_repeat_count += 1
                    if prompt_repeat_count >= 5:
                        prompt_livelock_detected = True
                        livelock_prompt_msg = state.message or "interactive prompt"
                else:
                    prompt_repeat_count = 1
                    last_prompt_text = msg_l
            else:
                prompt_repeat_count = 0
                last_prompt_text = ""
            msg_clean = (state.message or "").strip()
            if msg_clean:
                smoke_msg_counts[msg_clean] += 1
            if not done and int(getattr(state, "depth", 0) or 0) > 0:
                d_now = int(getattr(state, "depth", 1) or 1)
                unique_positions.add(
                    (
                        int(getattr(state, "dungeon_num", 0) or 0),
                        d_now,
                        int(state.y),
                        int(state.x),
                    )
                )
                positions_history.append((d_now, int(state.y), int(state.x)))
            if len(positions_history) >= 40:
                if len(set(positions_history[-40:])) <= 3:
                    oscillation_detected = True
            act_name = info["action_name"]
            raw_act_names.append(act_name)
            actions_taken.append(f"{active_fn}:{act_name}")
            if done:
                hit_smoke_cap = (
                    harness.step_count >= smoke_cap
                    and int(getattr(state, "depth", 0) or 0) > 0
                    and int(getattr(state, "hp", 0) or 0) > 0
                )
                if not hit_smoke_cap:
                    smoke_ended = True
                    smoke_end_reason = (
                        str(getattr(state, "end_reason", "") or "")
                        or str(info.get("end_reason", "") or "")
                        or str(getattr(state, "message", "") or "").split(".")[0][:80]
                        or "ended"
                    )
                break

        # Snapshot smoke step count BEFORE any extra smoke seeds or replay fixtures reuse `harness`
        smoke_steps = int(harness.step_count)

        fn_summary = ", ".join(f"{k}:{v}" for k, v in runner.tracer.fn_calls.items())
        act_counts = ", ".join(
            f"{k}:{v}" for k, v in Counter(raw_act_names).most_common(6)
        )
        params_summary = (
            ", ".join(
                f"{k}={v.value}" for k, v in runner.tracer.registered_params.items()
            )
            or "none"
        )
        sample_acts = " -> ".join(actions_taken[:10])

        # Multi-seed smoke sweep on diverse seeds [1000, 1001] when running smoke_only
        if not has_replay and smoke_only:
            extra_smoke_steps = min(100, max(50, int(max_turns)))
            for extra_s in (1000, 1001):
                if int(seed) == extra_s:
                    continue
                active_seed = int(extra_s)
                extra_runner = compiled.create_runner()
                est = harness.reset(seed=extra_s)
                for _ in range(extra_smoke_steps):
                    eact, _ = extra_runner.next_action(est)
                    est, edone, _ = harness.step(eact)
                    if edone:
                        break
            active_seed = int(seed)

        warn_lines: list[str] = []
        for h_warn in audit_policy_ast_hygiene(spliced_policy):
            warn_lines.append(f"- AST HYGIENE WARNING: {h_warn}")
        if prompt_livelock_detected:
            warn_lines.append(
                f"- INTERACTIVE PROMPT LIVELOCK DETECTED: The prompt '{livelock_prompt_msg}' was triggered repeatedly without resolving! "
                "Ensure your prompt handling answers the prompt cleanly instead of looping indefinitely."
            )
        if smoke_msg_counts:
            top_msgs = ", ".join(
                f"{m!r}: {c}x" for m, c in smoke_msg_counts.most_common(4) if c >= 2
            )
            if top_msgs:
                warn_lines.append(f"- REPEATED MESSAGES: {top_msgs}")
        if len(unique_positions) == 1 and smoke_steps >= 15:
            warn_lines.append(
                "- WARNING: Hero stayed on 1 tile ((y,x) did not change) across all steps! Verify movement logic."
            )
        if oscillation_detected:
            warn_lines.append(
                "- BEHAVIORAL WARNING: 2-tile or 3-tile ping-pong oscillation detected (visited <= 3 unique tiles over 40 steps). Ensure you use persistent targets and pos_history cycle breaking!"
            )
        if len(raw_act_names) >= 20:
            stall_acts = sum(1 for a in raw_act_names if a in ("SEARCH", "WAIT"))
            if stall_acts / len(raw_act_names) > 0.5:
                warn_lines.append(
                    f"- BEHAVIORAL WARNING: High action stall rate ({stall_acts}/{len(raw_act_names)} actions were SEARCH or WAIT)."
                )
        warn_block = ("\n" + "\n".join(warn_lines)) if warn_lines else ""

        if smoke_only:
            registry.last_passed_code = (
                spliced_policy
                if (registry.mode == "SWARM_PEER" and has_policy_class(spliced_policy))
                else code
            )
            registry.last_dry_run_mode = "check"
            registry.last_dry_run_passed = True
            registry.last_passed_skills = dict(cand_skills)
            if skill_name and skill_code is not None:
                registry.last_passed_skill_name = str(skill_name)
                registry.last_passed_skill_code = str(skill_code)
            registry.last_passed_pkg = CandidatePackage(
                code=registry.last_passed_code,
                skill_name=str(skill_name or ""),
                skill_code=str(skill_code or ""),
                skills_snapshot=dict(cand_skills),
            )
            return (
                f"Dry-Run Smoke Check PASSED ({smoke_steps} steps on seed {seed}).{warn_block}",
                "PASSED (smoke check)",
            )

        replay_checks_block, active_seed = _run_replay_and_frontier_checks(
            registry, compiled, seed=int(seed), replay_run_id=replay_run_id
        )
        warn_block += _check_claimed_deficit_trigger_miss(
            registry, compiled, runner, changed_fns, cand_skills
        )

        if not smoke_only and registry.scenario_gate in ("advisory", "strict"):
            try:
                from lox.env.benchmarks.nethack_scenarios import run_scenario_gate

                sg = run_scenario_gate(
                    spliced_policy,
                    cand_skills,
                    seed=int(seed),
                    max_steps=40,
                    include_archetypes=False,
                    include_prompt_torture=True,
                )
                if sg.get("exceptions", 0) > 0 or sg.get("prompt_livelocks", 0) > 0:
                    msg_sg = (
                        f"Prompt-Torture Scenario Gate flagged issues: passed={sg['passed_count']}/{sg['total_count']}, "
                        f"exc={sg['exceptions']}, livelocks={sg['prompt_livelocks']}."
                    )
                    if registry.scenario_gate == "strict":
                        return (msg_sg, "RUNTIME_ERROR")
                    warn_block += f"\n- SCENARIO_GATE ADVISORY: {msg_sg}"
            except Exception as sg_exc:
                if registry.scenario_gate == "strict":
                    return (
                        f"Prompt-Torture Scenario Gate raised internal error: {type(sg_exc).__name__}: {sg_exc}",
                        "RUNTIME_ERROR",
                    )
                warn_block += f"\n- SCENARIO_GATE ERROR: {sg_exc}"

        micro_pilot_block = ""
        if (
            not smoke_only
            and registry.mode == "SWARM_PEER"
            and registry.pool is not None
        ):
            micro_pilot_block = run_peer_paired_micro_pilot(
                registry=registry,
                policy_code=spliced_policy,
                cand_skills=cand_skills,
            )
            last_mp_rec = getattr(registry, "last_peer_micro_pilot", None)
            mp_exc = (
                opt_int(last_mp_rec, "policy_exceptions", 0)
                if isinstance(last_mp_rec, dict)
                else 0
            )
            if mp_exc > 0:
                return (
                    f"Dry-Run FAILED: Paired micro-pilot raised {mp_exc} runtime exception(s). "
                    f"Fix all exceptions before finalizing.{micro_pilot_block}",
                    "RUNTIME_ERROR",
                )

        registry.last_passed_code = (
            spliced_policy
            if (registry.mode == "SWARM_PEER" and has_policy_class(spliced_policy))
            else code
        )
        registry.last_dry_run_mode = "check"
        registry.last_dry_run_passed = True
        registry.last_passed_skills = dict(cand_skills)
        if skill_name and skill_code is not None:
            registry.last_passed_skill_name = str(skill_name)
            registry.last_passed_skill_code = str(skill_code)
        registry.last_passed_pkg = CandidatePackage(
            code=registry.last_passed_code,
            skill_name=str(skill_name or ""),
            skill_code=str(skill_code or ""),
            skills_snapshot=dict(cand_skills),
        )
        disp_state = (
            last_live_state
            if (
                smoke_ended
                and int(getattr(state, "depth", 0) or 0) == 0
                and last_live_state is not None
            )
            else state
        )
        status_suffix = (
            f", Status=ENDED({smoke_end_reason})" if smoke_ended else ", Status=ALIVE"
        )
        return (
            f"Dry-Run PASSED ({smoke_steps} steps on seed {seed}).\n"
            f"- Final State: Dlvl={disp_state.depth}, Pos=({disp_state.y},{disp_state.x}), UniqueTiles={len(unique_positions)}, HP={disp_state.hp}/{disp_state.max_hp}, Turn={disp_state.turn}{status_suffix}\n"
            f"- Functions Called: {{{fn_summary}}}\n"
            f"- Action Breakdown: {{{act_counts}}}\n"
            f"- Registered Params: {{{params_summary}}}\n"
            f"- First 10 Actions: {sample_acts}{replay_checks_block}{warn_block}{micro_pilot_block}",
            f"PASSED ({smoke_steps} steps)",
        )
    except Exception as exc:
        tb = traceback.format_exc(limit=-8)
        extra_hint = ""
        if isinstance(exc, ImportError) and "skills." in str(exc):
            extra_hint = (
                "\nHINT: Wrap every new or modified skill inside "
                '`<skill_code module="name">...</skill_code>` inside `dry_run_policy(code=...)` '
                "(alongside `<policy_code>...</policy_code>` if editing `policy.py`). "
                "When editing an existing skill, keep existing exports or emit a function patch — "
                "partial patches splice cleanly without dropping unchanged helpers."
            )
        registry.last_dry_run_diag = f"{type(exc).__name__}: {exc}{extra_hint}"
        return (
            f"Runtime Error during Dry-Run on seed {active_seed}:\n{type(exc).__name__}: {exc}\n{tb}{extra_hint}",
            "RUNTIME_ERROR",
        )
