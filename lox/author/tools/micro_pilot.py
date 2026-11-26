"""
Thread-safe Paired CRN Micro-Pilot evaluator for Swarm Peer `dry_run_policy`.
Evaluates Guard + Target + Probe seeds at a strictly matched step horizon for both
candidate and incumbent baseline, persisting 100% of ticks into DuckDB `runs`/`ticks`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_PEER_MICRO_PILOT_LOCK = threading.Lock()


def first_divergence_byte(a: bytes, b: bytes) -> int | None:
    if a == b:
        return None
    lim = min(len(a), len(b))
    for i in range(lim):
        if a[i] != b[i]:
            return i
    return lim


def run_peer_paired_micro_pilot(
    registry: Any,
    policy_code: str,
    cand_skills: dict[str, str],
) -> str:
    """Runs a thread-safe paired CRN micro-pilot (D_succ guards + D_fail targets + unseen probes)."""
    if registry.pool is None or registry.max_peer_micro_pilots <= 0:
        return ""
    if registry.peer_micro_pilots_used >= registry.max_peer_micro_pilots:
        return (
            f"\n- Paired Micro-Pilot: skipped (cap {registry.peer_micro_pilots_used}/"
            f"{registry.max_peer_micro_pilots} reached; submit or refine without extra micro-pilots)."
        )

    guard_seeds = list(dict.fromkeys(int(s) for s in registry.peer_guard_seeds[:2]))
    raw_targets = list(registry.peer_target_seeds[:2])
    if not raw_targets and registry.claimed_deficit is not None:
        raw_targets = list(getattr(registry.claimed_deficit, "sample_seeds", []) or [])[
            :2
        ]
    target_seeds = [
        int(s)
        for s in dict.fromkeys(int(x) for x in raw_targets)
        if int(s) not in guard_seeds
    ]
    probe_seeds = [
        int(s)
        for s in dict.fromkeys(int(x) for x in registry.peer_probe_seeds[:12])
        if int(s) not in guard_seeds and int(s) not in target_seeds
    ]
    all_seeds = guard_seeds + target_seeds + probe_seeds
    if not all_seeds:
        return ""

    base_by_seed: dict[int, dict[str, Any]] = {
        int(r["seed"]): r
        for r in (registry.baseline_runs or [])
        if isinstance(r, dict) and "seed" in r and not r.get("is_shard")
    }

    # Include ALL selected seeds (guards, targets, probes) when sizing step horizon so a
    # long guard seed is never artificially truncated against a longer cached incumbent run!
    all_cached_horizons = [
        max(
            int(base_by_seed[s].get("steps", 0) or 0),
            int(base_by_seed[s].get("turns", 0) or 0),
        )
        for s in all_seeds
        if s in base_by_seed
    ]
    if not all_cached_horizons and registry.claimed_deficit is not None:
        all_cached_horizons = [
            int(getattr(registry.claimed_deficit, "avg_turns", 0) or 0)
        ]
    max_cached_horizon = max(all_cached_horizons, default=0)
    pool_cap = int(
        getattr(registry.pool, "max_steps", registry.max_steps) or registry.max_steps
    )
    desired_cap = 6500 if max_cached_horizon >= 4000 else max(4000, max_cached_horizon)
    micro_max_steps = max(200, min(pool_cap, desired_cap))

    # Any seed missing from cache OR cached with a longer run than `micro_max_steps`
    # must be evaluated at the exact same `micro_max_steps` so comparison is strictly apples-to-apples.
    missing_base = [s for s in all_seeds if s not in base_by_seed]
    horizon_mismatch_base = [
        s
        for s in all_seeds
        if s in base_by_seed
        and int(base_by_seed[s].get("steps", 0) or 0) > micro_max_steps
    ]
    base_eval_seeds = list(dict.fromkeys(missing_base + horizon_mismatch_base))

    attempt_idx = registry.peer_micro_pilots_used + 1
    peer_tag = str(registry.peer_id or "peer")
    base_batch_id = f"peer_mp_inc_g{registry.generation}_{peer_tag}_a{attempt_idx}_{int(time.time() * 1000)}"
    cand_batch_id = f"peer_mp_cand_g{registry.generation}_{peer_tag}_a{attempt_idx}_{int(time.time() * 1000)}"
    new_base_runs: list[dict[str, Any]] = []
    try:
        base_pol_eval = str(
            registry.baseline_policy_code or registry.current_policy_code or ""
        )
        base_skills_eval = (
            registry.baseline_skills
            if registry.baseline_skills is not None
            else registry.skills
        )
        with _PEER_MICRO_PILOT_LOCK:
            if base_eval_seeds and base_pol_eval.strip():
                base_res = registry.pool.evaluate_batch(
                    batch_id=base_batch_id,
                    policy_code=base_pol_eval,
                    macros_code="",
                    skills=base_skills_eval,
                    target_episodes=len(base_eval_seeds),
                    priority_seeds=base_eval_seeds,
                    max_steps=micro_max_steps,
                    record_ticks=True,
                    show_progress=False,
                )
                for br in base_res.get("runs", []) or []:
                    if isinstance(br, dict) and "seed" in br:
                        sd_i = int(br["seed"])
                        base_by_seed[sd_i] = dict(br)
                        if sd_i in missing_base:
                            registry.baseline_runs.append(dict(br))
                        new_base_runs.append(dict(br))

            cand_res = registry.pool.evaluate_batch(
                batch_id=cand_batch_id,
                policy_code=policy_code,
                macros_code="",
                skills=cand_skills,
                target_episodes=len(all_seeds),
                priority_seeds=all_seeds,
                max_steps=micro_max_steps,
                record_ticks=True,
                show_progress=False,
            )
    except Exception as exc:
        return f"\n- Paired Micro-Pilot: unavailable ({type(exc).__name__}: {exc})"

    registry.peer_micro_pilots_used += 1
    cand_runs_list = [
        r
        for r in (cand_res.get("runs", []) or [])
        if isinstance(r, dict) and "seed" in r
    ]
    cand_by_seed = {int(r["seed"]): r for r in cand_runs_list}
    if not cand_by_seed:
        return ""

    mp_ticks_logged = 0
    if registry.db_path:
        try:
            from lox.db.logger import BatchDatabaseLogger

            db_log = BatchDatabaseLogger(registry.db_path)
            mp_ticks_logged += db_log.save_runs_and_ticks(
                batch_id=cand_batch_id,
                generation=int(registry.generation),
                run_kind="micro_pilot",
                peer_id=peer_tag,
                runs=cand_runs_list,
                attempt_num=attempt_idx,
                run_id_prefix=f"g{registry.generation}_{peer_tag}_mp{attempt_idx}",
            )
            paired_base_runs = (
                new_base_runs
                if len(new_base_runs) == len(all_seeds)
                else [base_by_seed[sd] for sd in all_seeds if sd in base_by_seed]
            )
            if paired_base_runs:
                mp_ticks_logged += db_log.save_runs_and_ticks(
                    batch_id=base_batch_id,
                    generation=int(registry.generation),
                    run_kind="micro_pilot_baseline",
                    peer_id=peer_tag,
                    runs=paired_base_runs,
                    attempt_num=attempt_idx,
                    run_id_prefix=f"g{registry.generation}_{peer_tag}_mp{attempt_idx}_base",
                )
        except Exception as mp_log_exc:
            from lox.ui.console import ui

            ui.warn(
                f"Peer `{peer_tag}` micro-pilot baseline tick save failed: {mp_log_exc}"
            )

    lines: list[str] = [
        f"\n### Paired Micro-Pilot Diff vs Incumbent ({registry.peer_micro_pilots_used}/{registry.max_peer_micro_pilots}, cap={micro_max_steps} steps)"
    ]
    if mp_ticks_logged > 0:
        lines.append(
            f"- Queryable Ticks Logged ({mp_ticks_logged:,} ticks in `peer_ticks` / `ticks`): "
            f"filter `WHERE generation = {registry.generation} AND peer_id = '{peer_tag}' AND attempt_num = {attempt_idx} AND seed = <seed>` "
            f"(`run_kind IN ('micro_pilot', 'micro_pilot_baseline')`) via `query_db` to diagnose any flat or regressed seed directly."
        )

    guard_regressions: list[str] = []
    target_evaluated = 0
    target_unchanged = 0
    probe_evaluated = 0
    probe_diverged = 0
    probe_improved = 0
    probe_regressed = 0
    net_wins = 0
    net_losses = 0
    net_ties = 0
    total_diverged = 0
    total_cand_exc = 0
    depth_deltas_all: list[float] = []

    def _fmt_row(role: str, sd: int) -> None:
        nonlocal target_evaluated, target_unchanged, probe_evaluated, probe_diverged
        nonlocal probe_improved, probe_regressed, net_wins, net_losses, net_ties
        nonlocal total_diverged, total_cand_exc
        cr = cand_by_seed.get(sd)
        if cr is None:
            return
        br = base_by_seed.get(sd)
        c_d = int(cr.get("max_depth", cr.get("depth", 1)) or 1)
        c_sc = int(cr.get("score", 0) or 0)
        c_t = int(cr.get("turns", 0) or 0)
        c_end = str(cr.get("end_reason", "unknown") or "unknown")
        c_srch = float(cr.get("pct_search", 0.0) or 0.0)
        c_exc = int(cr.get("policy_exceptions", 0) or 0)
        c_esc = int(cr.get("auto_esc_count", 0) or 0)
        if br is None:
            esc_s = f", auto_esc={c_esc}" if c_esc else ""
            lines.append(
                f"  * {role} seed={sd}: Dlvl={c_d}, S={c_sc}, T={c_t}, end={c_end}, search={c_srch:.1f}%, exc={c_exc}{esc_s} (no incumbent cache)"
            )
            return

        b_d = int(br.get("max_depth", br.get("depth", 1)) or 1)
        b_sc = int(br.get("score", 0) or 0)
        b_t = int(br.get("turns", 0) or 0)
        b_end = str(br.get("end_reason", "unknown") or "unknown")
        b_srch = float(br.get("pct_search", 0.0) or 0.0)
        b_exc = int(br.get("policy_exceptions", 0) or 0)
        b_esc = int(br.get("auto_esc_count", 0) or 0)

        b_bytes = bytes(br.get("action_bytes") or b"")
        c_bytes = bytes(cr.get("action_bytes") or b"")
        if b_bytes and c_bytes:
            div_byte = first_divergence_byte(b_bytes, c_bytes)
        else:
            div_byte = (
                None
                if (
                    c_d == b_d
                    and c_sc == b_sc
                    and c_t == b_t
                    and c_end == b_end
                    and c_exc == b_exc
                )
                else 0
            )

        exc_note = f", exc={b_exc}->{c_exc}" if (b_exc or c_exc) else ""
        if c_esc > b_esc:
            exc_note += f", auto_esc={b_esc}->{c_esc}"
        total_cand_exc += c_exc
        if div_byte is not None:
            total_diverged += 1
        d_diff = float(c_d - b_d)
        depth_deltas_all.append(d_diff)
        if c_d > b_d or (
            c_d == b_d and c_sc >= b_sc + 50 and c_exc <= b_exc and c_esc <= b_esc + 1
        ):
            net_wins += 1
        elif c_d < b_d or c_exc > b_exc or c_esc > b_esc + 1:
            net_losses += 1
        else:
            net_ties += 1

        stall_ends = {"stalled_zero_turn", "max_steps", "crash"}
        if role == "GUARD":
            if div_byte is None:
                lines.append(
                    f"  * GUARD seed={sd}: ZERO DRIFT (Dlvl {c_d}, S={c_sc}, T={c_t}, end={c_end})"
                )
            else:
                regressed = (
                    c_d < b_d
                    or c_exc > b_exc
                    or c_esc > b_esc + 1
                    or (c_end in stall_ends and b_end not in stall_ends)
                )
                if regressed:
                    guard_regressions.append(
                        f"seed={sd} (drift @ byte {div_byte}: Dlvl {b_d}->{c_d}, S {b_sc}->{c_sc}{exc_note})"
                    )
                    lines.append(
                        f"  * GUARD seed={sd}: ⚠️ REGRESSION (DRIFT @ byte {div_byte} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}{exc_note})"
                    )
                else:
                    lines.append(
                        f"  * GUARD seed={sd}: OK (non-regressing drift @ byte {div_byte} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}{exc_note}; do NOT over-tighten)"
                    )
        elif role == "PROBE":
            probe_evaluated += 1
            if div_byte is not None:
                probe_diverged += 1
            div_txt = (
                "UNCHANGED (0-byte drift)"
                if div_byte is None
                else f"GENERALIZED (drift @ byte {div_byte})"
            )
            regressed = (
                c_d < b_d
                or c_exc > b_exc
                or c_esc > b_esc + 1
                or (c_end in stall_ends and b_end not in stall_ends)
            )
            improved = (
                c_d > b_d or (c_d == b_d and c_sc >= b_sc + 50 and c_exc <= b_exc)
            ) and c_esc <= b_esc + 1
            if improved:
                probe_improved += 1
            elif regressed:
                probe_regressed += 1
            status_tag = (
                "✅ IMPROVED" if improved else ("⚠️ REGRESSED" if regressed else div_txt)
            )
            lines.append(
                f"  * PROBE seed={sd}: {status_tag} (Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}, search {b_srch:.1f}%->{c_srch:.1f}%{exc_note})"
            )
        else:
            target_evaluated += 1
            if div_byte is None:
                target_unchanged += 1
            improved = (
                c_d > b_d
                or (
                    c_d == b_d
                    and b_end in stall_ends
                    and c_end not in stall_ends
                    and c_exc <= b_exc
                )
                or (c_d == b_d and c_sc >= b_sc + 50 and c_exc <= b_exc)
                or (
                    c_d == b_d
                    and b_srch - c_srch >= 10.0
                    and c_sc >= b_sc
                    and c_exc <= b_exc
                )
            )
            regressed = c_d < b_d or c_exc > b_exc
            div_txt = "ZERO DRIFT" if div_byte is None else f"drift @ byte {div_byte}"
            if improved:
                tag = "✅ RESCUED/IMPROVED"
            elif regressed:
                tag = "⚠️ REGRESSED"
            else:
                tag = "UNCHANGED" if div_byte is None else "FLAT"
            lines.append(
                f"  * TARGET seed={sd}: {tag} ({div_txt} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}, search {b_srch:.1f}%->{c_srch:.1f}%{exc_note})"
            )

    for sd in guard_seeds:
        _fmt_row("GUARD", sd)
    for sd in target_seeds:
        _fmt_row("TARGET", sd)
    for sd in probe_seeds:
        _fmt_row("PROBE", sd)

    if depth_deltas_all:
        avg_dd = sum(depth_deltas_all) / len(depth_deltas_all)
        probe_part = (
            f", PROBE diverged={probe_diverged}/{probe_evaluated}, regressed={probe_regressed}/{probe_evaluated}"
            if probe_evaluated > 0
            else ""
        )
        lines.append(
            f"- Micro-Pilot Net Summary: W/L/T={net_wins}/{net_losses}/{net_ties} across {len(depth_deltas_all)} seeds (avg_Δdepth={avg_dd:+.2f}{probe_part})"
        )
        mp_rec: dict[str, Any] = {
            "attempt": int(attempt_idx),
            "code": str(policy_code or ""),
            "skills": dict(cand_skills or {}),
            "avg_depth_delta": round(float(avg_dd), 4),
            "wins": int(net_wins),
            "losses": int(net_losses),
            "ties": int(net_ties),
            "diverged": int(total_diverged),
            "probe_diverged": int(probe_diverged),
            "policy_exceptions": int(total_cand_exc),
            "guard_regressions": len(guard_regressions),
        }
        registry.last_peer_micro_pilot = mp_rec
        if total_cand_exc == 0:
            prev_best = registry.best_peer_micro_pilot
            cur_rank = (
                1 if total_diverged > 0 else 0,
                1 if len(guard_regressions) == 0 else 0,
                float(avg_dd),
                int(net_wins - net_losses),
            )
            prev_rank = (
                (
                    1 if int(prev_best.get("diverged", 0) or 0) > 0 else 0,
                    1 if int(prev_best.get("guard_regressions", 0) or 0) == 0 else 0,
                    float(prev_best.get("avg_depth_delta", -999.0) or -999.0),
                    int(prev_best.get("wins", 0) or 0)
                    - int(prev_best.get("losses", 0) or 0),
                )
                if isinstance(prev_best, dict)
                else None
            )
            if prev_rank is None or cur_rank > prev_rank:
                registry.best_peer_micro_pilot = mp_rec

    if guard_regressions:
        lines.append(
            "- ⚠️ GUARD DRIFT WARNING: Guard seed(s) regressed: "
            + "; ".join(guard_regressions)
            + ". Fix the regressing logic cleanly (do NOT add artificial depth<=X / xp_level<=Y caps), then re-test with `dry_run_policy`."
        )
    if probe_evaluated > 0 and (
        probe_regressed >= 2
        or (
            probe_regressed > probe_improved
            and depth_deltas_all
            and (sum(depth_deltas_all) / len(depth_deltas_all)) <= -0.25
        )
    ):
        lines.append(
            f"- ⚠️ GENERALIZATION REGRESSION WARNING: {probe_regressed}/{probe_evaluated} unseen PROBE seed(s) regressed. "
            "Your change hurts normal unseen runs (e.g., overly aggressive cap or unguarded flee loop) — soften/fix the mechanism before submitting."
        )
    if (
        probe_evaluated >= 4
        and probe_diverged == 0
        and not (target_evaluated > 0 and target_unchanged == target_evaluated)
    ):
        lines.append(
            f"- ℹ️ ZERO-PROBE-DRIFT NOTE: Fired on 0/{probe_evaluated} unseen PROBE seeds. "
            "Verify your precondition is not over-tightened to `depth<=X`/`xl<=Y` or a 1-seed edge case."
        )
    if target_evaluated > 0 and target_unchanged == target_evaluated:
        bonus_note = ""
        if (
            registry.peer_micro_pilots_used >= registry.max_peer_micro_pilots
            and not registry._dead_patch_bonus_granted
        ):
            registry._dead_patch_bonus_granted = True
            registry.max_peer_micro_pilots += 1
            bonus_note = (
                f" (+1 bonus micro-pilot granted -> {registry.peer_micro_pilots_used}/"
                f"{registry.max_peer_micro_pilots} so you can verify the broadened fix.)"
            )
        lines.append(
            "- ⚠️ DEAD-PATCH WARNING: All TARGET seed(s) were UNCHANGED (0-byte drift). "
            "Your precondition is too narrow or unreachable — broaden it to the general failure mechanism and re-test before submitting."
            + bonus_note
        )
    elif depth_deltas_all and total_diverged == 0:
        lines.append(
            "- ⚠️ ZERO-DIVERGENCE WARNING: 0-byte behavioral drift across all evaluated micro-pilot seeds. "
            "Your change never fired — inspect preconditions and re-test before submitting."
        )
    if (
        depth_deltas_all
        and net_wins == 0
        and (sum(depth_deltas_all) / len(depth_deltas_all)) <= -0.50
    ):
        lines.append(
            f"- ⚠️ NEGATIVE MICRO-PILOT WARNING: avg_Δdepth={sum(depth_deltas_all) / len(depth_deltas_all):+.2f} "
            f"with 0 wins across {len(depth_deltas_all)} seeds. Revise or revert before submitting."
        )
    return "\n".join(lines)
