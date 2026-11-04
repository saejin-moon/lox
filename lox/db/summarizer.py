"""
Domain-Agnostic Run Summarizer, Outlier Detector & Statistical Contrast Engine ("Hard Code Nothing").
1. `summarize_episode_ticks`: Compresses raw tick records into a 3-part `summary_digest`
   (Run-Length Encoded timeline, Top Inflection Windows, Stagnation/Oscillation Loops) + deduplicated `game_log`.
2. `mark_outlier_good_runs`: Flags statistical exception runs (`is_outlier_good = True`) in a batch via Z-score & IQR.
3. `compute_function_stats`: Computes Ochiai Good-Run correlation & Liblit Failure Importance per function.
4. `build_batch_contrast_digest`: Summarizes what separated `is_outlier_good = True` runs from normal runs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

import numpy as np


def summarize_episode_ticks(
    ticks: list[dict[str, Any]],
    final_info: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """
    Compresses a sequence of raw episode ticks into:
    - `summary_digest` (str): Compact, LLM-digestible 3-part causal summary
    - `game_log` (str): Chronological deduplicated log of game messages
    - `milestones` (dict): Structured milestone stats (depths reached, min/max HP, best AC, max XP level)
    """
    if not ticks:
        return (
            "Empty run (0 ticks).",
            "",
            {"depths_visited": [1], "best_ac": 10, "max_xl": 1},
        )

    # 1. Milestones & deduplicated game log
    depths_seen: list[int] = []
    depth_first_turn: dict[int, int] = {}
    best_ac = int(ticks[0].get("ac", 10))
    max_xl = int(ticks[0].get("xp_level", 1))
    log_lines: list[str] = []
    last_msg = ""
    msg_repeat = 0

    # Domain-agnostic progress expansion & stagnation tracking
    visited_positions: set[tuple[int, int]] = set()
    max_composite_progress = -1.0
    last_progress_turn = 0

    for t in ticks:
        d = int(t.get("depth", 1))
        turn = int(t.get("turn", 0))
        ac = int(t.get("ac", 10))
        xl = int(t.get("xp_level", 1))
        sc = float(t.get("score", 0))
        ach = float(t.get("achievements", 0))
        pos = (int(t.get("y", 0)), int(t.get("x", 0)))

        # Universal composite progress monotonic evaluation
        prog = d * 10.0 + xl * 2.0 + sc * 0.01 + ach * 100.0
        is_expansion = False
        if prog > max_composite_progress:
            max_composite_progress = prog
            is_expansion = True
        if pos not in visited_positions:
            visited_positions.add(pos)
            is_expansion = True
        if is_expansion:
            last_progress_turn = turn

        if d not in depth_first_turn:
            depth_first_turn[d] = turn
            depths_seen.append(d)
        if ac < best_ac:
            best_ac = ac
        if xl > max_xl:
            max_xl = xl

        msg = str(t.get("message", "")).strip()
        if msg:
            if msg == last_msg:
                msg_repeat += 1
            else:
                if msg_repeat > 1 and log_lines:
                    log_lines[-1] += f" (x{msg_repeat})"
                act = t.get("action_name", "")
                log_lines.append(f"[T{turn} D{d} {act}] {msg}")
                last_msg = msg
                msg_repeat = 1
    if msg_repeat > 1 and log_lines:
        log_lines[-1] += f" (x{msg_repeat})"

    # Keep first 25 and last 45 log messages so game_log never explodes context
    if len(log_lines) > 70:
        game_log = "\n".join(
            log_lines[:25]
            + [f"... ({len(log_lines) - 70} intermediate messages omitted) ..."]
            + log_lines[-45:]
        )
    else:
        game_log = "\n".join(log_lines)

    # 2. Part A: Run-Length Encoded (RLE) Phase Timeline by (depth, active_function)
    segments: list[dict[str, Any]] = []
    for t in ticks:
        d = t.get("depth", 1)
        fn = t.get("active_function", "root")
        act = t.get("action_name", "")
        hp = t.get("hp", 0)
        turn = t.get("turn", 0)
        pos = (t.get("y", 0), t.get("x", 0))
        msg = str(t.get("message", "")).strip()

        if not segments or segments[-1]["depth"] != d or segments[-1]["fn"] != fn:
            segments.append(
                {
                    "depth": d,
                    "fn": fn,
                    "start_turn": turn,
                    "end_turn": turn,
                    "steps": 1,
                    "start_hp": hp,
                    "end_hp": hp,
                    "actions": {act: 1},
                    "positions": {pos},
                    "messages": {msg} if msg else set(),
                }
            )
        else:
            seg = segments[-1]
            seg["end_turn"] = turn
            seg["steps"] += 1
            seg["end_hp"] = hp
            acts = seg["actions"]
            acts[act] = acts.get(act, 0) + 1
            seg["positions"].add(pos)
            if msg:
                seg["messages"].add(msg)

    # Collapse consecutive micro-segments if there are too many
    rle_lines: list[str] = []
    display_segs = segments if len(segments) <= 14 else (segments[:6] + segments[-8:])
    for seg in display_segs:
        top_acts_list = sorted(
            seg["actions"].items(), key=lambda kv: kv[1], reverse=True
        )[:3]
        top_acts = ", ".join(f"{k}:{v}" for k, v in top_acts_list)
        hp_str = (
            f"HP:{seg['start_hp']}->{seg['end_hp']}"
            if seg["start_hp"] != seg["end_hp"]
            else f"HP:{seg['end_hp']}"
        )
        uniq_tiles = len(seg["positions"])
        rle_lines.append(
            f"  [T{seg['start_turn']}..{seg['end_turn']} ({seg['steps']}s) | Dlvl:{seg['depth']} | "
            f"fn={seg['fn']} | {hp_str} | tiles={uniq_tiles} | acts=({top_acts})]"
        )
    if len(segments) > 14:
        rle_lines.insert(
            6, f"  ... ({len(segments) - 14} intermediate phase segments) ..."
        )

    # 3. Part B: Critical Inflection Windows (sharpest HP/Depth/Score shifts) + Final Termination Window
    inflection_lines: list[str] = []
    n = len(ticks)
    window = 5
    candidates: list[tuple[float, int]] = []
    inflection_fn = ""
    pre_inflection_fn = ""
    max_hp_drop = 0
    best_drop_idx = -1

    for i in range(0, max(1, n - window), 1):
        t0 = ticks[i]
        t1 = ticks[min(n - 1, i + window)]
        drop = int(t0.get("hp", 0)) - int(t1.get("hp", 0))
        max_hp = max(1, int(t0.get("max_hp", 1)))
        if drop > max_hp_drop and drop / max_hp >= 0.25:
            max_hp_drop = drop
            best_drop_idx = i

    if best_drop_idx >= 0:
        inflection_fn = str(ticks[best_drop_idx].get("active_function", ""))
        pre_idx = max(0, best_drop_idx - 5)
        pre_inflection_fn = str(ticks[pre_idx].get("active_function", ""))

    for i in range(0, max(1, n - window), window):
        t0 = ticks[i]
        t1 = ticks[min(n - 1, i + window)]
        hp_diff = abs(int(t1.get("hp", 0)) - int(t0.get("hp", 0)))
        d_diff = abs(int(t1.get("depth", 1)) - int(t0.get("depth", 1))) * 15.0
        sc_diff = abs(int(t1.get("score", 0)) - int(t0.get("score", 0))) * 0.1
        mag = hp_diff + d_diff + sc_diff
        if mag >= 4.0:
            candidates.append((mag, i))

    candidates.sort(reverse=True)
    chosen_indices = sorted(idx for _, idx in candidates[:3])
    for idx in chosen_indices:
        sub = ticks[idx : min(n, idx + window + 1)]
        t_start = sub[0].get("turn", 0)
        t_end = sub[-1].get("turn", 0)
        d0, d1 = sub[0].get("depth", 1), sub[-1].get("depth", 1)
        hp0, hp1 = sub[0].get("hp", 0), sub[-1].get("hp", 0)
        msgs = [
            str(s.get("message", "")).strip()
            for s in sub
            if str(s.get("message", "")).strip()
        ]
        msg_str = " / ".join(dict.fromkeys(msgs))[:120] or "no msg"
        fns = "/".join(dict.fromkeys(str(s.get("active_function", "")) for s in sub))
        inflection_lines.append(
            f"  * T{t_start}..{t_end} (D{d0}->D{d1}, HP {hp0}->{hp1}, fn={fns}): {msg_str}"
        )

    # Final 8 ticks before end
    tail = ticks[-min(8, n) :]
    tail_str = " -> ".join(
        f"T{s.get('turn', 0)}({s.get('active_function', '')}:{s.get('action_name', '')} HP={s.get('hp', 0)} '{str(s.get('message', ''))[:35]}')"
        for s in tail
    )

    # 4. Part C: Stagnation / Loop Detection (segments with >=40 steps and <=3 unique tiles)
    stalls: list[str] = []
    stall_fns: set[str] = set()
    for seg in segments:
        if seg["steps"] >= 40 and len(seg["positions"]) <= 3:
            fn_name = str(seg.get("fn", ""))
            if fn_name:
                stall_fns.add(fn_name)
            top_a = sorted(seg["actions"].items(), key=lambda kv: kv[1], reverse=True)[
                :2
            ]
            msgs_lower = " ".join(seg.get("messages", set())).lower()
            if "swap places with" in msgs_lower or "in the way" in msgs_lower:
                cause = "pet displacement loop (yield WAIT to let pet move)"
            elif "locked" in msgs_lower:
                cause = "locked door stall (needs seq(Action.KICK, dir))"
            elif "try to move the boulder" in msgs_lower:
                cause = "boulder blockage (path blocked by boulder)"
            elif "solid stone" in msgs_lower or "doorway" in msgs_lower:
                cause = "wall/doorway diagonal collision"
            elif "descend the stairs" in msgs_lower:
                cause = "staircase / --More-- pagination pause"
            else:
                cause = f"{len(seg['positions'])} tiles oscillation"

            stalls.append(
                f"fn={seg['fn']} on Dlvl {seg['depth']} [{cause}] (T{seg['start_turn']}..{seg['end_turn']}, "
                f"{seg['steps']} steps: {top_a})"
            )

    t_last = ticks[-1]
    overview = (
        f"Depth={t_last.get('depth', 1)} (Max={max(depths_seen) if depths_seen else 1}) | "
        f"Turns={t_last.get('turn', 0)} (Steps={n}) | "
        f"HP={t_last.get('hp', 0)}/{t_last.get('max_hp', 0)} | AC={best_ac} | XL={max_xl}"
    )
    stall_line = f"Stalls: {'; '.join(stalls[:3])}" if stalls else "Stalls: None"

    summary_digest = "\n".join(
        [
            f"Overview: {overview}",
            f"{stall_line}",
            "Phase Timeline (RLE):",
            *rle_lines,
            "Key Inflections:",
            *(inflection_lines or ["  * None"]),
            f"Final Sequence: {tail_str}",
        ]
    )

    d1_ticks = [t for t in ticks if t.get("depth", 1) == 1]
    stairs_found = any(
        t.get("action_name") == "DOWN"
        or t.get("raw_key") == ord(">")
        or "descend the stairs" in str(t.get("message", "")).lower()
        for t in ticks
    )

    t_end = int(ticks[-1].get("turn", len(ticks))) if ticks else 0
    stagnation_turns = max(0, t_end - last_progress_turn)
    stagnation_ratio = float(stagnation_turns / max(1, t_end))

    # Recurrence rate and active function attribution over stagnation window
    stagnant_ticks = [t for t in ticks if int(t.get("turn", 0)) >= last_progress_turn]
    stagnation_positions = {
        (int(t.get("y", 0)), int(t.get("x", 0))) for t in stagnant_ticks
    }
    if len(stagnant_ticks) >= 40:
        recurrence_rate = float(
            1.0 - (len(stagnation_positions) / max(1, len(stagnant_ticks)))
        )
    else:
        recurrence_rate = 0.0

    stagnation_fn_turns: dict[str, int] = {}
    for t in stagnant_ticks:
        fn = str(t.get("active_function", "root") or "root")
        stagnation_fn_turns[fn] = stagnation_fn_turns.get(fn, 0) + 1

    top_stagnation_fn = (
        max(stagnation_fn_turns.items(), key=lambda kv: kv[1])[0]
        if stagnation_fn_turns
        else ""
    )

    milestones = {
        "max_depth": max(depths_seen) if depths_seen else 1,
        "depths_visited": depths_seen,
        "depth_first_turn": {str(k): v for k, v in depth_first_turn.items()},
        "best_ac": best_ac,
        "max_xl": max_xl,
        "stall_count": len(stalls),
        "stall_functions": list(stall_fns),
        "inflection_function": inflection_fn,
        "pre_inflection_function": pre_inflection_fn,
        "unique_tiles_d1": (
            len({(t.get("y", 0), t.get("x", 0)) for t in d1_ticks}) if d1_ticks else 0
        ),
        "stairs_down_found": stairs_found,
        "stagnation_ratio": round(stagnation_ratio, 4),
        "recurrence_rate": round(recurrence_rate, 4),
        "last_progress_turn": last_progress_turn,
        "stagnation_turns": stagnation_turns,
        "top_stagnation_fn": top_stagnation_fn,
        "stagnation_fn_turns": stagnation_fn_turns,
    }
    return summary_digest, game_log, milestones


def mark_outlier_good_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Computes statistical progression scores across a batch of runs and sets:
    - `outlier_z_score` (float)
    - `batch_percentile` (float, 0..100)
    - `is_outlier_good` (bool): True if the run is a genuine statistical outlier for how well it did.
    """
    n = len(runs)
    if n == 0:
        return runs

    # Composite progression index: prioritizes max_depth, then xp_level, then AC improvement, then score, then turns survived
    scores = np.array(
        [
            float(r.get("max_depth", 1)) * 100.0
            + float(r.get("xp_level", 1)) * 15.0
            + max(0.0, 10.0 - float(r.get("final_ac", 10))) * 4.0
            + float(r.get("score", 0)) * 0.05
            + min(float(r.get("turns", 0)), 2000.0) * 0.005
            for r in runs
        ],
        dtype=np.float64,
    )

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    q75 = float(np.percentile(scores, 75))
    q25 = float(np.percentile(scores, 25))
    iqr = q75 - q25
    p90 = float(np.percentile(scores, 90))
    max_s = float(np.max(scores))

    # Order ranks for percentiles
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / float(n) * 100.0

    for idx, r in enumerate(runs):
        s = float(scores[idx])
        z = (s - mean_s) / std_s if std_s > 1e-6 else 0.0
        pct = float(ranks[idx])
        r["outlier_z_score"] = round(z, 3)
        r["outlier_zscore"] = round(z, 3)
        r["batch_percentile"] = round(pct, 1)

        # Must be a genuine statistical exception above the batch mean/median:
        # Either Z >= 1.65 (top ~5%), OR > Q75 + 1.5*IQR, OR in top 10% with strictly higher score than median
        is_outlier = bool(
            std_s > 1e-6
            and s > mean_s
            and (
                z >= 1.65
                or (iqr > 0 and s >= q75 + 1.5 * iqr)
                or (s >= p90 and s == max_s)
            )
        )
        r["is_outlier_good"] = is_outlier

    return runs


def compute_function_stats(
    batch_id: str,
    generation_or_runs: int | list[dict[str, Any]],
    runs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Computes per-function telemetry and Ochiai / Liblit statistical correlation
    with `is_outlier_good` runs vs failed/normal runs.
    """
    if isinstance(generation_or_runs, list) and runs is None:
        generation = 0
        runs = generation_or_runs
    else:
        generation = int(generation_or_runs)  # type: ignore[arg-type]
        runs = runs or []

    total_runs = len(runs)
    if total_runs == 0:
        return []

    good_runs = [r for r in runs if r.get("is_outlier_good")]
    n_good = len(good_runs)

    all_fns: set[str] = set()
    parsed_trig: list[dict[str, int]] = []
    parsed_turns: list[dict[str, int]] = []

    batch_total_stagnation_turns = 0
    fn_stagnation_turns_map: Counter[str] = Counter()

    for r in runs:
        raw_t = r.get("triggered_functions") or r.get("functions_triggered") or "{}"
        t_dict = json.loads(raw_t) if isinstance(raw_t, str) else dict(raw_t or {})
        u_dict = r.get("fn_turns") or r.get("function_turns") or {}
        if isinstance(u_dict, str):
            u_dict = json.loads(u_dict)
        parsed_trig.append(t_dict)
        parsed_turns.append(u_dict)
        all_fns.update(t_dict.keys())

        ms = r.get("milestones") or {}
        if isinstance(ms, str):
            try:
                ms = json.loads(ms)
            except Exception:
                ms = {}
        s_dict = ms.get("stagnation_fn_turns") or {}
        for f_name, t_cnt in s_dict.items():
            fn_stagnation_turns_map[f_name] += int(t_cnt)
            batch_total_stagnation_turns += int(t_cnt)

    stats_rows: list[dict[str, Any]] = []
    for fn in sorted(all_fns):
        eps_trig = 0
        total_calls = 0
        total_turns = 0
        depths_when_trig: list[int] = []
        deaths_while_active = 0

        e_good = 0  # outlier good runs where fn executed
        e_bad = 0  # non-outlier runs where fn executed

        for idx, r in enumerate(runs):
            c = int(parsed_trig[idx].get(fn, 0))
            t = int(parsed_turns[idx].get(fn, 0))
            if c > 0:
                eps_trig += 1
                total_calls += c
                total_turns += t
                depths_when_trig.append(int(r.get("max_depth", 1)))
                if r.get("is_outlier_good"):
                    e_good += 1
                else:
                    e_bad += 1
                ms = r.get("milestones") or {}
                if isinstance(ms, str):
                    try:
                        ms = json.loads(ms)
                    except Exception:
                        ms = {}
                pre_inf = r.get("pre_inflection_function") or ms.get(
                    "pre_inflection_function"
                )
                st_fns = r.get("stall_functions") or ms.get("stall_functions") or []
                last_fn = r.get("last_active_function")
                end_clean = str(r.get("end_reason", "")).strip().lower()
                is_success = bool(
                    r.get("is_success")
                    or r.get("is_outlier_good")
                    or end_clean in ("ascended", "success", "completed", "won")
                )
                is_failed = not is_success
                if is_failed:
                    stag_ratio = float(ms.get("stagnation_ratio", 0.0))
                    top_stag_fn = str(ms.get("top_stagnation_fn", ""))
                    if stag_ratio >= 0.50 and top_stag_fn:
                        if fn == top_stag_fn:
                            deaths_while_active += 1
                    else:
                        if (
                            fn == pre_inf
                            or fn in st_fns
                            or (
                                not pre_inf
                                and not st_fns
                                and (last_fn == fn or last_fn is None)
                            )
                        ):
                            deaths_while_active += 1

        trig_rate = eps_trig / float(total_runs)
        avg_d = float(np.mean(depths_when_trig)) if depths_when_trig else 0.0
        death_rate = deaths_while_active / float(max(1, eps_trig))

        # Depth-stratified progression rate
        depth_progression_gains: list[float] = []
        max_d_seen = max((int(r.get("max_depth", 1)) for r in runs), default=1)
        for d in range(1, max(6, max_d_seen + 1)):
            at_d = [r for r in runs if int(r.get("max_depth", 1)) >= d]
            if not at_d:
                break
            adv_base = sum(1 for r in at_d if int(r.get("max_depth", 1)) > d) / float(
                len(at_d)
            )
            fn_at_d = [
                r
                for idx_r, r in enumerate(runs)
                if int(r.get("max_depth", 1)) >= d
                and int(parsed_trig[idx_r].get(fn, 0)) > 0
            ]
            if fn_at_d:
                adv_fn = sum(
                    1 for r in fn_at_d if int(r.get("max_depth", 1)) > d
                ) / float(len(fn_at_d))
                depth_progression_gains.append(adv_fn - adv_base)

        prog_bonus = (
            float(np.mean(depth_progression_gains)) if depth_progression_gains else 0.0
        )

        # Ochiai similarity with within-depth progression weighting
        denom_good = math.sqrt(float(max(1, n_good) * max(1, eps_trig)))
        base_ochiai = e_good / denom_good if n_good > 0 and eps_trig > 0 else 0.0
        if base_ochiai >= 1.0:
            ochiai_good = 1.0
        else:
            ochiai_good = (
                max(
                    0.0,
                    min(1.0, base_ochiai * 0.7 + max(0.0, prog_bonus + 0.5) * 0.3),
                )
                if n_good > 0
                else max(0.0, min(1.0, 0.5 + prog_bonus))
            )

        # Liblit failure importance (dies while active vs overall active rate)
        liblit_fail = (
            death_rate
            * math.log(deaths_while_active + 1.0)
            / math.log(total_runs + 1.0)
        )

        fn_stag_turns = fn_stagnation_turns_map.get(fn, 0)
        stag_share = (
            fn_stag_turns / float(max(1, batch_total_stagnation_turns))
            if batch_total_stagnation_turns > 0
            else 0.0
        )
        livelock_score = (
            (fn_stag_turns / float(max(1, total_turns))) * trig_rate
            if total_turns > 0
            else 0.0
        )

        stats_rows.append(
            {
                "batch_id": batch_id,
                "generation": generation,
                "function_name": fn,
                "episodes_triggered": eps_trig,
                "trigger_rate": round(trig_rate, 4),
                "total_calls": total_calls,
                "total_turns_active": total_turns,
                "avg_depth_when_triggered": round(avg_d, 2),
                "death_rate_while_active": round(death_rate, 4),
                "ochiai_good_run_score": round(ochiai_good, 4),
                "ochiai_good_score": round(ochiai_good, 4),
                "liblit_failure_score": round(liblit_fail, 4),
                "liblit_fail_importance": round(liblit_fail, 4),
                "stagnation_turns": fn_stag_turns,
                "stagnation_share": round(stag_share, 4),
                "livelock_score": round(livelock_score, 4),
            }
        )

    return stats_rows


def build_batch_contrast_digest(
    batch_stats: dict[str, Any],
    runs: list[dict[str, Any]],
    fn_stats: list[dict[str, Any]],
) -> str:
    """
    Builds a high-signal markdown digest contrasting `is_outlier_good = True` runs
    against median/normal runs in the batch so the Author Agent immediately sees what worked.
    """
    if not runs:
        return "No runs recorded yet."

    outliers = [r for r in runs if r.get("is_outlier_good")]
    normals = [r for r in runs if not r.get("is_outlier_good")]

    out_d = (
        float(np.mean([r.get("max_depth", 1) for r in outliers])) if outliers else 0.0
    )
    norm_d = (
        float(np.mean([r.get("max_depth", 1) for r in normals])) if normals else 0.0
    )
    out_s = float(np.mean([r.get("score", 0) for r in outliers])) if outliers else 0.0
    norm_s = float(np.mean([r.get("score", 0) for r in normals])) if normals else 0.0
    out_t = float(np.mean([r.get("turns", 0) for r in outliers])) if outliers else 0.0
    norm_t = float(np.mean([r.get("turns", 0) for r in normals])) if normals else 0.0

    lines = [
        f"### Batch Overview (`{batch_stats.get('batch_id', 'latest')}`):",
        f"- **Episodes Recorded**: {len(runs)} (Discarded untriggered: {batch_stats.get('episodes_discarded', 0)})",
        f"- **Overall Depth**: Avg={batch_stats.get('avg_depth', 0.0):.2f} | P90={batch_stats.get('p90_depth', 0.0):.1f} | Max={batch_stats.get('max_depth', 1)}",
        f"- **Outlier Good Runs (`is_outlier_good = TRUE`, count={len(outliers)}) vs Normal Runs (count={len(normals)})**:",
        f"  * Outlier Good Runs: Avg MaxDepth={out_d:.2f}, Avg Score={out_s:.1f}, Avg Turns={out_t:.1f}",
        f"  * Normal Runs:       Avg MaxDepth={norm_d:.2f}, Avg Score={norm_s:.1f}, Avg Turns={norm_t:.1f}",
    ]

    if fn_stats:
        lines.append("\n### Function Execution & Correlation Table (`function_stats`):")
        lines.append(
            "| Function | Trigger Rate | Total Turns | Avg Depth | Death Rate Active | Good-Run Ochiai | Fail Score |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fs in sorted(
            fn_stats, key=lambda x: x["ochiai_good_run_score"], reverse=True
        )[:12]:
            lines.append(
                f"| `{fs['function_name']}` | {fs['trigger_rate'] * 100:.1f}% | {fs['total_turns_active']:,} | "
                f"{fs['avg_depth_when_triggered']:.2f} | {fs['death_rate_while_active'] * 100:.1f}% | "
                f"**{fs['ochiai_good_run_score']:.3f}** | {fs['liblit_failure_score']:.3f} |"
            )

    # Top death reasons
    end_counts = Counter(str(r.get("end_reason", "ended")) for r in runs)
    top_ends = ", ".join(
        f"{k}: {v} ({v * 100.0 / len(runs):.1f}%)" for k, v in end_counts.most_common(5)
    )
    lines.append(f"\n- **Termination Breakdown**: {top_ends}")

    # Dynamic failure telemetry & environmental diagnostics
    max_d = int(batch_stats.get("max_depth", 1))
    norm_d = float(batch_stats.get("avg_depth", 1.0))
    max_steps_runs = sum(1 for r in runs if r.get("end_reason") == "max_steps")
    max_steps_rate = max_steps_runs / float(len(runs)) if runs else 0.0

    # Aggregate environmental stall causes directly from summary_digest
    stall_causes: Counter[str] = Counter()
    for r in runs:
        digest = str(r.get("summary_digest", ""))
        for line in digest.splitlines():
            if line.startswith("Stalls:"):
                for m in re.finditer(r"\[(.*?)\]", line):
                    cause = m.group(1).strip()
                    if cause:
                        stall_causes[cause] += 1

    lines.append(
        "\n### 📊 Dynamic Failure Telemetry & Frontier Bottleneck Diagnostics:\n"
        f"- **Step Truncation Rate**: {max_steps_rate * 100:.1f}% ({max_steps_runs}/{len(runs)} runs reached step limit without advancing past Dlvl {max_d})."
    )
    top_killers = Counter(
        str(r.get("killer_name") or r.get("end_reason", "ended"))
        for r in runs
        if r.get("end_reason") == "died"
    ).most_common(5)
    if top_killers:
        killer_items = [
            f"{k}: {count} ({count * 100.0 / len(runs):.1f}%)"
            for k, count in top_killers
        ]
        lines.append(f"- **Top Lethal Monsters/Hazards**: {'; '.join(killer_items)}")
    if stall_causes:
        stall_items = [
            f"**{cause}**: {count} runs ({count * 100.0 / len(runs):.1f}%)"
            for cause, count in stall_causes.most_common(5)
        ]
        lines.append(f"- **Top Environmental Stalls**: {'; '.join(stall_items)}")

    # Unique tiles explored
    tiles_seen = [
        r.get("unique_tiles") for r in runs if r.get("unique_tiles") is not None
    ]
    if tiles_seen:
        lines.append(
            f"- **Exploration Tile Coverage**: Avg {np.mean(tiles_seen):.1f} unique tiles mapped per run."
        )

    # Stairs down discovery rate
    stairs_seen = sum(
        1
        for r in runs
        if r.get("stairs_down_found")
        or "descend the stairs" in str(r.get("summary_digest", "")).lower()
    )
    lines.append(
        f"- **Down Stairs (`>`) Discovery Rate**: {stairs_seen * 100.0 / len(runs):.1f}% ({stairs_seen}/{len(runs)} runs found or descended stairs)."
    )

    # Domain-agnostic chronic stagnation & livelock sink telemetry
    chronic_runs = 0
    recurrence_values: list[float] = []
    for r in runs:
        ms = r.get("milestones") or {}
        if isinstance(ms, str):
            try:
                ms = json.loads(ms)
            except Exception:
                ms = {}
        s_ratio = float(ms.get("stagnation_ratio", 0.0))
        rec_r = float(ms.get("recurrence_rate", 0.0))
        if s_ratio >= 0.50:
            chronic_runs += 1
        if rec_r > 0.0:
            recurrence_values.append(rec_r)

    chronic_rate = chronic_runs / float(max(1, len(runs)))
    top_livelock_fns = sorted(
        [fs for fs in fn_stats if fs.get("stagnation_turns", 0) > 0],
        key=lambda x: x.get("stagnation_turns", 0),
        reverse=True,
    )[:3]

    if chronic_rate >= 0.20 or top_livelock_fns:
        avg_rec = float(np.mean(recurrence_values)) if recurrence_values else 0.0
        lines.append(
            "\n### 🚨 Frontier Bottleneck & Root-Cause Telemetry:\n"
            f"- **Chronic Stagnation Rate**: {chronic_rate * 100:.1f}% ({chronic_runs}/{len(runs)} runs spent >50% of runtime stalled after their last milestone).\n"
            f"- **State Recurrence Rate during Stagnation**: {avg_rec:.2f} (values > 0.85 indicate compact limit cycles / oscillation)."
        )
        if top_livelock_fns:
            lines.append(
                "- **Livelock Sinks (Functions consuming turns during zero-progress stagnation)**:"
            )
            for l_fs in top_livelock_fns:
                lines.append(
                    f"  * `{l_fs['function_name']}`: {l_fs.get('stagnation_turns', 0):,} stagnation turns "
                    f"({l_fs.get('stagnation_share', 0.0) * 100:.1f}% of stalled time in batch, livelock_score={l_fs.get('livelock_score', 0.0):.3f})"
                )
            lines.append(
                "- **Universal Causal Invariant for Author Agent**:\n"
                "  * Chronic failures (high stagnation ratio) are CAUSED by the Livelock Sink above, NOT by the final death message.\n"
                "  * Do NOT patch secondary resource consumers (e.g. food/health regeneration). You MUST modify the Livelock Sink to break the zero-progress cycle!"
            )

    # Sample top 2 Outlier Good Run summaries right in the digest
    if outliers:
        top_outliers = sorted(
            outliers, key=lambda x: x.get("outlier_z_score", 0.0), reverse=True
        )[:2]
        lines.append(
            "\n### Sample Statistical Outlier Good Runs (`is_outlier_good = TRUE`):"
        )
        for r in top_outliers:
            lines.append(
                f"#### Outlier Run `{r['run_id']}` (Seed={r['seed']}, Steps={r.get('steps', 0)}, Z={r['outlier_z_score']:+.2f}, Pctl={r['batch_percentile']:.1f}%):\n"
                f"```text\n{r.get('summary_digest', '')}\n```"
            )

    # Sample stalled/normal runs for replay & diagnosis
    if normals:
        target_d = max(1, int(round(norm_d)))
        frontier_samples = [
            r for r in normals if int(r.get("max_depth", 1)) >= target_d
        ][:2]
        stalled_samples = frontier_samples or normals[:2]
        lines.append(
            f"\n### Sample Frontier Failure/Stall Runs (Dlvl {target_d} Bottleneck for Prefix Replay / Diagnosis):"
        )
        for r in stalled_samples:
            lines.append(
                f"#### Run `{r['run_id']}` (Seed={r['seed']}, Steps={r.get('steps', 0)}, Depth={r.get('max_depth', 1)}, End={r.get('end_reason', 'ended')}):\n"
                f"```text\n{r.get('summary_digest', '')}\n```\n"
                f'*Replay Hint: `dry_run_policy(code=..., replay_run_id="{r["run_id"]}", replay_steps={max(0, int(r.get("steps", 0)) - 25)})`*'
            )

    return "\n".join(lines)
