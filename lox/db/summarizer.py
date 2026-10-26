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

    for t in ticks:
        d = int(t.get("depth", 1))
        turn = int(t.get("turn", 0))
        ac = int(t.get("ac", 10))
        xl = int(t.get("xp_level", 1))
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
        d = int(t.get("depth", 1))
        fn = str(t.get("active_function", "root"))
        act = str(t.get("action_name", "WAIT"))
        hp = int(t.get("hp", 0))
        turn = int(t.get("turn", 0))
        pos = (int(t.get("y", 0)), int(t.get("x", 0)))

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
                    "actions": Counter({act: 1}),
                    "positions": {pos},
                }
            )
        else:
            seg = segments[-1]
            seg["end_turn"] = turn
            seg["steps"] += 1
            seg["end_hp"] = hp
            seg["actions"][act] += 1
            seg["positions"].add(pos)

    # Collapse consecutive micro-segments if there are too many
    rle_lines: list[str] = []
    display_segs = segments if len(segments) <= 14 else (segments[:6] + segments[-8:])
    for seg in display_segs:
        top_acts = ", ".join(f"{k}:{v}" for k, v in seg["actions"].most_common(3))
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
    for seg in segments:
        if seg["steps"] >= 40 and len(seg["positions"]) <= 3:
            top_a = seg["actions"].most_common(2)
            stalls.append(
                f"fn={seg['fn']} on Dlvl {seg['depth']} (T{seg['start_turn']}..{seg['end_turn']}, "
                f"{seg['steps']} steps on {len(seg['positions'])} tiles: {top_a})"
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

    milestones = {
        "max_depth": max(depths_seen) if depths_seen else 1,
        "depths_visited": depths_seen,
        "depth_first_turn": {str(k): v for k, v in depth_first_turn.items()},
        "best_ac": best_ac,
        "max_xl": max_xl,
        "stall_count": len(stalls),
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

    for r in runs:
        raw_t = r.get("triggered_functions") or r.get("functions_triggered") or "{}"
        t_dict = json.loads(raw_t) if isinstance(raw_t, str) else dict(raw_t or {})
        u_dict = r.get("fn_turns") or r.get("function_turns") or {}
        if isinstance(u_dict, str):
            u_dict = json.loads(u_dict)
        parsed_trig.append(t_dict)
        parsed_turns.append(u_dict)
        all_fns.update(t_dict.keys())

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
                last_fn = r.get("last_active_function")
                if (
                    last_fn == fn or (last_fn is None and not r.get("is_outlier_good"))
                ) and r.get("end_reason") != "ascended":
                    deaths_while_active += 1

        trig_rate = eps_trig / float(total_runs)
        avg_d = float(np.mean(depths_when_trig)) if depths_when_trig else 0.0
        death_rate = deaths_while_active / float(max(1, eps_trig))

        # Ochiai similarity with `is_outlier_good = True`
        denom_good = math.sqrt(float(max(1, n_good) * max(1, eps_trig)))
        ochiai_good = e_good / denom_good if n_good > 0 and eps_trig > 0 else 0.0

        # Liblit failure importance (dies while active vs overall active rate)
        liblit_fail = (
            death_rate
            * math.log(deaths_while_active + 1.0)
            / math.log(total_runs + 1.0)
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
                f"#### Outlier Run `{r['run_id']}` (Seed={r['seed']}, Z={r['outlier_z_score']:+.2f}, Pctl={r['batch_percentile']:.1f}%):\n"
                f"```text\n{r.get('summary_digest', '')}\n```"
            )

    return "\n".join(lines)
