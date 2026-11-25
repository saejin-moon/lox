"""
Campaign progression reporting utilities (`STATUS.md` dashboard & adaptive step horizons).
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lox.db.schema import NON_SHARD_RUNS_SQL, connect_db


def compute_adaptive_max_steps(
    baseline_runs: list[dict[str, Any]] | float | int, base_max_steps: int = 5000
) -> int:
    """
    Dynamically scales the episode horizon from 5,000 up to 50,000 steps
    matching depth progression tiers to prevent false timeouts on deep runs.
    Accepts either baseline runs or a depth number directly.
    """
    if isinstance(baseline_runs, (int, float)):
        p90_d = float(baseline_runs)
    elif not baseline_runs:
        return base_max_steps
    else:
        clean_depths = [
            int(r["max_depth"]) if r.get("max_depth") is not None else 1
            for r in baseline_runs
            if not r.get("is_shard")
        ]
        if not clean_depths:
            return base_max_steps
        p90_d = float(np.percentile(clean_depths, 90))

    if p90_d < 4.0:
        tier = 5000
    elif p90_d < 8.0:
        tier = 10000
    elif p90_d < 13.0:
        tier = 18000
    elif p90_d <= 20.0:
        tier = 30000
    else:
        tier = 50000
    return int(min(50000, max(base_max_steps, tier)))


def write_status_report(
    workspace_dir: str,
    db_path: str,
    gen: int = 0,
    start_time: float | None = None,
    baseline_runs: list[dict[str, Any]] | None = None,
    active_max_steps: int = 5000,
    current_policy: str = "",
    skills_count: int = 0,
    niches: list[str] | None = None,
) -> None:
    """Atomically writes an updated campaign status report to `STATUS.md`."""
    uptime_sec = max(1.0, time.time() - (start_time or time.time()))
    hours, rem = divmod(int(uptime_sec), 3600)
    mins, secs = divmod(rem, 60)
    days, hours = divmod(hours, 24)
    uptime_str = (
        f"{days}d {hours}h {mins}m {secs}s" if days > 0 else f"{hours}h {mins}m {secs}s"
    )

    con = connect_db(db_path, read_only=True)
    try:
        row_prom = con.execute(
            "SELECT COUNT(*) FROM policies WHERE is_promoted = TRUE"
        ).fetchone()
        total_promotions = (
            int(row_prom[0]) if row_prom and row_prom[0] is not None else 0
        )
    except Exception:
        total_promotions = 0

    gen0_stats: dict[str, Any] = {
        "avg_depth": 1.0,
        "p90_depth": 1.0,
        "max_depth": 1,
        "avg_ac": 10.0,
        "avg_xl": 1.0,
        "ascensions": 0,
    }
    try:
        g0_row = con.execute(
            """
            SELECT avg_depth, p90_depth, max_depth
            FROM batches
            WHERE generation = 0 AND hypothesis_passed = TRUE
            ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
        if g0_row:
            gen0_stats["avg_depth"] = float(g0_row[0]) if g0_row[0] is not None else 1.0
            gen0_stats["p90_depth"] = float(g0_row[1]) if g0_row[1] is not None else 1.0
            gen0_stats["max_depth"] = int(g0_row[2]) if g0_row[2] is not None else 1
    except Exception:
        pass

    clean_runs = [r for r in (baseline_runs or []) if not r.get("is_shard")]
    depths = [
        int(r["max_depth"]) if r.get("max_depth") is not None else 1 for r in clean_runs
    ]
    c_depth = float(np.mean(depths)) if depths else 1.0
    c_p90 = float(np.percentile(depths, 90)) if depths else 1.0
    c_max = int(max(depths)) if depths else 1
    c_ac = (
        float(
            np.mean(
                [
                    int(r["final_ac"]) if r.get("final_ac") is not None else 10
                    for r in clean_runs
                ]
            )
        )
        if clean_runs
        else 10.0
    )
    c_xl = (
        float(
            np.mean(
                [
                    int(r["xp_level"]) if r.get("xp_level") is not None else 1
                    for r in clean_runs
                ]
            )
        )
        if clean_runs
        else 1.0
    )
    c_asc = sum(1 for r in clean_runs if r.get("end_reason") == "ascended")

    discoveries: list[dict[str, Any]] = []
    try:
        cur = con.execute(
            "SELECT * FROM discoveries WHERE mode != 'PILOT' ORDER BY created_at DESC LIMIT 15"
        )
        d_cols = [d[0] for d in (cur.description or [])]
        discoveries = [dict(zip(d_cols, r)) for r in cur.fetchall()]
    except Exception:
        discoveries = []

    deepest_outlier: dict[str, Any] | None = None
    try:
        cur = con.execute(
            f"""
            SELECT run_id, seed, max_depth, xp_level, final_ac, end_reason, summary_digest
            FROM runs
            WHERE is_outlier_good = TRUE AND {NON_SHARD_RUNS_SQL}
            ORDER BY max_depth DESC, score DESC
            LIMIT 1
            """
        )
        r_cols = [d[0] for d in (cur.description or [])]
        row = cur.fetchone()
        if row:
            deepest_outlier = dict(zip(r_cols, row))
    except Exception:
        deepest_outlier = None
    finally:
        con.close()

    pol_lines = len(current_policy.splitlines()) if current_policy else 0
    active_niches = niches or ["peer"]

    lines = [
        "# LOX Campaign Status Dashboard",
        f"**Last Updated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Campaign Uptime**: {uptime_str} | **Current Generation**: {gen} | **Total Promotions**: {total_promotions} | **Active Max Steps**: {active_max_steps:,}",
        "",
        "## Progression Overview",
        "| Metric | Gen 0 Baseline | Current Champion |",
        "|---|---|---|",
        f"| **Avg Depth** | {gen0_stats['avg_depth']:.2f} | **{c_depth:.2f}** |",
        f"| **P90 Depth** | {gen0_stats['p90_depth']:.2f} | **{c_p90:.2f}** |",
        f"| **Max Depth** | {gen0_stats['max_depth']} | **{c_max}** |",
        f"| **Avg AC** | {gen0_stats['avg_ac']:.1f} | **{c_ac:.1f}** |",
        f"| **Avg XL** | {gen0_stats['avg_xl']:.1f} | **{c_xl:.1f}** |",
        f"| **Ascensions** | {gen0_stats['ascensions']} | **{c_asc}** |",
        "",
        "## Promoted Breakthroughs",
    ]
    if discoveries:
        lines.append(
            "| Gen | Mode | Target Functions | Δ | Depth | Binding Constraint | Verified Rule |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for d in discoveries:
            tf = d.get("target_functions") or ""
            if isinstance(tf, str) and tf.startswith("["):
                try:
                    tf = ", ".join(json.loads(tf))
                except Exception:
                    pass
            m_delta = float(d["mean_delta"]) if d.get("mean_delta") is not None else 0.0
            a_depth = float(d["avg_depth"]) if d.get("avg_depth") is not None else 1.0
            lines.append(
                f"| {d.get('generation', 0)} | {d.get('mode') or ''} | `{tf}` | "
                f"{m_delta:+.2f} | Dlvl {a_depth:.1f} | "
                f"{str(d.get('binding_constraint') or '')[:30]} | {str(d.get('verified_c_rule') or '')[:40]} |"
            )
    else:
        lines.append("*(No promoted discoveries recorded yet)*")

    lines.extend(["", "## Deepest Statistical Outlier Run"])
    if deepest_outlier:
        lines.extend(
            [
                f"- **Run ID**: `{deepest_outlier.get('run_id')}` (Seed: {deepest_outlier.get('seed')})",
                f"- **Floor Reached**: Dlvl {deepest_outlier.get('max_depth')}, XL {deepest_outlier.get('xp_level')}, AC {deepest_outlier.get('final_ac')}",
                f"- **End Reason**: {deepest_outlier.get('end_reason')}",
                "```text",
                str(deepest_outlier.get("summary_digest") or "").strip(),
                "```",
            ]
        )
    else:
        lines.append("*(No outlier runs recorded yet)*")

    lines.extend(
        [
            "",
            "## Codebase Health & Diversity Archive",
            f"- `policy.py` lines: **{pol_lines}**",
            f"- `skills/` modules: **{skills_count}**",
            f"- Archive Niches: {', '.join(f'`{n}`' for n in active_niches)}",
        ]
    )

    status_path = Path(workspace_dir) / "STATUS.md"
    tmp_path = Path(workspace_dir) / "STATUS.md.tmp"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, status_path)
