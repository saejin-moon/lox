#!/usr/bin/env python3
"""
Morning / Evening check-in CLI for ongoing multi-week LOX campaigns.
Safely queries DuckDB in read-only mode while `run_swarm.py` is actively evaluating.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.db.schema import (
    DEFAULT_DB_PATH,
    NON_CHAMPION_BATCH_MODES,
    NON_SHARD_RUNS_SQL,
    connect_db,
)


def show_status(db_path: str = DEFAULT_DB_PATH, since_hours: float = 12.0) -> None:
    now = datetime.datetime.now()
    since_dt = now - datetime.timedelta(hours=since_hours)

    print("=" * 72)
    print(" 🛡️  LOX CAMPAIGN CHECK-IN DASHBOARD")
    print(f" Report Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f" Window: Last {since_hours:.1f} hours (since {since_dt.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    print(f" Database: {db_path}")
    print("=" * 72)

    status_md = os.path.join(os.path.dirname(db_path), "workspace", "STATUS.md")
    if os.path.exists(status_md):
        print(f"\n[Latest STATUS.md from {status_md}]:\n")
        try:
            with open(status_md, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Print first 40 lines
                lines = content.splitlines()
                print("\n".join(lines[:45]))
                if len(lines) > 45:
                    print(f"... ({len(lines) - 45} more lines in `{status_md}`)")
        except Exception as e:
            print(f"(Could not read STATUS.md: {e})")
        print("\n" + "-" * 72)

    if not os.path.exists(db_path):
        print(f"\nNo DuckDB file found at `{db_path}`.")
        return

    try:
        con = connect_db(db_path, read_only=True)
    except Exception as exc:
        print(f"\nCould not open DuckDB in read-only mode: {exc}")
        return

    try:
        # Window stats (exclude non-champion modes using canonical list from lox.db.schema)
        _excluded_modes = ", ".join(f"'{m}'" for m in NON_CHAMPION_BATCH_MODES)
        _valid_mode_sql = f"UPPER(COALESCE(mode, '')) NOT IN ({_excluded_modes})"
        batch_row = con.execute(
            f"""
            SELECT COUNT(*), MIN(generation), MAX(generation),
                   AVG(avg_depth), MAX(max_depth)
            FROM batches
            WHERE created_at >= ? AND {_valid_mode_sql}
            """,
            (since_dt,),
        ).fetchone()

        win_count, min_gen, max_gen, win_avg_d, win_max_d = batch_row or (
            0,
            None,
            None,
            0.0,
            0,
        )
        win_count = int(win_count or 0)
        win_avg_d = float(win_avg_d or 0.0)
        win_max_d = int(win_max_d or 0)

        prom_count = (
            con.execute(
                f"""
            SELECT COUNT(*)
            FROM batches
            WHERE hypothesis_passed = TRUE AND created_at >= ? AND {_valid_mode_sql}
            """,
                (since_dt,),
            ).fetchone()[0]
            or 0
        )

        total_batches = (
            con.execute(
                f"SELECT COUNT(*) FROM batches WHERE {_valid_mode_sql}"
            ).fetchone()[0]
            or 0
        )
        total_runs = (
            con.execute(
                f"SELECT COUNT(*) FROM runs WHERE {NON_SHARD_RUNS_SQL}"
            ).fetchone()[0]
            or 0
        )
        cost_row = con.execute(
            f"""
            SELECT COALESCE(SUM(total_cost_usd), 0.0),
                   COALESCE(SUM(total_llm_tokens), 0),
                   COALESCE(AVG(avg_score), 0.0)
            FROM batches
            WHERE created_at >= ? AND {_valid_mode_sql}
            """,
            (since_dt,),
        ).fetchone()
        win_cost, win_tokens, win_avg_score = cost_row or (0.0, 0, 0.0)

        print("\n📊 RECENT ACTIVITY & DEPTH VELOCITY:")
        print(
            f"  • Generations Run: {win_count} (Gen {min_gen or 0} -> {max_gen or 0}) | Total All-Time: {total_batches}"
        )
        print(f"  • Promoted Changes: {prom_count} in window")
        print(
            f"  • Window Avg Depth: {win_avg_d:.2f} | Max Depth: {win_max_d} | Avg Score: {float(win_avg_score or 0.0):.1f}"
        )
        print(
            f"  • Window LLM Spend: ${float(win_cost or 0.0):.4f} ({int(win_tokens or 0):,} tokens) | Total Episodes: {total_runs:,}"
        )

        # Recent discoveries
        try:
            disc_cur = con.execute(
                """
                SELECT generation, mode, target_functions, mean_delta, avg_depth, binding_constraint, verified_c_rule
                FROM discoveries
                WHERE created_at >= ?
                ORDER BY generation DESC
                LIMIT 10
                """,
                (since_dt,),
            )
            discs = disc_cur.fetchall()
            if discs:
                print("\n💡 PROMOTED DISCOVERIES IN WINDOW:")
                for d in discs:
                    gen_val, mode_val, tf, delta, depth, constraint, rule = d
                    g_i = int(gen_val) if gen_val is not None else 0
                    d_f = float(delta) if delta is not None else 0.0
                    dep_f = float(depth) if depth is not None else 1.0
                    c_s = str(constraint or "")[:35]
                    r_s = str(rule or "")[:40]
                    print(
                        f"  • Gen {g_i:03d} [{mode_val or ''} on {tf or ''}]: Δ={d_f:+.2f}, Dlvl {dep_f:.1f} | {c_s} | {r_s}"
                    )
            else:
                print("\n💡 PROMOTED DISCOVERIES IN WINDOW: (None in this window)")
        except Exception as disc_exc:
            print(f"\n⚠️ Could not query discoveries: {disc_exc}")

        # Deepest runs in window
        try:
            run_cur = con.execute(
                f"""
                SELECT run_id, seed, max_depth, score, xp_level, final_ac, end_reason
                FROM runs
                WHERE created_at >= ? AND {NON_SHARD_RUNS_SQL}
                ORDER BY max_depth DESC, score DESC
                LIMIT 5
                """,
                (since_dt,),
            )
            deepest = run_cur.fetchall()
            if deepest:
                print("\n🏆 DEEPEST RUNS IN WINDOW:")
                for r in deepest:
                    rid, s, md, sc, xl, ac, reason = r
                    md_s = int(md or 1)
                    sc_s = int(sc or 0)
                    xl_s = int(xl or 1)
                    ac_s = int(ac if ac is not None else 10)
                    print(
                        f"  • `{rid}` (Seed={s}): Dlvl {md_s}, Score {sc_s:,}, XL {xl_s}, AC {ac_s} -> {reason}"
                    )
        except Exception as run_exc:
            print(f"\n⚠️ Could not query deepest runs: {run_exc}")

    finally:
        con.close()
    print("\n" + "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check in on active LOX campaign progression (safe read-only DuckDB inspector)."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=12.0,
        help="Report activity over the last N hours (default: 12.0)",
    )
    args = parser.parse_args()
    show_status(db_path=args.db_path, since_hours=args.since_hours)


if __name__ == "__main__":
    main()
