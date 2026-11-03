#!/usr/bin/env python3
"""
Morning / Evening check-in CLI for ongoing multi-week LOX campaigns.
Safely queries DuckDB in read-only mode while `run_lox.py` is actively evaluating.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.db.schema import DEFAULT_DB_PATH, connect_db


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
        # Window stats
        batch_row = con.execute(
            """
            SELECT COUNT(*), MIN(generation), MAX(generation),
                   AVG(avg_depth), MAX(max_depth)
            FROM batches
            WHERE created_at >= ?
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
                """
            SELECT COUNT(*)
            FROM batches
            WHERE hypothesis_passed = TRUE AND created_at >= ?
            """,
                (since_dt,),
            ).fetchone()[0]
            or 0
        )

        total_batches = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0] or 0
        total_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] or 0

        print("\n📊 RECENT ACTIVITY & DEPTH VELOCITY:")
        print(
            f"  • Generations Run: {win_count} (Gen {min_gen or 0} -> {max_gen or 0}) | Total All-Time: {total_batches}"
        )
        print(f"  • Promoted Changes: {prom_count} in window")
        print(f"  • Window Avg Depth: {win_avg_d:.2f} | Window Max Depth: {win_max_d}")
        print(f"  • Total Recorded Episodes: {total_runs:,}")

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
                    print(
                        f"  • Gen {gen_val:03d} [{mode_val} on {tf}]: Δ={delta:+.2f}, Dlvl {depth:.1f} | {constraint[:35]} | {rule[:40]}"
                    )
            else:
                print("\n💡 PROMOTED DISCOVERIES IN WINDOW: (None in this window)")
        except Exception:
            pass

        # Deepest runs in window
        try:
            run_cur = con.execute(
                """
                SELECT run_id, seed, max_depth, score, xp_level, final_ac, end_reason
                FROM runs
                WHERE created_at >= ?
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
                    print(
                        f"  • `{rid}` (Seed={s}): Dlvl {md}, Score {sc:,}, XL {xl}, AC {ac} -> {reason}"
                    )
        except Exception:
            pass

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
