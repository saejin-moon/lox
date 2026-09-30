#!/usr/bin/env python3
"""
LOX 2.0 High-Volume Telemetry Collector.
Runs multi-episode, multi-role NetHack campaigns to amass comprehensive telemetry
in data/lox.duckdb for deep empirical analysis by the LLM Author Agent.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_nethack import run_nethack_eval
from lox.author.tools import DuckDBToolRegistry
from lox.telemetry.tokens import get_token_usage_summary


def collect_campaign(
    roles: list[str],
    episodes_per_role: int = 10,
    max_turns: int = 500,
    db_path: str = "data/lox.duckdb",
) -> None:
    print("=" * 70)
    print("LOX 2.0 High-Volume Telemetry Data Collection Campaign")
    print(f"Target Roles: {roles}")
    print(f"Episodes per role: {episodes_per_role} (Total: {len(roles) * episodes_per_role})")
    print(f"Max turns per episode: {max_turns}")
    print(f"Database Target: {db_path}")
    print("=" * 70)

    t0 = time.perf_counter()

    for idx, role in enumerate(roles):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"collect_{role}_{ts}"
        print(f"\n[{idx+1}/{len(roles)}] Launching collection batch for role: {role.upper()} (Run ID: {run_id})...")
        run_nethack_eval(
            episodes=episodes_per_role,
            max_turns=max_turns,
            role=role,
            run_id=run_id,
            db_path=db_path,
            record_telemetry=True,
        )

    wall_total = time.perf_counter() - t0

    # Print comprehensive DuckDB inspection using the Author Agent's analytical tools
    print("\n" + "=" * 70)
    print("CAMPAIGN COMPLETED: DuckDB Telemetry Audit & Inspection")
    print("=" * 70)

    registry = DuckDBToolRegistry(db_path=db_path)
    print("\n" + registry.get_duckdb_schema())
    print("\n" + registry.get_death_taxonomy(window=10))
    print("\n" + registry.get_floor_pacing_stats(depth=1))
    print("\n" + registry.get_action_distribution())

    token_stats = get_token_usage_summary(db_path=db_path)
    print("\n### Cumulative Token Spend:")
    print(f"- Total LLM Calls:      {token_stats['total_calls']}")
    print(f"- Total Prompt Tokens:  {token_stats['prompt_tokens']:,}")
    print(f"- Total Output Tokens:  {token_stats['completion_tokens']:,}")
    print(f"- Total Tokens:         {token_stats['total_tokens']:,}")
    print(f"- Cumulative Cost:      ${token_stats['total_cost_usd']:.4f} USD")
    print(f"- Total Campaign Wall:  {wall_total:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="valkyrie,barbarian", help="Comma-separated roles")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per role")
    parser.add_argument("--max-turns", type=int, default=400, help="Max turns per episode")
    parser.add_argument("--db-path", default="data/lox.duckdb", help="Path to DuckDB database")
    args = parser.parse_args()

    role_list = [r.strip().lower() for r in args.roles.split(",") if r.strip()]
    collect_campaign(
        roles=role_list,
        episodes_per_role=args.episodes,
        max_turns=args.max_turns,
        db_path=args.db_path,
    )
