"""
Consolidates Parquet telemetry partitions into DuckDB and prints analytical reports.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lox.telemetry import DuckDBConsolidator


def main():
    parser = argparse.ArgumentParser(description="DuckDB Telemetry Consolidator")
    parser.add_argument("--db-path", default="data/lox_telemetry.duckdb", help="Path to DuckDB database")
    parser.add_argument("--parquet-dir", default="logs/parquet", help="Parquet partition directory")
    args = parser.parse_args()

    print("=" * 80)
    print("LOX-ψ DuckDB Telemetry Consolidator")
    print(f"Database: {args.db_path} | Source: {args.parquet_dir}")
    print("=" * 80)

    consolidator = DuckDBConsolidator(db_path=args.db_path, parquet_dir=args.parquet_dir)
    counts = consolidator.consolidate()
    print(f"Consolidation complete: {counts['episodes']} total episodes, {counts['ticks']} total ticks.\n")

    print("1. Evaluation Summary View (v_eval_summary):")
    print("-" * 80)
    try:
        print(consolidator.query_formatted("SELECT * FROM v_eval_summary LIMIT 10"))
    except Exception as e:
        print(f"Error querying v_eval_summary: {e}")

    print("\n2. Persona Performance Matrix (v_persona_performance):")
    print("-" * 80)
    try:
        print(consolidator.query_formatted("SELECT * FROM v_persona_performance LIMIT 10"))
    except Exception as e:
        print(f"Error querying v_persona_performance: {e}")

    print("\n3. Lethal Taxonomy & Root Causes (v_lethal_taxonomy):")
    print("-" * 80)
    try:
        print(consolidator.query_formatted("SELECT * FROM v_lethal_taxonomy LIMIT 10"))
    except Exception as e:
        print(f"Error querying v_lethal_taxonomy: {e}")

    print("\n4. Decision Loop Latency Stats (v_latency_stats):")
    print("-" * 80)
    try:
        print(consolidator.query_formatted("SELECT * FROM v_latency_stats LIMIT 10"))
    except Exception as e:
        print(f"Error querying v_latency_stats: {e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
