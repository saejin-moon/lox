"""
Interactive or CLI DuckDB SQL Query Runner for LOX-ψ Telemetry.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lox.telemetry import DuckDBConsolidator


def main():
    parser = argparse.ArgumentParser(description="Query LOX-ψ DuckDB Telemetry")
    parser.add_argument("query", nargs="?", default=None, help="SQL query to execute")
    parser.add_argument("--db-path", default="data/lox_telemetry.duckdb", help="Path to DuckDB database")
    args = parser.parse_args()

    consolidator = DuckDBConsolidator(db_path=args.db_path)

    if not os.path.exists(args.db_path):
        print(f"Error: Database {args.db_path} does not exist. Run a benchmark first.")
        sys.exit(1)

    if args.query:
        print(consolidator.query_formatted(args.query))
    else:
        print("Interactive DuckDB Query Mode (type 'exit' or 'quit' to exit):")
        while True:
            try:
                sql = input("lox-duckdb> ").strip()
                if not sql:
                    continue
                if sql.lower() in ("exit", "quit", "q"):
                    break
                print(consolidator.query_formatted(sql))
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
