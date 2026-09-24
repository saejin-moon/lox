#!/usr/bin/env python3
"""
Telemetry Maintenance: Consolidate Parquet files into DuckDB and clean up raw files.
Preserves DuckDB database integrity and frees disk space.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import duckdb
from lox.telemetry.duckdb_consolidator import DuckDBConsolidator


def main():
    consolidator = DuckDBConsolidator()
    print("[*] Consolidating pending Parquet files into DuckDB...")
    stats = consolidator.consolidate(clean_parquet=False)
    print(f"    - Total episodes in DuckDB: {stats['episodes']}")
    print(f"    - Total ticks in DuckDB: {stats['ticks']}")
    print(f"    - Total LLM queries in DuckDB: {stats['llm_queries']}")

    print("[*] Safely purging intermediate Parquet files...")
    cleanup = consolidator.clean_parquet_files()
    print(f"    - Deleted files: {cleanup['deleted_files']}")
    print(f"    - Freed space: {cleanup['freed_mb']} MB")

    print("[*] Running DuckDB VACUUM...")
    consolidator.vacuum()
    print("[+] Telemetry maintenance complete!")


if __name__ == "__main__":
    main()
