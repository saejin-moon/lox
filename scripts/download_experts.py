#!/usr/bin/env python3
"""
Downloads and compiles real human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs),
custom `.txt` dumplogs, `.ttyrec`/`.ttyrec.bz2` files (e.g. NLD-AA / NLD-NAO), or direct URLs
into DuckDB's `expert_runs` and `expert_ticks` tables.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.ui.console import ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & compile expert NetHack runs into DuckDB (expert_runs & expert_ticks)."
    )
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH, help="Path to DuckDB database."
    )
    parser.add_argument(
        "--max-ascensions",
        type=int,
        default=50,
        help="Max human ascensions to ingest from archive.alt.org.",
    )
    parser.add_argument(
        "--fetch-dumplogs",
        type=int,
        default=15,
        help="Number of detailed human dumplogs to fetch.",
    )
    parser.add_argument(
        "--dumplog",
        action="append",
        default=[],
        help="Path to a local NetHack .txt dumplog file or directory of .txt dumplogs (repeatable).",
    )
    parser.add_argument(
        "--ttyrec",
        action="append",
        default=[],
        help="Path to a local .ttyrec / .ttyrec.bz2 file or directory (e.g., from NLD-AA / NLD-NAO) (repeatable).",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Direct URL to an online NetHack dumplog .txt file (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if expert_runs already meets --max-ascensions.",
    )
    args = parser.parse_args()

    ui.info(f"Ingesting expert NetHack runs into `{args.db_path}`...")
    res = ingest_expert_runs_into_duckdb(
        db_path=args.db_path,
        max_ascensions=args.max_ascensions,
        fetch_dumplogs=args.fetch_dumplogs,
        dumplog_paths=args.dumplog,
        ttyrec_paths=args.ttyrec,
        urls=args.url,
        force=args.force,
    )
    con = connect_db(args.db_path, read_only=True)
    try:
        n_runs = con.execute("SELECT COUNT(*) FROM expert_runs").fetchone()[0]
        n_ticks = con.execute("SELECT COUNT(*) FROM expert_ticks").fetchone()[0]
    finally:
        con.close()

    ui.info(
        f"Expert corpus ready: {n_runs} runs in `expert_runs` "
        f"(+{res['newly_added']} processed), {n_ticks} rows in `expert_ticks`."
    )


if __name__ == "__main__":
    main()
