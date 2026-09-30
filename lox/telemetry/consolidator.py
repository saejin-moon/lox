"""
LOX 2.0 Telemetry Consolidator.
Consolidates raw partitioned Parquet telemetry files into a single, high-performance DuckDB database.
Safely cleans up raw parquet partition directories post-run.
"""
from __future__ import annotations

import glob
import os
import shutil
import duckdb


def consolidate_run(
    run_id: str,
    db_path: str = "data/lox.duckdb",
    telemetry_dir: str = "data/telemetry",
    cleanup: bool = True,
) -> dict[str, Any]:
    """
    Consolidates ticks and episode records for a given run_id into DuckDB.
    Deletes the raw parquet partition directory if cleanup=True.
    """
    run_dir = os.path.join(telemetry_dir, run_id)
    if not os.path.exists(run_dir):
        return {"run_id": run_id, "ticks_added": 0, "episodes_added": 0, "cleaned_up": False}

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = duckdb.connect(db_path)

    # Initialize tables if they do not exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            run_id VARCHAR,
            episode_id VARCHAR,
            turn INTEGER,
            depth INTEGER,
            hp INTEGER,
            max_hp INTEGER,
            hunger VARCHAR,
            y INTEGER,
            x INTEGER,
            action VARCHAR,
            message VARCHAR,
            reward FLOAT
        );
        CREATE TABLE IF NOT EXISTS episodes (
            run_id VARCHAR,
            episode_id VARCHAR,
            depth INTEGER,
            score INTEGER,
            turns INTEGER,
            death_reason VARCHAR,
            solved BOOLEAN,
            wall_sec FLOAT
        );
    """)

    # Vectorized SQL merge for ticks
    ticks_pattern = os.path.join(run_dir, "ticks_part_*.parquet")
    ticks_files = glob.glob(ticks_pattern)
    ticks_added = 0
    if ticks_files:
        con.execute(f"INSERT INTO ticks SELECT * FROM read_parquet('{ticks_pattern}')")
        res = con.execute("SELECT COUNT(*) FROM ticks WHERE run_id = ?", [run_id]).fetchone()
        ticks_added = res[0] if res else 0

    # Vectorized SQL merge for episodes
    ep_file = os.path.join(run_dir, "episodes.parquet")
    episodes_added = 0
    if os.path.exists(ep_file):
        con.execute(f"INSERT INTO episodes SELECT * FROM read_parquet('{ep_file}')")
        res = con.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ?", [run_id]).fetchone()
        episodes_added = res[0] if res else 0

    con.close()

    # Cleanup raw parquet directory
    if cleanup:
        shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "run_id": run_id,
        "ticks_added": ticks_added,
        "episodes_added": episodes_added,
        "cleaned_up": cleanup,
    }
