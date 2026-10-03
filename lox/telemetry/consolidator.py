"""
LOX 2.0 Telemetry Consolidator.
Consolidates raw partitioned Parquet telemetry files into a single, high-performance DuckDB database.
Safely cleans up raw parquet partition directories post-run.
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Any
import duckdb


def init_db(db_path: str = "data/lox.duckdb") -> duckdb.DuckDBPyConnection:
    """Initializes DuckDB tables and runs migrations if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = duckdb.connect(db_path)

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
            reward FLOAT,
            closest_hostile_name VARCHAR,
            closest_hostile_dist FLOAT,
            hostiles_in_fov INTEGER,
            tile_type VARCHAR,
            dungeon_branch VARCHAR,
            target_y INTEGER,
            target_x INTEGER,
            stairs_down_y INTEGER,
            stairs_down_x INTEGER,
            stairs_down_turn INTEGER,
            tiles_visited_count INTEGER,
            unvisited_frontier_count INTEGER,
            dead_ends_count INTEGER,
            adjacent_monsters VARCHAR,
            ac INTEGER,
            xl INTEGER,
            active_subroutine VARCHAR,
            food_count INTEGER,
            potion_count INTEGER,
            scroll_count INTEGER,
            dagger_count INTEGER,
            weapon_in_hand VARCHAR
        );
        CREATE TABLE IF NOT EXISTS episodes (
            run_id VARCHAR,
            episode_id VARCHAR,
            depth INTEGER,
            score INTEGER,
            turns INTEGER,
            death_reason VARCHAR,
            solved BOOLEAN,
            wall_sec FLOAT,
            role VARCHAR,
            gold INTEGER,
            max_depth INTEGER,
            steps INTEGER,
            attacks INTEGER,
            descents INTEGER,
            searches INTEGER,
            eats INTEGER,
            prayers INTEGER,
            death_category VARCHAR
        );
        CREATE TABLE IF NOT EXISTS events (
            run_id VARCHAR,
            episode_id VARCHAR,
            turn INTEGER,
            depth INTEGER,
            event_type VARCHAR,
            message VARCHAR,
            details VARCHAR
        );
        CREATE TABLE IF NOT EXISTS token_usage (
            run_id VARCHAR,
            session_id VARCHAR,
            timestamp TIMESTAMP,
            provider VARCHAR,
            model VARCHAR,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            estimated_cost_usd FLOAT,
            trigger_reason VARCHAR,
            tools_called VARCHAR
        );
    """)

    # Check and add any missing columns in episodes table if upgraded from earlier schema
    for col, col_type in [
        ("role", "VARCHAR"),
        ("gold", "INTEGER"),
        ("max_depth", "INTEGER"),
        ("steps", "INTEGER"),
        ("attacks", "INTEGER"),
        ("descents", "INTEGER"),
        ("searches", "INTEGER"),
        ("eats", "INTEGER"),
        ("prayers", "INTEGER"),
        ("death_category", "VARCHAR"),
        ("inventory_at_death", "VARCHAR"),
        ("last_5_actions", "VARCHAR"),
        ("turns_dl1", "INTEGER"),
        ("turns_dl2", "INTEGER"),
        ("turns_mines", "INTEGER"),
    ]:
        con.execute(f"ALTER TABLE episodes ADD COLUMN IF NOT EXISTS {col} {col_type};")

    # Check and add any missing columns in ticks table
    for col, col_type in [
        ("closest_hostile_name", "VARCHAR"),
        ("closest_hostile_dist", "FLOAT"),
        ("hostiles_in_fov", "INTEGER"),
        ("tile_type", "VARCHAR"),
        ("dungeon_branch", "VARCHAR"),
        ("target_y", "INTEGER"),
        ("target_x", "INTEGER"),
        ("stairs_down_y", "INTEGER"),
        ("stairs_down_x", "INTEGER"),
        ("stairs_down_turn", "INTEGER"),
        ("tiles_visited_count", "INTEGER"),
        ("unvisited_frontier_count", "INTEGER"),
        ("dead_ends_count", "INTEGER"),
        ("adjacent_monsters", "VARCHAR"),
        ("ac", "INTEGER"),
        ("xl", "INTEGER"),
        ("active_subroutine", "VARCHAR"),
        ("food_count", "INTEGER"),
        ("potion_count", "INTEGER"),
        ("scroll_count", "INTEGER"),
        ("dagger_count", "INTEGER"),
        ("weapon_in_hand", "VARCHAR"),
    ]:
        con.execute(f"ALTER TABLE ticks ADD COLUMN IF NOT EXISTS {col} {col_type};")

    return con


def consolidate_run(
    run_id: str,
    db_path: str = "data/lox.duckdb",
    telemetry_dir: str = "data/telemetry",
    cleanup: bool = True,
) -> dict[str, Any]:
    """
    Consolidates ticks, episode records, and events for a given run_id into DuckDB.
    Deletes the raw parquet partition directory if cleanup=True.
    """
    run_dir = os.path.join(telemetry_dir, run_id)
    if not os.path.exists(run_dir):
        return {
            "run_id": run_id,
            "ticks_added": 0,
            "episodes_added": 0,
            "events_added": 0,
            "cleaned_up": False,
        }

    con = init_db(db_path)

    # Vectorized SQL merge for ticks
    ticks_pattern = os.path.join(run_dir, "ticks_*part_*.parquet")
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

    # Vectorized SQL merge for events
    ev_file = os.path.join(run_dir, "events.parquet")
    events_added = 0
    if os.path.exists(ev_file):
        con.execute(f"INSERT INTO events SELECT * FROM read_parquet('{ev_file}')")
        res = con.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", [run_id]).fetchone()
        events_added = res[0] if res else 0

    con.close()

    # Cleanup raw parquet directory
    if cleanup:
        shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "run_id": run_id,
        "ticks_added": ticks_added,
        "episodes_added": episodes_added,
        "events_added": events_added,
        "cleaned_up": cleanup,
    }
