import os
import duckdb
import pytest
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.consolidator import consolidate_run


def test_parquet_and_duckdb_consolidation(tmp_path):
    run_id = "test_run_123"
    telem_dir = str(tmp_path / "telemetry")
    db_path = str(tmp_path / "test.duckdb")

    logger = ParquetLogger(run_id=run_id, base_dir=telem_dir, flush_interval=10)

    # Log 25 ticks
    for t in range(25):
        logger.log_tick(
            episode_id="ep_01",
            turn=t,
            depth=1,
            hp=16,
            max_hp=16,
            hunger="NORMAL",
            y=5,
            x=10,
            action="step_to_frontier",
            message="You move east.",
            reward=0.0,
        )

    # Log episode summary
    logger.log_episode(
        episode_id="ep_01",
        depth=1,
        score=100,
        turns=25,
        death_reason="Survived",
        solved=True,
        wall_sec=0.15,
    )

    # Log an event
    logger.log_event(
        episode_id="ep_01",
        turn=10,
        depth=1,
        event_type="combat_kill",
        message="You kill the jackal!",
        details="jackal",
    )

    logger.close()

    # Raw parquet directory should exist
    run_dir = os.path.join(telem_dir, run_id)
    assert os.path.exists(run_dir)

    # Consolidate into DuckDB and cleanup
    stats = consolidate_run(run_id=run_id, db_path=db_path, telemetry_dir=telem_dir, cleanup=True)
    assert stats["ticks_added"] == 25
    assert stats["episodes_added"] == 1
    assert stats["events_added"] == 1
    assert stats["cleaned_up"] is True

    # Raw parquet directory should be deleted
    assert not os.path.exists(run_dir)

    # Query DuckDB
    con = duckdb.connect(db_path, read_only=True)
    ticks_count = con.execute("SELECT COUNT(*) FROM ticks WHERE run_id = ?", [run_id]).fetchone()[0]
    ep_count = con.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ?", [run_id]).fetchone()[0]
    ev_count = con.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", [run_id]).fetchone()[0]
    con.close()

    assert ticks_count == 25
    assert ep_count == 1
    assert ev_count == 1
