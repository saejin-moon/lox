"""
Unit tests for ParquetLogger and DuckDBConsolidator telemetry pipeline.
"""

import os
import tempfile
import time
import pytest
import duckdb

from corp.telemetry.parquet_logger import ParquetLogger, TickRecord, EpisodeRecord
from corp.telemetry.duckdb_consolidator import DuckDBConsolidator


def test_parquet_logger_and_duckdb_consolidation():
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_dir = os.path.join(tmpdir, "parquet")
        duckdb_path = os.path.join(tmpdir, "telemetry.duckdb")

        logger = ParquetLogger(
            base_dir=parquet_dir,
            run_id="test_run_1",
            tick_batch_size=5,  # Small batch for testing
        )

        ep_id = "test_ep_001"
        # Log 10 ticks (should trigger 2 flushes)
        for i in range(10):
            tick = TickRecord(
                episode_id=ep_id,
                run_id="test_run_1",
                step=i + 1,
                turn=i + 1,
                timestamp=time.time(),
                depth=1,
                dungeon_number=0,
                level_number=1,
                x=10 + i,
                y=10,
                hp=16,
                max_hp=16,
                energy=5,
                max_energy=5,
                ac=10,
                xp_level=1,
                xp_points=0,
                gold=0,
                hunger_state=1,
                encumbrance=0,
                condition_bits=0,
                predicate_mask=0,
                active_nogoods_count=5,
                action_name="STEP" if i % 2 == 0 else "SEARCH",
                action_index=1 if i % 2 == 0 else 75,
                action_key_char="l" if i % 2 == 0 else "s",
                action_args_json="{}",
                decision_latency_us=45.2 + i,
                visible_hostiles_count=0,
                closest_monster_name="",
                closest_monster_dist=-1,
                closest_monster_speed=0,
                closest_monster_threat=0.0,
                cycle_detected=False,
                llm_invoked=False,
                reward=0.0,
            )
            logger.log_tick(tick)

        # Log episode summary
        ep = EpisodeRecord(
            episode_id=ep_id,
            run_id="test_run_1",
            eval_type="fast_prelim",
            seed=42,
            mode="random",
            character="valkyrie",
            role="valkyrie",
            race="human",
            gender="female",
            alignment="lawful",
            persona_resilience=0.92,
            persona_ranged=0.25,
            persona_mana=0.10,
            persona_stealth=0.30,
            persona_alignment=1.0,
            total_turns=10,
            total_steps=10,
            max_depth=1,
            final_score=15,
            final_hp=16,
            final_gold=0,
            is_ascended=False,
            death_message="Survived",
            death_category="none",
            nogoods_active=5,
            nogoods_synthesized=0,
            mean_sps=520.0,
            wall_duration_sec=0.02,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        logger.log_episode(ep)
        logger.flush_all()

        # Check that parquet files were created
        ticks_files = os.listdir(os.path.join(parquet_dir, "ticks"))
        episodes_files = os.listdir(os.path.join(parquet_dir, "episodes"))
        assert len(ticks_files) >= 1
        assert len(episodes_files) >= 1

        # Consolidate into DuckDB
        consolidator = DuckDBConsolidator(
            db_path=duckdb_path,
            parquet_dir=parquet_dir,
        )
        counts = consolidator.consolidate()
        assert counts["episodes"] == 1
        assert counts["ticks"] == 10

        # Query views
        summary_rows = consolidator.query("SELECT run_id, eval_type, episodes_count, median_depth, avg_sps FROM v_eval_summary")
        assert len(summary_rows) == 1
        assert summary_rows[0][0] == "test_run_1"
        assert summary_rows[0][2] == 1  # 1 episode

        latency_rows = consolidator.query("SELECT action_name, action_count, p50_latency_us FROM v_latency_stats")
        assert len(latency_rows) == 2  # STEP and SEARCH
