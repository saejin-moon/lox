"""
Telemetry event-stream tests: message classification, parquet event stream,
DuckDB consolidation, and the episode-level emitter.
"""
from __future__ import annotations

import os

import duckdb
import pytest

from lox.telemetry.message_class import classify_message, SIGNIFICANT_CLASSES
from lox.telemetry.parquet_logger import EventRecord, ParquetLogger


class TestMessageClass:
    @pytest.mark.parametrize("msg,expected", [
        ("You die...", "death"),
        ("You faint from lack of food.  The kobold hits!", "hunger"),
        ("The jackal bites!", "damage"),
        ("You pray to Tyr.", "prayer"),
        ("Welcome to experience level 5!", "levelup"),
        ("The shopkeeper says: get out!", "shop"),
        ("You trigger a trap!", "trap"),
        ("You are confused.", "status"),
        ("You descend the staircase.", "stair"),
        ("You pick up a dagger.", "item"),
        ("", None),
        ("  ", None),
        ("Nothing notable happens.", None),
    ])
    def test_classification(self, msg, expected):
        assert classify_message(msg) == expected

    def test_all_classes_significant(self):
        for c in ("death", "hunger", "damage", "prayer", "shop", "trap", "status",
                  "stair", "item", "levelup"):
            assert c in SIGNIFICANT_CLASSES

    def test_death_precedes_damage(self):
        assert classify_message("You die... the kobold bites!") == "death"


class TestParquetEvents:
    def test_events_written(self, tmp_path):
        logger = ParquetLogger(base_dir=str(tmp_path), run_id="r1", tick_batch_size=100)
        logger.log_event(EventRecord("r1", "ep1", 1, 10, 1, "message", "damage", "hit"))
        logger.log_event(EventRecord("r1", "ep1", 2, 11, 1, "anomaly", "BURST_DAMAGE", "big"))
        logger.log_event(EventRecord("r1", "ep1", 3, 12, 1, "message", "hunger", "faint"))
        logger.flush_all()
        files = os.listdir(os.path.join(tmp_path, "events"))
        assert len(files) == 1 and files[0].startswith("events_r1_")
        rows = duckdb.sql(
            f"SELECT kind, code FROM read_parquet('{tmp_path}/events/*.parquet') "
            "ORDER BY step").fetchall()
        assert rows == [("message", "damage"), ("anomaly", "BURST_DAMAGE"),
                        ("message", "hunger")]

    def test_flush_all_includes_events(self, tmp_path):
        logger = ParquetLogger(base_dir=str(tmp_path), run_id="r1", tick_batch_size=100)
        logger.log_event(EventRecord("r1", "ep1", 1, 1, 1, "message", "item", "pickup"))
        logger.flush_all()
        assert os.listdir(os.path.join(tmp_path, "events"))


class TestEventEmission:
    def _agent(self, logger):
        from lox.agent.episode_runner import EpisodeRunnerMixin

        class Dummy(EpisodeRunnerMixin):
            pass

        a = Dummy()
        a.parquet_logger = logger
        a.episode_id = "ep1"
        a.step_counter = 5
        from tests.test_shop_and_dungeon_graph import make_blstats
        a.current_blstats = make_blstats(hp=8, depth=2, turn=100)
        a.max_turn_reached = 100
        a.max_depth_reached = 2
        a.current_message = "The jackal bites!"
        a._last_message_class = None
        a._last_anomaly_code = None
        return a

    def test_message_transition_and_dedupe(self, tmp_path):
        logger = ParquetLogger(base_dir=str(tmp_path), run_id="r1", tick_batch_size=100)
        a = self._agent(logger)
        a._log_events({})
        a._log_events({})                      # same class → deduped
        a.current_message = "You descend the staircase."
        a._log_events({})                      # class transition
        logger.flush_all()
        rows = duckdb.sql(
            f"SELECT kind, code FROM read_parquet('{tmp_path}/events/*.parquet') "
            "ORDER BY turn").fetchall()
        assert rows == [("message", "damage"), ("message", "stair")]

    def test_anomalies_persisted(self, tmp_path):
        class FakeAnomaly:
            condition_name = "BURST_DAMAGE"
            message = "Burst HP drop: -5 HP"

        logger = ParquetLogger(base_dir=str(tmp_path), run_id="r1", tick_batch_size=100)
        a = self._agent(logger)
        a.current_message = ""          # isolate the anomaly stream
        a._log_events({"anomalies": [FakeAnomaly()]})
        a._log_events({"anomalies": [FakeAnomaly()]})   # deduped
        logger.flush_all()
        rows = duckdb.sql(
            f"SELECT kind, code FROM read_parquet('{tmp_path}/events/*.parquet')").fetchall()
        assert rows == [("anomaly", "BURST_DAMAGE")]

    def test_no_logger_is_noop(self):
        a = self._agent(None)
        a._log_events({"anomalies": []})  # must not raise


class TestConsolidation:
    def test_consolidator_ingests_events(self, tmp_path):
        from lox.telemetry.duckdb_consolidator import DuckDBConsolidator
        parquet = str(tmp_path / "parquet")
        logger = ParquetLogger(base_dir=parquet, run_id="r1", tick_batch_size=2)
        for i in range(3):
            logger.log_event(EventRecord("r1", "ep1", i, i + 1, 1, "message", "damage", "hit"))
        logger.flush_all()
        db = str(tmp_path / "t.duckdb")
        con = DuckDBConsolidator(db_path=db, parquet_dir=parquet)
        stats = con.consolidate(clean_parquet=False)
        assert stats["events"] == 3
        rows = duckdb.connect(db, read_only=True).execute(
            "SELECT kind, code, occurrences FROM v_event_summary").fetchall()
        assert rows == [("message", "damage", 3)]
        # idempotent on re-consolidate
        assert con.consolidate(clean_parquet=False)["events"] == 3