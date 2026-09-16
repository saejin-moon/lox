"""
Telemetry Subsystem: High-performance streaming Parquet Logger.
Buffers tick-level and episode-level telemetry in memory and writes snappy-compressed
columnar Parquet partitions via PyArrow with zero simulation overhead.
"""

from dataclasses import dataclass, asdict, field
import json
import os
import time
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq


TICK_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("run_id", pa.string()),
    ("step", pa.int64()),
    ("turn", pa.int64()),
    ("timestamp", pa.float64()),
    ("depth", pa.int32()),
    ("dungeon_number", pa.int32()),
    ("level_number", pa.int32()),
    ("x", pa.int32()),
    ("y", pa.int32()),
    ("hp", pa.int32()),
    ("max_hp", pa.int32()),
    ("energy", pa.int32()),
    ("max_energy", pa.int32()),
    ("ac", pa.int32()),
    ("xp_level", pa.int32()),
    ("xp_points", pa.int32()),
    ("gold", pa.int32()),
    ("hunger_state", pa.int32()),
    ("encumbrance", pa.int32()),
    ("condition_bits", pa.int64()),
    ("predicate_mask", pa.int64()),
    ("active_nogoods_count", pa.int32()),
    ("action_name", pa.string()),
    ("action_index", pa.int32()),
    ("action_key_char", pa.string()),
    ("action_args_json", pa.string()),
    ("decision_latency_us", pa.float64()),
    ("visible_hostiles_count", pa.int32()),
    ("closest_monster_name", pa.string()),
    ("closest_monster_dist", pa.int32()),
    ("closest_monster_speed", pa.int32()),
    ("closest_monster_threat", pa.float64()),
    ("cycle_detected", pa.bool_()),
    ("llm_invoked", pa.bool_()),
    ("reward", pa.float64()),
])

EPISODE_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("run_id", pa.string()),
    ("eval_type", pa.string()),
    ("seed", pa.int64()),
    ("mode", pa.string()),
    ("character", pa.string()),
    ("role", pa.string()),
    ("race", pa.string()),
    ("gender", pa.string()),
    ("alignment", pa.string()),
    ("persona_resilience", pa.float64()),
    ("persona_ranged", pa.float64()),
    ("persona_mana", pa.float64()),
    ("persona_stealth", pa.float64()),
    ("persona_alignment", pa.float64()),
    ("total_turns", pa.int64()),
    ("total_steps", pa.int64()),
    ("max_depth", pa.int32()),
    ("final_score", pa.int64()),
    ("final_hp", pa.int32()),
    ("final_gold", pa.int32()),
    ("is_ascended", pa.bool_()),
    ("death_message", pa.string()),
    ("death_category", pa.string()),
    ("nogoods_active", pa.int32()),
    ("nogoods_synthesized", pa.int32()),
    ("mean_sps", pa.float64()),
    ("wall_duration_sec", pa.float64()),
    ("timestamp", pa.string()),
])

LLM_QUERY_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("episode_id", pa.string()),
    ("run_id", pa.string()),
    ("step", pa.int64()),
    ("turn", pa.int64()),
    ("timestamp", pa.float64()),
    ("trigger_type", pa.string()),        # 'autopsy', 'deadlock_cycle', 'zero_progress_stall', etc.
    ("provider", pa.string()),            # 'mock', 'gemini', 'openrouter', 'llama_cpp'
    ("model", pa.string()),
    ("prompt_preview", pa.string()),
    ("tokens_consumed", pa.int32()),
    ("latency_ms", pa.float64()),
    ("turns_since_last_query", pa.int64()),
    ("wall_seconds_since_last_query", pa.float64()),
])


@dataclass(slots=True)
class TickRecord:
    episode_id: str
    run_id: str
    step: int
    turn: int
    timestamp: float
    depth: int
    dungeon_number: int
    level_number: int
    x: int
    y: int
    hp: int
    max_hp: int
    energy: int
    max_energy: int
    ac: int
    xp_level: int
    xp_points: int
    gold: int
    hunger_state: int
    encumbrance: int
    condition_bits: int
    predicate_mask: int
    active_nogoods_count: int
    action_name: str
    action_index: int
    action_key_char: str
    action_args_json: str
    decision_latency_us: float
    visible_hostiles_count: int
    closest_monster_name: str
    closest_monster_dist: int
    closest_monster_speed: int
    closest_monster_threat: float
    cycle_detected: bool
    llm_invoked: bool
    reward: float


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    run_id: str
    eval_type: str
    seed: int
    mode: str
    character: str
    role: str
    race: str
    gender: str
    alignment: str
    persona_resilience: float
    persona_ranged: float
    persona_mana: float
    persona_stealth: float
    persona_alignment: float
    total_turns: int
    total_steps: int
    max_depth: int
    final_score: int
    final_hp: int
    final_gold: int
    is_ascended: bool
    death_message: str
    death_category: str
    nogoods_active: int
    nogoods_synthesized: int
    mean_sps: float
    wall_duration_sec: float
    timestamp: str


@dataclass(slots=True)
class LLMQueryRecord:
    query_id: str
    episode_id: str
    run_id: str
    step: int
    turn: int
    timestamp: float
    trigger_type: str
    provider: str
    model: str
    prompt_preview: str
    tokens_consumed: int
    latency_ms: float
    turns_since_last_query: int
    wall_seconds_since_last_query: float


class ParquetLogger:
    """
    Streaming Parquet telemetry logger. Buffers records in memory and flushes
    to snappy-compressed Parquet files upon reaching batch size.
    """

    def __init__(
        self,
        base_dir: str = "logs/parquet",
        run_id: str | None = None,
        tick_batch_size: int = 500,
    ):
        self.base_dir = base_dir
        self.run_id = run_id or f"run_{int(time.time())}"
        self.tick_batch_size = tick_batch_size

        self.ticks_dir = os.path.join(base_dir, "ticks")
        self.episodes_dir = os.path.join(base_dir, "episodes")
        self.llm_queries_dir = os.path.join(base_dir, "llm_queries")
        os.makedirs(self.ticks_dir, exist_ok=True)
        os.makedirs(self.episodes_dir, exist_ok=True)
        os.makedirs(self.llm_queries_dir, exist_ok=True)

        self._tick_buffer: list[dict[str, Any]] = []
        self._episode_buffer: list[dict[str, Any]] = []
        self._llm_query_buffer: list[dict[str, Any]] = []
        self._tick_file_counter = 0
        self._episode_file_counter = 0
        self._llm_query_file_counter = 0

    def log_tick(self, record: TickRecord):
        """Buffers a tick frame and flushes if batch threshold reached."""
        self._tick_buffer.append(asdict(record))
        if len(self._tick_buffer) >= self.tick_batch_size:
            self._flush_ticks()

    def log_episode(self, record: EpisodeRecord):
        """Buffers an episode summary record."""
        self._episode_buffer.append(asdict(record))
        self._flush_episodes()

    def log_llm_query(self, record: LLMQueryRecord):
        """Buffers and immediately flushes an LLM query record."""
        self._llm_query_buffer.append(asdict(record))
        self._flush_llm_queries()

    def _flush_ticks(self):
        if not self._tick_buffer:
            return
        table = pa.Table.from_pylist(self._tick_buffer, schema=TICK_SCHEMA)
        out_path = os.path.join(
            self.ticks_dir,
            f"ticks_{self.run_id}_{self._tick_file_counter:04d}.parquet",
        )
        pq.write_table(table, out_path, compression="SNAPPY")
        self._tick_buffer.clear()
        self._tick_file_counter += 1

    def _flush_episodes(self):
        if not self._episode_buffer:
            return
        table = pa.Table.from_pylist(self._episode_buffer, schema=EPISODE_SCHEMA)
        out_path = os.path.join(
            self.episodes_dir,
            f"episodes_{self.run_id}_{self._episode_file_counter:04d}.parquet",
        )
        pq.write_table(table, out_path, compression="SNAPPY")
        self._episode_buffer.clear()
        self._episode_file_counter += 1

    def _flush_llm_queries(self):
        if not self._llm_query_buffer:
            return
        table = pa.Table.from_pylist(self._llm_query_buffer, schema=LLM_QUERY_SCHEMA)
        out_path = os.path.join(
            self.llm_queries_dir,
            f"llm_queries_{self.run_id}_{self._llm_query_file_counter:04d}.parquet",
        )
        pq.write_table(table, out_path, compression="SNAPPY")
        self._llm_query_buffer.clear()
        self._llm_query_file_counter += 1

    def flush_all(self):
        """Flushes any remaining in-memory buffers."""
        self._flush_ticks()
        self._flush_episodes()
        self._flush_llm_queries()
