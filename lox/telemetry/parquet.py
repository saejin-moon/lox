"""
LOX 2.0 Parquet Telemetry Logger.
Streaming, zero-lock PyArrow Parquet writer for per-tick, per-episode, and per-event telemetry.
"""
from __future__ import annotations

import os
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq


TICK_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("episode_id", pa.string()),
    ("turn", pa.int32()),
    ("depth", pa.int32()),
    ("hp", pa.int32()),
    ("max_hp", pa.int32()),
    ("hunger", pa.string()),
    ("y", pa.int32()),
    ("x", pa.int32()),
    ("action", pa.string()),
    ("message", pa.string()),
    ("reward", pa.float32()),
])

EPISODE_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("episode_id", pa.string()),
    ("depth", pa.int32()),
    ("score", pa.int32()),
    ("turns", pa.int32()),
    ("death_reason", pa.string()),
    ("solved", pa.bool_()),
    ("wall_sec", pa.float32()),
    ("role", pa.string()),
    ("gold", pa.int32()),
    ("max_depth", pa.int32()),
    ("steps", pa.int32()),
    ("attacks", pa.int32()),
    ("descents", pa.int32()),
    ("searches", pa.int32()),
    ("eats", pa.int32()),
    ("prayers", pa.int32()),
    ("death_category", pa.string()),
])

EVENT_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("episode_id", pa.string()),
    ("turn", pa.int32()),
    ("depth", pa.int32()),
    ("event_type", pa.string()),
    ("message", pa.string()),
    ("details", pa.string()),
])


class ParquetLogger:
    """Streams execution data into chunked Parquet partition files."""

    def __init__(self, run_id: str, base_dir: str = "data/telemetry", flush_interval: int = 500):
        self.run_id = run_id
        self.run_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.flush_interval = flush_interval

        self.tick_buffer: list[dict[str, Any]] = []
        self.episode_buffer: list[dict[str, Any]] = []
        self.event_buffer: list[dict[str, Any]] = []
        self.tick_part_count = 0

    def log_tick(
        self,
        episode_id: str,
        turn: int,
        depth: int,
        hp: int,
        max_hp: int,
        hunger: str,
        y: int,
        x: int,
        action: str,
        message: str = "",
        reward: float = 0.0,
    ) -> None:
        self.tick_buffer.append({
            "run_id": self.run_id,
            "episode_id": episode_id,
            "turn": int(turn),
            "depth": int(depth),
            "hp": int(hp),
            "max_hp": int(max_hp),
            "hunger": str(hunger),
            "y": int(y),
            "x": int(x),
            "action": str(action),
            "message": str(message)[:80],
            "reward": float(reward),
        })

        if len(self.tick_buffer) >= self.flush_interval:
            self.flush_ticks()

    def log_episode(
        self,
        episode_id: str,
        depth: int,
        score: int,
        turns: int,
        death_reason: str,
        solved: bool,
        wall_sec: float,
        role: str = "valkyrie",
        gold: int = 0,
        max_depth: int = 1,
        steps: int = 0,
        attacks: int = 0,
        descents: int = 0,
        searches: int = 0,
        eats: int = 0,
        prayers: int = 0,
        death_category: str = "unknown",
    ) -> None:
        self.episode_buffer.append({
            "run_id": self.run_id,
            "episode_id": episode_id,
            "depth": int(depth),
            "score": int(score),
            "turns": int(turns),
            "death_reason": str(death_reason),
            "solved": bool(solved),
            "wall_sec": float(wall_sec),
            "role": str(role),
            "gold": int(gold),
            "max_depth": int(max_depth),
            "steps": int(steps),
            "attacks": int(attacks),
            "descents": int(descents),
            "searches": int(searches),
            "eats": int(eats),
            "prayers": int(prayers),
            "death_category": str(death_category),
        })

    def log_event(
        self,
        episode_id: str,
        turn: int,
        depth: int,
        event_type: str,
        message: str = "",
        details: str = "",
    ) -> None:
        self.event_buffer.append({
            "run_id": self.run_id,
            "episode_id": episode_id,
            "turn": int(turn),
            "depth": int(depth),
            "event_type": str(event_type),
            "message": str(message)[:100],
            "details": str(details)[:150],
        })

    def flush_ticks(self) -> None:
        if not self.tick_buffer:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        table = pa.Table.from_pylist(self.tick_buffer, schema=TICK_SCHEMA)
        part_path = os.path.join(self.run_dir, f"ticks_part_{self.tick_part_count:04d}.parquet")
        pq.write_table(table, part_path)
        self.tick_part_count += 1
        self.tick_buffer.clear()

    def flush_episodes(self) -> None:
        if not self.episode_buffer:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        table = pa.Table.from_pylist(self.episode_buffer, schema=EPISODE_SCHEMA)
        ep_path = os.path.join(self.run_dir, "episodes.parquet")
        pq.write_table(table, ep_path)
        self.episode_buffer.clear()

    def flush_events(self) -> None:
        if not self.event_buffer:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        table = pa.Table.from_pylist(self.event_buffer, schema=EVENT_SCHEMA)
        ev_path = os.path.join(self.run_dir, "events.parquet")
        pq.write_table(table, ev_path)
        self.event_buffer.clear()

    def flush(self) -> None:
        self.flush_ticks()
        self.flush_episodes()
        self.flush_events()

    def close(self) -> None:
        self.flush()
