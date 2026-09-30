"""
LOX 2.0 Dynamic Synthesis Trigger Engine.
Monitors execution events and fires LLM synthesis triggers only when needed.
Replaces wasteful fixed "every N episodes" polling with high-signal reactive triggers.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any
from lox.telemetry.recorder import FlightRecorder


class TriggerType(Enum):
    NONE = auto()
    STALL = auto()           # >150 turns on single level without descent
    CLUSTER_DEATH = auto()   # 3+ recent deaths sharing identical fatality cause
    MILESTONE = auto()       # New depth or achievement milestone reached
    BATCH_END = auto()       # Periodic evaluation batch boundary


class DynamicTriggerEngine:
    """Evaluates whether the current game state warrants an LLM authoring session."""

    def __init__(self, stall_threshold: int = 150, cluster_threshold: int = 3):
        self.stall_threshold = stall_threshold
        self.cluster_threshold = cluster_threshold
        self.max_depth_seen = 1

    def check_turn(self, turns_on_level: int, depth: int) -> tuple[TriggerType, str]:
        """Checks per-turn progression for stalls or depth breakthroughs."""
        if depth > self.max_depth_seen:
            self.max_depth_seen = depth
            return TriggerType.MILESTONE, f"New depth record reached: Depth {depth}"

        if turns_on_level >= self.stall_threshold:
            return TriggerType.STALL, f"Floor pacing stall: {turns_on_level} turns spent on Depth {depth} without descending."

        return TriggerType.NONE, ""

    def check_death_cluster(self, recorder: FlightRecorder) -> tuple[TriggerType, str]:
        """Checks recent deaths for recurring cluster fatalities."""
        taxonomy = recorder.get_recent_death_taxonomy(window=5)
        for cause, count in taxonomy.items():
            if count >= self.cluster_threshold:
                return (
                    TriggerType.CLUSTER_DEATH,
                    f"Death cluster detected: {count}/5 recent runs terminated by '{cause}'.",
                )
        return TriggerType.NONE, ""
