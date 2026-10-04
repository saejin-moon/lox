"""
LOX Optimal Dynamic Synthesis Trigger Engine.
Monitors execution events and fires LLM synthesis triggers only when needed.
Features a 2-Tier Reactive Trigger Architecture:
1. Online Intra-Episode Interventions: Floor Stagnation, Starvation Crisis.
2. Offline Cross-Episode Batch Interventions: Fatal Taxonomy Cluster, Milestone Breakthrough.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any
from lox.telemetry.recorder import FlightRecorder


class TriggerType(Enum):
    NONE = auto()
    STALL = auto()           # >=80 turns on single level with no new frontier
    STARVATION = auto()      # hunger >= WEAK with 0 food in inventory
    CLUSTER_DEATH = auto()   # 3+ recent deaths sharing identical fatality cause
    MILESTONE = auto()       # New depth or achievement milestone reached
    BATCH_END = auto()       # Periodic evaluation batch boundary


class DynamicTriggerEngine:
    """Evaluates whether the current game state warrants an LLM authoring session."""

    def __init__(self, stall_threshold: int = 80, cluster_threshold: int = 3):
        self.stall_threshold = stall_threshold
        self.cluster_threshold = cluster_threshold
        self.max_depth_seen = 1

    def check_turn(
        self,
        turns_on_level: int,
        depth: int,
        hunger_state: str = "NORMAL",
        food_count: int = 1,
        has_frontier: bool = False,
    ) -> tuple[TriggerType, str]:
        """
        Checks per-turn progression for online interventions:
        - Milestone depth breakthrough
        - Starvation crisis without food
        - Floor stagnation stall
        """
        if depth > self.max_depth_seen:
            self.max_depth_seen = depth
            return TriggerType.MILESTONE, f"New depth record reached: Depth {depth}"

        # Online Starvation Trigger: WEAK or FAINTING without food
        if hunger_state in ("WEAK", "FAINTING") and food_count == 0:
            return (
                TriggerType.STARVATION,
                f"Starvation crisis: Hero is {hunger_state} with 0 food items in inventory.",
            )

        # Online Floor Stagnation Trigger: >=80 turns on level without new frontiers
        if turns_on_level >= self.stall_threshold and not has_frontier:
            return (
                TriggerType.STALL,
                f"Floor pacing stall: {turns_on_level} turns on Depth {depth} with no unvisited frontier.",
            )

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
