"""
LOX 2.0 Flight Recorder: Lightweight in-memory circular telemetry.
Stores the last 100 turns of execution state to generate high-signal autopsies for the LLM.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnSnapshot:
    turn: int
    depth: int
    hp: int
    max_hp: int
    hunger: str
    pos: tuple[int, int]
    action_name: str
    message: str


class FlightRecorder:
    """Ring buffer holding the most recent execution steps."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque[TurnSnapshot] = deque(maxlen=capacity)
        self.death_causes: list[str] = []

    def record_turn(
        self,
        turn: int,
        depth: int,
        hp: int,
        max_hp: int,
        hunger: str,
        pos: tuple[int, int],
        action_name: str,
        message: str = "",
    ) -> None:
        self.buffer.append(
            TurnSnapshot(
                turn=turn,
                depth=depth,
                hp=hp,
                max_hp=max_hp,
                hunger=hunger,
                pos=pos,
                action_name=action_name,
                message=message[:80],
            )
        )

    def record_death(self, death_message: str) -> None:
        self.death_causes.append(death_message)

    def get_recent_death_taxonomy(self, window: int = 5) -> dict[str, int]:
        recent = self.death_causes[-window:]
        counts: dict[str, int] = {}
        for d in recent:
            counts[d] = counts.get(d, 0) + 1
        return counts

    def generate_compact_status_report(self, trigger_reason: str, cluster_note: str = "") -> str:
        """
        Generates an ultra-compact status report (<180 tokens) with high signal density:
        - Incident header
        - Last 5 steps transition table
        - Historical cluster note
        """
        if not self.buffer:
            return f"[INCIDENT: {trigger_reason}]\nNo turn telemetry recorded."

        last = self.buffer[-1]
        lines = [
            f"[INCIDENT: {trigger_reason}]",
            f"State: Depth {last.depth} | Turn {last.turn} | HP {last.hp}/{last.max_hp} | Hunger {last.hunger}",
            "",
            "Recent Steps (Last 5):",
            "| T | HP | Action | Message |",
            "| :--- | :--- | :--- | :--- |",
        ]

        recent_snaps = list(self.buffer)[-5:]
        for s in recent_snaps:
            msg = s.message.replace("|", "/") if s.message else "-"
            lines.append(f"| {s.turn} | {s.hp}/{s.max_hp} | `{s.action_name}` | {msg} |")

        if cluster_note:
            lines.append(f"\n{cluster_note}")

        return "\n".join(lines)

    def generate_autopsy_report(self, death_reason: str) -> str:
        """Legacy alias pointing to compact status report."""
        return self.generate_compact_status_report(trigger_reason=death_reason)
