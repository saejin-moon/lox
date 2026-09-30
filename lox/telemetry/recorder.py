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

    def generate_autopsy_report(self, death_reason: str) -> str:
        """Generates a compact, high-signal markdown autopsy report for the LLM author."""
        if not self.buffer:
            return f"### Episode Autopsy\n- **Death Reason**: {death_reason}\n- No turn telemetry recorded.\n"

        last_snap = self.buffer[-1]
        lines = [
            "### Episode Autopsy",
            f"- **Fatal Outcome**: `{death_reason}`",
            f"- **Dungeon Depth**: {last_snap.depth} | **Total Turns**: {last_snap.turn}",
            f"- **Final HP**: {last_snap.hp}/{last_snap.max_hp} | **Hunger**: {last_snap.hunger}",
            "",
            "#### Last 8 Steps Before Termination:",
            "| Turn | HP | Action | In-Game Message |",
            "| :--- | :--- | :--- | :--- |",
        ]

        recent_snaps = list(self.buffer)[-8:]
        for s in recent_snaps:
            msg = s.message.replace("|", "/") if s.message else "-"
            lines.append(f"| {s.turn} | {s.hp}/{s.max_hp} | `{s.action_name}` | {msg} |")

        return "\n".join(lines)
