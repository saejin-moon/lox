"""
LOX Flight Recorder: Lightweight in-memory circular telemetry.
Stores the last 100 turns of execution state to generate high-signal tactical autopsies for the LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


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
    tile_type: str = "room"
    closest_hostile_name: str = ""
    closest_hostile_dist: float = 999.0
    hostiles_in_fov: int = 0
    dungeon_branch: str = "dungeon"


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
        tile_type: str = "room",
        closest_hostile_name: str = "",
        closest_hostile_dist: float = 999.0,
        hostiles_in_fov: int = 0,
        dungeon_branch: str = "dungeon",
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
                tile_type=tile_type,
                closest_hostile_name=closest_hostile_name,
                closest_hostile_dist=closest_hostile_dist,
                hostiles_in_fov=hostiles_in_fov,
                dungeon_branch=dungeon_branch,
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

    def get_last_10_turns_trajectory(self) -> str:
        """Returns readable trajectory of HP progression and actions over the last 10 ticks."""
        if not self.buffer:
            return "No ticks recorded."
        recent = list(self.buffer)[-10:]
        hp_traj = " -> ".join(f"{s.hp}" for s in recent)
        last_s = recent[-1]
        threat_desc = (
            f"{last_s.closest_hostile_name} (dist {last_s.closest_hostile_dist:.1f})"
            if last_s.closest_hostile_name
            else "none in FOV"
        )
        return (
            f"Pre-Death HP Trajectory (last 10 ticks): {hp_traj}\n"
            f"Action at Death: `{last_s.action_name}` on tile type `{last_s.tile_type}` in branch `{last_s.dungeon_branch}`\n"
            f"Closest Threat at Death: {threat_desc}\n"
            f'Last Game Message: "{last_s.message or "None"}"'
        )

    def generate_compact_status_report(
        self, trigger_reason: str, cluster_note: str = ""
    ) -> str:
        """Generates compact status report with recent steps and tactical details."""
        if not self.buffer:
            return f"[INCIDENT: {trigger_reason}]\nNo turn telemetry recorded."

        last = self.buffer[-1]
        lines = [
            f"[INCIDENT: {trigger_reason}]",
            f"State: Depth {last.depth} ({last.dungeon_branch}) | Turn {last.turn} | HP {last.hp}/{last.max_hp} | Hunger {last.hunger}",
            "",
            "Recent Steps (Last 5):",
            "| T | HP | Action | Threat | Message |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        recent_snaps = list(self.buffer)[-5:]
        for s in recent_snaps:
            msg = s.message.replace("|", "/") if s.message else "-"
            threat = (
                f"{s.closest_hostile_name[:10]} ({s.closest_hostile_dist:.1f})"
                if s.closest_hostile_name
                else "-"
            )
            lines.append(
                f"| {s.turn} | {s.hp}/{s.max_hp} | `{s.action_name}` | {threat} | {msg} |"
            )

        if cluster_note:
            lines.append(f"\n{cluster_note}")

        return "\n".join(lines)

    def generate_autopsy_report(self, death_reason: str) -> str:
        """Legacy alias pointing to compact status report."""
        return self.generate_compact_status_report(trigger_reason=death_reason)
