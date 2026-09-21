"""Agent episode dataclasses — extracted from corp_agent (behavior-preserving)."""

from dataclasses import dataclass, field

from corp.planner.htn import Task


@dataclass
class LockedIntent:
    """Represents a multi-turn committed intent to prevent heuristic fluttering."""
    name: str  # e.g. "DOOR_KICK", "RETREAT_FUNNEL", "ALTAR_TEST", "DIP_FOUNTAIN"
    target_pos: tuple[int, int]
    subtasks: list[Task] = field(default_factory=list)
    remaining_steps: int = 0
    abort_on_emergency: bool = True


@dataclass
class EpisodeResult:
    """Summary metrics of a completed NetHack episode."""
    turns: int
    max_depth: int
    final_score: int
    final_hp: int
    is_ascended: bool
    death_message: str
    active_nogoods_count: int
    steps_per_second: float
