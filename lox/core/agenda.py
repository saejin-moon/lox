"""
LOX Hybrid HTN-BT Goal Agenda:
Provides high-level strategic goal management above low-latency reactive reflex nodes.
Directs non-emergency macro selection across long dungeon horizons and depth-tiered progression phases.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from lox.core.types import AgendaView


class DungeonPhase(str, Enum):
    """Depth-tiered strategic macro phases in NetHack ascension progression."""

    EARLY_RUSH = "early_rush"          # Depths 1-2: Rapid exploration & descent, scoop surface loot, no dead-end searching
    EARLY_SCALING = "early_scaling"    # Depths 3-5: Excalibur, mithril/iron body armor, poison resistance, altar BUC
    MID_BRANCHES = "mid_branches"      # Depths 6-10: Sokoban solve, Mines avoidance, temple priest donation
    DEEP_DUNGEON = "deep_dungeon"      # Depths 11-19: Medusa crossing, Quest portal, corridor defense
    CASTLE_ASCENSION = "castle"        # Depths 20+: Drawbridge breach, Wand of Wishing, Ascension kit


class GoalDirective(str, Enum):
    """Core strategic milestones in NetHack progression."""

    EXPLORE_FLOOR = "explore_floor"
    COLLECT_POISON_RES = "collect_poison_res"
    FORGE_EXCALIBUR = "forge_excalibur"
    UPGRADE_ARMOR = "upgrade_armor"
    TEST_BUC_ALTAR = "test_buc_altar"
    CLEAR_MINES = "clear_mines"
    SOLVE_SOKOBAN = "solve_sokoban"
    DONATE_TO_PRIEST = "donate_to_priest"
    DESCEND_STAIRS = "descend_stairs"


class GoalAgenda:
    """
    Maintains a goal stack for long-horizon task coordination.
    High-priority reflexes (Combat, Panic Elbereth, Emergency Prayer) preempt the agenda,
    while non-emergency exploration/macros defer to the top of the stack and the active dungeon phase.
    """

    def __init__(self, initial_goals: list[GoalDirective] | None = None):
        self._stack: list[GoalDirective] = initial_goals or [
            GoalDirective.DESCEND_STAIRS,
            GoalDirective.EXPLORE_FLOOR,
        ]
        self._last_depth = 1
        self._current_phase = DungeonPhase.EARLY_RUSH

    @staticmethod
    def get_phase(depth: int, branch: str = "") -> DungeonPhase:
        if depth >= 20:
            return DungeonPhase.CASTLE_ASCENSION
        elif depth >= 11:
            return DungeonPhase.DEEP_DUNGEON
        elif depth >= 6 or branch == "sokoban":
            return DungeonPhase.MID_BRANCHES
        elif depth >= 3:
            return DungeonPhase.EARLY_SCALING
        else:
            return DungeonPhase.EARLY_RUSH

    @property
    def current_phase(self) -> DungeonPhase:
        return self._current_phase

    @property
    def active_goal(self) -> GoalDirective:
        return self._stack[-1] if self._stack else GoalDirective.EXPLORE_FLOOR

    @property
    def stack(self) -> list[GoalDirective]:
        return list(self._stack)

    def push(self, goal: GoalDirective) -> None:
        """Pushes a new strategic priority to the top of the agenda."""
        if goal in self._stack:
            self._stack.remove(goal)
        self._stack.append(goal)

    def pop(self) -> GoalDirective | None:
        """Pops the active goal upon milestone completion."""
        if len(self._stack) > 1:
            return self._stack.pop()
        return self._stack[0] if self._stack else None

    def remove(self, goal: GoalDirective) -> None:
        """Removes a goal from the agenda if present."""
        if goal in self._stack:
            self._stack.remove(goal)
        if not self._stack:
            self._stack.append(GoalDirective.EXPLORE_FLOOR)

    def evaluate_milestones(self, obs: Any) -> None:
        """
        Dynamically adjusts active phase and goals based on observation state.
        """
        depth = getattr(obs.hero, "depth", 1)
        branch = getattr(obs.hero, "dungeon_branch", "")
        self._current_phase = self.get_phase(depth, branch)
        active = self.active_goal

        # 0. Floor transition: reset agenda stack on new dungeon level
        if depth != getattr(self, "_last_depth", depth):
            self._last_depth = depth
            self._stack = [GoalDirective.DESCEND_STAIRS, GoalDirective.EXPLORE_FLOOR]
            # In Phase 0 (Early rush), prioritize rapid descent
            if self._current_phase == DungeonPhase.EARLY_RUSH:
                stairs_known = getattr(obs.spatial, "stairs_down_known", False)
                if stairs_known:
                    self._stack = [GoalDirective.DESCEND_STAIRS]
            return

        # 1. Floor exploration complete or Phase 0 down stairs discovered -> transition to descent
        if active == GoalDirective.EXPLORE_FLOOR:
            stairs_known = getattr(obs.spatial, "stairs_down_known", False)
            has_frontier = getattr(obs.spatial, "has_unvisited_frontier", True)
            if stairs_known and (not has_frontier or self._current_phase == DungeonPhase.EARLY_RUSH):
                self.pop()

        # 2. Poison resistance acquired
        elif active == GoalDirective.COLLECT_POISON_RES:
            has_poison_res = getattr(obs.hero, "has_poison_res", False) or getattr(
                obs.status, "has_poison_res", False
            )
            if has_poison_res:
                self.pop()

        # 3. Excalibur forged
        elif active == GoalDirective.FORGE_EXCALIBUR:
            can_forge = getattr(obs.dungeon, "can_forge_excalibur", False)
            if not can_forge:
                # Either forged or no longer eligible
                self.pop()

        # 4. Altar BUC tests complete
        elif active == GoalDirective.TEST_BUC_ALTAR:
            epistemic = getattr(obs, "epistemic", None)
            if epistemic is not None and epistemic.untested_buc_count == 0:
                self.pop()

        # 5. Sokoban solved
        elif active == GoalDirective.SOLVE_SOKOBAN:
            if not getattr(obs.dungeon, "can_solve_sokoban", False):
                self.pop()

    def create_view(self) -> AgendaView:
        return AgendaView(
            active_goal=self.active_goal.value,
            goal_stack=[g.value for g in self._stack],
            dungeon_phase=self.current_phase.value,
        )
