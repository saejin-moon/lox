"""
LOX Hybrid HTN-BT Goal Agenda:
Provides high-level strategic goal management above low-latency reactive reflex nodes.
Directs non-emergency macro selection across long dungeon horizons.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class GoalDirective(str, Enum):
    """Core strategic milestones in NetHack progression."""
    EXPLORE_FLOOR = "explore_floor"
    COLLECT_POISON_RES = "collect_poison_res"
    FORGE_EXCALIBUR = "forge_excalibur"
    TEST_BUC_ALTAR = "test_buc_altar"
    CLEAR_MINES = "clear_mines"
    SOLVE_SOKOBAN = "solve_sokoban"
    DESCEND_STAIRS = "descend_stairs"


from lox.core.types import AgendaView


class GoalAgenda:
    """
    Maintains a goal stack for long-horizon task coordination.
    High-priority reflexes (Combat, Panic Elbereth, Emergency Prayer) preempt the agenda,
    while non-emergency exploration/macros defer to the top of the stack.
    """

    def __init__(self, initial_goals: list[GoalDirective] | None = None):
        self._stack: list[GoalDirective] = initial_goals or [
            GoalDirective.DESCEND_STAIRS,
            GoalDirective.EXPLORE_FLOOR,
        ]
        self._last_depth = 1

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
        Dynamically adjusts and auto-completes goals based on observation state.
        """
        active = self.active_goal

        # 0. Floor transition: reset agenda stack on new dungeon level
        depth = getattr(obs.hero, "depth", 1)
        if depth != getattr(self, "_last_depth", depth):
            self._last_depth = depth
            self._stack = [GoalDirective.DESCEND_STAIRS, GoalDirective.EXPLORE_FLOOR]
            return

        # 1. Floor exploration complete -> transition to descent
        if active == GoalDirective.EXPLORE_FLOOR:
            stairs_known = getattr(obs.spatial, "stairs_down_known", False)
            has_frontier = getattr(obs.spatial, "has_unvisited_frontier", True)
            if stairs_known and not has_frontier:
                self.pop()

        # 2. Poison resistance acquired
        elif active == GoalDirective.COLLECT_POISON_RES:
            has_poison_res = getattr(obs.hero, "has_poison_res", False) or getattr(obs.status, "has_poison_res", False)
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

    def create_view(self) -> AgendaView:
        return AgendaView(
            active_goal=self.active_goal.value,
            goal_stack=[g.value for g in self._stack],
        )
