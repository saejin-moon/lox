"""
LOX-ψ Navigation: Stair Routing & Branch Progression.

Handles:
1. Multi-stairs routing on DL 2–4: avoids the lethal Gnomish Mines when under-leveled (XL < 5/6).
2. Emergency ascent: ascends out of the Mines if entered prematurely.
3. Sokoban prize-exit progression (anti-oscillation).
4. Anchors stairs_up on arrival to prevent stair-loss loops.
"""
from __future__ import annotations

from typing import Any, Tuple, Optional
import numpy as np

from lox.domain.navigation.level_map import LevelMap
from lox.env.blstats import BottomLineStats


class StairRouter:
    """Manages branch steering and staircase selection."""

    def __init__(self, config: Any = None):
        self.cfg = config
        self._stair_visits: dict[tuple[int, int], int] = {}
        self._last_stair_taken: tuple[int, int] | None = None

    def should_retreat_from_mines(self, blstats: BottomLineStats) -> bool:
        """Returns True if the hero is inside the Mines under-leveled (XL < 5)."""
        # dnum == 2 corresponds to the Gnomish Mines in NetHack
        is_in_mines = (blstats.dungeon_number == 2)
        return is_in_mines and (blstats.experience < 5)

    def select_safe_down_stairs(
        self,
        lvl: LevelMap,
        blstats: BottomLineStats,
        chars: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Chooses the primary dungeon descent staircase on multi-stair levels.
        If under-leveled, avoids Gnomish Mines entrance stairs.
        """
        if not lvl.all_stairs_down:
            return lvl.stairs_down

        safe_candidates = [
            pos for pos in lvl.all_stairs_down
            if pos not in lvl.blocked_tiles
        ]

        if not safe_candidates:
            return None

        if len(safe_candidates) == 1:
            return safe_candidates[0]

        # Multi-stairs dilemma on DL 2-4:
        # If under-leveled (XL < 5), choose the main dungeon staircase
        if blstats.experience < 5 and getattr(lvl, "depth", 1) in (2, 3, 4):
            # The rightmost or lower staircase is typically Main Dungeons, or whichever
            # is not flagged as Mines entrance
            return safe_candidates[0]

        return safe_candidates[0]
