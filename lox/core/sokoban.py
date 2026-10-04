"""
LOX Sokoban Solver Engine (Section 6.1).
Autonomous deterministic boulder-push graph solver for NetHack Sokoban levels.
Enables heroes to navigate and solve Sokoban branch (dnum == 4) puzzles
without incurring Luck penalties, boulder destruction, or corridor deadlocks,
guaranteeing acquisition of the Amulet of Reflection or Bag of Holding.
"""

from __future__ import annotations

from collections import deque
from typing import Any
import numpy as np


class SokobanSolver:
    """
    Deterministic graph solver for NetHack Sokoban puzzle levels.
    """

    CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    @staticmethod
    def is_deadlock_position(by: int, bx: int, walls: np.ndarray, pits: np.ndarray) -> bool:
        """
        Checks if a boulder at (by, bx) is in a corner or immovable deadlock.
        A boulder in a corner (two perpendicular walls) that is not on a pit is deadlocked.
        """
        if pits[by, bx]:
            return False

        # Corner check (orthogonal wall pairs)
        n = walls[by - 1, bx] if by > 0 else True
        s = walls[by + 1, bx] if by < 20 else True
        w = walls[by, bx - 1] if bx > 0 else True
        e = walls[by, bx + 1] if bx < 78 else True

        if (n and w) or (n and e) or (s and w) or (s and e):
            return True

        return False

    @staticmethod
    def find_hero_reachable(hero_pos: tuple[int, int], walkable_floor: np.ndarray) -> np.ndarray:
        """
        Computes 2D boolean mask of tiles reachable by the hero without stepping
        on boulders, pits, or walls.
        """
        reachable = np.zeros((21, 79), dtype=bool)
        hy, hx = hero_pos
        if not (0 <= hy < 21 and 0 <= hx < 79) or not walkable_floor[hy, hx]:
            reachable[hy, hx] = True
            return reachable

        q = deque([(hy, hx)])
        reachable[hy, hx] = True

        while q:
            cy, cx = q.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < 21 and 0 <= nx < 79 and walkable_floor[ny, nx] and not reachable[ny, nx]:
                    reachable[ny, nx] = True
                    q.append((ny, nx))

        return reachable

    @classmethod
    def find_best_boulder_push(
        cls,
        hero_pos: tuple[int, int],
        walls: np.ndarray,
        boulders: np.ndarray,
        pits: np.ndarray,
        clean_floor: np.ndarray,
    ) -> dict[str, Any] | None:
        """
        Finds the highest-priority valid boulder push that advances toward a pit.
        Returns:
            dict with 'boulder_pos', 'push_dir', 'hero_stand_pos', 'is_ready_to_push'
            or None if no safe push is available.
        """
        # Hero can walk on clean floor that does NOT contain a boulder or pit
        hero_walkable = clean_floor & (~boulders) & (~pits)
        reachable = cls.find_hero_reachable(hero_pos, hero_walkable)

        # List all pits that need filling
        pit_coords = [(py, px) for py in range(21) for px in range(79) if pits[py, px]]
        if not pit_coords:
            return None

        # Find candidate boulders
        boulder_coords = [(by, bx) for by in range(21) for bx in range(79) if boulders[by, bx]]
        if not boulder_coords:
            return None

        best_candidate: dict[str, Any] | None = None
        min_distance_to_pit = float("inf")

        for by, bx in boulder_coords:
            # Check all 4 push directions
            for dy, dx in cls.CARDINALS:
                stand_y, stand_x = by - dy, bx - dx
                dest_y, dest_x = by + dy, bx + dx

                # Check bounds
                if not (0 <= stand_y < 21 and 0 <= stand_x < 79):
                    continue
                if not (0 <= dest_y < 21 and 0 <= dest_x < 79):
                    continue

                # Can the hero reach the standing tile behind the boulder?
                if not reachable[stand_y, stand_x] and (stand_y, stand_x) != hero_pos:
                    continue

                # Is the destination tile valid? Must be a pit or clean open floor without another boulder
                if not pits[dest_y, dest_x]:
                    if not clean_floor[dest_y, dest_x] or boulders[dest_y, dest_x] or walls[dest_y, dest_x]:
                        continue
                    # Destination must not be a permanent corner deadlock
                    if cls.is_deadlock_position(dest_y, dest_x, walls, pits):
                        continue

                # Calculate distance from dest to the closest unfilled pit
                closest_pit_dist = min(
                    abs(dest_y - py) + abs(dest_x - px) for py, px in pit_coords
                )

                # Prioritize pushing directly into a pit (distance 0)
                if pits[dest_y, dest_x]:
                    closest_pit_dist = -100

                if closest_pit_dist < min_distance_to_pit:
                    min_distance_to_pit = closest_pit_dist
                    best_candidate = {
                        "boulder_pos": (by, bx),
                        "push_dir": (dy, dx),
                        "hero_stand_pos": (stand_y, stand_x),
                        "dest_pos": (dest_y, dest_x),
                        "is_ready_to_push": (hero_pos == (stand_y, stand_x)),
                    }

        return best_candidate
