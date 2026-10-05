"""
LOX Digging Router Engine (Section 6.2).
Straight-line cardinal tunneling and Bresenham ray-carving router for NetHack.
Enables rapid transit through Gehennom mazes and dense stone walls using wands of digging
and pickaxes, cutting turn expenditure from 15,000+ turns down to dozens of turns.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DiggingRouter:
    """
    Computes straight-line digging vectors and identifies breachable walls
    between hero and destination stairs/portals.
    """

    @staticmethod
    def get_cardinal_tunnel_direction(
        hero_pos: tuple[int, int], target_pos: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Determines the dominant cardinal direction vector (dy, dx) from hero to target.
        """
        hy, hx = hero_pos
        ty, tx = target_pos
        dy = ty - hy
        dx = tx - hx

        if abs(dy) >= abs(dx):
            return (1 if dy > 0 else -1, 0)
        else:
            return (0, 1 if dx > 0 else -1)

    @staticmethod
    def bresenham_ray(
        start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Returns list of coordinates along the Bresenham line from start to end.
        """
        y0, x0 = start
        y1, x1 = end
        points = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0

        while True:
            points.append((curr_y, curr_x))
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

        return points

    @classmethod
    def find_dig_target(
        cls,
        hero_pos: tuple[int, int],
        target_pos: tuple[int, int],
        walkable: np.ndarray,
        chars: np.ndarray,
    ) -> dict[str, Any] | None:
        """
        Analyzes the path from hero to target. If blocked by solid rock or walls,
        proposes the cardinal or ray direction to zap a wand of digging or dig with a pickaxe.
        """
        if hero_pos == target_pos:
            return None

        cardinal_dir = cls.get_cardinal_tunnel_direction(hero_pos, target_pos)
        adj_y = hero_pos[0] + cardinal_dir[0]
        adj_x = hero_pos[1] + cardinal_dir[1]

        # If adjacent tile in dominant direction is non-walkable wall or stone, it's an immediate dig target
        if 0 <= adj_y < 21 and 0 <= adj_x < 79 and not walkable[adj_y, adj_x]:
            return {
                "direction": cardinal_dir,
                "target_tile": (adj_y, adj_x),
                "is_adjacent": True,
            }

        # Otherwise scan along the ray for the first wall encounter
        ray = cls.bresenham_ray(hero_pos, target_pos)
        for i, (ry, rx) in enumerate(ray[1:], 1):
            if 0 <= ry < 21 and 0 <= rx < 79 and not walkable[ry, rx]:
                # Direction from previous point
                prev_y, prev_x = ray[i - 1]
                return {
                    "direction": (ry - prev_y, rx - prev_x),
                    "target_tile": (ry, rx),
                    "is_adjacent": (i == 1),
                }

        return None
