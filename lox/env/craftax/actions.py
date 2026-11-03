"""
Craftax Discrete Action Enumeration & Primitives ("Hard Code Nothing").
1-to-1 mapping of native Craftax actions.
"""

from __future__ import annotations

import enum


class CraftaxAction(enum.IntEnum):
    """Native discrete actions in Crafter / Craftax-Classic."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16


ACTION_DELTAS: dict[CraftaxAction, tuple[int, int]] = {
    CraftaxAction.LEFT: (0, -1),
    CraftaxAction.RIGHT: (0, 1),
    CraftaxAction.UP: (-1, 0),
    CraftaxAction.DOWN: (1, 0),
}
