"""LevelMap / LevelStore — extracted from navigation_manager (behavior-preserving)."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from lox.env.blstats import BottomLineStats, HungerState
from nle import nethack
from lox.navigation.astar import GridAStar, PathNode
from lox.navigation.frontier import FrontierExplorer
from lox.planner.htn import Task, PrimitiveTask
from lox.planner.guards import HTNGuards


@dataclass
class LevelMap:
    """Persistent 2D spatial representation for a single dungeon branch and depth level."""
    depth: int = 1
    dnum: int = 0
    dlevel: int = 1
    walkable: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=bool))
    mapped: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=bool))
    walls: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=bool))
    visited: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=np.int32))
    searched: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=np.int32))
    stairs_down: tuple[int, int] | None = None
    stairs_up: tuple[int, int] | None = None
    all_stairs_down: set[tuple[int, int]] = field(default_factory=set)
    all_stairs_up: set[tuple[int, int]] = field(default_factory=set)
    doors: set[tuple[int, int]] = field(default_factory=set)
    corridors: set[tuple[int, int]] = field(default_factory=set)
    fountains: set[tuple[int, int]] = field(default_factory=set)
    altars: set[tuple[int, int]] = field(default_factory=set)
    thrones: set[tuple[int, int]] = field(default_factory=set)
    traps: set[tuple[int, int]] = field(default_factory=set)
    containers: set[tuple[int, int]] = field(default_factory=set)
    gold_piles: set[tuple[int, int]] = field(default_factory=set)
    floor_corpses: dict[tuple[int, int], int] = field(default_factory=dict)
    conveyor_corpses: dict[tuple[int, int], tuple[int, str]] = field(default_factory=dict)
    consumed_corpses: set[tuple[int, int]] = field(default_factory=set)
    floor_food: set[tuple[int, int]] = field(default_factory=set)
    collected_food: set[tuple[int, int]] = field(default_factory=set)
    # Level-scoped interaction tracking (resolves Flaws 4 & 6)
    visited_thrones: set[tuple[int, int]] = field(default_factory=set)
    disarmed_traps: set[tuple[int, int]] = field(default_factory=set)
    looted_containers: set[tuple[int, int]] = field(default_factory=set)
    collected_gold: set[tuple[int, int]] = field(default_factory=set)
    dead_end_searches: dict[tuple[int, int], int] = field(default_factory=dict)
    door_attempts: dict[tuple[int, int], int] = field(default_factory=dict)
    has_shops: bool = False
    shop_doors: set[tuple[int, int]] = field(default_factory=set)
    boulders: set[tuple[int, int]] = field(default_factory=set)
    blocked_tiles: set[tuple[int, int]] = field(default_factory=set)
    turns_spent: int = 0


class LevelStore(dict):
    """
    Composite dictionary mapping (dnum, dlevel) -> LevelMap,
    with backward compatibility for single-integer depth lookups.
    """
    def __getitem__(self, key):
        if isinstance(key, int):
            for (dnum, dlevel), lvl in self.items():
                if lvl.depth == key:
                    return lvl
            return super().__getitem__((0, key))
        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, int):
            for (dnum, dlevel), lvl in self.items():
                if lvl.depth == key:
                    return True
            return super().__contains__((0, key))
        return super().__contains__(key)


