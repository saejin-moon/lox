"""
Domain Layer: Spatial Navigation Manager.
Governs cost-weighted Grid A* pathfinding, frontier exploration, corridor dead-end
secret door searches, and stairs navigation.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats
from corp.navigation.astar import GridAStar
from corp.navigation.frontier import FrontierExplorer
from corp.planner.htn import Task, PrimitiveTask


@dataclass
class LevelMap:
    """Persistent 2D spatial representation for a single dungeon depth level."""
    depth: int
    walkable: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=bool))
    visited: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=np.int32))
    searched: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=np.int32))
    stairs_down: tuple[int, int] | None = None
    stairs_up: tuple[int, int] | None = None


class NavigationManager:
    """
    Directs spatial traversal and exploration across NetHack's 80x21 grid.
    Strictly enforces:
    1. Grid corner-clipping interlock (via GridAStar).
    2. Hazard cost weighting (traps, lava, water).
    3. Corridor dead-end secret door search bursts.
    4. Autonomous stair descent once floor frontiers are exhausted.
    """

    WALKABLE_CHARS = {
        ord("."),  # Room floor
        ord("#"),  # Corridor
        ord("+"),  # Door
        ord("<"),  # Stairs up
        ord(">"),  # Stairs down
        ord("_"),  # Altar
        ord("{"),  # Fountain
        ord("\\"), # Throne
        ord("^"),  # Trap
        ord("$"),  # Gold
        ord(")"),  # Weapon
        ord("["),  # Armor
        ord("?"),  # Scroll
        ord("!"),  # Potion
        ord("/"),  # Wand
        ord("="),  # Ring
        ord("\""), # Amulet
        ord("("),  # Tool
        ord("%"),  # Food
        ord("*"),  # Gem
    }

    def __init__(self):
        self.levels: dict[int, LevelMap] = {}
        self.astar = GridAStar()
        self.frontier_explorer = FrontierExplorer()
        self.dead_end_searches: dict[tuple[int, int], int] = {}

    def get_or_create_level(self, depth: int, shape: tuple[int, int] = (21, 79)) -> LevelMap:
        if depth not in self.levels:
            self.levels[depth] = LevelMap(
                depth=depth,
                walkable=np.zeros(shape, dtype=bool),
                visited=np.zeros(shape, dtype=np.int32),
                searched=np.zeros(shape, dtype=np.int32),
            )
        return self.levels[depth]

    def update_map(
        self,
        chars: np.ndarray,
        blstats: BottomLineStats,
    ) -> LevelMap:
        """
        Updates the internal topological map using current sensory chars.
        """
        lvl = self.get_or_create_level(blstats.depth, chars.shape)
        py, px = blstats.y, blstats.x

        rows, cols = chars.shape
        for r in range(rows):
            for c in range(cols):
                ch = int(chars[r, c])
                if ch in self.WALKABLE_CHARS:
                    lvl.walkable[r, c] = True
                    if ch == ord(">"):
                        lvl.stairs_down = (r, c)
                    elif ch == ord("<"):
                        lvl.stairs_up = (r, c)

        lvl.visited[py, px] += 1
        return lvl

    def evaluate_navigation_turn(
        self,
        chars: np.ndarray,
        blstats: BottomLineStats,
        force_descend: bool = False,
    ) -> Task:
        """
        Calculates the next navigation action:
        1. If on stairs down and ready to descend -> DESCEND.
        2. If dead-end corridor with unsearched neighbors -> SEARCH.
        3. If stairs down known and level explored -> path to stairs down.
        4. If unexplored frontiers remain -> path to nearest frontier.
        5. Fallback -> localized SEARCH or random walkable step.
        """
        lvl = self.update_map(chars, blstats)
        py, px = blstats.y, blstats.x

        # 1. On stairs down check
        if lvl.stairs_down == (py, px) and (force_descend or self._is_level_mapped(lvl, chars)):
            return Task("DESCEND", is_primitive=True)

        # 2. Corridor dead-end secret door search burst
        if chars[py, px] == ord("#") and self._is_corridor_dead_end(lvl, py, px):
            searches_here = self.dead_end_searches.get((py, px), 0)
            if searches_here < 4:
                self.dead_end_searches[(py, px)] = searches_here + 1
                return Task("SEARCH", is_primitive=True)

        # 3. Path to stairs down if level mapped or descending
        hazard_costs = self._compute_hazard_costs(chars, lvl)
        if lvl.stairs_down and (force_descend or self._is_level_mapped(lvl, chars)):
            if (py, px) != lvl.stairs_down:
                path = self.astar.find_path(
                    (py, px),
                    lvl.stairs_down,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                )
                if path and len(path) > 1:
                    next_node = path[1]
                    return Task("STEP", is_primitive=True, args={"delta": (next_node.row - py, next_node.col - px)})

        # 4. Path to nearest frontier
        frontier = self.frontier_explorer.find_nearest_frontier(
            (py, px),
            lvl.walkable,
            chars,
        )
        if frontier:
            path = self.astar.find_path(
                (py, px),
                frontier,
                lvl.walkable,
                hazard_costs=hazard_costs,
            )
            if path and len(path) > 1:
                next_node = path[1]
                return Task("STEP", is_primitive=True, args={"delta": (next_node.row - py, next_node.col - px)})

        # 5. Fallback: Localized search or step to least-visited neighbor
        best_delta = self._find_least_visited_step(py, px, lvl, chars)
        if best_delta is not None:
            return Task("STEP", is_primitive=True, args={"delta": best_delta})

        return Task("SEARCH", is_primitive=True)

    def _is_level_mapped(self, lvl: LevelMap, chars: np.ndarray) -> bool:
        """Determines if the current dungeon level has no remaining reachable frontiers."""
        # Simple heuristic: if we have visited > 120 tiles and known stairs down exist
        visited_count = int(np.count_nonzero(lvl.visited > 0))
        return visited_count >= 80 and lvl.stairs_down is not None

    def _is_corridor_dead_end(self, lvl: LevelMap, r: int, c: int) -> bool:
        """Checks if a corridor tile has only 1 walkable corridor neighbor."""
        walkable_neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < lvl.walkable.shape[0] and 0 <= nc < lvl.walkable.shape[1]:
                if lvl.walkable[nr, nc]:
                    walkable_neighbors += 1
        return walkable_neighbors <= 1

    def _compute_hazard_costs(self, chars: np.ndarray, lvl: LevelMap) -> np.ndarray:
        """Computes cost penalties for known traps, water, and loops."""
        costs = np.zeros(chars.shape, dtype=np.float32)
        # Trap penalty (+100)
        costs[chars == ord("^")] += 100.0
        # Visited loop damping (+2.0 per visit)
        costs += np.minimum(lvl.visited * 2.0, 50.0).astype(np.float32)
        return costs

    def _find_least_visited_step(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
    ) -> tuple[int, int] | None:
        """Finds adjacent walkable step with lowest visit count."""
        best_delta: tuple[int, int] | None = None
        min_visits = float("inf")

        # 8 directions
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    if lvl.walkable[nr, nc]:
                        visits = lvl.visited[nr, nc]
                        if visits < min_visits:
                            min_visits = visits
                            best_delta = (dr, dc)

        return best_delta
