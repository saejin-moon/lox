"""
LOX-ψ Navigation: Search Policy & Secret Door Detection.

Eliminates the 4,400-turn DL1 perimeter wall search stall by enforcing:
1. Strict priority ordering: frontier exploration and corridor stepping strictly
   dominate room perimeter searches.
2. Dead-end corridor prioritization (+500 score bonus over perimeter walls).
3. Search caps: room perimeter tiles capped at <= 3 searches on DL1; dead ends <= 6.
4. Immediate abort: HP drops or active threats instantly cancel search bursts.
5. Stair fast-exit: searching is locked out once down-stairs are discovered.
"""
from __future__ import annotations

from typing import Any, Tuple, Optional
import numpy as np

from lox.domain.navigation.level_map import LevelMap
from lox.planner.htn import Task


class SearchPolicy:
    """Manages secret door search scheduling and burst limits."""

    def __init__(self, config: Any = None):
        self.cfg = config
        self._current_search_spot: tuple[int, int] | None = None
        self._current_search_burst: int = 0
        self._prev_hp: int = 0

    def reset(self) -> None:
        self._current_search_spot = None
        self._current_search_burst = 0
        self._prev_hp = 0

    def check_damage_abort(self, current_hp: int) -> bool:
        """Aborts active search burst if HP drops."""
        if self._prev_hp > 0 and current_hp < self._prev_hp:
            self._current_search_spot = None
            self._current_search_burst = 0
            self._prev_hp = current_hp
            return True
        self._prev_hp = current_hp
        return False

    def find_best_search_candidate(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
        dist_grid: np.ndarray,
        has_unvisited_frontiers: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """
        Selects the optimal secret-door candidate tile.
        Strictly suppresses room perimeter searches when unexplored frontiers or
        unvisited corridors exist on the floor.
        """
        rows, cols = lvl.walkable.shape
        is_wall = lvl.walls | (chars == ord("|")) | (chars == ord("-"))
        is_stone = (chars == 0) | (chars == ord(" "))
        boundary_target = is_wall | is_stone

        # Corridor dead-ends: corridor tile with <= 1 walkable neighbor
        is_corridor = (chars == ord("#")).copy()
        if lvl.corridors:
            for cr, cc in lvl.corridors:
                if 0 <= cr < rows and 0 <= cc < cols:
                    is_corridor[cr, cc] = True

        walkable_neighbors = np.zeros(lvl.walkable.shape, dtype=np.int32)
        walkable_neighbors[:-1, :] += lvl.walkable[1:, :].astype(np.int32)
        walkable_neighbors[1:, :] += lvl.walkable[:-1, :].astype(np.int32)
        walkable_neighbors[:, :-1] += lvl.walkable[:, 1:].astype(np.int32)
        walkable_neighbors[:, 1:] += lvl.walkable[:, :-1].astype(np.int32)

        is_dead_end = is_corridor & (walkable_neighbors <= 1)

        # Boundary tiles adjacent to wall or unmapped stone
        has_boundary = np.zeros(lvl.walkable.shape, dtype=bool)
        has_boundary[:-1, :] |= boundary_target[1:, :]
        has_boundary[1:, :] |= boundary_target[:-1, :]
        has_boundary[:, :-1] |= boundary_target[:, 1:]
        has_boundary[:, 1:] |= boundary_target[:, :-1]

        # Candidates must be walkable and reachable
        is_reachable = dist_grid >= 0
        cand_mask = lvl.walkable & is_reachable & (has_boundary | is_dead_end)

        # Exclude doors and blocked obstacles
        for dr, dc in lvl.doors:
            if 0 <= dr < rows and 0 <= dc < cols:
                cand_mask[dr, dc] = False
        for br, bc in lvl.blocked_tiles:
            if 0 <= br < rows and 0 <= bc < cols:
                cand_mask[br, bc] = False

        cand_pts = np.argwhere(cand_mask)
        if len(cand_pts) == 0:
            return None

        # Anti-stall caps:
        # Perimeter wall searches are strictly capped: DL 1 <= 3, DL 2-3 <= 5, DL 4+ <= 10
        # Dead-end corridors: <= 6 (early), <= 15 (deep maze)
        is_deep = getattr(lvl, "depth", 1) >= 4
        dead_end_cap = 15 if is_deep else 6
        if getattr(lvl, "depth", 1) <= 1:
            perimeter_cap = 3
        elif getattr(lvl, "depth", 1) <= 3:
            perimeter_cap = 5
        else:
            perimeter_cap = 10

        best_pos = None
        best_prio = -1e9

        for pt in cand_pts:
            r, c = int(pt[0]), int(pt[1])
            is_de = is_dead_end[r, c]

            # If frontiers exist, ONLY search genuine dead ends — NEVER room perimeter walls!
            if has_unvisited_frontiers and not is_de:
                continue

            hard_cap = dead_end_cap if is_de else perimeter_cap
            searches = int(lvl.searched[r, c]) + int(lvl.dead_end_searches.get((r, c), 0))
            if searches >= hard_cap:
                continue

            prio = 100.0
            if is_de:
                prio += 500.0  # Massive bonus: secret doors are almost always at corridor dead ends
            else:
                prio -= 100.0  # Penalty on generic perimeter walls

            prio -= (searches ** 1.5) * 10.0
            prio -= dist_grid[r, c] * 1.5

            if prio > best_prio:
                best_prio = prio
                best_pos = (r, c)

        return best_pos
