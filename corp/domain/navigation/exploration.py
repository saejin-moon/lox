"""Frontier exploration, search-tile selection, hazard costs — extracted from navigation_manager."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from nle import nethack
from corp.navigation.astar import GridAStar, PathNode
from corp.navigation.frontier import FrontierExplorer
from corp.planner.htn import Task, PrimitiveTask
from corp.planner.guards import HTNGuards
from corp.domain.navigation.level_map import LevelMap


class ExplorationMixin:
    def find_nearest_unvisited(self, py: int, px: int, lvl: LevelMap) -> tuple[int, int] | None:
        """Runs BFS on walkable grid to find the closest reachable unvisited walkable tile."""
        # Stairs down is excluded so the agent explores the floor before diving
        cand_unvisited = lvl.walkable & (lvl.visited == 0)
        if 0 <= py < cand_unvisited.shape[0] and 0 <= px < cand_unvisited.shape[1]:
            cand_unvisited[py, px] = False
        if lvl.stairs_down is not None:
            cand_unvisited[lvl.stairs_down[0], lvl.stairs_down[1]] = False
        if not np.any(cand_unvisited):
            return None

        type(self)._bfs_epoch += 1
        epoch = type(self)._bfs_epoch
        if epoch >= 2000000000:
            type(self)._bfs_visited = [0] * type(self).SIZE
            type(self)._bfs_epoch = 1
            epoch = 1

        cols = type(self).COLS
        rows = type(self).ROWS
        start_idx = py * cols + px
        type(self)._bfs_visited[start_idx] = epoch

        from collections import deque
        queue = deque([start_idx])
        cand_flat = cand_unvisited.ravel()
        walkable_flat = lvl.walkable.ravel()
        dir_dr = type(self).DIR_DR
        dir_dc = type(self).DIR_DC
        neighbor_deltas = type(self).NEIGHBOR_DELTAS
        visited = type(self)._bfs_visited

        while queue:
            curr_idx = queue.popleft()
            if cand_flat[curr_idx]:
                return (curr_idx // cols, curr_idx % cols)

            curr_r = curr_idx // cols
            curr_c = curr_idx % cols

            for i in range(8):
                nr = curr_r + dir_dr[i]
                nc = curr_c + dir_dc[i]
                if 0 <= nr < rows and 0 <= nc < cols:
                    n_idx = curr_idx + neighbor_deltas[i]
                    if walkable_flat[n_idx] and visited[n_idx] != epoch:
                        visited[n_idx] = epoch
                        queue.append(n_idx)
        return None

    def _compute_bfs_distances(self, py: int, px: int, lvl: LevelMap) -> np.ndarray:
        """
        Computes 8-way BFS step distances from (py, px) across lvl.walkable in <0.04 ms.
        Enforces diagonal corner-cutting interlocks. Unreachable tiles have value -1.
        """
        rows = type(self).ROWS
        cols = type(self).COLS
        dist_grid = np.full((rows, cols), -1, dtype=np.int32)
        if not (0 <= py < rows and 0 <= px < cols) or not lvl.walkable[py, px]:
            return dist_grid

        from collections import deque
        queue = deque([(py, px)])
        dist_grid[py, px] = 0

        dir_dr = type(self).DIR_DR
        dir_dc = type(self).DIR_DC
        walkable = lvl.walkable

        while queue:
            cr, cc = queue.popleft()
            cd = dist_grid[cr, cc]

            for i in range(8):
                dr, dc = dir_dr[i], dir_dc[i]
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dist_grid[nr, nc] == -1 and walkable[nr, nc]:
                        if dr != 0 and dc != 0:
                            if not walkable[cr + dr, cc] or not walkable[cr, cc + dc]:
                                continue
                        dist_grid[nr, nc] = cd + 1
                        queue.append((nr, nc))

        return dist_grid



    def find_best_search_tile(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
        dist_grid: np.ndarray | None = None,
    ) -> tuple[int, int] | None:
        """
        AutoAscend-inspired prioritized secret door searcher.
        Identifies reachable walkable tiles adjacent to walls or unmapped stone,
        or corridor dead ends, prioritizing least-searched tiles with corridor dead-end bonuses.
        """
        if dist_grid is None:
            dist_grid = self._compute_bfs_distances(py, px, lvl)

        rows, cols = lvl.walkable.shape
        is_wall = lvl.walls | (chars == ord("|")) | (chars == ord("-"))
        is_stone = (chars == 0) | (chars == ord(" "))
        boundary_target = is_wall | is_stone

        # Shift in 4 cardinal directions to find walkable tiles adjacent to wall/stone
        has_boundary = np.zeros(lvl.walkable.shape, dtype=bool)
        has_boundary[:-1, :] |= boundary_target[1:, :]   # North of boundary
        has_boundary[1:, :] |= boundary_target[:-1, :]   # South of boundary
        has_boundary[:, :-1] |= boundary_target[:, 1:]   # West of boundary
        has_boundary[:, 1:] |= boundary_target[:, :-1]   # East of boundary

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

        # Candidates must be walkable AND REACHABLE!
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

        # Hard cumulative cap (stall fix): a tile searched up to the adaptive limit is
        # permanently excluded so a lone dead-end candidate can never loop forever
        s = self.cfg.search
        if lvl.stairs_down is None:
            hard_cap = s.hard_cap_deep if lvl.depth >= 4 else s.hard_cap_early
        else:
            hard_cap = s.hard_cap_stairs

        best_pos = None
        best_prio = -1e9

        for pt in cand_pts:
            r, c = int(pt[0]), int(pt[1])
            searches = int(lvl.searched[r, c]) + int(lvl.dead_end_searches.get((r, c), 0))
            if searches >= hard_cap:
                continue
            prio = 100.0
            if is_dead_end[r, c]:
                prio += 250.0
            prio -= (searches ** 1.5) * 6.0
            dist = dist_grid[r, c]
            prio -= dist * 1.5
            if prio > best_prio:
                best_prio = prio
                best_pos = (r, c)

        return best_pos

    def _find_unsearched_dead_end(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
    ) -> tuple[int, int] | None:
        """Finds the closest reachable corridor dead end that has been searched fewer than max_searches times."""
        # Adaptive limits (Phase 3): maze levels on DL 4+ need deep search bursts to find stairs
        s = self.cfg.search
        if lvl.stairs_down is None:
            max_searches = s.dead_end_none_deep if lvl.depth >= 4 else s.dead_end_none
        else:
            max_searches = s.dead_end_stairs
        # Burst search: if already standing on a dead end, continue searching up to max_searches times
        if (py, px) in lvl.corridors and self._is_corridor_dead_end(lvl, py, px):
            if lvl.dead_end_searches.get((py, px), 0) < max_searches:
                return (py, px)

        pts = lvl.corridors if lvl.corridors else [(int(pt[0]), int(pt[1])) for pt in np.argwhere(lvl.walkable & (chars == ord("#")))]
        candidates = []
        for r, c in pts:
            if self._is_corridor_dead_end(lvl, r, c):
                if lvl.dead_end_searches.get((r, c), 0) < max_searches:
                    candidates.append((r, c))
        if not candidates:
            return None
        candidates.sort(key=lambda p: abs(p[0] - py) + abs(p[1] - px))
        return candidates[0]

    def _is_level_mapped(self, lvl: LevelMap, chars: np.ndarray) -> bool:
        """Determines if the current dungeon level has no remaining reachable frontiers."""
        visited_count = int(np.count_nonzero(lvl.visited > 0))
        return visited_count >= 35 and lvl.stairs_down is not None

    def _is_corridor_dead_end(self, lvl: LevelMap, r: int, c: int) -> bool:
        """Checks if a corridor tile has only 1 walkable corridor neighbor."""
        walkable_neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < lvl.walkable.shape[0] and 0 <= nc < lvl.walkable.shape[1]:
                if lvl.walkable[nr, nc]:
                    walkable_neighbors += 1
        return walkable_neighbors <= 1

    def _compute_hazard_costs(
        self, chars: np.ndarray, lvl: LevelMap, glyphs: np.ndarray | None = None
    ) -> np.ndarray:
        """Computes cost penalties for known traps, water, loops, and hostile monsters."""
        costs = np.zeros(chars.shape, dtype=np.float32)
        nav = self.cfg.navigation
        # Trap penalty (+100 for visible ^, +500 for known lvl.traps)
        costs[chars == ord("^")] += nav.trap_cost
        for tr, tc in lvl.traps:
            if (tr, tc) not in lvl.disarmed_traps:
                costs[tr, tc] += nav.trap_cost
        # Visited loop damping (+2.0 per visit)
        costs += np.minimum(lvl.visited * nav.visit_cost_per, nav.visit_cost_cap).astype(np.float32)
        if glyphs is not None:
            # Monster/Pet tile penalty (+1000) so A* routes around entities during exploration
            mon_mask = (
                (glyphs >= nethack.GLYPH_MON_OFF) & (glyphs < nethack.GLYPH_OBJ_OFF)
            )
            costs[mon_mask] += self.cfg.navigation.monster_cost
        return costs

    def _find_least_visited_step(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
        glyphs: np.ndarray | None = None,
    ) -> tuple[int, int] | None:
        """Finds adjacent walkable step with lowest visit count, strictly respecting movement rules."""
        best_delta: tuple[int, int] | None = None
        min_visits = float("inf")

        # Prioritize 4 cardinal directions first, then 4 diagonals
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            nr, nc = py + dr, px + dc
            if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                if lvl.walkable[nr, nc]:
                    # Never step into another human (@) or any monster during fallback navigation
                    if chars[nr, nc] == ord("@") and (nr, nc) != (py, px):
                        continue
                    if glyphs is not None:
                        g = int(glyphs[nr, nc])
                        if nethack.glyph_is_monster(g):
                            continue
                    # Corner clipping interlock: diagonal step through wall corner forbidden
                    if dr != 0 and dc != 0:
                        if not lvl.walkable[py + dr, px] or not lvl.walkable[py, px + dc]:
                            continue
                        # Doorway diagonal interlock: forbidden into/out of door tiles
                        if (
                            chars[py, px] in (ord("+"), ord("'"))
                            or chars[nr, nc] in (ord("+"), ord("'"))
                            or (py, px) in lvl.doors
                            or (nr, nc) in lvl.doors
                        ):
                            continue
                    visits = lvl.visited[nr, nc]
                    if self.last_step_delta is not None and (dr, dc) == (-self.last_step_delta[0], -self.last_step_delta[1]):
                        visits += 20
                    if visits < min_visits:
                        min_visits = visits
                        best_delta = (dr, dc)

        return best_delta

    def _find_unsearched_wall_tile(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
    ) -> tuple[int, int] | None:
        """
        Finds the closest walkable room tile adjacent to a wall (| or -)
        that has been searched fewer than max_searches times, using vectorized morphological dilation.
        Excludes corridors (#) to prevent endless search bursts while navigating corridors.
        """
        # Adaptive limits (Phase 3): DL 4+ maze levels hide secret doors requiring 15-25 searches
        s = self.cfg.search
        if lvl.stairs_down is None:
            max_searches = s.wall_none_deep if lvl.depth >= 4 else s.wall_none
        else:
            max_searches = s.wall_stairs
        is_wall = lvl.walls | (chars == ord("|")) | (chars == ord("-"))
        has_wall = np.zeros_like(is_wall)
        has_wall[1:, :] |= is_wall[:-1, :]
        has_wall[:-1, :] |= is_wall[1:, :]
        has_wall[:, 1:] |= is_wall[:, :-1]
        has_wall[:, :-1] |= is_wall[:, 1:]

        # Strictly room tiles only: never search corridors (#) as walls
        is_room = (chars == ord(".")) | ((chars == ord("@")) & (chars != ord("#")))
        if lvl.corridors:
            for cr, cc in lvl.corridors:
                is_room[cr, cc] = False

        # Burst searching: if standing on a room wall tile searched fewer than max_searches times, stay and search
        if 0 <= py < has_wall.shape[0] and 0 <= px < has_wall.shape[1]:
            if is_room[py, px] and has_wall[py, px] and lvl.searched[py, px] < max_searches:
                return (py, px)

        cand_mask = is_room & (lvl.searched < max_searches) & has_wall & lvl.walkable
        cand_indices = np.argwhere(cand_mask)
        if len(cand_indices) == 0:
            return None

        dists = np.abs(cand_indices[:, 0] - py) + np.abs(cand_indices[:, 1] - px)
        best_idx = int(np.argmin(dists))
        return (int(cand_indices[best_idx, 0]), int(cand_indices[best_idx, 1]))

    def _get_candidate_wall_tiles(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Returns candidate room wall tiles sorted by Manhattan distance."""
        # Adaptive limits (Phase 3): DL 4+ maze levels hide secret doors requiring 15-25 searches
        s = self.cfg.search
        if lvl.stairs_down is None:
            max_searches = s.wall_none_deep if lvl.depth >= 4 else s.wall_none
        else:
            max_searches = s.wall_stairs
        is_wall = lvl.walls | (chars == ord("|")) | (chars == ord("-"))
        has_wall = np.zeros_like(is_wall)
        has_wall[1:, :] |= is_wall[:-1, :]
        has_wall[:-1, :] |= is_wall[1:, :]
        has_wall[:, 1:] |= is_wall[:, :-1]
        has_wall[:, :-1] |= is_wall[:, 1:]

        is_room = (chars == ord(".")) | ((chars == ord("@")) & (chars != ord("#")))
        if lvl.corridors:
            for cr, cc in lvl.corridors:
                is_room[cr, cc] = False

        cand_mask = is_room & (lvl.searched < max_searches) & has_wall & lvl.walkable
        cand_indices = np.argwhere(cand_mask)
        if len(cand_indices) == 0:
            return []

        dists = np.abs(cand_indices[:, 0] - py) + np.abs(cand_indices[:, 1] - px)
        sorted_order = np.argsort(dists)
        return [(int(cand_indices[i, 0]), int(cand_indices[i, 1])) for i in sorted_order]

