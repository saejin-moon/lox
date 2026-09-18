"""
Cost-Weighted Grid A* Pathfinding for NetHack 80x21 dungeon grid.
Enforces 8-way movement, corner-clipping interlocks, and hazard cost penalties.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Tuple, List, Optional
import numpy as np
import numpy.typing as npt


# Direction vectors: (dr, dc, action_index, key_char)
DIRECTIONS: list[tuple[int, int, int, str]] = [
    (-1, 0, 0, "k"),   # North
    (0, 1, 1, "l"),    # East
    (1, 0, 2, "j"),    # South
    (0, -1, 3, "h"),   # West
    (-1, 1, 4, "u"),   # North-East
    (1, 1, 5, "n"),    # South-East
    (1, -1, 6, "b"),   # South-West
    (-1, -1, 7, "y"),  # North-West
]


@dataclass(slots=True, frozen=True)
class PathNode:
    row: int
    col: int
    action_index: int = 0
    key_char: str = ""


class GridAStar:
    """
    Finds optimal hazard-weighted paths on an 80x21 grid in <0.35 ms.
    """

    ROWS = 21
    COLS = 79
    SIZE = ROWS * COLS

    # Octile distance heuristic for 8-way movement
    SQRT2 = math.sqrt(2.0)

    # Reusable flat buffers to avoid per-call heap dict/set allocations
    _epoch = 0
    _visited_epoch = [0] * SIZE
    _seen_epoch = [0] * SIZE
    _g_scores = [float("inf")] * SIZE
    _parent_idx = [-1] * SIZE
    _parent_act = [-1] * SIZE

    # Precomputed direction tables
    DIR_DELTAS = [dr * 79 + dc for dr, dc, _, _ in DIRECTIONS]
    DIR_ACTS = [act for _, _, act, _ in DIRECTIONS]
    DIR_KEYS = [key for _, _, _, key in DIRECTIONS]
    DIR_IS_DIAG = [(dr != 0 and dc != 0) for dr, dc, _, _ in DIRECTIONS]
    DIR_DR = [dr for dr, dc, _, _ in DIRECTIONS]
    DIR_DC = [dc for dr, dc, _, _ in DIRECTIONS]

    @classmethod
    def octile_distance(cls, r1: int, c1: int, r2: int, c2: int) -> float:
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return float(dr + dc + (cls.SQRT2 - 2.0) * min(dr, dc))

    @classmethod
    def find_path(
        cls,
        start: tuple[int, int],  # (row, col)
        goal: tuple[int, int],   # (row, col)
        walkable_mask: npt.NDArray[np.bool_],  # Shape (21, 79), True if walkable
        hazard_costs: npt.NDArray[np.float64] | None = None,  # Shape (21, 79), extra costs
        doorway_mask: npt.NDArray[np.bool_] | None = None,    # Shape (21, 79), True if door
    ) -> Optional[List[PathNode]]:
        """
        Computes cost-weighted A* path from start to goal.
        Returns list of PathNode leading from start to goal, or None if unreachable.
        """
        start_r, start_c = start
        goal_r, goal_c = goal

        if not (0 <= start_r < cls.ROWS and 0 <= start_c < cls.COLS):
            return None
        if not (0 <= goal_r < cls.ROWS and 0 <= goal_c < cls.COLS):
            return None
        if start == goal:
            return []

        cls._epoch += 1
        epoch = cls._epoch
        if epoch >= 2000000000:
            cls._visited_epoch = [0] * cls.SIZE
            cls._seen_epoch = [0] * cls.SIZE
            cls._epoch = 1
            epoch = 1

        start_idx = start_r * cls.COLS + start_c
        goal_idx = goal_r * cls.COLS + goal_c

        cls._seen_epoch[start_idx] = epoch
        cls._g_scores[start_idx] = 0.0

        # Heuristic tie-breaking factor (1.0 + 1e-4) to prefer paths directly toward goal
        h_start = cls.octile_distance(start_r, start_c, goal_r, goal_c) * 1.0001
        open_set = [(h_start, 0.0, start_idx)]

        g_scores = cls._g_scores
        visited_epoch = cls._visited_epoch
        seen_epoch = cls._seen_epoch
        parent_idx = cls._parent_idx
        parent_act = cls._parent_act
        dir_keys = cls.DIR_KEYS

        # Ravel masks for zero-overhead 1D indexing
        walkable_flat = walkable_mask.ravel()
        hazard_flat = hazard_costs.ravel() if hazard_costs is not None else None
        doorway_flat = doorway_mask.ravel() if doorway_mask is not None else None

        rows = cls.ROWS
        cols = cls.COLS
        dir_dr = cls.DIR_DR
        dir_dc = cls.DIR_DC
        dir_deltas = cls.DIR_DELTAS
        dir_is_diag = cls.DIR_IS_DIAG
        dir_acts = cls.DIR_ACTS
        sqrt2 = cls.SQRT2
        oct_dist = cls.octile_distance

        while open_set:
            f, current_g, curr_idx = heapq.heappop(open_set)

            if visited_epoch[curr_idx] == epoch:
                continue
            visited_epoch[curr_idx] = epoch

            if curr_idx == goal_idx:
                path: list[PathNode] = []
                c_idx = goal_idx
                while c_idx != start_idx:
                    p_idx = parent_idx[c_idx]
                    act = parent_act[c_idx]
                    path.append(PathNode(row=c_idx // cols, col=c_idx % cols, action_index=act, key_char=dir_keys[act]))
                    c_idx = p_idx
                path.reverse()
                return path

            curr_r = curr_idx // cols
            curr_c = curr_idx % cols

            for i in range(8):
                dr = dir_dr[i]
                dc = dir_dc[i]
                nr = curr_r + dr
                nc = curr_c + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                n_idx = curr_idx + dir_deltas[i]
                if not walkable_flat[n_idx] and n_idx != goal_idx:
                    continue

                is_diag = dir_is_diag[i]
                if is_diag:
                    o1_idx = curr_idx + dr * cols
                    o2_idx = curr_idx + dc
                    if not walkable_flat[o1_idx] or not walkable_flat[o2_idx]:
                        continue
                    if doorway_flat is not None:
                        if doorway_flat[curr_idx] or doorway_flat[n_idx]:
                            continue

                step_cost = sqrt2 if is_diag else 1.0
                if hazard_flat is not None:
                    step_cost += float(hazard_flat[n_idx])

                tentative_g = current_g + step_cost

                if seen_epoch[n_idx] != epoch:
                    seen_epoch[n_idx] = epoch
                    g_scores[n_idx] = float("inf")

                if tentative_g < g_scores[n_idx]:
                    g_scores[n_idx] = tentative_g
                    parent_idx[n_idx] = curr_idx
                    parent_act[n_idx] = dir_acts[i]
                    f_score = tentative_g + oct_dist(nr, nc, goal_r, goal_c) * 1.0001
                    heapq.heappush(open_set, (f_score, tentative_g, n_idx))

        return None  # Unreachable
