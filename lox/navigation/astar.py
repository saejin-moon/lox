"""
Cost-Weighted Grid A* Pathfinding for NetHack 80x21 dungeon grid.
Enforces 8-way movement, corner-clipping interlocks, and hazard cost penalties.
Accelerated with Numba JIT (<20 µs/call) with zero-overhead pure Python fallback.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Tuple, List, Optional
import numpy as np
import numpy.typing as npt

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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

_DIR_DELTAS = np.array([dr * 79 + dc for dr, dc, _, _ in DIRECTIONS], dtype=np.int32)
_DIR_ACTS = np.array([act for _, _, act, _ in DIRECTIONS], dtype=np.int32)
_DIR_KEYS = [key for _, _, _, key in DIRECTIONS]
_DIR_IS_DIAG = np.array([(dr != 0 and dc != 0) for dr, dc, _, _ in DIRECTIONS], dtype=np.bool_)
_DIR_DR = np.array([dr for dr, dc, _, _ in DIRECTIONS], dtype=np.int32)
_DIR_DC = np.array([dc for dr, dc, _, _ in DIRECTIONS], dtype=np.int32)
_SQRT2 = math.sqrt(2.0)


@dataclass(slots=True, frozen=True)
class PathNode:
    row: int
    col: int
    action_index: int = 0
    key_char: str = ""


if _HAS_NUMBA:
    @numba.njit(fastmath=True, nogil=True)
    def _octile_distance(r1: int, c1: int, r2: int, c2: int) -> float:
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return float(dr + dc + (_SQRT2 - 2.0) * min(dr, dc))

    @numba.njit(fastmath=True, nogil=True)
    def _astar_kernel(
        start_r: int, start_c: int, goal_r: int, goal_c: int,
        walkable_flat: np.ndarray,
        hazard_flat: np.ndarray,
        doorway_flat: np.ndarray,
        has_hazard: bool,
        has_doorway: bool,
        visited: np.ndarray,
        seen: np.ndarray,
        g_scores: np.ndarray,
        parent_idx: np.ndarray,
        parent_act: np.ndarray,
        epoch: int,
    ) -> int:
        start_idx = start_r * 79 + start_c
        goal_idx = goal_r * 79 + goal_c

        seen[start_idx] = epoch
        g_scores[start_idx] = 0.0

        h_start = _octile_distance(start_r, start_c, goal_r, goal_c) * 1.0001
        open_set = [(h_start, 0.0, start_idx)]

        while len(open_set) > 0:
            f, current_g, curr_idx = heapq.heappop(open_set)

            if visited[curr_idx] == epoch:
                continue
            visited[curr_idx] = epoch

            if curr_idx == goal_idx:
                return 1  # Success

            curr_r = curr_idx // 79
            curr_c = curr_idx % 79

            for i in range(8):
                dr = _DIR_DR[i]
                dc = _DIR_DC[i]
                nr = curr_r + dr
                nc = curr_c + dc

                if not (0 <= nr < 21 and 0 <= nc < 79):
                    continue

                n_idx = curr_idx + _DIR_DELTAS[i]
                if not walkable_flat[n_idx] and n_idx != goal_idx:
                    continue

                is_diag = _DIR_IS_DIAG[i]
                if is_diag:
                    o1_idx = curr_idx + dr * 79
                    o2_idx = curr_idx + dc
                    if not walkable_flat[o1_idx] or not walkable_flat[o2_idx]:
                        continue
                    if has_doorway:
                        if doorway_flat[curr_idx] or doorway_flat[n_idx]:
                            continue

                step_cost = _SQRT2 if is_diag else 1.0
                if has_hazard:
                    step_cost += float(hazard_flat[n_idx])

                tentative_g = current_g + step_cost

                if seen[n_idx] != epoch:
                    seen[n_idx] = epoch
                    g_scores[n_idx] = 1e9

                if tentative_g < g_scores[n_idx]:
                    g_scores[n_idx] = tentative_g
                    parent_idx[n_idx] = curr_idx
                    parent_act[n_idx] = _DIR_ACTS[i]
                    f_score = tentative_g + _octile_distance(nr, nc, goal_r, goal_c) * 1.0001
                    heapq.heappush(open_set, (f_score, tentative_g, n_idx))

        return 0  # Unreachable
else:
    def _octile_distance(r1: int, c1: int, r2: int, c2: int) -> float:
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return float(dr + dc + (_SQRT2 - 2.0) * min(dr, dc))


class GridAStar:
    """
    Finds optimal hazard-weighted paths on an 80x21 grid in <20 µs via Numba JIT.
    """

    ROWS = 21
    COLS = 79
    SIZE = ROWS * COLS
    SQRT2 = _SQRT2

    # Reusable flat buffers
    _epoch = 0
    _visited_np = np.zeros(SIZE, dtype=np.int32)
    _seen_np = np.zeros(SIZE, dtype=np.int32)
    _g_scores_np = np.zeros(SIZE, dtype=np.float64)
    _parent_idx_np = np.zeros(SIZE, dtype=np.int32)
    _parent_act_np = np.zeros(SIZE, dtype=np.int32)

    # Empty dummy buffers for hazard / doorway flags
    _empty_float_flat = np.zeros(SIZE, dtype=np.float64)
    _empty_bool_flat = np.zeros(SIZE, dtype=np.bool_)

    # Precomputed direction tables for compatibility
    DIR_DELTAS = list(_DIR_DELTAS)
    DIR_ACTS = list(_DIR_ACTS)
    DIR_KEYS = _DIR_KEYS
    DIR_IS_DIAG = list(_DIR_IS_DIAG)
    DIR_DR = list(_DIR_DR)
    DIR_DC = list(_DIR_DC)

    @classmethod
    def octile_distance(cls, r1: int, c1: int, r2: int, c2: int) -> float:
        return _octile_distance(r1, c1, r2, c2)

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
        if epoch >= 2000000000 or epoch == 1:
            cls._visited_np.fill(0)
            cls._seen_np.fill(0)
            cls._epoch = 1
            epoch = 1

        walkable_flat = np.ascontiguousarray(walkable_mask).ravel()
        has_hazard = hazard_costs is not None
        hazard_flat = np.ascontiguousarray(hazard_costs).ravel() if has_hazard else cls._empty_float_flat

        has_doorway = doorway_mask is not None
        doorway_flat = np.ascontiguousarray(doorway_mask).ravel() if has_doorway else cls._empty_bool_flat

        if _HAS_NUMBA:
            found = _astar_kernel(
                start_r, start_c, goal_r, goal_c,
                walkable_flat, hazard_flat, doorway_flat,
                has_hazard, has_doorway,
                cls._visited_np, cls._seen_np, cls._g_scores_np,
                cls._parent_idx_np, cls._parent_act_np,
                epoch,
            )
            if not found:
                return None

            start_idx = start_r * cls.COLS + start_c
            goal_idx = goal_r * cls.COLS + goal_c
            p_idx = cls._parent_idx_np
            p_act = cls._parent_act_np
            cols = cls.COLS
            dir_keys = cls.DIR_KEYS

            path: list[PathNode] = []
            c_idx = goal_idx
            while c_idx != start_idx:
                parent = int(p_idx[c_idx])
                act = int(p_act[c_idx])
                path.append(PathNode(row=int(c_idx // cols), col=int(c_idx % cols), action_index=act, key_char=dir_keys[act]))
                c_idx = parent
            path.reverse()
            return path

        # Fallback pure-Python path if Numba is unavailable
        start_idx = start_r * cls.COLS + start_c
        goal_idx = goal_r * cls.COLS + goal_c
        cls._seen_np[start_idx] = epoch
        cls._g_scores_np[start_idx] = 0.0

        h_start = _octile_distance(start_r, start_c, goal_r, goal_c) * 1.0001
        open_set = [(h_start, 0.0, start_idx)]

        visited = cls._visited_np
        seen = cls._seen_np
        g_scores = cls._g_scores_np
        p_idx = cls._parent_idx_np
        p_act = cls._parent_act_np
        dir_keys = cls.DIR_KEYS
        cols = cls.COLS

        while open_set:
            f, current_g, curr_idx = heapq.heappop(open_set)
            if visited[curr_idx] == epoch:
                continue
            visited[curr_idx] = epoch

            if curr_idx == goal_idx:
                path = []
                c_idx = goal_idx
                while c_idx != start_idx:
                    parent = p_idx[c_idx]
                    act = p_act[c_idx]
                    path.append(PathNode(row=c_idx // cols, col=c_idx % cols, action_index=act, key_char=dir_keys[act]))
                    c_idx = parent
                path.reverse()
                return path

            curr_r = curr_idx // cols
            curr_c = curr_idx % cols

            for i in range(8):
                dr = _DIR_DR[i]
                dc = _DIR_DC[i]
                nr = curr_r + dr
                nc = curr_c + dc

                if not (0 <= nr < cls.ROWS and 0 <= nc < cols):
                    continue

                n_idx = curr_idx + _DIR_DELTAS[i]
                if not walkable_flat[n_idx] and n_idx != goal_idx:
                    continue

                is_diag = _DIR_IS_DIAG[i]
                if is_diag:
                    o1_idx = curr_idx + dr * cols
                    o2_idx = curr_idx + dc
                    if not walkable_flat[o1_idx] or not walkable_flat[o2_idx]:
                        continue
                    if has_doorway and (doorway_flat[curr_idx] or doorway_flat[n_idx]):
                        continue

                step_cost = _SQRT2 if is_diag else 1.0
                if has_hazard:
                    step_cost += float(hazard_flat[n_idx])

                tentative_g = current_g + step_cost
                if seen[n_idx] != epoch:
                    seen[n_idx] = epoch
                    g_scores[n_idx] = float("inf")

                if tentative_g < g_scores[n_idx]:
                    g_scores[n_idx] = tentative_g
                    p_idx[n_idx] = curr_idx
                    p_act[n_idx] = _DIR_ACTS[i]
                    f_score = tentative_g + _octile_distance(nr, nc, goal_r, goal_c) * 1.0001
                    heapq.heappush(open_set, (f_score, tentative_g, n_idx))

        return None
