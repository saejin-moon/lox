"""
Frontier Exploration: Identifies unexplored boundaries and secret door candidates on 80x21 grid.
"""

from collections import deque
from typing import Optional, List, Tuple
import numpy as np
import numpy.typing as npt


try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(fastmath=True, nogil=True)
def _bfs_frontier_kernel(
    start_idx: int,
    frontier_flat: np.ndarray,
    walkable_flat: np.ndarray,
    rows: int,
    cols: int,
    dir_dr: np.ndarray,
    dir_dc: np.ndarray,
    neighbor_deltas: np.ndarray,
    visited: np.ndarray,
    epoch: int,
) -> int:
    size = rows * cols
    queue = np.empty(size, dtype=np.int32)
    head = 0
    tail = 0

    visited[start_idx] = epoch
    queue[tail] = start_idx
    tail += 1

    while head < tail:
        curr_idx = queue[head]
        head += 1

        if frontier_flat[curr_idx]:
            return curr_idx

        curr_r = curr_idx // cols
        curr_c = curr_idx % cols

        for i in range(8):
            nr = curr_r + dir_dr[i]
            nc = curr_c + dir_dc[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                n_idx = curr_idx + neighbor_deltas[i]
                if walkable_flat[n_idx] and visited[n_idx] != epoch:
                    visited[n_idx] = epoch
                    queue[tail] = n_idx
                    tail += 1

    return -1


# Warmup JIT at import
try:
    _dummy_f = np.zeros(1659, dtype=bool)
    _dummy_w = np.ones(1659, dtype=bool)
    _dummy_v = np.zeros(1659, dtype=np.int32)
    _dummy_dr = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
    _dummy_dc = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
    _dummy_d = _dummy_dr * 79 + _dummy_dc
    _bfs_frontier_kernel(0, _dummy_f, _dummy_w, 21, 79, _dummy_dr, _dummy_dc, _dummy_d, _dummy_v, 1)
except Exception:
    pass


class FrontierExplorer:
    """
    Identifies unexplored boundary tiles and schedules BFS exploration goals.
    """

    ROWS = 21
    COLS = 79
    SIZE = ROWS * COLS

    CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ALL_NEIGHBORS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    NEIGHBOR_DELTAS = np.array([dr * 79 + dc for dr, dc in ALL_NEIGHBORS], dtype=np.int32)
    DIR_DR = np.array([dr for dr, _ in ALL_NEIGHBORS], dtype=np.int32)
    DIR_DC = np.array([dc for _, dc in ALL_NEIGHBORS], dtype=np.int32)

    _bfs_epoch = 0
    _bfs_visited = np.zeros(SIZE, dtype=np.int32)

    @classmethod
    def get_frontier_mask(
        cls,
        walkable_mask: npt.NDArray[np.bool_],
        unmapped_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.bool_]:
        """
        Returns a boolean mask where True indicates a walkable tile adjacent to unmapped space.
        Optimized via 2D morphological dilation with 8 directional slice shifts.
        """
        rows, cols = cls.ROWS, cls.COLS
        unmapped_dilated = np.zeros((rows, cols), dtype=bool)

        # 8 directional neighbor shifts
        unmapped_dilated[1:, :] |= unmapped_mask[:-1, :]
        unmapped_dilated[:-1, :] |= unmapped_mask[1:, :]
        unmapped_dilated[:, 1:] |= unmapped_mask[:, :-1]
        unmapped_dilated[:, :-1] |= unmapped_mask[:, 1:]
        unmapped_dilated[1:, 1:] |= unmapped_mask[:-1, :-1]
        unmapped_dilated[1:, :-1] |= unmapped_mask[:-1, 1:]
        unmapped_dilated[:-1, 1:] |= unmapped_mask[1:, :-1]
        unmapped_dilated[:-1, :-1] |= unmapped_mask[1:, 1:]

        return walkable_mask & unmapped_dilated

    @classmethod
    def find_nearest_frontier(
        cls,
        start: tuple[int, int],
        walkable_mask: npt.NDArray[np.bool_],
        unmapped_mask: npt.NDArray[np.bool_],
    ) -> Optional[Tuple[int, int]]:
        """
        Runs BFS from player start position to find the closest reachable frontier tile.
        """
        frontier_mask = cls.get_frontier_mask(walkable_mask, unmapped_mask)

        start_r, start_c = start
        if not (0 <= start_r < cls.ROWS and 0 <= start_c < cls.COLS):
            return None

        start_idx = start_r * cls.COLS + start_c
        frontier_flat = frontier_mask.ravel()
        if frontier_flat[start_idx]:
            return start

        walkable_flat = walkable_mask.ravel()

        cls._bfs_epoch += 1
        epoch = cls._bfs_epoch
        if not isinstance(cls._bfs_visited, np.ndarray) or epoch >= 2000000000 or epoch == 1:
            cls._bfs_visited = np.zeros(cls.SIZE, dtype=np.int32)
            cls._bfs_epoch = 1
            epoch = 1

        target_idx = _bfs_frontier_kernel(
            start_idx=start_idx,
            frontier_flat=frontier_flat,
            walkable_flat=walkable_flat,
            rows=cls.ROWS,
            cols=cls.COLS,
            dir_dr=cls.DIR_DR,
            dir_dc=cls.DIR_DC,
            neighbor_deltas=cls.NEIGHBOR_DELTAS,
            visited=cls._bfs_visited,
            epoch=epoch,
        )

        if target_idx >= 0:
            return (int(target_idx // cls.COLS), int(target_idx % cls.COLS))
        return None

    @classmethod
    def is_corridor_dead_end(
        cls,
        pos: tuple[int, int],
        walkable_mask: npt.NDArray[np.bool_],
    ) -> bool:
        """
        Returns True if pos is a walkable tile with exactly 1 walkable neighbor.
        Dead-end corridors are prime candidates for secret doors in NetHack.
        """
        r, c = pos
        if not (0 <= r < cls.ROWS and 0 <= c < cls.COLS):
            return False
        if not walkable_mask[r, c]:
            return False

        walkable_neighbors = 0
        for dr, dc in cls.CARDINALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cls.ROWS and 0 <= nc < cls.COLS:
                if walkable_mask[nr, nc]:
                    walkable_neighbors += 1

        return walkable_neighbors <= 1
