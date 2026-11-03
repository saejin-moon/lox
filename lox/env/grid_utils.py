"""
Generic Numba-accelerated 2D Grid Math & Pathfinding Utilities ("Hard Code Nothing").
Zero game rules: pure numerical BFS distance transforms and A* delta extraction.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def bfs_distance_grid(
    passable: np.ndarray,
    sy: int,
    sx: int,
    allow_diagonal: bool = True,
    no_diag: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes shortest-path step distances from (sy, sx) across any boolean `passable` 2D grid.
    Unreachable tiles have distance -1.
    If `no_diag` is provided, diagonal transitions into or out of True tiles are skipped.
    """
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if not (0 <= sy < h and 0 <= sx < w):
        return dist
    dist[sy, sx] = 0
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    qy[tail] = sy
    qx[tail] = sx
    tail += 1

    if allow_diagonal:
        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        n_dirs = 8
    else:
        dys = (-1, 1, 0, 0, 0, 0, 0, 0)
        dxs = (0, 0, -1, 1, 0, 0, 0, 0)
        n_dirs = 4

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        head += 1
        cd = dist[cy, cx] + 1
        for i in range(n_dirs):
            ny = cy + dys[i]
            nx = cx + dxs[i]
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1 and passable[ny, nx]:
                if i >= 4 and no_diag is not None:
                    if no_diag[cy, cx] or no_diag[ny, nx]:
                        continue
                dist[ny, nx] = cd
                qy[tail] = ny
                qx[tail] = nx
                tail += 1
    return dist


@numba.njit(cache=True)
def astar_next_step(
    passable: np.ndarray,
    sy: int,
    sx: int,
    ty: int,
    tx: int,
    allow_diagonal: bool = True,
    no_diag: np.ndarray | None = None,
) -> tuple[int, int]:
    """
    Returns the first step delta `(dy, dx)` along the shortest path from `(sy, sx)` to `(ty, tx)`
    on `passable`. Returns `(0, 0)` if already at target or unreachable.
    If `no_diag` is provided, diagonal transitions into or out of True tiles are skipped.
    """
    if sy == ty and sx == tx:
        return (0, 0)
    h, w = passable.shape
    if not (0 <= sy < h and 0 <= sx < w and 0 <= ty < h and 0 <= tx < w):
        return (0, 0)

    # Run BFS backwards from target (ty, tx) so first step from (sy, sx) is O(1) neighbor lookup
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[ty, tx] = 0
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    qy[tail] = ty
    qx[tail] = tx
    tail += 1

    if allow_diagonal:
        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        n_dirs = 8
    else:
        dys = (-1, 1, 0, 0, 0, 0, 0, 0)
        dxs = (0, 0, -1, 1, 0, 0, 0, 0)
        n_dirs = 4

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        head += 1
        if cy == sy and cx == sx:
            break
        cd = dist[cy, cx] + 1
        for i in range(n_dirs):
            ny = cy + dys[i]
            nx = cx + dxs[i]
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1:
                if i >= 4 and no_diag is not None:
                    if no_diag[cy, cx] or no_diag[ny, nx]:
                        continue
                if passable[ny, nx] or (ny == sy and nx == sx):
                    dist[ny, nx] = cd
                    qy[tail] = ny
                    qx[tail] = nx
                    tail += 1

    best_d = 999999
    best_dy = 0
    best_dx = 0
    for i in range(n_dirs):
        ny = sy + dys[i]
        nx = sx + dxs[i]
        if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] != -1:
            if i >= 4 and no_diag is not None:
                if no_diag[sy, sx] or no_diag[ny, nx]:
                    continue
            if dist[ny, nx] < best_d:
                best_d = dist[ny, nx]
                best_dy = dys[i]
                best_dx = dxs[i]
    return (best_dy, best_dx)
