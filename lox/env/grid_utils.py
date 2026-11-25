"""
High-throughput 2D Grid Math & Pathfinding Utilities ("Hard Code Nothing").
Zero game heuristics: pure numerical BFS distance transforms and A* delta extraction.
Accelerated via native PyO3 Rust `lox_core` when available, with seamless Numba fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

lox_core: Any = None
try:
    import lox_core as _lox_core_mod

    lox_core = _lox_core_mod
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

_NUMBA_BFS_KERNEL: Any = None
_NUMBA_ASTAR_KERNEL: Any = None


def _get_numba_bfs_kernel() -> Any:
    global _NUMBA_BFS_KERNEL
    if _NUMBA_BFS_KERNEL is None:
        import numba

        @numba.njit(cache=True)
        def _numba_bfs_distance_grid(
            passable: np.ndarray,
            sy: int,
            sx: int,
            allow_diagonal: bool = True,
            no_diag: np.ndarray | None = None,
        ) -> np.ndarray:
            """Computes shortest-path step distances from (sy, sx) using Numba JIT."""
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
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and dist[ny, nx] == -1
                        and passable[ny, nx]
                    ):
                        if i >= 4 and no_diag is not None:
                            if no_diag[cy, cx] or no_diag[ny, nx]:
                                continue
                        dist[ny, nx] = cd
                        qy[tail] = ny
                        qx[tail] = nx
                        tail += 1
            return dist

        _NUMBA_BFS_KERNEL = _numba_bfs_distance_grid
    return _NUMBA_BFS_KERNEL


def _numba_bfs_distance_grid(
    passable: np.ndarray,
    sy: int,
    sx: int,
    allow_diagonal: bool = True,
    no_diag: np.ndarray | None = None,
) -> np.ndarray:
    return _get_numba_bfs_kernel()(passable, sy, sx, allow_diagonal, no_diag)


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
    Delegates to SIMD-optimized `lox_core` when available, falling back to Numba.
    """
    if HAS_RUST_CORE:
        try:
            # Ensure contiguous boolean array for PyO3 numpy (zero-copy when already bool & C-contiguous)
            c_passable = (
                passable
                if isinstance(passable, np.ndarray)
                and passable.dtype == np.bool_
                and passable.flags.c_contiguous
                else np.ascontiguousarray(passable, dtype=bool)
            )
            c_no_diag = (
                (
                    no_diag
                    if isinstance(no_diag, np.ndarray)
                    and no_diag.dtype == np.bool_
                    and no_diag.flags.c_contiguous
                    else np.ascontiguousarray(no_diag, dtype=bool)
                )
                if no_diag is not None
                else None
            )
            return lox_core.fast_bfs_distance_grid(
                c_passable, int(sy), int(sx), bool(allow_diagonal), c_no_diag
            )
        except ValueError:
            raise
        except Exception:
            pass
    return _numba_bfs_distance_grid(passable, sy, sx, allow_diagonal, no_diag)


def _get_numba_astar_kernel() -> Any:
    global _NUMBA_ASTAR_KERNEL
    if _NUMBA_ASTAR_KERNEL is None:
        import numba

        @numba.njit(cache=True)
        def _numba_astar_next_step_impl(
            passable: np.ndarray,
            sy: int,
            sx: int,
            ty: int,
            tx: int,
            allow_diagonal: bool = True,
            no_diag: np.ndarray | None = None,
        ) -> tuple[int, int]:
            """Computes first step delta (dy, dx) using Numba JIT backwards BFS."""
            if sy == ty and sx == tx:
                return (0, 0)
            h, w = passable.shape
            if not (0 <= sy < h and 0 <= sx < w and 0 <= ty < h and 0 <= tx < w):
                return (0, 0)

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
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and dist[ny, nx] != -1
                    and passable[ny, nx]
                ):
                    if i >= 4 and no_diag is not None:
                        if no_diag[sy, sx] or no_diag[ny, nx]:
                            continue
                    if dist[ny, nx] < best_d:
                        best_d = dist[ny, nx]
                        best_dy = dys[i]
                        best_dx = dxs[i]
            return (best_dy, best_dx)

        _NUMBA_ASTAR_KERNEL = _numba_astar_next_step_impl
    return _NUMBA_ASTAR_KERNEL


def _numba_astar_next_step(
    passable: np.ndarray,
    sy: int,
    sx: int,
    ty: int,
    tx: int,
    allow_diagonal: bool = True,
    no_diag: np.ndarray | None = None,
) -> tuple[int, int]:
    return _get_numba_astar_kernel()(passable, sy, sx, ty, tx, allow_diagonal, no_diag)


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
    Delegates to SIMD-optimized `lox_core` when available, falling back to Numba.
    """
    if HAS_RUST_CORE:
        try:
            c_passable = (
                passable
                if isinstance(passable, np.ndarray)
                and passable.dtype == np.bool_
                and passable.flags.c_contiguous
                else np.ascontiguousarray(passable, dtype=bool)
            )
            c_no_diag = (
                (
                    no_diag
                    if isinstance(no_diag, np.ndarray)
                    and no_diag.dtype == np.bool_
                    and no_diag.flags.c_contiguous
                    else np.ascontiguousarray(no_diag, dtype=bool)
                )
                if no_diag is not None
                else None
            )
            return lox_core.fast_astar_next_step(
                c_passable,
                int(sy),
                int(sx),
                int(ty),
                int(tx),
                bool(allow_diagonal),
                c_no_diag,
            )
        except ValueError:
            raise
        except Exception:
            pass
    return _numba_astar_next_step(passable, sy, sx, ty, tx, allow_diagonal, no_diag)
