"""Domain-scoped tests for LOX Rust acceleration core."""

from __future__ import annotations

import lox_core
import numpy as np

from lox.env.grid_utils import _numba_astar_next_step, _numba_bfs_distance_grid
from lox.env.harness import NetHackHarness
from lox.env.state import FloorMemory


def test_rust_bfs_and_astar_equivalence():
    rng = np.random.default_rng(42)
    passable = rng.random((21, 79)) > 0.3
    passable[10, 40] = True
    passable[15, 60] = True

    # 1. Compare BFS
    dist_rust = lox_core.fast_bfs_distance_grid(passable, 10, 40, allow_diagonal=True)
    dist_numba = _numba_bfs_distance_grid(passable, 10, 40, allow_diagonal=True)
    np.testing.assert_array_equal(dist_rust, dist_numba)

    # 2. Compare A* step
    step_rust = lox_core.fast_astar_next_step(
        passable, 10, 40, 15, 60, allow_diagonal=True
    )
    step_numba = _numba_astar_next_step(passable, 10, 40, 15, 60, allow_diagonal=True)
    assert step_rust == step_numba


def test_rust_frontier_doors_and_readonly_floor_memory():
    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=42)
        assert isinstance(st.adjacent_closed_doors(), list)
        frontier = st.unexplored_frontier_mask()
        assert frontier.shape == (21, 79) and frontier.dtype == np.bool_
    finally:
        h.close()

    fm = FloorMemory()
    assert fm.passable_mask.flags.writeable is False
    assert fm.copy_passable_mask().flags.writeable is True
