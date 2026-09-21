"""Shared test fixtures: isolate module/class-level state between tests.

The navigation primitives (GridAStar, FrontierExplorer) keep epoch-guarded
visited buffers as CLASS-level state — correct at runtime (every search bumps
the epoch so stale entries are ignored) but a cross-test contamination vector
if any future code forgets to bump. The autouse fixture below forces fresh
buffers around every test so ordering can never leak map topology between
cases (R5→R6 handoff checklist item 1: the two transiently-flaky combat tests
test_combat_manager_standard_melee / test_fragile_role_elbereth_threshold).
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_navigation_class_state():
    """Fresh epoch buffers for GridAStar / FrontierExplorer around each test."""
    from corp.navigation import astar as astar_mod
    from corp.navigation import frontier as frontier_mod

    astar_mod.GridAStar._epoch = 0
    astar_mod.GridAStar._visited_epoch = [0] * astar_mod.GridAStar.SIZE
    astar_mod.GridAStar._seen_epoch = [0] * astar_mod.GridAStar.SIZE
    frontier_mod.FrontierExplorer._bfs_epoch = 0
    frontier_mod.FrontierExplorer._bfs_visited = [0] * frontier_mod.FrontierExplorer.SIZE
    yield
    astar_mod.GridAStar._epoch = 0
    astar_mod.GridAStar._visited_epoch = [0] * astar_mod.GridAStar.SIZE
    astar_mod.GridAStar._seen_epoch = [0] * astar_mod.GridAStar.SIZE
    frontier_mod.FrontierExplorer._bfs_epoch = 0
    frontier_mod.FrontierExplorer._bfs_visited = [0] * frontier_mod.FrontierExplorer.SIZE
