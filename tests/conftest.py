"""Shared test fixtures: isolate module/class-level state between tests.

The navigation primitives (GridAStar, FrontierExplorer) keep epoch-guarded
visited buffers as CLASS-level state — correct at runtime (every search bumps
the epoch so stale entries are ignored) but a cross-test contamination vector
if any future code forgets to bump. The autouse fixture below forces fresh
buffers around every test so ordering can never leak map topology between
cases (R5→R6 handoff checklist item 1: the two transiently-flaky combat tests
test_combat_manager_standard_melee / test_fragile_role_elbereth_threshold).
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _sandbox_policy_compiler_dirs(tmp_path, monkeypatch):
    """S1: the acceptance path writes the authoring tree + compiled artifact +
    archive. Sandbox every test's compiler output dirs into its tmp dir so no test
    can ever pollute data/program, data/compiled, or data/programs/archive."""
    from lox.policy import compiler

    monkeypatch.setattr(compiler, "PROGRAM_DIR", str(tmp_path / "program_tree"))
    monkeypatch.setattr(compiler, "COMPILED_DIR", str(tmp_path / "compiled"))
    monkeypatch.setattr(compiler, "ARCHIVE_DIR", str(tmp_path / "programs_archive"))
    yield


@pytest.fixture(autouse=True)
def _reset_navigation_class_state():
    """Fresh epoch buffers for GridAStar / FrontierExplorer around each test."""
    from lox.navigation import astar as astar_mod
    from lox.navigation import frontier as frontier_mod

    astar_mod.GridAStar._epoch = 0
    astar_mod.GridAStar._visited_epoch = [0] * astar_mod.GridAStar.SIZE
    astar_mod.GridAStar._seen_epoch = [0] * astar_mod.GridAStar.SIZE
    frontier_mod.FrontierExplorer._bfs_epoch = 0
    frontier_mod.FrontierExplorer._bfs_visited = np.zeros(frontier_mod.FrontierExplorer.SIZE, dtype=np.int32)
    yield
    astar_mod.GridAStar._epoch = 0
    astar_mod.GridAStar._visited_epoch = [0] * astar_mod.GridAStar.SIZE
    astar_mod.GridAStar._seen_epoch = [0] * astar_mod.GridAStar.SIZE
    frontier_mod.FrontierExplorer._bfs_epoch = 0
    frontier_mod.FrontierExplorer._bfs_visited = np.zeros(frontier_mod.FrontierExplorer.SIZE, dtype=np.int32)
