"""Domain-scoped tests for LOX."""

from __future__ import annotations

import lox_core
import numpy as np

from lox.env.actions import Action
from lox.env.grid_utils import _numba_astar_next_step, _numba_bfs_distance_grid
from lox.env.harness import NetHackHarness
from lox.env.state import FloorMemory
from lox.lang.bt import (
    ActionLeaf,
    BehaviorTree,
    Blackboard,
    PrioritySelector,
)


def test_rust_blackboard():
    bb = lox_core.Blackboard()
    assert len(bb) == 0
    assert not bb.has_var("health")
    assert bb.get_var("health", default=100) == 100

    bb.set_var("health", 42)
    assert bb.has_var("health")
    assert bb.get_var("health") == 42
    assert bb["health"] == 42
    assert "health" in bb

    bb["target"] = (12, 34)
    assert bb["target"] == (12, 34)

    bb.clear_vars()
    assert len(bb) == 0
    assert not bb.has_var("health")


def test_rust_behavior_tree_execution():
    bb = lox_core.Blackboard()

    # Priority 1: Dialog (returns None when no prompt)
    leaf_dialog = lox_core.ActionLeaf(
        lambda st: Action.SPACE if getattr(st, "in_prompt", False) else None,
        name="DialogLeaf",
    )

    # Priority 2: Threat (returns WAIT if hostile adjacent)
    def threat_fn(st, b):
        if getattr(st, "in_danger", False):
            b.set_var("alert", True)
            return Action.WAIT
        return None

    leaf_threat = lox_core.ActionLeaf(threat_fn, name="ThreatLeaf")

    # Priority 3: Exploration (fallback SEARCH)
    leaf_nav = lox_core.ActionLeaf(lambda st: Action.SEARCH, name="NavLeaf")

    root = lox_core.PrioritySelector(
        [leaf_dialog, leaf_threat, leaf_nav], name="RootSelector"
    )
    tree = lox_core.BehaviorTree(root, bb)

    class MockState:
        in_prompt = False
        in_danger = False

    st = MockState()

    # Turn 1: Fallback to nav
    act = tree.tick(st)
    assert act == Action.SEARCH
    assert not bb.has_var("alert")

    # Turn 2: Threat active
    st.in_danger = True
    act = tree.tick(st)
    assert act == Action.WAIT
    assert bb.get_var("alert") is True

    # Turn 3: Dialog takes highest priority
    st.in_prompt = True
    act = tree.tick(st)
    assert act == Action.SPACE


def test_rust_sequence_execution():
    bb = lox_core.Blackboard()

    cond = lox_core.ConditionLeaf(
        lambda st, b: getattr(st, "can_move", False), name="CanMove"
    )
    act_leaf = lox_core.ActionLeaf(lambda st: Action.DOWN, name="MoveDown")

    seq = lox_core.Sequence([cond, act_leaf], name="TryMove")

    class MockState:
        can_move = False

    st = MockState()
    status, act = seq.tick(st, bb)
    assert status == lox_core.NodeStatus.FAILURE
    assert act is None

    st.can_move = True
    status, act = seq.tick(st, bb)
    assert status == lox_core.NodeStatus.SUCCESS
    assert act == Action.DOWN


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


def test_phase0_rust_and_bt_preempt_and_blacklist():
    bb = Blackboard()
    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=42)
        assert isinstance(st.adjacent_closed_doors(), list)
        frontier = st.unexplored_frontier_mask()
        assert frontier.shape == (21, 79) and frontier.dtype == np.bool_

        bb.floor_blacklist_add(st, (5, 5), ttl=3)
        assert bb.floor_is_blacklisted(st, (5, 5)) is True

        # Preempt_if interrupts min_commit immediately
        ticks = {"explore": 0, "escape": 0}
        danger = {"active": False}

        def do_escape(s, _bb):
            if danger["active"]:
                ticks["escape"] += 1
                return Action.WAIT
            return None

        def do_explore(s, _bb):
            ticks["explore"] += 1
            return Action.S

        tree = BehaviorTree(
            PrioritySelector(
                [
                    ActionLeaf(do_escape, name="escape"),
                    ActionLeaf(
                        do_explore,
                        name="explore",
                        min_commit=5,
                        preempt_if=lambda _s, _bb: danger["active"],
                    ),
                ],
                name="root",
            ),
            blackboard=bb,
        )
        tree.tick(st)
        assert ticks["explore"] == 1
        danger["active"] = True
        act = tree.tick(st)
        assert act == Action.WAIT and ticks["escape"] == 1
    finally:
        h.close()

    fm = FloorMemory()
    assert fm.passable_mask.flags.writeable is False
    assert fm.copy_passable_mask().flags.writeable is True
