"""Domain-scoped tests for LOX."""

from __future__ import annotations

from typing import Any

from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.bt import (
    ActionLeaf,
    BehaviorTree,
    Blackboard,
    ConditionLeaf,
    Inverter,
    NodeStatus,
    PrioritySelector,
    Sequence,
)
from lox.lang.compiler import compile_policy_package


# ---------------------------------------------------------------------------
# 1. Behavior Tree Engine Unit Tests
# ---------------------------------------------------------------------------
def test_behavior_tree_composites_and_leaves():
    bb = Blackboard()
    bb.set_var("danger", True)

    def is_in_danger(st: Any, b: Blackboard) -> bool:
        return b.get_var("danger", False)

    def flee_action(st: Any, b: Blackboard) -> Action:
        return Action.WAIT

    def explore_action(st: Any) -> Action:
        return Action.SEARCH

    cond_danger = ConditionLeaf(is_in_danger, name="DangerCheck")
    act_flee = ActionLeaf(flee_action, name="FleeAct")
    flee_branch = Sequence([cond_danger, act_flee], name="FleeBranch")

    act_explore = ActionLeaf(explore_action, name="ExploreAct")

    root = PrioritySelector([flee_branch, act_explore], name="RootSelector")
    tree = BehaviorTree(root=root, blackboard=bb)

    # When danger is True, flee_branch executes and returns Action.WAIT
    act1 = tree.tick(state=None)
    assert act1 == Action.WAIT

    # When danger is False, flee_branch fails, fallback to act_explore (Action.SEARCH)
    bb.set_var("danger", False)
    act2 = tree.tick(state=None)
    assert act2 == Action.SEARCH


def test_behavior_tree_inverter():
    bb = Blackboard()
    cond = ConditionLeaf(lambda st: False)
    inv = Inverter(cond)
    status, _ = inv.tick(None, bb)
    assert status == NodeStatus.SUCCESS


def test_policy_runner_native_bt_support():
    pol_code = """
from lox.lang.bt import BehaviorTree, PrioritySelector, ActionLeaf
from lox.env.actions import Action

class Policy:
    def __init__(self):
        leaf1 = ActionLeaf(lambda st: Action.SEARCH)
        self.tree = BehaviorTree(root=leaf1)
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()
    assert runner._mode == "bt"

    class DummyState:
        hp = 20
        max_hp = 20
        turn = 1

    st = DummyState()
    act, fn = runner.next_action(st)
    assert act == Action.SEARCH


def test_bt_bound_method_leaves():
    class Helper:
        def is_ok(self, state) -> bool:
            return bool(state.hp > 0)

        def act_step(self, state):
            return Action.WAIT

    h = Helper()
    bb = Blackboard()

    class DummyState:
        hp = 10
        depth = 1
        dungeon_num = 0
        y = 5
        x = 5

    seq_node = Sequence([ConditionLeaf(h.is_ok), ActionLeaf(h.act_step)])
    status, act = seq_node.tick(DummyState(), bb)
    assert act == Action.WAIT
    assert status.name == "SUCCESS"
    assert not bb.leaf_errors


def test_bt_leaf_attribution_ignores_step_wrappers_and_hysteresis_commits():
    from lox.lang.compiler import compile_policy_package

    code = """
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector

def _leaf_a(state, bb):
    cnt = bb.get_var("cnt", 0) + 1
    bb.set_var("cnt", cnt)
    # Only returns None on turn 2 if ticked normally unless hysteresis holds it
    return Action.EAST if cnt in (1, 2, 3) else None

def _leaf_b(state, bb):
    return Action.WEST

def _osc_gate(state, bb, act):
    # Post-step wrapper like Gen 10 used in policy.py
    return act

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        self.tree = BehaviorTree(
            root=PrioritySelector([
                ActionLeaf(_leaf_a, name="leaf_a", min_commit=2),
                ActionLeaf(_leaf_b, name="leaf_b"),
            ]),
            blackboard=self.blackboard,
        )

    def step(self, state):
        act = self.tree.tick(state)
        return _osc_gate(state, self.blackboard, act) or Action.WAIT
"""
    compiled = compile_policy_package(code, "")
    runner = compiled.create_runner()
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        seen_fns: list[str] = []
        for _ in range(4):
            act, fn_name = runner.next_action(st)
            seen_fns.append(fn_name)
            st, done, _ = h.step(act)
            if done:
                break
        # Must attribute to winning ActionLeaf names, NOT `_osc_gate`!
        assert "_osc_gate" not in runner.tracer.fn_turns
        assert seen_fns[0] == "leaf_a"
        assert "leaf_b" in seen_fns
    finally:
        h.close()

    # Verify PrioritySelector min_commit holds commitment when higher-priority child flips
    from lox.lang.bt import ActionLeaf, Blackboard, PrioritySelector

    bb = Blackboard()
    calls: list[str] = []

    def high_fn(st, b):
        calls.append("high")
        # False on turn 0, True on turn 1!
        return "HIGH_ACT" if b.get_var("turn", 0) >= 1 else None

    def low_fn(st, b):
        calls.append("low")
        return "LOW_ACT"

    sel = PrioritySelector(
        [
            ActionLeaf(high_fn, name="high_leaf"),
            ActionLeaf(low_fn, name="low_leaf", min_commit=2),
        ]
    )
    bb.set_var("turn", 0)
    _, act0 = sel.tick(None, bb)
    assert act0 == "LOW_ACT"
    assert bb.last_active_leaf == "low_leaf"

    # Turn 1: low_leaf commitment (min_commit=2) holds before high_leaf preempts on Turn 2
    bb.set_var("turn", 1)
    _, act1 = sel.tick(None, bb)
    assert act1 == "LOW_ACT"
    assert bb.last_active_leaf == "low_leaf"

    bb.set_var("turn", 2)
    _, act2 = sel.tick(None, bb)
    assert act2 == "HIGH_ACT"
    assert bb.last_active_leaf == "high_leaf"
