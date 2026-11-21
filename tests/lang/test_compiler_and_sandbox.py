"""Domain-scoped tests for LOX."""

from __future__ import annotations

import pytest

from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_and_macros
from lox.lang.instrument import PolicyLoopError
from lox.lang.validator import (
    PolicyValidationError,
    splice_functions_into_policy,
    validate_python_code,
)


def test_full_python_311_features_supported():
    code = """
@dataclass
class Target:
    y: int
    x: int
    priority: float

def helper_score(d: int) -> float:
    return 1.0 / max(1, d)

class Policy:
    def __init__(self):
        self.visited = set()
        self.history = deque(maxlen=10)
        self.flee_hp = self.param("flee_hp", 0.30, 0.10, 0.70)

    def combat(self, state):
        for m in state.visible_monsters():
            match m.char:
                case 'e':
                    state = yield Action.WAIT
                    return state
                case _:
                    if abs(m.y - state.y) <= 1 and abs(m.x - state.x) <= 1:
                        state = yield DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
                        return state
        state = yield Action.SEARCH
        return state

    def run(self, state):
        while True:
            self.visited.add((state.y, state.x))
            self.history.append(state.turn)
            scores = {k: helper_score(k) for k in range(1, 4)}
            t = Target(y=state.y, x=state.x, priority=scores[1])
            if state.hp / max(1, state.max_hp) < self.flee_hp:
                state = yield Action.WAIT
            else:
                state = yield from self.combat(state)
"""
    validate_python_code(code)
    compiled = compile_policy_and_macros(code)
    fn_names = compiled.policy_function_names
    assert "combat" in fn_names
    assert "run" in fn_names
    assert "helper_score" in fn_names

    harness = NetHackHarness(max_steps=50)
    try:
        st = harness.reset(seed=101)
        runner = compiled.create_runner(param_overrides={"flee_hp": 0.45})
        for _ in range(10):
            act, active_fn = runner.next_action(st)
            st, done, _ = harness.step(act)
            if done:
                break

        # Verify ExecutionTracer tracked function calls, active turns, and tunable parameters
        assert runner.tracer.call_counts.get("run", 0) == 1
        assert runner.tracer.call_counts.get("combat", 0) >= 1
        assert runner.tracer.call_counts.get("helper_score", 0) >= 1
        assert runner.tracer.active_turns.get("combat", 0) >= 1
        assert "flee_hp" in runner.tracer.registered_params
        assert runner.tracer.registered_params["flee_hp"]["value"] == 0.45
    finally:
        harness.close()


def test_sandbox_blocks_unsafe_constructs():
    with pytest.raises(PolicyValidationError, match="forbidden in policy sandbox"):
        validate_python_code(
            "import os\nclass Policy:\n    def run(self, state):\n        yield Action.WAIT"
        )

    with pytest.raises(PolicyValidationError, match="forbidden in policy sandbox"):
        validate_python_code(
            "class Policy:\n    def run(self, state):\n        eval('1+1')\n        yield Action.WAIT"
        )

    with pytest.raises(PolicyValidationError, match="forbidden in policy sandbox"):
        validate_python_code(
            "class Policy:\n    def run(self, state):\n        x = ().__class__.__base__\n        yield Action.WAIT"
        )


def test_infinite_zero_yield_loop_watchdog():
    bad_code = """
class Policy:
    def run(self, state):
        while True:
            x = 1 + 1
        yield Action.WAIT
"""
    compiled = compile_policy_and_macros(bad_code)
    harness = NetHackHarness(max_steps=20)
    try:
        st = harness.reset(seed=7)
        runner = compiled.create_runner()
        with pytest.raises(PolicyLoopError, match="Infinite zero-yield loop"):
            runner.next_action(st)
    finally:
        harness.close()


def test_call_stack_attribution_and_caller_splice_filtering():
    # 1. Verify that when specialist `handle_combat` calls leaf helper `get_passable_mask`,
    # the turn is attributed to `handle_combat` rather than getting stuck on `get_passable_mask`.
    code = """
class Policy:
    def get_passable_mask(self, state):
        return state.chars == ord('.')

    def handle_combat(self, state):
        _ = self.get_passable_mask(state)
        return Action.SEARCH

    def run(self, state):
        while True:
            act = self.handle_combat(state)
            state = yield act
"""
    compiled = compile_policy_and_macros(code)
    harness = NetHackHarness(max_steps=10)
    try:
        st = harness.reset(seed=55)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st)
        assert active_fn == "handle_combat"
        assert runner.tracer.fn_turns.get("handle_combat", 0) == 1
        assert runner.tracer.fn_turns.get("get_passable_mask", 0) == 0
    finally:
        harness.close()

    # 2. Verify that when a patch modifies `run` alongside a new specialist `pray_when_weak`,
    # `splice_functions_into_policy` filters out root entrypoint `run` from `changed_names`.
    patch_with_caller = """
def pray_when_weak(self, state):
    return Action.PRAY

def run(self, state):
    while True:
        if state.hunger_state >= 3:
            act = self.pray_when_weak(state)
        else:
            act = self.handle_combat(state)
        state = yield act
"""
    _, changed_fns = splice_functions_into_policy(code, patch_with_caller)
    assert changed_fns == ["pray_when_weak"]


def test_phase2_memory_sequence_coroutines_halting_and_floor_scoping():
    from types import SimpleNamespace

    from lox.lang.bt import (
        ActionLeaf,
        Blackboard,
        MemorySequence,
        NodeStatus,
        PrioritySelector,
        bt_leaf,
    )
    from lox.lang.validator import PolicyValidationError

    bb = Blackboard()
    closed = []

    def multi_step_gen(state, b):
        try:
            s1 = yield Action.EAST
            yield (NodeStatus.SUCCESS, Action.SOUTH if s1 is not None else Action.WAIT)
        finally:
            closed.append(True)

    leaf_gen = ActionLeaf(multi_step_gen, name="coro_leaf")
    seq_node = MemorySequence(
        [leaf_gen, ActionLeaf(lambda s: Action.NORTH, name="done_leaf")]
    )
    sel = PrioritySelector(
        [
            ActionLeaf(
                lambda s, b: Action.WAIT if b.get_var("urgent") else None, name="urgent"
            ),
            seq_node,
        ]
    )

    st_f1 = SimpleNamespace(dungeon_num=0, level_num=1)
    st_f2 = SimpleNamespace(dungeon_num=0, level_num=2)
    bb.floor_set(st_f1, "shops", [(5, 5)])
    assert bb.floor_get(st_f1, "shops") == [(5, 5)]
    assert bb.floor_get(st_f2, "shops", []) == []

    # Step 1: coroutine runs step 1 -> RUNNING
    status0, act0 = sel.tick(st_f1, bb)
    assert status0 == NodeStatus.RUNNING and act0 == Action.EAST
    assert id(seq_node) in bb._seq_indices

    # Step 2: urgent branch preempts -> halts MemorySequence and closes generator
    bb.set_var("urgent", True)
    status1, act1 = sel.tick(st_f1, bb)
    assert status1 == NodeStatus.SUCCESS and act1 == Action.WAIT
    assert closed == [True]
    assert id(seq_node) not in bb._seq_indices

    # Step 3: resume from scratch when urgent clears, complete coroutine then next child
    bb.set_var("urgent", False)
    _, act2 = sel.tick(st_f1, bb)
    assert act2 == Action.EAST
    status3, act3 = sel.tick(st_f1, bb)
    assert status3 == NodeStatus.SUCCESS and act3 == Action.NORTH

    # Floor-scoped update + traceback recording
    cnt = bb.floor_update(st_f2, "visits", lambda v: (v or 0) + 1)
    assert cnt == 1
    try:
        raise ValueError("boom")
    except ValueError as exc:
        bb.record_leaf_error("bad_leaf", exc)
    assert "test_phase2_memory_sequence" in bb.leaf_errors[-1][1]
    assert "ValueError: boom" in bb.leaf_errors[-1][1]

    # @bt_leaf + PrioritySelector.from_modules + circular check
    class Mod:
        @bt_leaf(priority=10)
        def low_hi(st):
            return Action.WAIT

        @bt_leaf(priority=1, before="low_hi")
        def first(st):
            return Action.SEARCH

    built = PrioritySelector.from_modules(Mod)
    assert [c.name for c in built.children] == ["first", "low_hi"]

    class BadMod:
        @bt_leaf(before="b")
        def a(st):
            return None

        @bt_leaf(before="a")
        def b(st):
            return None

    with pytest.raises(PolicyValidationError):
        PrioritySelector.from_modules(BadMod)


def test_subgenerator_skills_and_emergency_interrupt():
    code = """
class Policy:
    def __init__(self):
        self.flee_hp_ratio = 0.50

    def breach_door(self, state):
        yield seq(Action.OPEN, Action.EAST)
        yield seq(Action.KICK, Action.EAST)
        yield seq(Action.KICK, Action.EAST)

    def run(self, state):
        while True:
            if state.hp / max(1, state.max_hp) < self.flee_hp_ratio:
                state = yield Action.WAIT
                continue
            # Return subgenerator directly
            state = yield from self.breach_door(state)
"""
    compiled = compile_policy_and_macros(code, "")
    runner = compiled.create_runner()

    harness = NetHackHarness(max_steps=50)
    st = harness.reset(seed=42)

    act1, fn1 = runner.next_action(st)
    act2, fn2 = runner.next_action(st)
    act3, fn3 = runner.next_action(st)

    assert act1 is not None
    assert act2 is not None
    assert act3 is not None
