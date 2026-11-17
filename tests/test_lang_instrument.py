"""
Tests for the Near-Full Python Policy Language, Sandbox Validator,
AST ExecutionTracer Instrumentation, Loop Watchdog, Function Splicing, and MacroLibrary.
"""

from __future__ import annotations

import pytest

from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_and_macros
from lox.lang.instrument import PolicyLoopError
from lox.lang.macros import MacroLibrary
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


def test_function_splicing_and_macro_library(tmp_path):
    base_policy = """
class Policy:
    def explore(self, state):
        state = yield Action.SEARCH
        return state

    def run(self, state):
        while True:
            state = yield from self.explore(state)
"""
    patch_code = """
def explore(self, state):
    if state.char_at(state.y, state.x) == '>':
        state = yield Action.DOWN
    else:
        state = yield Action.WAIT
    return state

def combat(self, state):
    state = yield Action.WAIT
    return state
"""
    spliced, modified_fns = splice_functions_into_policy(base_policy, patch_code)
    assert set(modified_fns) == {"explore", "combat"}
    validate_python_code(spliced)
    assert "def combat(self, state):" in spliced
    assert "Action.DOWN" in spliced

    # Test MacroLibrary
    macros_file = tmp_path / "macros.py"
    lib = MacroLibrary(path=str(macros_file))
    saved = lib.upsert_macros(
        """
def macro_engrave_elbereth(state):
    \"\"\"Engraves Elbereth in dust with fingers.\"\"\"
    yield seq(Action.ENGRAVE, "-", "Elbereth\\r")
"""
    )
    assert saved == ["macro_engrave_elbereth"]
    docs = lib.describe_macros()
    assert "macro_engrave_elbereth(state)" in docs
    assert "Engraves Elbereth" in docs


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


def test_macro_contract_linting():
    from lox.lang.validator import check_macro_contracts

    macros_code = """
def find_corridors(state):
    return state.chars == ord('#')

def kick_door(state, direction):
    yield seq(Action.KICK, direction)
"""

    # 1. Valid policy call
    valid_policy = """
class Policy:
    def run(self, state):
        mask = macros.find_corridors(state)
        yield from macros.kick_door(state, Action.EAST)
"""
    errors = check_macro_contracts(valid_policy, macros_code)
    assert errors == []

    # 2. Too few arguments
    too_few_args = """
class Policy:
    def run(self, state):
        mask = macros.find_corridors()
"""
    errs = check_macro_contracts(too_few_args, macros_code)
    assert len(errs) == 1
    assert "requires at least 1 argument" in errs[0]

    # 3. Direct yield instead of yield from for generator macro
    bad_yield = """
class Policy:
    def run(self, state):
        state = yield macros.kick_door(state, Action.NORTH)
"""
    errs = check_macro_contracts(bad_yield, macros_code)
    assert any("is a generator. Use 'yield from" in e for e in errs)


def test_bottom_up_macro_harvester(tmp_path):
    from lox.lang.macros import MacroHarvester, MacroLibrary

    macros_file = str(tmp_path / "macros.py")
    ml = MacroLibrary(path=macros_file)

    champion_code = """
class Policy:
    def __init__(self):
        self.step_count = 0

    def find_stairs_down(self, state):
        '''Locates downstairs coordinates.'''
        stairs = np.argwhere(state.chars == ord('>'))
        return tuple(stairs[0]) if len(stairs) > 0 else None

    def stateful_method(self, state):
        self.step_count += 1
        return self.step_count

    def run(self, state):
        while True:
            target = self.find_stairs_down(state)
            state = yield Action.SEARCH
"""
    fn_stats = [
        {"function_name": "find_stairs_down", "ochiai_good_run_score": 0.45},
        {"function_name": "stateful_method", "ochiai_good_run_score": 0.50},
    ]

    harvested = MacroHarvester.harvest_from_policy(
        policy_code=champion_code,
        macro_lib=ml,
        fn_stats=fn_stats,
    )

    assert harvested == ["find_stairs_down"]
    saved_code = ml.load_code()
    assert "def find_stairs_down(state):" in saved_code
    assert "Locates downstairs coordinates." in saved_code
    assert "stateful_method" not in saved_code
    assert "run" not in saved_code


def test_bt_leaf_attribution_ignores_step_wrappers_and_hysteresis_commits():
    from lox.env.harness import NetHackHarness
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


def test_phase2_memory_sequence_coroutines_halting_and_floor_scoping():
    from types import SimpleNamespace

    from lox.env.actions import Action
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


def test_phase2_skill_splice_three_way_merge_and_symtable_gate():
    import time

    from lox.author.tools import merge_skill_module_update
    from lox.lang.compiler import _build_base_namespace, compile_policy_package
    from lox.lang.instrument import ExecutionTracer
    from lox.lang.validator import (
        PolicyValidationError,
        check_undefined_symbols,
        three_way_ast_merge,
    )

    # 1. splice_functions_into_policy preserves top-level imports & constants when target_class is None
    base_skill = "from lox.env.actions import Action\nLIMIT = 3\ndef a(st):\n    return Action.WAIT\ndef b(st):\n    return None\n"
    patch_skill = (
        "from lox.env.actions import key\nLIMIT = 7\ndef b(st):\n    return key(' ')\n"
    )
    spliced = merge_skill_module_update(base_skill, patch_skill)
    assert "Action, key" in spliced or ("Action" in spliced and "key" in spliced)
    assert "LIMIT = 7" in spliced
    assert "def a(st):" in spliced
    assert "def b(st):" in spliced

    # 2. three_way_ast_merge handles disjoint helpers + PrioritySelector children
    base_pol = """
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector
from lox.env.actions import Action

class Policy:
    def __init__(self):
        self.bb = Blackboard()
        self.tree = self.build_tree()

    def build_tree(self):
        return PrioritySelector([ActionLeaf(lambda s: Action.WAIT, name='w')])

    def step(self, state):
        return self.tree.tick(state) or Action.WAIT
"""
    pol_a = base_pol.replace(
        "class Policy:",
        "def leaf_a(s):\n    return Action.SEARCH\n\nclass Policy:",
    ).replace(
        "ActionLeaf(lambda s: Action.WAIT, name='w')",
        "ActionLeaf(leaf_a, name='a'), ActionLeaf(lambda s: Action.WAIT, name='w')",
    )
    pol_b = base_pol.replace(
        "class Policy:",
        "def leaf_b(s):\n    return Action.EAST\n\nclass Policy:",
    ).replace(
        "ActionLeaf(lambda s: Action.WAIT, name='w')",
        "ActionLeaf(lambda s: Action.WAIT, name='w'), ActionLeaf(leaf_b, name='b')",
    )
    merged = three_way_ast_merge(base_pol, pol_a, pol_b)
    assert merged is not None
    assert "def leaf_a" in merged and "def leaf_b" in merged
    assert "leaf_a" in merged and "leaf_b" in merged
    # Same-function conflicting mutation must return None
    pol_a2 = pol_a.replace("Action.SEARCH", "Action.WEST")
    assert three_way_ast_merge(base_pol, pol_a, pol_a2) is None

    # 3. Fast symtable gate catches dormant NameError & missing skill export
    allowed = frozenset(_build_base_namespace(ExecutionTracer()).keys())
    t0 = time.perf_counter()
    check_undefined_symbols(merged, allowed_globals=allowed)
    assert (time.perf_counter() - t0) < 0.05

    bad_skill = {"sub": "def f(st):\n    return dormant_missing_helper(st)\n"}
    with pytest.raises(PolicyValidationError, match="dormant_missing_helper"):
        check_undefined_symbols(
            "class Policy:\n    def step(self, state):\n        return Action.WAIT\n",
            skills=bad_skill,
            allowed_globals=allowed,
        )

    good_skill = {"sub": "def existing_fn(st):\n    return Action.WAIT\n"}
    with pytest.raises(PolicyValidationError, match="missing_export"):
        check_undefined_symbols(
            "from skills.sub import missing_export\nclass Policy:\n    def step(self, state):\n        return Action.WAIT\n",
            skills=good_skill,
            allowed_globals=allowed,
        )

    # Also verify compile_policy_package invokes check_undefined_symbols automatically
    with pytest.raises(PolicyValidationError):
        compile_policy_package(
            "class Policy:\n    def step(self, state):\n        return undefined_xyz(state)\n"
        )
