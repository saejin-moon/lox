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


def test_v3_tracer_primitive_fastpath_and_leaf_subfunction_attribution():
    """Verifies Component 1 & 5: ExecutionTracer unmasks `explore._open_adjacent_door` while ignoring non-action helpers and preserving Action IntEnum fast-path."""
    from lox.lang.compiler import compile_policy_package
    from lox.lang.starter import build_clean_option_a_starter

    pol, skills = build_clean_option_a_starter()
    skills["goals/explore"] = '''"""Explore goal calling a leaf action helper."""
from lox.env.actions import Action
import skills.goals.core as goals


def _is_blocked(y, x):
    return (y + x) < 0


def _open_adjacent_door(state):
    if _is_blocked(state.y, state.x):
        return None
    return Action.SEARCH


def _explore_step(state, ws):
    _ = ws.floor(state)
    return _open_adjacent_door(state)


goals.register_goal("explore", priority=10, trigger=lambda s, w: True, step=_explore_step)
'''
    pol2 = "import skills.goals.explore  # noqa: F401\n" + pol
    compiled = compile_policy_package(pol2, "", skills=skills)
    runner = compiled.create_runner()
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        act, fn = runner.next_action(st)
        assert act == Action.SEARCH
        # Unmasks sub-function under goal wrapper instead of opaque `explore` or `ws.floor`!
        assert fn == "explore._open_adjacent_door", fn
        assert runner.tracer.fn_calls.get("_is_blocked", 0) >= 1
        assert runner.tracer.fn_calls.get("explore._open_adjacent_door", 0) >= 1
        # Verify compute_function_stats retains the unmasked sub-function for DuckDB `function_stats`
        from lox.db.summarizer import compute_function_stats

        fstats = compute_function_stats(
            "b_test",
            1,
            [
                {
                    "triggered_functions": dict(runner.tracer.fn_calls),
                    "fn_turns": dict(runner.tracer.fn_turns),
                    "max_depth": 2,
                    "is_outlier_good": True,
                    "end_reason": "killed",
                }
            ],
        )
        fn_map = {row["function_name"]: row for row in fstats}
        assert "explore._open_adjacent_door" in fn_map
        assert fn_map["explore._open_adjacent_door"]["total_turns_active"] >= 1
    finally:
        h.close()
