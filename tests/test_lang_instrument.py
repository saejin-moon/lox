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
