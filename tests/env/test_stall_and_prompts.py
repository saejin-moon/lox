"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.env import Action, NetHackHarness
from lox.env.state import GameState
from lox.lang.compiler import compile_policy_package


def test_turn_advanced_and_consecutive_stalls():
    harness = NetHackHarness(max_steps=50)
    try:
        state = harness.reset(seed=42)
        assert state is not None
        assert state.turn_advanced is True
        assert state.consecutive_stalls == 0

        # Action.SEARCH consumes 1 turn in NetHack
        state, done, info = harness.step(Action.SEARCH)
        assert state.turn_advanced is True
        assert state.consecutive_stalls == 0
        assert state.time_delta == 1
    finally:
        harness.close()


def test_zero_turn_stall_truncation():
    # Set a low stall limit of 10 zero-turn steps to verify fast truncation
    harness = NetHackHarness(max_steps=500, max_zero_turn_stalls=10)
    try:
        state = harness.reset(seed=42)
        # In NetHack, walking into a solid stone wall consumes 0 turns!
        # Find a wall direction and repeat it
        for _ in range(25):
            # Try walking NW or N into boundary wall if at top
            state, done, info = harness.step(Action.NW)
            if done:
                break

        assert done is True
        assert info.get("stalled_zero_turn") is True
        assert harness.consecutive_zero_turn_steps >= 10
    finally:
        harness.close()


def test_interactive_yn_prompt_passthrough():
    harness = NetHackHarness(max_steps=50)
    try:
        state = harness.reset(seed=42)
        # Yield Action.PRAY -> prompts "Are you sure you want to pray? [yn] (n)"
        state, done, info = harness.step(Action.PRAY)
        assert not done
        assert state.in_yn_prompt is True
        assert "pray" in state.message.lower() or "[yn]" in state.message.lower()

        # Step 2: Policy can answer 'y' across steps!
        state, done, info = harness.step("y")
        assert not done
        # Tyr responds to the prayer
        msg = state.message.lower()
        assert "pray" in msg or "tyr" in msg or "finish" in msg or "pleased" in msg
    finally:
        harness.close()


def test_lean_observation_keys_compatibility():
    harness = NetHackHarness(max_steps=50, lean_obs=True)
    try:
        state = harness.reset(seed=123)
        assert isinstance(state, GameState)
        assert state.hp > 0
        assert len(state.tty_lines) == 24
        assert len(state.tty_text.splitlines()) == 24
        assert state.terrain_at(state.y, state.x) is not None
        assert state.description_at(state.y, state.x) is not None
    finally:
        harness.close()


def test_yn_prompt_clears_after_n_reply_no_se_loop():
    import numpy as np

    # 1. Real NLE verification: after answering 'n' to Pray? [yn] (n), internal[1] flips 0
    # and in_yn_prompt must become False even though '(n) n' stays in state.message.
    harness = NetHackHarness(max_steps=50)
    try:
        state = harness.reset(seed=42)
        state, _, _ = harness.step(Action.PRAY)
        assert state.internal[1] == 1
        assert state.in_yn_prompt is True

        state, _, _ = harness.step("n")
        assert state.internal[1] == 0
        assert "[yn]" in state.message.lower()
        assert state.in_yn_prompt is False
    finally:
        harness.close()

    # 2. Synthetic/mock GameState without internal array still rejects echoed ') n' replies.
    blank = np.full((21, 79), ord("."), dtype=np.uint8)
    obs_active = {
        "chars": blank,
        "glyphs": np.zeros((21, 79), dtype=np.int32),
        "colors": np.zeros((21, 79), dtype=np.int8),
        "specials": np.zeros((21, 79), dtype=np.int8),
        "blstats": np.zeros(27, dtype=np.int32),
        "inv_glyphs": np.zeros(55, dtype=np.int32),
        "inv_strs": np.zeros((55, 80), dtype=np.uint8),
        "inv_letters": np.zeros(55, dtype=np.uint8),
        "inv_oclasses": np.zeros(55, dtype=np.uint8),
        "message": np.frombuffer(
            b"Really attack the dwarf? [yn] (n)\x00", dtype=np.uint8
        ),
    }
    obs_echoed = {
        **obs_active,
        "message": np.frombuffer(
            b"Really attack the dwarf? [yn] (n) n\x00", dtype=np.uint8
        ),
    }
    assert GameState(obs_active).in_yn_prompt is True
    assert GameState(obs_echoed).in_yn_prompt is False


# ---------------------------------------------------------------------------
# 5. Environment & Execution Purification ("Hard Code Nothing") Tests
# ---------------------------------------------------------------------------
def test_environment_purification_no_auto_esc():
    """PolicyRunner must NOT silently replace actions with ESC when in a text prompt."""
    pol_code = """
class Policy:
    def step(self, state):
        return Action.WAIT
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()

    class DummyPromptState:
        hp = 20
        max_hp = 20
        turn = 1
        in_text_prompt = True
        in_yn_prompt = False

    st = DummyPromptState()

    # Call next_action multiple times: Action.WAIT must NOT be rewritten to ESC
    for _ in range(5):
        act, _ = runner.next_action(st)
        assert act == Action.WAIT
    assert runner.auto_esc_count == 0


def test_environment_purification_no_low_hp_preemption():
    """PolicyRunner must NOT cancel active subgenerators solely because HP is low."""
    pol_code = """
class Policy:
    def subgen(self, state):
        while True:
            yield Action.SEARCH

    def run(self, state):
        yield from self.subgen(state)
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()

    class DummyLowHpState:
        hp = 1
        max_hp = 20
        turn = 1
        in_text_prompt = False
        in_yn_prompt = False

    st = DummyLowHpState()

    act1, fn1 = runner.next_action(st)
    assert act1 == Action.SEARCH

    act2, fn2 = runner.next_action(st)
    assert act2 == Action.SEARCH
    assert runner.subgen_preemptions == 0
