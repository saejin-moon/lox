"""
Unit and integration tests for:
1. Zero-turn action awareness (turn_advanced, consecutive_stalls, time_delta)
2. Interactive decision prompt passthrough and safety watchdog
3. Zero-turn runaway stall truncation (stalled_zero_turn)
4. Lean observation keys throughput and GameState compatibility
"""

from __future__ import annotations

from lox.env import Action, NetHackHarness
from lox.env.state import GameState


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
