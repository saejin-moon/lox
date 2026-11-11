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
