import numpy as np
import pytest
from lox.core.types import Action
from lox.envs.nethack import NetHackAdapter


def test_nethack_adapter_step():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)

    assert obs is not None
    assert obs.hero.hp > 0
    assert obs.hero.depth == 1
    assert len(obs.inventory) > 0

    # Test taking a step
    obs, reward, term, trunc, info = adapter.step(Action(name="search"))
    assert obs.hero.turn >= 1

    adapter.close()


def test_nethack_interlocks():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)

    # Test floating eye safety check: monster 28 is floating eye
    fake_glyphs = np.zeros((21, 79), dtype=np.int32)
    fake_glyphs[5, 5] = 28  # Floating eye at (5, 5)
    assert adapter.is_target_floating_eye(fake_glyphs, 5, 5) is True
    assert adapter.is_target_floating_eye(fake_glyphs, 5, 6) is False

    # Test prayer cooldown
    assert adapter.can_safely_pray(turn=500) is True
    adapter.last_prayer_turn = 450
    assert adapter.can_safely_pray(turn=500) is False  # 50 turns < 350
    assert adapter.can_safely_pray(turn=850) is True   # 400 turns >= 350

    adapter.close()
