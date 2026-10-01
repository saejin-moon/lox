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


def test_door_glyph_discrimination_and_open_fallback():
    import nle.nethack as nh
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)

    # Fake glyph array: (5, 5) is closed door, (5, 6) is a spellbook object with ASCII '+'
    fake_glyphs = np.zeros((21, 79), dtype=np.int32)
    fake_glyphs[5, 5] = nh.GLYPH_CMAP_OFF + 15  # Closed door
    fake_glyphs[5, 6] = nh.GLYPH_OBJ_OFF + 300  # Spellbook object (char '+')

    doors_mask = adapter._get_doors_mask(fake_glyphs)
    assert doors_mask[5, 5] == True
    assert doors_mask[5, 6] == False  # Must NOT be marked as door!

    # Test open_door fallback to wait when no door adjacent
    t_before = obs.hero.turn
    obs, reward, term, trunc, info = adapter.step(Action(name="open_door"))
    # Graceful fallback to wait consumes 1 turn
    assert obs.hero.turn >= t_before

    adapter.close()


def test_unified_stride2_dead_ends_mask():
    adapter = NetHackAdapter()
    adapter.reset(seed=42)

    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    walkable = np.zeros((21, 79), dtype=bool)

    # 1. Setup a corridor: (5, 10) to (5, 12)
    # (5, 10) is a dead end (1 neighbor: (5, 11))
    # (5, 11) is a through-corridor (2 neighbors: (5, 10) and (5, 12))
    # (5, 12) is a dead end (1 neighbor: (5, 11))
    for cx in (10, 11, 12):
        chars[5, cx] = ord("#")
        walkable[5, cx] = True

    # 2. Setup a 3x3 room from (10, 10) to (12, 12) surrounded by walls
    # Walls at y=9, y=13 and x=9, x=13
    for cx in range(9, 14):
        chars[9, cx] = ord("-")
        chars[13, cx] = ord("-")
    for cy in range(9, 14):
        chars[cy, 9] = ord("|")
        chars[cy, 13] = ord("|")
    # Floors inside room
    for cy in range(10, 13):
        for cx in range(10, 13):
            chars[cy, cx] = ord(".")
            walkable[cy, cx] = True

    mask = adapter._compute_dead_ends_mask(chars, walkable)

    # Corridors: (5, 10) and (5, 12) are dead ends, (5, 11) is not
    assert mask[5, 10] == True
    assert mask[5, 12] == True
    assert mask[5, 11] == False

    # Room corners: (10, 10), (10, 12), (12, 10), (12, 12) have adj_wall >= 2
    # They MUST be candidates regardless of parity!
    assert mask[10, 10] == True
    assert mask[10, 12] == True
    assert mask[12, 10] == True
    assert mask[12, 12] == True

    # Check search threshold filtering
    adapter.searched_count[5, 10] = 15
    mask_after = adapter._compute_dead_ends_mask(chars, walkable)
    assert mask_after[5, 10] == False  # Corridor with 15 searches is pruned!

    adapter.searched_count[10, 10] = 10
    mask_after2 = adapter._compute_dead_ends_mask(chars, walkable)
    assert mask_after2[10, 10] == False  # Room wall with 10 searches is pruned!

    adapter.close()

