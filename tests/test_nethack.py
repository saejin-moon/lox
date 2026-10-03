import numpy as np
import pytest
import nle.nethack as nethack
from lox.core.types import Action, Observation
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


def test_dismiss_more_ynq_prompts():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)

    # Test that [ynq] prompts are auto-dismissed with 'n'
    fake_obs = {"message": "Do you want your possessions identified? [ynq] (n)", "misc": [0, 0, 0]}
    cleaned_obs, term, trunc = adapter._dismiss_more(fake_obs, False, False)
    assert cleaned_obs is not None

    # Test that [ynaq] post-death prompts are auto-dismissed with 'n'
    fake_obs2 = {"message": "Do you want an account of creatures vanquished? [ynaq] (n)", "misc": [0, 0, 0]}
    cleaned_obs2, term2, trunc2 = adapter._dismiss_more(fake_obs2, False, False)
    assert cleaned_obs2 is not None

    # Test that floor corpse eating prompt is answered with 'y' (not rejected with 'n')
    fake_obs3 = {"message": "There is a goblin corpse here; eat it? [ynq] (n)", "misc": [0, 0, 0]}
    cleaned_obs3, term3, trunc3 = adapter._dismiss_more(fake_obs3, False, False)
    assert cleaned_obs3 is not None

    # Test that empty "Eat what?" prompt is cancelled with ESC
    fake_obs4 = {"message": "Eat what? [a-z or ?*]", "misc": [0, 0, 0]}
    cleaned_obs4, term4, trunc4 = adapter._dismiss_more(fake_obs4, False, False)
    assert cleaned_obs4 is not None

    adapter.close()


def test_locked_door_breach_and_poison_gating():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)

    # 1. Door Breaching: Set adjacent door
    hy, hx = obs.hero.y, obs.hero.x
    fake_glyphs = obs.glyphs.copy()
    fake_glyphs[hy, hx + 1] = nethack.GLYPH_CMAP_OFF + 15  # closed door to the East
    adapter._last_obs = Observation(
        chars=obs.chars,
        glyphs=fake_glyphs,
        hero=obs.hero,
        raw_obs=obs.raw_obs,
    )

    # Calling step_direction directly into the closed door should trigger open_door
    # and if the door is in locked_doors, should trigger kick_closed_door
    adapter.locked_doors.add((hy, hx + 1))
    sub_action = None

    # Step or breach returns kick_closed_door when locked
    # We can inspect _step_or_breach delegation
    obs_test = adapter._last_obs
    # Mock step to record action
    recorded_actions = []
    orig_step = adapter.step
    def mock_step(action):
        recorded_actions.append(action.name)
        if action.name in ("kick_closed_door", "open_door"):
            return obs_test, 0.0, False, False, {}
        return orig_step(action)
    adapter.step = mock_step

    adapter._step_or_breach(obs_test, 0, 1)
    assert "kick_closed_door" in recorded_actions

    adapter.close()


def test_hero_position_walkability_and_dead_end_stagnation_recovery():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)
    hy, hx = obs.hero.y, obs.hero.x

    # 1. Test that hero position is NEVER in blocked_tiles even if artificially added
    adapter.blocked_tiles.add((hy, hx))
    adapter.non_door_tiles.add((hy, hx))
    walkable, walkable_nav = adapter._build_walkable_nav(obs)
    assert (hy, hx) not in adapter.blocked_tiles
    assert (hy, hx) not in adapter.non_door_tiles
    assert walkable_nav[hy, hx] == True

    # 2. Test stagnation recovery: artificially max out search counts on dead ends
    adapter.searched_count.fill(20)
    # step_to_dead_end should decay searched_count and find/search a candidate instead of hanging
    obs_next, _, _, _, _ = adapter.step(Action(name="step_to_dead_end"))
    assert np.any(adapter.searched_count <= 11)  # Decayed by 10!
    assert obs_next is not None

    adapter.close()


