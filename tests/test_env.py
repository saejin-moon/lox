"""
Unit tests for Phase 1: Sensory Boundary, AutoMoreWrapper, InventoryNormalizer,
AnomalySentry, and FlightRecorder.
"""

import numpy as np
import pytest
from nle import nethack

from lox.env.blstats import BottomLineStats, ConditionFlag, HungerState
from lox.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from lox.env.anomaly_sentry import AnomalySentry, AnomalySeverity
from lox.env.flight_recorder import FlightRecorderRingBuffer
from lox.env.nle_wrapper import make_env


def test_make_env_reset():
    env = make_env("NetHackChallenge-v0")
    obs, info = env.reset()

    assert "glyphs" in obs
    assert "blstats" in obs
    assert "blstats" in info
    blstats = info["blstats"]
    assert isinstance(blstats, BottomLineStats)
    assert blstats.hp > 0
    assert blstats.max_hp > 0
    assert blstats.depth >= 1

    tracker = info["inventory_tracker"]
    assert isinstance(tracker, InventoryNormalizer)
    active_items = tracker.get_active_items()
    assert len(active_items) > 0

    # Ensure items have stable UIDs
    for item in active_items:
        assert item.uid is not None
        assert len(item.uid) > 0
        assert item.current_letter != ""

    env.close()


def test_env_stepping_and_flight_recorder():
    env = make_env("NetHackChallenge-v0", flight_recorder_capacity=20)
    obs, info = env.reset()

    # Step a few times (Wait / rest action '.')
    action_wait = nethack.ACTIONS.index(ord("."))
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(action_wait)
        if term or trunc:
            break

    recorder = env.flight_recorder
    records = recorder.get_recent_history(10)
    assert len(records) >= 5

    summary = recorder.export_autopsy_summary(recent_turns=5)
    assert "Flight Recorder Autopsy" in summary
    assert "Final HP" in summary

    env.close()


def test_inventory_normalizer_letter_shift():
    normalizer = InventoryNormalizer()

    # Initial state: 3 items in slots a, b, c
    inv_strs = np.zeros((55, 80), dtype=np.uint8)
    inv_letters = np.zeros(55, dtype=np.uint8)
    inv_glyphs = np.zeros(55, dtype=np.int16)

    # Item a: long sword (glyph 100)
    inv_letters[0] = ord("a")
    for idx, c in enumerate(b"a - a blessed long sword"):
        inv_strs[0, idx] = c
    inv_glyphs[0] = 100

    # Item b: ration (glyph 200)
    inv_letters[1] = ord("b")
    for idx, c in enumerate(b"b - a food ration"):
        inv_strs[1, idx] = c
    inv_glyphs[1] = 200

    # Item c: healing potion (glyph 300)
    inv_letters[2] = ord("c")
    for idx, c in enumerate(b"c - a potion of extra healing"):
        inv_strs[2, idx] = c
    inv_glyphs[2] = 300

    normalizer.synchronize(inv_strs, inv_letters, inv_glyphs, turn=1)

    sword_uid = normalizer.letter_to_uid["a"]
    ration_uid = normalizer.letter_to_uid["b"]
    potion_uid = normalizer.letter_to_uid["c"]

    assert sword_uid != ration_uid != potion_uid

    # Now drop the food ration: item c shifts to letter b!
    inv_strs_2 = np.zeros((55, 80), dtype=np.uint8)
    inv_letters_2 = np.zeros(55, dtype=np.uint8)
    inv_glyphs_2 = np.zeros(55, dtype=np.int16)

    inv_letters_2[0] = ord("a")
    for idx, c in enumerate(b"a - a blessed long sword"):
        inv_strs_2[0, idx] = c
    inv_glyphs_2[0] = 100

    inv_letters_2[1] = ord("b")  # Potion shifted into slot 'b'!
    for idx, c in enumerate(b"b - a potion of extra healing"):
        inv_strs_2[1, idx] = c
    inv_glyphs_2[1] = 300

    normalizer.synchronize(inv_strs_2, inv_letters_2, inv_glyphs_2, turn=2)

    # The potion's UID MUST remain identical to potion_uid!
    assert normalizer.letter_to_uid["b"] == potion_uid
    assert normalizer.get_letter(potion_uid) == "b"
    assert normalizer.active_items[ration_uid].is_active is False


def test_anomaly_sentry_burst_and_lethal():
    sentry = AnomalySentry()

    raw_bl_1 = np.zeros(27, dtype=np.int64)
    raw_bl_1[nethack.NLE_BL_HP] = 20
    raw_bl_1[nethack.NLE_BL_HPMAX] = 20
    raw_bl_1[nethack.NLE_BL_TIME] = 10
    raw_bl_1[nethack.NLE_BL_X] = 10
    raw_bl_1[nethack.NLE_BL_Y] = 10

    bl_1 = BottomLineStats.from_blstats(raw_bl_1)
    events_1 = sentry.evaluate(bl_1)
    assert len(events_1) == 0

    # Step 2: Sudden burst damage: HP drops from 20 to 12 (-8 HP, 40% loss)
    raw_bl_2 = np.zeros(27, dtype=np.int64)
    raw_bl_2[nethack.NLE_BL_HP] = 12
    raw_bl_2[nethack.NLE_BL_HPMAX] = 20
    raw_bl_2[nethack.NLE_BL_TIME] = 11
    raw_bl_2[nethack.NLE_BL_X] = 10
    raw_bl_2[nethack.NLE_BL_Y] = 10

    bl_2 = BottomLineStats.from_blstats(raw_bl_2)
    events_2 = sentry.evaluate(bl_2)
    assert any(e.condition_name == "BURST_DAMAGE" for e in events_2)

    # Step 3: Lethal status: Petrification bit turned on
    raw_bl_3 = np.zeros(27, dtype=np.int64)
    raw_bl_3[nethack.NLE_BL_HP] = 12
    raw_bl_3[nethack.NLE_BL_HPMAX] = 20
    raw_bl_3[nethack.NLE_BL_TIME] = 12
    raw_bl_3[nethack.NLE_BL_X] = 10
    raw_bl_3[nethack.NLE_BL_Y] = 10
    raw_bl_3[nethack.NLE_BL_CONDITION] = ConditionFlag.STONE

    bl_3 = BottomLineStats.from_blstats(raw_bl_3)
    events_3 = sentry.evaluate(bl_3)
    assert any(e.condition_name == "LETHAL_STATUS" for e in events_3)


def test_auto_more_wishing_and_payment_interception():
    from lox.env.auto_more import AutoMoreWrapper
    import gymnasium as gym

    class MockInnerEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.actions = [type("Act", (), {"value": i})() for i in range(128)]

    mock_env = MockInnerEnv()
    wrapper = AutoMoreWrapper(mock_env)

    # 1. Wish prompt detection
    obs_wish = {"message": np.frombuffer(b"For what do you wish?\x00", dtype=np.uint8)}
    assert wrapper._is_wish_prompt(obs_wish) is True

    # 2. Payment confirmation
    obs_pay = {"message": np.frombuffer(b"Pay 40 zm for a food ration? [yn]\x00", dtype=np.uint8)}
    assert wrapper._is_yn_prompt(obs_pay) is True
    assert wrapper._resolve_yn_action(obs_pay) == AutoMoreWrapper.ACTION_Y


def test_auto_more_menu_dismissal():
    from lox.env.auto_more import AutoMoreWrapper
    import gymnasium as gym

    # Verify menu patterns (X of Y) and (end) are recognized as needing pagination/dismissal
    class MockInnerEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.actions = [type("Act", (), {"value": i})() for i in range(128)]

    mock_env = MockInnerEnv()
    wrapper = AutoMoreWrapper(mock_env)

    obs_menu_p1 = {"tty_chars": np.frombuffer(b"Skills (1 of 2) ...", dtype=np.uint8)}
    obs_menu_p2 = {"tty_chars": np.frombuffer(b"Skills (2 of 2) ...", dtype=np.uint8)}
    obs_menu_end = {"tty_chars": np.frombuffer(b"Skills (end) ...", dtype=np.uint8)}

    assert wrapper._is_more(obs_menu_p1) is True
    assert wrapper._is_more(obs_menu_p2) is True
    assert wrapper._is_more(obs_menu_end) is True
    assert wrapper.ACTION_ESC == 38
    assert wrapper.ACTION_SPACE == 107

