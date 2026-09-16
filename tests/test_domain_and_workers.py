"""
Unit tests for Domain Managers (Combat, Navigation, Inventory) and ActionDispatcher.
"""

import pytest
import numpy as np
import gymnasium as gym
import nle

from corp.env.blstats import BottomLineStats, ConditionFlag, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.domain.combat_manager import TacticalCombatManager, MonsterTrack
from corp.domain.navigation_manager import NavigationManager
from corp.domain.inventory_manager import InventoryManager
from corp.workers.dispatcher import ActionDispatcher, build_action_tables
from corp.planner.htn import Task


def make_test_blstats(
    hp: int = 16,
    max_hp: int = 16,
    x: int = 10,
    y: int = 10,
    turn: int = 10,
    depth: int = 1,
    hunger_state: int = HungerState.NORMAL,
    condition_mask: int = 0,
) -> BottomLineStats:
    raw = np.zeros(26, dtype=np.int32)
    raw[nle.nethack.NLE_BL_X] = x
    raw[nle.nethack.NLE_BL_Y] = y
    raw[nle.nethack.NLE_BL_HP] = hp
    raw[nle.nethack.NLE_BL_HPMAX] = max_hp
    raw[nle.nethack.NLE_BL_DEPTH] = depth
    raw[nle.nethack.NLE_BL_TIME] = turn
    raw[nle.nethack.NLE_BL_HUNGER] = hunger_state
    raw[nle.nethack.NLE_BL_CONDITION] = condition_mask
    return BottomLineStats.from_blstats(raw)


def test_action_dispatcher_tables():
    env = gym.make("NetHackChallenge-v0")
    delta_to_idx, name_to_idx, char_to_idx = build_action_tables(env)

    # Check cardinal directions
    assert (-1, 0) in delta_to_idx  # North
    assert (0, 1) in delta_to_idx   # East
    assert (1, 0) in delta_to_idx   # South
    assert (0, -1) in delta_to_idx  # West

    # Check core primitives
    assert "SEARCH" in name_to_idx
    assert "WAIT" in name_to_idx
    assert "EAT" in name_to_idx
    assert "WIELD" in name_to_idx
    assert "ENGRAVE" in name_to_idx
    assert "PRAY" in name_to_idx


def test_combat_manager_floating_eye_lockout():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=10, y=10)

    # Create synthetic glyphs array with floating eye adjacent to player at (10, 11)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 9] = ord(".")   # Walkable retreat tile to West
    chars[10, 11] = ord("e")  # Eye to East

    # Find floating eye monster id
    eye_mon_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "floating eye" in pm.mname:
            eye_mon_id = m
            break
    assert eye_mon_id is not None
    glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + eye_mon_id

    # Player is NOT blind
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    # Must NOT be melee attack!
    assert task.name != "MELEE_ATTACK"
    # Should step away
    assert task.name in ("STEP", "WAIT")


def test_combat_manager_emergency_elbereth():
    combat_mgr = TacticalCombatManager()
    # Player at critical HP (2 out of 16)
    blstats = make_test_blstats(hp=2, max_hp=16, x=10, y=10)

    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # Place jackal adjacent at (10, 11)
    jackal_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "jackal" in pm.mname:
            jackal_id = m
            break
    assert jackal_id is not None
    glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + jackal_id

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    assert task.name == "ENGRAVE_DUST"
    assert task.args.get("text") == "Elbereth"


def test_combat_manager_standard_melee():
    combat_mgr = TacticalCombatManager()
    # Healthy player (16/16)
    blstats = make_test_blstats(hp=16, max_hp=16, x=10, y=10)

    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # Place goblin adjacent at (9, 10) (North)
    goblin_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "goblin" in pm.mname:
            goblin_id = m
            break
    assert goblin_id is not None
    glyphs[9, 10] = nle.nethack.GLYPH_MON_OFF + goblin_id

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    assert task.name == "MELEE_ATTACK"
    assert task.args["delta"] == (-1, 0)


def test_inventory_manager_nutrition():
    inv_mgr = InventoryManager()
    # Normal hunger -> no eating
    blstats = make_test_blstats(hunger_state=HungerState.NORMAL)
    inv = InventoryNormalizer()
    item = NormalizedItem(current_letter="b", raw_str="a food ration")
    inv.active_items[item.uid] = item
    assert inv_mgr.evaluate_resource_turn(blstats, inv) is None

    # Hungry -> eat ration
    blstats_hungry = make_test_blstats(hunger_state=HungerState.HUNGRY)
    task = inv_mgr.evaluate_resource_turn(blstats_hungry, inv)
    assert task is not None
    assert task.name == "EAT"
    assert task.args["slot"] == "b"


def test_inventory_manager_prayer_cooldown():
    inv_mgr = InventoryManager()
    # Turn 50, critical HP -> initial cooldown (300) not satisfied
    blstats_early = make_test_blstats(hp=3, max_hp=16, turn=50)
    inv = InventoryNormalizer()
    assert inv_mgr.evaluate_resource_turn(blstats_early, inv) is None

    # Turn 150, critical HP -> emergency cutoff (101) satisfied!
    blstats_emergency = make_test_blstats(hp=3, max_hp=16, turn=150)
    task = inv_mgr.evaluate_resource_turn(blstats_emergency, inv)
    assert task is not None
    assert task.name == "PRAY"


def test_navigation_manager_frontier_and_search():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5)

    # Map with floor and unmapped space
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord(".")
    chars[5, 7] = ord(".")

    task = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task is not None
    # Will path towards frontier (5, 7) or search
    assert task.name in ("STEP", "SEARCH")


def test_combat_manager_healing_rest_and_peaceful():
    combat_mgr = TacticalCombatManager()
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # 1. Damaged player (10/20 HP) with NO hostiles -> should REST (WAIT)
    blstats_damaged = make_test_blstats(hp=10, max_hp=20, hunger_state=HungerState.NORMAL)
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats_damaged)
    assert task is not None
    assert task.name == "WAIT"

    # 2. Healthy player (20/20 HP) -> no combat action needed (returns None)
    blstats_healthy = make_test_blstats(hp=20, max_hp=20, hunger_state=HungerState.NORMAL)
    assert combat_mgr.evaluate_combat_turn(glyphs, chars, blstats_healthy) is None

    # 3. Peaceful shopkeeper nearby -> should NOT attack
    shopkeeper_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "shopkeeper" in pm.mname:
            shopkeeper_id = m
            break
    if shopkeeper_id is not None:
        glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + shopkeeper_id
        monsters = combat_mgr.scan_monsters(glyphs, blstats_healthy)
        assert len(monsters) == 0  # Ignored peaceful entity


def test_inventory_manager_floor_pickup():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    inv = InventoryNormalizer()

    # Step on food ration message -> issue PICKUP
    msg = "You see here a food ration."
    task = inv_mgr.evaluate_resource_turn(blstats, inv, message=msg)
    assert task is not None
    assert task.name == "PICKUP"

    # Subsequent check at same location should not repeat pickup
    task2 = inv_mgr.evaluate_resource_turn(blstats, inv, message=msg)
    assert task2 is None or task2.name != "PICKUP"

