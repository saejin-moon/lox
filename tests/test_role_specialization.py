"""
Unit tests for Role Specialization, Metadata Parsing, Intrinsic Nutrition,
and Tactical Door Closing.
"""

import pytest
import numpy as np
import gymnasium as gym
import nle

from corp.env.blstats import BottomLineStats, ConditionFlag, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.planner.persona import parse_character_metadata
from corp.domain.inventory_manager import InventoryManager
from corp.domain.combat_manager import TacticalCombatManager, MonsterTrack
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
    score: int = 0,
    experience: int = 1,
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
    raw[nle.nethack.NLE_BL_SCORE] = score
    raw[nle.nethack.NLE_BL_EXP] = experience
    return BottomLineStats.from_blstats(raw)


def test_parse_character_metadata():
    # Standard NetHack startup messages
    valk_msg = "Welcome to NetHack! You are a neutral female human Valkyrie."
    meta = parse_character_metadata(valk_msg)
    assert meta["role"] == "valkyrie"
    assert meta["race"] == "human"
    assert meta["gender"] == "female"
    assert meta["alignment"] == "neutral"

    barb_msg = "Welcome to NetHack! You are a chaotic male orc Barbarian."
    meta = parse_character_metadata(barb_msg)
    assert meta["role"] == "barbarian"
    assert meta["race"] == "orc"
    assert meta["gender"] == "male"
    assert meta["alignment"] == "chaotic"

    wiz_msg = "You are a chaotic female elf Wizard."
    meta = parse_character_metadata(wiz_msg)
    assert meta["role"] == "wizard"
    assert meta["race"] == "elf"
    assert meta["gender"] == "female"
    assert meta["alignment"] == "chaotic"

    monk_msg = "You are a lawful male human Monk."
    meta = parse_character_metadata(monk_msg)
    assert meta["role"] == "monk"
    assert meta["race"] == "human"
    assert meta["gender"] == "male"
    assert meta["alignment"] == "lawful"

    priest_msg = "You are a lawful male human Priest."
    meta = parse_character_metadata(priest_msg)
    assert meta["role"] == "priest"
    assert meta["race"] == "human"
    assert meta["gender"] == "male"
    assert meta["alignment"] == "lawful"


def test_priest_automatic_uncursed_identification():
    # Normal role leaves unknown items as UNKNOWN
    normal_tracker = InventoryNormalizer(role="valkyrie")
    inv_strs = np.zeros((1, 80), dtype=np.uint8)
    inv_letters = np.zeros(1, dtype=np.uint8)
    inv_oclasses = np.zeros(1, dtype=np.uint8)
    line = b"a - a silver bell"
    inv_strs[0, :len(line)] = list(line)
    inv_letters[0] = ord("a")
    normal_tracker.synchronize(inv_strs, inv_letters, inv_oclasses)
    items = list(normal_tracker.active_items.values())
    assert len(items) == 1
    assert items[0].buc_state == "UNKNOWN"

    # Priest role automatically identifies items not marked blessed/cursed as UNCURSED
    priest_tracker = InventoryNormalizer(role="priest")
    priest_tracker.synchronize(inv_strs, inv_letters, inv_oclasses)
    p_items = list(priest_tracker.active_items.values())
    assert len(p_items) == 1
    assert p_items[0].buc_state == "UNCURSED"

    # Cursed item remains CURSED
    inv_strs_cursed = np.zeros((1, 80), dtype=np.uint8)
    line_c = b"b - a cursed mace"
    inv_strs_cursed[0, :len(line_c)] = list(line_c)
    inv_letters[0] = ord("b")
    priest_tracker.synchronize(inv_strs_cursed, inv_letters, inv_oclasses)
    cursed_item = [it for it in priest_tracker.active_items.values() if it.current_letter == "b"][0]
    assert cursed_item.buc_state == "CURSED"


def test_corpse_consumption_poison_resistance():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hunger_state=HungerState.HUNGRY)

    tracker = InventoryNormalizer()
    kobold_corpse = NormalizedItem(
        uid="c1",
        raw_str="a fresh kobold corpse",
        current_letter="a",
        oclass=nle.nethack.FOOD_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["c1"] = kobold_corpse

    # Valkyrie (human): rejects kobold corpse because it's poisonous
    task_valk = inv_mgr.evaluate_resource_turn(blstats, tracker, role="valkyrie", race="human")
    assert task_valk is None

    # Tourist (human): rejects kobold corpse
    task_tourist = inv_mgr.evaluate_resource_turn(blstats, tracker, role="tourist", race="human")
    assert task_tourist is None

    # Barbarian: innate poison resistance -> eats kobold corpse!
    task_barb = inv_mgr.evaluate_resource_turn(blstats, tracker, role="barbarian", race="human")
    assert task_barb is not None
    assert task_barb.name == "EAT"
    assert task_barb.args["slot"] == "a"

    # Orcish rogue: innate poison resistance -> eats kobold corpse!
    task_orc = inv_mgr.evaluate_resource_turn(blstats, tracker, role="rogue", race="orc")
    assert task_orc is not None
    assert task_orc.name == "EAT"
    assert task_orc.args["slot"] == "a"

    # Healer: innate poison resistance -> eats kobold corpse!
    task_healer = inv_mgr.evaluate_resource_turn(blstats, tracker, role="healer", race="human")
    assert task_healer is not None
    assert task_healer.name == "EAT"

    # Fatal corpses (cockatrice / chickatrice) must NEVER be eaten even by Barbarian
    tracker.active_items.clear()
    chickatrice = NormalizedItem(
        uid="c2",
        raw_str="a fresh chickatrice corpse",
        current_letter="b",
        oclass=nle.nethack.FOOD_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["c2"] = chickatrice
    task_fatal = inv_mgr.evaluate_resource_turn(blstats, tracker, role="barbarian", race="human")
    assert task_fatal is None


def test_spellcaster_armor_filtering():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats()

    # Wizard with plate mail in inventory should NOT equip it
    tracker_wiz = InventoryNormalizer(role="wizard")
    plate_mail = NormalizedItem(
        uid="a1",
        raw_str="an uncursed plate mail",
        current_letter="a",
        oclass=nle.nethack.ARMOR_CLASS,
        buc_state="UNCURSED",
    )
    tracker_wiz.active_items["a1"] = plate_mail
    task_wiz = inv_mgr.evaluate_resource_turn(blstats, tracker_wiz, role="wizard", race="human")
    assert task_wiz is None  # Rejects metallic plate mail

    # Wizard with robe in inventory SHOULD equip it
    robe = NormalizedItem(
        uid="a2",
        raw_str="an uncursed robe",
        current_letter="b",
        oclass=nle.nethack.ARMOR_CLASS,
        buc_state="UNCURSED",
    )
    tracker_wiz.active_items["a2"] = robe
    task_robe = inv_mgr.evaluate_resource_turn(blstats, tracker_wiz, role="wizard", race="human")
    assert task_robe is not None
    assert task_robe.name == "WEAR"
    assert task_robe.args["slot"] == "b"


def test_monk_restrictions():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats()

    tracker_monk = InventoryNormalizer(role="monk")
    leather_armor = NormalizedItem(
        uid="m1",
        raw_str="an uncursed leather jacket",
        current_letter="a",
        oclass=nle.nethack.ARMOR_CLASS,
        buc_state="UNCURSED",
    )
    sword = NormalizedItem(
        uid="m2",
        raw_str="an uncursed long sword",
        current_letter="b",
        oclass=nle.nethack.WEAPON_CLASS,
        buc_state="UNCURSED",
    )
    tracker_monk.active_items["m1"] = leather_armor
    tracker_monk.active_items["m2"] = sword

    # Monk must NOT wear body armor (preserves Monk AC bonus)
    # Monk must NOT wield weapons (preserves martial arts damage)
    task_monk = inv_mgr.evaluate_resource_turn(blstats, tracker_monk, role="monk", race="human")
    assert task_monk is None


def test_priest_blunt_weapon_preference():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats()

    tracker_priest = InventoryNormalizer(role="priest")
    sword = NormalizedItem(
        uid="p1",
        raw_str="an uncursed long sword",
        current_letter="a",
        oclass=nle.nethack.WEAPON_CLASS,
        buc_state="UNCURSED",
    )
    mace = NormalizedItem(
        uid="p2",
        raw_str="an uncursed mace",
        current_letter="b",
        oclass=nle.nethack.WEAPON_CLASS,
        buc_state="UNCURSED",
    )
    tracker_priest.active_items["p1"] = sword

    # Priest with only edged sword: must NOT wield sword
    task_sword = inv_mgr.evaluate_resource_turn(blstats, tracker_priest, role="priest", race="human")
    assert task_sword is None

    # Priest with mace: CAN wield mace
    tracker_priest.active_items["p2"] = mace
    task_mace = inv_mgr.evaluate_resource_turn(blstats, tracker_priest, role="priest", race="human")
    assert task_mace is not None
    assert task_mace.name == "WIELD"
    assert task_mace.args["slot"] == "b"


def test_unicorn_horn_status_cure():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(condition_mask=ConditionFlag.BLIND)

    tracker = InventoryNormalizer()
    uni_horn = NormalizedItem(
        uid="u1",
        raw_str="an uncursed unicorn horn",
        current_letter="u",
        oclass=nle.nethack.TOOL_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["u1"] = uni_horn

    task = inv_mgr.evaluate_resource_turn(blstats, tracker)
    assert task is not None
    assert task.name == "APPLY"
    assert task.args["slot"] == "u"


def test_tactical_door_closing():
    env = gym.make("NetHackChallenge-v0")
    env.reset()
    combat_mgr = TacticalCombatManager()
    dispatcher = ActionDispatcher(env)

    # Hero is at (10, 10). Adjacent open door is at (10, 11) (East).
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 11] = ord("'")  # Open door
    chars[10, 9] = ord("#")   # Corridor behind

    blstats = make_test_blstats(x=10, y=10)

    # Pursuing monster on other side of door at (10, 12)
    monster = MonsterTrack(
        pos=(10, 12),
        name="jackal",
        level=1,
        speed=12,
        ac=7,
        distance=2,
        is_adjacent=False,
        is_instakill=False,
        threat_score=1.0,
    )

    door_task = combat_mgr._evaluate_door_closing(chars, blstats.y, blstats.x, [monster])
    assert door_task is not None
    assert door_task.name == "CLOSE_DOOR"
    assert door_task.args["delta"] == (0, 1)

    # Dispatcher executes CLOSE action followed by direction
    obs, r, term, trunc, info = dispatcher.dispatch(door_task)
    assert "glyphs" in obs
