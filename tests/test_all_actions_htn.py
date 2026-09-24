"""
Comprehensive unit tests for All Incorporated Actions in HTN and ActionDispatcher:
- ActionDispatcher multi-keystroke handlers (FIRE, QUIVER, ZAP, CAST, PUTON, REMOVE, WIPE, UNTRAP, FORCE, LOOT, SIT, TURN, TWOWEAPON, SWAP, RUB, CHAT)
- TacticalCombatManager: WIPE, TURN_UNDEAD, ZAP_WAND, FIRE, THROW
- InventoryManager: READ (teleport/remove curse), PUTON (slow digestion/free action), RUB (magic lamp)
- NavigationManager: SIT (throne), LOOT (container), UNTRAP (adjacent trap)
"""

import pytest
import numpy as np
import gymnasium as gym
import nle

from lox.env.blstats import BottomLineStats, ConditionFlag, HungerState
from lox.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from lox.domain.combat_manager import TacticalCombatManager, MonsterTrack
from lox.domain.inventory_manager import InventoryManager
from lox.domain.navigation_manager import NavigationManager
from lox.workers.dispatcher import ActionDispatcher, build_action_tables
from lox.planner.htn import Task


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


def test_dispatcher_all_actions():
    env = gym.make("NetHackChallenge-v0")
    env.reset()
    dispatcher = ActionDispatcher(env)

    # Test FIRE
    obs, r, term, trunc, info = dispatcher.dispatch(Task("FIRE", is_primitive=True, args={"delta": (0, 1)}))
    assert "glyphs" in obs

    # Test QUIVER
    obs, r, term, trunc, info = dispatcher.dispatch(Task("QUIVER", is_primitive=True, args={"slot": "a"}))
    assert "glyphs" in obs

    # Test ZAP_WAND
    obs, r, term, trunc, info = dispatcher.dispatch(Task("ZAP_WAND", is_primitive=True, args={"slot": "z", "delta": (0, 1)}))
    assert "glyphs" in obs

    # Test CAST_SPELL
    obs, r, term, trunc, info = dispatcher.dispatch(Task("CAST_SPELL", is_primitive=True, args={"spell": "a", "delta": (0, 1)}))
    assert "glyphs" in obs

    # Test PUTON
    obs, r, term, trunc, info = dispatcher.dispatch(Task("PUTON", is_primitive=True, args={"slot": "p", "hand": "r"}))
    assert "glyphs" in obs

    # Test REMOVE
    obs, r, term, trunc, info = dispatcher.dispatch(Task("REMOVE", is_primitive=True, args={"slot": "p"}))
    assert "glyphs" in obs

    # Test WIPE
    obs, r, term, trunc, info = dispatcher.dispatch(Task("WIPE", is_primitive=True))
    assert "glyphs" in obs

    # Test UNTRAP
    obs, r, term, trunc, info = dispatcher.dispatch(Task("UNTRAP", is_primitive=True, args={"delta": (0, 1)}))
    assert "glyphs" in obs

    # Test FORCE_LOCK
    obs, r, term, trunc, info = dispatcher.dispatch(Task("FORCE_LOCK", is_primitive=True, args={"delta": (0, 1)}))
    assert "glyphs" in obs

    # Test LOOT_CONTAINER
    obs, r, term, trunc, info = dispatcher.dispatch(Task("LOOT_CONTAINER", is_primitive=True, args={"delta": (0, 0)}))
    assert "glyphs" in obs

    # Test SIT
    obs, r, term, trunc, info = dispatcher.dispatch(Task("SIT", is_primitive=True))
    assert "glyphs" in obs

    # Test TURN_UNDEAD
    obs, r, term, trunc, info = dispatcher.dispatch(Task("TURN_UNDEAD", is_primitive=True))
    assert "glyphs" in obs

    # Test TWOWEAPON
    obs, r, term, trunc, info = dispatcher.dispatch(Task("TWOWEAPON", is_primitive=True))
    assert "glyphs" in obs

    # Test SWAP_WEAPON
    obs, r, term, trunc, info = dispatcher.dispatch(Task("SWAP_WEAPON", is_primitive=True))
    assert "glyphs" in obs

    # Test RUB
    obs, r, term, trunc, info = dispatcher.dispatch(Task("RUB", is_primitive=True, args={"slot": "l"}))
    assert "glyphs" in obs

    # Test CHAT
    obs, r, term, trunc, info = dispatcher.dispatch(Task("CHAT", is_primitive=True, args={"delta": (0, 1), "amount": 400}))
    assert "glyphs" in obs


def test_combat_manager_wipe_and_turn_undead():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(condition_mask=ConditionFlag.BLIND)
    glyphs = np.zeros((21, 79), dtype=np.int16)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)

    # 1. Blinded by venom -> triggers WIPE
    task_wipe = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats, role="valkyrie", message="A spray of venom hits your face! You can't see!"
    )
    assert task_wipe is not None
    assert task_wipe.name == "WIPE"

    # 2. Priest seeing hostile undead -> triggers TURN_UNDEAD
    blstats_normal = make_test_blstats()
    # Put two zombies
    glyphs[10, 13] = 100  # monster glyph
    glyphs[10, 14] = 101  # monster glyph
    chars[10, 13] = ord("Z")
    chars[10, 14] = ord("Z")

    # Mock scan_monsters to return undead monsters
    m1 = MonsterTrack(pos=(10, 13), name="kobold zombie", level=1, speed=6, ac=10, distance=3, is_adjacent=False, is_instakill=False, threat_score=1.0)
    m2 = MonsterTrack(pos=(10, 14), name="human zombie", level=2, speed=6, ac=9, distance=4, is_adjacent=False, is_instakill=False, threat_score=2.0)
    combat_mgr.scan_monsters = lambda g, b: [m1, m2]

    task_turn = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats_normal, role="priest", message=""
    )
    assert task_turn is not None
    assert task_turn.name == "TURN_UNDEAD"


def test_combat_manager_ranged_wand_and_quiver():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(x=10, y=10)
    glyphs = np.zeros((21, 79), dtype=np.int16)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 10] = ord("@")

    # Hostile floating eye at (10, 13) (East, distance 3)
    eye = MonsterTrack(pos=(10, 13), name="floating eye", level=2, speed=0, ac=9, distance=3, is_adjacent=False, is_instakill=True, threat_score=200.0)
    combat_mgr.scan_monsters = lambda g, b: [eye]

    inv_tracker = InventoryNormalizer()
    wand = NormalizedItem(uid="w1", raw_str="an uncursed wand of striking (0:4)", current_letter="z", oclass=nle.nethack.WAND_CLASS, buc_state="UNCURSED")
    inv_tracker.active_items["w1"] = wand

    # High threat instakill with wand -> ZAP_WAND
    task_wand = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=inv_tracker)
    assert task_wand is not None
    assert task_wand.name == "ZAP_WAND"
    assert task_wand.args["slot"] == "z"
    assert task_wand.args["delta"] == (0, 1)

    # Without wand, but with quivered arrows -> FIRE
    inv_tracker.active_items.clear()
    arrow = NormalizedItem(uid="a1", raw_str="25 uncursed arrows (in quiver)", current_letter="f", oclass=nle.nethack.WEAPON_CLASS, buc_state="UNCURSED")
    inv_tracker.active_items["a1"] = arrow
    task_fire = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=inv_tracker)
    assert task_fire is not None
    assert task_fire.name == "FIRE"
    assert task_fire.args["delta"] == (0, 1)


def test_inventory_manager_emergency_scrolls_and_rings():
    inv_mgr = InventoryManager()

    # 1. Critical HP (4/20 HP) with scroll of teleportation -> READ
    blstats_crit = make_test_blstats(hp=4, max_hp=20)
    tracker = InventoryNormalizer()
    tele_scroll = NormalizedItem(uid="s1", raw_str="an uncursed scroll of teleportation", current_letter="t", oclass=nle.nethack.SCROLL_CLASS, buc_state="UNCURSED")
    tracker.active_items["s1"] = tele_scroll

    task_read = inv_mgr.evaluate_resource_turn(blstats_crit, tracker)
    assert task_read is not None
    assert task_read.name == "READ"
    assert task_read.args["slot"] == "t"

    # 2. Cursed weapon wielded with scroll of remove curse -> READ
    blstats_normal = make_test_blstats(hp=20, max_hp=20)
    tracker.active_items.clear()
    cursed_sword = NormalizedItem(uid="w1", raw_str="a cursed broadsword (weapon in hand)", current_letter="a", oclass=nle.nethack.WEAPON_CLASS, buc_state="CURSED", equipped=True)
    rc_scroll = NormalizedItem(uid="s2", raw_str="an uncursed scroll of remove curse", current_letter="r", oclass=nle.nethack.SCROLL_CLASS, buc_state="UNCURSED")
    tracker.active_items["w1"] = cursed_sword
    tracker.active_items["s2"] = rc_scroll

    task_rc = inv_mgr.evaluate_resource_turn(blstats_normal, tracker)
    assert task_rc is not None
    assert task_rc.name == "READ"
    assert task_rc.args["slot"] == "r"

    # 3. Uncursed Ring of Slow Digestion -> PUTON
    tracker.active_items.clear()
    ring = NormalizedItem(uid="r1", raw_str="an uncursed ring of slow digestion", current_letter="d", oclass=nle.nethack.RING_CLASS, buc_state="UNCURSED")
    tracker.active_items["r1"] = ring
    task_ring = inv_mgr.evaluate_resource_turn(blstats_normal, tracker)
    assert task_ring is not None
    assert task_ring.name == "PUTON"
    assert task_ring.args["slot"] == "d"

    # 4. Cursed Ring must NEVER be put on!
    tracker.active_items.clear()
    cursed_ring = NormalizedItem(uid="r2", raw_str="a cursed ring of slow digestion", current_letter="c", oclass=nle.nethack.RING_CLASS, buc_state="CURSED")
    tracker.active_items["r2"] = cursed_ring
    task_cursed_ring = inv_mgr.evaluate_resource_turn(blstats_normal, tracker)
    assert task_cursed_ring is None

    # 5. Magic lamp rubbing -> RUB
    tracker.active_items.clear()
    lamp = NormalizedItem(uid="l1", raw_str="an uncursed magic lamp", current_letter="m", oclass=nle.nethack.TOOL_CLASS, buc_state="UNCURSED")
    tracker.active_items["l1"] = lamp
    task_rub = inv_mgr.evaluate_resource_turn(blstats_normal, tracker)
    assert task_rub is not None
    assert task_rub.name == "RUB"
    assert task_rub.args["slot"] == "m"


def test_navigation_manager_throne_container_and_trap():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=10, y=10)

    # 1. Standing on a throne ('\') -> SIT
    chars_throne = np.full((21, 79), ord("."), dtype=np.uint8)
    chars_throne[10, 10] = ord("\\")
    # Throne sitting is disabled: NetHack 3.6.6 throne effects include fatal electric
    # shocks ("You get zapped!") and hostile summonings — the agent walks over thrones.
    task_sit = nav_mgr.evaluate_navigation_turn(chars_throne, blstats)
    assert task_sit.name != "SIT"

    # Second turn on same throne should not sit again
    task_sit_2 = nav_mgr.evaluate_navigation_turn(chars_throne, blstats)
    assert task_sit_2.name != "SIT"

    # 2. Standing on a container ('(') -> LOOT
    nav_mgr.reset()
    chars_box = np.full((21, 79), ord("."), dtype=np.uint8)
    chars_box[10, 10] = ord("(")
    task_loot = nav_mgr.evaluate_navigation_turn(chars_box, blstats)
    assert task_loot.name == "LOOT"
    assert task_loot.args["delta"] == (0, 0)

    # 3. Discovered trap ('^') adjacent at (10, 11) -> UNTRAP
    nav_mgr.reset()
    chars_trap = np.full((21, 79), ord("."), dtype=np.uint8)
    chars_trap[10, 10] = ord("@")
    chars_trap[10, 11] = ord("^")
    task_untrap = nav_mgr.evaluate_navigation_turn(chars_trap, blstats)
    assert task_untrap.name == "UNTRAP"
    assert task_untrap.args["delta"] == (0, 1)
