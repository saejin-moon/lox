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
from corp.workers.dispatcher import ActionDispatcher, MicroActionFiber, build_action_tables
from corp.epistemic.epistemic_manager import EpistemicManager
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

    # Turn 150, critical HP -> initial cooldown (300) still unready
    blstats_mid = make_test_blstats(hp=3, max_hp=16, turn=150)
    assert inv_mgr.evaluate_resource_turn(blstats_mid, inv) is None

    # Turn 301, critical HP -> initial cooldown (300) satisfied!
    blstats_emergency = make_test_blstats(hp=3, max_hp=16, turn=301)
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

    # 1. Damaged player (7/20 HP, <= 40%) with NO hostiles -> should REST (WAIT)
    blstats_damaged = make_test_blstats(hp=7, max_hp=20, hunger_state=HungerState.NORMAL)
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


def test_inventory_manager_emergency_quaff():
    inv_mgr = InventoryManager()
    # Critical HP (5 out of 20, <= 50%)
    blstats = make_test_blstats(hp=5, max_hp=20)
    inv = InventoryNormalizer()
    item = NormalizedItem(current_letter="c", raw_str="a potion of extra healing", buc_state="UNCURSED")
    inv.active_items[item.uid] = item

    task = inv_mgr.evaluate_resource_turn(blstats, inv)
    assert task is not None
    assert task.name == "QUAFF"
    assert task.args["slot"] == "c"


def test_inventory_manager_excalibur_dip():
    inv_mgr = InventoryManager()
    # Valkyrie at XL 5, standing on fountain with uncursed long sword
    blstats = make_test_blstats(experience=5, x=5, y=5)
    inv = InventoryNormalizer()
    sword = NormalizedItem(current_letter="b", raw_str="an uncursed long sword", buc_state="UNCURSED")
    inv.active_items[sword.uid] = sword

    task = inv_mgr.evaluate_resource_turn(blstats, inv, is_on_fountain=True, role="valkyrie")
    assert task is not None
    assert task.name == "DIP"
    assert task.args["slot"] == "b"


def test_inventory_manager_dynamic_weapon_upgrade():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats()
    inv = InventoryNormalizer()
    # Wielding dagger (rank 4), has katana (rank 9) in pack
    dagger = NormalizedItem(current_letter="a", raw_str="an uncursed dagger (weapon in hand)", equipped=True, buc_state="UNCURSED")
    katana = NormalizedItem(current_letter="b", raw_str="an uncursed katana", equipped=False, buc_state="UNCURSED")
    inv.active_items[dagger.uid] = dagger
    inv.active_items[katana.uid] = katana

    task = inv_mgr.evaluate_resource_turn(blstats, inv)
    assert task is not None
    assert task.name == "WIELD"
    assert task.args["slot"] == "b"


def test_inventory_manager_6slot_armor_optimization():
    inv_mgr = InventoryManager()
    blstats = make_test_blstats()
    inv = InventoryNormalizer()
    # Unarmored, has helmet and boots in pack
    helm = NormalizedItem(current_letter="h", raw_str="an uncursed helm of telepathy", equipped=False, buc_state="UNCURSED")
    boots = NormalizedItem(current_letter="f", raw_str="uncursed speed boots", equipped=False, buc_state="UNCURSED")
    inv.active_items[helm.uid] = helm
    inv.active_items[boots.uid] = boots

    task = inv_mgr.evaluate_resource_turn(blstats, inv)
    assert task is not None
    assert task.name == "WEAR"
    assert task.args["slot"] in ("h", "f")


def test_navigation_manager_fountain_targeting():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord(".")
    chars[5, 7] = ord("{")  # Fountain
    chars[6, 5] = ord(">")  # Stairs down

    # When target_fountain is True, paths towards fountain at (5, 7) rather than stairs down at (6, 5)
    task = nav_mgr.evaluate_navigation_turn(chars, blstats, target_fountain=True)
    assert task is not None
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)  # Moves East towards fountain


def test_navigation_manager_door_interaction_kick():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord("+")  # Closed door to East

    # First attempt -> OPEN
    task1 = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task1 is not None
    assert task1.name == "OPEN"
    assert task1.args["delta"] == (0, 1)

    # Second attempt on same door -> KICK
    task2 = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task2 is not None
    assert task2.name == "KICK"
    assert task2.args["delta"] == (0, 1)


def test_navigation_manager_frontier_exploration():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord(".")
    chars[5, 7] = ord("#")  # Corridor opening into unmapped space

    task = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task is not None
    assert task.name == "STEP"


def test_skill_worker_enhancement():
    from corp.domain.skill_worker import SkillWorker

    worker = SkillWorker()
    blstats = make_test_blstats(experience=1, turn=10)

    # Normal turn, no message -> None
    assert worker.evaluate_skill_turn(blstats, message="You hit the jackal.") is None

    # Message notification -> Task("ENHANCE")
    task = worker.evaluate_skill_turn(blstats, message="You feel more confident in your weapon skills.")
    assert task is not None
    assert task.name == "ENHANCE"

    # Immediate next turn -> throttled
    blstats_next = make_test_blstats(experience=1, turn=11)
    assert worker.evaluate_skill_turn(blstats_next, message="") is None

    # Level up to XL 2 after 60 turns -> Task("ENHANCE")
    blstats_xl2 = make_test_blstats(experience=2, turn=75)
    task2 = worker.evaluate_skill_turn(blstats_xl2, message="")
    assert task2 is not None
    assert task2.name == "ENHANCE"


def test_tactical_combat_corridor_funneling():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=5, y=5)

    # Grid: Player at (5, 5) on room floor '.'
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord(".")
    chars[5, 4] = ord("#")  # Corridor to West (funnel target!)
    chars[5, 6] = ord(".")
    chars[4, 5] = ord(".")
    chars[6, 5] = ord(".")

    # Glyphs: Two adjacent jackals (North and East)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    # Set monster glyphs at (4, 5) and (5, 6)
    glyphs[4, 5] = nle.nethack.GLYPH_MON_OFF + 1  # jackal
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + 1  # jackal

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    # Swarm of 2 monsters in open room -> funnel step to West corridor (0, -1)
    assert task.name == "STEP"
    assert task.args["delta"] == (0, -1)


def test_tactical_combat_elbereth_defense_ward():
    combat_mgr = TacticalCombatManager()
    # Critical HP (3 / 16 <= 25%)
    blstats = make_test_blstats(hp=3, max_hp=16, x=5, y=5)

    chars = np.full((21, 79), ord("#"), dtype=np.uint8)  # In corridor
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    # Find actual jackal monster id
    jackal_id = next(m for m in range(nle.nethack.NUMMONS) if nle.nethack.permonst(m).mname == "jackal")
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + jackal_id  # Adjacent jackal

    # Turn 1: Emergency Elbereth Dust Engraving
    task1 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task1 is not None
    assert task1.name == "ENGRAVE_DUST"
    assert task1.args["text"] == "Elbereth"

    # Turn 2: Standing on Elbereth with low HP -> WAIT for 1 grace turn to let enemy flee, DO NOT ATTACK!
    task2 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task2 is not None
    assert task2.name == "WAIT"

    # Turn 3: Enemy did NOT flee and is still adjacent -> RETREAT (critical-HP universal
    # retreat: at 3/16 HP with no healing, never trade blows — escape step first).
    # NOTE: the corridor chars are all '#', so the hero is cornered in this fixture;
    # add an open tile east so a retreat step exists.
    chars[5, 4] = ord(".")
    task3 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task3 is not None
    assert task3.name == "STEP"

    # Cornered (all neighbors solid wall): melee is the last resort — never suicide wait
    chars[:, :] = ord("|")
    chars[5, 5] = ord(".")  # hero tile
    chars[5, 6] = ord("|")
    task4 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task4 is not None
    assert task4.name == "MELEE_ATTACK"


def test_tactical_combat_ranged_projectile_throwing():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=5, y=5)

    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    # Gas spore at (5, 8) - distance 3 East, straight cardinal line
    spore_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "gas spore" in pm.mname.lower():
            spore_id = m
            break
    assert spore_id is not None
    glyphs[5, 8] = nle.nethack.GLYPH_MON_OFF + spore_id

    # Mock inventory tracker carrying daggers
    inv = InventoryNormalizer()
    inv_strs = np.zeros((55, 80), dtype=np.uint8)
    inv_letters = np.zeros(55, dtype=np.uint8)
    inv_glyphs = np.zeros(55, dtype=np.int32)

    dagger_str = b"a - 5 daggers"
    inv_strs[0, :len(dagger_str)] = [c for c in dagger_str]
    inv_letters[0] = ord("a")
    inv_glyphs[0] = nle.nethack.GLYPH_OBJ_OFF + 1

    inv.synchronize(inv_strs, inv_letters, inv_glyphs, turn=10)

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=inv)
    assert task is not None
    assert task.name == "THROW"
    assert task.args["slot"] == "a"
    assert task.args["delta"] == (0, 1)  # East


def test_navigation_manager_door_lockpick_apply():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord("+")  # Closed door to East

    # Inventory with lock pick
    inv = InventoryNormalizer()
    inv_strs = np.zeros((55, 80), dtype=np.uint8)
    inv_letters = np.zeros(55, dtype=np.uint8)
    inv_glyphs = np.zeros(55, dtype=np.int32)

    pick_str = b"b - an uncursed lock pick"
    inv_strs[0, :len(pick_str)] = [c for c in pick_str]
    inv_letters[0] = ord("b")
    inv_glyphs[0] = nle.nethack.GLYPH_OBJ_OFF + 10
    inv.synchronize(inv_strs, inv_letters, inv_glyphs, turn=10)

    # Attempt 1: OPEN
    task1 = nav_mgr.evaluate_navigation_turn(chars, blstats, inv_tracker=inv)
    assert task1.name == "OPEN"

    # Attempt 2: APPLY lock pick (rather than kick!)
    task2 = nav_mgr.evaluate_navigation_turn(chars, blstats, inv_tracker=inv)
    assert task2.name == "APPLY"
    assert task2.args["slot"] == "b"
    assert task2.args["delta"] == (0, 1)


def test_tactical_combat_peaceful_entity_ignore():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=5, y=5)

    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)

    # Find pony / horse monster id
    horse_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "pony" in pm.mname.lower() or "horse" in pm.mname.lower():
            horse_id = m
            break
    assert horse_id is not None
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + horse_id

    # Horse is adjacent, but in PEACEFUL_NAMES -> scan_monsters ignores it
    monsters = combat_mgr.scan_monsters(glyphs, blstats)
    assert len(monsters) == 0

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is None  # Does not attack peaceful horse!


def test_tactical_combat_grid_bug_diagonal_tactics():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=5, y=5)

    # Grid bug monster ID
    grid_bug_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "grid bug" in pm.mname.lower():
            grid_bug_id = m
            break
    assert grid_bug_id is not None

    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)

    # Case 1: Grid bug is ORTHOGONAL at (5, 6) (East)
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + grid_bug_id
    task1 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task1 is not None
    # Hero steps to an adjacent tile (e.g. North (-1, 0) or South (1, 0)) to make grid bug diagonal
    assert task1.name == "STEP"
    sr, sc = task1.args["delta"]
    # Check that new pos is diagonal to grid bug at (5, 6)
    new_r, new_c = 5 + sr, 5 + sc
    assert abs(new_r - 5) == 1 and abs(new_c - 6) == 1

    # Case 2: Grid bug is DIAGONAL at (4, 6) (NE)
    glyphs.fill(nle.nethack.NO_GLYPH)
    glyphs[4, 6] = nle.nethack.GLYPH_MON_OFF + grid_bug_id
    task2 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task2 is not None
    # Hero strikes diagonally with zero retaliation risk!
    assert task2.name == "MELEE_ATTACK"
    assert task2.args["delta"] == (-1, 1)


def test_tactical_combat_elbereth_coordinate_invalidation():
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=4, max_hp=16, x=5, y=5, depth=1)

    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + 1  # Adjacent jackal

    # Turn 1: Engraves Elbereth at (1, 5, 5)
    task1 = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task1.name == "ENGRAVE_DUST"
    assert combat_mgr.elbereth_pos == (1, 5, 5)
    assert combat_mgr.elbereth_turns > 0

    # Turn 2: Hero moves to (5, 4) - Elbereth is invalidated because hero moved off the tile!
    blstats_moved = make_test_blstats(hp=4, max_hp=16, x=4, y=5, depth=1)
    combat_mgr.evaluate_combat_turn(glyphs, chars, blstats_moved)
    assert combat_mgr.elbereth_pos is None
    assert combat_mgr.elbereth_turns == 0


def test_micro_action_fiber_protocol_and_interruption():
    env = gym.make("NetHackChallenge-v0")
    env.reset()
    dispatcher = ActionDispatcher(env)

    # 1. Test fiber creation for single-step task
    fiber_step = dispatcher.create_fiber(Task("STEP", is_primitive=True, args={"delta": (0, 1)}))
    assert not fiber_step.is_finished
    assert len(fiber_step.action_sequence) == 1
    assert fiber_step.next_action() is not None
    assert fiber_step.is_finished

    # 2. Test fiber creation for multi-step task (QUAFF: QUAFF cmd + slot 'a')
    fiber_quaff = dispatcher.create_fiber(Task("QUAFF", is_primitive=True, args={"slot": "a"}))
    assert len(fiber_quaff.action_sequence) == 2
    assert fiber_quaff.cursor == 0
    act1 = fiber_quaff.next_action()
    assert act1 == dispatcher.name_to_index["QUAFF"]
    assert not fiber_quaff.is_finished
    act2 = fiber_quaff.next_action()
    assert act2 == dispatcher.char_to_index["a"]
    assert fiber_quaff.is_finished

    # 3. Test fiber interruption protocol
    fiber_engrave = dispatcher.create_fiber(Task("ENGRAVE_DUST", is_primitive=True, args={"text": "Elbereth"}))
    assert len(fiber_engrave.action_sequence) > 5
    fiber_engrave.interrupt()
    assert fiber_engrave.is_finished
    assert fiber_engrave.next_action() is None

    # 4. Test cooperative step_callback interrupt during execute_fiber
    fiber_multi = dispatcher.create_fiber(Task("QUAFF", is_primitive=True, args={"slot": "a"}))
    step_calls = 0

    def abort_callback(obs, r, term, trunc, info):
        nonlocal step_calls
        step_calls += 1
        return False  # Abort immediately after first micro-action

    dispatcher.execute_fiber(fiber_multi, step_callback=abort_callback)
    assert step_calls == 1
    assert fiber_multi.interrupted


def test_inventory_manager_shannon_safe_gate_veto():
    inv_mgr = InventoryManager()
    epistemic_mgr = EpistemicManager()
    blstats = make_test_blstats()
    inv = InventoryNormalizer()

    # Case 1: Untested weapon with UNKNOWN BUC (default 10% cursed prior)
    untested_sword = NormalizedItem(
        current_letter="a",
        raw_str="a broadsword",
        buc_state="UNKNOWN",
    )
    inv.active_items[untested_sword.uid] = untested_sword

    # With epistemic_mgr passed, ShannonSafeGate vetoes equipping untested weapon (P(Cursed) = 0.10 > 0.05)
    task_vetoed = inv_mgr.evaluate_resource_turn(blstats, inv, epistemic_mgr=epistemic_mgr)
    assert task_vetoed is None

    # Once confirmed uncursed (e.g. drop on altar), it is safe to equip!
    untested_sword.buc_state = "UNCURSED"
    belief = epistemic_mgr.get_or_create_belief(untested_sword)
    belief.collapse_buc("UNCURSED")
    task_safe = inv_mgr.evaluate_resource_turn(blstats, inv, epistemic_mgr=epistemic_mgr)
    assert task_safe is not None
    assert task_safe.name == "WIELD"
    assert task_safe.args["slot"] == "a"

    # Case 2: Emergency quaffing vetoes lethal potion candidates
    blstats_low_hp = make_test_blstats(hp=5, max_hp=16)
    mystery_potion = NormalizedItem(
        current_letter="b",
        raw_str="a potion of healing",
        buc_state="UNCURSED",
    )
    inv.active_items[mystery_potion.uid] = mystery_potion

    # Inject lethal candidate into belief state
    pot_belief = epistemic_mgr.get_or_create_belief(mystery_potion)
    pot_belief.candidate_identities = ["potion of healing", "potion of death"]
    pot_belief.identity_probs = np.array([0.5, 0.5])

    # ShannonSafeGate detects P(potion of death) = 0.50 > 0.01 and vetoes quaffing
    task_quaff = inv_mgr.evaluate_resource_turn(blstats_low_hp, inv, epistemic_mgr=epistemic_mgr)
    assert task_quaff is None or task_quaff.name != "QUAFF"


def test_inventory_readied_missile_no_quiver_loop():
    inv_mgr = InventoryManager()
    inv = InventoryNormalizer()
    blstats = make_test_blstats()

    # Darts marked '(at the ready)' should NOT trigger QUIVER task
    item_darts = NormalizedItem(
        current_letter="a",
        raw_str="34 +2 darts (at the ready)",
        buc_state="UNCURSED",
    )
    inv.active_items[item_darts.uid] = item_darts

    task = inv_mgr.evaluate_resource_turn(blstats, inv, role="rogue")
    assert task is None or task.name != "QUIVER"


def test_combat_readied_missile_firing():
    combat_mgr = TacticalCombatManager()
    inv = InventoryNormalizer()

    # Darts marked '(at the ready)' are recognized as quivered/ready
    item_darts = NormalizedItem(
        current_letter="a",
        raw_str="34 +2 darts (at the ready)",
        buc_state="UNCURSED",
    )
    inv.active_items[item_darts.uid] = item_darts

    slot, is_quivered = combat_mgr._find_quivered_or_missile(inv)
    assert slot == "a"
    assert is_quivered is True


def test_navigation_diagonal_doorway_orthogonal_decomposition():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[6, 6] = ord("+")  # Closed door at (6, 6)

    lvl = nav_mgr.update_map(chars, blstats)
    from corp.navigation.astar import PathNode
    # Hero is at (5, 5), next node in path is (6, 6) (diagonal into door)
    next_node = PathNode(row=6, col=6, action_index=0, key_char="")

    task = nav_mgr._step_or_open(5, 5, next_node, chars, lvl)
    assert task.name == "STEP"
    # Should decompose into orthogonal step (1, 0) or (0, 1), not diagonal (1, 1)
    assert task.args["delta"] in ((1, 0), (0, 1))


def test_navigation_diagonal_closed_door_alignment():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 4] = ord(".")
    chars[6, 4] = ord("+")  # Closed door diagonal at (6, 4)

    task = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task is not None
    # Hero should step orthogonally to align with the door
    assert task.name == "STEP"
    assert task.args["delta"] in ((0, -1), (1, 0))


def test_navigation_pet_swap_wait_interlock():
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # When message indicates pet swap, agent should issue WAIT to let pet move
    task = nav_mgr.evaluate_navigation_turn(
        chars,
        blstats,
        message="You swap places with your little dog.",
    )
    assert task is not None
    assert task.name == "WAIT"


def test_tactical_combat_killer_bee_lethal_poison_avoidance():
    """Verifies that un-resistant heroes do NOT melee killer bees, prioritizing Elbereth or retreat."""
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=16, x=5, y=5)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    # Killer bee is monster ID 1 in NLE
    glyphs[5, 6] = nle.nethack.GLYPH_MON_OFF + 1

    # Without poison resistance: must NOT close into melee!
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, has_poison_res=False)
    assert task is not None
    assert task.name != "MELEE_ATTACK"
    assert task.name in ("ENGRAVE_DUST", "STEP", "FIRE", "THROW", "ZAP_WAND", "WAIT")

    # With poison resistance: safe to attack in melee!
    combat_mgr.elbereth_pos = None
    combat_mgr.elbereth_turns = 0
    combat_mgr.elbereth_cooldown = 0
    task_res = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, has_poison_res=True)
    assert task_res is not None
    assert task_res.name == "MELEE_ATTACK"
    assert task_res.args["delta"] == (0, 1)


def test_navigation_diagonal_door_monster_avoidance():
    """Verifies that orthogonal doorway alignment avoids stepping into a monster."""
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 4] = ord(".")
    chars[6, 5] = ord(".")
    chars[6, 4] = ord("+")  # Closed door diagonal at (6, 4)

    glyphs = np.zeros((21, 79), dtype=np.int32)
    # Monster standing on (5, 4)
    glyphs[5, 4] = nle.nethack.GLYPH_MON_OFF + 12

    task = nav_mgr.evaluate_navigation_turn(chars, blstats, glyphs=glyphs)
    assert task is not None
    # Must NOT step into (5, 4) containing the monster! Should step into (6, 5) instead.
    assert task.args.get("delta") != (0, -1)


def test_agent_cycle_perturbation_monster_bump_guard():
    """Verifies that cycle perturbation never steps into an adjacent monster or pet."""
    from corp.agent.corp_agent import CORPAgent
    from corp.deliberative.providers.mock_provider import MockProvider
    from corp.env.nle_wrapper import make_env
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    # Place player at (10, 10) in an empty room
    agent.current_blstats = make_test_blstats(x=10, y=10)
    agent.current_chars = np.full((21, 79), ord("."), dtype=np.uint8)
    agent.current_glyphs = np.zeros((21, 79), dtype=np.int32)
    lvl_map = agent.nav_mgr.get_or_create_level(agent.current_blstats)
    lvl_map.walkable[:] = False

    # Surrounding player: 7 walls, 1 walkable tile containing a monster
    lvl_map.walkable[10, 11] = True
    agent.current_glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + 12  # Monster

    # Trigger cycle detector stall at (10, 10)
    agent.cycle_detector.pos_history = [(10, 10), (10, 10), (10, 10), (10, 10)]

    task = agent.select_action()
    # Must NOT step into (10, 11) to bump attack the monster! Should fall back to SEARCH
    assert task.name != "STEP"
    assert task.name in ("SEARCH", "WAIT")


def test_peaceful_prompt_extraction_and_unwalkable_mask():
    """Verifies that 'Really attack the hobbit?' extracts peaceful name and marks coordinate unwalkable."""
    from corp.agent.corp_agent import CORPAgent
    from corp.deliberative.providers.mock_provider import MockProvider
    from corp.env.nle_wrapper import make_env
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    agent.current_blstats = make_test_blstats(x=10, y=10)
    agent.current_chars = np.full((21, 79), ord("."), dtype=np.uint8)
    agent.current_glyphs = np.zeros((21, 79), dtype=np.int32)
    lvl_map = agent.nav_mgr.get_or_create_level(agent.current_blstats)
    lvl_map.walkable[:] = True

    # Monster at (10, 11): use a pony (a genuinely peaceful species) so the prompt-named
    # pacification applies. Previously ANY adjacent monster was position-poisoned here,
    # leaving hostile attackers permanently unattackable.
    pony_id = None
    for m in range(nle.nethack.NUMMONS):
        if "pony" in nle.nethack.permonst(m).mname.lower():
            pony_id = m
            break
    assert pony_id is not None
    agent.current_glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + pony_id
    agent.current_message = "Really attack the pony? [yn] (n)"

    agent.select_action()

    # Verify extracted into PEACEFUL_NAMES
    assert "pony" in agent.combat_mgr.PEACEFUL_NAMES
    # Verify coordinate recorded as peaceful
    assert (10, 11) in agent.combat_mgr.peaceful_positions
    # Verify coordinate marked unwalkable on the level map
    assert not lvl_map.walkable[10, 11]


def test_navigation_conveyor_corpse_pathing():
    """Verifies that hero without poison resistance prioritizes pathing to fresh conveyor corpses."""
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=5, y=5, depth=1, turn=100)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    lvl = nav_mgr.get_or_create_level(1)
    lvl.walkable[:] = True

    # Killer bee corpse at (5, 9) spawned on turn 95 (5 turns ago, fresh)
    lvl.conveyor_corpses[(5, 9)] = (95, "killer bee")

    # Hero without poison resistance should path towards the conveyor corpse!
    task = nav_mgr.evaluate_navigation_turn(chars, blstats, has_poison_res=False)
    assert task is not None
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)  # Steps East towards (5, 9)


def test_combat_manager_heavy_hitter_kiting():
    """Verifies that wounded hero kites/retreats from heavy hitters instead of trading fatal melee."""
    combat_mgr = TacticalCombatManager()
    # Wounded hero: 10/20 HP (50% HP)
    blstats = make_test_blstats(hp=10, max_hp=20, x=10, y=10)

    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 9] = ord(".")   # Safe retreat tile to West
    chars[10, 11] = ord("h")  # Gnome lord to East

    # Find gnome lord monster id
    gnome_lord_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "gnome lord" in pm.mname:
            gnome_lord_id = m
            break
    assert gnome_lord_id is not None
    glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + gnome_lord_id

    # Wounded hero should NOT melee attack the gnome lord!
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    assert task.name != "MELEE_ATTACK"
    assert task.name in ("ENGRAVE_DUST", "STEP", "WAIT", "FIRE", "THROW")


def test_floating_eye_active_attacker_prioritization():
    """Verifies that active attackers (like newts) are prioritized over adjacent floating eyes."""
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=16, max_hp=20, x=10, y=10)

    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # Eye at (10, 11) East, Newt at (10, 9) West
    eye_id, newt_id = None, None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "floating eye" in pm.mname:
            eye_id = m
        if "newt" in pm.mname:
            newt_id = m
    assert eye_id is not None and newt_id is not None

    glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + eye_id
    glyphs[10, 9] = nle.nethack.GLYPH_MON_OFF + newt_id

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    assert task is not None
    # Must attack the newt (West: delta (0, -1)), NOT wait for the eye!
    assert task.name == "MELEE_ATTACK"
    assert task.args["delta"] == (0, -1)


def test_tactical_healing_rest_suppressed_when_damaged():
    """Verifies that resting is suppressed if damage was recently taken."""
    combat_mgr = TacticalCombatManager()
    combat_mgr._prev_hp = 20
    # Took 5 damage this turn with no monsters in glyphs (e.g. unseen/dark attacker)
    blstats = make_test_blstats(hp=15, max_hp=20, x=10, y=10)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
    # Must NOT be WAIT (resting while attacked is fatal)
    assert task is None or task.name != "WAIT"


def test_domestic_animal_defense_when_attacked():
    """Verifies that if a dog attacks the hero, it is marked hostile and attacked."""
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=18, max_hp=20, x=10, y=10)
    glyphs = np.full((21, 79), nle.nethack.NO_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    dog_id = None
    for m in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(m)
        if "little dog" in pm.mname:
            dog_id = m
            break
    assert dog_id is not None
    glyphs[10, 11] = nle.nethack.GLYPH_MON_OFF + dog_id

    # The dog attacks the hero
    task = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats, message="The little dog bites!"
    )
    assert task is not None
    # Must defend and attack the dog!
    assert task.name == "MELEE_ATTACK"
    assert task.args["delta"] == (0, 1)










