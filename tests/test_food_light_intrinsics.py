"""
Unit tests for Food Clock Enhancements, Light Source Management,
Intrinsic Resistance Tracking, and Fragile Role Combat Adjustments.
"""

import pytest
import numpy as np
import nle

from corp.env.blstats import BottomLineStats, HungerState, ConditionFlag
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.domain.inventory_manager import InventoryManager
from corp.domain.navigation_manager import NavigationManager, LevelMap
from corp.domain.combat_manager import TacticalCombatManager, MonsterTrack
from corp.planner.htn import Task


def make_test_blstats(
    hp: int = 16,
    max_hp: int = 16,
    x: int = 10,
    y: int = 10,
    turn: int = 10,
    depth: int = 1,
    dnum: int = 0,
    hunger_state: int = HungerState.NORMAL,
    condition_mask: int = 0,
) -> BottomLineStats:
    raw = np.zeros(26, dtype=np.int32)
    raw[nle.nethack.NLE_BL_X] = x
    raw[nle.nethack.NLE_BL_Y] = y
    raw[nle.nethack.NLE_BL_HP] = hp
    raw[nle.nethack.NLE_BL_HPMAX] = max_hp
    raw[nle.nethack.NLE_BL_DEPTH] = depth
    raw[nle.nethack.NLE_BL_DNUM] = dnum
    raw[nle.nethack.NLE_BL_TIME] = turn
    raw[nle.nethack.NLE_BL_HUNGER] = hunger_state
    raw[nle.nethack.NLE_BL_CONDITION] = condition_mask
    return BottomLineStats.from_blstats(raw)


def test_food_clock_eat_corpse_when_normal():
    """Verify that fresh safe corpses on the floor are eaten when at NORMAL hunger (not satiated)."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hunger_state=HungerState.NORMAL)
    tracker = InventoryNormalizer()

    # Standing on a goblin corpse
    msg = "You see here a goblin corpse."
    task = inv_mgr.evaluate_resource_turn(blstats, tracker, message=msg)
    assert task is not None
    assert task.name == "EAT"
    assert task.args.get("slot") == ""  # Slot "" designates floor eating

    # When Satiated, agent must NOT eat
    blstats_satiated = make_test_blstats(hunger_state=HungerState.SATIATED)
    task_satiated = inv_mgr.evaluate_resource_turn(blstats_satiated, tracker, message=msg)
    assert task_satiated is None


def test_food_clock_poison_resistance_corpse():
    """Verify that poisonous corpses are avoided without resistance and consumed with resistance."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hunger_state=HungerState.NORMAL)
    tracker = InventoryNormalizer()

    # Standing on a poisonous snake corpse
    msg = "You see here a snake corpse."

    # Without poison resistance -> do not eat!
    task_unprotected = inv_mgr.evaluate_resource_turn(blstats, tracker, message=msg, has_poison_res=False, role="valkyrie")
    assert task_unprotected is None

    # With poison resistance -> safe to eat!
    task_protected = inv_mgr.evaluate_resource_turn(blstats, tracker, message=msg, has_poison_res=True, role="valkyrie")
    assert task_protected is not None
    assert task_protected.name == "EAT"


def test_navigation_corpse_tracking_and_rot_pruning():
    """Verify that NavigationManager tracks % food items and prunes them after 25 turns."""
    nav_mgr = NavigationManager()
    blstats = make_test_blstats(x=10, y=10, turn=50)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 12] = ord("%")  # Food 2 steps East

    lvl = nav_mgr.update_map(chars, blstats)
    assert (10, 12) in lvl.floor_corpses
    assert lvl.floor_corpses[(10, 12)] == 50

    # Advance 10 turns (turn 60) -> still fresh
    blstats_turn60 = make_test_blstats(x=10, y=10, turn=60)
    lvl = nav_mgr.update_map(chars, blstats_turn60)
    assert (10, 12) in lvl.floor_corpses

    # Advance 30 turns (turn 80) -> rotten (> 25 turns old), must be pruned!
    blstats_turn80 = make_test_blstats(x=10, y=10, turn=80)
    lvl = nav_mgr.update_map(chars, blstats_turn80)
    assert (10, 12) not in lvl.floor_corpses


def test_light_source_application():
    """Verify that unlit lamps are applied on deep levels or in the Gnomish Mines."""
    inv_mgr = InventoryManager()
    # Depth 3
    blstats_d3 = make_test_blstats(depth=3)
    tracker = InventoryNormalizer()
    lamp = NormalizedItem(
        uid="lamp1",
        raw_str="an uncursed oil lamp",
        current_letter="a",
        oclass=nle.nethack.TOOL_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["lamp1"] = lamp

    task = inv_mgr.evaluate_resource_turn(blstats_d3, tracker)
    assert task is not None
    assert task.name == "APPLY"
    assert task.args.get("slot") == "a"

    # When already lit, must NOT re-apply
    lamp_lit = NormalizedItem(
        uid="lamp1",
        raw_str="an uncursed oil lamp (lit)",
        current_letter="a",
        oclass=nle.nethack.TOOL_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["lamp1"] = lamp_lit
    task_lit = inv_mgr.evaluate_resource_turn(blstats_d3, tracker)
    assert task_lit is None


def test_light_source_out_of_oil_lockout():
    """Verify that out-of-power/oil lamps are not repeatedly applied."""
    inv_mgr = InventoryManager()
    blstats_d3 = make_test_blstats(depth=3)
    tracker = InventoryNormalizer()
    lamp = NormalizedItem(
        uid="lamp1",
        raw_str="an uncursed oil lamp",
        current_letter="a",
        oclass=nle.nethack.TOOL_CLASS,
        buc_state="UNCURSED",
    )
    tracker.active_items["lamp1"] = lamp

    # Initial apply
    task = inv_mgr.evaluate_resource_turn(blstats_d3, tracker)
    assert task.name == "APPLY"

    # Message indicates lamp is out of power
    task_next = inv_mgr.evaluate_resource_turn(blstats_d3, tracker, message="This lamp has run out of power.")
    assert "a" in inv_mgr.failed_light_slots
    assert task_next is None


def test_fragile_role_elbereth_threshold():
    """Verify that fragile roles engrave Elbereth at higher HP threshold (75% HP)."""
    combat_mgr = TacticalCombatManager()
    # Wizard with 11/15 HP (73% HP) adjacent to hostile goblin
    blstats = make_test_blstats(hp=11, max_hp=15, x=10, y=10)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 11] = ord("g")  # adjacent goblin

    glyphs = np.full((21, 79), 2359, dtype=np.int16)  # 2359 is floor glyph
    glyphs[10, 11] = 69  # 69 is goblin glyph

    # Valkyrie at 11/15 (73% > 60%) will melee attack
    task_valk = combat_mgr.evaluate_combat_turn(
        chars=chars,
        glyphs=glyphs,
        blstats=blstats,
        role="valkyrie",
    )
    assert task_valk.name == "MELEE_ATTACK"

    # Wizard at 11/15 (73% <= 75%) will trigger Elbereth to protect low HP pool
    task_wiz = combat_mgr.evaluate_combat_turn(
        chars=chars,
        glyphs=glyphs,
        blstats=blstats,
        role="wizard",
    )
    assert task_wiz.name == "ENGRAVE_DUST"


def test_corp_agent_intrinsic_tracking():
    """Verify that CORPAgent dynamically updates intrinsics from message history."""
    from corp.env.nle_wrapper import make_env
    from corp.agent.corp_agent import CORPAgent
    from corp.deliberative.providers.mock_provider import MockProvider

    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    # Initially empty or starting role intrinsic
    starting_intrinsics = set(agent.intrinsics)

    # Simulate eating a poisonous corpse and gaining poison resistance
    agent.current_message = "You feel healthy."
    # Simulate select_action parsing
    agent.select_action()
    assert "poison_res" in agent.intrinsics

    # Simulate gaining shock resistance
    agent.current_message = "Your health currently feels amplified!"
    agent.select_action()
    assert "shock_res" in agent.intrinsics

    # Simulate gaining telepathy
    agent.current_message = "You feel totally together, man."
    agent.select_action()
    assert "telepathy" in agent.intrinsics


def test_safe_poison_res_conveyor_eating():
    """Verify that safe non-poisonous corpses (centipede, cave spider, shrieker) are eaten even without poison resistance."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hunger_state=HungerState.NORMAL)
    tracker = InventoryNormalizer()

    # Standing on a fresh centipede corpse
    msg_centipede = "You see here a centipede corpse."
    task = inv_mgr.evaluate_resource_turn(blstats, tracker, message=msg_centipede, has_poison_res=False, role="valkyrie")
    assert task is not None
    assert task.name == "EAT"

    # Standing on a fresh shrieker corpse
    msg_shrieker = "You see here a shrieker corpse."
    task_shrieker = inv_mgr.evaluate_resource_turn(blstats, tracker, message=msg_shrieker, has_poison_res=False, role="valkyrie")
    assert task_shrieker is not None
    assert task_shrieker.name == "EAT"


def test_poisonous_conveyor_eating_when_healthy():
    """Verify that poisonous conveyors (killer bee, soldier ant) are eaten when healthy and skipped when wounded."""
    inv_mgr = InventoryManager()
    tracker = InventoryNormalizer()
    msg_bee = "You see here a killer bee corpse."

    # Healthy hero (20/20 HP) without poison resistance -> eat killer bee to gain intrinsic!
    blstats_healthy = make_test_blstats(hp=20, max_hp=20, hunger_state=HungerState.NORMAL)
    task_eat = inv_mgr.evaluate_resource_turn(blstats_healthy, tracker, message=msg_bee, has_poison_res=False, role="valkyrie")
    assert task_eat is not None
    assert task_eat.name == "EAT"

    # Wounded hero (10/20 HP) -> avoid poisonous corpse to prevent lethal damage!
    blstats_low = make_test_blstats(hp=10, max_hp=20, hunger_state=HungerState.NORMAL)
    task_skip = inv_mgr.evaluate_resource_turn(blstats_low, tracker, message=msg_bee, has_poison_res=False, role="valkyrie")
    assert task_skip is None


def test_giant_spider_kiting_without_poison_res():
    """Verify that giant spiders are kited or warded with Elbereth without poison res, and attacked with poison res."""
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=25, max_hp=25, x=10, y=10)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 11] = ord("s")  # adjacent giant spider
    glyphs = np.full((21, 79), 2359, dtype=np.int16)
    # Glyph for giant spider
    spider_mon_id = nle.nethack.glyph_to_mon(2359)
    # Mock scan_monsters
    spider = MonsterTrack(
        pos=(10, 11), name="giant spider", level=5, speed=12, ac=4,
        distance=1, is_adjacent=True, is_instakill=True, threat_score=350.0
    )
    combat_mgr.scan_monsters = lambda g, b, has_poison_res=False: [spider]

    # Without poison resistance -> do NOT melee attack! Engrave Elbereth!
    task_ward = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats, role="samurai", has_poison_res=False
    )
    assert task_ward is not None
    assert task_ward.name == "ENGRAVE_DUST"

    # With poison resistance -> safe to melee attack!
    task_melee = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats, role="samurai", has_poison_res=True
    )
    assert task_melee is not None
    assert task_melee.name == "MELEE_ATTACK"


def test_wand_wielder_cover_evasion():
    """Verify that a hero ducks into adjacent cover when facing a wand-wielder in straight line of fire."""
    combat_mgr = TacticalCombatManager()
    blstats = make_test_blstats(hp=25, max_hp=25, x=10, y=10)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 11] = ord(".")
    chars[10, 12] = ord(".")
    chars[10, 13] = ord(".")  # Gnome lord at (10, 13), East distance 3
    # Put a corridor step to the North at (9, 10)
    chars[9, 10] = ord("#")

    glyphs = np.full((21, 79), 2359, dtype=np.int16)
    gnome_lord = MonsterTrack(
        pos=(10, 13), name="gnome lord", level=3, speed=8, ac=6,
        distance=3, is_adjacent=False, is_instakill=False, threat_score=15.0
    )
    combat_mgr.scan_monsters = lambda g, b, has_poison_res=False: [gnome_lord]

    # Without reflection -> duck into cover at (9, 10) (delta (-1, 0))!
    task = combat_mgr.evaluate_combat_turn(
        glyphs, chars, blstats, role="valkyrie", has_reflection=False
    )
    assert task is not None
    assert task.name == "STEP"
    assert task.args["delta"] == (-1, 0)


def test_armor_takeoff_and_upgrade():
    """Verify that an equipped inferior armor is taken off when a strictly superior upgrade is found."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hp=20, max_hp=20)
    tracker = InventoryNormalizer()

    # Hero is wearing leather armor (rank 3)
    worn_leather = NormalizedItem(
        uid="a1", raw_str="an uncursed leather armor (being worn)", current_letter="a",
        oclass=nle.nethack.ARMOR_CLASS, buc_state="UNCURSED", equipped=True
    )
    # Hero has dwarvish mithril-coat (rank 9) in inventory
    better_mithril = NormalizedItem(
        uid="a2", raw_str="an uncursed dwarvish mithril-coat", current_letter="b",
        oclass=nle.nethack.ARMOR_CLASS, buc_state="UNCURSED", equipped=False
    )
    tracker.active_items["a1"] = worn_leather
    tracker.active_items["a2"] = better_mithril

    # Should issue TAKEOFF for leather armor ('a')
    task = inv_mgr.evaluate_resource_turn(blstats, tracker, role="valkyrie")
    assert task is not None
    assert task.name == "TAKEOFF"
    assert task.args["slot"] == "a"


def test_twoweapon_shield_avoidance():
    """Verify that Samurai and Barbarian roles do not equip shields to protect #twoweapon."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(hp=20, max_hp=20)
    tracker = InventoryNormalizer()

    # Hero has a shield in inventory
    shield = NormalizedItem(
        uid="s1", raw_str="an uncursed large shield", current_letter="s",
        oclass=nle.nethack.ARMOR_CLASS, buc_state="UNCURSED", equipped=False
    )
    tracker.active_items["s1"] = shield

    # Valkyrie CAN wear shield
    task_valk = inv_mgr.evaluate_resource_turn(blstats, tracker, role="valkyrie")
    assert task_valk is not None
    assert task_valk.name == "WEAR"
    assert task_valk.args["slot"] == "s"

    # Samurai will NOT wear shield
    task_samu = inv_mgr.evaluate_resource_turn(blstats, tracker, role="samurai")
    assert task_samu is None

    # Barbarian will NOT wear shield
    task_barb = inv_mgr.evaluate_resource_turn(blstats, tracker, role="barbarian")
    assert task_barb is None


def test_sokoban_branch_navigation():
    """Verify that in Sokoban (dnum=3), navigation ascends '<' on floors 1-3."""
    nav_mgr = NavigationManager()
    blstats_soko1 = make_test_blstats(x=10, y=10, dnum=3)
    blstats_soko1 = BottomLineStats(
        x=10, y=10, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=20, max_hp=20,
        depth=1, gold=0, energy=10, max_energy=10, ac=10, monster_level=1,
        experience=5, turn=100, hunger_state=0, encumbrance=0,
        dungeon_number=3, level_number=1, condition_bits=0, alignment=0,
    )
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 10] = ord("@")
    chars[10, 11] = ord("<")  # Stairs up at (10, 11)

    lvl = nav_mgr.update_map(chars, blstats_soko1)
    task = nav_mgr.evaluate_navigation_turn(chars, blstats_soko1)
    assert task is not None
    # Step towards stairs up
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)

    # Standing on stairs up -> ASCEND
    blstats_on_stairs = BottomLineStats(
        x=11, y=10, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=20, max_hp=20,
        depth=1, gold=0, energy=10, max_energy=10, ac=10, monster_level=1,
        experience=5, turn=101, hunger_state=0, encumbrance=0,
        dungeon_number=3, level_number=1, condition_bits=0, alignment=0,
    )
    task_ascend = nav_mgr.evaluate_navigation_turn(chars, blstats_on_stairs)
    assert task_ascend is not None
    assert task_ascend.name == "ASCEND"


def test_starvation_prayer_triggers_when_weak_and_foodless():
    """Verify that when starving (WEAK) and carrying no food, emergency prayer triggers if safe."""
    inv_mgr = InventoryManager()
    blstats = make_test_blstats(turn=350, hunger_state=HungerState.WEAK, hp=20, max_hp=20)
    tracker = InventoryNormalizer()  # No food items

    task = inv_mgr.evaluate_resource_turn(blstats, tracker, role="valkyrie")
    assert task is not None
    assert task.name == "PRAY"
    assert inv_mgr.prayer_state.last_prayer_turn == 350


def test_search_burst_aborts_on_damage():
    """Verify that taking damage or receiving hit messages immediately aborts an ongoing search burst."""
    nav_mgr = NavigationManager()
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    blstats = make_test_blstats(turn=100, hp=20, max_hp=20, x=5, y=5)

    # Simulate ongoing search burst at (5, 5)
    nav_mgr._current_search_spot = (5, 5)
    nav_mgr._current_search_burst = 3
    nav_mgr._prev_hp = 20

    # Hero takes damage (HP drops 20 -> 16) with attack message
    blstats_damaged = make_test_blstats(turn=101, hp=16, max_hp=20, x=5, y=5)
    task = nav_mgr.evaluate_navigation_turn(chars, blstats_damaged, message="The hobbit hits!")

    # Search burst must be cancelled
    assert nav_mgr._current_search_spot is None
    assert nav_mgr._current_search_burst == 0


def test_diagonal_closed_door_skips_blocked_or_shop_doors():
    """Verify that a diagonally adjacent door is NOT aligned to if it is already in blocked_tiles or shop_doors."""
    nav_mgr = NavigationManager()
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 10] = ord("@")  # Player at (10, 10)
    chars[9, 10] = ord(".")   # North floor
    chars[10, 9] = ord(".")   # West floor
    chars[9, 11] = ord("+")   # Diagonal closed door at NE (9, 11)

    blstats = make_test_blstats(x=10, y=10)
    lvl = nav_mgr.update_map(chars, blstats)
    lvl.walkable[9, 10] = True
    lvl.walkable[10, 9] = True
    lvl.visited[9, 10] = 5  # North has 5 visits
    lvl.visited[10, 9] = 0  # West has 0 visits

    # CASE 1: When diagonal door is NOT blocked, diagonal alignment overrides visit counts and steps North
    task_unblocked = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task_unblocked.name == "STEP"
    assert task_unblocked.args.get("delta") == (-1, 0)

    # CASE 2: When diagonal door is blocked, diagonal alignment is SKIPPED, so fallback picks West (0 visits)
    lvl.blocked_tiles.add((9, 11))
    task_blocked = nav_mgr.evaluate_navigation_turn(chars, blstats)
    assert task_blocked.name == "STEP"
    assert task_blocked.args.get("delta") == (0, -1)


def test_hallucination_does_not_corrupt_walkable_map():
    """Verify that when hallucinating, random glyphs do not corrupt lvl.walkable."""
    nav_mgr = NavigationManager()
    # Chars contains a fake wall disguised as gold '$'
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[5, 5] = ord("$")

    # Hallucinating bit set
    blstats = make_test_blstats(x=10, y=10, condition_mask=ConditionFlag.HALLU)
    lvl = nav_mgr.update_map(chars, blstats)

    # (5, 5) must NOT be marked walkable because hero is hallucinating!
    assert not lvl.walkable[5, 5]



