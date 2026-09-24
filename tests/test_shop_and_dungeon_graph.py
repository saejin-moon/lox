"""
Tests for Economy Subsystem (ShopManager), Macro Dungeon Graph (DungeonGraph),
Plan Locking (LockedIntent), and Specialized Pioneer Episode ID Formatting.
"""

import numpy as np
import pytest

from lox.env.blstats import BottomLineStats, HungerState
from lox.domain.shop_manager import ShopManager, SCROLL_PRICE_TIERS, POTION_PRICE_TIERS, WAND_PRICE_TIERS
from lox.domain.dungeon_graph import DungeonGraph
from lox.domain.navigation_manager import NavigationManager
from lox.agent.lox_agent import LoxAgent, LockedIntent
from lox.planner.htn import Task
from lox.planner.guards import HTNGuards
from lox.env.nle_wrapper import make_env


def test_shop_manager_price_inversion():
    """Verifies that ShopManager accurately maps shopkeeper sell offers to NetHack 3.6.6 base price identities."""
    shop_mgr = ShopManager()

    # 1. 10zm offer on scroll -> uniquely identifies Scroll of Identify
    cands_identify = shop_mgr.resolve_scroll_price(10)
    assert cands_identify == ["scroll of identify"]

    # Greedy shopkeeper (offers 7-8zm) -> still maps to Scroll of Identify
    cands_greedy_id = shop_mgr.resolve_scroll_price(8)
    assert "scroll of identify" in cands_greedy_id

    # 2. 25zm offer on scroll -> Scroll of Light
    cands_light = shop_mgr.resolve_scroll_price(25)
    assert cands_light == ["scroll of light"]

    # 3. 40zm offer on scroll -> Enchant Weapon / Remove Curse
    cands_80 = shop_mgr.resolve_scroll_price(40)
    assert "scroll of enchant weapon" in cands_80
    assert "scroll of remove curse" in cands_80

    # 4. 150zm offer on scroll -> Genocide / Charging
    cands_300 = shop_mgr.resolve_scroll_price(150)
    assert "scroll of genocide" in cands_300
    assert "scroll of charging" in cands_300

    # 5. Potions: 50zm offer -> Extra Healing / Cure Blindness
    cands_pot_100 = shop_mgr.resolve_potion_price(50)
    assert "potion of extra healing" in cands_pot_100

    # 6. Wands: 250zm offer -> Wand of Wishing / Wand of Death
    cands_wand_500 = shop_mgr.resolve_wand_price(250)
    assert "wand of wishing" in cands_wand_500


def make_blstats(
    x=10, y=10, hp=20, max_hp=20, depth=1, gold=0,
    experience=1, turn=10, hunger_state=0, dungeon_number=0, level_number=1,
    ac=10, condition_bits=0, alignment=0
) -> BottomLineStats:
    return BottomLineStats(
        x=x, y=y, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=hp, max_hp=max_hp,
        depth=depth, gold=gold, energy=10, max_energy=10, ac=ac, monster_level=1,
        experience=experience, turn=turn, hunger_state=hunger_state, encumbrance=0,
        dungeon_number=dungeon_number, level_number=level_number,
        condition_bits=condition_bits, alignment=alignment,
    )


def test_shop_manager_temple_priest_donation():
    """Verifies that ShopManager calculates correct 400 * XL donation to temple priests."""
    shop_mgr = ShopManager()
    blstats = make_blstats(x=10, y=10, hp=30, max_hp=30, experience=3, gold=1500, turn=200)
    chars = np.full((21, 79), ord("."))
    # Place priest at (10, 11)
    chars[10, 11] = ord("@")

    # Update perception with temple shrine message
    shop_mgr.update_perception(blstats, message="Welcome to the shrine of Tyr!", chars=chars)
    assert shop_mgr.priest_pos == (10, 11)

    # Evaluate donation: XL=3 requires 1200 gold
    task = shop_mgr.evaluate_temple_donation(blstats, chars, has_adjacent_hostiles=False)
    assert task is not None
    assert task.name == "CHAT"
    assert task.args["delta"] == (0, 1)
    assert task.args["amount"] == 1200

    # If hero has insufficient gold (< 1200), no donation task
    poor_blstats = make_blstats(x=10, y=10, hp=30, max_hp=30, experience=3, gold=500, turn=201)
    assert shop_mgr.evaluate_temple_donation(poor_blstats, chars, has_adjacent_hostiles=False) is None


def test_dungeon_graph_mines_and_sokoban_policies():
    """Verifies DungeonGraph topology tracking and Gnomish Mines entry/retreat policies."""
    graph = DungeonGraph()

    # Turn on Dungeons of Doom DL 1
    bl_dod = make_blstats(x=10, y=10, hp=15, max_hp=15, experience=1, dungeon_number=0, level_number=1)
    node = graph.record_step(bl_dod)
    assert node.dnum == 0
    assert node.dlevel == 1
    assert node.branch_name == "Dungeons of Doom"

    # Under-leveled hero (XL 1, no light) should NOT enter or descend deep into Gnomish Mines
    bl_mines_low = make_blstats(x=10, y=10, hp=15, max_hp=15, experience=1, dungeon_number=2, level_number=1)
    assert not graph.should_enter_mines(bl_mines_low, has_light=False, has_infravision=False)
    assert graph.should_ascend_from_mines(bl_mines_low, has_light=False, has_infravision=False)

    # Level 5 hero CAN enter Gnomish Mines
    bl_mines_high = make_blstats(x=10, y=10, hp=45, max_hp=45, experience=5, dungeon_number=2, level_number=1)
    assert graph.should_enter_mines(bl_mines_high, has_light=False, has_infravision=False)
    assert not graph.should_ascend_from_mines(bl_mines_high, has_light=False, has_infravision=False)

    # Sokoban detection
    bl_soko = make_blstats(x=10, y=10, hp=40, max_hp=40, experience=6, dungeon_number=3, level_number=1)
    assert graph.is_in_sokoban(bl_soko)


def test_htn_guards_mines_descent_interlock():
    """Verifies HTNGuards prevents diving deeper into Gnomish Mines if under-leveled."""
    bl_mines_low = make_blstats(
        x=10, y=10, hp=15, max_hp=15, experience=2, dungeon_number=2, level_number=1, depth=3
    )
    # can_descend must be False for low-level hero in Mines
    assert not HTNGuards.can_descend(bl_mines_low, has_poison_res=False)

    # If hero is XL >= 5, descent is permitted
    bl_mines_ready = make_blstats(
        x=10, y=10, hp=35, max_hp=35, experience=5, dungeon_number=2, level_number=1, depth=3
    )
    assert HTNGuards.can_descend(bl_mines_ready, has_poison_res=False)


def test_navigation_excalibur_fountain_dip():
    """Verifies that NavigationManager issues Task('DIP') when standing on a fountain with long sword."""
    nav = NavigationManager()
    chars = np.full((21, 79), ord("."))
    chars[10, 10] = ord("{")  # Fountain
    blstats = make_blstats(x=10, y=10, hp=40, max_hp=40, experience=5, turn=100)

    task = nav.evaluate_navigation_turn(
        chars=chars,
        blstats=blstats,
        target_fountain=True,
        long_sword_slot="a",
    )
    assert task is not None
    assert task.name == "DIP"
    assert task.args["slot"] == "a"


def test_locked_intent_door_kicking_and_emergency_break():
    """Verifies that LockedIntent executes door kicks committedly but breaks on emergency health drop."""
    env = make_env()
    agent = LoxAgent(env=env)
    obs, info = agent.reset()

    # Simulate locked door at (10, 11) with player at (10, 10)
    agent.current_blstats = make_blstats(
        x=10, y=10, hp=20, max_hp=20, experience=2, turn=10, depth=1
    )
    agent.current_chars = np.full((21, 79), ord("."))
    agent.current_chars[10, 11] = ord("+")  # Closed door
    agent.current_glyphs = np.zeros((21, 79), dtype=np.int32)
    agent.current_message = "The door is locked."

    # Set locked door kick intent
    agent.locked_intent = LockedIntent(
        name="DOOR_KICK",
        target_pos=(10, 11),
        remaining_steps=10,
    )

    # 1. Normal turn: executes KICK
    task1 = agent.select_action()
    assert task1.name == "KICK"
    assert task1.args["delta"] == (0, 1)
    assert agent.locked_intent.remaining_steps == 9

    # 2. Emergency turn: HP drops to 5 (<= 55% of max 20)
    agent.current_blstats = make_blstats(
        x=10, y=10, hp=5, max_hp=20, experience=2, turn=11, depth=1
    )
    task2 = agent.select_action()
    # Locked intent MUST be released immediately
    assert agent.locked_intent is None

    env.close()


def test_specialized_role_episode_id_prefix():
    """Verifies that specialized pioneer roles produce formatted episode IDs with role tags."""
    env = make_env()

    # Valkyrie agent
    agent_valk = LoxAgent(env=env, role="valkyrie")
    assert agent_valk.episode_id.startswith("ep_valk_")

    # Barbarian agent
    agent_barb = LoxAgent(env=env, role="barbarian")
    assert agent_barb.episode_id.startswith("ep_barb_")

    # Samurai agent
    agent_samu = LoxAgent(env=env, role="samurai")
    assert agent_samu.episode_id.startswith("ep_samu_")

    # Generalist agent (default)
    agent_gen = LoxAgent(env=env)
    assert agent_gen.episode_id.startswith("ep_gen_")

    env.close()


def test_shop_door_protection_and_unlock_tools():
    """Verifies that shop doors are never kicked and unlocking tools are applied first."""
    from lox.domain.inventory_manager import InventoryManager
    from lox.navigation.astar import PathNode

    # 1. Test get_unlock_tool
    inv_mgr = InventoryManager()
    class DummyTracker:
        pass
    class DummyItem:
        def __init__(self, letter, raw_str):
            self.current_letter = letter
            self.raw_str = raw_str

    tracker = DummyTracker()
    tracker.active_items = {
        "a": DummyItem("a", "a +1 long sword (weapon in hands)"),
        "b": DummyItem("b", "a skeleton key"),
        "c": DummyItem("c", "an uncursed food ration"),
    }
    tool_slot = inv_mgr.get_unlock_tool(tracker)
    assert tool_slot == "b"

    # 2. Test NavigationManager door unlocking with key
    nav = NavigationManager()
    nav.unlock_tool_slot = "b"
    lvl = nav.get_or_create_level(1)
    chars = np.full((21, 79), ord("."))
    chars[10, 11] = ord("+")
    next_node = PathNode(10, 11, 1, "l")

    # Attempt 0 -> OPEN
    task0 = nav._step_or_open(10, 10, next_node, chars, lvl)
    assert task0.name == "OPEN"

    # Attempt 1 -> APPLY skeleton key
    task1 = nav._step_or_open(10, 10, next_node, chars, lvl)
    assert task1.name == "APPLY"
    assert task1.args["slot"] == "b"
    assert task1.args["delta"] == (0, 1)

    # Attempt 2 -> OPEN
    task2 = nav._step_or_open(10, 10, next_node, chars, lvl)
    assert task2.name == "OPEN"

    # 3. Test Shop Door Interlock: if door is in lvl.shop_doors, NEVER KICK!
    lvl.shop_doors.add((10, 11))
    task3 = nav._step_or_open(10, 10, next_node, chars, lvl)
    # Must NOT be KICK! Must return SEARCH and mark unwalkable
    assert task3.name == "SEARCH"
    assert not lvl.walkable[10, 11]


def test_corpse_freshness_and_rotten_rejection():
    """Verifies that InventoryManager eats fresh corpses and lichens, but rejects rotten or combat-threatened corpses."""
    from lox.domain.inventory_manager import InventoryManager
    from lox.domain.navigation_manager import LevelMap
    inv_mgr = InventoryManager()

    class DummyTracker:
        active_items = {}

    tracker = DummyTracker()
    lvl = LevelMap()

    # Case 1: Fresh corpse on floor (spawned at turn 100, current turn 110 <= 25 turns)
    lvl.floor_corpses[(10, 10)] = 100
    blstats_fresh = make_blstats(x=10, y=10, hp=20, max_hp=20, turn=110, hunger_state=1)
    task_fresh = inv_mgr.evaluate_resource_turn(
        blstats_fresh, tracker, message="There is a newt corpse here.", lvl_map=lvl, has_adjacent_hostiles=False
    )
    assert task_fresh is not None
    assert task_fresh.name == "EAT"

    # Case 2: Stale corpse on floor (spawned at turn 100, current turn 130 > 25 turns)
    blstats_stale = make_blstats(x=10, y=10, hp=20, max_hp=20, turn=130, hunger_state=1)
    task_stale = inv_mgr.evaluate_resource_turn(
        blstats_stale, tracker, message="There is a newt corpse here.", lvl_map=lvl, has_adjacent_hostiles=False
    )
    # Must REFUSE to eat!
    assert task_stale is None

    # Case 3: Lichen corpse on floor (never rots, even if 500 turns old!)
    blstats_lichen = make_blstats(x=10, y=10, hp=20, max_hp=20, turn=600, hunger_state=1)
    task_lichen = inv_mgr.evaluate_resource_turn(
        blstats_lichen, tracker, message="There is a lichen corpse here.", lvl_map=lvl, has_adjacent_hostiles=False
    )
    assert task_lichen is not None
    assert task_lichen.name == "EAT"

    # Case 4: Adjacent hostile is attacking: NEVER eat multi-turn floor corpse during active combat!
    task_combat = inv_mgr.evaluate_resource_turn(
        blstats_fresh, tracker, message="There is a newt corpse here.", lvl_map=lvl, has_adjacent_hostiles=True
    )
    assert task_combat is None


def test_navigation_domestic_animal_and_peaceful_avoidance():
    """Verifies that NavigationManager does not bump-attack non-pet monsters like ponies, but swaps with pets."""
    from nle import nethack
    from lox.navigation.astar import PathNode
    nav = NavigationManager()
    lvl = nav.get_or_create_level(1)
    chars = np.full((21, 79), ord("."))
    glyphs = np.zeros((21, 79), dtype=np.int32)
    next_node = PathNode(10, 11, 1, "l")

    # 1. Non-pet monster (e.g. wild pony or domestic dog glyph)
    glyphs[10, 11] = nethack.GLYPH_MON_OFF + 50  # Hostile/neutral monster
    task_mon = nav._step_or_open(10, 10, next_node, chars, lvl, glyphs=glyphs)
    # Must WAIT, not STEP/attack!
    assert task_mon.name == "WAIT"

    # 2. Pet monster (e.g. starting kitten/little dog)
    glyphs[10, 11] = nethack.GLYPH_PET_OFF + 50
    task_pet = nav._step_or_open(10, 10, next_node, chars, lvl, glyphs=glyphs)
    # Pets can be displaced safely
    assert task_pet.name == "STEP"
    assert task_pet.args["delta"] == (0, 1)


def test_multiple_stairs_down_branch_steering():
    """Verifies that update_map steers away from Gnomish Mines when under-leveled."""
    from lox.domain.dungeon_graph import DungeonGraph
    graph = DungeonGraph()
    # Record Dod level 2 node
    node = graph.get_or_create_node(0, 2)
    # Stair at (5, 5) leads to Mines (2, 1); stair at (8, 8) leads to DoD Level 3 (0, 3)
    node.stair_connections[(5, 5)] = (2, 1)
    node.stair_connections[(8, 8)] = (0, 3)

    nav = NavigationManager()
    chars = np.full((21, 79), ord("."))
    chars[5, 5] = ord(">")
    chars[8, 8] = ord(">")

    # Under-leveled hero (XL 1, no light)
    bl_low = make_blstats(x=10, y=10, experience=1, dungeon_number=0, level_number=2)
    lvl_low = nav.update_map(chars, bl_low, dungeon_graph=graph)
    # Must avoid Mines staircase at (5, 5) and choose (8, 8)
    assert lvl_low.stairs_down == (8, 8)
    assert (5, 5) in lvl_low.all_stairs_down
    assert (8, 8) in lvl_low.all_stairs_down


def test_active_shop_purchasing_and_unpaid_debt_drop():
    """Verifies that ShopManager issues PAY when gold > 0, and DROP when broke."""
    from lox.domain.shop_manager import ShopManager
    shop_mgr = ShopManager()

    class MockItem:
        def __init__(self, letter, raw_str):
            self.current_letter = letter
            self.raw_str = raw_str

    class MockTracker:
        def __init__(self, items):
            self.active_items = {it.current_letter: it for it in items}

    # Case 1: Hero has 100 gold and carries unpaid food ration -> PAY
    bl_rich = make_blstats(x=5, y=5, gold=100)
    tracker_unpaid = MockTracker([MockItem("b", "an uncursed food ration (unpaid, 40 zorkmids)")])
    assert shop_mgr.has_unpaid_items(tracker_unpaid) is True
    task_pay = shop_mgr.evaluate_shop_payment(bl_rich, tracker_unpaid, "Will you please pay for that?")
    assert task_pay is not None
    assert task_pay.name == "PAY"

    # Case 2: Hero has 0 gold and carries unpaid wand -> DROP to avoid wrath
    bl_broke = make_blstats(x=5, y=5, gold=0)
    task_drop = shop_mgr.evaluate_shop_payment(bl_broke, tracker_unpaid, "Will you please pay for that?")
    assert task_drop is not None
    assert task_drop.name == "DROP"
    assert task_drop.args["slot"] == "b"


def test_sokoban_boulder_pathfinding():
    """Verifies that pushable boulders in Sokoban are navigable by A*."""
    nav = NavigationManager()
    lvl = nav.get_or_create_level(3, 1)  # Sokoban dnum=3, floor=1
    chars = np.full((21, 79), ord(" "))
    # Room with floor
    chars[5:15, 5:15] = ord(".")
    lvl.walkable[5:15, 5:15] = True

    # Boulder at (10, 10), open floor at (10, 11) -> pushable East!
    chars[10, 10] = ord("0")
    lvl.walkable[10, 10] = False
    hazard_costs = np.zeros(chars.shape, dtype=np.float32)

    soko_w, soko_c = nav._get_sokoban_walkable_and_costs(chars, lvl, hazard_costs)
    assert soko_w[10, 10] is True or soko_w[10, 10] == 1
    assert soko_c[10, 10] >= 30.0

    # Stepping into the boulder issues STEP delta
    from lox.navigation.astar import PathNode
    next_node = PathNode(10, 10, 1, "l")
    task = nav._step_or_open(10, 9, next_node, chars, lvl)
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)


def test_amulet_of_reflection_and_wishing():
    """Verifies that amulet of reflection is equipped and wand of wishing is zapped."""
    from lox.domain.inventory_manager import InventoryManager
    from lox.env.inventory_tracker import NormalizedItem
    inv_mgr = InventoryManager()
    blstats = make_blstats(x=5, y=5, experience=10)

    class MockInv:
        def __init__(self, items):
            self.active_items = {it.uid: it for it in items}

    # 1. Amulet of reflection equipping
    am = NormalizedItem(uid="u1", current_letter="a", raw_str="an uncursed amulet of reflection", buc_state="UNCURSED")
    inv = MockInv([am])
    eq_task = inv_mgr._evaluate_equipment(blstats, inv, role="valkyrie")
    assert eq_task is not None
    assert eq_task.name == "PUTON"
    assert eq_task.args["slot"] == "a"

    # 2. Wand of wishing zapping
    wand = NormalizedItem(uid="u2", current_letter="z", raw_str="a wand of wishing (0:3)", buc_state="UNCURSED")
    inv_wand = MockInv([wand])
    res_task = inv_mgr.evaluate_resource_turn(blstats, inv_wand)
    assert res_task is not None
    assert res_task.name == "ZAP"
    assert res_task.args["slot"] == "z"


def test_castle_drawbridge_wand_blasting():
    """Verifies that closed drawbridges on DL 25 are blasted with wand of striking."""
    nav = NavigationManager()
    lvl = nav.get_or_create_level(0, 25, depth=25)  # Castle level
    chars = np.full((21, 79), ord("."))
    from lox.navigation.astar import PathNode
    from lox.env.inventory_tracker import NormalizedItem
    next_node = PathNode(10, 11, 1, "l")

    class MockInv:
        def __init__(self, items):
            self.active_items = {it.uid: it for it in items}

    wand = NormalizedItem(uid="w1", current_letter="y", raw_str="an uncursed wand of striking", buc_state="UNCURSED")
    inv = MockInv([wand])

    task = nav._step_or_open(
        10, 10, next_node, chars, lvl,
        message="The drawbridge is closed.",
        inv_tracker=inv,
    )
    assert task.name == "ZAP"
    assert task.args["slot"] == "y"
    assert task.args["delta"] == (0, 1)


def test_blocked_stairs_retreat_and_step_off():
    """Verifies that standing on a blocked staircase steps off safely and never descends."""
    nav = NavigationManager()
    chars = np.full((21, 79), ord("."))
    blstats = make_blstats(x=5, y=5, depth=3, level_number=3, dungeon_number=0, hp=20, max_hp=20, experience=2)
    lvl = nav.get_or_create_level(0, 3)

    # Stair at (5, 5) is blocked (leads to Mines while underleveled)
    lvl.blocked_tiles.add((5, 5))
    lvl.stairs_down = (5, 5)

    # 1. step_towards_stairs must return None when stairs_down is blocked
    task_nav = nav.step_towards_stairs(5, 5, lvl, chars)
    assert task_nav is None

    # 2. evaluate_navigation_turn must step off the blocked tile onto an adjacent walkable tile
    task_step = nav.evaluate_navigation_turn(
        chars, blstats, inv_tracker=None, message="There is a staircase down here."
    )
    assert task_step.name == "STEP"
    assert task_step.args["delta"] != (0, 0)
    nr = 5 + task_step.args["delta"][0]
    nc = 5 + task_step.args["delta"][1]
    assert (nr, nc) not in lvl.blocked_tiles
    assert lvl.walkable[nr, nc]

    # 3. Message-based stair detection must not overwrite lvl.stairs_down with a blocked tile
    lvl.stairs_down = None
    nav.update_map(chars, blstats, message="There is a staircase down here.")
    assert lvl.stairs_down is None



