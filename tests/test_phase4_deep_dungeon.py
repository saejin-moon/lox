"""
Phase 4 regression tests: Deep Dungeon Capabilities.
Verifies Minetown temple routing directive, water crossing (levitation),
and Medusa encounter preparation.
"""

import numpy as np
import pytest
from nle import nethack

from corp.domain.macro_director import MacroAscensionDirector, AscensionPhase
from corp.domain.medusa_handler import MedusaHandler
from corp.domain.navigation_manager import NavigationManager
from corp.domain.dungeon_graph import DungeonGraph
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem


def create_stats(
    hp: int = 20,
    max_hp: int = 20,
    x: int = 10,
    y: int = 10,
    depth: int = 1,
    exp: int = 6,
    hunger: int = 1,
    dungeon_number: int = 0,
    level_number: int = 1,
    turn: int = 100,
    condition_bits: int = 0,
) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[0] = x
    raw[1] = y
    raw[10] = hp
    raw[11] = max_hp
    raw[12] = depth
    raw[18] = exp
    raw[20] = turn
    raw[21] = hunger
    raw[23] = dungeon_number
    raw[24] = level_number
    raw[25] = condition_bits
    return BottomLineStats.from_blstats(raw)


def make_floor():
    return np.full((21, 79), ord("."), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Minetown temple routing
# ---------------------------------------------------------------------------

def test_minetown_directive():
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.MINETOWN_PROTECTION

    # DL 3 in Dungeons of Doom, Minetown not yet visited: route into the Mines branch
    blstats = create_stats(depth=3, level_number=3, exp=6)
    assert md.get_navigation_directive(blstats) == "GOTO_MINETOWN"

    # Already visited Minetown: resume normal descent
    md.state.minetown_visited = True
    assert md.get_navigation_directive(blstats) is None

    # DL outside the Mines entry range: no directive
    md.state.minetown_visited = False
    blstats_dl1 = create_stats(depth=1, level_number=1, exp=6)
    assert md.get_navigation_directive(blstats_dl1) is None

    # Donation target scales with XL
    bl = create_stats(exp=5)
    assert md.get_minetown_donation_target(bl) == 400 * 5 * 5


def test_minetown_navigation_descends_known_mines_stairs():
    """GOTO_MINETOWN routes to the recorded Mines-branch stairs and descends."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.MINETOWN_PROTECTION
    chars = make_floor()
    chars[10, 12] = ord(">")  # Mines branch stairs east

    bl = DungeonGraph()
    stats = create_stats(x=10, y=10, hp=40, max_hp=40, depth=3, level_number=3, exp=6)
    nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md, dungeon_graph=bl)

    # Record that the '>' at (10, 12) leads to the Gnomish Mines (dnum 2)
    node = bl.get_or_create_node(0, 3)
    node.stair_connections[(10, 12)] = (2, 1)

    task = nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md, dungeon_graph=bl)
    assert task is not None
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)  # toward the Mines stairs, not random frontiers


# ---------------------------------------------------------------------------
# Water crossing
# ---------------------------------------------------------------------------

def test_levitation_boots_activate_near_water():
    """Adjacent wide water body + carried levitation boots: issue WEAR."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    # Water body east of hero (2 tiles within the 3x3 scan -> not a lone sink)
    chars[10, 11] = ord("}")
    chars[11, 11] = ord("}")

    inv = InventoryNormalizer()
    boots = NormalizedItem(current_letter="b", raw_str="an uncursed pair of levitation boots", buc_state="UNCURSED")
    inv.active_items[boots.uid] = boots

    stats = create_stats(x=10, y=10, hp=20, max_hp=20, exp=6)
    task = nav_mgr.evaluate_navigation_turn(chars, stats, inv_tracker=inv)
    assert task is not None
    assert task.name == "WEAR"
    assert task.args["slot"] == "b"


def test_water_walkable_when_levitating():
    """Levitating heroes can path across water tiles."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    chars[10, 11] = ord("}")

    # LEVITATING = 1024
    stats = create_stats(x=10, y=10, hp=20, max_hp=20, exp=6, condition_bits=1024)
    lvl = nav_mgr.update_map(chars, stats)
    assert lvl.walkable[10, 11]


def test_lone_sink_does_not_trigger_levitation():
    """A single '}' tile is a sink (harmless): never waste levitation on it."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    chars[10, 11] = ord("}")  # lone sink

    inv = InventoryNormalizer()
    boots = NormalizedItem(current_letter="b", raw_str="an uncursed pair of levitation boots", buc_state="UNCURSED")
    inv.active_items[boots.uid] = boots

    stats = create_stats(x=10, y=10, hp=20, max_hp=20, exp=6)
    task = nav_mgr.evaluate_navigation_turn(chars, stats, inv_tracker=inv)
    assert task is None or task.name != "WEAR"


# ---------------------------------------------------------------------------
# Medusa handler
# ---------------------------------------------------------------------------

def test_medusa_noop_far_from_medusa():
    mh = MedusaHandler()
    bl = create_stats(depth=5, exp=8)
    inv = InventoryNormalizer()
    assert mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=False) is None


def test_medusa_equips_reflection_amulet():
    mh = MedusaHandler()
    bl = create_stats(depth=20, exp=10)
    inv = InventoryNormalizer()
    amulet = NormalizedItem(current_letter="a", raw_str="an uncursed amulet of reflection", buc_state="UNCURSED")
    inv.active_items[amulet.uid] = amulet

    task = mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=False)
    assert task is not None
    assert task.name == "PUTON"
    assert task.args["slot"] == "a"
    # After equipping, no further prep needed
    assert mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=False) is None


def test_medusa_equips_carried_reflection_shield():
    mh = MedusaHandler()
    bl = create_stats(depth=19, exp=10)
    inv = InventoryNormalizer()
    shield = NormalizedItem(current_letter="b", raw_str="an uncursed shield of reflection", buc_state="UNCURSED")
    inv.active_items[shield.uid] = shield

    task = mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=False)
    assert task is not None
    assert task.name == "WEAR"


def test_medusa_applies_blindfold_without_reflection():
    """No reflection available: apply a blindfold so the gaze cannot petrify."""
    mh = MedusaHandler()
    bl = create_stats(depth=20, exp=10)
    inv = InventoryNormalizer()
    blindfold = NormalizedItem(current_letter="c", raw_str="an uncursed blindfold", buc_state="UNCURSED")
    inv.active_items[blindfold.uid] = blindfold

    task = mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=False)
    assert task is not None
    assert task.name == "APPLY"
    assert task.args["slot"] == "c"


def test_medusa_prep_skipped_when_reflective():
    mh = MedusaHandler()
    bl = create_stats(depth=20, exp=10)
    inv = InventoryNormalizer()
    assert mh.evaluate_medusa_prep(bl, inv_tracker=inv, has_reflection=True) is None


def test_perseus_statue_loot_flag():
    mh = MedusaHandler()
    bl = create_stats(depth=20, dungeon_number=0, exp=10)
    assert mh.should_loot_perseus_statue(bl, has_reflection=False) is True
    assert mh.should_loot_perseus_statue(bl, has_reflection=True) is False
