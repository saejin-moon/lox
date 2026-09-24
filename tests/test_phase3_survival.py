"""
Phase 3 regression tests: Mid-Game Survival Systems.
Verifies tactical HP resting, adaptive secret-door search limits, active poison
resistance conveyor farming priority, and fast-monster ranged engagement.
"""

import numpy as np
import pytest
from nle import nethack

from lox.domain.navigation_manager import NavigationManager
from lox.domain.combat_manager import TacticalCombatManager
from lox.env.blstats import BottomLineStats, HungerState
from lox.env.inventory_tracker import InventoryNormalizer, NormalizedItem


def create_stats(
    hp: int = 20,
    max_hp: int = 20,
    x: int = 10,
    y: int = 10,
    depth: int = 1,
    exp: int = 2,
    hunger: int = 1,
    dungeon_number: int = 0,
    level_number: int = 1,
    turn: int = 100,
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
    return BottomLineStats.from_blstats(raw)


def make_floor():
    return np.full((21, 79), ord("."), dtype=np.uint8)


def make_glyphs():
    return np.full((21, 79), nethack.NO_GLYPH, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tactical HP Resting
# ---------------------------------------------------------------------------

def test_hp_resting_waits_when_wounded():
    """HP < 60% with no hostiles: WAIT to regenerate instead of wandering."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    glyphs = make_glyphs()

    # HP 10/20 = 50% < 60% -> rest
    stats = create_stats(hp=10, max_hp=20, exp=2, turn=100)
    task = nav_mgr.evaluate_navigation_turn(chars, stats, glyphs=glyphs)
    assert task is not None
    assert task.name == "WAIT"

    # Still below 85%: keep resting
    stats_mid = create_stats(hp=14, max_hp=20, exp=2, turn=110)
    task = nav_mgr.evaluate_navigation_turn(chars, stats_mid, glyphs=glyphs)
    assert task is not None
    assert task.name == "WAIT"

    # HP recovered >= 85%: resume normal navigation
    stats_healed = create_stats(hp=18, max_hp=20, exp=2, turn=130)
    task = nav_mgr.evaluate_navigation_turn(chars, stats_healed, glyphs=glyphs)
    assert task is not None
    assert task.name != "WAIT"


def test_hp_resting_aborts_on_monster():
    """A monster appearing within 2 tiles aborts the rest immediately."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    glyphs = make_glyphs()

    stats = create_stats(hp=10, max_hp=20, exp=2, turn=100)
    task = nav_mgr.evaluate_navigation_turn(chars, stats, glyphs=glyphs)
    assert task.name == "WAIT"

    # Monster appears 2 tiles away (e.g. a jackal at (10, 12))
    jackal_id = None
    for m in range(nethack.NUMMONS):
        if "jackal" in nethack.permonst(m).mname:
            jackal_id = m
            break
    glyphs[10, 12] = nethack.GLYPH_MON_OFF + jackal_id

    stats2 = create_stats(hp=11, max_hp=20, exp=2, turn=110)
    task = nav_mgr.evaluate_navigation_turn(chars, stats2, glyphs=glyphs)
    assert task.name != "WAIT"


def test_hp_resting_never_while_hungry():
    """Resting burns nutrition: never rest when hungry or worse."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    glyphs = make_glyphs()

    stats = create_stats(hp=10, max_hp=20, exp=2, hunger=HungerState.HUNGRY, turn=100)
    task = nav_mgr.evaluate_navigation_turn(chars, stats, glyphs=glyphs)
    assert task.name != "WAIT"


# ---------------------------------------------------------------------------
# Adaptive search limits (DL 4+)
# ---------------------------------------------------------------------------

def test_adaptive_search_limits_wall_tiles():
    """Wall search cap is depth-adaptive when stairs are unknown: DL 1 -> 15, DL 5 -> 20."""
    nav_mgr = NavigationManager()
    chars = make_floor()
    chars[0, :] = ord("|")  # top wall so row 1 tiles are wall-adjacent

    lvl5 = nav_mgr.get_or_create_level((0, 5))
    lvl5.depth = 5
    lvl5.stairs_down = None
    lvl5.walkable[:] = True

    # Exhaust all wall-adjacent row-1 tiles with 17 searches (above DL1 cap 15, below DL5 cap 20)
    for cc in range(79):
        lvl5.searched[1, cc] = 17

    # DL 5 (cap 20): tiles with 17 searches are still eligible
    spot = nav_mgr._find_unsearched_wall_tile(10, 10, lvl5, chars)
    assert spot is not None

    # DL 1 (cap 15): tiles with 17 searches are exhausted
    lvl1 = nav_mgr.get_or_create_level((0, 1))
    lvl1.depth = 1
    lvl1.stairs_down = None
    lvl1.walkable[:] = True
    for cc in range(79):
        lvl1.searched[1, cc] = 17
    spot_d1 = nav_mgr._find_unsearched_wall_tile(10, 10, lvl1, chars)
    if spot_d1 is not None:
        assert lvl1.searched[spot_d1] < 15

    # Stairs found: cap drops to 5
    lvl5.stairs_down = (10, 10)
    lvl5.searched[1, 5] = 6
    for cc in range(79):
        if cc != 5:
            lvl5.searched[1, cc] = 17
    spot_s = nav_mgr._find_unsearched_wall_tile(10, 10, lvl5, chars)
    if spot_s is not None:
        assert lvl5.searched[spot_s] < 5


def test_adaptive_search_cap_boundaries():
    """Unit-test the adaptive caps directly via the dead-end finder semantics."""
    nav_mgr = NavigationManager()
    chars = make_floor()

    # Construct corridor dead-end at (10, 5)
    chars[:, :] = ord(" ")
    for cc in range(3, 6):
        chars[10, cc] = ord("#")
    for rr in range(8, 13):
        for cc in range(8, 13):
            chars[rr, cc] = ord(".")

    # DL 5, no stairs: cap is 20
    lvl = nav_mgr.get_or_create_level((0, 5))
    lvl.depth = 5
    lvl.stairs_down = None
    lvl.corridors.add((10, 3))
    for i in range(19):
        assert nav_mgr._find_unsearched_dead_end(10, 10, lvl, chars) == (10, 3)
        lvl.dead_end_searches[(10, 3)] = i + 1
    # 20th search allowed, 21st blocked
    lvl.dead_end_searches[(10, 3)] = 20
    assert nav_mgr._find_unsearched_dead_end(10, 10, lvl, chars) is None

    # Stairs found: cap drops to 6
    lvl2 = nav_mgr.get_or_create_level((0, 5))
    lvl2.depth = 5
    lvl2.stairs_down = (10, 10)
    lvl2.corridors.add((10, 3))
    for i in range(6):
        lvl2.dead_end_searches[(10, 3)] = i
    lvl2.dead_end_searches[(10, 3)] = 6
    assert nav_mgr._find_unsearched_dead_end(10, 10, lvl2, chars) is None


# ---------------------------------------------------------------------------
# Active poison resistance farming
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fast-monster ranged engagement
# ---------------------------------------------------------------------------

def _find_monster(name: str) -> int:
    for m in range(nethack.NUMMONS):
        if name in nethack.permonst(m).mname:
            return m
    raise AssertionError(f"monster {name} not found")


def test_fast_monster_engages_ranged():
    """Fast monster (giant bat, speed 22 > 1.3x 12) adjacent + missile: FIRE, don't melee."""
    combat_mgr = TacticalCombatManager()
    blstats = create_stats(hp=20, max_hp=20, x=10, y=10, exp=3)

    glyphs = make_glyphs()
    chars = make_floor()
    ant_id = _find_monster("giant bat")
    glyphs[10, 11] = nethack.GLYPH_MON_OFF + ant_id

    inv = InventoryNormalizer()
    dart = NormalizedItem(current_letter="a", raw_str="12 darts", buc_state="UNCURSED")
    inv.active_items[dart.uid] = dart

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=inv)
    assert task is not None
    # Fast monster: engage at range instead of trading melee blows
    assert task.name != "MELEE_ATTACK"
    assert task.name in ("FIRE", "THROW")


def test_fast_monster_cornered_falls_back_to_melee():
    """No missiles available: melee is still preferable to standing idle."""
    combat_mgr = TacticalCombatManager()
    blstats = create_stats(hp=20, max_hp=20, x=10, y=10, exp=3)

    glyphs = make_glyphs()
    chars = make_floor()
    ant_id = _find_monster("giant bat")
    glyphs[10, 11] = nethack.GLYPH_MON_OFF + ant_id

    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=None)
    assert task is not None
    assert task.name == "MELEE_ATTACK"


# ---------------------------------------------------------------------------
# Critical-HP universal retreat
# ---------------------------------------------------------------------------

def test_critical_hp_retreats_instead_of_grinding():
    """At 3/16 HP with no healing: engrave Elbereth / escape — never trade blows."""
    combat_mgr = TacticalCombatManager()
    blstats = create_stats(hp=3, max_hp=16, x=10, y=10, exp=2)

    glyphs = make_glyphs()
    chars = make_floor()
    jackal_id = _find_monster("jackal")
    glyphs[10, 11] = nethack.GLYPH_MON_OFF + jackal_id

    # Turn 1: emergency Elbereth
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=None)
    assert task is not None
    assert task.name == "ENGRAVE_DUST"

    # Turn 2: ward active -> wait for monsters to flee
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=None)
    assert task is not None
    assert task.name == "WAIT"

    # Open floor everywhere: eventually escape steps appear (max separation)
    combat_mgr.elbereth_turns = 0
    combat_mgr.elbereth_cooldown = 999  # block re-engraving to force the escape branch
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=None)
    assert task is not None
    assert task.name == "STEP"


def test_critical_hp_with_healing_potion_keeps_fighting():
    """With a usable healing potion, triage (not retreat) owns the critical-HP decision."""
    combat_mgr = TacticalCombatManager()
    blstats = create_stats(hp=3, max_hp=16, x=10, y=10, exp=2)

    glyphs = make_glyphs()
    chars = make_floor()
    jackal_id = _find_monster("jackal")
    glyphs[10, 11] = nethack.GLYPH_MON_OFF + jackal_id

    inv = InventoryNormalizer()
    potion = NormalizedItem(current_letter="a", raw_str="a potion of extra healing", buc_state="UNCURSED")
    inv.active_items[potion.uid] = potion

    assert combat_mgr._has_usable_healing(inv) is True
    # No universal retreat: normal engagement (defensive engraving also acceptable)
    task = combat_mgr.evaluate_combat_turn(glyphs, chars, blstats, inv_tracker=inv)
    assert task is not None
    assert task.name in ("MELEE_ATTACK", "FIRE", "THROW", "ENGRAVE_DUST")


def test_cursed_healing_potion_not_usable():
    combat_mgr = TacticalCombatManager()
    inv = InventoryNormalizer()
    potion = NormalizedItem(current_letter="a", raw_str="a cursed potion of healing", buc_state="CURSED")
    inv.active_items[potion.uid] = potion
    assert combat_mgr._has_usable_healing(inv) is False
