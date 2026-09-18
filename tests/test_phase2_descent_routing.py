"""
Phase 2 regression tests: Descent & Branch Routing.
Verifies MacroDirector navigation directive wiring, aggressive descent priority,
mines retreat policy, Sokoban prize-exit (anti-oscillation), and rest-on-stairs guard.
"""

import numpy as np
import pytest

from corp.domain.macro_director import MacroAscensionDirector, AscensionPhase
from corp.domain.navigation_manager import NavigationManager
from corp.env.blstats import BottomLineStats, HungerState
from corp.planner.guards import HTNGuards


def create_stats(
    hp: int = 15,
    max_hp: int = 15,
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
    raw[18] = exp  # XL at NLE_BL_XP (18)
    raw[20] = turn
    raw[21] = hunger
    raw[23] = dungeon_number
    raw[24] = level_number
    return BottomLineStats.from_blstats(raw)


def make_floor():
    """21x79 room floor grid."""
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    return chars


# ---------------------------------------------------------------------------
# MacroDirector.get_navigation_directive
# ---------------------------------------------------------------------------

def test_directive_deep_descent():
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.DEEP_DESCENT
    blstats = create_stats(depth=10, exp=8)
    assert md.get_navigation_directive(blstats) == "DESCEND"

    # Non-descent phases: no directive
    md2 = MacroAscensionDirector()
    assert md2.get_navigation_directive(blstats) is None


def test_directive_sokoban_entry_and_exit():
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.SOKOBAN_PROGRESSION

    # DL 5-9 in Dungeons of Doom during Sokoban phase: hunt the branch entry
    blstats_dl6 = create_stats(depth=6, level_number=6, exp=6)
    assert md.get_navigation_directive(blstats_dl6) == "ENTER_SOKOBAN"

    # Too deep (>= 9): phase auto-transitions to MINETOWN_PROTECTION, so no Sokoban hunt
    blstats_dl9 = create_stats(depth=9, level_number=9, exp=6)
    md.update_state(blstats_dl9, role="valkyrie")
    assert md.state.current_phase == AscensionPhase.MINETOWN_PROTECTION

    # In Sokoban with prize collected: exit directive
    blstats_soko = create_stats(depth=8, level_number=3, dungeon_number=3, exp=6)
    assert md.get_navigation_directive(blstats_soko) is None
    md.state.sokoban_prize_collected = True
    assert md.get_navigation_directive(blstats_soko) == "EXIT_SOKOBAN"


def test_directive_mines_retreat():
    md = MacroAscensionDirector()

    # Mine's End (dlevel 10 in dnum 2): ascend, there are no downstairs below
    blstats_mine_end = create_stats(depth=12, dungeon_number=2, level_number=10, exp=7)
    assert md.get_navigation_directive(blstats_mine_end) == "ASCEND_FROM_MINES"

    # Minetown visited + temple donations complete: ascend to resume progression
    md.state.minetown_visited = True
    md.state.temple_donations_count = 3
    blstats_minetown = create_stats(depth=6, dungeon_number=2, level_number=4, exp=7)
    assert md.get_navigation_directive(blstats_minetown) == "ASCEND_FROM_MINES"

    # Normal Mines level, nothing special: no directive
    md.state.temple_donations_count = 0
    blstats_mines_mid = create_stats(depth=5, dungeon_number=2, level_number=4, exp=7)
    assert md.get_navigation_directive(blstats_mines_mid) is None


def test_mines_retreat_navigation_ascends():
    """Fully explored Mines level with no downstairs must ascend (Phase 2: black-hole fix)."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    chars = make_floor()

    stats = create_stats(x=10, y=10, hp=40, max_hp=40, exp=8, dungeon_number=2, level_number=4, depth=6)
    nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)

    # Simulate a fully explored dead-end Mines level: no downstairs, nothing unvisited
    lvl = nav_mgr.get_or_create_level((2, 4))
    lvl.visited[:] = 1
    lvl.turns_spent = 100

    stats2 = create_stats(x=10, y=10, hp=40, max_hp=40, exp=8, dungeon_number=2, level_number=4, depth=6)
    task = nav_mgr.evaluate_navigation_turn(chars, stats2, macro_director=md)
    assert task is not None
    assert task.name == "ASCEND"


def test_sokoban_prize_exit_descends():
    """After collecting the Sokoban prize, the agent must descend out of Sokoban, not oscillate."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    md.state.sokoban_prize_collected = True
    chars = make_floor()
    chars[10, 15] = ord(">")  # stairs down (exit chain)
    chars[5, 5] = ord("<")

    # Hero on DL 3 of Sokoban (< 4): old logic would route UP ('<'), oscillating with floor 4
    stats = create_stats(x=10, y=10, hp=40, max_hp=40, exp=6, dungeon_number=3, level_number=3, depth=8)
    lvl = nav_mgr.update_map(chars, stats)
    assert lvl.stairs_down == (10, 15)

    task = nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    assert task is not None
    # Must route DOWN (towards the '>' exit), never ascend '<' after prize collection
    if task.name == "STEP":
        assert task.args["delta"] == (0, 1)
    else:
        assert task.name in ("DESCEND", "OPEN")


def test_sokoban_entry_hunt_ascends():
    """Macro directive ENTER_SOKOBAN routes to '<' upstairs instead of wandering."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.SOKOBAN_PROGRESSION
    chars = make_floor()
    chars[10, 15] = ord("<")  # stairs up -> potential Sokoban branch entry

    stats = create_stats(x=10, y=10, hp=40, max_hp=40, exp=6, depth=6, level_number=6)
    lvl = nav_mgr.update_map(chars, stats)
    assert lvl.stairs_up == (10, 15)

    task = nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    assert task is not None
    # Route toward the '<' at (10, 15), not to unvisited frontiers
    if task.name == "STEP":
        assert task.args["delta"] == (0, 1)
    else:
        assert task.name in ("ASCEND", "OPEN")


def test_aggressive_descent_before_container_looting():
    """Descent must be evaluated BEFORE container looting (Phase 2 reorder)."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    chars = make_floor()
    chars[10, 9] = ord("(")  # unlooted container adjacent (west)
    chars[10, 15] = ord(">")  # stairs down (east)

    stats = create_stats(x=10, y=10, hp=20, max_hp=20, exp=2, depth=1)
    nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    lvl = nav_mgr.get_or_create_level(stats)
    lvl.turns_spent = 100  # high turns -> should_descend triggers

    task = nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    assert task is not None
    assert task.name == "STEP"
    # Move EAST towards stairs, not WEST towards the container
    assert task.args["delta"] == (0, 1)


def test_directive_descend_overrides_farming_deferral():
    """DEEP_DESCENT directive must bypass MacroDirector stair-farming deferral."""
    nav_mgr = NavigationManager()
    md = MacroAscensionDirector()
    md.state.current_phase = AscensionPhase.DEEP_DESCENT
    chars = make_floor()
    chars[10, 15] = ord(">")

    # DL 2, unvisited > 15, turns < 150, XL 2: should_defer_stairs_for_farming would defer
    stats = create_stats(x=10, y=10, hp=20, max_hp=20, exp=2, depth=2, level_number=2)
    assert md.should_defer_stairs_for_farming(stats, unvisited_count=30, turns_spent=100) is True

    nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    lvl = nav_mgr.get_or_create_level(stats)
    lvl.turns_spent = 100

    task = nav_mgr.evaluate_navigation_turn(chars, stats, macro_director=md)
    assert task is not None
    assert task.name == "STEP"
    assert task.args["delta"] == (0, 1)  # toward stairs at (10, 15)


def test_should_rest_on_stairs_guard():
    """On stairs with sub-threshold HP: WAIT to regenerate instead of wandering off."""
    # HP 40%: below descent threshold (45%), above critical triage (20%) -> rest
    stats = create_stats(hp=8, max_hp=20, exp=2)
    assert HTNGuards.should_rest_on_stairs(stats, on_stairs_down=True) is True
    assert HTNGuards.should_rest_on_stairs(stats, on_stairs_down=False) is False

    # Healthy HP: no rest needed
    stats_healthy = create_stats(hp=20, max_hp=20, exp=2)
    assert HTNGuards.should_rest_on_stairs(stats_healthy, on_stairs_down=True) is False

    # Critical HP (< 20%): emergency triage handles it, not rest
    stats_critical = create_stats(hp=3, max_hp=20, exp=2)
    assert HTNGuards.should_rest_on_stairs(stats_critical, on_stairs_down=True) is False


def test_rest_on_stairs_navigation_returns_wait():
    nav_mgr = NavigationManager()
    chars = make_floor()
    chars[10, 10] = ord(">")  # hero standing on stairs down

    # HP 8/20 = 40%: can_descend False, not critical -> rest on stairs
    stats = create_stats(x=10, y=10, hp=8, max_hp=20, exp=2, depth=3, level_number=3, turn=500)
    nav_mgr.evaluate_navigation_turn(chars, stats)
    lvl = nav_mgr.get_or_create_level(stats)
    assert lvl.stairs_down == (10, 10)

    task = nav_mgr.evaluate_navigation_turn(chars, stats)
    assert task is not None
    assert task.name == "WAIT"
