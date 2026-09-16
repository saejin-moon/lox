"""
Unit tests for Holistic Cognitive Impasse Evaluator (System 2 trigger heuristic).
"""

import pytest
import numpy as np

from corp.deliberative.impasse_evaluator import HolisticImpasseEvaluator, ImpasseEvaluation
from corp.domain.navigation_manager import LevelMap
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import NormalizedItem
from tests.test_domain_and_workers import make_test_blstats


def test_micro_oscillation_rejection():
    evaluator = HolisticImpasseEvaluator(min_stall_turns=15)
    blstats = make_test_blstats(hp=16, max_hp=16, x=10, y=10, turn=50)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    lvl_map = LevelMap(depth=1, walkable=np.ones((21, 79), dtype=bool), visited=np.zeros((21, 79), dtype=np.int32))

    # Record 3 stalls (micro-oscillation)
    for _ in range(3):
        evaluator.record_stall("NAVIGATE")

    result = evaluator.evaluate(
        blstats=blstats,
        chars=chars,
        lvl_map=lvl_map,
        inv_items=[],
        failed_task="NAVIGATE",
    )

    assert not result.should_trigger
    assert "Micro-oscillation" in result.diagnosis
    assert result.score == 0.0


def test_stagnation_with_unexplored_frontiers_rejection():
    evaluator = HolisticImpasseEvaluator(min_stall_turns=5, stagnation_window=30)
    lvl_map = LevelMap(depth=1, walkable=np.ones((21, 79), dtype=bool), visited=np.zeros((21, 79), dtype=np.int32))
    # Mark some visited, but leave plenty of unvisited walkable tiles
    lvl_map.visited[10, 10] = 5
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    # Simulate 25 turns of zero progress
    for t in range(25):
        blstats = make_test_blstats(hp=16, max_hp=16, x=10, y=10, turn=100 + t, score=100)
        evaluator.record_step(blstats, lvl_map)
        evaluator.record_stall("NAVIGATE")

    result = evaluator.evaluate(
        blstats=blstats,
        chars=chars,
        lvl_map=lvl_map,
        inv_items=[],
        failed_task="NAVIGATE",
        prayer_cooldown_remaining=500,
    )

    # Frontiers still exist on the floor, so topology is NOT exhausted
    assert result.topology_score == 0.0
    # No tools in inventory, so affordance is 0
    assert result.affordance_score == 0.0
    # Not enough to trigger
    assert not result.should_trigger


def test_true_macro_impasse_trigger():
    evaluator = HolisticImpasseEvaluator(min_stall_turns=5, stagnation_window=30)
    
    # Fully explored floor: all walkable tiles visited, walls fully searched
    walkable = np.zeros((21, 79), dtype=bool)
    walkable[10:15, 10:20] = True
    visited = np.zeros((21, 79), dtype=np.int32)
    visited[walkable] = 5
    searched = np.zeros((21, 79), dtype=np.int32)
    searched[walkable] = 15  # Searched > 12 times

    lvl_map = LevelMap(depth=1, walkable=walkable, visited=visited, searched=searched)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[walkable] = ord(".")

    # Simulate 25 turns of zero progress
    for t in range(25):
        blstats = make_test_blstats(
            hp=6, max_hp=16, x=10, y=10, turn=100 + t, score=100, hunger_state=HungerState.WEAK
        )
        evaluator.record_step(blstats, lvl_map)
        evaluator.record_stall("NAVIGATE")

    # Inventory with escape tools and unidentified scrolls
    items = [
        NormalizedItem(raw_str="a pick-axe (weapon in hands)", equipped=True),
        NormalizedItem(raw_str="a scroll labeled FOOBAR"),
        NormalizedItem(raw_str="a wand of digging"),
    ]

    result = evaluator.evaluate(
        blstats=blstats,
        chars=chars,
        lvl_map=lvl_map,
        inv_items=items,
        failed_task="NAVIGATE",
    )

    assert result.stagnation_score >= 0.5
    assert result.topology_score == 1.0
    assert result.affordance_score >= 0.7
    assert result.urgency_multiplier >= 2.0
    assert result.score >= 1.0
    assert result.should_trigger


def test_peaceful_chokepoint_dilemma():
    evaluator = HolisticImpasseEvaluator(min_stall_turns=3)
    blstats = make_test_blstats(
        hp=10, max_hp=16, x=10, y=10, turn=200, hunger_state=HungerState.HUNGRY
    )
    chars = np.full((21, 79), ord("#"), dtype=np.uint8)
    lvl_map = LevelMap(depth=1, walkable=np.ones((21, 79), dtype=bool), visited=np.zeros((21, 79), dtype=np.int32))

    for _ in range(5):
        evaluator.record_stall("NAVIGATE")

    result = evaluator.evaluate(
        blstats=blstats,
        chars=chars,
        lvl_map=lvl_map,
        inv_items=[NormalizedItem(raw_str="a wand of teleportation")],
        failed_task="NAVIGATE",
        adjacent_monster_names=["shopkeeper"],
        message="really attack shopkeeper?",
    )

    assert result.dilemma_score >= 0.9
    assert result.should_trigger
