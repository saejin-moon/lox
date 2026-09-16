"""
Unit tests for Phase 3: Fast HTN Planner, Persona Profiler, CDCL Nogood Store,
and Spatial Navigation (Grid A* and Frontier Exploration).
"""

import numpy as np
import pytest

from corp.planner.predicates import PredicateBit, compile_predicate_mask
from corp.planner.persona import PersonaProfiler, PersonaTraitVector
from corp.planner.nogood import NogoodStore, NogoodEntry
from corp.planner.guards import HTNGuards
from corp.planner.htn import (
    HTNPlanner,
    Task,
    PrimitiveTask,
    CompoundTask,
    Method,
    PlanSignature,
    CycleDetector,
    HTNDeadlockException,
)
from corp.navigation.astar import GridAStar, PathNode
from corp.navigation.frontier import FrontierExplorer
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem


def create_stats(
    hp: int = 15,
    max_hp: int = 15,
    con: int = 12,
    dex: int = 12,
    intel: int = 12,
    wis: int = 12,
    ac: int = 10,
    max_energy: int = 10,
    hunger: int = 1,
) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[10] = hp
    raw[11] = max_hp
    raw[5] = con
    raw[4] = dex
    raw[6] = intel
    raw[7] = wis
    raw[16] = ac
    raw[15] = max_energy
    raw[21] = hunger
    return BottomLineStats.from_blstats(raw)


def test_predicate_mask_compilation():
    stats = create_stats(hp=3, max_hp=20, hunger=HungerState.WEAK)  # 15% HP -> critical
    mask = compile_predicate_mask(stats)

    assert mask & PredicateBit.HP_CRITICAL_BELOW_20_PCT
    assert mask & PredicateBit.HP_LOW_BELOW_50_PCT
    assert mask & PredicateBit.HUNGER_WEAK_OR_FAINTING
    assert not (mask & PredicateBit.IS_BLIND)


def test_persona_profiler():
    # Valkyrie-like: High HP, high con, good AC (negative or low number)
    valk_stats = create_stats(hp=20, max_hp=20, con=18, ac=3)
    valk_persona = PersonaProfiler.derive_persona(valk_stats)
    assert valk_persona.resilience > 0.85

    # Wizard-like: Low HP, low con, high int/wis, high energy
    wiz_stats = create_stats(hp=10, max_hp=10, con=10, ac=10, intel=18, wis=18, max_energy=30)
    wiz_persona = PersonaProfiler.derive_persona(wiz_stats)
    assert wiz_persona.mana > 0.85
    assert wiz_persona.resilience < 0.45


def test_nogood_store_evaluation():
    store = NogoodStore()

    # Case 1: Melee against floating eye without blindness
    state_mask = PredicateBit.ADJACENT_FLOATING_EYE
    forbidden, reason = store.is_forbidden(state_mask, "MELEE_ATTACK")
    assert forbidden is True
    assert "gaze" in reason.lower()

    # Case 2: Once blinded, melee is allowed!
    state_mask_blind = PredicateBit.ADJACENT_FLOATING_EYE | PredicateBit.IS_BLIND
    forbidden, reason = store.is_forbidden(state_mask_blind, "MELEE_ATTACK")
    assert forbidden is False

    # Case 3: Cockatrice barehanded pickup
    state_mask_rice = PredicateBit.ADJACENT_COCKATRICE
    forbidden, _ = store.is_forbidden(state_mask_rice, "PICKUP")
    assert forbidden is True

    # With gloves, pickup allowed
    state_mask_rice_gloves = PredicateBit.ADJACENT_COCKATRICE | PredicateBit.GLOVES_EQUIPPED
    forbidden, _ = store.is_forbidden(state_mask_rice_gloves, "PICKUP")
    assert forbidden is False


def test_htn_persona_weighted_decomposition():
    planner = HTNPlanner()

    # Methods for compound task 'TACTICAL_COMBAT'
    method_melee = Method(
        name="MELEE_TRADE",
        target_task="TACTICAL_COMBAT",
        preconditions=lambda s: True,
        subtasks_fn=lambda s: [PrimitiveTask("MELEE_ATTACK")],
        priority_weights=(4.0, -3.0, 0.0, 0.0, 0.0),  # High resilience preferred
    )

    method_kite = Method(
        name="KITE_RANGED",
        target_task="TACTICAL_COMBAT",
        preconditions=lambda s: True,
        subtasks_fn=lambda s: [PrimitiveTask("STEP_AWAY"), PrimitiveTask("THROW_MISSILE")],
        priority_weights=(-3.0, 4.0, 0.0, 0.0, 0.0),  # High ranged preferred
    )

    planner.register_method(method_melee)
    planner.register_method(method_kite)

    # High resilience persona: should choose MELEE_TRADE
    tank_persona = PersonaTraitVector(resilience=0.9, ranged=0.2, mana=0.1, stealth=0.3, alignment=1.0)
    plan_tank = planner.plan(CompoundTask("TACTICAL_COMBAT"), state=None, persona=tank_persona)
    assert len(plan_tank) == 1
    assert plan_tank[0].name == "MELEE_ATTACK"

    # Fragile kiter persona: should choose KITE_RANGED
    kiter_persona = PersonaTraitVector(resilience=0.1, ranged=0.9, mana=0.2, stealth=0.5, alignment=0.5)
    plan_kiter = planner.plan(CompoundTask("TACTICAL_COMBAT"), state=None, persona=kiter_persona)
    assert len(plan_kiter) == 2
    assert plan_kiter[0].name == "STEP_AWAY"
    assert plan_kiter[1].name == "THROW_MISSILE"


def test_cycle_detector():
    detector = CycleDetector(window_size=6, max_repetitions=3)
    sig1 = PlanSignature("STEP", (10, 10), turn=1)
    sig2 = PlanSignature("STEP", (10, 11), turn=2)

    assert not detector.record_and_check(sig1)  # 1st
    assert not detector.record_and_check(sig2)  # 1st
    assert not detector.record_and_check(sig1)  # 2nd
    assert not detector.record_and_check(sig2)  # 2nd
    assert detector.record_and_check(sig1)      # 3rd -> Cycle detected!


def test_grid_astar_pathfinding():
    walkable = np.zeros((21, 79), dtype=bool)
    # Create a 5x5 room from row 5-9, col 5-9
    walkable[5:10, 5:10] = True

    start = (5, 5)
    goal = (9, 9)

    path = GridAStar.find_path(start, goal, walkable)
    assert path is not None
    assert len(path) == 4  # 4 diagonal steps from (5,5) to (9,9)
    assert path[-1].row == 9 and path[-1].col == 9


def test_grid_astar_corner_clipping():
    walkable = np.zeros((21, 79), dtype=bool)
    # L-shaped corner: (5,5) and (6,6) are walkable, but (5,6) and (6,5) are walls
    walkable[5, 5] = True
    walkable[6, 6] = True
    # (5, 6) is a wall (False)
    # (6, 5) is a wall (False)

    start = (5, 5)
    goal = (6, 6)

    # Diagonal move MUST be blocked by corner clipping interlock!
    path = GridAStar.find_path(start, goal, walkable)
    assert path is None  # Cannot clip through corner


def test_grid_astar_hazard_avoidance():
    walkable = np.zeros((21, 79), dtype=bool)
    walkable[5:8, 5:8] = True  # 3x3 room

    hazard_costs = np.zeros((21, 79), dtype=float)
    # Put a dangerous pit trap right in the middle (6, 6)
    hazard_costs[6, 6] = 500.0

    start = (5, 6)
    goal = (7, 6)

    path = GridAStar.find_path(start, goal, walkable, hazard_costs=hazard_costs)
    assert path is not None
    # Path should route around (6, 6), not step through the pit
    coords = [(node.row, node.col) for node in path]
    assert (6, 6) not in coords


def test_frontier_explorer():
    walkable = np.zeros((21, 79), dtype=bool)
    unmapped = np.ones((21, 79), dtype=bool)

    # Explored a 3x3 room
    walkable[10:13, 10:13] = True
    unmapped[10:13, 10:13] = False

    # (10, 10) is a room perimeter tile adjacent to unmapped space
    frontier = FrontierExplorer.find_nearest_frontier((11, 11), walkable, unmapped)
    assert frontier is not None
    # Frontier should be on the boundary
    assert frontier[0] in (10, 12) or frontier[1] in (10, 12)

    # Corridor dead end test
    corridor_map = np.zeros((21, 79), dtype=bool)
    corridor_map[5, 5] = True
    corridor_map[5, 6] = True  # (5,5) only connects to (5,6)
    assert FrontierExplorer.is_corridor_dead_end((5, 5), corridor_map) is True
    assert FrontierExplorer.is_corridor_dead_end((5, 6), corridor_map) is True
