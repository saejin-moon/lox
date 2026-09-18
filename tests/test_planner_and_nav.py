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
from corp.domain.navigation_manager import NavigationManager
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
    x: int = 0,
    y: int = 0,
    depth: int = 1,
    exp: int = 1,
) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[0] = x
    raw[1] = y
    raw[10] = hp
    raw[11] = max_hp
    raw[12] = depth
    raw[19] = exp
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


def test_nogood_store_inverted_action_index():
    store = NogoodStore()
    # Check that _by_action is populated with pre-seeded interlocks
    assert "MELEE_ATTACK" in store._by_action
    assert "PICKUP" in store._by_action
    assert "WIELD" in store._by_action
    assert "STEP_AWAY" in store._by_action
    assert "ZAP_RAY" in store._by_action

    # Unregistered action immediately returns False in O(1)
    forbidden, reason = store.is_forbidden(0xFFFFFFFFFFFFFFFF, "EAT")
    assert forbidden is False
    assert reason == ""

    # Adding new Nogood updates both entries and _by_action
    new_entry = NogoodEntry(
        mask=PredicateBit.IS_BLIND,
        target_val=PredicateBit.IS_BLIND,
        forbidden_action="READ",
        reason="Cannot read while blind",
    )
    store.add_nogood(new_entry)
    assert "READ" in store._by_action
    assert new_entry in store._by_action["READ"]

    forbidden, reason = store.is_forbidden(PredicateBit.IS_BLIND, "READ")
    assert forbidden is True
    assert "blind" in reason.lower()


def test_enriched_predicate_mask_compilation():
    from corp.domain.combat_manager import MonsterTrack

    stats = create_stats(hp=18, max_hp=20, depth=1)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 10] = ord(">")  # Standing on stairs down
    stats_stairs = create_stats(hp=18, max_hp=20, depth=1, x=10, y=10)

    # 1. Spatial flags
    mask_stairs = compile_predicate_mask(stats_stairs, chars=chars)
    assert mask_stairs & PredicateBit.STANDING_ON_STAIRS_DOWN

    # 2. Monster adjacency flags
    m_eye = MonsterTrack(pos=(10, 11), name="floating eye", level=2, speed=0, ac=9, distance=1, is_adjacent=True, is_instakill=True, threat_score=100.0)
    m_cockatrice = MonsterTrack(pos=(11, 10), name="cockatrice", level=5, speed=6, ac=6, distance=1, is_adjacent=True, is_instakill=True, threat_score=150.0)
    m_fast = MonsterTrack(pos=(9, 10), name="soldier ant", level=3, speed=18, ac=3, distance=1, is_adjacent=True, is_instakill=False, threat_score=50.0)

    mask_mon = compile_predicate_mask(stats_stairs, chars=chars, monsters=[m_eye, m_cockatrice, m_fast])
    assert mask_mon & PredicateBit.ADJACENT_FLOATING_EYE
    assert mask_mon & PredicateBit.ADJACENT_COCKATRICE
    assert mask_mon & PredicateBit.ADJACENT_MONSTER_FASTER
    assert mask_mon & PredicateBit.HOSTILE_COUNT_GE_2


def test_htn_guards_should_descend():
    from corp.planner.guards import HTNGuards

    stats_healthy = create_stats(hp=18, max_hp=20, exp=2)
    # Healthy XL 2 with stairs known -> should descend
    assert HTNGuards.should_descend(stats_healthy, stairs_down_known=True, unvisited_count=30, turns_spent=40) is True

    # Healthy XL 1 with mostly explored floor (unvisited <= 25) -> should descend
    stats_xl1 = create_stats(hp=18, max_hp=20, exp=1)
    assert HTNGuards.should_descend(stats_xl1, stairs_down_known=True, unvisited_count=20, turns_spent=50) is True

    # High turns spent (>= 100) -> should descend
    assert HTNGuards.should_descend(stats_xl1, stairs_down_known=True, unvisited_count=40, turns_spent=105) is True

    # Critical HP (4 / 20 = 20%) -> forbidden!
    stats_critical = create_stats(hp=4, max_hp=20, exp=2)
    assert HTNGuards.should_descend(stats_critical, stairs_down_known=True, unvisited_count=10, turns_spent=100) is False

    # Adjacent hostiles -> forbidden!
    assert HTNGuards.should_descend(stats_healthy, stairs_down_known=True, has_adjacent_hostiles=True) is False


def test_navigation_step_towards_stairs():
    nav_mgr = NavigationManager()
    stats = create_stats(hp=20, max_hp=20, x=5, y=5, depth=1)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[8, 8] = ord(">")

    lvl = nav_mgr.update_map(chars, stats)
    assert lvl.stairs_down == (8, 8)

    # 1. Step towards stairs when away
    task_step = nav_mgr.step_towards_stairs(5, 5, lvl, chars)
    assert task_step is not None
    assert task_step.name in ("STEP", "OPEN")

    # 2. Descend directly when on stairs down
    task_descend = nav_mgr.step_towards_stairs(8, 8, lvl, chars)
    assert task_descend is not None
    assert task_descend.name == "DESCEND"


def test_navigation_gold_and_container_looting():
    nav_mgr = NavigationManager()
    stats = create_stats(hp=20, max_hp=20, x=5, y=5, depth=1)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[5, 5] = ord("@")
    chars[5, 6] = ord("$")  # Gold adjacent

    # 1. Pathing towards gold
    task_step = nav_mgr.evaluate_navigation_turn(chars, stats)
    assert task_step.name == "STEP"
    assert task_step.args["delta"] == (0, 1)

    # 2. Standing on gold -> PICKUP
    stats_on_gold = create_stats(hp=20, max_hp=20, x=6, y=5, depth=1)
    chars[6, 5] = ord("$")
    task_pickup = nav_mgr.evaluate_navigation_turn(chars, stats_on_gold)
    assert task_pickup.name == "PICKUP"


def test_inventory_manager_twoweapon():
    from corp.domain.inventory_manager import InventoryManager
    from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem

    inv_mgr = InventoryManager()
    tracker = InventoryNormalizer()
    stats = create_stats(hp=20, max_hp=20)

    # Valkyrie wielding a long sword and carrying a dagger, with no shield
    sword = NormalizedItem(uid="s1", raw_str="an uncursed long sword (weapon in hand)", current_letter="a", buc_state="UNCURSED", equipped=True)
    dagger = NormalizedItem(uid="d1", raw_str="an uncursed dagger", current_letter="b", buc_state="UNCURSED", equipped=False)
    tracker.active_items["s1"] = sword
    tracker.active_items["d1"] = dagger

    task_two = inv_mgr.evaluate_resource_turn(stats, tracker, role="valkyrie")
    assert task_two is not None
    assert task_two.name == "TWOWEAPON"


def test_branch_aware_level_representation():
    nav_mgr = NavigationManager()

    # Dungeons of Doom Dlvl 3 (dnum=0, dlevel=3, depth=3)
    stats_dod = create_stats(depth=3)
    lvl_dod = nav_mgr.get_or_create_level(stats_dod)
    lvl_dod.walkable[5, 5] = True
    lvl_dod.stairs_down = (10, 10)

    # Gnomish Mines Dlvl 1 (dnum=2, dlevel=1, depth=3)
    # Both are at depth 3, but in different branches!
    stats_mines_raw = BottomLineStats(
        x=5, y=5, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=20, max_hp=20,
        depth=3, gold=50, energy=10, max_energy=10, ac=5, monster_level=1,
        experience=1, turn=100, hunger_state=0, encumbrance=0,
        dungeon_number=2, level_number=1, condition_bits=0, alignment=0,
    )
    lvl_mines = nav_mgr.get_or_create_level(stats_mines_raw)

    # Must be distinct levels!
    assert lvl_dod is not lvl_mines
    assert lvl_dod.dnum == 0 and lvl_dod.dlevel == 3
    assert lvl_mines.dnum == 2 and lvl_mines.dlevel == 1
    assert lvl_mines.stairs_down is None  # Not corrupted by DoD stairs
    assert not lvl_mines.walkable[5, 5]   # Not corrupted by DoD walkable mask

    # Test LevelStore backward compatibility
    assert nav_mgr.levels[3] is lvl_dod  # By depth
    assert nav_mgr.levels[(0, 3)] is lvl_dod
    assert nav_mgr.levels[(2, 1)] is lvl_mines


def test_level_scoped_coordinate_caches():
    nav_mgr = NavigationManager()
    lvl1 = nav_mgr.get_or_create_level(1)
    lvl2 = nav_mgr.get_or_create_level(2)

    # Throne and container interaction on level 1
    lvl1.visited_thrones.add((5, 5))
    lvl1.looted_containers.add((6, 6))
    lvl1.collected_gold.add((7, 7))
    lvl1.dead_end_searches[(8, 8)] = 10

    # Level 2 must remain unaffected (no cross-level coordinate collision)
    assert (5, 5) not in lvl2.visited_thrones
    assert (6, 6) not in lvl2.looted_containers
    assert (7, 7) not in lvl2.collected_gold
    assert (8, 8) not in lvl2.dead_end_searches

    # Aggregate properties on nav_mgr reflect level state
    assert (5, 5) in nav_mgr.visited_thrones
    assert (6, 6) in nav_mgr.looted_containers
    assert (7, 7) in nav_mgr.collected_gold
    assert nav_mgr.dead_end_searches[(8, 8)] == 10

    # Reset clears everything
    nav_mgr.reset()
    assert len(nav_mgr.levels) == 0
    assert len(nav_mgr.visited_thrones) == 0
    assert len(nav_mgr.looted_containers) == 0


def test_navigation_manager_instakill_monster_avoidance():
    import nle.nethack as nh
    nav_mgr = NavigationManager()
    stats = create_stats(x=5, y=5)

    # 3x3 room floor
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    glyphs = np.full((21, 79), nh.NO_GLYPH, dtype=np.int32)
    for r in range(4, 7):
        for c in range(4, 7):
            chars[r, c] = ord(".")
    chars[5, 5] = ord("@")

    # Find gas spore monster id
    spore_id = None
    for m in range(nh.NUMMONS):
        pm = nh.permonst(m)
        if "gas spore" in pm.mname.lower():
            spore_id = m
            break
    assert spore_id is not None

    # Place gas spore directly East at (5, 6)
    glyphs[5, 6] = nh.GLYPH_MON_OFF + spore_id

    # Make (5, 6) the next node candidate in a test step
    next_node = PathNode(5, 6, 1, "l")
    lvl = nav_mgr.get_or_create_level(stats)
    lvl.walkable[5, 6] = True

    # Step into (5, 6) must be vetoed by instakill guard and return WAIT!
    task = nav_mgr._step_or_open(5, 5, next_node, chars, lvl, glyphs=glyphs)
    assert task.name == "WAIT"


def test_find_best_search_tile_prioritizes_dead_ends():
    nav_mgr = NavigationManager()
    stats = create_stats(x=5, y=5)
    lvl = nav_mgr.get_or_create_level(stats)

    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    # Room at (4..6, 4..6)
    for r in range(4, 7):
        for c in range(4, 7):
            chars[r, c] = ord(".")
            lvl.walkable[r, c] = True

    # Corridor from (5, 7) to (5, 10)
    for c in range(7, 11):
        chars[5, c] = ord("#")
        lvl.walkable[5, c] = True
        lvl.corridors.add((5, c))

    # Dead end at (5, 10) has only 1 neighbor (5, 9)
    best_tile = nav_mgr.find_best_search_tile(5, 5, lvl, chars)
    # Must prioritize the corridor dead end at (5, 10)
    assert best_tile == (5, 10)

    # If dead end has been searched 25 times, priority shifts to unsearched room boundaries
    lvl.dead_end_searches[(5, 10)] = 25
    next_best = nav_mgr.find_best_search_tile(5, 5, lvl, chars)
    assert next_best != (5, 10)
    assert next_best is not None


def test_search_reachability_and_burst_searching():
    """Verifies that unreachable tiles are filtered out and burst searching executes 5 times on arrival."""
    nav_mgr = NavigationManager()
    stats = create_stats(x=5, y=5)
    lvl = nav_mgr.get_or_create_level(stats)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)

    # Reachable room at (4..6, 4..6)
    for r in range(4, 7):
        for c in range(4, 7):
            chars[r, c] = ord(".")
            lvl.walkable[r, c] = True

    # Disconnected unreachable room at (15..17, 15..17)
    for r in range(15, 18):
        for c in range(15, 18):
            chars[r, c] = ord(".")
            lvl.walkable[r, c] = True

    # 1. BFS distances must show -1 for unreachable room
    dist_grid = nav_mgr._compute_bfs_distances(5, 5, lvl)
    assert dist_grid[5, 5] == 0
    assert dist_grid[15, 15] == -1

    # 2. find_best_search_tile must select a reachable tile, never the unreachable room
    best_tile = nav_mgr.find_best_search_tile(5, 5, lvl, chars)
    assert best_tile is not None
    assert dist_grid[best_tile[0], best_tile[1]] >= 0

    # 3. Burst searching: when standing on search tile, evaluates 5 consecutive SEARCH tasks
    lvl.visited[lvl.walkable] = 1
    target_pos = best_tile
    stats_at_target = create_stats(x=target_pos[1], y=target_pos[0])
    nav_mgr._current_search_spot = None
    nav_mgr._current_search_burst = 0

    for i in range(1, 6):
        task = nav_mgr.evaluate_navigation_turn(chars, stats_at_target)
        assert task.name == "SEARCH"
        assert nav_mgr._current_search_burst == i



