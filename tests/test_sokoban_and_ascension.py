"""
Tests for Sokoban, Digging, Castle Drawbridge Solver, and Ascension mechanisms.
"""

from __future__ import annotations

import numpy as np

from lox.core.digging import DiggingRouter
from lox.core.sokoban import SokobanSolver
from lox.core.types import (
    CombatView,
    DungeonView,
    HeroState,
    HeroStatus,
    InventoryView,
    Item,
    Observation,
    SpatialView,
)
from lox.envs.solvers.castle_solver import CastleDrawbridgeSolver


def test_sokoban_deadlock_detection():
    walls = np.zeros((21, 79), dtype=bool)
    pits = np.zeros((21, 79), dtype=bool)

    # Place walls to create a top-left corner at (5, 5)
    walls[4, 5] = True  # North wall
    walls[5, 4] = True  # West wall

    # Boulder at (5, 5) is in a corner deadlock if not on a pit
    assert SokobanSolver.is_deadlock_position(5, 5, walls, pits) is True

    # If it is on a pit, it is not deadlocked
    pits[5, 5] = True
    assert SokobanSolver.is_deadlock_position(5, 5, walls, pits) is False


def test_sokoban_solver_best_push():
    walls = np.zeros((21, 79), dtype=bool)
    boulders = np.zeros((21, 79), dtype=bool)
    pits = np.zeros((21, 79), dtype=bool)
    clean_floor = np.ones((21, 79), dtype=bool)

    # Pit at (10, 15)
    pits[10, 15] = True

    # Boulder at (10, 12), Hero at (10, 11)
    boulders[10, 12] = True
    hero_pos = (10, 11)

    # Push from (10, 11) into (10, 12) pushes boulder to (10, 13) towards pit (10, 15)
    best_push = SokobanSolver.find_best_boulder_push(
        hero_pos, walls, boulders, pits, clean_floor
    )
    assert best_push is not None
    assert best_push["boulder_pos"] == (10, 12)
    assert best_push["push_dir"] == (0, 1)  # East
    assert best_push["hero_stand_pos"] == (10, 11)
    assert best_push["is_ready_to_push"] is True


def test_digging_router_cardinal_direction():
    hero_pos = (10, 20)
    # Target directly south
    assert DiggingRouter.get_cardinal_tunnel_direction(hero_pos, (18, 20)) == (1, 0)
    # Target directly north
    assert DiggingRouter.get_cardinal_tunnel_direction(hero_pos, (2, 20)) == (-1, 0)
    # Target east
    assert DiggingRouter.get_cardinal_tunnel_direction(hero_pos, (10, 30)) == (0, 1)
    # Target west
    assert DiggingRouter.get_cardinal_tunnel_direction(hero_pos, (10, 10)) == (0, -1)


def test_digging_router_bresenham_ray():
    ray = DiggingRouter.bresenham_ray((5, 5), (5, 9))
    assert ray == [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]

    ray_diag = DiggingRouter.bresenham_ray((2, 2), (5, 5))
    assert ray_diag[0] == (2, 2)
    assert ray_diag[-1] == (5, 5)
    assert len(ray_diag) == 4


def test_digging_router_find_dig_target():
    walkable = np.ones((21, 79), dtype=bool)
    # Wall block at x=15
    walkable[:, 15] = False
    chars = np.full((21, 79), ord("."), dtype=np.uint8)

    target = DiggingRouter.find_dig_target((10, 10), (10, 20), walkable, chars)
    assert target is not None
    assert target["target_tile"] == (10, 15)
    assert target["direction"] == (0, 1)


def test_castle_drawbridge_detection():
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    # Moat at (10, 39)
    chars[10, 39] = ord("}")
    # Drawbridge '#' at (10, 40)
    chars[10, 40] = ord("#")

    detected = CastleDrawbridgeSolver.detect_drawbridge(chars)
    assert detected == (10, 40)


def test_castle_drawbridge_plan_step():
    solver = CastleDrawbridgeSolver()
    drawbridge_pos = (10, 40)

    # Case 1: Hero adjacent to drawbridge at (10, 39) - danger, step back!
    hero_adj = HeroState(
        y=10,
        x=39,
        hp=50,
        max_hp=50,
        energy=10,
        max_energy=10,
        ac=0,
        level=10,
        depth=26,
        dungeon_num=0,
        gold=100,
        score=5000,
        turn=1000,
        turns_on_level=10,
        hunger_state=0,
        dungeon_branch="dungeon",
    )
    obs_adj = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=hero_adj,
        inventory=InventoryView([]),
        status=HeroStatus(),
        combat=CombatView(),
        spatial=SpatialView(),
        dungeon=DungeonView(),
    )
    action_adj = solver.plan_step(obs_adj, drawbridge_pos)
    assert action_adj.name == "step_direction"
    assert action_adj.direction == (0, -1)  # Step away to the west

    # Case 2: Hero at distance 2 orthogonal: (10, 38) with wand of striking
    hero_dist2 = HeroState(
        y=10,
        x=38,
        hp=50,
        max_hp=50,
        energy=10,
        max_energy=10,
        ac=0,
        level=10,
        depth=26,
        dungeon_num=0,
        gold=100,
        score=5000,
        turn=1001,
        turns_on_level=11,
        hunger_state=0,
        dungeon_branch="dungeon",
    )
    wand_item = Item(slot="a", name="wand of striking (0:4)", category="wand")
    inv = InventoryView([wand_item])
    obs_dist2 = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=hero_dist2,
        inventory=inv,
        status=HeroStatus(),
        combat=CombatView(),
        spatial=SpatialView(),
        dungeon=DungeonView(),
    )
    action_dist2 = solver.plan_step(obs_dist2, drawbridge_pos)
    assert action_dist2.name == "zap_offensive_wand"
    assert action_dist2.direction == (0, 1)  # Fire east towards (10, 40)


def test_ascension_inventory_properties():
    items = [
        Item(
            slot="a", name="gray dragon scale mail", category="armor", is_equipped=True
        ),
        Item(slot="b", name="bag of holding", category="tool"),
        Item(slot="c", name="speed boots", category="armor", is_equipped=True),
        Item(slot="d", name="wand of wishing (0:3)", category="wand"),
        Item(slot="e", name="wand of striking (0:5)", category="wand"),
        Item(slot="f", name="magic marker (0:80)", category="tool"),
        Item(slot="g", name="unicorn horn", category="tool"),
    ]
    inv = InventoryView(items)

    assert inv.has_gdsm is True
    assert inv.has_sdsm is False
    assert inv.has_bag_of_holding is True
    assert inv.has_speed_boots is True
    assert inv.has_wand_of_wishing is True
    assert inv.has_wand_of_striking is True
    assert inv.has_magic_marker is True
    assert inv.has_unicorn_horn is True
    assert inv.get_striking_slot() == "e"
    assert inv.get_unicorn_horn_slot() == "g"


def test_bag_of_holding_explosion_safety():
    from lox.core.epistemic import EpistemicEngine, ShannonSafeGate

    epistemic = EpistemicEngine()
    # Safe items
    assert epistemic.is_safe_for_bag_of_holding("potion of extra healing") is True
    assert epistemic.is_safe_for_bag_of_holding("scroll of identify") is True
    assert epistemic.is_safe_for_bag_of_holding("wand of death (0:4)") is True

    # Lethal explosive items
    assert epistemic.is_safe_for_bag_of_holding("wand of cancellation") is False
    assert epistemic.is_safe_for_bag_of_holding("wand of tricks") is False
    assert epistemic.is_safe_for_bag_of_holding("bag of holding") is False
    assert epistemic.is_safe_for_bag_of_holding("bag of tricks") is False

    safe, msg = ShannonSafeGate.can_safely_insert_bag_of_holding("wand of cancellation")
    assert safe is False
    assert "cancellation" in msg


def test_sokoban_entrance_detection_and_sweep():
    from lox.envs.nethack import NetHackAdapter
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=101)

    # Simulate arrival on DL 6 with explicit descent
    adapter.last_depth = 6
    adapter.last_dnum = 0
    adapter.arrival_stairs_up = (10, 20)
    adapter.known_stairs_up_set = {(10, 20)}
    adapter.sokoban_entrance_pos = None

    # Discovered a second staircase up at (15, 30) (Sokoban entrance)
    adapter.known_stairs_up_set.add((15, 30))

    # Re-extract observation with DL 6 in blstats
    raw_obs = dict(adapter._last_raw_obs)
    bl = np.array(raw_obs["blstats"])
    bl[12] = 6
    bl[23] = 0
    raw_obs["blstats"] = bl
    obs = adapter._extract_obs(raw_obs)

    assert adapter.sokoban_entrance_pos == (15, 30)
    assert obs.dungeon.has_sokoban_entrance is True
    assert obs.dungeon.sokoban_entrance_pos == (15, 30)
    adapter.close()


def test_latest_policy_sokoban_execution():
    from lox.dsl.compiler import compile_policy
    with open("data/latest_policy.py") as f:
        compiled = compile_policy(f.read())
    assert compiled is not None

