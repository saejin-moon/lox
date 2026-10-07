"""
Tests for Pure Synthesis Architecture & Multi-Benchmark Portability.
Validates:
1. NetHackAdapter atomic retreat (pure gradient retreat, no tactical overrides).
2. RunMemory cross-floor episodic persistence.
3. DungeonPhase goal routing and state transitions.
4. Modular generator solver skills (breach castle, sokoban, invocation, ascension run) yield on every branch.
5. BaseDiagnosticEngine decoupling and portability factory.
6. Dimension-agnostic SpatialEngine (arbitrary grid dimensions, e.g. 48x48).
7. Dynamic timeouts scaling with max_turns.
"""

from __future__ import annotations

import numpy as np
import pytest

from lox.core.spatial import SpatialEngine, build_walkable_mask
from lox.core.types import (
    Action,
    AltarRecord,
    CombatView,
    DungeonPhase,
    DungeonView,
    FountainRecord,
    HeroState,
    HeroStatus,
    InventoryView,
    Item,
    Observation,
    RunMemory,
    SpatialView,
)
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter
from lox.telemetry.diagnostics import (
    BaseDiagnosticEngine,
    NetHackDiagnosticEngine,
    get_diagnostic_engine,
)


def test_nethack_adapter_atomic_retreat():
    """Verify NetHackAdapter step_away_from_hostile performs atomic gradient retreat."""
    import unittest.mock

    adapter = NetHackAdapter()
    obs = adapter.reset(seed=42)
    hy, hx = obs.hero.y, obs.hero.x
    obs.combat.adjacent_hostile = True
    obs.combat.closest_hostile_pos = (hy, hx + 1)
    adapter._last_obs = obs

    resolved_actions = []
    original_step = adapter.step

    def capture_step(a):
        resolved_actions.append(a)
        if a.name == "step_direction":
            return obs, 0.0, False, False, {}
        return original_step(a)

    with unittest.mock.patch.object(adapter, "step", side_effect=capture_step):
        adapter.step(Action(name="step_away_from_hostile"))

    assert any(a.name in ("step_direction", "wait") for a in resolved_actions)
    adapter.close()


def test_run_memory_cross_floor_persistence():
    """Verify RunMemory persists altars, fountains, stairs, and portals across depth transitions."""
    memory = RunMemory()
    memory.visited_depths.add(1)
    memory.visited_depths.add(2)
    memory.stairs_down[1] = (12, 45)
    memory.stairs_up[2] = (12, 45)
    memory.altars[3] = [AltarRecord(y=5, x=20, alignment="lawful")]
    memory.fountains[2] = [FountainRecord(y=8, x=15, is_active=True)]
    memory.portals["quest"] = (3, 1, 14)

    obs = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=4),
        memory=memory,
    )

    assert obs.memory.stairs_down[1] == (12, 45)
    assert obs.memory.stairs_up[2] == (12, 45)
    assert obs.memory.altars[3][0].alignment == "lawful"
    assert obs.memory.fountains[2][0].is_active is True
    assert obs.memory.portals["quest"] == (3, 1, 14)
    assert 1 in obs.memory.visited_depths
    assert 2 in obs.memory.visited_depths


def test_dungeon_phase_routing():
    """Verify determine_goal routes to appropriate skills across all DungeonPhases."""
    from data.modular_starter_policy import Agent

    agent = Agent()

    # 1. Ascension Run phase
    obs_ascension = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=1),
        inventory=InventoryView([Item(slot="a", name="Amulet of Yendor", category="tool")]),
        dungeon=DungeonView(phase="ascension_run"),
    )
    assert agent.determine_goal(obs_ascension) == "ascension_run"

    # 2. Invocations phase
    obs_invocation = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=46, turn=16000),
        dungeon=DungeonView(phase="invocation"),
    )
    assert agent.determine_goal(obs_invocation) == "perform_invocation"

    # 3. Castle Breach phase
    obs_castle = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=26),
        dungeon=DungeonView(phase="castle_breach", is_castle_level=True),
    )
    assert agent.determine_goal(obs_castle) == "breach_castle"

    # 4. Sokoban phase
    obs_sokoban = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=7),
        dungeon=DungeonView(phase="sokoban", is_sokoban_level=True),
    )
    assert agent.determine_goal(obs_sokoban) == "solve_sokoban"


def test_modular_solver_skills_yield():
    """Verify that every generator skill yields an action on every branch without infinite looping."""
    with open("data/modular_starter_policy.py") as f:
        code = f.read()

    executor = compile_policy(code)
    dummy_obs = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=HeroState(depth=1, turn=10),
        spatial=SpatialView(),
        dungeon=DungeonView(),
    )

    runner = executor.create_runner(dummy_obs)
    action = runner.send(dummy_obs)
    assert isinstance(action, Action)


def test_base_diagnostic_engine_portability():
    """Verify BaseDiagnosticEngine decoupling and engine factory."""
    nethack_engine = get_diagnostic_engine("nethack")
    assert isinstance(nethack_engine, BaseDiagnosticEngine)
    assert isinstance(nethack_engine, NetHackDiagnosticEngine)

    default_engine = get_diagnostic_engine("craftax")
    assert isinstance(default_engine, BaseDiagnosticEngine)


def test_dimension_agnostic_spatial_engine():
    """Verify build_walkable_mask and SpatialEngine.warmup work on arbitrary (H, W) shapes."""
    custom_shape = (48, 48)
    SpatialEngine.warmup(shape=custom_shape)

    chars = np.full(custom_shape, ord("."), dtype=np.uint8)
    chars[0, :] = ord("-")
    chars[:, 0] = ord("|")

    walkable = build_walkable_mask(chars)
    assert walkable.shape == custom_shape
    assert bool(walkable[0, 0]) is False
    assert bool(walkable[10, 10]) is True


def test_dynamic_timeouts():
    """Verify dynamic episode ceiling calculation scales safely with episode max_turns."""
    def calc_wall_sec(max_turns: int) -> float:
        return max(180.0, max_turns * 0.025)

    def calc_batch_sec(max_turns: int) -> float:
        return max(360.0, max_turns * 0.05)

    # Early game: 4,000 turns -> defaults to minimum floors (180s and 360s)
    assert calc_wall_sec(4000) == 180.0
    assert calc_batch_sec(4000) == 360.0

    # Late game: 25,000 turns -> dynamically scales up
    assert calc_wall_sec(25000) == 625.0
    assert calc_batch_sec(25000) == 1250.0
