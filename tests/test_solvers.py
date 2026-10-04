"""
Unit tests for LOX Algorithmic Sub-Solvers (Altar BUC and Poison Harvesting).
"""
from __future__ import annotations

import numpy as np
import pytest
from lox.core.types import (
    Observation,
    HeroState,
    DungeonView,
    FloorCorpse,
    Item,
    InventoryView,
    CombatView,
)
from lox.core.epistemic import EpistemicEngine
from lox.envs.solvers.altar_solver import AltarBUCSolver
from lox.envs.solvers.poison_solver import PoisonResHarvestSolver


def test_altar_buc_solver_state_machine():
    solver = AltarBUCSolver()
    epistemic = EpistemicEngine()

    # Step 1: Hero not on altar -> navigates to altar
    obs_not_on_altar = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=5, x=5),
        dungeon=DungeonView(standing_on_altar=False, adjacent_altar=True),
        inventory=InventoryView([Item(slot="a", name="iron helm", category="armor")]),
    )
    act1 = solver.plan_step(obs_not_on_altar, epistemic, known_altar_pos=(5, 6))
    assert act1 is not None
    assert act1.name == "step_to"
    assert act1.target_pos == (5, 6)

    # Step 2: Hero standing on altar with unconfirmed item -> drops item
    obs_on_altar = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=5, x=6),
        dungeon=DungeonView(standing_on_altar=True),
        inventory=InventoryView([Item(slot="a", name="iron helm", category="armor")]),
    )
    act2 = solver.plan_step(obs_on_altar, epistemic)
    assert act2 is not None
    assert act2.name == "drop"
    assert act2.slot == "a"

    # Step 3: Finished dropping queue -> transitions to pickup
    act3 = solver.plan_step(obs_on_altar, epistemic)
    assert act3 is not None
    assert act3.name == "pickup"

    # Step 4: Done -> returns None
    act4 = solver.plan_step(obs_on_altar, epistemic)
    assert act4 is None


def test_poison_harvest_solver():
    # Hero without poison resistance: detects fresh soldier ant corpse
    obs = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, has_poison_res=False),
        combat=CombatView(hostile_count_fov=0),
        corpses=[
            FloorCorpse(name="soldier ant corpse", y=10, x=12, drop_turn=10, age_turns=5, is_poisonous=True, is_fresh=True),
            FloorCorpse(name="jackal corpse", y=10, x=15, drop_turn=10, age_turns=5, is_fresh=True),
        ],
    )
    act = PoisonResHarvestSolver.plan_step(obs)
    assert act is not None
    assert act.name == "step_to"
    assert act.target_pos == (10, 12)

    # If standing on the ant corpse -> eats it
    obs_standing = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=12, has_poison_res=False),
        combat=CombatView(hostile_count_fov=0),
        corpses=[
            FloorCorpse(name="soldier ant corpse", y=10, x=12, drop_turn=10, age_turns=5, is_poisonous=True, is_fresh=True),
        ],
    )
    act_standing = PoisonResHarvestSolver.plan_step(obs_standing)
    assert act_standing is not None
    assert act_standing.name == "eat_floor_corpse"

    # If hero already has poison resistance -> returns None
    obs_immune = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, has_poison_res=True),
        corpses=[FloorCorpse(name="soldier ant corpse", y=10, x=12, is_poisonous=True)],
    )
    assert PoisonResHarvestSolver.plan_step(obs_immune) is None
