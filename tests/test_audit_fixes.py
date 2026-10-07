"""
Tests verifying the fixes implemented from the comprehensive codebase audit.
"""

from __future__ import annotations

import ast
import numpy as np
import pyarrow as pa
import pytest

from lox.core.tree import Blackboard
from lox.core.types import FloorCorpse, HeroState, InventoryView, Item, Observation
from lox.dsl.compiler import compile_policy, _compile_condition_node
from lox.envs.solvers.castle_solver import CastleDrawbridgeSolver
from lox.knowledge.invariants import REGISTRY
from lox.telemetry.diagnostics import CausalTimelineAnalyzer
from lox.telemetry.parquet import EPISODE_SCHEMA


def test_parquet_timeout_last_5_actions_compatibility():
    """Verify that timeout fallback last_5_actions string is compatible with pyarrow string schema."""
    fallback_record = {
        "run_id": "test_run",
        "episode_id": "test_timeout_ep",
        "depth": 1,
        "score": 0,
        "turns": 100,
        "death_reason": "TimeoutError",
        "solved": False,
        "wall_sec": 180.0,
        "role": "valkyrie",
        "gold": 0,
        "max_depth": 1,
        "steps": 100,
        "attacks": 0,
        "descents": 0,
        "searches": 0,
        "eats": 0,
        "prayers": 0,
        "death_category": "timeout",
        "inventory_at_death": "",
        "last_5_actions": "wait",
        "turns_dl1": 100,
        "turns_dl2": 0,
        "turns_mines": 0,
        "killer": "timeout",
        "ac_at_death": 10,
        "hp_at_death": 0,
        "max_hp_at_death": 15,
        "excalibur_forged": False,
        "root_cause": "PACING_STALL",
        "turns_fainting": 0,
        "has_body_armor": False,
        "is_oscillating": False,
    }
    table = pa.Table.from_pylist([fallback_record], schema=EPISODE_SCHEMA)
    assert table.num_rows == 1
    assert table.column("last_5_actions")[0].as_py() == "wait"


def test_diagnostics_missed_artifact_with_final_score():
    """Verify that MISSED_ARTIFACT triggers when score is passed as final_score."""
    ep_summary = {
        "episode_id": "ep_test_artifact",
        "final_depth": 5,
        "ep_turns": 1500,
        "final_score": 600,
        "inventory_at_death": "a +1 long sword (weapon in hand)",
        "death_reason": "killed by an orc",
        "killer": "orc",
    }
    dossier = CausalTimelineAnalyzer.analyze_episode(ep_summary)
    assert any("MISSED_ARTIFACT" in d for d in dossier.causal_drivers)


def test_observation_properties_and_compiler_predicates():
    """Verify new properties on HeroState, InventoryView, and Observation."""
    hero = HeroState(can_pray=True, hp=15, max_hp=15)
    assert hero.can_safely_pray is True

    inv = InventoryView()
    inv.append(Item(slot="a", name="a food ration", category="food"))
    assert inv.has_carried_food is True

    obs = Observation(
        chars=np.full((21, 79), ord(".")),
        glyphs=None,
        hero=hero,
        inventory=inv,
        corpses=[
            FloorCorpse(name="newt corpse", y=0, x=0, drop_turn=10, age_turns=5, is_poisonous=False, is_deadly=False, is_fresh=True)
        ],
    )
    assert obs.floor_corpse_adjacent is True
    assert obs.corpse_is_safe is True
    assert obs.corpse_is_fresh is True
    assert obs.corpse_is_deadly is False

    # Test Python policy access to properties
    code = """
class Agent:
    def run(self, obs):
        while True:
            if obs.floor_corpse_adjacent and obs.corpse_is_safe and obs.hero.can_safely_pray and obs.inventory.has_carried_food:
                obs = yield wait()
            else:
                obs = yield wait()
"""
    tree = compile_policy(code)
    runner = tree.create_runner(obs)
    action = runner.send(None)
    assert action.name == "wait"

    # Test DSL BT condition node evaluation of bare predicate names
    bb = Blackboard(obs)
    cond_node = ast.parse("floor_corpse_adjacent and corpse_is_safe and can_safely_pray and has_carried_food", mode="eval").body
    evaluator = _compile_condition_node(cond_node)
    assert evaluator(bb) is True


def test_castle_solver_avoids_moat_on_retreat():
    """Verify that CastleDrawbridgeSolver checks backward tile and avoids water."""
    solver = CastleDrawbridgeSolver()
    chars = np.full((21, 79), ord("."))
    # Set hero at (10, 20), closed drawbridge at (10, 21) (dist == 1)
    # The normal backward step would be (10, 19). Let's make (10, 19) moat water '}'!
    chars[10, 19] = ord("}")
    hero = HeroState(y=10, x=20)
    obs = Observation(chars=chars, glyphs=None, hero=hero)

    action = solver.plan_step(obs, drawbridge_pos=(10, 21))
    assert action is not None
    assert action.name == "step_direction"
    # It must NOT step directly into (10, 19) (direction (0, -1))
    assert action.direction != (0, -1)


def test_invariant_registry_len():
    """Verify InvariantRegistry supports len() and has 47 invariants."""
    assert len(REGISTRY) == 47
