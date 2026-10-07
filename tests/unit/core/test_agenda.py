"""
Unit tests for LOX Hybrid HTN-BT Goal Agenda and DSL integration.
"""

from __future__ import annotations

import numpy as np

from lox.core.agenda import AgendaView, GoalAgenda, GoalDirective
from lox.core.types import (
    EpistemicView,
    HeroState,
    Observation,
)
from lox.dsl.compiler import compile_policy


def test_goal_agenda_stack_operations():
    agenda = GoalAgenda()
    assert agenda.active_goal == GoalDirective.EXPLORE_FLOOR

    agenda.push(GoalDirective.COLLECT_POISON_RES)
    assert agenda.active_goal == GoalDirective.COLLECT_POISON_RES

    agenda.push(GoalDirective.FORGE_EXCALIBUR)
    assert agenda.active_goal == GoalDirective.FORGE_EXCALIBUR

    popped = agenda.pop()
    assert popped == GoalDirective.FORGE_EXCALIBUR
    assert agenda.active_goal == GoalDirective.COLLECT_POISON_RES


def test_goal_agenda_milestone_evaluation():
    agenda = GoalAgenda(
        [GoalDirective.DESCEND_STAIRS, GoalDirective.COLLECT_POISON_RES]
    )
    assert agenda.active_goal == GoalDirective.COLLECT_POISON_RES

    obs_no_res = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, has_poison_res=False),
    )
    agenda.evaluate_milestones(obs_no_res)
    assert agenda.active_goal == GoalDirective.COLLECT_POISON_RES

    # Now hero acquires poison res: milestone evaluates and auto-pops
    obs_res = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, has_poison_res=True),
    )
    agenda.evaluate_milestones(obs_res)
    assert agenda.active_goal == GoalDirective.DESCEND_STAIRS


def test_policy_with_goal_agenda_and_epistemic_predicates():
    policy_code = """
class Agent:
    def __init__(self):
        self.agenda = GoalAgenda([GoalDirective.DESCEND_STAIRS, GoalDirective.COLLECT_POISON_RES])

    def run(self, obs):
        while True:
            # 1. Reflex Combat
            if obs.combat.adjacent_hostile:
                obs = yield melee_attack_hostile()
                continue

            # 2. Epistemic Safe Equipping
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 3. Strategic Agenda
            if obs.agenda.is_active(GOAL_COLLECT_POISON_RES):
                obs = yield harvest_poison_res()
                continue

            obs = yield descend()
"""
    executor = compile_policy(policy_code)
    obs = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10),
        epistemic=EpistemicView(can_safely_wear_armor=True),
        agenda=AgendaView(active_goal="collect_poison_res"),
    )
    runner = executor.create_runner(obs)
    act = runner.send(obs)
    assert act.name == "harvest_poison_res"


def test_depth_tiered_dungeon_phases():
    from lox.core.agenda import DungeonPhase

    # Phase 0: Depths 1-2
    assert GoalAgenda.get_phase(1) == DungeonPhase.EARLY_RUSH
    assert GoalAgenda.get_phase(2) == DungeonPhase.EARLY_RUSH

    # Phase 1: Depths 3-5
    assert GoalAgenda.get_phase(3) == DungeonPhase.EARLY_SCALING
    assert GoalAgenda.get_phase(5) == DungeonPhase.EARLY_SCALING

    # Phase 2: Depths 6-10 or Sokoban
    assert GoalAgenda.get_phase(6) == DungeonPhase.MID_BRANCHES
    assert GoalAgenda.get_phase(10) == DungeonPhase.MID_BRANCHES
    assert GoalAgenda.get_phase(3, branch="sokoban") == DungeonPhase.MID_BRANCHES

    # Phase 3: Depths 11-19
    assert GoalAgenda.get_phase(11) == DungeonPhase.DEEP_DUNGEON
    assert GoalAgenda.get_phase(18) == DungeonPhase.DEEP_DUNGEON

    # Phase 4: Depths 20+
    assert GoalAgenda.get_phase(20) == DungeonPhase.CASTLE_ASCENSION
    assert GoalAgenda.get_phase(35) == DungeonPhase.CASTLE_ASCENSION


def test_depth_gated_policy_execution():
    policy_code = """
class Agent:
    def run(self, obs):
        while True:
            if obs.agenda.dungeon_phase == PHASE_EARLY_RUSH:
                obs = yield step_to_stairs_down()
            elif obs.agenda.dungeon_phase == PHASE_EARLY_SCALING:
                obs = yield dip_excalibur()
            else:
                obs = yield wait()
"""
    executor = compile_policy(policy_code)

    # Observation at DL 1 (Early rush)
    agenda1 = GoalAgenda()
    obs1 = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, depth=1),
    )
    agenda1.evaluate_milestones(obs1)
    obs1.agenda = agenda1.create_view()
    runner1 = executor.create_runner(obs1)
    act1 = runner1.send(obs1)
    assert act1.name == "step_to_stairs_down"

    # Observation at DL 4 (Early scaling)
    agenda4 = GoalAgenda()
    obs4 = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, depth=4),
    )
    agenda4.evaluate_milestones(obs4)
    obs4.agenda = agenda4.create_view()
    runner4 = executor.create_runner(obs4)
    act4 = runner4.send(obs4)
    assert act4.name == "dip_excalibur"

