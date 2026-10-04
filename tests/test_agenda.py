"""
Unit tests for LOX Hybrid HTN-BT Goal Agenda and DSL integration.
"""
from __future__ import annotations

import numpy as np
import pytest
from lox.core.agenda import GoalAgenda, GoalDirective, AgendaView
from lox.core.types import Observation, HeroState, DungeonView, HeroStatus, EpistemicView
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
    agenda = GoalAgenda([GoalDirective.DESCEND_STAIRS, GoalDirective.COLLECT_POISON_RES])
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
