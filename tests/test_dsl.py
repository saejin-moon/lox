import numpy as np
import pytest

from lox.core.types import HeroState, HungerState, Observation
from lox.dsl.compiler import compile_policy
from lox.dsl.parser import DSLValidationError


def test_dsl_compile_and_execute():
    policy_code = """
def emergency_survival():
    if hp_frac <= 0.30 and can_safely_pray:
        pray()
    elif hp_frac <= 0.50 and has_healing:
        quaff_healing()

def handle_nutrition():
    if hunger_state >= WEAK:
        eat_carried_food()

def explore():
    step_to_frontier()

plan = [
    emergency_survival,
    handle_nutrition,
    explore,
]
"""

    tree = compile_policy(policy_code)
    assert tree is not None

    obs = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=None,
        hero=HeroState(hp=10, max_hp=50),  # hp_frac = 0.20 <= 0.30
    )

    # With can_safely_pray = True
    memory = {"can_safely_pray": True}
    act = tree.execute(obs, memory=memory)
    assert act is not None
    assert act.name == "pray"

    # With can_safely_pray = False, but has_healing = True and hp_frac = 0.40 <= 0.50
    memory = {"can_safely_pray": False, "has_healing": True}
    obs.hero.hp = 20  # hp_frac = 0.40
    act = tree.execute(obs, memory=memory)
    assert act is not None
    assert act.name == "quaff_healing"

    # High HP, but hunger is WEAK
    obs.hero.hp = 50
    obs.hero.hunger_state = HungerState.WEAK
    act = tree.execute(obs, memory=memory)
    assert act is not None
    assert act.name == "eat_carried_food"

    # Default -> explore
    obs.hero.hunger_state = HungerState.NORMAL
    act = tree.execute(obs, memory=memory)
    assert act is not None
    assert act.name == "step_to_frontier"


def test_dsl_shorthand_syntax():
    shorthand_code = """
macro combat:
    if adjacent_hostile:
        melee_attack_hostile()

macro explore:
    step_to_frontier()

plan:
    combat
    explore
"""
    tree = compile_policy(shorthand_code)
    assert tree is not None


def test_dsl_disallowed_syntax_rejected():
    # Attempting to use an import
    bad_code = """
import os
def exploit():
    wait()
"""
    with pytest.raises(DSLValidationError):
        compile_policy(bad_code)

    # Attempting to call an unknown unapproved function
    bad_func = """
def hack():
    os_system_rm_rf()
"""
    with pytest.raises(DSLValidationError):
        compile_policy(bad_func)

    # Attempting to access dunder attribute
    bad_dunder = """
class ExploitAgent:
    def run(self, obs):
        x = self.__class__
        yield wait()
"""
    with pytest.raises(DSLValidationError):
        compile_policy(bad_dunder)


def test_dsl_generator_and_while_loop_allowed():
    code = """
class Agent:
    def __init__(self):
        self.step_count = 0

    def run(self, obs):
        while True:
            self.step_count += 1
            if obs.hero.hp_frac < 0.2:
                obs = yield pray()
            else:
                obs = yield wait()
"""
    executor = compile_policy(code)
    assert executor is not None
    obs = Observation(chars=None, glyphs=None, hero=HeroState(hp=5, max_hp=50))
    runner = executor.create_runner(obs)
    # Prime / run
    act = runner.send(obs) if hasattr(runner, "send") else next(runner)
    assert act is not None
    assert act.name == "pray"


def test_dsl_pass_statement_allowed():
    code_with_pass = """
def fallback():
    if hp_frac < 0.2:
        pray()
    else:
        pass

plan = [fallback]
"""
    tree = compile_policy(code_with_pass)
    assert tree is not None


def test_dsl_step_to_coordinate_args():
    code = """
class Agent:
    def run(self, obs):
        while True:
            obs = yield step_to(12, 34)
"""
    executor = compile_policy(code)
    obs = Observation(chars=None, glyphs=None, hero=HeroState())
    runner = executor.create_runner(obs)
    act = runner.send(obs) if hasattr(runner, "send") else next(runner)
    assert act.name == "step_to"
    assert act.target_pos == (12, 34)


def test_dsl_lambda_syntax():
    code = """
class Agent:
    def run(self, obs):
        items = sorted(obs.inventory, key=lambda it: it.slot)
        safe = list(filter(lambda c: c.is_safe, obs.corpses))
        obs = yield wait()
"""
    executor = compile_policy(code)
    assert executor is not None


def test_dsl_lambda_exploit_rejected():
    code = """
class Agent:
    def run(self, obs):
        fn = lambda x: x.__class__
        obs = yield wait()
"""
    with pytest.raises(DSLValidationError):
        compile_policy(code)


def test_policy_contract_near_duplicate_method_rejected():
    """Verify that near-duplicate methods (e.g. typos like skill_explore_and_div) are rejected."""
    code = """
class Agent:
    def determine_goal(self, obs):
        return "explore"

    def run(self, obs):
        while True:
            goal = self.determine_goal(obs)
            if goal == "explore":
                obs = yield from self.skill_explore_and_dive(obs)

    def skill_explore_and_dive(self, obs):
        obs = yield wait()
        return obs

    def skill_explore_and_div(self, obs):
        obs = yield wait()
        return obs
"""
    with pytest.raises(DSLValidationError, match="near-duplicate methods with suspected typo"):
        compile_policy(code)


def test_policy_contract_nonexistent_subroutine_rejected():
    """Verify that calling self.nonexistent() is caught with suggestions."""
    code = """
class Agent:
    def determine_goal(self, obs):
        return "combat"

    def run(self, obs):
        while True:
            goal = self.determine_goal(obs)
            if goal == "combat":
                obs = yield from self.skill_combatt(obs)

    def skill_combat(self, obs):
        obs = yield wait()
        return obs
"""
    with pytest.raises(DSLValidationError, match="calls nonexistent subroutine 'self.skill_combatt.*Did you mean 'skill_combat'"):
        compile_policy(code)


def test_policy_contract_unhandled_goal_rejected():
    """Verify that a goal returned by determine_goal() that is never handled in run() is rejected."""
    code = """
class Agent:
    def determine_goal(self, obs):
        if obs.hero.depth >= 10:
            return "solve_castle"
        return "combat"

    def run(self, obs):
        while True:
            goal = self.determine_goal(obs)
            if goal == "combat":
                obs = yield wait()

    def skill_combat(self, obs):
        obs = yield wait()
        return obs
"""
    with pytest.raises(DSLValidationError, match="Goal.*solve_castle.*never handled in run"):
        compile_policy(code)


def test_policy_contract_orphan_goal_rejected():
    """Verify that a goal checked in run() that is never returned by determine_goal() is rejected."""
    code = """
class Agent:
    def determine_goal(self, obs):
        return "combat"

    def run(self, obs):
        while True:
            goal = self.determine_goal(obs)
            if goal == "combat":
                obs = yield wait()
            elif goal == "pacing_decay":
                obs = yield wait()
"""
    with pytest.raises(DSLValidationError, match="Goal.*pacing_decay.*never returned by determine_goal"):
        compile_policy(code)
