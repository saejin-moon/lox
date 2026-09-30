import numpy as np
import pytest
from lox.core.types import Observation, HeroState, HungerState, Action
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
    # Attempting to use a while loop
    bad_code = """
def infinite():
    while True:
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
