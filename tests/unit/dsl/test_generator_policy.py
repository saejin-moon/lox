from lox.core.types import HeroState, Observation
from lox.dsl.compiler import compile_policy


def test_class_agent_generator_execution():
    policy_code = """
class Agent:
    def __init__(self):
        self.step_count = 0
        self.mode = "explore"

    def run(self, obs):
        while True:
            self.step_count += 1
            if obs.hero.hp_frac < 0.2:
                self.mode = "emergency"
                obs = yield pray()
            elif obs.combat.adjacent_hostile:
                self.mode = "combat"
                obs = yield melee_attack_hostile()
            else:
                self.mode = "explore"
                obs = yield step_to_frontier()
"""
    executor = compile_policy(policy_code)
    assert executor is not None

    obs1 = Observation(chars=None, glyphs=None, hero=HeroState(hp=50, max_hp=50))
    runner = executor.create_runner(obs1)

    # Step 1: High HP, no hostiles -> explore
    act1 = runner.send(obs1)
    assert act1.name == "step_to_frontier"

    # Step 2: Enemy adjacent -> combat
    obs2 = Observation(chars=None, glyphs=None, hero=HeroState(hp=50, max_hp=50))
    obs2.combat.adjacent_hostile = True
    act2 = runner.send(obs2)
    assert act2.name == "melee_attack_hostile"

    # Step 3: Low HP -> pray
    obs3 = Observation(chars=None, glyphs=None, hero=HeroState(hp=5, max_hp=50))
    act3 = runner.send(obs3)
    assert act3.name == "pray"


def test_generator_with_subroutines_yield_from():
    policy_code = """
class Agent:
    def combat_routine(self, obs):
        for _ in range(2):
            obs = yield melee_attack_hostile()

    def run(self, obs):
        while True:
            if obs.combat.adjacent_hostile:
                yield from self.combat_routine(obs)
            else:
                obs = yield step_to_frontier()
"""
    executor = compile_policy(policy_code)
    assert executor is not None

    obs1 = Observation(chars=None, glyphs=None, hero=HeroState(hp=50, max_hp=50))
    obs1.combat.adjacent_hostile = True
    runner = executor.create_runner(obs1)

    act1 = runner.send(obs1)
    assert act1.name == "melee_attack_hostile"
    act2 = runner.send(obs1)
    assert act2.name == "melee_attack_hostile"
