"""
Tests for ExecutionGuard, LoopGuardTransformer, and infinite loop protections.
Ensures zero-yield busy loops in policies cannot hang worker processes or dry-run validation.
"""
from lox.dsl.compiler import compile_policy, ExecutionGuard, LoopGuardTransformer
from lox.core.types import Observation, HeroState, Action


def test_infinite_loop_guard_interception():
    # Policy with an internal zero-yield while loop
    bad_code = """
class Agent:
    def run(self, obs):
        while True:
            # Zero-yield spin
            x = 1 + 1
"""
    tree = compile_policy(bad_code)
    obs = Observation(chars=None, glyphs=None, hero=HeroState())
    runner = tree.create_runner(obs)
    # Runner must gracefully intercept the infinite loop and return Action(wait) without hanging
    act = runner.send(obs)
    assert act.name == "wait"


def test_generator_with_bad_branch_fallback():
    # Policy that yields once, then enters an infinite loop
    bad_code = """
class Agent:
    def run(self, obs):
        obs = yield wait()
        while True:
            pass
"""
    tree = compile_policy(bad_code)
    obs = Observation(chars=None, glyphs=None, hero=HeroState())
    runner = tree.create_runner(obs)
    act1 = runner.send(obs)
    assert act1.name == "wait"

    # Second step hits the infinite loop
    act2 = runner.send(obs)
    assert act2.name == "wait"


def test_healthy_policy_unaffected_by_guard():
    # Normal policy yielding actions across multiple steps
    good_code = """
class Agent:
    def __init__(self):
        self.step_num = 0
    def run(self, obs):
        while True:
            self.step_num += 1
            obs = yield search()
"""
    tree = compile_policy(good_code)
    obs = Observation(chars=None, glyphs=None, hero=HeroState())
    runner = tree.create_runner(obs)
    for _ in range(10):
        act = runner.send(obs)
        assert act.name == "search"
