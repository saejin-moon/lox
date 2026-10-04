import ast
import pytest
from lox.author.agent import AuthorAgent
from lox.core.types import Observation, HeroState
import numpy as np


BASE_POLICY = '''class Agent:
    def __init__(self):
        self.retreat_count = 0

    def run(self, obs):
        while True:
            if obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue
            obs = yield step_to_frontier()

    def handle_combat(self, obs):
        while obs.combat.adjacent_hostile:
            obs = yield melee_attack_hostile()
        return obs
'''


def test_splice_single_method_replacement():
    patch = '''def handle_combat(self, obs):
    while obs.combat.adjacent_hostile:
        if obs.combat.closest_hostile_name == "floating eye":
            obs = yield step_away_from_hostile()
        else:
            obs = yield melee_attack_hostile()
    return obs
'''
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "class Agent:" in spliced
    assert "self.retreat_count = 0" in spliced
    assert "step_to_frontier()" in spliced
    assert "closest_hostile_name == 'floating eye'" in spliced
    assert "step_away_from_hostile()" in spliced

    # Check valid Python AST
    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    method_names = [m.name for m in agent_cls.body if isinstance(m, ast.FunctionDef)]
    assert method_names == ["__init__", "run", "handle_combat"]


def test_splice_class_wrapped_patch():
    patch = '''class Agent:
    def handle_combat(self, obs):
        while obs.combat.adjacent_hostile:
            obs = yield throw_dagger()
        return obs
'''
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "class Agent:" in spliced
    assert "throw_dagger()" in spliced
    assert "step_to_frontier()" in spliced


def test_splice_missing_self_injection():
    # Model omitted 'self' in the definition
    patch = '''def handle_combat(obs):
    while obs.combat.adjacent_hostile:
        obs = yield engrave_dust_elbereth()
    return obs
'''
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "engrave_dust_elbereth()" in spliced
    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    handle_combat = [m for m in agent_cls.body if isinstance(m, ast.FunctionDef) and m.name == "handle_combat"][0]
    assert len(handle_combat.args.args) >= 1
    assert handle_combat.args.args[0].arg == "self"


def test_splice_append_new_helper_method():
    patch = '''def handle_loot(self, obs):
    if obs.spatial.has_nearby_loot:
        obs = yield step_to_loot()
    return obs
'''
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "handle_loot" in spliced
    assert "step_to_loot()" in spliced
    assert "handle_combat" in spliced

    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    method_names = [m.name for m in agent_cls.body if isinstance(m, ast.FunctionDef)]
    assert method_names == ["__init__", "run", "handle_combat", "handle_loot"]


def test_splice_full_class_passthrough():
    full_new_class = '''class Agent:
    def __init__(self):
        self.flag = True

    def run(self, obs):
        while True:
            obs = yield wait()
'''
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, full_new_class)
    assert "self.flag = True" in spliced
    assert "wait()" in spliced


def test_splice_empty_base_or_patch():
    assert AuthorAgent.splice_policy_methods("", BASE_POLICY) == BASE_POLICY
    assert AuthorAgent.splice_policy_methods(BASE_POLICY, "") == BASE_POLICY
    assert AuthorAgent.splice_policy_methods(BASE_POLICY, "   ") == BASE_POLICY


def test_splice_syntax_error_resilience():
    broken_patch = "def broken(obs: return 42"
    res = AuthorAgent.splice_policy_methods(BASE_POLICY, broken_patch)
    assert res == broken_patch
