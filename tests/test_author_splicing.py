import ast

from lox.author.agent import AuthorAgent

BASE_POLICY = """class Agent:
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
"""


def test_splice_single_method_replacement():
    patch = """def handle_combat(self, obs):
    while obs.combat.adjacent_hostile:
        if obs.combat.closest_hostile_name == "floating eye":
            obs = yield step_away_from_hostile()
        else:
            obs = yield melee_attack_hostile()
    return obs
"""
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
    patch = """class Agent:
    def handle_combat(self, obs):
        while obs.combat.adjacent_hostile:
            obs = yield throw_dagger()
        return obs
"""
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "class Agent:" in spliced
    assert "throw_dagger()" in spliced
    assert "step_to_frontier()" in spliced


def test_splice_missing_self_injection():
    # Model omitted 'self' in the definition
    patch = """def handle_combat(obs):
    while obs.combat.adjacent_hostile:
        obs = yield engrave_dust_elbereth()
    return obs
"""
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "engrave_dust_elbereth()" in spliced
    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    handle_combat = [
        m
        for m in agent_cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "handle_combat"
    ][0]
    assert len(handle_combat.args.args) >= 1
    assert handle_combat.args.args[0].arg == "self"


def test_splice_append_new_helper_method():
    patch = """def handle_loot(self, obs):
    if obs.spatial.has_nearby_loot:
        obs = yield step_to_loot()
    return obs
"""
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, patch)
    assert "handle_loot" in spliced
    assert "step_to_loot()" in spliced
    assert "handle_combat" in spliced

    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    method_names = [m.name for m in agent_cls.body if isinstance(m, ast.FunctionDef)]
    assert method_names == ["__init__", "run", "handle_combat", "handle_loot"]


def test_splice_full_class_passthrough():
    full_new_class = """class Agent:
    def __init__(self):
        self.flag = True

    def run(self, obs):
        while True:
            obs = yield wait()
"""
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


def test_splice_phase_subroutine_into_latest_policy():
    with open("data/latest_policy.py") as f:
        policy_code = f.read()

    patch = """def phase_early_scaling(self, obs):
    if obs.dungeon.can_forge_excalibur:
        obs = yield dip_excalibur()
        return obs
    obs = yield step_to_stairs_down()
    return obs
"""
    spliced = AuthorAgent.splice_policy_methods(policy_code, patch)
    assert "def phase_early_scaling(self, obs):" in spliced
    assert "dip_excalibur()" in spliced
    assert "skill_combat" in spliced
    assert "skill_scavenge_armor" in spliced
    assert "skill_explore_and_dive" in spliced
    assert "determine_goal" in spliced

    # Check compilation
    parsed = ast.parse(spliced)
    agent_cls = [n for n in parsed.body if isinstance(n, ast.ClassDef)][0]
    method_names = {m.name for m in agent_cls.body if isinstance(m, ast.FunctionDef)}
    assert "phase_early_scaling" in method_names
    assert "skill_combat" in method_names


def test_splice_whitespace_mismatched_outer_indentation():
    """Verify that patches with mismatched outer indentation levels (e.g. def at 0, next def at 4, bodies at 8) splice cleanly."""
    from lox.dsl.compiler import compile_policy

    # Exact pattern that previously failed in Gen 2 / Gen 3 synthesis log
    malformed_patch = """def determine_goal(self, obs) -> str:
        if obs.hero.dungeon_branch == 'mines':
            return 'escape_mines'
        return 'explore_and_dive'

    def skill_explore_and_dive(self, obs):
        if obs.spatial.stairs_down_known:
            obs = yield step_to_stairs_down()
            return obs
        obs = yield step_to_frontier()
        return obs
"""
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, malformed_patch)
    assert "class Agent:" in spliced
    assert "def determine_goal(self, obs)" in spliced
    assert "def skill_explore_and_dive(self, obs)" in spliced

    # Must compile cleanly via compiler
    compiled = compile_policy(spliced)
    assert compiled is not None


def test_splice_tabs_and_excessive_indentation():
    """Verify that patches with mixed tabs and excessive indents are normalized and spliced cleanly."""
    from lox.dsl.compiler import compile_policy

    tabbed_patch = "def skill_scavenge_armor(self, obs):\n\tif obs.inventory.has_unworn_armor:\n\t\tobs = yield wear_armor()\n\t\treturn obs\n\tobs = yield step_to_frontier()\n\treturn obs\n"
    spliced = AuthorAgent.splice_policy_methods(BASE_POLICY, tabbed_patch)
    assert "class Agent:" in spliced
    assert "def skill_scavenge_armor(self, obs):" in spliced

    compiled = compile_policy(spliced)
    assert compiled is not None


def test_repair_python_indentation_utility():
    """Verify that repair_python_indentation repairs unaligned unindents and preserves docstrings."""
    from lox.dsl.parser import repair_python_indentation

    code = """def determine_goal(self, obs) -> str:
        \"\"\"
        Multiline docstring.
          Inner indent.
        \"\"\"
        if obs.hero.dungeon_branch == 'mines':
            return 'escape_mines'
        return 'explore_and_dive'

    def skill_combat(self, obs):
        obs = yield wait()
        return obs
"""
    repaired = repair_python_indentation(code)
    parsed = ast.parse(repaired)
    func_names = [n.name for n in parsed.body if isinstance(n, ast.FunctionDef)]
    assert func_names == ["determine_goal", "skill_combat"]


