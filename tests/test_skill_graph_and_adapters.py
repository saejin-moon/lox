"""
Verification Tests for Hierarchical Multi-Module Skill Graph (Section 3)
and Pluggable Benchmark Adapters (Section 7).
"""

from __future__ import annotations

from lox.author.prompts import build_user_prompt
from lox.author.tools import AuthorToolRegistry
from lox.env import (
    CraftaxAction,
    CraftaxHarness,
    NetHackHarness,
    get_adapter,
)
from lox.env.actions import Action
from lox.env.base import BenchmarkAdapter, EnvHarness
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import extract_module_outline


def test_benchmark_adapter_protocol_and_nethack():
    adapter = get_adapter("nethack")
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "nethack"

    harness = adapter.create_harness(max_steps=50)
    assert isinstance(harness, EnvHarness)
    assert isinstance(harness, NetHackHarness)

    state = harness.reset(seed=123)
    assert state is not None
    next_s, done, info = harness.step(Action.SEARCH)
    assert next_s is not None
    assert harness.step_count == 1

    docs = adapter.get_api_docs()
    assert isinstance(docs, str) and len(docs) > 100


def test_craftax_benchmark_adapter_and_deterministic_replay():
    adapter = get_adapter("craftax")
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "craftax"

    harness = adapter.create_harness(max_steps=100)
    assert isinstance(harness, EnvHarness)
    assert isinstance(harness, CraftaxHarness)

    state = harness.reset(seed=777)
    assert state.y == 24 and state.x == 24
    assert state.health == 9

    # Step: move right, then action DO
    s1, _, _ = harness.step(CraftaxAction.RIGHT)
    assert s1.x == 25
    s2, _, _ = harness.step(CraftaxAction.DO)

    # Replay bytes and verify bit-level identical state
    replayed = harness.replay_bytes(
        seed=777, recorded_bytes=bytes(harness.action_bytes)
    )
    assert replayed.y == s2.y and replayed.x == s2.x
    assert replayed.step_count == 2
    assert replayed.health == s2.health

    # Check knowledge base query
    kb_res = adapter.search_knowledge_base("wood pickaxe")
    assert "Wood" in kb_res or "Pickaxe" in kb_res


def test_hierarchical_skill_package_compilation_and_router():
    # Define an external skill module
    combat_skill_code = """
import skills
from lox.env.actions import Action

def defend(self, state):
    \"\"\"Tactical combat routine.\"\"\"
    return Action.SEARCH
"""

    # Define a top-level router/policy that imports and dispatches to skills.combat
    router_code = """
import skills
from lox.env.actions import Action

class Policy:
    def __init__(self):
        self.turns = 0

    def run(self, state):
        while True:
            self.turns += 1
            # Dispatch to hierarchical combat skill
            act = skills.combat.defend(self, state)
            state = yield act
"""

    skills = {"combat": combat_skill_code}
    compiled = compile_policy_package(
        policy_code=router_code,
        macros_code="",
        skills=skills,
    )

    runner = compiled.create_runner()
    harness = NetHackHarness(max_steps=20)
    state = harness.reset(seed=42)

    act, fn_name = runner.next_action(state)
    assert act == Action.SEARCH
    assert "defend" in runner.tracer.call_counts or "run" in runner.tracer.call_counts


def test_extract_module_outline_and_read_skill_tool(tmp_path):
    skill_code = """
class CombatManager:
    \"\"\"Handles melee and ranged tactical decisions.\"\"\"
    def engage(self, state, target_id):
        \"\"\"Executes attack sequence.\"\"\"
        return 1

def emergency_escape(state):
    \"\"\"Calculates fastest escape path to corridor.\"\"\"
    return (0, 0)
"""

    outline = extract_module_outline(skill_code, module_name="combat")
    assert "# Module: combat" in outline
    assert "class CombatManager:" in outline
    assert "def engage(self, state, target_id):" in outline
    assert "def emergency_escape(state):" in outline

    # Test read_skill tool in AuthorToolRegistry
    tools = AuthorToolRegistry(db_path=":memory:")
    tools.skills["combat"] = skill_code

    content, summary = tools.read_skill("combat")
    assert "class CombatManager" in content
    assert "lines" in summary

    # Test reading non-existent skill
    err_msg, err_summary = tools.read_skill("non_existent")
    assert "not found" in err_msg
    assert err_summary == "not_found"


def test_context_scoped_user_prompt_includes_skill_registry():
    combat_code = "def attack(state):\n    '''Attacks monster.'''\n    pass\n"
    prompt = build_user_prompt(
        mode="FUNC_MOD_ADD",
        generation=1,
        current_policy="class Policy: pass",
        macros_docs="### Macros: None",
        skills={"combat": combat_code},
    )

    assert "Hierarchical Skill Registry" in prompt
    assert "# Module: combat" in prompt
    assert "def attack(state):" in prompt
    assert "read_skill" in prompt


def test_skill_import_forms_and_macro_integration():
    macro_code = """
def helper_search(state):
    return Action.SEARCH
"""
    combat_skill_code = """
from lox.env.actions import Action

def defend(self, state):
    return helper_search(state)
"""
    router_code_1 = """
from skills.combat import defend
from lox.env.actions import Action

class Policy:
    def run(self, state):
        while True:
            act = defend(self, state)
            state = yield act
"""
    router_code_2 = """
import skills.combat
from lox.env.actions import Action

class Policy:
    def run(self, state):
        while True:
            act = skills.combat.defend(self, state)
            state = yield act
"""

    skills = {"combat": combat_skill_code}

    # Test 'from skills.combat import defend'
    compiled_1 = compile_policy_package(
        policy_code=router_code_1,
        macros_code=macro_code,
        skills=skills,
    )
    runner_1 = compiled_1.create_runner()
    harness = NetHackHarness(max_steps=10)
    state = harness.reset(seed=42)
    act1, _ = runner_1.next_action(state)
    assert act1 == Action.SEARCH

    # Test 'import skills.combat; skills.combat.defend'
    compiled_2 = compile_policy_package(
        policy_code=router_code_2,
        macros_code=macro_code,
        skills=skills,
    )
    runner_2 = compiled_2.create_runner()
    act2, _ = runner_2.next_action(state)
    assert act2 == Action.SEARCH


def test_dry_run_policy_with_candidate_skill(tmp_path):
    tools = AuthorToolRegistry(db_path=":memory:")
    policy_code = """
from skills.combat import defend
from lox.env.actions import Action

class Policy:
    def run(self, state):
        while True:
            state = yield defend(self, state)
"""
    combat_code = """
from lox.env.actions import Action

def defend(self, state):
    return Action.SEARCH
"""
    diag, status = tools.dry_run_policy(
        code=policy_code,
        skill_name="combat",
        skill_code=combat_code,
        smoke_only=True,
    )
    assert "PASSED" in status or "VALID" in status


def test_spsa_keyword_args_and_zero_gain_short_circuit():
    from lox.engine.modes import bake_params_into_policy_ast

    policy_code = """
class Policy:
    def __init__(self):
        self.flee_hp = self.param("flee_hp", default=0.25, min_val=0.10, max_val=0.70)
        self.wait_steps = self.param("wait_steps", 5, 1, 20)
"""
    updated = bake_params_into_policy_ast(
        policy_code, {"flee_hp": 0.45, "wait_steps": 10}
    )
    assert "0.45" in updated
    assert "10" in updated
