"""Domain-scoped tests for LOX skills, splicing, and bundling."""

from __future__ import annotations

import ast

import pytest

from lox.author.tools import merge_skill_module_update
from lox.env import NetHackHarness
from lox.env.actions import Action
from lox.lang.bundle import (
    extract_json_payload,
    extract_skill_imports,
    load_skills_dir,
    unpack_code_bundle,
    write_skills_dir,
)
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import (
    PolicyValidationError,
    splice_functions_into_policy,
    three_way_ast_merge,
    validate_python_code,
)


def test_function_splicing():
    base_policy = """
class Policy:
    def explore(self, state):
        state = yield Action.SEARCH
        return state

    def run(self, state):
        while True:
            state = yield from self.explore(state)
"""
    patch_code = """
def explore(self, state):
    if state.char_at(state.y, state.x) == '>':
        state = yield Action.DOWN
    else:
        state = yield Action.WAIT
    return state

def combat(self, state):
    state = yield Action.WAIT
    return state
"""
    spliced, modified_fns = splice_functions_into_policy(base_policy, patch_code)
    assert set(modified_fns) == {"explore", "combat"}
    validate_python_code(spliced)
    assert "def combat(self, state):" in spliced
    assert "Action.DOWN" in spliced


def test_phase2_skill_splice_three_way_merge_and_symtable_gate():
    import time

    from lox.lang.compiler import _build_base_namespace, compile_policy_package
    from lox.lang.instrument import ExecutionTracer
    from lox.lang.validator import check_undefined_symbols

    # 1. merge_skill_module_update preserves top-level imports & constants
    base_skill = "from lox.env.actions import Action\nLIMIT = 3\ndef a(st):\n    return Action.WAIT\ndef b(st):\n    return None\n"
    patch_skill = (
        "from lox.env.actions import key\nLIMIT = 7\ndef b(st):\n    return key(' ')\n"
    )
    spliced = merge_skill_module_update(base_skill, patch_skill)
    assert "Action, key" in spliced or ("Action" in spliced and "key" in spliced)
    assert "LIMIT = 7" in spliced
    assert "def a(st):" in spliced
    assert "def b(st):" in spliced

    # 2. three_way_ast_merge handles disjoint top-level helpers cleanly
    base_pol = """
from lox.env.actions import Action

class Policy:
    def step(self, state):
        return Action.WAIT
"""
    pol_a = base_pol.replace(
        "class Policy:",
        "def leaf_a(s):\n    return Action.SEARCH\n\nclass Policy:",
    )
    pol_b = base_pol.replace(
        "class Policy:",
        "def leaf_b(s):\n    return Action.EAST\n\nclass Policy:",
    )
    merged = three_way_ast_merge(base_pol, pol_a, pol_b)
    assert merged is not None
    assert "def leaf_a" in merged and "def leaf_b" in merged
    # Same-function conflicting mutation must return None
    pol_a2 = pol_a.replace("Action.SEARCH", "Action.WEST")
    assert three_way_ast_merge(base_pol, pol_a, pol_a2) is None

    # 3. Fast symtable gate catches dormant NameError & missing skill export
    allowed = frozenset(_build_base_namespace(ExecutionTracer()).keys())
    t0 = time.perf_counter()
    check_undefined_symbols(merged, allowed_globals=allowed)
    assert (time.perf_counter() - t0) < 0.05

    bad_skill = {"sub": "def f(st):\n    return dormant_missing_helper(st)\n"}
    with pytest.raises(PolicyValidationError, match="dormant_missing_helper"):
        check_undefined_symbols(
            "class Policy:\n    def step(self, state):\n        return Action.WAIT\n",
            skills=bad_skill,
            allowed_globals=allowed,
        )

    good_skill = {"sub": "def existing_fn(st):\n    return Action.WAIT\n"}
    with pytest.raises(PolicyValidationError, match="missing_export"):
        check_undefined_symbols(
            "from skills.sub import missing_export\nclass Policy:\n    def step(self, state):\n        return Action.WAIT\n",
            skills=good_skill,
            allowed_globals=allowed,
        )

    with pytest.raises(PolicyValidationError):
        compile_policy_package(
            "class Policy:\n    def step(self, state):\n        return undefined_xyz(state)\n"
        )


def test_hierarchical_skill_package_compilation_and_router():
    combat_skill_code = """
import skills
from lox.env.actions import Action

def defend(self, state):
    \"\"\"Tactical combat routine.\"\"\"
    return Action.SEARCH
"""

    router_code = """
import skills
from lox.env.actions import Action

class Policy:
    def __init__(self):
        self.turns = 0

    def run(self, state):
        while True:
            self.turns += 1
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
    try:
        state = harness.reset(seed=42)
        act, fn_name = runner.next_action(state)
        assert act == Action.SEARCH
        assert (
            "defend" in runner.tracer.call_counts or "run" in runner.tracer.call_counts
        )
    finally:
        harness.close()


def test_skill_import_forms_and_transitive_skill_integration():
    helper_skill_code = """
from lox.env.actions import Action

def helper_search(state):
    return Action.SEARCH
"""
    combat_skill_code = """
from skills.helpers import helper_search

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

    skills = {"combat": combat_skill_code, "helpers": helper_skill_code}

    compiled_1 = compile_policy_package(
        policy_code=router_code_1,
        skills=skills,
    )
    runner_1 = compiled_1.create_runner()
    harness = NetHackHarness(max_steps=10)
    try:
        state = harness.reset(seed=42)
        act1, _ = runner_1.next_action(state)
        assert act1 == Action.SEARCH

        compiled_2 = compile_policy_package(
            policy_code=router_code_2,
            skills=skills,
        )
        runner_2 = compiled_2.create_runner()
        act2, _ = runner_2.next_action(state)
        assert act2 == Action.SEARCH
    finally:
        harness.close()


def test_bundle_ssot_unpack_skills_dir_and_imports(tmp_path):
    raw_verbatim = "def f():\n    return 1\n"
    pol_v, sk_v = unpack_code_bundle(raw_verbatim, preserve_raw_verbatim=True)
    assert pol_v == raw_verbatim
    assert sk_v == {}

    bundle_src = """<think>thinking</think>
<arg_key>code</arg_key><arg_value>ignore</arg_value>
<skill_code module="nav/core">
```python
import math
CONST = 1
def a():
    return 1
def c():
    return 3
```
</skill_code>
<skill_code module="nav/core">
def b():
    return 2
</skill_code>
### skill: survival.food
```python
def eat():
    return 3
```
<policy_code>
from skills import nav
from skills.survival import *
class Policy:
    def step(self, state):
        return Action.WAIT
</policy_code>
"""
    pol, skills = unpack_code_bundle(
        bundle_src, merge_fn=merge_skill_module_update, include_markdown_skills=True
    )
    assert "class Policy:" in pol
    assert "nav/core" in skills
    assert "def a()" in skills["nav/core"]
    assert "def b()" in skills["nav/core"]
    assert "survival/food" in skills

    s_dir = tmp_path / "skills"
    write_skills_dir(s_dir, skills, sync=True)
    loaded = load_skills_dir(s_dir)
    assert set(loaded.keys()) == {"nav/core", "survival/food"}
    assert (s_dir / "nav" / "__init__.py").exists()

    imp_info = extract_skill_imports(ast.parse(pol), set(loaded.keys()))
    assert "nav/core" in imp_info.referenced_keys
    assert "survival/food" in imp_info.referenced_keys
    assert extract_json_payload('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_skills_called_from_policy_and_traced():
    helper_skill = """
from lox.env.actions import Action

def skill_safe_search(state):
    \"\"\"Yields a search action and returns updated state.\"\"\"
    state = yield Action.SEARCH
    return state

def skill_is_low_hp(state, threshold=0.35):
    return (state.hp / max(1, state.max_hp)) < threshold
"""
    policy_code = """
from skills.helpers import skill_is_low_hp, skill_safe_search

class Policy:
    def run(self, state):
        while True:
            if skill_is_low_hp(state, 0.5):
                state = yield Action.WAIT
            else:
                state = yield from skill_safe_search(state)
"""
    compiled = compile_policy_package(
        policy_code=policy_code, skills={"helpers": helper_skill}
    )
    harness = NetHackHarness(max_steps=30)
    try:
        st = harness.reset(seed=77)
        runner = compiled.create_runner()
        for _ in range(5):
            act, active_fn = runner.next_action(st)
            st, done, _ = harness.step(act)
            if done:
                break
        assert runner.tracer.fn_calls.get("skill_is_low_hp", 0) >= 1
        assert runner.tracer.fn_calls.get("skill_safe_search", 0) >= 1
        assert runner.tracer.fn_turns.get("skill_safe_search", 0) >= 1
    finally:
        harness.close()


def test_merge_and_splice_dataclass_skill_no_corruption():
    from lox.lang.starter import build_clean_option_a_starter
    from lox.lang.validator import (
        run_pyright_on_package,
        splice_functions_into_policy,
    )

    existing = '''"""Explore goal."""

from dataclasses import dataclass
from lox.env.actions import Action


@dataclass
class ExploreState:
    target: tuple[int, int] | None = None


def _nearest_frontier(state, ws):
    return None


def _explore_trigger(state, ws):
    return True


def _explore_step(state, ws):
    t = _nearest_frontier(state, ws)
    return Action.SEARCH if t is None else Action.WAIT
'''
    full_rewrite = '''"""Explore goal (refactored)."""

from dataclasses import dataclass
from lox.env.actions import Action


@dataclass
class ExploreState:
    target: tuple[int, int] | None = None
    last_turn: int = 0


def _find_frontier(state, ws):
    return (int(state.y), int(state.x))


def _explore_trigger(state, ws):
    return True


def _explore_step(state, ws):
    t = _find_frontier(state, ws)
    return Action.SEARCH if t is None else Action.WAIT
'''
    merged = merge_skill_module_update(existing, full_rewrite)
    assert "_nearest_frontier" not in merged
    assert "def _find_frontier" in merged
    assert merged.count("def _explore_step") == 1

    patch = """# PATCH
def _pick_door(state):
    return None

def _explore_step(state, ws):
    return _pick_door(state) or Action.WAIT
"""
    spliced, changed = splice_functions_into_policy(existing, patch)
    assert "_pick_door" in changed
    assert spliced.splitlines()[0].startswith('"""') or spliced.splitlines()[
        0
    ].startswith("'")
    assert spliced.count("def _explore_step") == 1
    assert spliced.index("from lox.env.actions import Action") < spliced.index(
        "def _pick_door"
    )

    pol, base_skills = build_clean_option_a_starter()
    base_skills["goals/explore"] = full_rewrite
    pol2 = "import skills.goals.explore  # noqa: F401\n" + pol
    assert run_pyright_on_package(pol2, skills=base_skills) == []


def test_topological_skill_import_order_before_shared():
    skills = {
        "aaa_guard": (
            "from skills.equip import pick_action\n\n"
            "def guard_fn(state):\n"
            "    return pick_action(state)\n"
        ),
        "equip": (
            "from lox.env.actions import Action\n"
            "from skills.shared import _delta_of\n\n"
            "def pick_action(state):\n"
            "    return Action.SEARCH if _delta_of(Action.N) == (0, -1) else Action.WAIT\n"
        ),
        "shared": (
            "from lox.env.actions import DIR_DELTAS\n\n"
            "def _delta_of(act):\n"
            "    return DIR_DELTAS.get(act)\n"
        ),
    }
    policy_src = (
        "from lox.env.actions import Action\n"
        "from skills.aaa_guard import guard_fn\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return guard_fn(state) or Action.WAIT\n"
    )
    compiled = compile_policy_package(policy_src, "", skills=skills)
    assert compiled._skill_order == ["shared", "equip", "aaa_guard"]
    runner = compiled.create_runner()
    assert runner is not None
