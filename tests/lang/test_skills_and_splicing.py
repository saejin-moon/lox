"""Domain-scoped tests for LOX."""

from __future__ import annotations

import ast

import pytest

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.tools import AuthorToolRegistry, merge_skill_module_update
from lox.db.schema import init_db
from lox.env import (
    NetHackHarness,
)
from lox.env.actions import Action
from lox.lang.bundle import (
    extract_json_payload,
    extract_skill_imports,
    load_skills_dir,
    unpack_code_bundle,
    write_skills_dir,
)
from lox.lang.compiler import compile_policy_and_macros, compile_policy_package
from lox.lang.macros import MacroLibrary
from lox.lang.validator import (
    PolicyValidationError,
    check_macro_contracts,
    splice_functions_into_policy,
    three_way_ast_merge,
    validate_python_code,
)


def test_function_splicing_and_macro_library(tmp_path):
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

    # Test MacroLibrary
    macros_file = tmp_path / "macros.py"
    lib = MacroLibrary(path=str(macros_file))
    saved = lib.upsert_macros(
        """
def macro_engrave_elbereth(state):
    \"\"\"Engraves Elbereth in dust with fingers.\"\"\"
    yield seq(Action.ENGRAVE, "-", "Elbereth\\r")
"""
    )
    assert saved == ["macro_engrave_elbereth"]
    docs = lib.describe_macros()
    assert "macro_engrave_elbereth(state)" in docs
    assert "Engraves Elbereth" in docs


def test_macro_contract_linting():

    macros_code = """
def find_corridors(state):
    return state.chars == ord('#')

def kick_door(state, direction):
    yield seq(Action.KICK, direction)
"""

    # 1. Valid policy call
    valid_policy = """
class Policy:
    def run(self, state):
        mask = macros.find_corridors(state)
        yield from macros.kick_door(state, Action.EAST)
"""
    errors = check_macro_contracts(valid_policy, macros_code)
    assert errors == []

    # 2. Too few arguments
    too_few_args = """
class Policy:
    def run(self, state):
        mask = macros.find_corridors()
"""
    errs = check_macro_contracts(too_few_args, macros_code)
    assert len(errs) == 1
    assert "requires at least 1 argument" in errs[0]

    # 3. Direct yield instead of yield from for generator macro
    bad_yield = """
class Policy:
    def run(self, state):
        state = yield macros.kick_door(state, Action.NORTH)
"""
    errs = check_macro_contracts(bad_yield, macros_code)
    assert any("is a generator. Use 'yield from" in e for e in errs)


def test_bottom_up_macro_harvester(tmp_path):
    from lox.lang.macros import MacroHarvester, MacroLibrary

    macros_file = str(tmp_path / "macros.py")
    ml = MacroLibrary(path=macros_file)

    champion_code = """
class Policy:
    def __init__(self):
        self.step_count = 0

    def find_stairs_down(self, state):
        '''Locates downstairs coordinates.'''
        stairs = np.argwhere(state.chars == ord('>'))
        return tuple(stairs[0]) if len(stairs) > 0 else None

    def stateful_method(self, state):
        self.step_count += 1
        return self.step_count

    def run(self, state):
        while True:
            target = self.find_stairs_down(state)
            state = yield Action.SEARCH
"""
    fn_stats = [
        {"function_name": "find_stairs_down", "ochiai_good_run_score": 0.45},
        {"function_name": "stateful_method", "ochiai_good_run_score": 0.50},
    ]

    harvested = MacroHarvester.harvest_from_policy(
        policy_code=champion_code,
        macro_lib=ml,
        fn_stats=fn_stats,
    )

    assert harvested == ["find_stairs_down"]
    saved_code = ml.load_code()
    assert "def find_stairs_down(state):" in saved_code
    assert "Locates downstairs coordinates." in saved_code
    assert "stateful_method" not in saved_code
    assert "run" not in saved_code


def test_phase2_skill_splice_three_way_merge_and_symtable_gate():
    import time

    from lox.author.tools import merge_skill_module_update
    from lox.lang.compiler import _build_base_namespace, compile_policy_package
    from lox.lang.instrument import ExecutionTracer
    from lox.lang.validator import (
        check_undefined_symbols,
    )

    # 1. splice_functions_into_policy preserves top-level imports & constants when target_class is None
    base_skill = "from lox.env.actions import Action\nLIMIT = 3\ndef a(st):\n    return Action.WAIT\ndef b(st):\n    return None\n"
    patch_skill = (
        "from lox.env.actions import key\nLIMIT = 7\ndef b(st):\n    return key(' ')\n"
    )
    spliced = merge_skill_module_update(base_skill, patch_skill)
    assert "Action, key" in spliced or ("Action" in spliced and "key" in spliced)
    assert "LIMIT = 7" in spliced
    assert "def a(st):" in spliced
    assert "def b(st):" in spliced

    # 2. three_way_ast_merge handles disjoint helpers + PrioritySelector children
    base_pol = """
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector
from lox.env.actions import Action

class Policy:
    def __init__(self):
        self.bb = Blackboard()
        self.tree = self.build_tree()

    def build_tree(self):
        return PrioritySelector([ActionLeaf(lambda s: Action.WAIT, name='w')])

    def step(self, state):
        return self.tree.tick(state) or Action.WAIT
"""
    pol_a = base_pol.replace(
        "class Policy:",
        "def leaf_a(s):\n    return Action.SEARCH\n\nclass Policy:",
    ).replace(
        "ActionLeaf(lambda s: Action.WAIT, name='w')",
        "ActionLeaf(leaf_a, name='a'), ActionLeaf(lambda s: Action.WAIT, name='w')",
    )
    pol_b = base_pol.replace(
        "class Policy:",
        "def leaf_b(s):\n    return Action.EAST\n\nclass Policy:",
    ).replace(
        "ActionLeaf(lambda s: Action.WAIT, name='w')",
        "ActionLeaf(lambda s: Action.WAIT, name='w'), ActionLeaf(leaf_b, name='b')",
    )
    merged = three_way_ast_merge(base_pol, pol_a, pol_b)
    assert merged is not None
    assert "def leaf_a" in merged and "def leaf_b" in merged
    assert "leaf_a" in merged and "leaf_b" in merged
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

    # Also verify compile_policy_package invokes check_undefined_symbols automatically
    with pytest.raises(PolicyValidationError):
        compile_policy_package(
            "class Policy:\n    def step(self, state):\n        return undefined_xyz(state)\n"
        )


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


def test_macros_called_from_policy_and_traced():
    macros_code = """
def macro_safe_search(state):
    \"\"\"Yields a search action and returns updated state.\"\"\"
    state = yield Action.SEARCH
    return state

def macro_is_low_hp(state, threshold=0.35):
    return (state.hp / max(1, state.max_hp)) < threshold
"""
    policy_code = """
class Policy:
    def run(self, state):
        while True:
            if macro_is_low_hp(state, 0.5):
                state = yield Action.WAIT
            else:
                state = yield from macro_safe_search(state)
"""
    compiled = compile_policy_and_macros(
        policy_code=policy_code, macros_code=macros_code
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
        assert runner.tracer.fn_calls.get("macro_is_low_hp", 0) >= 1
        assert runner.tracer.fn_calls.get("macro_safe_search", 0) >= 1
        assert runner.tracer.fn_turns.get("macro_safe_search", 0) >= 1
    finally:
        harness.close()


def test_real_state_macro_verification(tmp_path):

    tools = AuthorToolRegistry(db_path=str(tmp_path / "test.duckdb"))

    # 1. Valid macro should execute on real states and return PASSED (macro)
    valid_macro = """
def get_walkable_mask(state):
    return state.chars == ord('.')

def step_wait(state):
    yield Action.WAIT

def is_safe_corpse(obj):
    return obj.is_corpse and 'poison' not in obj.name.lower()

def execute_move(state, dy, dx):
    target_y = state.y + dy
    return DIR_DELTAS.get((dy, dx), Action.WAIT)

def conditional_generator(state):
    for o in state.visible_objects():
        if o.is_corpse:
            yield seq(Action.EAT, 'y')
"""
    msg, status = tools.dry_run_policy(code=valid_macro, is_macro=True)
    assert status == "PASSED (macro)"
    assert "Real-State Macro Verification PASSED" in msg

    # 2. Broken macro with invalid attribute lookup should fail with RUNTIME_ERROR
    broken_macro = """
def bad_macro(state):
    return state.non_existent_attribute_xyz
"""
    msg, status = tools.dry_run_policy(code=broken_macro, is_macro=True)
    assert status == "RUNTIME_ERROR"
    assert "non_existent_attribute_xyz" in msg

    # 3. Invalid generator yield should be caught as INVALID_YIELD
    bad_generator = """
def bad_gen(state):
    yield 123456789  # Not a valid Action / sequence
"""
    msg, status = tools.dry_run_policy(code=bad_generator, is_macro=True)
    assert status == "INVALID_YIELD"


def test_dynamic_macro_category_discovery(tmp_path):
    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_dyn.duckdb")
    con = init_db(db_path)
    con.close()

    meta = SwarmMetaAgent(
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
    )
    categories = meta.discover_macro_categories(generation=1, n_categories=4)
    assert len(categories) >= 3
    for cat in categories:
        assert "category_id" in cat
        assert "name" in cat
        assert "target_depth_tier" in cat
        assert "target_metrics" in cat

    # Verify category decomposition stored in LanceDB
    docs = meta.theories.search_theories("decomposition", limit=5)
    assert len(docs) >= 1
    assert docs[0]["doc_type"] == "CATEGORY_DECOMPOSITION"


def test_merge_and_splice_dataclass_skill_no_corruption():
    from lox.author.tools import merge_skill_module_update
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

    # Partial patch onto @dataclass skill must not insert new defs before docstring/imports
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
    from lox.lang.compiler import compile_bt_subtree, compile_policy_package

    skills = {
        "aaa_guard": (
            "from skills.equip import pick_action\n\n"
            "def guard_fn(state, bb):\n"
            "    return pick_action(state, bb)\n"
        ),
        "equip": (
            "from lox.env.actions import Action\n"
            "from skills.shared import _delta_of\n\n"
            "def pick_action(state, bb):\n"
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
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n"
        "from skills.aaa_guard import guard_fn\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(\n"
        "            PrioritySelector([ActionLeaf(guard_fn, name='g')]),\n"
        "            blackboard=self.blackboard,\n"
        "        )\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    compiled = compile_policy_package(policy_src, "", skills=skills)
    assert compiled._skill_order == ["shared", "equip", "aaa_guard"]
    runner = compiled.create_runner()
    assert runner is not None

    subtree = compile_bt_subtree(
        "from lox.lang.bt import ActionLeaf\n"
        "from skills.aaa_guard import guard_fn\n"
        "subtree = ActionLeaf(guard_fn, name='g_sub')\n",
        skills=skills,
    )
    assert subtree is not None
