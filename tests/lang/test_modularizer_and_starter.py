"""Domain-scoped tests for LOX."""

from __future__ import annotations

import ast

from lox.author.directives import prune_unused_skills
from lox.db.schema import connect_db
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_package
from lox.lang.modularize import _modularize_skill_module


def test_option_a_starter_bare_zero_heuristics_and_2level_packages():
    from lox.lang.starter import build_clean_option_a_starter
    from lox.lang.validator import (
        normalize_skill_key,
        run_pyright_on_package,
        validate_cross_module_contracts,
    )

    pol, skills = build_clean_option_a_starter()
    assert "PrioritySelector" not in pol
    assert "Blackboard" not in pol
    assert set(skills.keys()) == {
        "world",
        "prompts",
        "survival/core",
        "combat/core",
        "navigation/core",
        "goals/core",
    }
    # Zero hardcoded monster/corpse/item lists
    combined = pol + "\n".join(skills.values())
    for forbidden in ("floating eye", "cockatrice", "food ration", "newt", "jackal"):
        assert forbidden not in combined.lower()

    validate_cross_module_contracts(pol, skills)
    assert run_pyright_on_package(pol, skills=skills) == []
    compiled = compile_policy_package(pol, "", skills=skills)
    runner = compiled.create_runner()
    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=42)
        for _ in range(15):
            act, fn = runner.next_action(st)
            st, done, _ = h.step(act)
            assert fn in ("fallback_step", "handle_prompts")
            if done:
                break
    finally:
        h.close()

    # Add a 2-level plugin goal + domain sub-state and verify Pyright, pruning, and attribution
    skills["goals/explore"] = '''"""Registered exploration stub goal."""

from lox.env.actions import Action
import skills.goals.core as goals


def _trig(state, ws):
    st = ws.domain("explore_mem", dict)
    st["seen"] = st.get("seen", 0) + 1
    return True


def _step(state, ws):
    return Action.SEARCH


goals.register_goal("explore_stub", priority=10, trigger=_trig, step=_step)
'''
    pol2 = "import skills.goals.explore  # noqa: F401\n" + pol
    assert normalize_skill_key("skills.goals.explore") == "goals/explore"
    validate_cross_module_contracts(pol2, skills)
    assert run_pyright_on_package(pol2, skills=skills) == []
    kept = prune_unused_skills(pol2, skills)
    assert "goals/explore" in kept

    compiled2 = compile_policy_package(pol2, "", skills=skills)
    runner2 = compiled2.create_runner()
    h2 = NetHackHarness(max_steps=40)
    try:
        st2 = h2.reset(seed=42)
        _act, active_fn = runner2.next_action(st2)
        assert active_fn == "explore_stub"
    finally:
        h2.close()


def test_starter_navigation_step_toward_returns_action():
    from lox.lang.starter import build_clean_option_a_starter

    _, skills = build_clean_option_a_starter()
    world_ns: dict = {}
    nav_ns: dict = {}
    exec(compile(skills["world"], "<world>", "exec"), world_ns)
    exec(compile(skills["navigation/core"], "<navigation/core>", "exec"), nav_ns)

    h = NetHackHarness(max_steps=20)
    try:
        st = h.reset(seed=42)
        ws = world_ns["WorldState"]()
        ws.observe(st)
        # Find an adjacent passable tile from GameState.passable_mask
        target = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = st.y + dy, st.x + dx
                if 0 <= ny < 21 and 0 <= nx < 79 and ws.passable_mask[ny, nx]:
                    target = (ny, nx)
                    break
            if target is not None:
                break
        assert target is not None
        act = nav_ns["step_toward"](st, ws, target)
        assert isinstance(act, (Action, int))
        assert act is not None
    finally:
        h.close()


def test_hybrid_side_cohesion_refactor_and_clean_split(tmp_path):
    from lox.lang.modularize import run_side_cohesion_refactor
    from lox.lang.starter import build_clean_option_a_starter

    pol, skills = build_clean_option_a_starter()
    pad = "\n".join(["    v = 1" for _ in range(245)])
    skills["survival/core"] = (
        '"""Survival domain."""\n'
        "from lox.env.actions import Action\n\n"
        "def _pray_ok(state, ws):\n"
        "    return False\n\n"
        "def _pray_timer(state):\n"
        f"{pad}\n"
        "    return True\n\n"
        "def pray_step(state, ws):\n"
        "    if _pray_ok(state, ws) and _pray_timer(state):\n"
        "        return Action.PRAY\n"
        "    return None\n\n"
        "def _eat_ok(state):\n"
        f"{pad}\n"
        "    return False\n\n"
        "def check_interrupt(state, ws):\n"
        "    if _eat_ok(state):\n"
        "        return Action.EAT\n"
        "    return None\n"
    )
    pol = pol.replace(
        "import skills.goals.core as goals",
        "import skills.goals.core as goals\nfrom skills.survival.core import pray_step",
    )

    def _mock_plan(sk, outline, existing):
        assert sk == "survival/core"
        assert any(item["name"] == "pray_step" for item in outline)
        return {
            "target_submodule": "survival/prayer",
            "move_symbols": ["_pray_ok", "_pray_timer", "pray_step"],
        }

    changed, new_pol, new_sk, msg = run_side_cohesion_refactor(
        pol,
        skills,
        max_loc=500,
        split_plan_fn=_mock_plan,
        verify_seeds=[42, 43],
        verify_steps=40,
    )
    assert changed is True, msg
    assert "survival/prayer" in new_sk
    assert "from skills.survival.prayer import pray_step" in new_pol
    assert "def pray_step" not in new_sk["survival/core"]


def test_ast_modularizer_hysteresis_incremental_and_orchestrator(tmp_path):
    import json

    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.engine.swarm_referee import _package_fingerprint
    from lox.lang.modularize import (
        maybe_auto_modularize,
        modularize_policy_source,
        verify_bit_identical,
    )

    monolith = (
        "from lox.env.actions import Action\n\n"
        "SHARED_CONST = 1\n\n"
        "def _shared_ok(state):\n"
        "    return state.hp >= SHARED_CONST\n\n"
        "def leaf_one(state):\n"
        "    if not _shared_ok(state):\n"
        "        return Action.WAIT\n"
        "    return None\n\n"
        "def leaf_two(state):\n"
        "    if _shared_ok(state) and state.step_count % 2 == 0:\n"
        "        return Action.SEARCH\n"
        "    return None\n\n"
        "def leaf_three(state):\n"
        "    return Action.WAIT if _shared_ok(state) else Action.SEARCH\n\n"
        "def dispatch_action(state):\n"
        "    for fn in (leaf_one, leaf_two, leaf_three):\n"
        "        act = fn(state)\n"
        "        if act is not None:\n"
        "            return act\n"
        "    return Action.WAIT\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return dispatch_action(state)\n"
    )
    # 1. Fresh monolith split + bit-identical check
    new_pol, new_skills = modularize_policy_source(monolith, num_modules=2)
    assert len(new_pol.splitlines()) < len(monolith.splitlines())
    assert "shared" in new_skills
    assert len(new_skills) >= 2
    verify_bit_identical(
        monolith, {}, new_pol, new_skills, seeds=[1000, 1001], steps=50
    )

    # 2. Below threshold hysteresis check skips splitting
    ch_skip, _, _, _ = maybe_auto_modularize(
        monolith, {}, verify_seeds=[1000], verify_steps=50
    )
    assert ch_skip is False

    # 3. Incremental split when existing `skills/` + imports are already present
    augmented = new_pol.replace(
        "class Policy:",
        "EXTRA_CONST = 2\n\n"
        "def _extra_helper(state):\n    return state.step_count >= EXTRA_CONST\n\n"
        "def leaf_four(state):\n    return Action.SEARCH if _extra_helper(state) else None\n\n"
        "def leaf_five(state):\n    return Action.WAIT if _extra_helper(state) else None\n\n"
        "class Policy:",
    ).replace(
        "return dispatch_action(state)",
        "return leaf_four(state) or leaf_five(state) or dispatch_action(state)",
    )
    inc_pol, inc_skills = modularize_policy_source(
        augmented, num_modules=2, existing_skills=new_skills
    )
    assert set(new_skills.keys()).issubset(set(inc_skills.keys()))
    assert inc_skills["shared"] == new_skills["shared"]
    assert any("leaf_four" in v for v in inc_skills.values())
    assert any("leaf_five" in v for v in inc_skills.values())
    verify_bit_identical(
        augmented, new_skills, inc_pol, inc_skills, seeds=[1000, 1001], steps=50
    )

    # 4. SwarmCampaignOrchestrator auto-modularizes a bloated monolith in run_generation
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    padded_helpers = "\n".join(
        f"def _noop_helper_{i}(state):\n    return state.hp > {i}\n" for i in range(135)
    )
    bloated_monolith = monolith.replace(
        "def dispatch_action(state):",
        f"{padded_helpers}\n\ndef dispatch_action(state):",
    )
    (ws_dir / "policy.py").write_text(bloated_monolith, encoding="utf-8")

    db_path = str(tmp_path / "swarm_mod.duckdb")
    orch = SwarmCampaignOrchestrator(
        db_path=db_path,
        workspace_dir=str(ws_dir),
        frontier_dir=str(tmp_path / "frontier"),
        provider="mock",
        num_workers=2,
        eval_workers=2,
        seeds_per_batch=2,
    )
    try:
        res = orch.run_generation(1)
        # Pre-gen side cohesion refactor writes modular files without falsely forcing promoted=True!
        assert res["promoted"] is False
        written_pol = (ws_dir / "policy.py").read_text(encoding="utf-8")
        assert len(written_pol.splitlines()) < 60
        skill_files = {
            str(p.relative_to(ws_dir / "skills"))
            .removesuffix(".py")
            .replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in (ws_dir / "skills").rglob("*.py")
            if p.name != "__init__.py"
        }
        assert "shared" in skill_files and len(skill_files) >= 2
        con = connect_db(db_path, read_only=True)
        try:
            row = con.execute(
                "SELECT skills_json FROM policies WHERE generation=1"
            ).fetchone()
            assert row is not None and len(json.loads(row[0])) == len(skill_files)
        finally:
            con.close()
        fp = _package_fingerprint(written_pol, skill_files)
        assert fp in orch._attempted_ast_modular_fps
    finally:
        orch.close()


def test_modularize_two_level_skill_dotted_imports():
    big_skill = "\n".join(
        [
            "SHARED_CONST = 7",
            "def _helper(x): return x + SHARED_CONST",
            "def alpha(s): return _helper(1)",
            "def beta(s): return _helper(2)",
            "def gamma(s): return _helper(3)",
            "def delta(s): return _helper(4)",
        ]
        + [
            f"def pad_{i}(x):\n    y = _helper(x)\n    return y + {i}\n"
            for i in range(45)
        ]
    )
    res = _modularize_skill_module("nav/core", big_skill, reserved_names=set())
    assert res is not None
    facade, submods = res
    assert "nav/core_shared" in submods
    assert "from skills.nav.core_shared import" in facade
    assert "/" not in "\n".join(facade.splitlines()[1:])
    ast.parse(facade)
    for k, src in submods.items():
        assert "/" not in "\n".join(src.splitlines()[1:])
        ast.parse(src)
