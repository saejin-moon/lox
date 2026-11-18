"""Tests for Option-A 2-level architecture, hybrid side refactoring, telemetry/referee fixes, and E3 hybrid loop."""

from __future__ import annotations


def test_option_a_starter_bare_zero_heuristics_and_2level_packages():
    from lox.author.swarm_meta import SwarmMetaAgent
    from lox.env.harness import NetHackHarness
    from lox.lang.compiler import compile_policy_package
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
    kept = SwarmMetaAgent.prune_unused_skills(pol2, skills)
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


def test_ide_navigation_tools_read_symbol_ranged_skill_and_architecture_map(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db
    from lox.lang.starter import build_clean_option_a_starter

    dbp = str(tmp_path / "tools.duckdb")
    init_db(dbp).close()
    pol, skills = build_clean_option_a_starter()
    skills["goals/explore"] = (
        '"""Explore goal."""\n'
        "from lox.env.actions import Action\n"
        "import skills.goals.core as goals\n\n"
        "def _exp_step(state, ws):\n"
        "    mem = ws.domain('exp_state', dict)\n"
        "    return Action.SEARCH\n\n"
        "goals.register_goal('explore_floor', priority=10, trigger=lambda s, w: True, step=_exp_step)\n"
    )
    pol = "import skills.goals.explore  # noqa: F401\n" + pol

    reg = AuthorToolRegistry(db_path=dbp, macros_path=str(tmp_path / "macros.py"))
    reg.configure_generation(
        pool=None,
        baseline_runs=[],
        skills=skills,
        generation=1,
        mode="SWARM_PEER",
    )
    reg.current_policy_code = pol

    sliced, s_tag = reg.read_skill("goals/explore", start_line=4, end_line=7)
    assert "5: def _exp_step" in sliced
    assert "lines 4-7/" in s_tag

    sym_txt, _ = reg.read_symbol("goals/explore._exp_step")
    assert "skills/goals/explore.py:5-7" in sym_txt
    assert "def _exp_step" in sym_txt

    arch_txt, arch_sum = reg.architecture_map()
    assert "explore_floor" in arch_txt
    assert "exp_state" in arch_txt
    assert "1 goals, 1 domains" == arch_sum


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


def test_referee_closes_noop_and_significance_loopholes():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee, _is_wait_zombie_run

    # Zombie threshold lowered to 65%
    assert (
        _is_wait_zombie_run({"turns": 200, "pct_wait": 72.0, "unique_tiles": 30})
        is True
    )
    assert (
        _is_wait_zombie_run({"turns": 200, "pct_wait": 40.0, "unique_tiles": 30})
        is False
    )

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s): return Action.N\n"
    )

    class _NoisyPool:
        def evaluate_batch(self, batch_id, policy_code, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                runs = [
                    {
                        "run_id": f"{batch_id}_r{i}",
                        "seed": s,
                        "depth": 3 if i < 20 else 2,
                        "max_depth": 3 if i < 20 else 2,
                        "score": 400,
                        "turns": 1500,
                        "final_ac": 4,
                        "xp_level": 3,
                        "pct_wait": 10.0,
                        "end_reason": "killed by bat",
                    }
                    for i, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            if "peer_noop" in batch_id:
                # Exact 0W/0L/32T tie despite different source
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"{batch_id}_r{i}",
                            "seed": s,
                            "depth": 3 if i < 20 else 2,
                            "max_depth": 3 if i < 20 else 2,
                            "score": 400,
                            "turns": 1500,
                            "final_ac": 4,
                            "xp_level": 3,
                            "pct_wait": 10.0,
                            "end_reason": "killed by bat",
                        }
                        for i, s in enumerate(s_list)
                    ],
                }
            # peer_noisy: wins 4, loses 5 (should NOT promote under strict gate!)
            runs = []
            for i, s in enumerate(s_list):
                d = 3 if i < 20 else 2
                if i < 4:
                    d = 5
                elif 4 <= i < 9:
                    d = 1
                runs.append(
                    {
                        "run_id": f"{batch_id}_r{i}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": 405,
                        "turns": 1500,
                        "final_ac": 4,
                        "xp_level": 3,
                        "pct_wait": 10.0,
                        "end_reason": "killed by bat",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    c_noop = PeerCandidate(
        peer_id="peer_noop",
        claimed_deficit_id="d1",
        claimed_label="noop",
        policy_code=base_pol + "\n_UNUSED = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    c_noisy = PeerCandidate(
        peer_id="peer_noisy",
        claimed_deficit_id="d2",
        claimed_label="noisy",
        policy_code=base_pol + "\n_ACTIVE = 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    seeds = [20000 + i for i in range(32)]
    verdict = SwarmReferee().evaluate_tournament(
        generation=10,
        candidates=[c_noop, c_noisy],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_NoisyPool(),
        seeds=seeds,
    )
    assert verdict.promoted is False
    # Behavioral 0W/0L no-op must never outrank active peer_noisy
    assert verdict.winner_id in ("incumbent", "peer_noisy")


def test_deficit_analyzer_includes_shortest_early_deaths(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.db.schema import connect_db, init_db

    dbp = str(tmp_path / "strat.duckdb")
    init_db(dbp).close()
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b_g1', 1, 'SWARM', TRUE)"
        )
        for i in range(64):
            # 16 ultra-short early deaths (turns=80..110) and 48 longer runs (turns=4000..4048)
            turns = 80 + i if i < 16 else 4000 + i
            er = "pray smite" if i < 16 else "max_steps"
            d = 1 if i < 16 else 3
            con.execute(
                """
                INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason)
                VALUES (?, 'b_g1', 1, ?, ?, ?, 100, ?, ?, ?)
                """,
                [f"r_{i}", 10100 + i, d, d, turns, turns, er],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(generation=2)
    hist_reasons = {er for er, _cnt, _ad in rep.end_reason_histogram}
    assert "pray smite" in hist_reasons
    assert any(c.cluster_id == "slice_early_terminations" for c in rep.fail_clusters)


def test_pool_preserves_last_depth_on_terminal_death():
    from lox.runner.pool import run_single_episode

    suicidal_pol = (
        "from lox.env.actions import Action, extcmd, key\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        if state.in_yn_prompt:\n"
        "            return key('y')\n"
        "        return extcmd('pray')\n"
    )
    res = run_single_episode(
        {
            "seed": 12304,
            "run_id": "test_death_depth",
            "role": "val-hum-law-fem",
            "max_steps": 120,
            "policy_code": suicidal_pol,
            "macros_code": "",
            "skills": {},
            "record_ticks": False,
        }
    )
    assert res["depth"] >= 1
    assert res["max_depth"] >= 1
