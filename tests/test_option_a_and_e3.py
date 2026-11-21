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
    from lox.engine.swarm_referee import SwarmReferee
    from lox.env.nethack.adapter import NetHackAdapter

    # All-depth horizon waste replaces the Goodharted pct_wait zombie heuristic
    adapter = NetHackAdapter()
    assert (
        adapter.is_horizon_waste(
            {"depth": 4, "end_reason": "max_steps", "turns": 10000}
        )
        is True
    )
    assert (
        adapter.is_horizon_waste({"depth": 2, "end_reason": "stalled_zero_turn"})
        is True
    )
    assert (
        adapter.is_horizon_waste({"depth": 4, "end_reason": "killed by gnome"}) is False
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


def test_starter_navigation_step_toward_returns_action():
    from lox.env.actions import Action
    from lox.env.harness import NetHackHarness
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


def test_read_symbol_resolves_skills_dotted_class_method(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db
    from lox.lang.starter import build_clean_option_a_starter

    dbp = str(tmp_path / "tools2.duckdb")
    init_db(dbp).close()
    pol, skills = build_clean_option_a_starter()
    reg = AuthorToolRegistry(db_path=dbp, macros_path=str(tmp_path / "macros.py"))
    reg.configure_generation(
        pool=None,
        baseline_runs=[],
        skills=skills,
        generation=1,
        mode="SWARM_PEER",
    )
    reg.current_policy_code = pol

    txt1, tag1 = reg.read_symbol("skills.goals.core.GoalRunner.tick")
    assert tag1 != "not_found"
    assert "GoalRunner.tick" in txt1
    assert "def tick(" in txt1

    txt2, tag2 = reg.read_symbol("goals/core.GoalRunner.tick")
    assert tag2 != "not_found"
    assert "def tick(" in txt2

    txt3, tag3 = reg.read_symbol("skills.navigation.core.step_toward")
    assert tag3 != "not_found"
    assert "def step_toward(" in txt3


def test_outcome_clustering_strips_fn_blame_and_pareto_gate_blocks_one_seed_noise(
    tmp_path,
):
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.engine.swarm_referee import SwarmReferee

    dbp = str(tmp_path / "outcome_split.duckdb")
    init_db(dbp).close()
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b_g5', 5, 'SWARM', TRUE)"
        )
        # All stalls share diverse last_active_function values, but genuinely split on objective outcome:
        # prompt_interaction vs zero_turn_dlvl2plus vs zero_turn_dlvl1
        for i in range(16):
            fn = ("combat_threat", "handle_prompts", "survival_interrupt")[i % 3]
            if i < 6:
                er, md = "Really attack the dwarf? [yn] (n)", 2
            elif i < 11:
                er, md = "stalled_zero_turn", 3
            else:
                er, md = "stalled_zero_turn", 1
            con.execute(
                """
                INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, last_active_function)
                VALUES (?, 'b_g5', 5, ?, ?, ?, 150, 500, 500, ?, ?)
                """,
                [f"r_{i}", 10500 + i, md, md, er, fn],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(generation=6)
    cids = {c.cluster_id for c in rep.fail_clusters}
    # Must sub-split by objective outcome signature, NOT by last_active_function
    assert not any(
        cid.endswith(("_combat_threat", "_handle_prompts", "_survival_interrupt"))
        for cid in cids
    ), cids
    assert any("__" in cid for cid in cids), cids
    brief = rep.format_brief()
    assert ", fns=" not in brief

    # 2. pareto_beats on 64 seeds must NOT promote a 33W-31L (+2 net) noisy AC sidegrade,
    #    and must persist iqm_delta / wilcoxon_p / wins / losses / reject_reason + preserve holdout_rejected
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s): return Action.N\n"
    )
    seeds64 = [21000 + i for i in range(64)]

    class _NoisyParetoPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 400,
                            "turns": 1500,
                            "final_ac": 5,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            # Exact G10-like noisy Pareto scenario:
            # 14 big wins (d=5, ac=4, sc=410) + 14 tiny wins (d=3, ac=4, sc=401) = 28 wins,
            # 14 big losses (d=2, ac=5, sc=390) + 13 med losses (d=2, ac=4, sc=405) = 27 losses, 9 ties (d=3, ac=5, sc=400)
            # -> avg_depth=3.016, avg_score=401.2 (>400), avg_ac=4.36 (improved!), wins=28, losses=27 (margin 1 < 4), p > 0.20
            runs = []
            for idx, s in enumerate(s_list):
                if idx < 14:
                    d, ac, sc = 5, 4, 410
                elif idx < 28:
                    d, ac, sc = 3, 4, 401
                elif idx < 42:
                    d, ac, sc = 2, 5, 390
                elif idx < 55:
                    d, ac, sc = 2, 4, 405
                else:
                    d, ac, sc = 3, 5, 400
                runs.append(
                    {
                        "run_id": f"p_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1500,
                        "final_ac": ac,
                        "xp_level": 3,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    c_noisy = PeerCandidate(
        peer_id="peer_noisy_pareto",
        claimed_deficit_id="slice_stalled",
        claimed_label="noisy",
        policy_code=base_pol + "\n_X = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    v_noise = SwarmReferee().evaluate_tournament(
        generation=11,
        candidates=[c_noisy],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_NoisyParetoPool(),
        seeds=seeds64,
    )
    assert v_noise.promoted is False
    att = v_noise.attempts_summary[0]
    assert att["wins"] > att["losses"] and (att["wins"] - att["losses"]) < 4
    assert att["reject_reason"] in ("cvar25_tail_regression", "below_sign_perm_alpha")
    BatchDatabaseLogger(db_path=dbp).save_tournament_attempts(v_noise.attempts_summary)
    con2 = connect_db(dbp, read_only=True)
    try:
        row = con2.execute(
            "SELECT wins, losses, reject_reason FROM tournament_attempts WHERE peer_id='peer_noisy_pareto'"
        ).fetchone()
        assert row == (att["wins"], att["losses"], att["reject_reason"])
    finally:
        con2.close()


def test_climbing_regime_does_not_throttle_peers_to_two(tmp_path, monkeypatch):
    from unittest.mock import patch

    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.schema import init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator

    dbp = str(tmp_path / "climb.duckdb")
    init_db(dbp).close()
    ws = tmp_path / "ws"
    fr = tmp_path / "fr"
    orch = SwarmCampaignOrchestrator(
        db_path=dbp,
        workspace_dir=str(ws),
        frontier_dir=str(fr),
        provider="mock",
        num_workers=4,
        seeds_per_batch=4,
        race_seeds=2,
    )
    seen_peers: list[str] = []

    class _FakePool:
        benchmark = "nethack"
        max_steps = 100

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 100,
                        "turns": 100,
                        "final_ac": 6,
                        "xp_level": 2,
                        "end_reason": "killed",
                    }
                    for s in (priority_seeds or [1])
                ],
            }

        def close(self):
            pass

    orch.eval_pool = _FakePool()

    def _fake_analyze(self, generation, current_policy="", skills=None):
        return DeficitReport(
            generation=generation,
            regime="CLIMBING",
            fail_clusters=[
                DeficitCluster("c1", "k1", "l1", run_count=5, sample_seeds=[100]),
                DeficitCluster("c2", "k2", "l2", run_count=5, sample_seeds=[101]),
            ],
        )

    def _fake_synth(
        self,
        generation,
        current_policy,
        current_skills,
        deficit_report,
        claimed_deficit,
        **kwargs,
    ):
        seen_peers.append(self.peer_id)
        return PeerCandidate(
            peer_id=self.peer_id,
            claimed_deficit_id=claimed_deficit.cluster_id,
            claimed_label=claimed_deficit.label,
            policy_code=current_policy,
            skills=dict(current_skills),
            sandbox_passed=False,
            error="stop",
        )

    try:
        with (
            patch("lox.author.swarm_deficits.DeficitAnalyzer.analyze", _fake_analyze),
            patch(
                "lox.author.swarm_peer.SwarmPeerAgent.run_cycle",
                _fake_synth,
            ),
        ):
            orch.run_generation(6)
    finally:
        orch.close()

    assert len(seen_peers) == 4, seen_peers


def test_g11_g12_surgical_zero_loss_and_pareto_depth_score_promotion():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s): return Action.N\n"
    )
    select32 = [131000 + i for i in range(32)]
    confirm64 = [1031000 + i for i in range(64)]

    # 1. Genuine surgical fix: 1W-0L-31T on S_select (advances ahead of net-losing 11W-12L churn!)
    #    and 5W-0L-59T on S_confirm (exact sign-perm p = 1/32 = 0.03125 < 0.05; OBF Look-1 p=0.03125
    #    triggers +64 OBF or 6W-0L passes Look-1 p=0.0156 < 0.030 immediately!).
    class _SurgicalZeroLossPool:
        def __init__(self, confirm_wins: int = 6):
            self.confirm_wins = confirm_wins

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 1500,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            if "peer_surgical" in batch_id:
                is_sel = "_sel_" in batch_id
                n_w = 1 if is_sel else self.confirm_wins
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"sg_{s}",
                            "seed": s,
                            "depth": 8 if idx < n_w else 3,
                            "max_depth": 8 if idx < n_w else 3,
                            "score": 2100 if idx < n_w else 500,
                            "turns": 1500,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for idx, s in enumerate(s_list)
                    ],
                }
            # Noisy 11W-12L churn candidate
            runs = []
            for idx, s in enumerate(s_list):
                d = 5 if idx < 11 else (1 if idx < 23 else 3)
                runs.append(
                    {
                        "run_id": f"ch_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": 520,
                        "turns": 1500,
                        "final_ac": 4,
                        "xp_level": 3,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    c_surg = PeerCandidate(
        peer_id="peer_surgical",
        claimed_deficit_id="slice_stalled",
        claimed_label="surgical",
        policy_code=base_pol + "\n_S = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    c_churn1 = PeerCandidate(
        peer_id="peer_churn1",
        claimed_deficit_id="slice_early_terminations",
        claimed_label="churn1",
        policy_code=base_pol + "\n_C1 = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    c_churn2 = PeerCandidate(
        peer_id="peer_churn2",
        claimed_deficit_id="slice_frontier_failures",
        claimed_label="churn2",
        policy_code=base_pol + "\n_C2 = 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    v_g12 = SwarmReferee().evaluate_tournament(
        generation=12,
        candidates=[c_churn1, c_churn2, c_surg],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_SurgicalZeroLossPool(confirm_wins=6),
        select_seeds=select32,
        seeds=confirm64,
    )
    assert v_g12.promoted is True, v_g12.reason
    assert v_g12.winner_id == "peer_surgical"

    # A 1W-0L fluke on S_confirm (p = 0.50) must NOT promote under v2!
    v_fluke = SwarmReferee().evaluate_tournament(
        generation=12,
        candidates=[c_surg],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_SurgicalZeroLossPool(confirm_wins=1),
        select_seeds=select32,
        seeds=confirm64,
    )
    assert v_fluke.promoted is False

    # 2. Old G11 pareto_axis_gain loophole (14W-10L-40T with tail collapse, p > 0.05) is now BLOCKED!
    class _G11ParetoPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 585,
                            "turns": 1500,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            runs = []
            for idx, s in enumerate(s_list):
                if idx < 14:
                    d, sc = (5 if idx < 3 else 4), 950
                elif idx < 24:
                    d, sc = 2, 480
                else:
                    d, sc = 3, 585
                runs.append(
                    {
                        "run_id": f"g11_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1500,
                        "final_ac": 4,
                        "xp_level": 3,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    c_g11 = PeerCandidate(
        peer_id="ortho_peer_2_peer_3",
        claimed_deficit_id="slice_stalled+slice_early_terminations",
        claimed_label="g11_ortho",
        policy_code=base_pol + "\n_G11 = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    v_g11 = SwarmReferee().evaluate_tournament(
        generation=11,
        candidates=[c_g11],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_G11ParetoPool(),
        seeds=confirm64,
    )
    assert v_g11.promoted is False, (
        "v2 deletes pareto_beats loophole on high-p/tail-risk candidates"
    )


def test_option_a_peer_prompt_micro_pilot_research_gate_and_search_slice(tmp_path):
    """Verifies G16-G25 plateau breakthrough: clean Option-A peer prompt, paired CRN micro-pilot diff,
    1-lookup empirical research gate, and high-search/horizon deficit slice."""
    from lox.author.prompts import build_system_prompt
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    # 1. Option-A peer system prompt drops contradictory BehaviorTree docs & teaches Zero-Drift + Schema-Safe rules
    peer_sys = build_system_prompt(is_swarm=True, is_peer=True)
    assert "Zero-Drift Guard Scoping" in peer_sys
    assert "Schema-Safe `ws.domain(...)`" in peer_sys
    assert "Paired Micro-Pilot Diff" in peer_sys
    assert "PrioritySelector" not in peer_sys
    assert "ActionLeaf" not in peer_sys
    assert "Blackboard" not in peer_sys

    # Legacy is_swarm=True, is_peer=False remains backward-compatible
    legacy_sys = build_system_prompt(is_swarm=True, is_peer=False)
    assert "PrioritySelector" in legacy_sys

    # 2. Relaxed RESEARCH_REQUIRED gate: 1 query_db lookup unlocks dry_run_policy
    dbp = str(tmp_path / "mp.duckdb")
    con = init_db(dbp)
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, parent_policy_id, candidate_policy_id, avg_depth, max_depth, avg_score, avg_turns, hypothesis_passed)
        VALUES ('b1', 18, 'SWARM', 'p0', 'p18', 2.5, 6, 420.0, 2200.0, TRUE)
        """
    )
    # Seed 101: Dlvl 6 guard (healthy); Seed 202: high-search starvation at Dlvl 3; Seed 203: shallow max_steps confinement
    con.execute(
        """
        INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, final_hp, max_hp, final_ac, xp_level, end_reason, pct_wait, pct_search, unique_tiles)
        VALUES
          ('r101', 'b1', 18, 101, 6, 6, 950, 1200, 1200, 18, 22, 2, 4, 'killed', 1.0, 4.0, 320),
          ('r202', 'b1', 18, 202, 3, 3, 410, 6864, 6864, 0, 18, 4, 3, 'starved', 2.0, 44.5, 190),
          ('r203', 'b1', 18, 203, 2, 2, 220, 4000, 4000, 16, 16, 4, 2, 'max_steps', 3.0, 52.0, 120)
        """
    )
    con.close()

    # 3. DeficitAnalyzer isolates pct_search >= 35%, turns >= 1500 into slice_high_search_or_horizon_burn
    #    while keeping Dlvl <= 2 max_steps inside slice_max_steps_confinement
    rep = DeficitAnalyzer(dbp).analyze(
        generation=19, current_policy="class Policy: pass"
    )
    cluster_map = {c.cluster_id: c for c in rep.fail_clusters}
    assert "slice_high_search_or_horizon_burn" in cluster_map
    assert 202 in cluster_map["slice_high_search_or_horizon_burn"].sample_seeds
    assert "slice_max_steps_confinement" in cluster_map
    assert 203 in cluster_map["slice_max_steps_confinement"].sample_seeds

    # 4. Paired CRN Micro-Pilot inside SWARM_PEER dry_run_policy (with exact first-divergence byte & 2-call cap)
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    cand_pol_v1 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    cand_pol_v2 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.E\n"
    )
    cand_pol_v3 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.W\n"
    )

    class _MockMicroPilotPool:
        max_steps = 8000

        def __init__(self):
            self.calls = 0

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            self.calls += 1
            # Target seed 202 had turns=6864 (>= 4000), so long-horizon cap scales to 6500
            assert kwargs.get("max_steps") == 6500
            runs = []
            for sd in priority_seeds or []:
                if sd == 101:
                    # Call 1 regresses guard seed @ byte 3; Call 2 has non-regressing drift; Call 3 has ZERO DRIFT
                    if self.calls == 1:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 2,
                                "score": 200,
                                "turns": 400,
                                "end_reason": "killed",
                                "pct_search": 5.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcXYZ",
                            }
                        )
                    elif self.calls == 2:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 6,
                                "score": 960,
                                "turns": 1200,
                                "end_reason": "killed",
                                "pct_search": 4.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcdeZ",
                            }
                        )
                    else:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 6,
                                "score": 950,
                                "turns": 1200,
                                "end_reason": "killed",
                                "pct_search": 4.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcdef",
                            }
                        )
                elif sd == 202:
                    # Call 3 simulates an over-tightened dead patch (ZERO DRIFT on target)
                    if self.calls == 3:
                        runs.append(
                            {
                                "seed": 202,
                                "max_depth": 3,
                                "score": 410,
                                "turns": 6864,
                                "end_reason": "starved",
                                "pct_search": 44.5,
                                "policy_exceptions": 0,
                                "action_bytes": b"ssssss",
                            }
                        )
                    else:
                        runs.append(
                            {
                                "seed": 202,
                                "max_depth": 5,
                                "score": 760,
                                "turns": 1800,
                                "end_reason": "killed",
                                "pct_search": 12.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"ssssNN",
                            }
                        )
            return {"batch_id": batch_id, "runs": runs}

    mp_pool = _MockMicroPilotPool()
    baseline_runs = [
        {
            "seed": 101,
            "max_depth": 6,
            "score": 950,
            "turns": 1200,
            "end_reason": "killed",
            "pct_search": 4.0,
            "policy_exceptions": 0,
            "action_bytes": b"abcdef",
        },
        {
            "seed": 202,
            "max_depth": 3,
            "score": 410,
            "turns": 6864,
            "end_reason": "starved",
            "pct_search": 44.5,
            "policy_exceptions": 0,
            "action_bytes": b"ssssss",
        },
    ]

    reg = AuthorToolRegistry(db_path=dbp)
    reg.configure_generation(
        pool=mp_pool,
        baseline_runs=baseline_runs,
        generation=19,
        mode="SWARM_PEER",
        peer_guard_seeds=[101],
        peer_target_seeds=[202],
        max_peer_micro_pilots=3,
    )
    reg.current_policy_code = base_pol
    reg.enforce_peer_research_gate = True

    # Blocked before empirical lookup
    _, st0 = reg.dry_run_policy(cand_pol_v1, seed=42, max_turns=15)
    assert st0 == "RESEARCH_REQUIRED"

    # Single query_db call unlocks dry_run_policy
    _, db_st = reg.query_db("SELECT seed, max_depth FROM runs LIMIT 3")
    assert "rows" in db_st

    # Micro-pilot 1: detects GUARD regression @ byte 3 + TARGET rescue
    out1, st1 = reg.dry_run_policy(cand_pol_v1, seed=42, max_turns=15)
    assert st1.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (1/3" in out1
    assert "GUARD seed=101: ⚠️ REGRESSION (DRIFT @ byte 3 -> Dlvl 6->2" in out1
    assert "GUARD DRIFT WARNING" in out1
    assert "TARGET seed=202: ✅ RESCUED/IMPROVED (drift @ byte 4 -> Dlvl 3->5" in out1
    assert mp_pool.calls == 1

    # Micro-pilot 2: non-regressing guard drift explicitly marked OK (do NOT over-tighten)
    out2, st2 = reg.dry_run_policy(cand_pol_v2, seed=42, max_turns=15)
    assert st2.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (2/3" in out2
    assert "GUARD seed=101: OK (non-regressing drift @ byte 5" in out2
    assert "do NOT over-tighten" in out2
    assert "GUARD DRIFT WARNING" not in out2
    assert mp_pool.calls == 2

    # Micro-pilot 3: over-tightened dead patch triggers DEAD-PATCH WARNING + 1 bonus retry (3 -> 4)
    out3, st3 = reg.dry_run_policy(cand_pol_v3, seed=42, max_turns=15)
    assert st3.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (3/3" in out3
    assert "GUARD seed=101: ZERO DRIFT (Dlvl 6" in out3
    assert "DEAD-PATCH WARNING" in out3
    assert "+1 bonus micro-pilot granted" in out3
    assert mp_pool.calls == 3

    # Micro-pilot 4: uses the 1 bonus retry
    out4, st4 = reg.dry_run_policy(
        cand_pol_v3 + "\n# bonus retry\n", seed=42, max_turns=15
    )
    assert st4.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (4/4" in out4
    assert mp_pool.calls == 4

    # Micro-pilot 5: respects cap 4/4 (does not call eval_pool again)
    out5, st5 = reg.dry_run_policy(
        cand_pol_v3 + "\n# cap check\n", seed=42, max_turns=15
    )
    assert st5.startswith("PASSED")
    assert "cap 4/4 reached" in out5
    assert mp_pool.calls == 4
    reg.close()


def test_stage3_tiebreaker_extension_and_confinement_priority(tmp_path):
    """Verifies Stage-3 +64-seed tiebreaker extension promotes genuine 8W-3L surgical near-misses
    while rejecting spurious near-misses, and verifies confinement_horizon priority weighting."""
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.schema import init_db
    from lox.engine.swarm_referee import SwarmReferee

    # 1. Confinement priority: 8 max_steps Dlvl<=2 runs outrank 10 generic early-termination runs
    dbp = str(tmp_path / "conf_prio.duckdb")
    con = init_db(dbp)
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b35', 35, 'SWARM', TRUE)"
    )
    for i in range(8):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles)
            VALUES (?, 'b35', 35, ?, 2, 2, 400, 9500, 10000, 'max_steps', 65.0, 180)
            """,
            [f"m_{i}", 13500 + i],
        )
    for i in range(10):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles)
            VALUES (?, 'b35', 35, ?, 2, 2, 150, 180, 180, 'killed', 5.0, 90)
            """,
            [f"e_{i}", 13600 + i],
        )
    con.close()
    rep = DeficitAnalyzer(dbp).analyze(
        generation=36, current_policy="class Policy: pass"
    )
    assert rep.fail_clusters[0].cluster_id == "slice_max_steps_confinement"

    # 2. Stage-3 Tiebreaker Extension: 8W-3L (p~0.084) on 64 seeds extends by +64 seeds (200_000+) -> 16W-6L (p<0.05) -> promotes!
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    select32 = [135000 + i for i in range(32)]
    seeds64 = [1035000 + i for i in range(64)]

    class _Stage3Pool:
        def __init__(self, stage3_holds: bool):
            self.stage3_holds = stage3_holds
            self.s3_seen = False

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if any(int(s) >= 20_000_000 for s in s_list):
                self.s3_seen = True
            # Realistic incumbent depth spread (Dlvl 2..6)
            base_depths = [2 + (i % 5) for i in range(len(s_list))]
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": base_depths[idx],
                            "max_depth": base_depths[idx],
                            "score": 700,
                            "turns": 2000,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for idx, s in enumerate(s_list)
                    ],
                }
            is_obf = any(int(s) >= 20_000_000 for s in s_list)
            runs = []
            for idx, s in enumerate(s_list):
                bd = base_depths[idx]
                if not is_obf:
                    # 64 seeds: 8W-3L-53T -> sign-perm p ~ 0.055 in [0.030, 0.120) -> triggers OBF +64!
                    if idx < 8:
                        d, sc = bd + 1, 900
                    elif idx < 11:
                        d, sc = max(1, bd - 1), 580
                    else:
                        d, sc = bd, 700
                else:
                    if self.stage3_holds:
                        # Holds consistently on +64 OBF seeds -> 16W-6L (p < 0.032) -> PROMOTES!
                        if idx < 8:
                            d, sc = bd + 1, 900
                        elif idx < 11:
                            d, sc = max(1, bd - 1), 580
                        else:
                            d, sc = bd, 700
                    else:
                        # Spurious near-miss collapses on +64 OBF seeds (2W-7L) -> REJECTED!
                        if idx < 2:
                            d, sc = bd + 1, 900
                        elif idx < 9:
                            d, sc = max(1, bd - 1), 580
                        else:
                            d, sc = bd, 700
                runs.append(
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 2000,
                        "final_ac": 4,
                        "xp_level": 3,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    cand = PeerCandidate(
        peer_id="peer_1",
        claimed_deficit_id="slice_median_outcomes",
        claimed_label="surgical_fix",
        policy_code=base_pol + "\n_S3 = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    pool_pass = _Stage3Pool(stage3_holds=True)
    v_pass = SwarmReferee().evaluate_tournament(
        generation=35,
        candidates=[cand],
        base_policy=base_pol,
        base_skills={},
        eval_pool=pool_pass,
        select_seeds=select32,
        seeds=seeds64,
    )
    assert pool_pass.s3_seen is True
    assert v_pass.promoted is True, v_pass.reason
    assert v_pass.best_pair.get("obf_extended") is True
    assert v_pass.winner_id == "peer_1"

    pool_fail = _Stage3Pool(stage3_holds=False)
    v_fail = SwarmReferee().evaluate_tournament(
        generation=35,
        candidates=[cand],
        base_policy=base_pol,
        base_skills={},
        eval_pool=pool_fail,
        select_seeds=select32,
        seeds=seeds64,
    )
    assert pool_fail.s3_seen is True
    assert v_fail.promoted is False


def test_g38_deficit_window_recurring_outcomes_referee_stall_rate_and_dead_patch_incubation(
    tmp_path,
):
    """Verifies G38 fixes:
    1. DeficitAnalyzer keeps all 64 runs of the latest generation (never drops max_steps/long runs),
       does not dump deep CONFINEMENT_STALL combat deaths into slice_stalled, and surfaces recurring
       terminal states dynamically with zero hardcoded NetHack strings.
    2. SwarmReferee stall_rate ignores non-terminal CONFINEMENT_STALL noise and Stage-3 admits
       high-IQM near-misses (iqm_delta >= 0.08, p < 0.35).
    3. incubate_candidate rejects 0-byte-divergence dead patches across evaluated target shards.
    """
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate
    from lox.db.schema import init_db
    from lox.engine.swarm_referee import SwarmReferee, _extract_eval_stats

    dbp = str(tmp_path / "g38.duckdb")
    con = init_db(dbp)
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b37', 37, 'SWARM', TRUE), ('b38', 38, 'SWARM', TRUE)"
    )
    # 64 short runs in Gen 37 (previously poisoned the LIMIT 96 turns-ASC stratifier!)
    for i in range(64):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles)
            VALUES (?, 'b37', 37, ?, 2, 2, 120, 150, 150, 'killed', 5.0, 80)
            """,
            [f"g37_{i}", 13700 + i],
        )
    # 64 realistic G38 runs: 9 max_steps Dlvl<=2, 7 recurring A, 6 recurring B, 42 deep combat deaths WITH CONFINEMENT_STALL
    for i in range(9):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles, summary_digest)
            VALUES (?, 'b38', 38, ?, 2, 2, 500, 9900, 10000, 'max_steps', 65.0, 200, 'CONFINEMENT_STALL on Dlvl 2')
            """,
            [f"g38_ms_{i}", 13800 + i],
        )
    for i in range(7):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles, summary_digest)
            VALUES (?, 'b38', 38, ?, 2, 2, 400, 6200, 5800, 'Recurring terminal outcome A', 45.0, 250, 'CONFINEMENT_STALL on Dlvl 1')
            """,
            [f"g38_ra_{i}", 13810 + i],
        )
    for i in range(6):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles, summary_digest)
            VALUES (?, 'b38', 38, ?, 4, 4, 750, 3900, 3800, 'Recurring terminal outcome B', 20.0, 500, 'CONFINEMENT_STALL on Dlvl 3')
            """,
            [f"g38_rb_{i}", 13820 + i],
        )
    for i in range(42):
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, end_reason, pct_search, unique_tiles, summary_digest)
            VALUES (?, 'b38', 38, ?, 5, 5, 850, 2400, 2400, 'combat_death_unique', 18.0, 750, 'CONFINEMENT_STALL on Dlvl 2')
            """,
            [f"g38_cb_{i}", 13830 + i],
        )
    con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=39, current_policy="class Policy: pass"
    )
    cmap = {c.cluster_id: c for c in rep.fail_clusters}
    # All 9 max_steps runs preserved (not dropped by stratifier!)
    assert "slice_max_steps_confinement" in cmap
    assert cmap["slice_max_steps_confinement"].run_count == 9
    # Recurring terminal outcomes surfaced dynamically
    rec_clusters = [
        c for c in rep.fail_clusters if c.kind == "recurring_terminal_state"
    ]
    assert len(rec_clusters) >= 2
    assert any("Recurring terminal outcome A" in c.label for c in rec_clusters)
    assert any("Recurring terminal outcome B" in c.label for c in rec_clusters)
    # Deep CONFINEMENT_STALL combat deaths are NOT dumped into slice_stalled
    assert cmap.get("slice_stalled") is None or cmap["slice_stalled"].run_count < 5

    # 2. _extract_eval_stats stall_rate ignores non-terminal CONFINEMENT_STALL on deep runs
    fake_runs = [
        {
            "depth": 5,
            "max_depth": 5,
            "score": 800,
            "turns": 2000,
            "end_reason": "killed",
            "summary_digest": "CONFINEMENT_STALL on Dlvl 2",
        }
        for _ in range(10)
    ]
    st = _extract_eval_stats({"runs": fake_runs}, loc_total=10)
    assert st["stall_count"] == 10
    assert st["stall_rate"] == 0.0

    # 3. Stage-3 triggers on high-IQM near-miss (iqm_delta >= 0.08, 0.15 <= p < 0.35)
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )

    class _HighIqmNearMissPool:
        def __init__(self):
            self.s3_seen = False

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            is_s3 = any(int(s) >= 20_000_000 for s in s_list)
            if is_s3:
                self.s3_seen = True
            base_depths = [2 + (i % 5) for i in range(len(s_list))]
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": base_depths[idx],
                            "max_depth": base_depths[idx],
                            "score": 680,
                            "turns": 2000,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for idx, s in enumerate(s_list)
                    ],
                }
            runs = []
            for idx, s in enumerate(s_list):
                bd = base_depths[idx]
                if not is_s3:
                    # Exact G37 peer_2 profile: 11W-7L, ΔD=+0.0625, ΔS=+36, ΔIQM>=+0.10, p in [0.15, 0.35)
                    if idx < 3:
                        d, sc = bd + 2, 1050
                    elif idx < 11:
                        d, sc = bd + 1, 900
                    elif idx < 13:
                        d, sc = max(1, bd - 2), 300
                    elif idx < 18:
                        d, sc = max(1, bd - 1), 620
                    else:
                        d, sc = bd, 680
                else:
                    # Holds strongly on Stage-3 fresh seeds -> 10W-2L -> promotes across 128 seeds
                    if idx < 10:
                        d, sc = bd + 1, 950
                    elif idx < 12:
                        d, sc = max(1, bd - 1), 620
                    else:
                        d, sc = bd, 680
                runs.append(
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 2000,
                        "final_ac": 4,
                        "xp_level": 3,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    hi_pool = _HighIqmNearMissPool()
    cand_hi = PeerCandidate(
        peer_id="peer_2",
        claimed_deficit_id="slice_early_terminations",
        claimed_label="high_iqm_nearmiss",
        policy_code=base_pol + "\n_HI = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    # Under v2 alpha-spent OBF (0.030 <= p < 0.12), a p=0.20 noise candidate does NOT peek/extend!
    v_hi = SwarmReferee().evaluate_tournament(
        generation=37,
        candidates=[cand_hi],
        base_policy=base_pol,
        base_skills={},
        eval_pool=hi_pool,
        select_seeds=[137000 + i for i in range(32)],
        seeds=[1037000 + i for i in range(64)],
    )
    assert hi_pool.s3_seen is False
    assert v_hi.promoted is False

    # 4. incubate_candidate rejects 0-byte-divergence dead patch when >=2 target seeds have action_bytes
    class _ZeroDivPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "seed": s,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 100,
                        "policy_exceptions": 0,
                        "action_bytes": b"identical_trace",
                    }
                    for s in (priority_seeds or [])
                ],
            }

    inc_shards = {
        101: {
            "seed": 101,
            "depth": 2,
            "max_depth": 2,
            "score": 100,
            "action_bytes": b"identical_trace",
        },
        201: {
            "seed": 201,
            "depth": 2,
            "max_depth": 2,
            "score": 100,
            "action_bytes": b"identical_trace",
        },
        202: {
            "seed": 202,
            "depth": 2,
            "max_depth": 2,
            "score": 100,
            "action_bytes": b"identical_trace",
        },
    }
    dead_cand = PeerCandidate(
        peer_id="peer_dead",
        claimed_deficit_id="slice_stalled",
        claimed_label="dead",
        policy_code=base_pol,
        skills={},
    )
    res_dead = incubate_candidate(
        dead_cand,
        deficit_report=DeficitReport(
            generation=39, regime="STAGNATING", d_succ_seeds=[101]
        ),
        claimed_deficit=DeficitCluster(
            cluster_id="slice_stalled",
            kind="stalled",
            label="stalled",
            sample_seeds=[201, 202],
        ),
        eval_pool=_ZeroDivPool(),
        incumbent_shard_runs=inc_shards,
        shard_seeds=[101, 201, 202],
    )
    assert res_dead.sandbox_passed is False
    assert "Zero behavioral divergence" in res_dead.error


def test_promotion_gate_v2_split_sample_anti_goodhart_safety_and_craftax():
    """Verifies every Promotion Gate v2 invariant:
    1. Anti-Goodhart NetHackAdapter + CraftaxAdapter progression_metric.
    2. Exact Paired Sign-Permutation & exact small-N Wilcoxon calibration (4W-0L=0.0625, 5W-0L=0.03125).
    3. Split-Sample Select(32) -> Confirm(64): zero fallback leakage on noise winner, unbiased S_confirm logging.
    4. 4 benchmark-agnostic safety invariants (horizon_waste_rate at D3+, auto_esc, cvar25) & AST bloat alpha bar.
    5. Multi-benchmark Craftax SwarmReferee end-to-end promotion.
    """
    import numpy as np

    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.hypothesis import (
        cvar_bottom_quantile,
        exact_sign_permutation_p,
        progression_metric,
        wilcoxon_signed_rank_p,
        winsorized_mean,
    )
    from lox.engine.swarm_referee import SwarmReferee
    from lox.env.craftax.adapter import CraftaxAdapter

    # 1. Anti-Goodhart NetHackAdapter & CraftaxAdapter progression_metric
    for depth in (1, 2, 3, 4):
        normal_run = {
            "depth": depth,
            "max_depth": depth,
            "xp_level": 2,
            "score": 180,
            "final_ac": 5,
            "end_reason": "killed by goblin",
        }
        camper_run = {
            "depth": depth,
            "max_depth": depth,
            "xp_level": 6,
            "score": 2500,
            "final_ac": 1,
            "turns": 10000,
            "end_reason": "max_steps",
        }
        assert progression_metric(camper_run) < progression_metric(normal_run), (
            f"D{depth} max_steps camper must score strictly below genuine D{depth} run"
        )

    craftax_adapter = CraftaxAdapter()
    assert (
        craftax_adapter.progression_metric({"achievements": 0, "steps": 10000}) == 0.0
    )
    assert (
        craftax_adapter.progression_metric({"achievements": 1, "steps": 10000})
        < 1.08 + 1e-9
    )
    assert progression_metric(
        {"achievements": 3, "steps": 200}, benchmark="craftax"
    ) > (progression_metric({"achievements": 2, "steps": 50000}, benchmark="craftax"))

    # 2. Exact Paired Sign-Permutation & Exact Wilcoxon (fixes 4W-0L anti-conservative normal approx!)
    p4, w4, l4, t4 = exact_sign_permutation_p(np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    assert (w4, l4, t4) == (4, 0, 2)
    assert abs(p4 - 0.0625) < 1e-9
    assert (
        abs(wilcoxon_signed_rank_p(np.array([1.0, 1.0, 1.0, 1.0]))[0] - 0.0625) < 1e-9
    )

    p5, w5, l5, _ = exact_sign_permutation_p(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert (w5, l5) == (5, 0)
    assert abs(p5 - 0.03125) < 1e-9
    assert (
        winsorized_mean([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.1) < 2.0
    )
    assert (
        abs(cvar_bottom_quantile([1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 0.25) - 1.5)
        < 1e-9
    )

    # 3. Split-Sample Select(32) -> Confirm(64): zero fallback leakage when C* is noise on S_confirm
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    select32 = [110000 + i for i in range(32)]
    confirm64 = [10110000 + i for i in range(64)]

    class _SplitNoFallbackPool:
        def __init__(self):
            self.confirm_evaluated: list[str] = []

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_conf_" in batch_id and "_inc_" not in batch_id:
                self.confirm_evaluated.append(batch_id)
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 1500,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            is_sel = "_sel_" in batch_id
            if "peer_overfit" in batch_id:
                # Crushes S_select (+3 depth), completely null on S_confirm!
                d = 6 if is_sel else 3
                sc = 1800 if is_sel else 500
            else:
                # peer_runnerup: smaller gain on S_select (+1), strong on S_confirm (+2)
                # Under strict split-sample, ONLY C* (peer_overfit) goes to S_confirm -> ZERO fallback!
                d = 4 if is_sel else 5
                sc = 900 if is_sel else 1400
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1500,
                        "end_reason": "killed",
                    }
                    for s in s_list
                ],
            }

    c_overfit = PeerCandidate(
        peer_id="peer_overfit",
        claimed_deficit_id="d1",
        claimed_label="overfit",
        policy_code=base_pol + "\n_OF = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    c_runner = PeerCandidate(
        peer_id="peer_runnerup",
        claimed_deficit_id="d2",
        claimed_label="runnerup",
        policy_code=base_pol + "\n_RU = 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    split_pool = _SplitNoFallbackPool()
    v_split = SwarmReferee().evaluate_tournament(
        generation=40,
        candidates=[c_overfit, c_runner],
        base_policy=base_pol,
        base_skills={},
        eval_pool=split_pool,
        select_seeds=select32,
        seeds=confirm64,
    )
    assert v_split.promoted is False
    assert len(split_pool.confirm_evaluated) == 1
    assert "peer_overfit" in split_pool.confirm_evaluated[0]
    assert len(v_split.winner_batch_eval["runs"]) == 64

    # 4. Safety Invariants: D4 horizon_waste_rate regression & auto_esc_regression & AST bloat alpha bar
    class _SafetyPool:
        def __init__(self, mode: str):
            self.mode = mode

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 1500,
                            "auto_esc_count": 0,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            runs = []
            for idx, s in enumerate(s_list):
                if self.mode == "horizon_waste_d4":
                    # Wins 50 seeds to D4, but 8 seeds hit D4 max_steps (12.5% > 5% tol)
                    er = "max_steps" if idx < 8 else "killed"
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": 4,
                            "max_depth": 4,
                            "score": 900,
                            "turns": 10000 if er == "max_steps" else 1500,
                            "auto_esc_count": 0,
                            "end_reason": er,
                        }
                    )
                elif self.mode == "auto_esc":
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": 4,
                            "max_depth": 4,
                            "score": 900,
                            "turns": 1500,
                            "auto_esc_count": 1 if idx < 5 else 0,
                            "end_reason": "killed",
                        }
                    )
                else:
                    # AST bloat bar: 5W-0L (exact p = 0.03125): passes alpha=0.05, fails bloated eff_alpha=0.025!
                    d = 5 if idx < 5 else 3
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": d,
                            "max_depth": d,
                            "score": 1200 if idx < 5 else 500,
                            "turns": 1500,
                            "auto_esc_count": 0,
                            "end_reason": "killed",
                        }
                    )
            return {"batch_id": batch_id, "runs": runs}

    big_base = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        + "".join(f"        x_{i} = {i} + 1\n" for i in range(15))
        + "        return Action.N\n"
    )
    bloated_cand_code = big_base + "".join(
        f"_BLOAT_{i} = {i} * 2 + 1\n" for i in range(25)
    )
    lean_cand_code = big_base + "_LEAN = 1\n"

    v_hw = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_hw",
                claimed_deficit_id="d1",
                claimed_label="hw",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("horizon_waste_d4"),
        seeds=confirm64,
    )
    assert v_hw.promoted is False
    assert v_hw.attempts_summary[0]["reject_reason"] == "horizon_waste_regression"

    v_esc = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_esc",
                claimed_deficit_id="d1",
                claimed_label="esc",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("auto_esc"),
        seeds=confirm64,
    )
    assert v_esc.promoted is False
    assert v_esc.attempts_summary[0]["reject_reason"] == "auto_esc_regression"

    # Lean 5W-0L (p=0.03125 < 0.05) promotes; bloated >+25% AST 5W-0L (0.03125 >= 0.025) is rejected!
    v_lean = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_lean",
                claimed_deficit_id="d1",
                claimed_label="lean",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("ast_bar"),
        seeds=confirm64,
    )
    assert v_lean.promoted is True, v_lean.reason

    v_bloat = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_bloat",
                claimed_deficit_id="d1",
                claimed_label="bloat",
                policy_code=bloated_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=45,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("ast_bar"),
        seeds=confirm64,
    )
    assert v_bloat.promoted is False
    assert (
        v_bloat.attempts_summary[0]["reject_reason"]
        == "ast_bloat_below_tightened_alpha"
    )

    # 5. Multi-Benchmark Craftax SwarmReferee integrity (zero NetHack assumptions)
    class _CraftaxPool:
        benchmark = "craftax"

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            is_inc = "_inc_" in batch_id
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"{'inc' if is_inc else 'c'}_{s}",
                        "seed": s,
                        "achievements": 2 if is_inc else 4,
                        "steps": 400,
                        "depth": 1,
                        "max_depth": 1,
                        "score": 0,
                        "end_reason": "died",
                    }
                    for s in s_list
                ],
            }

    v_cx = SwarmReferee().evaluate_tournament(
        generation=1,
        candidates=[
            PeerCandidate(
                peer_id="p_cx",
                claimed_deficit_id="d1",
                claimed_label="craftax_ach",
                policy_code=base_pol + "\n_CX = 1\n",
                skills={},
                sandbox_passed=True,
                loc_total=5,
            )
        ],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_CraftaxPool(),
        select_seeds=[100001 + i for i in range(8)],
        seeds=[10000001 + i for i in range(16)],
    )
    assert v_cx.promoted is True, v_cx.reason
    assert v_cx.best_pair["mean_delta"] >= 1.9


def test_plateau_fixes_gate_telemetry_and_probes(tmp_path):
    """
    Verifies all three plateau-breaking pillars:
    1. Sparsity-preserving Winsorized delta preserves 3W/0L/61T while capping dense outliers.
    2. Split-Sample Referee eliminates unsafe pre-merge (Gen 42 regression), ranks credible
       surgical winners over noisy coin-flips, builds safe stacks from individually safe winners,
       and widens OBF Look-1 extension band to p < 0.16 (3W/0L -> +64 extension).
    3. DeficitAnalyzer reports domain-agnostic per-depth residence & action concentration
       telemetry (with expert parity) and prioritizes high action-burn above recurring symptom slices.
    4. Paired Micro-Pilot displays informational PROBE rows without false dead-patch warnings.
    """
    import numpy as np

    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import connect_db, init_db
    from lox.engine.hypothesis import (
        sparse_winsorized_mean,
        sparse_winsorized_stats,
        winsorized_mean,
    )
    from lox.engine.swarm_referee import SwarmReferee

    # 1. Sparse Winsorized Delta vs naive Winsorization
    sparse_3w = np.array([2.0, 2.0, 2.0] + [0.0] * 61, dtype=np.float64)
    assert (
        winsorized_mean(sparse_3w, 0.05) == 0.0
    )  # Old naive winsor clipped 3W/0L to 0
    sw_mean, sw_se = sparse_winsorized_stats(sparse_3w, 0.05)
    assert sw_mean > 0.09
    assert sw_se > 0.0
    assert (
        sparse_winsorized_mean(
            [100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.1
        )
        < 2.0
    )

    # 2A. Gen 42 Regression: No unsafe pre-merge & precision-aware S_select ranking
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    base_skills = {
        "combat/core": "def fight(s):\n    return 1\n",
        "goals/explore": "def explore(s):\n    return 1\n",
        "survival/core": "def heal(s):\n    return 1\n",
    }
    select32 = [142000 + i for i in range(32)]
    confirm64 = [10142000 + i for i in range(64)]

    class _Gen42Pool:
        def __init__(self):
            self.seen_batches: list[str] = []

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            self.seen_batches.append(batch_id)
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 1500,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            is_sel = "_sel_" in batch_id
            runs = []
            for idx, s in enumerate(s_list):
                if "peer_toxic" in batch_id:
                    # Fails cvar25_tail_regression badly on bottom quartile
                    d = 1 if idx < 10 else 4
                    sc = 100 if idx < 10 else 900
                elif "peer_noisy" in batch_id:
                    # High-variance 16W/14L coin flip on S_select (p > 0.25), slightly positive raw mean
                    if is_sel:
                        if idx < 16:
                            d, sc = 5, 1100
                        elif idx < 30:
                            d, sc = 1, 200
                        else:
                            d, sc = 3, 500
                    else:
                        d, sc = 3, 500
                elif "peer_clean" in batch_id:
                    # Surgical 7W/1L on S_select (p < 0.04) and strong 8W/0L on S_confirm!
                    if is_sel:
                        if idx < 7:
                            d, sc = 5, 1100
                        elif idx == 7:
                            d, sc = 2, 300
                        else:
                            d, sc = 3, 500
                    else:
                        d, sc = (5, 1100) if idx < 8 else (3, 500)
                else:
                    d, sc = 3, 500
                runs.append(
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1500,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    p_clean = PeerCandidate(
        peer_id="peer_clean",
        claimed_deficit_id="d1",
        claimed_label="clean_fix",
        policy_code=base_pol,
        skills={
            **base_skills,
            "combat/core": "def fight(s):\n    return 2\n",
        },
        touched_skills={"combat/core"},
        touched_skill_symbols={"combat/core": {"fight"}},
        sandbox_passed=True,
        loc_total=12,
    )
    p_toxic = PeerCandidate(
        peer_id="peer_toxic",
        claimed_deficit_id="d2",
        claimed_label="toxic_fix",
        policy_code=base_pol,
        skills={
            **base_skills,
            "goals/explore": "def explore(s):\n    return 2\n",
        },
        touched_skills={"goals/explore"},
        touched_skill_symbols={"goals/explore": {"explore"}},
        sandbox_passed=True,
        loc_total=12,
    )
    p_noisy = PeerCandidate(
        peer_id="peer_noisy",
        claimed_deficit_id="d3",
        claimed_label="noisy_fix",
        policy_code=base_pol,
        skills={
            **base_skills,
            "survival/core": "def heal(s):\n    return 2\n",
        },
        touched_skills={"survival/core"},
        touched_skill_symbols={"survival/core": {"heal"}},
        sandbox_passed=True,
        loc_total=12,
    )
    g42_pool = _Gen42Pool()
    v42 = SwarmReferee(max_tournament_peers=3).evaluate_tournament(
        generation=42,
        candidates=[p_toxic, p_noisy, p_clean],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=g42_pool,
        select_seeds=select32,
        seeds=confirm64,
    )
    assert not any("ortho_" in bid for bid in g42_pool.seen_batches), (
        f"Mode A must NOT pre-merge unverified ortho pairs: {g42_pool.seen_batches}"
    )
    assert not any(
        "peer_toxic" in bid and "stack_" in bid for bid in g42_pool.seen_batches
    )
    assert v42.promoted is True, v42.reason
    assert v42.winner_id == "peer_clean"

    # 2B. Safe stack IS built when 2 individually safe positive winners exist
    class _SafeStackPool:
        def __init__(self):
            self.seen_batches: list[str] = []

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            self.seen_batches.append(batch_id)
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                d_val = 3
            elif "stack_" in batch_id:
                d_val = 6
            else:
                d_val = 4
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"r_{s}",
                        "seed": s,
                        "depth": d_val,
                        "max_depth": d_val,
                        "score": 400 * d_val,
                        "turns": 1500,
                        "end_reason": "killed",
                    }
                    for s in s_list
                ],
            }

    p_safe2 = PeerCandidate(
        peer_id="peer_safe2",
        claimed_deficit_id="d2",
        claimed_label="safe2",
        policy_code=base_pol,
        skills={
            **base_skills,
            "goals/explore": "def explore(s):\n    return 3\n",
        },
        touched_skills={"goals/explore"},
        touched_skill_symbols={"goals/explore": {"explore"}},
        sandbox_passed=True,
        loc_total=12,
    )
    stk_pool = _SafeStackPool()
    v_stk = SwarmReferee(max_tournament_peers=3).evaluate_tournament(
        generation=43,
        candidates=[p_clean, p_safe2],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=stk_pool,
        select_seeds=select32[:8],
        seeds=confirm64[:16],
    )
    assert any("stack_" in bid for bid in stk_pool.seen_batches)
    assert v_stk.promoted is True
    assert v_stk.winner_id.startswith("stack_")

    # 2C. Widened OBF Look-1 trigger (3W/0L -> p=0.125 < 0.16 triggers +64 OBF extension!)
    class _ObfSparsePool:
        def __init__(self):
            self.obf_called = False

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_obf_" in batch_id:
                self.obf_called = True
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 1500,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            # 3W/0L on S_confirm (p=0.125), +4W/0L on OBF extension (pooled 7W/0L -> p=0.0078 < 0.032!)
            win_n = 4 if "_obf_" in batch_id else (6 if "_sel_" in batch_id else 3)
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": 5 if idx < win_n else 3,
                        "max_depth": 5 if idx < win_n else 3,
                        "score": 1100 if idx < win_n else 500,
                        "turns": 1500,
                        "end_reason": "killed",
                    }
                    for idx, s in enumerate(s_list)
                ],
            }

    obf_pool = _ObfSparsePool()
    v_obf = SwarmReferee().evaluate_tournament(
        generation=44,
        candidates=[p_clean],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=obf_pool,
        select_seeds=select32,
        seeds=confirm64,
    )
    assert obf_pool.obf_called is True
    assert v_obf.promoted is True, v_obf.reason
    assert v_obf.best_pair.get("obf_extended") is True

    # 3. Domain-agnostic efficiency telemetry & upstream action-burn priority over recurring symptoms
    db_path = str(tmp_path / "eff_test.duckdb")
    init_db(db_path)
    con = connect_db(db_path, read_only=False)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, avg_depth, max_depth, avg_score, hypothesis_passed)
            VALUES ('b_g10', 10, 'SWARM', 2.5, 4, 400.0, TRUE)
            """
        )
        for i in range(4):
            con.execute(
                """
                INSERT INTO runs (
                    run_id, batch_id, generation, seed, depth, max_depth, turns,
                    score, end_reason, unique_tiles, pct_wait, pct_search,
                    starved_with_food, died_unarmored, auto_esc_count, policy_exceptions,
                    last_active_function, summary_digest, is_shard
                )
                VALUES (?, 'b_g10', 10, ?, 2, 2, 2600, 300, 'fainted from exhaustion', 80, 2.0, 54.0,
                        FALSE, FALSE, 0, 0, 'loop_fn', '', FALSE)
                """,
                [f"r_{i}", 300 + i],
            )
            # 10 ticks on depth 1: 6 ACT_SCAN, 4 ACT_STEP
            for step_i in range(10):
                act = "ACT_SCAN" if step_i < 6 else "ACT_STEP"
                con.execute(
                    """
                    INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, action_name, active_function, message)
                    VALUES (?, ?, ?, 1, 12, 12, ?, 'loop_fn', '')
                    """,
                    [f"r_{i}", step_i, step_i, act],
                )
        # Expert ticks parity on depth 1 (span = 120 turns)
        con.execute(
            """
            INSERT INTO expert_ticks (run_id, step_idx, turn, depth, hp, max_hp, action_name)
            VALUES ('exp_1', 0, 1, 1, 14, 14, 'ACT_STEP'),
                   ('exp_1', 1, 121, 1, 14, 14, 'ACT_STEP')
            """
        )
    finally:
        con.close()

    rep = DeficitAnalyzer(db_path).analyze(generation=11, current_policy=base_pol)
    assert "D1" in rep.efficiency_summary
    assert "ACT_SCAN 60.0%" in rep.efficiency_summary
    assert "vs expert ~120t" in rep.efficiency_summary
    assert "Floor Residence & Action Concentration Telemetry" in rep.format_brief()
    cluster_ids = [c.cluster_id for c in rep.fail_clusters]
    assert "slice_high_search_or_horizon_burn" in cluster_ids
    assert cluster_ids[0] == "slice_high_search_or_horizon_burn"
    assert any(cid.startswith("slice_recurring_outcome_") for cid in cluster_ids)

    # 4. Micro-Pilot PROBE rows (informational only; no spurious dead-patch warning)
    class _MicroProbePool:
        max_steps = 1000

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"mp_{s}",
                        "seed": s,
                        "depth": 4 if s == 20 else 3,
                        "max_depth": 4 if s == 20 else 3,
                        "score": 700 if s == 20 else 500,
                        "turns": 500,
                        "pct_search": 10.0,
                        "end_reason": "killed",
                        "action_bytes": b"XY" if s == 20 else b"AA",
                    }
                    for s in s_list
                ],
            }

    reg = AuthorToolRegistry(db_path=db_path)
    reg.current_policy_code = base_pol
    reg.configure_generation(
        pool=_MicroProbePool(),
        baseline_runs=[
            {
                "seed": 10,
                "depth": 3,
                "max_depth": 3,
                "score": 500,
                "turns": 500,
                "pct_search": 10.0,
                "end_reason": "killed",
                "action_bytes": b"AA",
            },
            {
                "seed": 20,
                "depth": 2,
                "max_depth": 2,
                "score": 300,
                "turns": 500,
                "pct_search": 40.0,
                "end_reason": "killed",
                "action_bytes": b"AA",
            },
            {
                "seed": 5000010,
                "depth": 3,
                "max_depth": 3,
                "score": 500,
                "turns": 500,
                "pct_search": 10.0,
                "end_reason": "killed",
                "action_bytes": b"AA",
            },
        ],
        peer_guard_seeds=[10],
        peer_target_seeds=[20],
        peer_probe_seeds=[5000010],
        max_peer_micro_pilots=2,
    )
    mp_out = reg._run_peer_paired_micro_pilot(base_pol, {})
    assert "GUARD seed=10: ZERO DRIFT" in mp_out
    assert "TARGET seed=20: ✅ RESCUED/IMPROVED" in mp_out
    assert "PROBE seed=5000010: UNCHANGED" in mp_out
    assert "DEAD-PATCH WARNING" not in mp_out


def test_cluster_stats_plateau_focus_and_incubation_guard(tmp_path):
    """
    Verifies Gen 49-51 fixes:
    1. DeficitAnalyzer includes cluster_stats[N=...] aggregate telemetry in evidence.
    2. incubate_candidate rejects depth_delta <= -0.35 across >= 4 paired shards.
    """
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate
    from lox.db.schema import connect_db, init_db

    # 1. cluster_stats aggregate telemetry in DeficitAnalyzer
    dbp = str(tmp_path / "cs_test.duckdb")
    init_db(dbp)
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b1', 10, 'SWARM', TRUE)"
        )
        for i in range(6):
            con.execute(
                """
                INSERT INTO runs (
                    run_id, batch_id, generation, seed, depth, max_depth, turns,
                    score, end_reason, unique_tiles, pct_wait, pct_search,
                    starved_with_food, died_unarmored, is_shard
                )
                VALUES (?, 'b1', 10, ?, 2, 2, 2200, 300, 'max_steps', 90, 5.0, 50.0, FALSE, TRUE, FALSE)
                """,
                [f"r_{i}", 100 + i],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=11, current_policy="class Policy: pass"
    )
    assert rep.fail_clusters
    assert "cluster_stats[N=6]:" in rep.fail_clusters[0].evidence
    assert "avg_search=50.0%" in rep.fail_clusters[0].evidence

    # 2. incubate_candidate rejects paired depth_delta <= -0.35 across >= 4 shards
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )

    class _RegPool:
        benchmark = "nethack"

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"inc_{s}",
                        "seed": s,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 200,
                        "turns": 500,
                        "policy_exceptions": 0,
                        "action_bytes": b"BB",
                    }
                    for s in s_list
                ],
            }

    inc_runs = {
        s: {
            "seed": s,
            "depth": 4,
            "max_depth": 4,
            "score": 600,
            "action_bytes": b"AA",
        }
        for s in (1, 2, 3, 4)
    }
    bad_cand = PeerCandidate(
        peer_id="p_reg",
        claimed_deficit_id="d1",
        claimed_label="reg",
        policy_code=base_pol,
        skills={},
    )
    res_cand = incubate_candidate(
        bad_cand,
        deficit_report=DeficitReport(generation=51, regime="PLATEAU"),
        claimed_deficit=DeficitCluster(cluster_id="d1", kind="k", label="l"),
        eval_pool=_RegPool(),
        incumbent_shard_runs=inc_runs,
        shard_seeds=[1, 2, 3, 4],
    )
    assert res_cand.sandbox_passed is False
    assert "Incubation paired depth regression" in res_cand.error
