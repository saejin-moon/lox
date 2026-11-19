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
    assert att["reject_reason"] == "below_sig_or_pareto_gate"
    BatchDatabaseLogger(db_path=dbp).save_tournament_attempts(v_noise.attempts_summary)
    con2 = connect_db(dbp, read_only=True)
    try:
        row = con2.execute(
            "SELECT wins, losses, reject_reason FROM tournament_attempts WHERE peer_id='peer_noisy_pareto'"
        ).fetchone()
        assert row == (att["wins"], att["losses"], "below_sig_or_pareto_gate")
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
    seeds64 = [31000 + i for i in range(64)]

    # 1. G12 scenario: 2W-0L-62T surgical fix (+0.16 depth, +50 score, +0.18 IQM, 0 losses) must promote
    #    and must NOT be cut in Stage-1 racing behind net-losing 11W-12L churn candidates
    class _SurgicalZeroLossPool:
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
                # 1 win in Stage 1 (seed 31005: depth 3 -> 8) + 1 win in Stage 2 (seed 31040: depth 3 -> 8) = 2W-0L-62T
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"sg_{s}",
                            "seed": s,
                            "depth": 8 if s in (31005, 31040) else 3,
                            "max_depth": 8 if s in (31005, 31040) else 3,
                            "score": 2100 if s in (31005, 31040) else 500,
                            "turns": 1500,
                            "final_ac": 4,
                            "xp_level": 3,
                            "end_reason": "killed",
                        }
                        for s in s_list
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
        eval_pool=_SurgicalZeroLossPool(),
        seeds=seeds64,
        race_seeds=32,
    )
    assert v_g12.promoted is True, v_g12.reason
    assert v_g12.winner_id == "peer_surgical"

    # 2. G11 scenario: 14W-10L-40T (+0.11 depth, +75 score, +0.098 IQM, integer p75 tied) promotes via pareto_axis_gain
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
        seeds=seeds64,
    )
    assert v_g11.promoted is True, v_g11.reason
    assert v_g11.winner_id == "ortho_peer_2_peer_3"


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

    # Micro-pilot 3: over-tightened dead patch triggers DEAD-PATCH WARNING
    out3, st3 = reg.dry_run_policy(cand_pol_v3, seed=42, max_turns=15)
    assert st3.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (3/3" in out3
    assert "GUARD seed=101: ZERO DRIFT (Dlvl 6" in out3
    assert "DEAD-PATCH WARNING" in out3
    assert mp_pool.calls == 3

    # Micro-pilot 4: respects 3-call cap (does not call eval_pool again)
    out4, st4 = reg.dry_run_policy(
        cand_pol_v3 + "\n# cap check\n", seed=42, max_turns=15
    )
    assert st4.startswith("PASSED")
    assert "cap 3/3 reached" in out4
    assert mp_pool.calls == 3
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
    seeds64 = [13500 + i for i in range(64)]

    class _Stage3Pool:
        def __init__(self, stage3_holds: bool):
            self.stage3_holds = stage3_holds
            self.s3_seen = False

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if any(int(s) >= 200_000 for s in s_list):
                self.s3_seen = True
            # Realistic incumbent depth spread (Dlvl 2..6) so IQM moves smoothly on mid-quartile seeds
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
            is_s3 = any(int(s) >= 200_000 for s in s_list)
            runs = []
            for idx, s in enumerate(s_list):
                bd = base_depths[idx]
                if not is_s3:
                    # 64 seeds: 7W-4L-53T -> ΔD=+0.047, ΔS=+14, ΔIQM>0, p=0.0777 (exact surgical near-miss!)
                    if idx < 7:
                        d, sc = bd + 1, 900
                    elif idx < 11:
                        d, sc = max(1, bd - 1), 580
                    else:
                        d, sc = bd, 700
                else:
                    if self.stage3_holds:
                        # Consistent +7W-4L on Stage-3 fresh seeds -> 14W-8L across 128 seeds (p=0.0187 < 0.05!)
                        if idx < 7:
                            d, sc = bd + 1, 900
                        elif idx < 11:
                            d, sc = max(1, bd - 1), 580
                        else:
                            d, sc = bd, 700
                    else:
                        # Spurious near-miss collapses on Stage-3 seeds (2W-7L)
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
        seeds=seeds64,
    )
    assert pool_pass.s3_seen is True
    assert v_pass.promoted is True, v_pass.reason
    assert v_pass.winner_id == "peer_1"

    pool_fail = _Stage3Pool(stage3_holds=False)
    v_fail = SwarmReferee().evaluate_tournament(
        generation=35,
        candidates=[cand],
        base_policy=base_pol,
        base_skills={},
        eval_pool=pool_fail,
        seeds=seeds64,
    )
    assert pool_fail.s3_seen is True
    assert v_fail.promoted is False
