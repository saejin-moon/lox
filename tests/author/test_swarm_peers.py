"""Domain-scoped tests for Option-A Peer Swarm agents, deficits, incubation, and referee."""

from __future__ import annotations

from lox.author.directives import extract_policy_branches, prune_unused_skills
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, init_db


def test_peer_swarm_tool_session_and_prune_unused_skills(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import get_schema_markdown

    db_path = str(tmp_path / "swarm_modes.duckdb")
    init_db(db_path).close()

    schema_md = get_schema_markdown(db_path)
    assert "`policies`" in schema_md
    assert "`discoveries`" in schema_md
    assert "`latest_pilot_ticks`" in schema_md
    assert "`llm_cost_rates`" in schema_md

    ag = AuthorAgent(provider="mock", db_path=db_path)
    calls = {"n": 0}

    def fake_llm(messages, allow_tools=True, tool_specs=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "query_db",
                            "arguments": '{"sql": "SELECT COUNT(*) FROM runs"}',
                        },
                    }
                ],
            }
        return {"content": '[{"category_id": "test_cat"}]', "tool_calls": []}

    ag._call_llm_api = fake_llm  # type: ignore[method-assign]
    tool_sess_out = ag.run_tool_session("sys", "usr", max_tool_turns=3)
    assert "test_cat" in tool_sess_out

    # Verify Gen 1 empty-DB check allows `AS runs` / `AS ticks` aliases on expert_ticks while blocking `FROM runs`
    tools = AuthorToolRegistry(db_path=db_path)
    tools.generation = 1
    out_ok, sum_ok = tools.query_db(
        "SELECT depth, COUNT(DISTINCT run_id) AS runs, COUNT(*) AS ticks FROM expert_ticks GROUP BY depth"
    )
    assert sum_ok != "error", out_ok
    out_err, sum_err = tools.query_db(
        "SELECT end_reason, COUNT(*) AS n FROM runs GROUP BY end_reason"
    )
    assert sum_err == "error"
    assert "disallowed" in out_err

    # Verify require_json nudges past Turn-1 plain-text planning in run_tool_session
    nudge_calls = {"n": 0}

    def fake_nudge_llm(messages, allow_tools=True, tool_specs=None):
        nudge_calls["n"] += 1
        if nudge_calls["n"] == 1:
            return {
                "content": "Investigate: diff g8 vs g9, look at runs, function stats.",
                "tool_calls": [],
            }
        return {
            "content": '{"needs_revision": false}',
            "tool_calls": [],
        }

    ag._call_llm_api = fake_nudge_llm  # type: ignore[method-assign]
    nudged_out = ag.run_tool_session(
        "sys",
        "usr",
        max_tool_turns=3,
        require_json=True,
        expected_json_type="object",
    )
    assert nudge_calls["n"] == 2
    assert '"needs_revision": false' in nudged_out

    # Verify prune_unused_skills removes orphaned files while preserving transitive imports
    prune_dir = tmp_path / "prune_skills"
    prune_dir.mkdir(parents=True)
    for nm, code_s in {
        "used_a": "from skills.used_helper import h\n",
        "used_helper": "h = 1\n",
        "orphaned_old": "x = 2\n",
    }.items():
        (prune_dir / f"{nm}.py").write_text(code_s, encoding="utf-8")
    kept = prune_unused_skills(
        "from skills.used_a import h\nclass Policy:\n    pass\n",
        {
            "used_a": "from skills.used_helper import h\n",
            "used_helper": "h = 1\n",
            "orphaned_old": "x = 2\n",
        },
        skills_dir=prune_dir,
    )
    assert sorted(kept.keys()) == ["used_a", "used_helper"]
    assert not (prune_dir / "orphaned_old.py").exists()
    assert (prune_dir / "used_helper.py").exists()


def test_democratized_peer_swarm_incubation_and_crn_tournament(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate
    from lox.engine.frontier import FrontierCheckpoint
    from lox.engine.swarm_referee import SwarmReferee

    db_path = str(tmp_path / "peer_swarm.duckdb")
    init_db(db_path).close()

    base_pol = (
        "from lox.env.actions import Action\n"
        "from skills.skill_a import fn_a\n"
        "from skills.skill_b import fn_b\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        act = fn_a(state)\n"
        "        if act is not None:\n"
        "            return act\n"
        "        act = fn_b(state)\n"
        "        if act is not None:\n"
        "            return act\n"
        "        return Action.WAIT\n"
    )
    base_skills = {
        "skill_a": "from lox.env.actions import Action\ndef fn_a(st):\n    return None\n",
        "skill_b": "from lox.env.actions import Action\ndef fn_b(st):\n    return Action.SEARCH\n",
    }
    branches = extract_policy_branches(base_pol, base_skills)
    assert any(b["fn_name"] == "fn_a" for b in branches)
    assert any(b["fn_name"] == "fn_b" for b in branches)

    logger = BatchDatabaseLogger(db_path)
    logger.save_batch(
        batch_id="b_g1",
        generation=1,
        mode="SWARM",
        parent_policy_id="policy_swarm_g0",
        candidate_policy_id="policy_swarm_g1",
        target_functions=["fn_b"],
        runs=[
            {
                "run_id": "r_good",
                "seed": 1000,
                "depth": 3,
                "max_depth": 3,
                "score": 450,
                "turns": 600,
                "steps": 600,
                "end_reason": "killed by grid bug",
                "unique_tiles": 140,
                "pct_search": 0.10,
                "triggered_functions": {"fn_b": 25},
                "fn_turns": {"fn_b": 600},
                "summary_digest": "Overview: Depth=3",
                "milestones": {"stall_count": 0, "max_depth": 3},
            },
            *[
                {
                    "run_id": f"r_stall_{i}",
                    "seed": 1001 + i,
                    "depth": 1,
                    "max_depth": 1,
                    "score": 15,
                    "turns": 500,
                    "steps": 500,
                    "end_reason": "max_steps",
                    "unique_tiles": 22,
                    "pct_search": 0.85,
                    "triggered_functions": {},
                    "fn_turns": {},
                    "summary_digest": "CONFINEMENT_STALL on Dlvl 1",
                    "milestones": {"stall_count": 1, "max_depth": 1},
                }
                for i in range(3)
            ],
        ],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.5, "max_depth": 3, "avg_score": 120.0},
        verdict={"passed": True},
    )

    analyzer = DeficitAnalyzer(db_path)
    report = analyzer.analyze(generation=2, current_policy=base_pol, skills=base_skills)
    assert report.fail_clusters
    assert "fn_a" in report.dormant_branches
    assert any(b["fn"] == "fn_b" for b in report.protected_branches)

    # Bad candidate raising runtime error is rejected in isolated sandbox incubation
    bad_cand = PeerCandidate(
        peer_id="peer_bad",
        claimed_deficit_id="confinement_stall",
        claimed_label="stall",
        policy_code=base_pol,
        skills={
            "skill_a": "def fn_a(st):\n    raise RuntimeError('boom')\n",
            "skill_b": base_skills["skill_b"],
        },
    )
    bad_res = incubate_candidate(
        bad_cand, deficit_report=report, claimed_deficit=DeficitCluster("c", "k", "l")
    )
    assert bad_res.sandbox_passed is False
    assert "RuntimeError" in bad_res.error or "boom" in bad_res.error

    # Disjoint skill updates merge cleanly via SwarmReferee.build_orthogonal_pair
    cand_a = PeerCandidate(
        peer_id="peer_1",
        claimed_deficit_id="c1",
        claimed_label="c1",
        policy_code=base_pol,
        skills={
            "skill_a": "from lox.env.actions import Action\ndef fn_a(st):\n    return Action.EAST\n",
            "skill_b": base_skills["skill_b"],
        },
        touched_skills={"skill_a"},
        touched_branches=set(),
        sandbox_passed=True,
        sandbox_metrics={"depth_delta": 0.2, "score_delta": 10.0, "wrapper_ratio": 0.0},
    )
    cand_b = PeerCandidate(
        peer_id="peer_2",
        claimed_deficit_id="c2",
        claimed_label="c2",
        policy_code=base_pol,
        skills={
            "skill_a": base_skills["skill_a"],
            "skill_b": "from lox.env.actions import Action\ndef fn_b(st):\n    return Action.WAIT\n",
        },
        touched_skills={"skill_b"},
        touched_branches=set(),
        sandbox_passed=True,
        sandbox_metrics={"depth_delta": 0.1, "score_delta": 5.0, "wrapper_ratio": 0.0},
    )
    referee = SwarmReferee()
    ortho = referee.build_orthogonal_pair([cand_a, cand_b], base_pol, base_skills)
    assert ortho is not None
    assert "Action.EAST" in ortho.skills["skill_a"]
    assert "Action.WAIT" in ortho.skills["skill_b"]

    class _FakePool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            is_inc = "Action.SEARCH" in (skills or {}).get("skill_b", "")
            depth = 2 if is_inc else 1
            score = 50.0 if is_inc else 5.0
            return {
                "batch_id": batch_id,
                "avg_depth": float(depth),
                "max_depth": depth,
                "avg_score": score,
                "p90_depth": float(depth),
                "max_score": int(score),
                "avg_turns": 10.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": depth,
                        "max_depth": depth,
                        "score": int(score),
                        "turns": 10,
                        "steps": 10,
                        "end_reason": "max_steps",
                        "fn_turns": {"fn_b": 10},
                        "summary_digest": "ok",
                    }
                    for sd in priority_seeds
                ],
            }

    verdict = referee.evaluate_tournament(
        generation=2,
        candidates=[cand_b],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_FakePool(),
        seeds=[1000, 1001],
    )
    assert verdict.promoted is False
    assert verdict.winner_id == "incumbent"
    assert verdict.winner_policy_code == base_pol

    class _ScoreCollapsePool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            is_inc = "Action.SEARCH" in (skills or {}).get("skill_b", "")
            avg_d = 2.75 if is_inc else 3.00
            avg_s = 700.0 if is_inc else 400.0
            return {
                "batch_id": batch_id,
                "avg_depth": avg_d,
                "max_depth": 3,
                "avg_score": avg_s,
                "p90_depth": 3.0,
                "max_score": int(avg_s),
                "avg_turns": 100.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": 3 if (not is_inc or idx < 3) else 2,
                        "max_depth": 3 if (not is_inc or idx < 3) else 2,
                        "score": int(avg_s),
                        "turns": 100,
                        "steps": 100,
                        "end_reason": "max_steps",
                        "fn_turns": {"fn_b": 100},
                        "summary_digest": "ok",
                    }
                    for idx, sd in enumerate(priority_seeds)
                ],
            }

    collapse_verdict = referee.evaluate_tournament(
        generation=3,
        candidates=[cand_b],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_ScoreCollapsePool(),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert collapse_verdict.promoted is False
    assert collapse_verdict.winner_id == "incumbent"

    eval_calls: list[str] = []

    class _CountingPool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            eval_calls.append(batch_id)
            return {
                "batch_id": batch_id,
                "avg_depth": 2.0,
                "max_depth": 2,
                "avg_score": 100.0,
                "p90_depth": 2.0,
                "max_score": 100,
                "avg_turns": 50.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 100,
                        "turns": 50,
                        "steps": 50,
                        "end_reason": "max_steps",
                        "fn_turns": {"fn_b": 50},
                        "summary_digest": "ok",
                    }
                    for sd in priority_seeds
                ],
            }

    c_pool = _CountingPool()
    referee.evaluate_tournament(
        generation=4,
        candidates=[cand_a],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
    )
    assert len(eval_calls) == 2
    cand_a2 = PeerCandidate(
        peer_id="peer_a2",
        claimed_deficit_id="explore",
        claimed_label="fresh variant",
        policy_code=base_pol + "\ndef _variant_step():\n    return 1\n",
        skills=base_skills,
        sandbox_passed=True,
        loc_total=len(base_pol.splitlines()),
    )
    eval_calls.clear()
    referee.evaluate_tournament(
        generation=5,
        candidates=[cand_a2],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
    )
    assert len(eval_calls) == 1

    cand_a3 = PeerCandidate(
        peer_id="peer_a3",
        claimed_deficit_id="explore",
        claimed_label="fresh tier variant",
        policy_code=base_pol + "\ndef _variant_step_2():\n    return 2\n",
        skills=base_skills,
        sandbox_passed=True,
        loc_total=len(base_pol.splitlines()),
    )
    eval_calls.clear()
    referee.evaluate_tournament(
        generation=6,
        candidates=[cand_a3],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
        max_steps=10000,
    )
    assert len(eval_calls) == 2

    deep_cp = FrontierCheckpoint(
        checkpoint_id="fc_test_d3",
        seed=1000,
        depth=3,
        milestone="dlvl_3",
        turns=50,
        action_bytes=bytes([ord("s")] * 20),
    )
    ok_res = incubate_candidate(
        cand_a,
        deficit_report=report,
        claimed_deficit=DeficitCluster("c", "k", "l"),
        frontier_checkpoints=[deep_cp],
    )
    assert ok_res.sandbox_passed is True
    assert "frontier_survived" in ok_res.sandbox_metrics

    large_pol = base_pol + "\n" + "\n".join(f"# pad {i}" for i in range(410))
    big_report = analyzer.analyze(
        generation=6, current_policy=large_pol, skills=base_skills
    )
    assert big_report.bloat_warning is True
    assert not any(c.cluster_id == "slice_modularize" for c in big_report.fail_clusters)
    assert "DITCH MONOLITH" in big_report.format_brief()


def test_swarm_peer_multi_skill_surgical_splice_and_placeholder_guard(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.author.tools import AuthorToolRegistry

    base_policy = (
        "from lox.env.actions import Action\n\n"
        "def leaf_wait(state):\n"
        "    return Action.WAIT\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return leaf_wait(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "t.duckdb"))
    tools.configure_generation(mode="SWARM_PEER", generation=2)
    tools.current_policy_code = base_policy

    res_sp, sum_sp = tools.dry_run_policy(
        code="def leaf_wait(state):\n    return Action.SEARCH\n",
        max_turns=20,
        smoke_only=True,
    )
    assert sum_sp.startswith("PASSED"), res_sp
    assert "class Policy:" in tools.last_passed_code
    assert "Action.SEARCH" in tools.last_passed_code

    thin_policy = (
        "from lox.env.actions import Action\n"
        "from skills.mod_a import fn_a\n"
        "from skills.mod_b import fn_b\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return fn_a(state) or fn_b(state) or Action.WAIT\n"
    )
    skills_bundle = {
        "mod_a": "def fn_a(state):\n    return None\n",
        "mod_b": "from lox.env.actions import Action\ndef fn_b(state):\n    return Action.SEARCH\n",
    }
    res_ms, sum_ms = tools.dry_run_policy(
        code=thin_policy,
        skills=skills_bundle,
        max_turns=20,
        smoke_only=True,
    )
    assert sum_ms.startswith("PASSED"), res_ms
    assert set(tools.last_passed_skills.keys()) >= {"mod_a", "mod_b"}

    has_stmts, has_cls = AuthorAgent._inspect_python_fragment(
        "# The complete class Policy: source is already validated above"
    )
    assert has_stmts is False and has_cls is False


def test_prune_empty_skills_unlinks_disk_and_deficit_pct_scale(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer

    sk_dir = tmp_path / "skills"
    sk_dir.mkdir()
    stale = sk_dir / "swarm_core.py"
    stale.write_text("x = 1\n", encoding="utf-8")
    kept = prune_unused_skills("class Policy:\n    pass\n", {}, skills_dir=sk_dir)
    assert kept == {}
    assert not stale.exists()

    db_p = str(tmp_path / "deficits.duckdb")
    init_db(db_p)
    con = connect_db(db_p, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, avg_depth, max_depth) VALUES ('b1', 1, 'SWARM', 3.0, 6)"
        )
        for idx in range(6):
            er = "killed" if idx < 4 else "stalled_zero_turn"
            pw = 20.0 if idx < 4 else 70.0
            fn = "fight_adjacent" if idx < 2 else "explore_frontier"
            con.execute(
                """
                INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, steps,
                                  end_reason, pct_wait, pct_search, last_active_function)
                VALUES (?, 'b1', 1, ?, 'val', ?, ?, 200, ?, 300, ?, ?, 5.0, ?)
                """,
                [
                    f"r{idx}",
                    1000 + idx,
                    4 if idx < 2 else 1,
                    4 if idx < 2 else 1,
                    500 if idx < 2 else 120,
                    er,
                    pw,
                    fn,
                ],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(db_p).analyze(2)
    cids = {c.cluster_id for c in rep.fail_clusters}
    assert "slice_stalled" in cids
    assert "slice_frontier_failures" in cids or "slice_early_terminations" in cids
    stalled = next(c for c in rep.fail_clusters if c.cluster_id == "slice_stalled")
    assert stalled.run_count == 2
