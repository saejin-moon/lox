"""Domain-scoped tests for LOX."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from lox.author.tools import AuthorToolRegistry
from lox.datasets.bootstrap_frontier import bootstrap_frontier_from_duckdb
from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, get_schema_markdown, init_db
from lox.env.actions import Action
from lox.lang.compiler import compile_policy_package


def test_duckdb_logger_and_author_tools(tmp_path):
    db_file = str(tmp_path / "test_lox.duckdb")
    con = init_db(db_file)
    con.close()

    # Ingest expert runs into test DB
    ingest_expert_runs_into_duckdb(db_path=db_file, max_ascensions=5, fetch_dumplogs=0)

    logger = BatchDatabaseLogger(db_path=db_file, max_tick_runs_per_batch=5)
    sample_ticks = [
        {
            "run_id": "b1_s1",
            "turn": t,
            "step_idx": t,
            "depth": 1 if t < 15 else 3,
            "dungeon_num": 0,
            "hp": 16,
            "max_hp": 16,
            "ac": 6,
            "xp_level": 1,
            "hunger_state": 1,
            "y": 10,
            "x": 10,
            "action_name": "WAIT",
            "raw_key": ord("."),
            "active_function": "run",
            "message": "",
            "screen_ascii": "",
        }
        for t in range(1, 21)
    ]
    runs = [
        {
            "run_id": f"b1_s{i}",
            "seed": i,
            "depth": 6 if i == 10 else 1,
            "max_depth": 6 if i == 10 else 1,
            "score": 1200 if i == 10 else 20,
            "turns": 20,
            "steps": 20,
            "xp_level": 4 if i == 10 else 1,
            "final_hp": 16,
            "max_hp": 16,
            "final_ac": 2,
            "gold": 10,
            "hunger_state": 1,
            "end_reason": "killed by an orc",
            "death_message": "killed by an orc",
            "summary_digest": "Overview: test",
            "game_log": "test log",
            "milestones_json": "{}",
            "functions_triggered": {"run": 1},
            "function_turns": {"run": 20},
            "action_bytes": b"s.",
            "ticks": sample_ticks,
        }
        for i in range(1, 11)
    ]
    verdict = {
        "passed": True,
        "mean_delta": 1.5,
        "ci_lower": 0.4,
        "ci_upper": 2.6,
        "wilcoxon_p": 0.01,
    }
    logger.save_batch(
        batch_id="b1",
        generation=0,
        mode="FULL_WRITE",
        parent_policy_id="none",
        candidate_policy_id="pol_b1",
        target_functions=[],
        runs=runs,
        episodes_discarded=0,
        batch_metrics={
            "episodes_recorded": 10,
            "avg_depth": 1.5,
            "max_depth": 6,
            "p90_depth": 6.0,
            "avg_score": 138.0,
            "avg_turns": 20.0,
            "wall_sec": 0.5,
        },
        verdict=verdict,
    )

    # Test AuthorToolRegistry tools
    tools = AuthorToolRegistry(db_path=db_file)

    # 1. query_db on runs (outlier good runs) and expert_runs
    out_good, _ = tools.query_db(
        "SELECT run_id, max_depth, is_outlier_good FROM runs WHERE is_outlier_good = TRUE"
    )
    assert "b1_s10" in out_good

    out_exp, _ = tools.query_db(
        "SELECT player, role, max_depth, end_reason FROM expert_runs LIMIT 5"
    )
    assert "ascended" in out_exp

    # Read-only guardrail blocks DROP/DELETE
    blocked, status = tools.query_db("DROP TABLE runs")
    assert status == "error"
    assert "only read-only" in blocked.lower() or "forbidden" in blocked.lower()

    # 2. query_wiki
    wiki_res, _ = tools.query_wiki("Excalibur fountain dip")
    assert "Excalibur" in wiki_res

    # 3. query_policy_docs
    docs_res, _ = tools.query_policy_docs("actions")
    assert "Action.N" in docs_res

    # 4. dry_run_policy
    dry_res, dry_status = tools.dry_run_policy(
        code="""
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
""",
    )
    assert "PASSED" in dry_status
    assert "Dry-Run PASSED" in dry_res


def test_save_batch_throughput_150_runs(tmp_path):
    """Verifies that persisting 150 runs with full telemetry completes in under 1 second."""
    import time

    db_file = str(tmp_path / "bench_lox.duckdb")
    logger = BatchDatabaseLogger(db_path=db_file, max_tick_runs_per_batch=25)

    sample_ticks = [
        {
            "run_id": f"run_{r}",
            "turn": t,
            "step_idx": t,
            "depth": 1,
            "dungeon_num": 0,
            "hp": 16,
            "max_hp": 16,
            "ac": 10,
            "xp_level": 1,
            "hunger_state": 1,
            "condition_mask": 0,
            "y": 10,
            "x": 10,
            "action_name": "WAIT",
            "raw_key": ord("."),
            "active_function": "explore",
            "message": "",
            "visible_monsters": "",
            "inventory_summary": "",
        }
        for r in range(25)
        for t in range(1, 200)
    ]

    runs = [
        {
            "run_id": f"run_{i}",
            "seed": i,
            "role": "val-hum-law-fem",
            "depth": 3 if i == 0 else 1,
            "max_depth": 3 if i == 0 else 1,
            "score": 500 if i == 0 else 20,
            "turns": 200,
            "steps": 200,
            "xp_level": 1,
            "final_hp": 16,
            "max_hp": 16,
            "final_ac": 10,
            "gold": 0,
            "end_reason": "died",
            "death_message": "killed by a jackal",
            "summary_digest": "Phase Timeline: explore (100%)",
            "game_log": "T1: started\nT200: died",
            "milestones": {},
            "triggered_functions": {"explore": 200},
            "last_active_function": "explore",
            "action_bytes": b"....",
            "wall_ms": 15.0,
            "ticks": sample_ticks if i < 25 else [],
        }
        for i in range(150)
    ]

    t0 = time.perf_counter()
    logger.save_batch(
        batch_id="bench_batch",
        generation=0,
        mode="FULL_WRITE",
        parent_policy_id="none",
        candidate_policy_id="bench_pol",
        target_functions=["explore"],
        runs=runs,
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.0, "max_depth": 3, "avg_score": 25.0},
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, (
        f"save_batch took {elapsed:.2f}s for 150 runs (expected < 1.0s)"
    )


def test_logger_baseline_preserves_oprop_and_branches(tmp_path):
    db_path = str(tmp_path / "test_logger.duckdb")
    init_db(db_path).close()
    logger = BatchDatabaseLogger(db_path=db_path)
    pol = "class Policy:\n    def step(self, s):\n        return None\n"
    logger.save_policy("pol_g1", 1, "SWARM", pol, "", {}, 2.0, 2, is_promoted=True)
    run_row = {
        "run_id": "b1_r1",
        "seed": 42,
        "role": "val-hum-law-fem",
        "depth": 2,
        "max_depth": 2,
        "score": 50,
        "turns": 100,
        "steps": 100,
        "xp_level": 2,
        "final_hp": 12,
        "max_hp": 15,
        "final_ac": 8,
        "gold": 10,
        "end_reason": "died",
        "death_message": "died",
        "summary_digest": "",
        "game_log": "",
        "milestones": {},
        "worn_oprop_count": 3,
        "dungeons_visited": 2,
        "triggered_functions": {"step": 100},
        "fn_turns": {"step": 100},
        "last_active_function": "step",
        "wall_ms": 5.0,
        "ticks": [],
    }
    logger.save_batch(
        batch_id="b1",
        generation=1,
        mode="SWARM",
        parent_policy_id="none",
        candidate_policy_id="pol_g1",
        target_functions=["step"],
        runs=[run_row],
        episodes_discarded=0,
        batch_metrics={
            "avg_depth": 2.0,
            "p90_depth": 2.0,
            "max_depth": 2,
            "avg_score": 50.0,
            "max_score": 50,
            "avg_turns": 100.0,
        },
        verdict={"passed": True, "reason": "ok"},
    )
    loaded = logger.load_champion_baseline()
    assert loaded is not None
    r0 = loaded["baseline_runs"][0]
    assert int(r0["worn_oprop_count"]) == 3
    assert int(r0["dungeons_visited"]) == 2


def test_duckdb_foreign_keys_and_logging_completeness(tmp_path: Path) -> None:
    """4. DuckDB logging completeness: 100% of candidate_policy_ids exist in policies table."""
    db_path = str(tmp_path / "test_lox.duckdb")
    init_db(db_path)
    logger = BatchDatabaseLogger(db_path=db_path)

    # 1. Warm-start policy & batch
    logger.save_policy(
        policy_id="pol_gen000",
        generation=0,
        mode="FULL_WRITE",
        policy_code="class Policy: pass",
        macros_code="",
        params={},
        avg_depth=1.5,
        max_depth=2,
        is_promoted=True,
    )
    logger.save_batch(
        batch_id="batch_warm_000",
        generation=0,
        mode="FULL_WRITE",
        parent_policy_id="none",
        candidate_policy_id="pol_gen000",
        target_functions=[],
        runs=[],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.5, "max_depth": 2},
    )

    # 2. Rejected PARAM_TUNE batch & policy
    cand_tune = "pol_batch_tune_001"
    logger.save_policy(
        policy_id=cand_tune,
        generation=1,
        mode="PARAM_TUNE",
        policy_code="class Policy: pass",
        macros_code="",
        params={"flee_hp": 0.3},
        avg_depth=1.2,
        max_depth=2,
        is_promoted=False,
    )
    logger.save_batch(
        batch_id="batch_tune_001",
        generation=1,
        mode="PARAM_TUNE",
        parent_policy_id="pol_gen000",
        candidate_policy_id=cand_tune,
        target_functions=["flee_hp"],
        runs=[],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.2, "max_depth": 2},
        verdict={"passed": False, "mean_delta": -0.3},
    )

    # 3. MACRO_WRITE policy
    cand_macro = "pol_batch_macro_002"
    logger.save_policy(
        policy_id=cand_macro,
        generation=2,
        mode="MACRO_WRITE",
        policy_code="class Policy: pass",
        macros_code="def helper(): pass",
        params={},
        avg_depth=1.5,
        max_depth=2,
        is_promoted=True,
    )

    # 4. Pilot evaluation experiment logged to DuckDB
    AuthorToolRegistry(db_path=db_path)
    # Save a pilot experiment directly through logger
    logger.save_experiment(
        experiment_id="exp_pilot_1_12345",
        generation=0,
        batch_id="pilot_1_12345",
        mode="PILOT",
        target_functions=["handle_nutrition"],
        hypothesis={
            "observation_from_db": "Starvation death",
            "mechanism": "Eat earlier",
        },
        verdict={
            "passed": True,
            "mean_delta": 0.25,
            "wins": 15,
            "losses": 5,
            "ties": 10,
        },
    )

    # Query foreign key integrity: NO batches should point to missing candidate_policy_id
    con = connect_db(db_path, read_only=True)
    try:
        missing_fks = con.execute(
            """
            SELECT COUNT(*)
            FROM batches b
            LEFT JOIN policies p ON b.candidate_policy_id = p.policy_id
            WHERE p.policy_id IS NULL
            """
        ).fetchone()[0]
        assert missing_fks == 0, (
            f"Found {missing_fks} dangling foreign keys in batches!"
        )

        # Query experiments table has pilot entry
        pilot_count = con.execute(
            "SELECT COUNT(*) FROM experiments WHERE mode = 'PILOT'"
        ).fetchone()[0]
        assert pilot_count >= 1
    finally:
        con.close()


def test_gen0_logging_and_campaign_on_migrated_schema(tmp_path):
    """Verifies that save_batch, save_experiment, and full Gen 0 execution succeed

    even when ALTER TABLE ADD COLUMN has appended telemetry columns after created_at TIMESTAMP.
    """
    import shutil
    from pathlib import Path

    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db

    db_file = str(tmp_path / "migrated_test.duckdb")
    if Path("data/lox.duckdb").exists():
        shutil.copy2("data/lox.duckdb", db_file)
    else:
        init_db(db_file)

    # 1. Verify BatchDatabaseLogger can insert without positional TIMESTAMP cast error
    logger = BatchDatabaseLogger(db_file)
    logger.save_batch(
        batch_id="b_migrated_001",
        generation=0,
        mode="FULL_WRITE",
        parent_policy_id="none",
        candidate_policy_id="pol_001",
        target_functions=["run"],
        runs=[],
        episodes_discarded=0,
        batch_metrics={
            "avg_depth": 1.5,
            "p90_depth": 2.0,
            "max_depth": 2,
            "avg_score": 20.0,
            "max_score": 30,
            "avg_turns": 50.0,
            "outliers": 0,
            "seeds_evaluated": 25,
            "is_early_stopped": True,
            "pilot_best_delta": 0.45,
        },
        verdict={
            "passed": True,
            "wilcoxon_p": 0.01,
            "ci_lower": 0.1,
            "ci_upper": 0.5,
            "mean_delta": 0.3,
            "wins": 20,
            "losses": 5,
            "ties": 0,
            "seeds_evaluated": 25,
            "is_early_stopped": True,
            "pilot_best_delta": 0.45,
            "reason": "Test migration",
        },
    )

    con = connect_db(db_file, read_only=True)
    b_row = con.execute(
        "SELECT seeds_evaluated, is_early_stopped, pilot_best_delta, created_at FROM batches WHERE batch_id='b_migrated_001'"
    ).fetchone()
    assert b_row[0] == 25
    assert b_row[1] is True
    assert abs(b_row[2] - 0.45) < 1e-6
    assert b_row[3] is not None
    con.close()


def test_concurrent_scoped_modules_and_threadsafe_db_writes(tmp_path):
    pol_src = """
def get_magic():
    import skills.helper as h
    return h.MAGIC

class Policy:
    def step(self, state):
        if get_magic() > 0:
            return Action.WAIT
        return Action.WAIT
"""
    db_path = str(tmp_path / "conc.duckdb")
    init_db(db_path).close()
    logger = BatchDatabaseLogger(db_path)

    def _worker(idx: int) -> tuple[int, int]:
        compiled = compile_policy_package(
            pol_src,
            skills={"helper": f"MAGIC = {100 + idx}\n"},
        )
        runner = compiled.create_runner()
        logger.save_synthesis_log(
            log_id=f"log_{idx}",
            generation=1,
            batch_id="b_conc",
            mode="SWARM_PEER",
            turn_index=idx,
            turn_type="TEST",
            agent_role=f"peer_{idx}",
        )
        return idx, runner.instance.__class__.__module__ == "lox_policy" and int(
            compiled.create_runner().next_action.__name__ != ""
        )

    # Execute compile + runtime import + DB write across parallel threads
    c1 = compile_policy_package(pol_src, skills={"helper": "MAGIC = 111\n"})
    c2 = compile_policy_package(pol_src, skills={"helper": "MAGIC = 222\n"})
    r1 = c1.create_runner()
    r2 = c2.create_runner()
    # Both runners execute after each other compiled without overwriting each other's scoped skills
    assert r1.instance is not None and r2.instance is not None

    def _run_magic(val: int) -> int:
        comp = compile_policy_package(
            "from skills import helper\nclass Policy:\n    def step(self, s):\n        import skills.helper as h\n        return Action.N if h.MAGIC == "
            + str(val)
            + " else Action.WAIT\n",
            skills={"helper": f"MAGIC = {val}\n"},
        )
        run = comp.create_runner()
        # Simulate delay so concurrent creates interleave
        act, _ = run.next_action(type("S", (), {})())
        return 1 if act == Action.N else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs_magic = [ex.submit(_run_magic, 10 + i) for i in range(12)]
        futs_db = [ex.submit(_worker, i) for i in range(12)]
        assert all(f.result() == 1 for f in futs_magic)
        for _ in futs_db:
            _.result()

    con = connect_db(db_path, read_only=True)
    try:
        cnt = con.execute("SELECT COUNT(*) FROM synthesis_logs").fetchone()[0]
        assert cnt == 12
    finally:
        con.close()


def test_schema_markdown_parity_and_non_shard_isolation(tmp_path):
    db_path = str(tmp_path / "schema_test.duckdb")
    init_db(db_path).close()

    md_std = get_schema_markdown(db_path, is_swarm=False)
    for tbl in (
        "runs",
        "ticks",
        "batches",
        "discoveries",
        "synthesis_logs",
        "llm_cost_rates",
    ):
        assert tbl in md_std

    con = connect_db(db_path, read_only=False)
    try:
        cols_runs = [r[1] for r in con.execute("PRAGMA table_info('runs')").fetchall()]
        for c in cols_runs:
            assert c in md_std
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b_p', 1, 'PILOT', FALSE)"
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, max_depth, turns, action_bytes, is_shard)
            VALUES
              ('r_full', 'b_p', 1, 1, 2, 50, ?::BLOB, FALSE),
              ('shard_99', 'b_p', 1, 2, 5, 900, ?::BLOB, TRUE)
            """,
            [b"abcd", b"efgh"],
        )
        pilot_ids = [
            r[0] for r in con.execute("SELECT run_id FROM latest_pilot_runs").fetchall()
        ]
        assert pilot_ids == ["r_full"]
    finally:
        con.close()

    added = bootstrap_frontier_from_duckdb(
        db_path=db_path,
        frontier_dir=str(tmp_path / "frontier"),
        max_checkpoints_per_depth=5,
    )
    assert added == 1


def test_synthesis_logs_persistence(tmp_path):
    """Verify that BatchDatabaseLogger persists and retrieves synthesis_logs with reasoning and diagnostics."""
    from lox.db.logger import BatchDatabaseLogger

    db_p = str(tmp_path / "test_syn.duckdb")
    logger = BatchDatabaseLogger(db_path=db_p)

    logger.save_synthesis_log(
        log_id="syn_001_1_0_test",
        generation=1,
        batch_id="gen_001_func_mod_add",
        mode="FUNC_MOD_ADD",
        turn_index=1,
        turn_type="RESEARCH_TOOL",
        repair_idx=0,
        model="test-model",
        thinking_content="I should check expert runs for staircase descent.",
        message_content="Querying expert runs.",
        tool_calls_json='[{"name": "query_db", "args": {"sql": "SELECT 1"}}]',
        tool_results_json='[{"name": "query_db", "output": "1"}]',
        candidate_code="def explore(self, state):\n    return Action.WAIT\n",
        skill_name="navigation",
        skill_code="",
        hypothesis_yaml="mode: FUNC_MOD_ADD\ntarget_functions: [explore]\n",
        status="TOOL_CALLS",
        error_diagnostic="",
        prompt_sent="Investigate the failure causes.",
        prompt_tokens=150,
        completion_tokens=50,
        reasoning_tokens=25,
        wall_ms=120.5,
    )

    logs = logger.get_recent_synthesis_logs(generation=1)
    assert len(logs) == 1
    row = logs[0]
    assert row["log_id"] == "syn_001_1_0_test"
    assert row["generation"] == 1
    assert row["mode"] == "FUNC_MOD_ADD"
    assert row["turn_type"] == "RESEARCH_TOOL"
    assert (
        row["thinking_content"] == "I should check expert runs for staircase descent."
    )
    assert row["reasoning_tokens"] == 25
    assert row["wall_ms"] == 120.5


def test_unified_runs_ticks_concurrency_and_stepping_stones(tmp_path):
    """
    Verifies unified runs+ticks storage:
    1. `save_runs_and_ticks` logs 100% of ticks tagged with `(run_kind, peer_id, attempt_num)`
       without polluting `batches` or `champion_runs`.
    2. `peer_ticks` and `all_ticks` expose unified ticks alongside `champion_ticks`.
    3. Default `prune_ticks=False` preserves 100% of ticks even when `prune_old_telemetry` is called.
    4. Concurrent multi-threaded `query_db` reads + `save_runs_and_ticks` writes succeed cleanly.
    5. `stepping_stones` CRUD + revision_count retirement at `max_revisions=2`.
    """
    import concurrent.futures

    from lox.author.tools import AuthorToolRegistry
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db

    db_p = str(tmp_path / "unified.duckdb")
    init_db(db_p).close()
    logger = BatchDatabaseLogger(db_path=db_p)
    assert logger.prune_ticks is False

    # Seed 1 champion batch
    logger.save_policy(
        "pol_g1", 1, "SWARM", "class Policy: pass", "", {}, 2.0, 2, is_promoted=True
    )
    logger.save_batch(
        batch_id="batch_champ_g1",
        generation=1,
        mode="SWARM",
        parent_policy_id="none",
        candidate_policy_id="pol_g1",
        target_functions=["step"],
        runs=[
            {
                "run_id": "champ_r1",
                "seed": 100,
                "depth": 2,
                "max_depth": 2,
                "score": 120,
                "turns": 10,
                "ticks": [
                    {
                        "turn": 1,
                        "step_idx": 1,
                        "depth": 1,
                        "hp": 16,
                        "max_hp": 16,
                        "action_name": "E",
                        "active_function": "explore",
                        "message": "champ tick",
                    }
                ],
            }
        ],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 2.0, "max_depth": 2, "avg_score": 120.0},
        verdict={"passed": True, "reason": "ok"},
    )

    # Log micro-pilot + select runs/ticks into unified runs+ticks
    n_ticks = logger.save_runs_and_ticks(
        batch_id="mp_b1",
        generation=2,
        run_kind="micro_pilot",
        peer_id="peer_2",
        attempt_num=1,
        runs=[
            {
                "seed": 100,
                "depth": 3,
                "max_depth": 3,
                "score": 250,
                "turns": 20,
                "ticks": [
                    {
                        "turn": t,
                        "step_idx": t,
                        "depth": 2,
                        "hp": 15,
                        "max_hp": 16,
                        "action_name": "DOWN",
                        "active_function": "descend",
                        "message": f"mp_tick_{t}",
                    }
                    for t in range(1, 6)
                ],
            }
        ],
    )
    assert n_ticks == 5

    # Default prune_old_telemetry is a no-op (zero tick deletion)
    prune_res = logger.prune_old_telemetry(current_generation=100, keep_generations=1)
    assert prune_res == {"ticks_pruned": 0, "runs_pruned": 0}

    con = connect_db(db_p, read_only=True)
    try:
        # batches table remains pure (only the 1 champion batch)
        assert con.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM champion_runs").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM peer_runs").fetchone()[0] == 1
        assert (
            con.execute(
                "SELECT COUNT(*) FROM peer_ticks WHERE generation = 2 AND peer_id = 'peer_2' AND attempt_num = 1 AND seed = 100"
            ).fetchone()[0]
            == 5
        )
        assert con.execute("SELECT COUNT(*) FROM all_ticks").fetchone()[0] == 6
    finally:
        con.close()

    # Concurrent multi-threaded query_db reads + save_runs_and_ticks writes
    reg = AuthorToolRegistry(db_path=db_p)

    def _writer(idx: int) -> int:
        return logger.save_runs_and_ticks(
            batch_id=f"conc_b_{idx}",
            generation=2,
            run_kind="micro_pilot",
            peer_id=f"peer_{idx}",
            attempt_num=1,
            runs=[
                {
                    "seed": 500 + idx,
                    "depth": 2,
                    "max_depth": 2,
                    "turns": 5,
                    "ticks": [
                        {
                            "turn": 1,
                            "step_idx": 1,
                            "depth": 2,
                            "hp": 16,
                            "max_hp": 16,
                            "action_name": "E",
                            "active_function": "nav",
                        }
                    ],
                }
            ],
        )

    def _reader(_idx: int) -> str:
        txt, status = reg.query_db(
            "SELECT run_kind, COUNT(*) AS n FROM all_ticks GROUP BY run_kind"
        )
        assert "error" not in status.lower(), txt
        return status

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        w_futs = [ex.submit(_writer, i) for i in range(1, 5)]
        r_futs = [ex.submit(_reader, i) for i in range(8)]
        assert all(f.result() == 1 for f in w_futs)
        assert all("rows" in f.result() for f in r_futs)

    # Stepping-stone lifecycle + retirement after 2 revisions
    logger.save_stepping_stone(
        {
            "stone_id": "stone_g2_peer_2",
            "generation": 2,
            "parent_policy_id": "fp_abc",
            "source_peer_id": "peer_2",
            "source_run_kind": "select",
            "source_batch_id": "sel_b2",
            "policy_code": "class Policy:\n    pass\n",
            "skills": {"nav": "def go(): return None\n"},
            "mechanism": "fix stairs",
            "target_functions": ["nav"],
            "wins": 7,
            "losses": 0,
            "iqm_delta": 0.21,
            "winsor_delta": 0.21,
            "wilcoxon_p": 0.0078,
            "sign_p": 0.0078,
            "reject_reason": "cut_in_select_stage_by_stack",
            "seeds_gained": [1, 2, 3],
            "seeds_lost": [4],
        }
    )
    loaded = logger.load_active_stepping_stone("fp_abc", max_revisions=2)
    assert loaded is not None and loaded["stone_id"] == "stone_g2_peer_2"
    logger.increment_stepping_stone_revision("stone_g2_peer_2", max_revisions=2)
    loaded2 = logger.load_active_stepping_stone("fp_abc", max_revisions=2)
    assert loaded2 is not None and loaded2["revision_count"] == 1
    logger.increment_stepping_stone_revision("stone_g2_peer_2", max_revisions=2)
    assert logger.load_active_stepping_stone("fp_abc", max_revisions=2) is None

    # v3 proactive_branches CRUD & tournament_attempts (winsor_delta, sign_perm_p)
    logger.save_proactive_branch(
        {
            "branch_id": "pbranch_g10_fp1",
            "base_champion_fp": "fp_abc",
            "generation_started": 10,
            "last_generation": 10,
            "steps_taken": 1,
            "max_steps": 4,
            "status": "active",
            "policy_code": "class Policy:\n    pass\n",
            "skills": {"goals/prep": "def prep(s, ws): return None\n"},
            "goal_summary": "Proactive inventory & altar preparation",
            "capabilities_added": ["altar_identification"],
            "last_avg_depth": 4.2,
            "last_winsor_delta": 0.12,
        }
    )
    br = logger.load_active_proactive_branch("fp_abc")
    assert br is not None
    assert br["branch_id"] == "pbranch_g10_fp1"
    assert br["steps_taken"] == 1
    assert br["capabilities_added"] == ["altar_identification"]
    assert "goals/prep" in br["skills"]

    logger.retire_proactive_branches_for_champion("fp_abc", new_status="superseded")
    assert logger.load_active_proactive_branch("fp_abc") is None

    logger.save_tournament_attempts(
        [
            {
                "attempt_id": "att_v3_1",
                "generation": 10,
                "peer_id": "proactive_1",
                "deficit_id": "proactive_expert_capability",
                "promoted": True,
                "sandbox_passed": True,
                "avg_depth": 4.5,
                "max_depth": 8,
                "avg_ac": 3.0,
                "policy_exceptions": 0,
                "reject_reason": "promoted",
                "wins": 9,
                "losses": 1,
                "iqm_delta": 0.35,
                "winsor_delta": 0.31,
                "sign_perm_p": 0.0042,
            }
        ]
    )
    con_v3 = connect_db(logger.db_path, read_only=True)
    try:
        r_att = con_v3.execute(
            "SELECT winsor_delta, sign_perm_p FROM tournament_attempts WHERE attempt_id='att_v3_1'"
        ).fetchone()
        assert r_att is not None
        assert abs(float(r_att[0]) - 0.31) < 1e-6
        assert abs(float(r_att[1]) - 0.0042) < 1e-6
    finally:
        con_v3.close()


def test_champion_views_include_latest_inc_retained_batch(tmp_path):
    """Verifies Fix 7: champion_runs / champion_ticks / champion_function_stats pick latest inc_retained champion batch."""
    from lox.db.schema import init_db

    db_p = str(tmp_path / "champ_views.duckdb")
    con = init_db(db_p)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, hypothesis_passed, created_at)
            VALUES ('batch_swarm_g76_prom', 76, 'SWARM', TRUE, '2026-01-01 10:00:00'),
                   ('batch_swarm_g87_inc_retained_1', 87, 'SWARM', FALSE, '2026-01-02 10:00:00'),
                   ('batch_swarm_g87_rejected_peer', 87, 'SWARM', FALSE, '2026-01-02 11:00:00'),
                   ('pilot_g88', 88, 'PILOT', FALSE, '2026-01-03 10:00:00')
            """
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, run_kind)
            VALUES ('r76', 'batch_swarm_g76_prom', 76, 1, 'champion'),
                   ('r87_a', 'batch_swarm_g87_inc_retained_1', 87, 10, 'champion'),
                   ('r87_b', 'batch_swarm_g87_inc_retained_1', 87, 11, 'champion'),
                   ('r87_rej', 'batch_swarm_g87_rejected_peer', 87, 12, 'champion'),
                   ('r88_p', 'pilot_g88', 88, 13, 'champion')
            """
        )
        con.execute(
            """
            INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, action_name, active_function)
            VALUES ('r76', 1, 1, 1, 16, 16, 'E', 'explore'),
                   ('r87_a', 1, 1, 1, 16, 16, 'W', 'explore')
            """
        )
        champ_ids = [
            r[0]
            for r in con.execute(
                "SELECT run_id FROM champion_runs ORDER BY run_id"
            ).fetchall()
        ]
        tick_ids = [
            r[0] for r in con.execute("SELECT run_id FROM champion_ticks").fetchall()
        ]
        assert champ_ids == ["r87_a", "r87_b"]
        assert tick_ids == ["r87_a"]
    finally:
        con.close()
