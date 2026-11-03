"""
Tests for Multi-Table DuckDB Schema, RLE + Inflection Summarizer,
Statistical Outlier Good-Run Detection (`is_outlier_good`), Ochiai/Liblit Function Stats,
Smart Tail-Based Tick Sampling, and AuthorToolRegistry (`query_db`, `query_wiki`, `query_policy_docs`, `dry_run_policy`).
"""

from __future__ import annotations

from lox.author.tools import AuthorToolRegistry
from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import init_db
from lox.db.summarizer import (
    compute_function_stats,
    mark_outlier_good_runs,
    summarize_episode_ticks,
)


def test_rle_summarizer_and_outlier_good_marker():
    ticks = [
        {
            "run_id": "r1",
            "turn": 1,
            "step_idx": 1,
            "depth": 1,
            "dungeon_num": 0,
            "hp": 16,
            "max_hp": 16,
            "ac": 6,
            "xp_level": 1,
            "hunger_state": 1,
            "y": 10,
            "x": 20,
            "action_name": "SEARCH",
            "raw_key": ord("s"),
            "active_function": "explore",
            "message": "Hello Valkyrie!",
            "screen_ascii": "",
        },
        {
            "run_id": "r1",
            "turn": 2,
            "step_idx": 2,
            "depth": 2,
            "dungeon_num": 0,
            "hp": 14,
            "max_hp": 16,
            "ac": 3,
            "xp_level": 2,
            "hunger_state": 1,
            "y": 10,
            "x": 21,
            "action_name": "DOWN",
            "raw_key": ord(">"),
            "active_function": "explore",
            "message": "You descend the stairs.",
            "screen_ascii": "",
        },
        {
            "run_id": "r1",
            "turn": 3,
            "step_idx": 3,
            "depth": 2,
            "dungeon_num": 0,
            "hp": 4,
            "max_hp": 16,
            "ac": 3,
            "xp_level": 2,
            "hunger_state": 2,
            "y": 10,
            "x": 21,
            "action_name": "NORTH",
            "raw_key": ord("k"),
            "active_function": "combat",
            "message": "The gnome hits! You are beginning to feel hungry.",
            "screen_ascii": "",
        },
    ]
    summary_digest, game_log, milestones = summarize_episode_ticks(ticks)
    assert "Overview:" in summary_digest
    assert "Phase Timeline (RLE):" in summary_digest
    assert "Key Inflections:" in summary_digest
    assert "D1->D2" in summary_digest
    assert "The gnome hits!" in game_log
    assert milestones["max_depth"] == 2
    assert milestones["best_ac"] == 3

    # Test statistical outlier marker (`is_outlier_good`) and Ochiai/Liblit function stats
    runs = []
    for i in range(20):
        is_star = i == 19
        runs.append(
            {
                "run_id": f"run_{i}",
                "seed": 100 + i,
                "depth": 9 if is_star else 1,
                "max_depth": 9 if is_star else 1,
                "score": 4500 if is_star else 50,
                "turns": 800 if is_star else 90,
                "steps": 800 if is_star else 90,
                "xp_level": 6 if is_star else 1,
                "final_hp": 25 if is_star else 0,
                "max_hp": 40 if is_star else 16,
                "final_ac": -2 if is_star else 6,
                "gold": 100 if is_star else 0,
                "hunger_state": 1,
                "end_reason": "ended on Dlvl 9" if is_star else "killed by a jackal",
                "death_message": "killed by a jackal",
                "summary_digest": summary_digest,
                "game_log": game_log,
                "milestones_json": "{}",
                "functions_triggered": (
                    {"explore": 10, "combat": 5, "forge_excalibur": 2}
                    if is_star
                    else {"explore": 10, "bad_panic": 4}
                ),
                "function_turns": (
                    {"explore": 500, "combat": 200, "forge_excalibur": 100}
                    if is_star
                    else {"explore": 60, "bad_panic": 30}
                ),
                "action_bytes": b"....",
                "ticks": ticks,
            }
        )

    mark_outlier_good_runs(runs)
    assert runs[19]["is_outlier_good"] is True
    assert runs[19]["outlier_zscore"] > 2.0
    assert sum(1 for r in runs if r["is_outlier_good"]) == 1

    fn_stats = compute_function_stats("batch_test", runs)
    by_name = {f["function_name"]: f for f in fn_stats}
    assert by_name["forge_excalibur"]["ochiai_good_score"] == 1.0
    assert by_name["bad_panic"]["liblit_fail_importance"] > 0.0


def test_duckdb_logger_and_author_tools(tmp_path):
    db_file = str(tmp_path / "test_lox.duckdb")
    macros_file = str(tmp_path / "macros.py")
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
    tools = AuthorToolRegistry(db_path=db_file, macros_path=macros_file)

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
    assert "forbidden" in blocked.lower()

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


def test_stall_cause_classification_and_progression_metrics():
    from lox.db.summarizer import summarize_episode_ticks
    from lox.engine.hypothesis import progression_metric

    # Simulate ticks with pet displacement loop
    pet_ticks = [
        {
            "turn": t,
            "step_idx": t,
            "depth": 1,
            "dungeon_num": 0,
            "hp": 16,
            "max_hp": 16,
            "energy": 2,
            "ac": 6,
            "xp_level": 1,
            "score": 0,
            "hunger_state": 1,
            "condition_mask": 0,
            "y": 10,
            "x": 20 if t % 2 == 0 else 21,
            "action_name": "E" if t % 2 == 0 else "W",
            "raw_key": ord("l") if t % 2 == 0 else ord("h"),
            "active_function": "explore",
            "message": "You swap places with your kitten." if t % 2 == 0 else "",
        }
        for t in range(1, 55)
    ]
    digest, _, milestones = summarize_episode_ticks(pet_ticks)
    assert "pet displacement loop" in digest
    assert "unique_tiles_d1" in milestones
    assert milestones["unique_tiles_d1"] == 2

    # Verify progression metric rewards stairs discovery and unique tiles
    run_stalled = {
        "max_depth": 1,
        "score": 0,
        "xp_level": 1,
        "final_ac": 6,
        "unique_tiles": 2,
        "stairs_down_found": False,
    }
    run_explored = {
        "max_depth": 1,
        "score": 0,
        "xp_level": 1,
        "final_ac": 6,
        "unique_tiles": 50,
        "stairs_down_found": True,
    }
    assert progression_metric(run_explored) > progression_metric(run_stalled) + 0.20


def test_dynamic_failure_telemetry_digest():
    from lox.db.summarizer import build_batch_contrast_digest

    batch_stats = {
        "batch_id": "b_test_01",
        "max_depth": 1,
        "avg_depth": 1.0,
        "p90_depth": 1.0,
        "episodes_discarded": 0,
    }
    runs = [
        {
            "run_id": "r1",
            "seed": 42,
            "max_depth": 1,
            "score": 0,
            "turns": 1000,
            "steps": 1000,
            "end_reason": "max_steps",
            "unique_tiles": 15,
            "stairs_down_found": False,
            "summary_digest": "Stalls: fn=explore on Dlvl 1 [locked door stall] (T10..T100)\nOverview: Depth=1",
        },
        {
            "run_id": "r2",
            "seed": 43,
            "max_depth": 1,
            "score": 0,
            "turns": 1000,
            "steps": 1000,
            "end_reason": "max_steps",
            "unique_tiles": 22,
            "stairs_down_found": True,
            "summary_digest": "Stalls: fn=explore on Dlvl 1 [pet displacement loop] (T50..T150)\nOverview: Depth=1",
        },
    ]
    fn_stats = [
        {
            "function_name": "explore",
            "trigger_rate": 1.0,
            "total_turns_active": 2000,
            "avg_depth_when_triggered": 1.0,
            "death_rate_while_active": 0.0,
            "ochiai_good_run_score": 0.0,
            "liblit_failure_score": 0.0,
        }
    ]

    digest = build_batch_contrast_digest(batch_stats, runs, fn_stats)

    # Must contain dynamic telemetry
    assert "Dynamic Failure Telemetry" in digest
    assert "locked door stall" in digest
    assert "pet displacement loop" in digest
    assert "Down Stairs (`>`) Discovery Rate" in digest
    assert "Avg 18.5 unique tiles mapped" in digest

    # Must NOT contain old misleading hardcoded text
    assert "Cyclic Positional Oscillations (Ping-Pong Loops)" not in digest
    assert "Stuck Detector Blindness" not in digest
    assert "Use a ring buffer of recent positions" not in digest
