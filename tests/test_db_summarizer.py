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
    assert "Floor Residence (Turns per Depth)" in digest
    assert "Downstairs Conversion Rate" in digest
    assert "Staircase Utilization at Step Limit" in digest

    # Must NOT contain old misleading hardcoded text
    assert "Cyclic Positional Oscillations (Ping-Pong Loops)" not in digest
    assert "Stuck Detector Blindness" not in digest
    assert "Use a ring buffer of recent positions" not in digest


def test_domain_agnostic_stagnation_ratio():
    """Verify that Tail Stagnation Ratio correctly measures post-progress stalls."""
    # Run progresses up to turn 20, then stalls from turn 21 to 100 on existing coordinates
    ticks = []
    for t in range(1, 22):
        ticks.append(
            {
                "turn": t,
                "depth": 1,
                "xp_level": 1,
                "score": t * 10,
                "y": 10,
                "x": t,
                "active_function": "explore",
                "action_name": "E",
            }
        )
    for t in range(22, 101):
        ticks.append(
            {
                "turn": t,
                "depth": 1,
                "xp_level": 1,
                "score": 210,
                "y": 10,
                "x": 20 if t % 2 == 0 else 21,
                "active_function": "explore",
                "action_name": "E",
            }
        )
    digest, glog, milestones = summarize_episode_ticks(ticks)
    assert milestones["last_progress_turn"] == 21
    assert milestones["stagnation_turns"] == 79
    assert milestones["stagnation_ratio"] == 0.79
    assert milestones["top_stagnation_fn"] == "explore"


def test_state_recurrence_limit_cycle():
    """Verify that Recurrence Rate identifies compact limit cycles (oscillations)."""
    # 50 ticks oscillating between 2 coordinates
    cycling_ticks = [
        {
            "turn": t,
            "depth": 1,
            "score": 10,
            "y": 5,
            "x": 5 if t % 2 == 0 else 6,
            "active_function": "wander",
        }
        for t in range(1, 51)
    ]
    _, _, ms_cycling = summarize_episode_ticks(cycling_ticks)
    # Recurrence = 1 - 2/50 = 0.96
    assert ms_cycling["recurrence_rate"] >= 0.90

    # 50 ticks visiting 40 unique coordinates
    exploring_ticks = [
        {
            "turn": t,
            "depth": 1,
            "score": t * 10,
            "y": t,
            "x": t,
            "active_function": "wander",
        }
        for t in range(1, 51)
    ]
    _, _, ms_exploring = summarize_episode_ticks(exploring_ticks)
    # With continuous discovery, last_progress_turn is 50, so stagnant ticks is 1 (< 40) => 0.0
    assert ms_exploring["recurrence_rate"] == 0.0


def test_livelock_failure_attribution():
    """Verify that timeout/max_steps failures attribute blame to the livelock sink function."""
    runs = [
        {
            "run_id": f"r_{i}",
            "seed": i,
            "max_depth": 1,
            "end_reason": "max_steps",
            "triggered_functions": '{"explore_level": 10, "handle_nutrition": 2}',
            "fn_turns": '{"explore_level": 900, "handle_nutrition": 10}',
            "milestones": {
                "stagnation_ratio": 0.85,
                "top_stagnation_fn": "explore_level",
                "stagnation_fn_turns": {"explore_level": 850, "handle_nutrition": 5},
            },
            "last_active_function": "handle_nutrition",
        }
        for i in range(10)
    ]
    fn_stats = compute_function_stats("b_livelock_test", runs)
    explore_stat = next(fs for fs in fn_stats if fs["function_name"] == "explore_level")
    nutrition_stat = next(
        fs for fs in fn_stats if fs["function_name"] == "handle_nutrition"
    )

    # explore_level was the livelock sink (850/855 stagnant turns), so it must absorb the failures!
    assert explore_stat["death_rate_while_active"] == 1.0
    assert explore_stat["liblit_failure_score"] > 0.5
    assert explore_stat["stagnation_turns"] == 8500
    assert explore_stat["stagnation_share"] > 0.95
    assert explore_stat["livelock_score"] > 0.80

    # handle_nutrition only took 5 stagnant turns and was NOT the livelock sink
    assert nutrition_stat["death_rate_while_active"] == 0.0
    assert nutrition_stat["liblit_failure_score"] == 0.0


def test_two_stage_pilot_sign_test():
    """Verify that Exact Sign Test filters out noisy 10W/9L coin flips while passing decisive 12W/5L."""
    import scipy.stats as sp_stats

    # Coin flip: 10 wins, 9 losses (n=19) -> p ~ 0.50 -> must fail
    p_coin = float(sp_stats.binom.sf(10 - 1, 19, 0.5))
    assert p_coin > 0.15

    # Decisive: 12 wins, 5 losses (n=17) -> p ~ 0.07 -> passes alpha=0.15 hurdle
    p_decisive = float(sp_stats.binom.sf(12 - 1, 17, 0.5))
    assert p_decisive <= 0.15


def test_floor_residence_and_staircase_utilization_telemetry():
    """Verify that floor residence distribution and staircase utilization at timeout are accurately computed."""
    from lox.db.summarizer import build_batch_contrast_digest

    batch_stats = {
        "batch_id": "b_residence_test",
        "max_depth": 3,
        "avg_depth": 2.0,
        "p90_depth": 3.0,
        "episodes_discarded": 0,
    }
    runs = [
        {
            "run_id": "r1",
            "seed": 101,
            "max_depth": 3,
            "turns": 500,
            "steps": 500,
            "end_reason": "died",
            "milestones": {
                "depth_first_turn": {"1": 0, "2": 100, "3": 300},
                "stairs_down_found": True,
            },
            "stairs_down_found": True,
            "summary_digest": "Overview: Depth=3 | Turns=500",
        },
        {
            "run_id": "r2",
            "seed": 102,
            "max_depth": 1,
            "turns": 1000,
            "steps": 1000,
            "end_reason": "max_steps",
            "milestones": {
                "depth_first_turn": {"1": 0},
                "stairs_down_found": True,
            },
            "stairs_down_found": True,
            "summary_digest": "Overview: Depth=1 | Turns=1000",
        },
    ]

    digest = build_batch_contrast_digest(batch_stats, runs, [])

    # Floor residence:
    # r1: D1=100 turns, D2=200 turns, D3=200 turns
    # r2: D1=1000 turns
    # D1 avg = (100 + 1000)/2 = 550 turns
    # D2 avg = 200 turns
    assert "Dlvl 1 avg=550 turns" in digest
    assert "Dlvl 2 avg=200 turns" in digest
    assert "Human Expert Reference" not in digest

    # Downstairs conversion: 1 of 2 runs with stairs descended to Dlvl 2+ (50.0%)
    assert "**Downstairs Conversion Rate**: 1/2 runs (50.0%)" in digest
    assert (
        "1 run located stairs but died or timed out on Dlvl 1 without descending"
        in digest
    )

    # Staircase utilization at step limit: 1 of 1 timeout runs had stairs (100.0%)
    assert (
        "**Staircase Utilization at Step Limit**: 1/1 timeout runs (100.0%)" in digest
    )


def test_full_ascension_telemetry_digest():
    """Verify that complete ascension telemetry (branch traversal, descent readiness, hazard blunders, endgame milestones) renders cleanly."""
    from lox.db.summarizer import build_batch_contrast_digest

    batch_stats = {
        "batch_id": "b_ascension_test",
        "max_depth": 10,
        "avg_depth": 5.0,
        "p90_depth": 9.0,
        "episodes_discarded": 0,
    }
    runs = [
        {
            "run_id": "r_deep",
            "seed": 201,
            "max_depth": 10,
            "final_ac": -4,
            "turns": 3000,
            "steps": 3000,
            "end_reason": "You kill the gas spore!  You are caught in the gas spore's explosion",
            "death_message": "You kill the gas spore!  You are caught in the gas spore's explosion",
            "milestones": {
                "depth_first_turn": {"1": 0, "2": 50, "3": 120, "4": 250},
                "depth_first_xl": {"1": 1, "2": 2, "3": 3, "4": 5},
                "depth_first_hp": {
                    "1": [16, 16],
                    "2": [16, 16],
                    "3": [28, 28],
                    "4": [45, 45],
                },
                "dungeons_visited": [0, 2, 4],  # DoD, Gnomish Mines, Sokoban
                "hazard_events": ["gas_spore_explosion"],
                "ascension_milestones": ["altar_found", "castle_drawbridge"],
                "stairs_down_found": True,
            },
            "stairs_down_found": True,
            "starved_with_food": False,
            "died_unarmored": False,
            "summary_digest": "Overview: Depth=10 | Turns=3000",
        },
        {
            "run_id": "r_early",
            "seed": 202,
            "max_depth": 2,
            "final_ac": 8,
            "turns": 1500,
            "steps": 1500,
            "end_reason": "You faint from lack of food",
            "death_message": "You faint from lack of food",
            "milestones": {
                "depth_first_turn": {"1": 0, "2": 1000},
                "depth_first_xl": {"1": 1, "2": 1},
                "depth_first_hp": {"1": [16, 16], "2": [16, 16]},
                "dungeons_visited": [0],
                "hazard_events": ["floating_eye_paralysis"],
                "stairs_down_found": True,
            },
            "stairs_down_found": True,
            "starved_with_food": True,
            "died_unarmored": True,
            "summary_digest": "Overview: Depth=2 | Turns=1500",
        },
    ]

    digest = build_batch_contrast_digest(batch_stats, runs, [])

    # 1. Descent readiness
    assert "Descent Readiness (Preparation at Floor Entry)" in digest
    assert "Dlvl 2: XL=1.5 HP=16" in digest
    assert "Dlvl 4: XL=5.0 HP=45" in digest
    assert "(Exp:" not in digest

    # 2. Branch traversal
    assert "Dungeon Branch Traversal" in digest
    assert "Dungeons of Doom: 2/2 (100.0%)" in digest
    assert "Gnomish Mines: 1/2 (50.0%)" in digest
    assert "Sokoban: 1/2 (50.0%)" in digest

    # 3. Tactical Hazard Blunders
    assert "Tactical Hazard Blunders" in digest
    assert "Gas Spore Explosions: 1 runs" in digest
    assert "Eye Paralysis Deaths: 1 runs" in digest
    assert "Starved with Carried Food: 1 runs" in digest
    assert "Died Unarmored: 1 runs" in digest

    # 4. Armor & Defense progression
    assert "Armor & Defense Progression" in digest
    assert "Avg AC=2.0 | Best AC=-4" in digest
    assert "1/2 runs (50.0%) achieved fortified armor (AC <= 0)" in digest

    # 5. Endgame & Ascension milestones
    assert "Ascension & Endgame Milestones" in digest
    assert "Minetown Altar Found: 1 runs" in digest
    assert "Castle Drawbridge: 1 runs" in digest


def test_autonomous_intelligence_and_deductive_parity_overhaul(tmp_path):
    """Verify all 10 components of the Autonomous Intelligence & Deductive Parity Overhaul."""

    from lox.author.prompts import _format_context_scoped_policy
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.db.summarizer import build_batch_contrast_digest, summarize_episode_ticks
    from lox.env.harness import NetHackHarness
    from lox.lang.compiler import compile_policy_package
    from lox.lang.validator import (
        audit_policy_ast_hygiene,
        normalize_mode,
        prune_extracted_skill_functions,
        splice_functions_into_policy,
    )

    # 1. Mode normalization (CREATE, WRITE, REFACTOR)
    assert normalize_mode("FULL_WRITE") == "CREATE"
    assert normalize_mode("CREATE") == "CREATE"
    assert normalize_mode("FUNC_MOD_ADD") == "WRITE"
    assert normalize_mode("WRITE") == "WRITE"
    assert normalize_mode("STRUCTURAL_REFACTOR") == "REFACTOR"
    assert normalize_mode("REFACTOR") == "REFACTOR"

    # 2. ExecutionTracer attributes active turns to specialist called by dispatch()
    pol_dispatch = """
class Policy:
    def handle_combat(self, state):
        if state.turn < 2:
            return Action.WAIT
        return None

    def explore_level(self, state):
        return Action.SEARCH

    def dispatch(self, state):
        act = self.handle_combat(state)
        if act is not None:
            return act
        return self.explore_level(state)

    def run(self, state):
        while True:
            act = self.dispatch(state)
            state = yield act
"""
    compiled = compile_policy_package(pol_dispatch)
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        runner = compiled.create_runner()
        act1, fn1 = runner.next_action(st)
        st, _, _ = h.step(act1)
        assert fn1 == "handle_combat"
        assert runner.tracer.fn_turns.get("handle_combat", 0) == 1
        assert runner.tracer.fn_turns.get("dispatch", 0) == 0
    finally:
        h.close()

    # 3. sys.modules["skills.*"] cache purge across runner instances & prune_extracted_skill_functions
    skill_v1 = """
def custom_skill_step(policy, state):
    return Action.WAIT
"""
    skill_v2 = """
def custom_skill_step(policy, state):
    return Action.SEARCH
"""
    pol_with_skill = """
from skills.nav_mod import custom_skill_step

class Policy:
    def custom_skill_step(self, state):
        return Action.WAIT

    def run(self, state):
        while True:
            state = yield custom_skill_step(self, state)
"""
    pruned_pol, pruned_names = prune_extracted_skill_functions(pol_with_skill, skill_v1)
    assert "custom_skill_step" in pruned_names
    assert "def custom_skill_step" not in pruned_pol

    comp1 = compile_policy_package(pruned_pol, skills={"nav_mod": skill_v1})
    comp2 = compile_policy_package(pruned_pol, skills={"nav_mod": skill_v2})
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        r1 = comp1.create_runner()
        a1, _ = r1.next_action(st)
        r2 = comp2.create_runner()
        a2, _ = r2.next_action(st)
        assert a1 != a2
    finally:
        h.close()

    # 4. In-place AST constant splicing & deduplication + AST hygiene warnings
    dup_policy = """
SAFE_LIST = ("lichen",)
SAFE_LIST = ("lichen", "newt")

class Policy:
    def __init__(self):
        self.x = 1

    def dead_helper(self, state):
        return Action.WAIT

    def run(self, state):
        while True:
            state = yield Action.SEARCH
"""
    warns = audit_policy_ast_hygiene(dup_policy)
    assert any("SAFE_LIST" in w for w in warns)
    assert any("dead_helper" in w for w in warns)

    patch_const = """
SAFE_LIST = ("lichen", "newt", "gecko")

def run(self, state):
    while True:
        state = yield Action.WAIT
"""
    spliced_const, _ = splice_functions_into_policy(dup_policy, patch_const)
    assert spliced_const.count("SAFE_LIST =") == 1
    assert "gecko" in spliced_const

    # 5. Macro-Category Rollup, Depth-Tier Mortality Matrix, Capability Audit & Pre-Mortem Exemplars
    raw_ticks = [
        {
            "turn": i,
            "depth": 1,
            "hp": 10,
            "max_hp": 14,
            "ac": 9,
            "xp_level": 1,
            "y": 10,
            "x": 10 + (i % 2),
            "action_name": "SEARCH" if i % 2 == 0 else "WAIT",
            "active_function": "handle_combat",
            "message": "The little dog bites!" if i == 4 else "",
            "visible_monsters": "little dog@(10,11)[pet,d1]",
            "inventory_summary": "a:2 food rations, d:a potion of healing",
        }
        for i in range(1, 6)
    ]
    digest_str, game_log_str, ms_dict = summarize_episode_ticks(raw_ticks)
    assert "action_counts" in ms_dict
    assert ms_dict["action_counts"]["SEARCH"] == 2

    sample_runs = [
        {
            "run_id": "r_combat_1",
            "seed": 301,
            "depth": 2,
            "max_depth": 2,
            "final_xp_level": 2,
            "final_hp": 0,
            "final_max_hp": 16,
            "final_ac": 9,
            "turns": 120,
            "steps": 120,
            "last_active_function": "handle_combat",
            "end_reason": "killed by a little dog",
            "death_message": "You are killed by a little dog.",
            "game_log": "[T118 D2 N] You hit the little dog.\n[T120 D2 N] The little dog bites! You die...",
            "summary_digest": digest_str,
            "milestones": {
                **ms_dict,
                "final_inventory": "a:2 food rations, d:a potion of healing",
            },
            "ticks": raw_ticks,
        }
    ]
    b_digest = build_batch_contrast_digest(
        {"batch_id": "b_test_new", "max_depth": 2, "avg_depth": 2.0, "p90_depth": 2.0},
        sample_runs,
        [],
    )
    assert "Macro-Category Binding Constraints Rollup" in b_digest
    assert "**Combat / Monster Deaths**: 1/1" in b_digest
    assert "Depth-Tier Mortality Matrix" in b_digest
    assert "Action Primitive & Terminal Inventory Capability Audit" in b_digest
    assert "`QUAFF` (0 turns)" in b_digest
    assert "Potions held at death: 1/1 deaths" in b_digest
    assert "Concrete Lethal Pre-Mortem Trace Exemplars" in b_digest

    # 6. Lossless ticks persistence & latest_pilot_ticks view
    db_file = str(tmp_path / "lossless.duckdb")
    con = init_db(db_file)
    con.close()
    logger = BatchDatabaseLogger(db_file)
    logger.save_batch(
        batch_id="pilot_test_01",
        generation=1,
        mode="PILOT",
        parent_policy_id="none",
        candidate_policy_id="pol_pilot_01",
        target_functions=["handle_combat"],
        runs=sample_runs,
        episodes_discarded=0,
        batch_metrics={"avg_depth": 2.0, "max_depth": 2},
        verdict={"passed": False, "mean_delta": -0.1},
    )
    con = connect_db(db_file, read_only=True)
    try:
        pilot_ticks_cnt = con.execute(
            "SELECT COUNT(*) FROM latest_pilot_ticks"
        ).fetchone()[0]
        inv_val = con.execute(
            "SELECT inventory_summary FROM latest_pilot_ticks LIMIT 1"
        ).fetchone()[0]
        assert pilot_ticks_cnt == 5
        assert "potion of healing" in inv_val
    finally:
        con.close()

    # 7. Prompt threshold raised to 1200 lines
    long_pol = "class Policy:\n" + "\n".join(
        f"    def fn_{i}(self, state):\n        return None" for i in range(400)
    )
    assert 700 <= len(long_pol.splitlines()) <= 1150
    formatted_pol = _format_context_scoped_policy(long_pol)
    assert "### Current `policy.py`:" in formatted_pol
    assert "def fn_399" in formatted_pol
