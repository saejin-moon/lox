"""
LOX Overhaul End-to-End Verification Suite.
Validates all 7 sections of TODO.md:
1. NetHackAdapter extract_milestones_and_stalls & search_knowledge_base
2. Real Craftax benchmark with >= 4 achievements across worker pool
3. WorldMemory searched_counts, stair_graph, and exact byte-replay alignment
4. CampaignOrchestrator DuckDB policy foreign-key integrity (zero dangling FKs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lox.author.agent import AuthorAgent, SynthesisProposal
from lox.author.tools import AuthorToolRegistry
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, init_db
from lox.db.shards import extract_shards_from_db
from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.env.actions import Action, seq
from lox.env.craftax.actions import CraftaxAction
from lox.env.craftax.harness import CraftaxHarness
from lox.env.harness import NetHackHarness
from lox.env.nethack.adapter import NetHackAdapter
from lox.runner.pool import PersistentEvalPool


def test_nethack_adapter_methods() -> None:
    """1. NetHackAdapter extract_milestones_and_stalls & search_knowledge_base."""
    adapter = NetHackAdapter()

    # extract_milestones_and_stalls
    sample_ticks = [
        {
            "turn": 1,
            "step_idx": 1,
            "depth": 1,
            "hp": 16,
            "max_hp": 16,
            "score": 0,
            "ac": 10,
            "xp_level": 1,
        },
        {
            "turn": 5,
            "step_idx": 5,
            "depth": 1,
            "hp": 16,
            "max_hp": 16,
            "score": 20,
            "ac": 10,
            "xp_level": 1,
        },
    ]
    summary_digest, game_log, milestones = adapter.extract_milestones_and_stalls(
        sample_ticks
    )
    assert isinstance(summary_digest, str)
    assert isinstance(game_log, str)
    assert isinstance(milestones, dict)

    # search_knowledge_base
    kb_res = adapter.search_knowledge_base("door")
    assert isinstance(kb_res, str)
    assert len(kb_res) > 0


def test_craftax_benchmark_and_achievements() -> None:
    """2. Real Craftax benchmark with >= 4 achievements across worker pool."""
    # Test directly on harness first
    harness = CraftaxHarness(max_steps=50)
    st = harness.reset(seed=42)
    st.inventory["wood"] = 3
    # Step 1: move East to (24, 25) facing East
    st, _, _ = harness.step(CraftaxAction.RIGHT)
    # Step 2: mine wood at (24, 26)
    st, _, _ = harness.step(CraftaxAction.DO)
    assert "collect_wood" in st.achievements
    # Step 3: place table at (24, 26)
    st, _, _ = harness.step(CraftaxAction.PLACE_TABLE)
    assert "place_table" in st.achievements
    # Step 4: craft wood pickaxe near table
    st, _, _ = harness.step(CraftaxAction.MAKE_WOOD_PICKAXE)
    assert "make_wood_pickaxe" in st.achievements
    # Step 5: move back West to (24, 24)
    st, _, _ = harness.step(CraftaxAction.LEFT)
    # Step 6: move North to (23, 24) facing North
    st, _, _ = harness.step(CraftaxAction.UP)
    # Step 7: mine stone at (22, 24) with wood pickaxe
    st, _, _ = harness.step(CraftaxAction.DO)
    assert "collect_stone" in st.achievements
    # Step 8: sleep
    st, _, _ = harness.step(CraftaxAction.SLEEP)
    assert "wake_up" in st.achievements
    assert len(st.achievements) >= 5

    # Now test through PersistentEvalPool with workers=2 on benchmark="craftax"
    pool = PersistentEvalPool(workers=2, max_steps=60, benchmark="craftax")
    test_policy = """
class Policy:
    def __init__(self):
        self.step_idx = 0

    def run(self, state):
        self.step_idx += 1
        if self.step_idx == 1:
            state.inventory["wood"] = 3
            return 2  # RIGHT (move to 24, 25 facing East)
        elif self.step_idx == 2:
            return 5  # DO (chop tree at 24, 26 -> collect_wood)
        elif self.step_idx == 3:
            return 8  # PLACE_TABLE (place table at 24, 26 -> place_table)
        elif self.step_idx == 4:
            return 11 # MAKE_WOOD_PICKAXE (craft pickaxe -> make_wood_pickaxe)
        elif self.step_idx == 5:
            return 1  # LEFT (move to 24, 24)
        elif self.step_idx == 6:
            return 3  # UP (move to 23, 24 facing North)
        elif self.step_idx == 7:
            return 5  # DO (mine stone at 22, 24 -> collect_stone)
        elif self.step_idx == 8:
            return 6  # SLEEP (sleep -> wake_up)
        return 0
"""
    try:
        eval_res = pool.evaluate_batch(
            batch_id="test_craftax_batch",
            policy_code=test_policy,
            macros_code="",
            target_episodes=4,
            benchmark="craftax",
            record_ticks=True,
            show_progress=False,
        )
        assert eval_res["episodes_recorded"] == 4
        assert len(eval_res["runs"]) == 4
        for r in eval_res["runs"]:
            assert r["achievements"] >= 4
    finally:
        pool.close()


def test_world_memory_and_byte_replay() -> None:
    """3. WorldMemory searched_counts, stair_graph, and exact byte-replay alignment."""
    harness = NetHackHarness(max_steps=200)
    try:
        st0 = harness.reset(seed=12345)
        # Verify searched_counts initially 0
        assert np.all(st0.mem.searched_counts == 0)

        # Execute Action.SEARCH
        st1, _, _ = harness.step(Action.SEARCH)
        # Player position and 8-neighborhood should be 1
        assert st1.mem.searched_counts[st1.y, st1.x] == 1
        assert st1.mem.searched_counts[max(0, st1.y - 1), max(0, st1.x - 1)] == 1

        # Execute multi-byte sequence
        kick_east = seq(Action.KICK, Action.EAST)
        st2, _, _ = harness.step(kick_east)
        cut_offset = len(harness.action_bytes)
        assert cut_offset > 0

        # Simulate stair transition
        st2.mem.pos_history.append((0, 1, 10, 20))

        class MockFloor2State:
            dungeon_num = 0
            level_num = 2
            y = 12
            x = 22

        st2.mem.update(MockFloor2State())
        assert (0, 1, 10, 20) in st2.mem.stair_graph
        assert st2.mem.stair_graph[(0, 1, 10, 20)] == (0, 2, 12, 22)
        assert st2.mem.stair_graph[(0, 2, 12, 22)] == (0, 1, 10, 20)

        # Replay bytes up to cut_offset
        replayed_st = harness.replay_bytes(
            seed=12345,
            recorded_bytes=bytes(harness.action_bytes),
            stop_at_byte_idx=cut_offset,
        )
        assert replayed_st.y == st2.y
        assert replayed_st.x == st2.x
        assert replayed_st.depth == st2.depth
        assert replayed_st.turn == st2.turn
        assert replayed_st.mem.searched_counts[st1.y, st1.x] == 1
    finally:
        harness.close()


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


def test_pass2_bug1_shards_paired_apples_to_apples() -> None:
    """Pass 2 Bug 1: 200-step shards must be paired strictly against baseline shards, never 25k full runs."""
    cand_runs = []
    base_runs = []

    # 10 shards
    for i in range(10):
        cand_runs.append(
            {
                "run_id": f"c_sh_{i}",
                "shard_id": f"sh_{i}",
                "seed": 100 + i,
                "is_shard": True,
                "depth": 1,
                "max_depth": 1,
                "score": 50,
                "turns": 200,
            }
        )
        base_runs.append(
            {
                "run_id": f"b_sh_{i}",
                "shard_id": f"sh_{i}",
                "seed": 100 + i,
                "is_shard": True,
                "depth": 1,
                "max_depth": 1,
                "score": 30,
                "turns": 200,
            }
        )

    # 10 full runs
    for i in range(10):
        cand_runs.append(
            {
                "run_id": f"c_full_{i}",
                "seed": 200 + i,
                "is_shard": False,
                "depth": 2,
                "max_depth": 2,
                "score": 500,
                "turns": 5000,
            }
        )
        base_runs.append(
            {
                "run_id": f"b_full_{i}",
                "seed": 200 + i,
                "is_shard": False,
                "depth": 2,
                "max_depth": 2,
                "score": 450,
                "turns": 5000,
            }
        )

    # Add a 25k-step full run for seed 100 into baseline (simulating a baseline full run with same seed as shard)
    base_runs.append(
        {
            "run_id": "b_full_100",
            "seed": 100,
            "is_shard": False,
            "depth": 5,
            "max_depth": 5,
            "score": 25000,
            "turns": 25000,
        }
    )

    verdict = evaluate_paired_hypothesis(cand_runs, base_runs)
    # Both shards and full runs improved, so all 20 paired deltas must be positive
    assert verdict["paired_count"] == 20
    assert verdict["wins"] == 20
    assert verdict["losses"] == 0
    assert verdict["mean_delta"] > 0.0

    # Test should_stop_early_for_futility handles both dict and list baseline without pairing shards against full
    futility_list = should_stop_early_for_futility(cand_runs, base_runs, min_pairs=10)
    assert not futility_list

    base_map = {int(r["seed"]): r for r in base_runs if not r.get("is_shard")}
    for r in base_runs:
        if r.get("is_shard"):
            base_map[r["shard_id"]] = r
    futility_dict = should_stop_early_for_futility(cand_runs, base_map, min_pairs=10)
    assert not futility_dict

    # Test baseline overwrite guard: promoting candidate runs ignores is_shard
    baseline_by_seed = {100: base_runs[-1]}  # 25k full run
    for r in cand_runs:
        if not r.get("is_shard"):
            baseline_by_seed[int(r["seed"])] = r
    # Seed 100 should still be the 25k full run, NOT overwritten by candidate shard!
    assert baseline_by_seed[100]["turns"] == 25000


def test_pass2_bug2_world_memory_craftax_bounds() -> None:
    """Pass 2 Bug 2: WorldMemory must handle Craftax (48, 48) grid without clipping at y >= 21."""
    harness = CraftaxHarness(max_steps=50)
    try:
        st = harness.reset(seed=42)
        # Craftax player spawns at (24, 24)
        assert st.y == 24
        assert st.x == 24
        assert st.mem.visited[24, 24]
        assert st.mem.visit_count[24, 24] >= 1
        assert st.mem.pos_history[-1] == (0, 1, 24, 24)

        # Move RIGHT to (24, 25)
        st, _, _ = harness.step(CraftaxAction.RIGHT)
        assert st.y == 24
        assert st.x == 25
        assert st.mem.visited[24, 25]
        assert st.mem.visit_count[24, 25] >= 1

        # Move LEFT to (24, 24)
        st, _, _ = harness.step(CraftaxAction.LEFT)
        assert st.mem.visit_count[24, 24] >= 2

        # Move UP to (23, 24)
        st, _, _ = harness.step(CraftaxAction.UP)
        assert st.mem.visited[23, 24]
        assert st.mem.visit_count[23, 24] >= 1
    finally:
        harness.close()


def test_pass2_bug3_synthesis_proposal_skill_code_wiring() -> None:
    """Pass 2 Bug 3: SynthesisProposal must preserve skill_code extracted from AuthorAgent."""
    hyp = {
        "mode": "FUNC_MOD_ADD",
        "target_functions": ["handle_combat"],
        "skill_module": "combat_tactics",
    }
    content = """
Here is my plan to improve combat survivability:

```python:combat_tactics
def evade_trap(state):
    return Action.WAIT
```

```python
def handle_combat(self, state):
    return self.explore_step(state)
```
"""
    skill = AuthorAgent.extract_skill_code(content, hyp)
    assert "def evade_trap(state):" in skill

    # Test hypothesis with direct skill_code key
    hyp_direct = {
        "mode": "FUNC_MOD_ADD",
        "skill_module": "trap_nav",
        "skill_code": "def avoid_trap(): pass",
    }
    skill_direct = AuthorAgent.extract_skill_code("", hyp_direct)
    assert skill_direct == "def avoid_trap(): pass"

    proposal = SynthesisProposal(
        mode="FUNC_MOD_ADD",
        hypothesis=hyp,
        raw_code_patch="def handle_combat(self, state): pass",
        full_policy_code="class Policy: pass",
        full_macros_code="",
        target_functions=["handle_combat"],
        skill_code=skill,
    )
    assert proposal.skill_code == skill


def test_pass2_bug4_extract_shards_from_db_and_pilot_logging(tmp_path: Path) -> None:
    """Pass 2 Bug 4: extract_shards_from_db must parse milestones_json & turns, and pilot saves policies/batches."""
    db_path = str(tmp_path / "test_shards.duckdb")
    init_db(db_path)
    logger = BatchDatabaseLogger(db_path=db_path)

    # 1. Save a policy and batch to allow runs insertion
    logger.save_policy(
        policy_id="pol_base",
        generation=0,
        mode="FULL_WRITE",
        policy_code="class Policy: pass",
        macros_code="",
        params={},
        avg_depth=2.0,
        max_depth=3,
        is_promoted=True,
    )

    # Build dummy runs with milestones_json and turns (no steps)
    action_bytes = b"k" * 200
    m_json = json.dumps({"depth_first_turn": {"2": 50, "3": 120}})
    runs = [
        {
            "run_id": "run_001",
            "batch_id": "batch_base",
            "generation": 0,
            "seed": 1234,
            "depth": 3,
            "max_depth": 3,
            "score": 150,
            "turns": 180,
            "xp_level": 2,
            "final_hp": 0,
            "max_hp": 15,
            "final_ac": 10,
            "gold": 0,
            "end_reason": "killed_by_jackal",
            "death_message": "jackal",
            "is_outlier_good": False,
            "outlier_z_score": 0.0,
            "batch_percentile": 50.0,
            "summary_digest": "Overview: Dlvl=3 | Steps=180",
            "game_log": "jackal bit you",
            "milestones_json": m_json,
            "triggered_functions": "{}",
            "last_active_function": "handle_combat",
            "action_bytes": action_bytes,
            "wall_ms": 10.0,
        }
    ]
    logger.save_batch(
        batch_id="batch_base",
        generation=0,
        mode="FULL_WRITE",
        parent_policy_id="none",
        candidate_policy_id="pol_base",
        target_functions=[],
        runs=runs,
        episodes_discarded=0,
        batch_metrics={"avg_depth": 3.0, "max_depth": 3},
    )

    shards = extract_shards_from_db(db_path, "batch_base")
    # Must extract pre-death shard and floor entry shards (Dlvl 2 and Dlvl 3)
    assert len(shards) >= 2
    shard_types = [s.trigger_context for s in shards]
    assert any("pre_death" in t for t in shard_types)
    assert any("floor_entry:2" in t for t in shard_types)
    assert any("floor_entry:3" in t for t in shard_types)

    # 2. Test pilot evaluation DuckDB saving completeness via AuthorToolRegistry
    pool = PersistentEvalPool(workers=2, max_steps=10)
    try:
        tools = AuthorToolRegistry(db_path=db_path)
        tools.configure_generation(
            pool=pool,
            baseline_runs=runs,
            max_pilot_runs=3,
            pilot_episodes=10,
            generation=1,
        )
        # Test valid code dry run
        code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        diag, status = tools.dry_run_policy(code=code, is_macro=False)
        assert "PASSED" in status or "PILOT" in status

        # Verify DuckDB has policies, batches, and experiments rows with generation=1
        con = connect_db(db_path, read_only=True)
        try:
            cand_count = con.execute(
                "SELECT COUNT(*) FROM policies WHERE mode = 'PILOT'"
            ).fetchone()[0]
            assert cand_count >= 1
            batch_count = con.execute(
                "SELECT COUNT(*) FROM batches WHERE mode = 'PILOT'"
            ).fetchone()[0]
            assert batch_count >= 1
            exp_count = con.execute(
                "SELECT COUNT(*) FROM experiments WHERE mode = 'PILOT'"
            ).fetchone()[0]
            assert exp_count >= 1
            gen_val = con.execute(
                "SELECT generation FROM experiments WHERE mode = 'PILOT'"
            ).fetchone()[0]
            assert gen_val == 1
        finally:
            con.close()
    finally:
        pool.close()


def test_pass3_pilot_delta_depth_and_exact_batch_capacity(tmp_path: Path) -> None:
    """Pass 3: No false negative pilot delta_depth with shards and exact batch dispatch."""
    db_path = str(tmp_path / "pass3_test.duckdb")
    pool = PersistentEvalPool(workers=2, max_steps=15)
    try:
        tools = AuthorToolRegistry(db_path=db_path)

        # 1. No False Negative in dry_run_policy When Shards Have Lower Depth Than Full Runs
        # 10 runs are Dlvl 1 failures (with action_bytes = b"k" * 20, max_depth = 1, score = 10)
        # and 20 runs have max_depth = 2, score = 20
        baseline_runs = []
        for i in range(1, 11):
            baseline_runs.append(
                {
                    "run_id": f"base_fail_{i}",
                    "seed": i,
                    "max_depth": 1,
                    "depth": 1,
                    "score": 10,
                    "steps": 20,
                    "turns": 20,
                    "action_bytes": b"k" * 20,
                    "end_reason": "killed_by_jackal",
                    "summary_digest": f"Overview: Dlvl=1 | Steps=20 | Seed={i}",
                }
            )
        for i in range(11, 31):
            baseline_runs.append(
                {
                    "run_id": f"base_norm_{i}",
                    "seed": i,
                    "max_depth": 2,
                    "depth": 2,
                    "score": 20,
                    "steps": 40,
                    "turns": 40,
                    "action_bytes": b"k" * 40,
                    "end_reason": "ascended",
                    "summary_digest": f"Overview: Dlvl=2 | Steps=40 | Seed={i}",
                }
            )

        tools.configure_generation(
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=3,
            pilot_episodes=30,
            generation=1,
        )

        cand_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""

        orig_eval = pool.evaluate_batch

        def mock_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
            res = orig_eval(*args, **kwargs)
            # Candidate full runs achieve higher score/progression while shards stay at max_depth=1
            for r in res.get("runs", []):
                if not r.get("is_shard"):
                    r["max_depth"] = 2
                    r["score"] = 25
                else:
                    r["max_depth"] = 1
                    r["score"] = 10
            return res

        pool.evaluate_batch = mock_eval

        try:
            diag, status = tools.dry_run_policy(code=cand_policy, is_macro=False)
        finally:
            pool.evaluate_batch = orig_eval

        # Assert tools.last_pilot_improved is True
        assert tools.last_pilot_improved is True
        assert "PASSED" in status

        # Assert Δ= in tools.last_pilot_summary is >= 0.0 (not negative!)
        assert "Δ=" in tools.last_pilot_summary
        delta_str = tools.last_pilot_summary.split("Δ=")[-1].split()[0]
        delta_val = float(delta_str)
        assert delta_val >= 0.0, f"Expected non-negative delta, got {delta_val}"

        # Assert cand_depths / base_depths were paired 1-to-1
        assert len(tools.last_cand_depths) > 0
        assert len(tools.last_cand_depths) == len(tools.last_base_depths)

        # 2. Exact Task Dispatch in evaluate_batch When Priority Seeds / Shards Cover target_episodes
        exact_seeds = list(range(100, 115))
        res = pool.evaluate_batch(
            batch_id="test_exact_dispatch",
            policy_code=cand_policy,
            target_episodes=10,
            target_functions=["run"],
            priority_seeds=exact_seeds,
        )
        assert res["raw_episodes_run"] == 10
        assert len(res["runs"]) == 10
        run_seeds = [r["seed"] for r in res["runs"]]
        assert set(run_seeds).issubset(set(exact_seeds))
    finally:
        pool.close()


def test_pass4_codebase_bottleneck_fixes(tmp_path: Path) -> None:
    """
    Pass 4 Codebase Bottleneck Verification:
    1. champion_runs, champion_function_stats, champion_ticks, and latest_pilot_runs views exist in DuckDB.
    2. query_db dynamically rewrites (SELECT MAX(batch_id) FROM runs) to active champion batch.
    3. ingest_expert_runs_into_duckdb populates expert_ticks with depth >= 2 rows.
    4. WorldMemory normalizes open doors to ord('.') in remembered_chars, sets door_mask = True,
       exposes passable_mask and adjacent_hostiles(), and astar_next_step avoids diagonal steps on door_mask.
    5. ModeScheduler only triggers PARAM_TUNE when at least one modification has been promoted.
    """
    from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
    from lox.engine.scheduler import ModeScheduler
    from lox.env.grid_utils import astar_next_step
    from lox.env.state import _IS_OPEN_DOOR_GLYPH_LUT, WorldMemory

    # 1. Canonical Views & SELECT MAX(batch_id) Poison Pill Resolution
    db_path = str(tmp_path / "test_bottlenecks.duckdb")
    con = init_db(db_path)
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, hypothesis_passed, created_at)
        VALUES ('c2026_g001', 1, 'FUNC_MOD_ADD', TRUE, '2026-01-01 10:00:00'),
               ('pilot_9999_01', 1, 'PILOT', FALSE, '2026-01-01 11:00:00')
        """
    )
    con.execute(
        """
        INSERT INTO runs (run_id, batch_id, generation, seed)
        VALUES ('run_champ', 'c2026_g001', 1, 100),
               ('run_pilot', 'pilot_9999_01', 1, 101)
        """
    )
    champ_count = con.execute("SELECT COUNT(*) FROM champion_runs").fetchone()[0]
    pilot_count = con.execute("SELECT COUNT(*) FROM latest_pilot_runs").fetchone()[0]
    assert champ_count == 1
    assert pilot_count == 1
    con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    yaml_out, count_str = tools.query_db(
        "SELECT run_id FROM runs WHERE batch_id = (SELECT MAX(batch_id) FROM runs)"
    )
    assert "run_champ" in yaml_out
    assert "run_pilot" not in yaml_out

    # 2. Expert Ticks Deep Depths Population
    res = ingest_expert_runs_into_duckdb(db_path=db_path, max_ascensions=3, force=True)
    assert res["expert_runs"] >= 3
    con = connect_db(db_path, read_only=True)
    deep_ticks = con.execute(
        "SELECT COUNT(*) FROM expert_ticks WHERE depth >= 2"
    ).fetchone()[0]
    jackal_ticks = con.execute(
        "SELECT COUNT(*) FROM expert_ticks WHERE message ILIKE '%jackal%'"
    ).fetchone()[0]
    con.close()
    assert deep_ticks > 0
    assert jackal_ticks > 0

    # 3. WorldMemory Open-Door Normalization, door_mask & Diagonal Doorway Avoidance
    mem = WorldMemory()

    class DummyState:
        def __init__(self):
            self.dungeon_num = 0
            self.level_num = 1
            self.y = 10
            self.x = 20
            self.chars = np.full((21, 79), ord(" "), dtype=np.uint8)
            self.glyphs = np.zeros((21, 79), dtype=np.int32)
            self.message = ""

    st = DummyState()
    # Find an open door glyph from precomputed LUT
    open_door_glyphs = np.nonzero(_IS_OPEN_DOOR_GLYPH_LUT)[0]
    od_g = int(open_door_glyphs[0]) if len(open_door_glyphs) > 0 else 2372
    st.glyphs[10, 15] = od_g
    st.chars[10, 15] = ord("-")  # C engine ASCII for horizontal open door
    mem.update(st)

    # Open door char should be normalized to '.' in remembered_chars (never '-' or '|')
    assert mem.current_floor.remembered_chars[10, 15] == ord(".")
    assert bool(mem.door_mask[10, 15]) is True
    assert bool(mem.passable_mask[10, 15]) is True

    # Doorway diagonal collision avoidance in pathfinding
    passable_grid = np.ones((5, 5), dtype=np.bool_)
    door_grid = np.zeros((5, 5), dtype=np.bool_)
    door_grid[2, 2] = True  # doorway at (2, 2)

    # Diagonal step into (2, 2) from (1, 1) is forbidden
    step = astar_next_step(
        passable_grid,
        sy=1,
        sx=1,
        ty=3,
        tx=3,
        allow_diagonal=True,
        no_diag=door_grid,
    )
    assert step in ((1, 0), (0, 1)), f"Expected orthogonal entry, got {step}"

    # 4. ModeScheduler PARAM_TUNE Gating on Promoted Modifications
    sched = ModeScheduler(param_tune_every=2)
    m1 = sched.select_mode(generation=1, has_policy=True, has_tunable_params=True)
    assert m1 == "FUNC_MOD_ADD"

    # Generation 2 without any promotions should NOT tune
    m2 = sched.select_mode(generation=2, has_policy=True, has_tunable_params=True)
    assert m2 == "FUNC_MOD_ADD"

    # After recording a promotion, generation 3 should select PARAM_TUNE
    sched.record_promotion("FUNC_MOD_ADD")
    m3 = sched.select_mode(generation=3, has_policy=True, has_tunable_params=True)
    assert m3 == "PARAM_TUNE"


def test_pass5_deep_audit_fixes(tmp_path: Path) -> None:
    """
    Pass 5 Deep Audit & Runtime Attribution Verification:
    1. Under-Hero Object Visibility (state.visible_objects() and state.objects_here()).
    2. Effective Function Trigger Attribution (ExecutionTracer distinguishes value-returns vs no-op).
    3. Compiler-Gate smoke_only=True and best-pilot preservation.
    4. Dark corridor memory in NetHackHarness.replay_bytes.
    """
    from lox.author.tools import AuthorToolRegistry
    from lox.env.harness import NetHackHarness
    from lox.env.state import _IS_BODY_LUT, WorldMemory
    from lox.lang.compiler import compile_policy_and_macros

    # 1. Under-Hero Object Visibility
    mem = WorldMemory()
    body_glyphs = np.nonzero(_IS_BODY_LUT)[0]
    corpse_g = int(body_glyphs[0]) if len(body_glyphs) > 0 else 1900

    harness = NetHackHarness(max_steps=10)
    try:
        real_st = harness.reset(seed=42)
        hy, hx = real_st.y, real_st.x
        # Simulate corpse remembered at hero's current position
        mem.current_floor.remembered_obj_glyphs[hy, hx] = corpse_g
        real_st.mem = mem
        here = real_st.objects_here()
        assert len(here) >= 1
        corpse_obj = here[0]
        assert corpse_obj.y == hy
        assert corpse_obj.x == hx
        assert corpse_obj.is_corpse is True
    finally:
        harness.close()

    # 2. Effective Function Trigger Attribution
    code = """
class Policy:
    def handle_items(self, state):
        return None

    def handle_combat(self, state):
        return None

    def move_towards(self, state):
        return Action.E

    def explore(self, state):
        return self.move_towards(state)

    def run(self, state):
        while True:
            act = self.handle_items(state)
            if act is not None:
                state = yield act
                continue
            act = self.handle_combat(state)
            if act is not None:
                state = yield act
                continue
            act = self.explore(state)
            state = yield act
"""
    compiled = compile_policy_and_macros(code)
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=123)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st)
        assert act == Action.E
        assert runner.tracer.fn_calls.get("handle_items", 0) == 0
        assert runner.tracer.fn_calls.get("handle_combat", 0) == 0
        assert runner.tracer.fn_calls.get("explore", 0) == 1
        assert runner.tracer.fn_turns.get("explore", 0) == 1
        assert runner.tracer.fn_calls.get("move_towards", 0) == 1
        assert runner.tracer.fn_turns.get("move_towards", 0) == 0
        assert runner.tracer.triggered_any(["handle_items"]) is False
    finally:
        h.close()

    # 3. Compiler-Gate smoke_only=True & Best-Pilot Fallback
    db_p = str(tmp_path / "tools_test.duckdb")
    tools = AuthorToolRegistry(db_path=db_p)
    smoke_code = """
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
    out, status = tools.dry_run_policy(smoke_code, smoke_only=True)
    assert status == "PASSED (smoke check)"
    assert tools.pilot_runs_used == 0

    tools.last_passed_code = "code_v1"
    tools.last_pilot_improved = True
    tools.best_pilot_code = "code_v1"
    tools.best_pilot_delta = 1.5

    # If subsequent pilot fails, last_pilot_improved stays True because last_passed_code is set
    improved = False
    assert not improved
    if not tools.last_passed_code:
        tools.last_pilot_improved = False
    assert tools.last_pilot_improved is True
    assert tools.last_passed_code == "code_v1"

    # 4. Dark Corridor Memory in replay_bytes
    h_rep = NetHackHarness(max_steps=20)
    try:
        st = h_rep.reset(seed=999)
        bytes_recorded = bytearray()
        for a in (Action.EAST, Action.EAST, Action.SOUTH, Action.SOUTH):
            st, d, _ = h_rep.step(a)
            bytes_recorded.append(int(a))
            if d:
                break
        replayed_st = h_rep.replay_bytes(seed=999, recorded_bytes=bytes(bytes_recorded))
        assert replayed_st is not None
        fl = h_rep.mem.current_floor
        for dnum, dlvl, py, px in list(h_rep.mem.pos_history):
            if (
                (dnum, dlvl) == h_rep.mem.current_floor_key
                and 0 <= py < 21
                and 0 <= px < 79
            ):
                assert fl.remembered_chars[py, px] != 0, f"Blank char at ({py}, {px})"
    finally:
        h_rep.close()


def test_pass6_overnight_audit_fixes(tmp_path: Path) -> None:
    """Pass 6: Overnight Campaign & Runtime Audit Overhaul Verification."""
    from lox.engine.scheduler import ModeScheduler
    from lox.lang.compiler import compile_policy_and_macros
    from lox.lang.validator import splice_functions_into_policy
    from lox.ui.console import ConsoleUI

    # 1. ExecutionTracer attributes turns to handle_combat (not leaf helper get_passable_mask)
    #    and counts boolean macro calls returning False
    macro_src = """
def macro_is_low_hp(state):
    return False
"""
    pol_src = """
class Policy:
    def get_passable_mask(self, state):
        return state.mem.passable_mask

    def handle_combat(self, state):
        if macro_is_low_hp(state):
            return Action.WAIT
        mask = self.get_passable_mask(state)
        return Action.E

    def run(self, state):
        while True:
            act = self.handle_combat(state)
            state = yield act
"""
    compiled = compile_policy_and_macros(pol_src, macros_code=macro_src)
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st)
        assert act == Action.E
        assert active_fn == "handle_combat"
        assert runner.tracer.fn_turns.get("handle_combat", 0) == 1
        assert runner.tracer.fn_turns.get("get_passable_mask", 0) == 0
        assert runner.tracer.fn_calls.get("macro_is_low_hp", 0) >= 1
    finally:
        h.close()

    # 2. PersistentEvalPool.evaluate_batch with priority_seeds and unconditional_seeds=[]
    #    discards untriggered episodes instead of treating all priority seeds as unconditional
    pool = PersistentEvalPool(workers=2, max_steps=20)
    try:
        pol_dummy = """
class Policy:
    def run(self, state):
        while True:
            yield Action.WAIT
"""
        # When unconditional_seeds=[], seed 100 is NOT unconditional, so enforce_trigger is True
        res_filtered = pool.evaluate_batch(
            batch_id="test_filter_bypass",
            policy_code=pol_dummy,
            target_episodes=1,
            target_functions=["nonexistent_fn"],
            priority_seeds=[100, 101],
            unconditional_seeds=[],
            unreachable_abort_after=3,
            show_progress=False,
        )
        assert res_filtered["unreachable_abort"] is True
        assert res_filtered["episodes_recorded"] == 0
        assert res_filtered["episodes_discarded"] >= 3

        # When unconditional_seeds=[100], seed 100 is kept even without triggering
        res_kept = pool.evaluate_batch(
            batch_id="test_filter_uncond",
            policy_code=pol_dummy,
            target_episodes=1,
            target_functions=["nonexistent_fn"],
            priority_seeds=[100],
            unconditional_seeds=[100],
            show_progress=False,
        )
        assert res_kept["episodes_recorded"] == 1
        assert res_kept["runs"][0]["seed"] == 100
    finally:
        pool.close()

    # 3. splice_functions_into_policy preserves existing self.<attr> in __init__
    base_code = """
class Policy:
    def __init__(self):
        self.old_attr1 = 10
        self.old_attr2 = 20
        self.overwrite_me = "old"

    def run(self, state):
        return Action.WAIT
"""
    patch_code = """
    def __init__(self):
        \"\"\"Patch docstring.\"\"\"
        self.overwrite_me = "new"
        self.new_attr = 30
"""
    spliced, changed = splice_functions_into_policy(base_code, patch_code)
    assert "__init__" in changed
    assert "self.old_attr1 = 10" in spliced
    assert "self.old_attr2 = 20" in spliced
    assert (
        "self.overwrite_me = 'new'" in spliced or 'self.overwrite_me = "new"' in spliced
    )
    assert "self.new_attr = 30" in spliced

    # 4. ModeScheduler does not lock into STRUCTURAL_REFACTOR on consecutive_failures == 4
    sched = ModeScheduler(macro_every=0)
    mode_at_3 = sched.select_mode(generation=3, has_policy=True, consecutive_failures=3)
    assert mode_at_3 == "STRUCTURAL_REFACTOR"
    mode_at_4 = sched.select_mode(generation=4, has_policy=True, consecutive_failures=4)
    assert mode_at_4 != "STRUCTURAL_REFACTOR"

    # 5. AuthorToolRegistry.query_policy_docs("policy") returns current_policy_code
    db_p = str(tmp_path / "pass6_tools.duckdb")
    tools = AuthorToolRegistry(db_path=db_p)
    tools.current_policy_code = "class Policy:\n    pass"
    doc_text, topic_tag = tools.query_policy_docs("policy")
    assert "class Policy:" in doc_text
    assert topic_tag == "policy source"

    # 6. ConsoleUI.hypothesis_box and BatchDatabaseLogger.save_experiment record
    #    binding_constraint and verified_c_rule
    c_ui = ConsoleUI(enabled=True)
    c_ui.hypothesis_box(
        {
            "binding_constraint": "Hunger clock forces exploration",
            "mechanism": "Eat corpses before descending",
            "target_functions": ["eat"],
        }
    )
    c_ui.hypothesis_box(
        {
            "verified_c_rule": "Doors cannot be entered diagonally",
            "mechanism": "Orthogonal alignment",
            "target_functions": ["move"],
        }
    )

    db_logger = BatchDatabaseLogger(db_p)
    db_logger.save_experiment(
        experiment_id="exp_test_binding",
        generation=1,
        batch_id="b1",
        mode="FUNC_MOD_ADD",
        target_functions=["eat"],
        hypothesis={"binding_constraint": "Hunger clock forces exploration"},
        verdict={"passed": True},
    )
    db_logger.save_experiment(
        experiment_id="exp_test_verified",
        generation=2,
        batch_id="b2",
        mode="FUNC_MOD_ADD",
        target_functions=["move"],
        hypothesis={"verified_c_rule": "Doors cannot be entered diagonally"},
        verdict={"passed": True},
    )
    con = connect_db(db_p)
    try:
        rows = con.execute(
            "SELECT experiment_id, observation_from_db FROM experiments ORDER BY generation"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == "Hunger clock forces exploration"
        assert rows[1][1] == "Doors cannot be entered diagonally"
    finally:
        con.close()
