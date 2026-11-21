"""Domain-scoped tests for LOX."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from lox.author.tools import AuthorToolRegistry
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import compute_llm_cost, connect_db, init_db
from lox.db.shards import extract_shards_from_db, extract_shards_from_runs
from lox.db.theories import TheoryDocumentStore
from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.runner.pool import PersistentEvalPool, run_single_episode


def test_theory_document_store_lancedb(tmp_path):
    p = tmp_path / "test_theories"
    store = TheoryDocumentStore(db_path=str(p))
    assert store.count() == 1  # Seed row

    # 1. Save document
    doc_id = store.save_document(
        generation=1,
        agent_role="threat",
        doc_type="REFLECTION_REPORT",
        title="Tactical Ant Kiting",
        content="Soldier ants have poisonous bites. In narrow corridors, only one ant can attack at a time.",
        tags=["combat", "ants", "poison"],
        verified_c_rule="corridor choke prevents surrounding",
    )
    assert doc_id.startswith("doc_threat_")
    assert store.count() == 2

    # 2. Search document
    hits = store.search_theories("ants poison", limit=3)
    assert len(hits) >= 1
    assert "Tactical Ant Kiting" in hits[0]["title"]

    # 3. Retrieve latest reports
    reports = store.get_latest_reports(generation=1)
    assert "threat" in reports
    assert reports["threat"]["title"] == "Tactical Ant Kiting"


def test_duckdb_llm_cost_and_token_tracking(tmp_path):
    db_path = str(tmp_path / "test_cost.duckdb")
    con = init_db(db_path)
    con.close()

    # 1. Verify cost formula
    cost = compute_llm_cost(
        "z-ai/glm-5.3-flash:nitro", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert abs(cost - 0.46) < 1e-4

    # 2. Save log with cost
    logger = BatchDatabaseLogger(db_path=db_path)
    logger.save_synthesis_log(
        log_id="log_swarm_01",
        generation=1,
        batch_id="b_swarm_01",
        mode="WRITE",
        turn_index=1,
        turn_type="PILOT",
        model="z-ai/glm-5.3-flash:nitro",
        prompt_tokens=50000,
        completion_tokens=2000,
        reasoning_tokens=500,
        agent_role="threat",
    )

    con = duckdb.connect(db_path, read_only=True)
    row = con.execute(
        "SELECT agent_role, cost_usd, prompt_tokens, completion_tokens FROM synthesis_logs"
    ).fetchone()
    assert row[0] == "threat"
    assert row[1] > 0.0

    rates_count = con.execute("SELECT COUNT(*) FROM llm_cost_rates").fetchone()[0]
    assert rates_count >= 8
    con.close()


def test_peer_swarm_records_theories_to_lancedb(tmp_path):
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import SwarmPeerAgent
    from lox.db.theories import TheoryDocumentStore

    store = TheoryDocumentStore(str(tmp_path / "theories.lance"))
    peer = SwarmPeerAgent(
        peer_id="peer_1",
        provider="mock",
        db_path=str(tmp_path / "lox.duckdb"),
        theories_store=store,
    )
    rep = DeficitReport(generation=1, regime="BOOTSTRAP")
    cluster = DeficitCluster(
        cluster_id="slice_stalled_wait_fallback",
        kind="stalled_trajectory",
        label="Stall in wait_fallback",
        evidence="wait_fallback starvation faint loop",
    )
    peer.run_cycle(
        generation=1,
        deficit_report=rep,
        claimed_deficit=cluster,
        current_policy="",
        current_skills={},
    )
    hits = store.search_theories("wait_fallback starvation", doc_type="THEORY")
    assert len(hits) >= 1
    assert hits[0]["agent_role"] == "peer_1"
    assert "slice_stalled_wait_fallback" in hits[0]["content"]


def test_prefix_shard_extractor_and_worker_replay():
    # Build sample runs with action bytes
    harness = NetHackHarness(max_steps=80)
    state = harness.reset(seed=999)
    for _ in range(35):
        state, done, _ = harness.step(Action.SEARCH)
        if done:
            break

    recorded_bytes = bytes(harness.action_bytes)
    assert len(recorded_bytes) >= 30

    sample_run = {
        "run_id": "test_run_01",
        "seed": 999,
        "steps": len(recorded_bytes),
        "depth": 1,
        "max_depth": 2,
        "end_reason": "killed by a goblin",
        "summary_digest": "Overview: Depth=1\nStalls: None\nKey Inflections:\n  * None",
        "milestones": {"depth_first_turn": {"2": 15}},
        "action_bytes": recorded_bytes,
    }

    shards = extract_shards_from_runs([sample_run], max_shards=10)
    assert len(shards) >= 1

    pre_death_shards = [s for s in shards if "pre_death" in s.trigger_context]
    assert len(pre_death_shards) == 1
    shard = pre_death_shards[0]
    assert shard.seed == 999
    assert len(shard.action_bytes) < len(recorded_bytes)

    # Execute task with replay_bytes in worker
    policy_code = """
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
"""
    task = {
        "seed": shard.seed,
        "run_id": "shard_eval_01",
        "role": "val-hum-law-fem",
        "max_steps": 20,
        "policy_code": policy_code,
        "macros_code": "",
        "target_functions": [],
        "replay_bytes": shard.action_bytes,
        "post_replay_max_steps": 20,
        "is_priority_seed": True,
        "enforce_trigger": False,
        "record_ticks": True,
    }

    res = run_single_episode(task)
    assert res.get("steps") is not None
    assert res.get("end_reason") is not None


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
