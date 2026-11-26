"""Domain-scoped tests for LOX."""

from __future__ import annotations

import concurrent.futures

from lox.datasets.bootstrap_frontier import bootstrap_frontier_from_duckdb
from lox.db.schema import connect_db, init_db
from lox.engine.frontier import FrontierArchive
from lox.engine.hypothesis import progression_metric
from lox.engine.map_elites import MAPElitesArchive


# ---------------------------------------------------------------------------
# 3. MAP-Elites 3D Quality-Diversity Archive Tests
# ---------------------------------------------------------------------------
def test_map_elites_archive(tmp_path):
    db_path = str(tmp_path / "test_map.duckdb")
    con = init_db(db_path)

    archive = MAPElitesArchive()
    assert archive.coverage() == 0.0

    # Policy A: Dlvl 1, unarmored (AC=10), slow tempo (< 0.08 tiles/turn)
    metrics_a = {
        "avg_depth": 1.0,
        "max_depth": 1,
        "avg_ac": 10.0,
        "avg_xp_level": 1.0,
        "avg_unique_tiles": 5.0,
        "avg_turns": 100.0,
        "avg_score": 10.0,
    }
    added_a, cell_a, elite_a = archive.add_candidate(
        policy_code="class Policy: pass # A",
        metrics=metrics_a,
        batch_id="b_001",
        generation=1,
        con=con,
    )
    assert added_a is True
    assert cell_a == "cell_v2_d0_v0_t0"
    assert archive.coverage() > 0.0

    # Policy B: Dlvl 1.5, armor/XL stepping stone (AC=5.0, XL=3), moderate tempo (0.20 tiles/turn)
    metrics_b = {
        "avg_depth": 1.5,
        "max_depth": 2,
        "avg_ac": 5.0,
        "avg_xp_level": 3.0,
        "avg_unique_tiles": 40.0,
        "avg_turns": 200.0,
        "avg_score": 150.0,
    }
    added_b, cell_b, elite_b = archive.add_candidate(
        policy_code="class Policy: pass # B",
        metrics=metrics_b,
        batch_id="b_002",
        generation=2,
        con=con,
    )
    assert added_b is True
    assert cell_b == "cell_v2_d1_v2_t2"

    # Policy A_worse: same cell as A, lower fitness -> rejected
    metrics_a_worse = {
        "avg_depth": 1.0,
        "max_depth": 1,
        "avg_ac": 10.0,
        "avg_xp_level": 1.0,
        "avg_unique_tiles": 1.0,
        "avg_turns": 50.0,
        "avg_score": 0.0,
    }
    added_aw, cell_aw, _ = archive.add_candidate(
        policy_code="class Policy: pass # Aw",
        metrics=metrics_a_worse,
        batch_id="b_003",
        generation=3,
        con=con,
    )
    assert added_aw is False
    assert cell_aw == "cell_v2_d0_v0_t0"

    # Sampling (including UCB1 curiosity)
    best_sample = archive.sample_elite(strategy="best")
    assert best_sample is not None
    assert best_sample.cell_id == cell_b

    curiosity_sample = archive.sample_elite(strategy="ucb_curiosity")
    assert curiosity_sample is not None
    archive.record_outcome(cell_b, promoted=True, con=con)

    # Load from DB
    new_archive = MAPElitesArchive()
    loaded_count = new_archive.load_from_db(con)
    assert loaded_count == 2
    assert "cell_v2_d0_v0_t0" in new_archive.cells
    assert "cell_v2_d1_v2_t2" in new_archive.cells
    assert new_archive.cells["cell_v2_d1_v2_t2"].promotions == 1

    con.close()


# ---------------------------------------------------------------------------
# 4. Go-Explore Frontier Checkpoints Tests
# ---------------------------------------------------------------------------
def test_frontier_archive_save_and_sample(tmp_path):
    frontier_dir = tmp_path / "frontier"
    archive = FrontierArchive(frontier_dir=str(frontier_dir))

    # Add checkpoint
    dummy_bytes = bytes([ord("k"), ord("k"), ord("l"), ord("h")])
    cp = archive.add_checkpoint(
        seed=12345,
        depth=4,
        milestone="DLVL_4",
        turns=650,
        action_bytes=dummy_bytes,
        game_log="You enter level 4.",
    )
    assert cp.checkpoint_id.startswith("fc_dlvl_4_")
    assert len(archive.checkpoints) == 1

    # Reload from disk
    new_archive = FrontierArchive(frontier_dir=str(frontier_dir))
    assert len(new_archive.checkpoints) == 1
    sampled = new_archive.sample_checkpoint(min_depth=3)
    assert sampled is not None
    assert sampled.depth == 4
    assert sampled.action_bytes == dummy_bytes


def test_frontier_archive_cross_instance_lock(tmp_path):
    fdir = str(tmp_path / "shared_frontier")

    def _add(i: int) -> str:
        arch = FrontierArchive(frontier_dir=fdir)
        cp = arch.add_checkpoint(
            seed=100 + i,
            depth=2,
            milestone=f"M{i}",
            turns=50 + i,
            action_bytes=bytes(f"act_{i}", "ascii"),
            checkpoint_id=f"cp_{i}",
        )
        return cp.checkpoint_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ids = list(ex.map(_add, range(10)))
    assert len(ids) == 10
    reader = FrontierArchive(frontier_dir=fdir)
    assert len(reader.checkpoints) == 10


def test_frontier_bootstrapping_and_checkpoints(tmp_path):
    frontier_dir = str(tmp_path / "frontier")
    seed_db = str(tmp_path / "seed_frontier.duckdb")
    con = init_db(seed_db)
    con.execute(
        """
        INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, action_bytes, game_log, is_shard)
        VALUES
          ('r_dlvl3_a', 'b1', 1, 101, 'Val', 3, 3, 250, 120, ?::BLOB, 'Turn 1\nTurn 120', FALSE),
          ('r_dlvl3_b', 'b1', 1, 102, 'Val', 3, 3, 180, 95, ?::BLOB, 'Turn 1\nTurn 95', FALSE),
          ('r_shard_ign', 'b1', 1, 103, 'Val', 3, 3, 999, 500, ?::BLOB, 'Shard run', TRUE)
        """,
        [b"abcde", b"fghij", b"zzzzz"],
    )
    con.close()

    count = bootstrap_frontier_from_duckdb(
        db_path=seed_db,
        frontier_dir=frontier_dir,
        max_checkpoints_per_depth=2,
    )
    assert count == 2


def test_map_elites_v2_calibration_deficits_confinement_and_private_symbols(tmp_path):
    from lox.author.prompts import build_user_prompt
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.db.schema import init_db
    from lox.engine.map_elites import MAPElitesArchive

    # 1. Empirical NetHack tiles/turns (0.04..0.35) & depths (1.1..2.6) spread across distinct MAP-Elites cells
    arch = MAPElitesArchive()
    c1, *_ = arch.compute_cell_id(
        {
            "avg_depth": 1.12,
            "avg_ac": 9.0,
            "avg_xp_level": 1.0,
            "avg_unique_tiles": 40.0,
            "avg_turns": 1000.0,
        }
    )
    c2, *_ = arch.compute_cell_id(
        {
            "avg_depth": 1.45,
            "avg_ac": 8.0,
            "avg_xp_level": 2.0,
            "avg_unique_tiles": 95.0,
            "avg_turns": 1000.0,
        }
    )
    c3, *_ = arch.compute_cell_id(
        {
            "avg_depth": 1.90,
            "avg_ac": 6.0,
            "avg_xp_level": 3.0,
            "avg_unique_tiles": 190.0,
            "avg_turns": 1000.0,
        }
    )
    c4, *_ = arch.compute_cell_id(
        {
            "avg_depth": 2.60,
            "avg_ac": 5.0,
            "avg_xp_level": 4.0,
            "avg_unique_tiles": 350.0,
            "avg_turns": 1000.0,
        }
    )
    assert len({c1, c2, c3, c4}) == 4

    # 2. Anti-Goodhart progression_metric strips farmable xl/score and applies -0.35 penalty on max_steps stalls
    norm_run = {
        "depth": 1,
        "max_depth": 1,
        "score": 50,
        "turns": 500,
        "end_reason": "killed",
        "unique_tiles": 60,
    }
    camp_run = {
        "depth": 1,
        "max_depth": 1,
        "score": 50,
        "turns": 5000,
        "end_reason": "max_steps",
        "unique_tiles": 60,
        "summary_digest": "CONFINEMENT_STALL",
    }
    assert progression_metric(norm_run) - progression_metric(camp_run) >= 0.35
    # Even a high-XL / high-score D2 room-camper must score strictly below a genuine D2 run
    norm_d2 = {
        "depth": 2,
        "max_depth": 2,
        "score": 150,
        "xp_level": 2,
        "end_reason": "killed",
    }
    camp_d2 = {
        "depth": 2,
        "max_depth": 2,
        "score": 1800,
        "xp_level": 6,
        "end_reason": "max_steps",
    }
    assert progression_metric(camp_d2) < progression_metric(norm_d2)

    # 3. DeficitAnalyzer isolates shallow max_steps confinement & surfaces raw game_log tail
    db_p = str(tmp_path / "confine.duckdb")
    init_db(db_p)
    con = connect_db(db_p, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, avg_depth, max_depth) VALUES ('b1', 1, 'SWARM', 1.5, 2)"
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, steps,
                              end_reason, unique_tiles, game_log, last_active_function)
            VALUES ('r_conf', 'b1', 1, 9001, 'val', 2, 2, 120, 4950, 5000, 'max_steps', 45,
                    'line1\nYou cannot pass through the closed obstacle.', 'explore_loop')
            """
        )
    finally:
        con.close()

    rep = DeficitAnalyzer(db_p).analyze(generation=2)
    cids = {c.cluster_id for c in rep.fail_clusters}
    assert "slice_max_steps_confinement" in cids
    conf_c = next(
        c for c in rep.fail_clusters if c.cluster_id == "slice_max_steps_confinement"
    )
    assert "closed obstacle" in conf_c.evidence

    # 4. Tier-1 SWARM_PEER compact Domain Tree Index includes public entrypoints without bloated private dumps
    prompt = build_user_prompt(
        generation=2,
        mode="SWARM_PEER",
        current_policy="class Policy:\n    def step(self, s): return None\n",
        skills={
            "nav": "_STALL_CAP = 5\n\ndef _private_helper(state):\n    return None\n\ndef public_step(state):\n    return None\n"
        },
        recent_discoveries=[],
    )
    assert "skills/nav.py" in prompt
    assert "public_step" in prompt
    assert "read_skill" in prompt


def test_map_elites_load_from_db_readonly_zero_warnings(tmp_path, caplog):
    """Verifies MAPElitesArchive.load_from_db on a read-only DuckDB connection emits zero purge warnings."""
    import duckdb

    db_path = str(tmp_path / "ro_map.duckdb")
    con_rw = init_db(db_path)
    try:
        arch = MAPElitesArchive()
        arch.add_candidate(
            policy_code="class Policy: pass",
            metrics={
                "avg_depth": 2.0,
                "max_depth": 2,
                "avg_ac": 8.0,
                "avg_xp_level": 2.0,
                "avg_unique_tiles": 40.0,
                "avg_turns": 100.0,
                "avg_score": 150.0,
            },
            batch_id="b1",
            generation=1,
            con=con_rw,
        )
        # Also insert a legacy v1 row to confirm read-only load skips DELETE quietly without logging a warning
        con_rw.execute(
            """
            INSERT INTO archive_policies (cell_id, depth_bin, caution_bin, resource_bin, fitness, policy_code, batch_id, generation, metrics_json)
            VALUES ('cell_d1_v0_t0', 1, 0, 0, 1.0, 'class Policy: pass', 'b0', 0, '{}')
            """
        )
    finally:
        con_rw.close()

    caplog.clear()
    with caplog.at_level("WARNING", logger="lox.engine.map_elites"):
        con_ro = duckdb.connect(db_path, read_only=True)
        try:
            loaded = MAPElitesArchive().load_from_db(con_ro)
        finally:
            con_ro.close()

    assert loaded == 1
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
