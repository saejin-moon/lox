"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.db.schema import init_db


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


def test_deficit_analyzer_filters_rejected_renders_memory_and_vitals(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer

    db_path = str(tmp_path / "def_clean.duckdb")
    con = init_db(db_path)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, hypothesis_passed, seeds_evaluated, avg_score, avg_depth, p90_depth, max_depth, avg_turns)
            VALUES ('b_prom_58', 58, 'SWARM', 'policy_swarm_g58', TRUE, 2, 650.0, 4.5, 5, 5, 1800.0),
                   ('b_rej_60', 60, 'SWARM', 'policy_swarm_g60', FALSE, 2, 100.0, 1.0, 1, 1, 50.0)
            """
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, final_ac, xp_level, end_reason, died_unarmored, milestones_json)
            VALUES ('r_prom', 'b_prom_58', 58, 1000, 'val-hum-law-fem', 5, 5, 650, 1800, 8, 4, 'killed by fire ant', FALSE, '{"unworn_armor_in_pack": true}'),
                   ('r_prom_armored', 'b_prom_58', 58, 1001, 'val-hum-law-fem', 5, 5, 650, 1800, 4, 4, 'killed by fire ant', FALSE, '{"unworn_armor_in_pack": true}'),
                   ('r_rej', 'b_rej_60', 60, 1000, 'val-hum-law-fem', 1, 1, 100, 50, 10, 1, 'stalled_bad_candidate', TRUE, '{}')
            """
        )
        con.execute(
            """
            INSERT INTO synthesis_logs (log_id, generation, mode, turn_index, turn_type, agent_role, hypothesis_yaml, status)
            VALUES ('s60', 60, 'SWARM_PEER', 8, 'SMOKE_CHECK', 'peer_4', 'hypothesis: H-wf-stall-fix failed', 'PASSED (smoke check)')
            """
        )
    finally:
        con.close()

    thin_pol = "\n".join(f"# l{i}" for i in range(45)) + "\nclass Policy:\n    pass\n"
    five_skills = {f"sk_{i}": "def fn():\n    return None\n" for i in range(5)}
    report = DeficitAnalyzer(db_path).analyze(
        generation=61, current_policy=thin_pol, skills=five_skills
    )
    assert report.bloat_warning is False
    assert "DITCH MONOLITH" not in report.format_brief()
    assert report.recent_attempts and "REJECTED" in report.recent_attempts[0]
    assert "H-wf-stall-fix" in report.recent_attempts[0]
    assert "avg_ac=6.0" in report.vital_summary
    brief = report.format_brief()
    assert "Recent Experiment & Rejection Memory" in brief
    assert "stalled_bad_candidate" not in brief
    assert any(
        c.cluster_id == "slice_unequipped_progression"
        and 1000 in c.sample_seeds
        and 1001 not in c.sample_seeds
        for c in report.fail_clusters
    )


def test_gen41_zero_row_sql_hints_and_deficit_telemetry_window(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
    from lox.author.tools import AuthorToolRegistry

    db_path = str(tmp_path / "hints.duckdb")
    con = init_db(db_path)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, seeds_evaluated, avg_score, avg_depth, p90_depth, max_depth, avg_turns)
            VALUES ('b14', 14, 'SWARM', 'pol_14', 4, 10.0, 1.0, 1, 1, 50.0),
                   ('b40', 40, 'SWARM', 'pol_40', 4, 50.0, 2.0, 2, 2, 120.0)
            """
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, turns, end_reason)
            VALUES ('run_g14_s1', 'b14', 14, 1, 'val-hum-law-fem', 1, 1, 50, 'died'),
                   ('run_g40_s1', 'b40', 40, 1, 'val-hum-law-fem', 2, 2, 120, 'died')
            """
        )
        con.execute(
            """
            INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, action_name, active_function)
            VALUES ('run_g40_s1', 0, 1, 1, 12, 12, 'WAIT', 'leaf_a')
            """
        )
    finally:
        con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    tools.configure_generation(mode="SWARM_PEER", generation=41)

    q_future, sum_future = tools.query_db(
        "SELECT run_id FROM runs WHERE generation = 41"
    )
    assert sum_future == "0 rows" and "results: []" in q_future
    assert "40" in q_future and "champion_runs" in q_future

    q_pruned, sum_pruned = tools.query_db(
        "SELECT * FROM ticks WHERE run_id = 'run_g14_s1'"
    )
    assert sum_pruned == "0 rows" and "results: []" in q_pruned
    assert "champion_ticks" in q_pruned and "40" in q_pruned

    analyzer = DeficitAnalyzer(db_path=db_path)
    report = analyzer.analyze(generation=41, current_policy="")
    assert report.latest_completed_gen == 40
    assert report.ticks_min_gen == 40
    report.fail_clusters = [
        DeficitCluster(
            cluster_id="c_test",
            kind="early_death",
            label="Early combat death",
            run_count=2,
            avg_depth=1.5,
            avg_turns=80.0,
            sample_run_ids=["run_g40_s1"],
            dominant_fns=["leaf_a"],
            evidence="hp dropped",
        )
    ]
    brief = report.format_brief()
    assert "latest_completed_gen=40" in brief
    assert "Gen 41 is currently running" in brief
    assert "run_ids=['run_g40_s1']" in brief


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


def test_gen60_rejection_memory_mech_and_tight_incubation_guard(tmp_path):
    """
    Verifies Gen 58-60 fixes:
    1. DeficitAnalyzer surfaces failed `mech=...` from `theories` inside `recent_attempts`.
    2. incubate_candidate rejects mild depth regressions (`<= -0.15`) when `>= 6` paired shards are evaluated.
    """
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate
    from lox.db.schema import connect_db, init_db

    # 1. Rejected mechanism surfaced in recent_attempts
    dbp = str(tmp_path / "mech_mem.duckdb")
    init_db(dbp)
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b60', 60, 'SWARM', TRUE)"
        )
        con.execute(
            """
            INSERT INTO tournament_attempts (
                attempt_id, generation, peer_id, deficit_id, promoted, sandbox_passed,
                avg_depth, max_depth, avg_ac, policy_exceptions, reject_reason, wins, losses
            )
            VALUES ('a60', 60, 'peer_1', 'slice_high_search_or_horizon_burn', FALSE, TRUE,
                    2.72, 6, 4.1, 0, 'horizon_waste_regression', 8, 22)
            """
        )
        con.execute(
            """
            INSERT INTO theories (doc_id, generation, agent_role, doc_type, title, content)
            VALUES ('th60', 60, 'peer_1', 'THEORY', 'Gen 60 peer_1',
                    'Peer peer_1\nObservation: obs\nMechanism: per-tile search cap 25 + open-floor patrol\nC/Wiki Rule: ')
            """
        )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=61, current_policy="class Policy: pass"
    )
    assert rep.recent_attempts
    assert any(
        "mech=per-tile search cap 25 + open-floor patrol" in ra
        for ra in rep.recent_attempts
    )

    # 2. v3 calibrated incubation: tolerates noisy -0.17 depth drift on 6 shards,
    #    rejects catastrophic <= -0.50 on 6 shards, and rejects >15 ms/step (>3x incumbent) latency regressions.
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )

    class _MildRegPool:
        benchmark = "nethack"

        def __init__(self, drop_count: int = 1, wall_ms: float = 200.0):
            self.drop_count = drop_count
            self.wall_ms = wall_ms

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"r_{s}",
                        "seed": s,
                        "depth": 3 if s <= self.drop_count else 4,
                        "max_depth": 3 if s <= self.drop_count else 4,
                        "score": 500,
                        "turns": 400,
                        "steps": 400,
                        "wall_ms": self.wall_ms,
                        "policy_exceptions": 0,
                        "action_bytes": b"BB",
                    }
                    for s in s_list
                ],
            }

    inc6 = {
        s: {
            "seed": s,
            "depth": 4,
            "max_depth": 4,
            "score": 500,
            "steps": 400,
            "wall_ms": 200.0,
            "action_bytes": b"AA",
        }
        for s in range(1, 7)
    }
    # Mild -0.17 drift across 6 shards now passes (noise tolerance)
    c6_mild = incubate_candidate(
        PeerCandidate(
            peer_id="p_mild6",
            claimed_deficit_id="d1",
            claimed_label="mild_reg",
            policy_code=base_pol,
            skills={},
        ),
        deficit_report=DeficitReport(generation=61, regime="PLATEAU"),
        claimed_deficit=DeficitCluster(cluster_id="d1", kind="k", label="l"),
        eval_pool=_MildRegPool(drop_count=1),
        incumbent_shard_runs=inc6,
        shard_seeds=list(range(1, 7)),
    )
    assert c6_mild.sandbox_passed is True

    # Catastrophic -0.67 (4 of 6 seeds drop by 1) is rejected (<= -0.50)
    c6_cat = incubate_candidate(
        PeerCandidate(
            peer_id="p_cat6",
            claimed_deficit_id="d1",
            claimed_label="cat_reg",
            policy_code=base_pol,
            skills={},
        ),
        deficit_report=DeficitReport(generation=61, regime="PLATEAU"),
        claimed_deficit=DeficitCluster(cluster_id="d1", kind="k", label="l"),
        eval_pool=_MildRegPool(drop_count=4),
        incumbent_shard_runs=inc6,
        shard_seeds=list(range(1, 7)),
    )
    assert c6_cat.sandbox_passed is False
    assert "Incubation paired depth regression" in c6_cat.error

    # Latency guard rejects > 15 ms/step and > 3x incumbent (20 ms/step vs 0.5 ms/step)
    c6_slow = incubate_candidate(
        PeerCandidate(
            peer_id="p_slow6",
            claimed_deficit_id="d1",
            claimed_label="slow_loop",
            policy_code=base_pol,
            skills={},
        ),
        deficit_report=DeficitReport(generation=61, regime="PLATEAU"),
        claimed_deficit=DeficitCluster(cluster_id="d1", kind="k", label="l"),
        eval_pool=_MildRegPool(drop_count=0, wall_ms=8000.0),
        incumbent_shard_runs=inc6,
        shard_seeds=list(range(1, 7)),
    )
    assert c6_slow.sandbox_passed is False
    assert "Incubation latency guard" in c6_slow.error

    inc4 = {s: inc6[s] for s in range(1, 5)}
    c4 = incubate_candidate(
        PeerCandidate(
            peer_id="p_ok4",
            claimed_deficit_id="d1",
            claimed_label="mild_4",
            policy_code=base_pol,
            skills={},
        ),
        deficit_report=DeficitReport(generation=61, regime="PLATEAU"),
        claimed_deficit=DeficitCluster(cluster_id="d1", kind="k", label="l"),
        eval_pool=_MildRegPool(drop_count=1),
        incumbent_shard_runs=inc4,
        shard_seeds=list(range(1, 5)),
    )
    assert c4.sandbox_passed is True


def test_gen62_single_mechanism_prompt_phrase_grouping_and_zero_loss_stack(tmp_path):
    """
    Verifies Gen 62 fixes:
    1. Peer system prompt instructs Single-Mechanism Isolation (no 3-4 sub-fix bundles).
    2. DeficitAnalyzer groups split terminal phrasings sharing `the <phrase>` (e.g. 3+3=6 runs)
       into a single `recurring_terminal_state` cluster.
    3. SwarmReferee allows a clean zero-loss 2W/0L (p=0.25) candidate to stack onto a p<0.05 winner.
    """
    from lox.author.prompts import build_system_prompt
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.schema import connect_db, init_db
    from lox.engine.swarm_referee import SwarmReferee

    # 1. Single-Mechanism Isolation prompt rule
    sys_p = build_system_prompt(is_swarm=True, is_peer=True)
    assert "Single-Mechanism Isolation" in sys_p

    # 2. Split phrase grouping in DeficitAnalyzer (3 + 3 = 6 runs >= recurring_end_min=4)
    dbp = str(tmp_path / "phrase_group.duckdb")
    init_db(dbp)
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b62', 62, 'SWARM', TRUE)"
        )
        for i in range(3):
            con.execute(
                """
                INSERT INTO runs (
                    run_id, batch_id, generation, seed, depth, max_depth, turns,
                    score, end_reason, unique_tiles, pct_wait, pct_search, is_shard
                )
                VALUES (?, 'b62', 62, ?, 3, 3, 1200, 400, 'You miss the crystal sentinel', 250, 0.0, 15.0, FALSE)
                """,
                [f"m_{i}", 6200 + i],
            )
            con.execute(
                """
                INSERT INTO runs (
                    run_id, batch_id, generation, seed, depth, max_depth, turns,
                    score, end_reason, unique_tiles, pct_wait, pct_search, is_shard
                )
                VALUES (?, 'b62', 62, ?, 3, 3, 1250, 420, 'You hit the crystal sentinel! You are stunned!', 260, 0.0, 15.0, FALSE)
                """,
                [f"h_{i}", 6210 + i],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=63, current_policy="class Policy: pass"
    )
    rec_cs = [c for c in rep.fail_clusters if c.kind == "recurring_terminal_state"]
    assert len(rec_cs) == 1
    assert rec_cs[0].run_count == 6
    assert "crystal sentinel" in rec_cs[0].label

    # 3. Clean zero-loss 2W/0L (p=0.25) stacks cleanly onto a 7W/0L (p=0.0078) winner -> 9W/0L stack wins!
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    base_skills = {
        "survival/avoid": "def avoid(s):\n    return 1\n",
        "goals/gear": "def equip(s):\n    return 1\n",
    }
    select32 = [162000 + i for i in range(32)]
    confirm64 = [10162000 + i for i in range(64)]

    class _ZeroLossStackPool:
        def __init__(self):
            self.seen_batches: list[str] = []

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            self.seen_batches.append(batch_id)
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                win_n = 0
            elif "stack_" in batch_id:
                win_n = 9
            elif "peer_a" in batch_id:
                win_n = 7
            else:
                win_n = 2
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"r_{s}",
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

    p_a = PeerCandidate(
        peer_id="peer_a",
        claimed_deficit_id="d1",
        claimed_label="a",
        policy_code=base_pol,
        skills={**base_skills, "survival/avoid": "def avoid(s):\n    return 2\n"},
        touched_skills={"survival/avoid"},
        touched_skill_symbols={"survival/avoid": {"avoid"}},
        sandbox_passed=True,
        loc_total=10,
    )
    p_b = PeerCandidate(
        peer_id="peer_b",
        claimed_deficit_id="d2",
        claimed_label="b",
        policy_code=base_pol,
        skills={**base_skills, "goals/gear": "def equip(s):\n    return 2\n"},
        touched_skills={"goals/gear"},
        touched_skill_symbols={"goals/gear": {"equip"}},
        sandbox_passed=True,
        loc_total=10,
    )
    zpool = _ZeroLossStackPool()
    v62 = SwarmReferee().evaluate_tournament(
        generation=62,
        candidates=[p_a, p_b],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=zpool,
        select_seeds=select32,
        seeds=confirm64,
    )
    assert any("stack_" in b for b in zpool.seen_batches)
    assert v62.promoted is True, v62.reason
    assert v62.winner_id == "stack_peer_a_peer_b"


def test_deficit_analyzer_excludes_non_champion_peer_and_pilot_runs(tmp_path):
    """
    Regression test for G68-G71 leak: `micro_pilot_baseline` (`peer_mp_inc_*`),
    `incubate_baseline` (`incubate_inc_*`), `select`, and `confirm_challenger`
    rows stored in the unified `runs` table must NEVER pollute `DeficitAnalyzer`
    (even though their batch_ids contain `_inc_`).
    """
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import init_db

    dbp = str(tmp_path / "no_leak.duckdb")
    init_db(dbp).close()
    logger = BatchDatabaseLogger(db_path=dbp)

    # 1. Seed 16 clean champion runs at Dlvl 2-3 (starved / max_steps)
    logger.save_policy(
        "pol_g71", 71, "SWARM", "class Policy: pass", "", {}, 2.5, 3, is_promoted=True
    )
    champ_runs = [
        {
            "run_id": f"champ_71_{i}",
            "seed": 1000 + i,
            "depth": 2,
            "max_depth": 2 if i < 12 else 3,
            "score": 200,
            "turns": 1500,
            "end_reason": "starved",
            "pct_search": 40.0,
            "unique_tiles": 200,
            "game_log": "T1500: You faint from lack of food.",
            "run_kind": "champion",
        }
        for i in range(16)
    ]
    logger.save_batch(
        batch_id="batch_swarm_g71_champ",
        generation=71,
        mode="SWARM",
        parent_policy_id="none",
        candidate_policy_id="pol_g71",
        target_functions=["step"],
        runs=champ_runs,
        episodes_discarded=0,
        batch_metrics={"avg_depth": 2.25, "max_depth": 3, "avg_score": 200.0},
        verdict={"passed": True, "reason": "ok"},
    )

    # 2. Flood unified `runs` with 12 duplicate Dlvl-11 guard copies whose batch_ids match `*_inc_*`
    phantom_runs = [
        {
            "seed": 99999,
            "depth": 11,
            "max_depth": 11,
            "score": 5000,
            "turns": 3750,
            "end_reason": "It bites!  It bites! | You die",
            "game_log": "[T3750 D11 N] It bites!  It bites! | You die",
        }
        for _ in range(12)
    ]
    logger.save_runs_and_ticks(
        batch_id="peer_mp_inc_g71_peer_1_a1_12345",
        generation=71,
        run_kind="micro_pilot_baseline",
        peer_id="peer_1",
        attempt_num=1,
        runs=phantom_runs,
    )
    logger.save_runs_and_ticks(
        batch_id="incubate_inc_g71_12345",
        generation=71,
        run_kind="incubate_baseline",
        peer_id="incumbent",
        runs=phantom_runs[:4],
    )

    rep = DeficitAnalyzer(dbp).analyze(
        generation=72, current_policy="class Policy: pass"
    )
    assert "N=16" in rep.vital_summary, rep.vital_summary
    assert 99999 not in rep.d_succ_seeds
    assert all("It bites!" not in c.label for c in rep.fail_clusters), [
        c.label for c in rep.fail_clusters
    ]


def test_v3_deficit_12gen_memory_fresh_guards_and_expert_parity_gaps(tmp_path):
    """Verifies Component 1: 12-gen rejection memory, fresh d_succ_seeds (<=10 gens), and domain-agnostic expert_parity_gaps."""
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.db.schema import init_db

    dbp = str(tmp_path / "v3_deficits.duckdb")
    con = init_db(dbp)
    try:
        # Old gen 1 champion with high depth (should be excluded from fresh d_succ_seeds when latest is gen 20)
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b1', 1, 'SWARM', TRUE), ('b20', 20, 'SWARM', TRUE)"
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, final_ac, xp_level, end_reason, run_kind)
            VALUES
              ('r_old_deep', 'b1', 1, 1111, 12, 12, 5000, 3000, 1, 8, 'killed', 'champion'),
              ('r_fresh_1', 'b20', 20, 2001, 4, 4, 900, 1200, 4, 4, 'killed', 'champion'),
              ('r_fresh_2', 'b20', 20, 2002, 5, 5, 1100, 1400, 3, 5, 'killed', 'champion'),
              ('r_fresh_3', 'b20', 20, 2003, 2, 2, 200, 600, 8, 2, 'starved', 'champion')
            """
        )
        # 10-gen-old rejection attempt (within 12-gen window, outside old 4-gen window)
        con.execute(
            """
            INSERT INTO tournament_attempts (
                attempt_id, generation, peer_id, deficit_id, promoted, sandbox_passed,
                avg_depth, max_depth, avg_ac, policy_exceptions, reject_reason, wins, losses,
                winsor_delta, sign_perm_p
            )
            VALUES ('att_g10', 10, 'reactive_1', 'slice_starvation', FALSE, TRUE, 3.8, 6, 4.0, 0, 'below_sign_perm_alpha', 6, 4, 0.14, 0.19)
            """
        )
        # Populate champion_ticks (heavy SEARCH, zero QUAFF/DROP) vs expert_ticks (active QUAFF/DROP/E, lower SEARCH)
        for i in range(40):
            con.execute(
                """
                INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, ac, xp_level, action_name, active_function)
                VALUES ('r_fresh_1', ?, ?, 2, 15, 20, 6, 2, ?, 'explore')
                """,
                [i, i + 1, "SEARCH" if i < 20 else "E"],
            )
        con.execute(
            "INSERT INTO expert_runs (run_id, source, role, max_depth, final_ac, xp_level, turns, score) VALUES ('exp_1', 'nao', 'Val', 40, -10, 14, 25000, 100000)"
        )
        for i in range(40):
            act_e = (
                "QUAFF"
                if i < 6
                else ("DROP" if i < 10 else ("SEARCH" if i < 12 else "E"))
            )
            con.execute(
                """
                INSERT INTO expert_ticks (run_id, step_idx, turn, depth, hp, max_hp, ac, xp_level, action_name)
                VALUES ('exp_1', ?, ?, 2, 20, 20, 2, 4, ?)
                """,
                [i, i + 1, act_e],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=21, current_policy="class Policy: pass"
    )
    # 1. 12-gen rejection memory includes Gen 10 with winsor_delta / sign_perm_p
    assert any(
        "Gen 10" in line and "WΔ=+0.14" in line and "p=0.190" in line
        for line in rep.recent_attempts
    ), rep.recent_attempts
    # 2. Fresh guard seeds exclude stale Gen 1 seed 1111
    assert 1111 not in rep.d_succ_seeds
    assert 2002 in rep.d_succ_seeds
    # 3. Domain-agnostic expert_parity_gaps detects unused QUAFF & overused SEARCH
    assert "State Parity:" in rep.expert_parity_gaps
    assert "QUAFF" in rep.expert_parity_gaps
    assert "SEARCH" in rep.expert_parity_gaps


def test_sparse_expert_ticks_full_batch_stalled_and_cooldown(tmp_path):
    """Verifies Fixes 1-3: sparse expert_ticks suppress fake action %, 256-run full clustering surfaces stalled_zero_turn + >=2.8% recurring outcomes, and cooldown tracks repeated rejections."""
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.db.schema import init_db

    dbp = str(tmp_path / "plan7_deficits.duckdb")
    con = init_db(dbp)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, hypothesis_passed)
            VALUES ('batch_swarm_g76_prom', 76, 'SWARM', TRUE),
                   ('batch_swarm_g87_inc_retained_1', 87, 'SWARM', FALSE)
            """
        )
        # 256 champion runs in Gen 87: 16 stalled_zero_turn (with high pct_search!), 9 floating eye split phrasings, rest combat
        for i in range(256):
            if i < 16:
                er, pct_s, turns, fn = (
                    "stalled_zero_turn",
                    42.0,
                    2200,
                    "explore._dir_to",
                )
            elif i < 21:
                er, pct_s, turns, fn = (
                    "You hit the floating eye! You are frozen",
                    20.0,
                    1200,
                    "explore._search_fallback",
                )
            elif i < 25:
                er, pct_s, turns, fn = (
                    "You miss the floating eye",
                    20.0,
                    1200,
                    "explore._search_fallback",
                )
            else:
                er, pct_s, turns, fn = f"killed_by_mob_{i % 40}", 25.0, 1500, "combat"
            con.execute(
                """
                INSERT INTO runs (
                    run_id, batch_id, generation, seed, depth, max_depth, score, turns,
                    final_ac, xp_level, end_reason, pct_search, last_active_function, run_kind
                )
                VALUES (?, 'batch_swarm_g87_inc_retained_1', 87, ?, 3, 3, 500, ?, 5, 3, ?, ?, ?, 'champion')
                """,
                [f"r87_{i}", 10870000 + i, turns, er, pct_s, fn],
            )
        # Champion ticks include WEAR via seq(WEAR,a), zero QUAFF
        for step in range(1, 201):
            act = "seq(WEAR,a)" if step == 1 else "SEARCH"
            con.execute(
                "INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, ac, xp_level, action_name, active_function) VALUES ('r87_0', ?, ?, 2, 16, 16, 5, 3, ?, 'explore')",
                [step, step, act],
            )
        # Sparse multi-depth expert_ticks (10 rows across depths 1..3, turns 1..9040 -> low density & span coverage)
        for idx_e, act_e in enumerate(
            [
                "WEAR",
                "WIELD",
                "QUAFF",
                "EAT",
                "OPEN",
                "KICK",
                "DOWN",
                "SEARCH",
                "QUAFF",
                "QUAFF",
            ],
            start=0,
        ):
            depth_e = (idx_e // 3) + 1
            turn_e = (depth_e - 1) * 3000 + (idx_e % 3) * 20 + 1
            con.execute(
                "INSERT INTO expert_ticks (run_id, step_idx, turn, depth, hp, max_hp, ac, xp_level, action_name) VALUES ('exp_sparse', ?, ?, ?, 20, 20, 2, 4, ?)",
                [idx_e + 1, turn_e, depth_e, act_e],
            )
        # Repeated rejections in last 4 gens trigger cooldown
        for g_fail in (85, 86, 87):
            con.execute(
                """
                INSERT INTO tournament_attempts (attempt_id, generation, peer_id, deficit_id, promoted, sandbox_passed)
                VALUES (?, ?, 'reactive_1', 'slice_high_search_or_horizon_burn', FALSE, FALSE)
                """,
                [f"att_{g_fail}", g_fail],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=88, current_policy="class Policy: pass"
    )
    # 1. Sparse expert_ticks must NOT emit fake action % or fake `vs expert ~Xt`
    assert "Under-Used/Missing Expert Actions" not in rep.expert_parity_gaps
    assert "vs expert " not in rep.expert_parity_gaps.split("vs expert@")[0]
    assert "vs expert ~" not in rep.efficiency_summary
    assert "sparse progression milestones" in rep.expert_parity_gaps
    # 2. All 16 stalled_zero_turn runs preserved (not truncated to 64 and not swallowed by is_high_burn)
    cmap = {c.cluster_id: c for c in rep.fail_clusters}
    assert "slice_stalled" in cmap and cmap["slice_stalled"].run_count == 16
    assert "explore._dir_to" in cmap["slice_stalled"].evidence
    # 3. 9/256 floating eye deaths surface via recurring_end_min = max(4, 256//36) = 7
    assert any("floating eye" in c.label.lower() for c in rep.fail_clusters)
    # 4. Cooldown flags slice_high_search_or_horizon_burn (failed 3x since Gen 76 promotion)
    assert rep.cooled_down_deficits.get("slice_high_search_or_horizon_burn") == 3
