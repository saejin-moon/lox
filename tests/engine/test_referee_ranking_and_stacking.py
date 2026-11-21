"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import init_db


def test_outcome_clustering_strips_fn_blame_and_pareto_gate_blocks_one_seed_noise(
    tmp_path,
):
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
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


def test_stage3_tiebreaker_extension_and_confinement_priority(tmp_path):
    """Verifies Stage-3 +64-seed tiebreaker extension promotes genuine 8W-3L surgical near-misses
    while rejecting spurious near-misses, and verifies confinement_horizon priority weighting."""
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
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
    from lox.db.schema import connect_db
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


def test_gen54_stack_precision_guard_protects_clean_winner():
    """
    Gen 54 regression: when peer_2 achieves 7W/0L (p=0.0078) on S_select and peer_4 has
    noisy 4W/2L (p=0.3438), SwarmReferee must NOT let a noisy stack_peer_2_peer_4 cut peer_2.
    """
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    base_skills = {
        "survival/avoid": "def avoid(s):\n    return 1\n",
        "combat/core": "def fight(s):\n    return 1\n",
    }
    select32 = [154000 + i for i in range(32)]
    confirm64 = [10154000 + i for i in range(64)]

    class _Gen54Pool:
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
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            is_sel = "_sel_" in batch_id
            runs = []
            for idx, s in enumerate(s_list):
                if "stack_peer_2_peer_4" in batch_id:
                    # Noisy stack: high variance on S_select (10W/4L, p=0.089) + big spike, fails S_confirm
                    if is_sel:
                        if idx < 10:
                            d, sc = 6, 1500
                        elif idx < 14:
                            d, sc = 2, 300
                        else:
                            d, sc = 3, 500
                    else:
                        d, sc = 3, 500
                elif "peer_2" in batch_id:
                    # Clean 7W/0L on S_select (p=0.0078) and 8W/0L on S_confirm!
                    win_n = 7 if is_sel else 8
                    d = 5 if idx < win_n else 3
                    sc = 1100 if idx < win_n else 500
                else:
                    # peer_4: noisy 4W/2L on S_select (p=0.3438)
                    if idx < 4:
                        d, sc = 5, 1100
                    elif idx < 6:
                        d, sc = 2, 300
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

    p2 = PeerCandidate(
        peer_id="peer_2",
        claimed_deficit_id="slice_frontier_failures",
        claimed_label="avoid_ranged",
        policy_code=base_pol,
        skills={**base_skills, "survival/avoid": "def avoid(s):\n    return 2\n"},
        touched_skills={"survival/avoid"},
        touched_skill_symbols={"survival/avoid": {"avoid"}},
        sandbox_passed=True,
        loc_total=10,
    )
    p4 = PeerCandidate(
        peer_id="peer_4",
        claimed_deficit_id="slice_frontier_failures",
        claimed_label="crossfire",
        policy_code=base_pol,
        skills={**base_skills, "combat/core": "def fight(s):\n    return 2\n"},
        touched_skills={"combat/core"},
        touched_skill_symbols={"combat/core": {"fight"}},
        sandbox_passed=True,
        loc_total=10,
    )
    v54 = SwarmReferee().evaluate_tournament(
        generation=54,
        candidates=[p2, p4],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_Gen54Pool(),
        select_seeds=select32,
        seeds=confirm64,
    )
    assert v54.promoted is True, v54.reason
    assert v54.winner_id == "peer_2"


def test_g9_eligible_winner_outranks_tie_noise_stage1_attempts_and_monster_research_gate(
    tmp_path,
):
    from lox.author.swarm_peer import PeerCandidate
    from lox.author.tools import AuthorToolRegistry
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.N\n"
    )

    class _G9Pool:
        def evaluate_batch(self, batch_id, policy_code, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_inc_" in batch_id:
                runs = [
                    {
                        "run_id": f"{batch_id}_r{i}",
                        "seed": s,
                        "depth": 2 if i < 14 else 1,
                        "max_depth": 4 if i < 4 else (2 if i < 14 else 1),
                        "score": 305,
                        "turns": 1200,
                        "final_ac": 5,
                        "xp_level": 2,
                        "end_reason": "killed by newt",
                    }
                    for i, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            if "peer_tie" in batch_id:
                # Middle-seed tile micro-bump -> slightly higher IQM_prog, but 0 depth gain & -1 score
                runs = [
                    {
                        "run_id": f"{batch_id}_r{i}",
                        "seed": s,
                        "depth": 2 if i < 14 else 1,
                        "max_depth": 4 if i < 4 else (2 if i < 14 else 1),
                        "score": 304,
                        "unique_tiles": 90 if 8 <= i < 24 else 40,
                        "turns": 1200,
                        "final_ac": 5,
                        "xp_level": 2,
                        "end_reason": "killed by newt",
                    }
                    for i, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            if "peer_s1_cut" in batch_id:
                # Stage-1 eliminated candidate with real AC=7 and XP=2 (not shard 10.0 / 1.0)
                runs = [
                    {
                        "run_id": f"{batch_id}_r{i}",
                        "seed": s,
                        "depth": 1,
                        "max_depth": 2 if i == 0 else 1,
                        "score": 110,
                        "turns": 600,
                        "final_ac": 7,
                        "xp_level": 2,
                        "end_reason": "killed by grid bug",
                    }
                    for i, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            # ortho_winner: gains max_depth=6 and +score on tail while tying middle IQR
            runs = [
                {
                    "run_id": f"{batch_id}_r{i}",
                    "seed": s,
                    "depth": 2 if i < 14 else 1,
                    "max_depth": 6 if i < 2 else (4 if i < 4 else (2 if i < 14 else 1)),
                    "score": 325 if i < 4 else 305,
                    "unique_tiles": 40,
                    "turns": 1200,
                    "final_ac": 4,
                    "xp_level": 3,
                    "end_reason": "killed by gnome",
                }
                for i, s in enumerate(s_list)
            ]
            return {"batch_id": batch_id, "runs": runs}

    seeds = [10800 + i for i in range(32)]
    cands = [
        PeerCandidate(
            peer_id="peer_tie",
            claimed_deficit_id="eat",
            claimed_label="tie noise",
            policy_code=base_pol + "\ndef _t():\n    return 1\n",
            skills={},
            sandbox_passed=True,
            loc_total=5,
        ),
        PeerCandidate(
            peer_id="peer_s1_cut",
            claimed_deficit_id="cut",
            claimed_label="cut in s1",
            policy_code=base_pol + "\ndef _c():\n    return 2\n",
            skills={},
            sandbox_passed=True,
            sandbox_metrics={"avg_depth": 3.57, "avg_score": 592.0},
            loc_total=5,
        ),
        PeerCandidate(
            peer_id="ortho_winner",
            claimed_deficit_id="melee+eat",
            claimed_label="real depth win",
            policy_code=base_pol + "\ndef _w():\n    return 3\n",
            skills={},
            sandbox_passed=True,
            loc_total=5,
        ),
    ]
    verdict = SwarmReferee().evaluate_tournament(
        generation=9,
        candidates=cands,
        base_policy=base_pol,
        base_skills={},
        eval_pool=_G9Pool(),
        seeds=seeds,
        race_seeds=16,
    )
    assert verdict.promoted is True
    assert verdict.winner_id == "ortho_winner"

    cut_row = next(a for a in verdict.attempts_summary if a["peer_id"] == "peer_s1_cut")
    assert cut_row["avg_depth"] < 2.0
    assert cut_row["max_depth"] == 2
    assert cut_row["avg_ac"] == 7.0
    assert cut_row["avg_xp_level"] == 2.0

    # query_monster satisfies RESEARCH_REQUIRED alongside query_theories
    class _StubTheories:
        def search_theories(self, query, limit=6, doc_type=None):
            return []

    dbp = str(tmp_path / "research_gate.duckdb")
    init_db(dbp)
    reg = AuthorToolRegistry(db_path=dbp)
    reg.theories_store = _StubTheories()
    reg.configure_generation(generation=10, mode="SWARM_PEER")
    reg.enforce_peer_research_gate = True
    reg.query_theories("floating eye")
    reg.query_monster("floating eye")
    _msg, st = reg.dry_run_policy(base_pol, seed=42, max_turns=15)
    assert st != "RESEARCH_REQUIRED"
