"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import init_db


def test_referee_closes_noop_and_significance_loopholes():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee
    from lox.env.nethack.adapter import NetHackAdapter

    # All-depth horizon waste replaces the Goodharted pct_wait zombie heuristic
    adapter = NetHackAdapter()
    assert (
        adapter.is_horizon_waste(
            {"depth": 4, "end_reason": "max_steps", "turns": 10000}
        )
        is True
    )
    assert (
        adapter.is_horizon_waste({"depth": 2, "end_reason": "stalled_zero_turn"})
        is True
    )
    assert (
        adapter.is_horizon_waste({"depth": 4, "end_reason": "killed by gnome"}) is False
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


def test_promotion_gate_v2_split_sample_anti_goodhart_safety():
    """Verifies every Promotion Gate v2 invariant:
    1. Anti-Goodhart NetHackAdapter progression_metric.
    2. Exact Paired Sign-Permutation & exact small-N Wilcoxon calibration (4W-0L=0.0625, 5W-0L=0.03125).
    3. Split-Sample Select(32) -> Confirm(64): zero fallback leakage on noise winner, unbiased S_confirm logging.
    4. 4 benchmark-agnostic safety invariants (horizon_waste_rate at D3+, auto_esc, cvar25) & AST bloat alpha bar.
    """
    import numpy as np

    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.hypothesis import (
        cvar_bottom_quantile,
        exact_sign_permutation_p,
        progression_metric,
        wilcoxon_signed_rank_p,
        winsorized_mean,
    )
    from lox.engine.swarm_referee import SwarmReferee

    # 1. Anti-Goodhart NetHackAdapter progression_metric
    for depth in (1, 2, 3, 4):
        normal_run = {
            "depth": depth,
            "max_depth": depth,
            "xp_level": 2,
            "score": 180,
            "final_ac": 5,
            "end_reason": "killed by goblin",
        }
        camper_run = {
            "depth": depth,
            "max_depth": depth,
            "xp_level": 6,
            "score": 2500,
            "final_ac": 1,
            "turns": 10000,
            "end_reason": "max_steps",
        }
        assert progression_metric(camper_run) < progression_metric(normal_run), (
            f"D{depth} max_steps camper must score strictly below genuine D{depth} run"
        )

    # 2. Exact Paired Sign-Permutation & Exact Wilcoxon (fixes 4W-0L anti-conservative normal approx!)
    p4, w4, l4, t4 = exact_sign_permutation_p(np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    assert (w4, l4, t4) == (4, 0, 2)
    assert abs(p4 - 0.0625) < 1e-9
    assert (
        abs(wilcoxon_signed_rank_p(np.array([1.0, 1.0, 1.0, 1.0]))[0] - 0.0625) < 1e-9
    )

    p5, w5, l5, _ = exact_sign_permutation_p(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert (w5, l5) == (5, 0)
    assert abs(p5 - 0.03125) < 1e-9
    assert (
        winsorized_mean([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.1) < 2.0
    )
    assert (
        abs(cvar_bottom_quantile([1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 0.25) - 1.5)
        < 1e-9
    )

    # 3. Split-Sample Select(32) -> Confirm(64): zero fallback leakage when C* is noise on S_confirm
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    select32 = [110000 + i for i in range(32)]
    confirm64 = [10110000 + i for i in range(64)]

    class _SplitNoFallbackPool:
        def __init__(self):
            self.confirm_evaluated: list[str] = []

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_conf_" in batch_id and "_inc_" not in batch_id:
                self.confirm_evaluated.append(batch_id)
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
            if "peer_overfit" in batch_id:
                # Crushes S_select (+3 depth), completely null on S_confirm!
                d = 6 if is_sel else 3
                sc = 1800 if is_sel else 500
            else:
                # peer_runnerup: smaller gain on S_select (+1), strong on S_confirm (+2)
                # Under strict split-sample, ONLY C* (peer_overfit) goes to S_confirm -> ZERO fallback!
                d = 4 if is_sel else 5
                sc = 900 if is_sel else 1400
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1500,
                        "end_reason": "killed",
                    }
                    for s in s_list
                ],
            }

    c_overfit = PeerCandidate(
        peer_id="peer_overfit",
        claimed_deficit_id="d1",
        claimed_label="overfit",
        policy_code=base_pol + "\n_OF = 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    c_runner = PeerCandidate(
        peer_id="peer_runnerup",
        claimed_deficit_id="d2",
        claimed_label="runnerup",
        policy_code=base_pol + "\n_RU = 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    split_pool = _SplitNoFallbackPool()
    v_split = SwarmReferee().evaluate_tournament(
        generation=40,
        candidates=[c_overfit, c_runner],
        base_policy=base_pol,
        base_skills={},
        eval_pool=split_pool,
        select_seeds=select32,
        seeds=confirm64,
    )
    assert v_split.promoted is False
    assert len(split_pool.confirm_evaluated) == 1
    assert "peer_overfit" in split_pool.confirm_evaluated[0]
    assert len(v_split.winner_batch_eval["runs"]) == 64

    # 4. Safety Invariants: D4 horizon_waste_rate regression & auto_esc_regression & AST bloat alpha bar
    class _SafetyPool:
        def __init__(self, mode: str):
            self.mode = mode

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
                            "auto_esc_count": 0,
                            "end_reason": "killed",
                        }
                        for s in s_list
                    ],
                }
            runs = []
            for idx, s in enumerate(s_list):
                if self.mode == "horizon_waste_d4":
                    # Wins 50 seeds to D4, but 8 seeds hit D4 max_steps (12.5% > 5% tol)
                    er = "max_steps" if idx < 8 else "killed"
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": 4,
                            "max_depth": 4,
                            "score": 900,
                            "turns": 10000 if er == "max_steps" else 1500,
                            "auto_esc_count": 0,
                            "end_reason": er,
                        }
                    )
                elif self.mode == "auto_esc":
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": 4,
                            "max_depth": 4,
                            "score": 900,
                            "turns": 1500,
                            "auto_esc_count": 1 if idx < 5 else 0,
                            "end_reason": "killed",
                        }
                    )
                else:
                    # AST bloat bar: 5W-0L (exact p = 0.03125): passes alpha=0.05, fails bloated eff_alpha=0.025!
                    d = 5 if idx < 5 else 3
                    runs.append(
                        {
                            "run_id": f"c_{s}",
                            "seed": s,
                            "depth": d,
                            "max_depth": d,
                            "score": 1200 if idx < 5 else 500,
                            "turns": 1500,
                            "auto_esc_count": 0,
                            "end_reason": "killed",
                        }
                    )
            return {"batch_id": batch_id, "runs": runs}

    big_base = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        + "".join(f"        x_{i} = {i} + 1\n" for i in range(15))
        + "        return Action.N\n"
    )
    bloated_cand_code = big_base + "".join(
        f"_BLOAT_{i} = {i} * 2 + 1\n" for i in range(25)
    )
    lean_cand_code = big_base + "_LEAN = 1\n"

    v_hw = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_hw",
                claimed_deficit_id="d1",
                claimed_label="hw",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("horizon_waste_d4"),
        seeds=confirm64,
    )
    assert v_hw.promoted is False
    assert v_hw.attempts_summary[0]["reject_reason"] == "horizon_waste_regression"

    v_esc = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_esc",
                claimed_deficit_id="d1",
                claimed_label="esc",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("auto_esc"),
        seeds=confirm64,
    )
    assert v_esc.promoted is False
    assert v_esc.attempts_summary[0]["reject_reason"] == "auto_esc_regression"

    # Lean 5W-0L (p=0.03125 < 0.05) promotes; bloated >+25% AST 5W-0L (0.03125 >= 0.025) is rejected!
    v_lean = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_lean",
                claimed_deficit_id="d1",
                claimed_label="lean",
                policy_code=lean_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=20,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("ast_bar"),
        seeds=confirm64,
    )
    assert v_lean.promoted is True, v_lean.reason

    v_bloat = SwarmReferee().evaluate_tournament(
        generation=41,
        candidates=[
            PeerCandidate(
                peer_id="p_bloat",
                claimed_deficit_id="d1",
                claimed_label="bloat",
                policy_code=bloated_cand_code,
                skills={},
                sandbox_passed=True,
                loc_total=45,
            )
        ],
        base_policy=big_base,
        base_skills={},
        eval_pool=_SafetyPool("ast_bar"),
        seeds=confirm64,
    )
    assert v_bloat.promoted is False
    assert (
        v_bloat.attempts_summary[0]["reject_reason"]
        == "ast_bloat_below_tightened_alpha"
    )


def test_swarm_referee_blocks_hard_stall_regression_on_minor_depth_tick():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )

    class _StubPool:
        def __init__(self, responses):
            self.responses = responses

        def evaluate_batch(self, batch_id, policy_code, **kwargs):
            for key, val in self.responses.items():
                if key in batch_id:
                    return val
            return self.responses["inc"]

    inc_runs = [
        {
            "run_id": f"inc_{i}",
            "seed": 1000 + i,
            "depth": 4 if i < 3 else 5,
            "max_depth": 4 if i < 3 else 5,
            "score": 600,
            "turns": 1500,
            "end_reason": "killed by grid bug",
            "summary_digest": "Overview: Depth=4",
        }
        for i in range(4)
    ]
    stalled_minor_runs = [
        {
            "run_id": f"st_{i}",
            "seed": 1000 + i,
            "depth": 5 if i < 2 else (3 if i == 2 else 4),
            "max_depth": 5 if i < 2 else (3 if i == 2 else 4),
            "score": 560,
            "turns": 400 if i == 0 else 1400,
            "end_reason": "stalled_zero_turn" if i == 0 else "killed by gnome",
            "summary_digest": "Overview: Stalled",
        }
        for i in range(4)
    ]
    clean_major_runs = [
        {
            "run_id": f"mj_{i}",
            "seed": 1000 + i,
            "depth": 5,
            "max_depth": 5,
            "score": 700,
            "turns": 1600,
            "end_reason": "killed by gnome",
            "summary_digest": "Overview: Depth=5",
        }
        for i in range(4)
    ]

    cand_stalled = PeerCandidate(
        peer_id="peer_stalled",
        claimed_deficit_id="c1",
        claimed_label="minor tick + stall",
        policy_code=base_pol + "\ndef _stalled_marker():\n    return 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    cand_clean = PeerCandidate(
        peer_id="peer_clean",
        claimed_deficit_id="c2",
        claimed_label="clean breakthrough",
        policy_code=base_pol + "\ndef _clean_marker():\n    return 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )

    referee = SwarmReferee()
    v1 = referee.evaluate_tournament(
        generation=52,
        candidates=[cand_stalled],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_StubPool(
            {"_inc_": {"runs": inc_runs}, "peer_stalled": {"runs": stalled_minor_runs}}
        ),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert v1.promoted is False

    v2 = SwarmReferee()
    v2_res = v2.evaluate_tournament(
        generation=52,
        candidates=[cand_stalled, cand_clean],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_StubPool(
            {
                "_inc_": {"runs": inc_runs},
                "peer_stalled": {"runs": stalled_minor_runs},
                "peer_clean": {"runs": clean_major_runs},
            }
        ),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert v2_res.promoted is True
    assert v2_res.winner_id == "peer_clean"


def test_swarm_referee_no_overwrite_noop_rejected_filter_and_parallel_tasks():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    noop_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        # comment only\n"
        "        return Action.WAIT\n"
    )
    cand_noop = PeerCandidate(
        peer_id="peer_noop",
        claimed_deficit_id="c1",
        claimed_label="noop",
        policy_code=noop_pol,
        skills={},
        sandbox_passed=True,
        loc_total=4,
    )
    cand_worse = PeerCandidate(
        peer_id="peer_worse",
        claimed_deficit_id="c2",
        claimed_label="worse",
        policy_code=base_pol + "\ndef _extra(s):\n    return None\n",
        skills={},
        sandbox_passed=True,
        loc_total=6,
    )
    cand_better = PeerCandidate(
        peer_id="peer_better",
        claimed_deficit_id="c3",
        claimed_label="better",
        policy_code=base_pol.replace("Action.WAIT", "Action.SEARCH"),
        skills={},
        sandbox_passed=True,
        loc_total=4,
    )

    referee = SwarmReferee()
    # 1. No-op candidate is filtered even when pool has size 1
    assert referee.select_finalists([cand_noop], base_pol, {}) == []

    # 2. Evaluate incumbent + 2 challengers via parallel `evaluate_tasks`
    class _ParallelStubPool:
        role = "val-hum-law-fem"
        max_steps = 100
        benchmark = "nethack"

        def __init__(self):
            self.task_batches = []

        def evaluate_tasks(self, tasks):
            self.task_batches.append(len(tasks))
            out = []
            for tk in tasks:
                rid = tk["run_id"]
                sd = tk["seed"]
                if "peer_better" in rid:
                    d, sc = 5, 500
                elif "peer_worse" in rid:
                    d, sc = 1, 10
                else:
                    d, sc = 3, 200
                out.append(
                    {
                        "run_id": rid,
                        "seed": sd,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 100,
                        "end_reason": "died",
                        "summary_digest": f"Overview: Depth={d}",
                    }
                )
            return out

    par_pool = _ParallelStubPool()
    verdict_rej = referee.evaluate_tournament(
        generation=55,
        candidates=[cand_worse],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert par_pool.task_batches == [
        4
    ]  # incumbent (2) + peer_worse (2) dispatched together
    assert verdict_rej.promoted is False
    assert "inc_retained" in verdict_rej.winner_batch_eval["batch_id"]
    assert verdict_rej.winner_batch_eval["avg_depth"] == 3.0

    # Rejected fingerprint is never re-admitted against the same incumbent
    assert referee.select_finalists([cand_worse], base_pol, {}) == []

    # When 0 finalists remain, retained incumbent uses a fresh cloned batch_id (never overwrites prior batch)
    v_empty = referee.evaluate_tournament(
        generation=56,
        candidates=[cand_worse],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert v_empty.promoted is False
    assert (
        v_empty.winner_batch_eval["batch_id"]
        != verdict_rej.winner_batch_eval["batch_id"]
    )
    assert "g56_inc_retained" in v_empty.winner_batch_eval["batch_id"]

    # Clean superior challenger wins
    v_win = referee.evaluate_tournament(
        generation=57,
        candidates=[cand_better],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert v_win.promoted is True
    assert v_win.winner_id == "peer_better"


def test_swarm_referee_stall_tol_holdout_telemetry_and_experiments_discoveries(
    tmp_path,
):
    from unittest.mock import patch

    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.schema import connect_db, init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    cand_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.DOWN\n"
    )

    # 1. Across 32 seeds, 1 extra hard stall with +0.125 depth and +40 score passes _passes_score_guard
    class _Pool32:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "p4" in batch_id and not batch_id.startswith("holdout_"):
                runs = [
                    {
                        "run_id": f"c_{s}",
                        "seed": s,
                        "depth": 3 if idx < 4 else 2,
                        "max_depth": 3 if idx < 4 else 2,
                        "score": 140,
                        "turns": 450,
                        "final_ac": 7,
                        "xp_level": 2,
                        "end_reason": "stalled_zero_turn" if idx == 0 else "killed",
                    }
                    for idx, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            if batch_id.startswith("holdout_") and "_inc_" not in batch_id:
                runs = [
                    {
                        "run_id": f"hc_{s}",
                        "seed": s,
                        "depth": 3,
                        "max_depth": 3,
                        "score": 150,
                        "turns": 500,
                        "final_ac": 7,
                        "xp_level": 2,
                    }
                    for s in s_list
                ]
                return {"batch_id": batch_id, "runs": runs}
            runs = [
                {
                    "run_id": f"i_{s}",
                    "seed": s,
                    "depth": 2,
                    "max_depth": 2,
                    "score": 100,
                    "turns": 400,
                    "final_ac": 7,
                    "xp_level": 2,
                    "end_reason": "killed",
                }
                for s in s_list
            ]
            return {"batch_id": batch_id, "runs": runs}

    ref = SwarmReferee()
    c4 = PeerCandidate(
        peer_id="p4",
        claimed_deficit_id="slice_early_terminations",
        claimed_label="Early termination gap",
        policy_code=cand_pol,
        skills={},
        touched_branches={"step"},
        sandbox_passed=True,
    )
    verdict32 = ref.evaluate_tournament(
        generation=41,
        candidates=[c4],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_Pool32(),
        seeds=[14100 + i for i in range(32)],
        holdout_seeds=[100000 + i for i in range(16)],
    )
    assert verdict32.promoted is True, verdict32.reason
    assert verdict32.holdout_rejected is False
    assert verdict32.attempts_summary[0]["holdout_avg_depth"] == 3.0

    # 2. Holdout rejection persists holdout_rejected=True and surfaces HOLD_REJECTED in DeficitAnalyzer
    class _HoldoutDropPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if batch_id.startswith("holdout_") and "_inc_" not in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"hd_{s}",
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 0,
                            "turns": 50,
                        }
                        for s in s_list
                    ],
                }
            if "_inc_" in batch_id:
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"hi_{s}",
                            "seed": s,
                            "depth": 2,
                            "max_depth": 2,
                            "score": 80,
                            "turns": 250,
                        }
                        for s in s_list
                    ],
                }
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"cp_{s}",
                        "seed": s,
                        "depth": 3,
                        "max_depth": 3,
                        "score": 200,
                        "turns": 400,
                    }
                    for s in s_list
                ],
            }

    db_p = str(tmp_path / "holdout.duckdb")
    init_db(db_p)
    v_h = SwarmReferee().evaluate_tournament(
        generation=5,
        candidates=[c4],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_HoldoutDropPool(),
        seeds=[2000 + i for i in range(8)],
        holdout_seeds=[100000 + i for i in range(8)],
    )
    assert v_h.promoted is False
    assert v_h.holdout_rejected is True
    assert "holdout" in v_h.reason.lower()
    BatchDatabaseLogger(db_path=db_p).save_tournament_attempts(v_h.attempts_summary)
    rep_h = DeficitAnalyzer(db_p).analyze(generation=6, current_policy=base_pol)
    assert any("HOLD_REJECTED" in line for line in rep_h.recent_attempts)

    # 3. SwarmOrchestrator persists experiments & discoveries and uses disjoint Split-Sample S_select -> S_confirm seeds
    seen_select_sets: list[list[int]] = []
    seen_confirm_sets: list[list[int]] = []

    class _OrchPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            if "_sel_" in batch_id and "_inc_" not in batch_id:
                seen_select_sets.append(list(s_list))
            if "_conf_" in batch_id and "_inc_" not in batch_id:
                seen_confirm_sets.append(list(s_list))
            is_cand = "p1" in batch_id
            d = 3 if is_cand else 2
            return {
                "batch_id": batch_id,
                "avg_depth": float(d),
                "max_depth": d,
                "avg_score": 200.0 if is_cand else 100.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": 200 if is_cand else 100,
                        "turns": 300,
                        "final_ac": 7,
                        "xp_level": 2,
                        "end_reason": "killed",
                    }
                    for s in s_list
                ],
            }

        def close(self):
            pass

    ws2 = tmp_path / "ws_exp"
    ws2.mkdir()
    (ws2 / "policy.py").write_text(base_pol, encoding="utf-8")
    db2 = str(tmp_path / "exp.duckdb")
    init_db(db2)

    with patch(
        "lox.engine.swarm_orchestrator.PersistentEvalPool", lambda *a, **k: _OrchPool()
    ):
        orch = SwarmCampaignOrchestrator(
            db_path=db2,
            workspace_dir=str(ws2),
            frontier_dir=str(tmp_path / "fr2"),
            provider="mock",
            model="mock-model",
            num_workers=2,
            seeds_per_batch=16,
        )
        with patch(
            "lox.engine.swarm_orchestrator.SwarmPeerAgent.run_cycle",
            return_value=PeerCandidate(
                peer_id="p1",
                claimed_deficit_id="slice_early_terminations",
                claimed_label="Early termination gap",
                policy_code=cand_pol,
                skills={},
                touched_branches={"step"},
                sandbox_passed=True,
            ),
        ):
            orch.run(generations=2)

    assert len(seen_select_sets) >= 1
    assert len(seen_confirm_sets) >= 1
    assert len(seen_confirm_sets[0]) == 16
    assert all(s >= 10_000_000 for s in seen_confirm_sets[0])
    assert set(seen_select_sets[0]).isdisjoint(set(seen_confirm_sets[0]))
    assert len(set(seen_confirm_sets[0])) == 16
    con2 = connect_db(db2, read_only=True)
    try:
        assert (
            con2.execute(
                "SELECT COUNT(*) FROM experiments WHERE mode='SWARM'"
            ).fetchone()[0]
            >= 1
        )
        assert (
            con2.execute(
                "SELECT COUNT(*) FROM discoveries WHERE mode='SWARM'"
            ).fetchone()[0]
            >= 1
        )
    finally:
        con2.close()


def test_swarm_referee_promotes_explorer_over_wait_zombie_incumbent_and_preserves_ticks(
    tmp_path,
):
    from lox.author.swarm_peer import PeerCandidate
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import connect_db
    from lox.engine.swarm_referee import SwarmReferee, _clone_incumbent_eval

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    cand_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.E\n"
    )

    recorded_tick_flags: list[tuple[str, bool]] = []

    class _ZombiePool:
        def evaluate_batch(self, batch_id, policy_code, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            recorded_tick_flags.append((batch_id, bool(kwargs.get("record_ticks"))))
            if "_inc_" in batch_id:
                runs = [
                    {
                        "run_id": f"{batch_id}_r{i:05d}",
                        "seed": s,
                        "depth": 1,
                        "max_depth": 1,
                        "score": 153,
                        "turns": 4742,
                        "unique_tiles": 1,
                        "pct_wait": 99.2,
                        "end_reason": "max_steps",
                        "ticks": [{"run_id": f"{batch_id}_r{i:05d}", "step_idx": 0}],
                    }
                    for i, s in enumerate(s_list)
                ]
                return {"batch_id": batch_id, "runs": runs}
            runs = [
                {
                    "run_id": f"{batch_id}_r{i:05d}",
                    "seed": s,
                    "depth": 4 if i < 2 else (2 if i < 12 else 1),
                    "max_depth": 4 if i < 2 else (2 if i < 12 else 1),
                    "score": 158,
                    "turns": 4500,
                    "unique_tiles": 120,
                    "pct_wait": 15.0,
                    "end_reason": "stalled_zero_turn" if i < 6 else "killed by newt",
                    "ticks": [{"run_id": f"{batch_id}_r{i:05d}", "step_idx": 0}],
                }
                for i, s in enumerate(s_list)
            ]
            return {"batch_id": batch_id, "runs": runs}

    cand = PeerCandidate(
        peer_id="peer_explorer",
        claimed_deficit_id="slice_max_steps_confinement",
        claimed_label="explore",
        policy_code=cand_pol,
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    seeds = [10300 + i for i in range(64)]
    verdict = SwarmReferee().evaluate_tournament(
        generation=3,
        candidates=[cand],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_ZombiePool(),
        seeds=seeds,
    )
    assert verdict.promoted is True
    assert verdict.winner_id == "peer_explorer"
    assert any("_inc_" in bid and rec for bid, rec in recorded_tick_flags)

    # _clone_incumbent_eval preserves ticks and rewrites their run_ids
    cloned = _clone_incumbent_eval(verdict.incumbent_batch_eval, generation=3)
    assert len(cloned["runs"]) == 64
    assert cloned["runs"][0]["ticks"][0]["run_id"] == cloned["runs"][0]["run_id"]

    # AuthorToolRegistry.read_policy / read_skill resolves MAP-Elites cells from archive_policies
    db_p = str(tmp_path / "map_elites.duckdb")
    init_db(db_p)
    con = connect_db(db_p, read_only=False)
    try:
        con.execute(
            """
            INSERT INTO archive_policies (
                cell_id, depth_bin, caution_bin, resource_bin, fitness,
                policy_code, skills_json, batch_id, generation, metrics_json
            ) VALUES (
                'cell_v2_d0_v1_t0', 0, 1, 0, 1.5,
                'class Policy:\n    pass\n', '{"exploration": "def go(): pass"}',
                'batch_swarm_g3_peer_1_1794946810', 3, '{"avg_depth": 1.2, "max_depth": 4}'
            )
            """
        )
    finally:
        con.close()

    reg = AuthorToolRegistry(db_path=db_p)
    p_text, _ = reg.read_policy("cell_v2_d0_v1_t0")
    assert "MAP_ELITES" in p_text
    assert "class Policy:" in p_text
    s_text, _ = reg.read_skill("exploration", policy_id="cell_v2_d0_v1_t0")
    assert "def go():" in s_text


def test_stepping_stone_hybrid_metric_and_select_record_ticks():
    """
    Verifies:
    1. Empirically calibrated Hybrid Stepping-Stone selector accepts real G54/G42/G35-class
       candidates (small-sample high-precision wins & guard-blocked high-upside breakthroughs)
       and rejects 50/50 coin-flip noise and policy_exception regressions.
    2. Split-Sample `S_select` runs with `record_ticks=True` and exposes `select_batches`,
       `confirm_challenger_eval`, and `qualifying_stepping_stone` on `TournamentVerdict`.
    """
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import (
        SwarmReferee,
        select_qualifying_stepping_stone,
    )

    def _cand(pid: str, code: str = "class Policy:\n    pass\n") -> PeerCandidate:
        return PeerCandidate(
            peer_id=pid,
            claimed_deficit_id="d1",
            claimed_label="lbl",
            policy_code=code,
            skills={},
            sandbox_passed=True,
            loc_total=2,
        )

    # 1. Coin-flip noise (32W/30L, p=0.45) & exception candidates are rejected
    noise = [
        {
            "candidate": _cand("peer_noise"),
            "policy_exceptions": 0,
            "wins": 32,
            "losses": 30,
            "iqm_delta": 0.02,
            "winsor_delta": 0.02,
            "wilcoxon_p": 0.447,
            "reject_reason": "below_sign_perm_alpha",
            "seeds_gained": [1, 2],
            "seeds_lost": [3, 4],
        },
        {
            "candidate": _cand("peer_exc"),
            "policy_exceptions": 1,
            "wins": 10,
            "losses": 0,
            "iqm_delta": 0.5,
            "winsor_delta": 0.5,
            "wilcoxon_p": 0.001,
            "reject_reason": "policy_exceptions_regression",
        },
    ]
    assert select_qualifying_stepping_stone(66, noise) is None

    # 2. Guard-blocked high-upside candidate (G42 peer_4: 14W/12L, iqm=+0.681, cvar25_tail_regression) qualifies
    guard_rescue = select_qualifying_stepping_stone(
        42,
        [
            {
                "candidate": _cand("peer_4"),
                "policy_exceptions": 0,
                "wins": 14,
                "losses": 12,
                "iqm_delta": 0.681,
                "winsor_delta": 0.681,
                "wilcoxon_p": 0.35,
                "reject_reason": "cvar25_tail_regression",
                "seeds_gained": [1, 2],
                "seeds_lost": [9],
            }
        ],
    )
    assert guard_rescue is not None
    assert guard_rescue["source_peer_id"] == "peer_4"

    # 3. Ranks lowest min_p first (G54 peer_2 7W/0L beats G35 peer_1 8W/3L)
    best = select_qualifying_stepping_stone(
        54,
        [
            {
                "candidate": _cand("peer_1"),
                "policy_exceptions": 0,
                "wins": 8,
                "losses": 3,
                "iqm_delta": 0.038,
                "winsor_delta": 0.038,
                "wilcoxon_p": 0.0841,
                "reject_reason": "below_sig_or_pareto_gate",
            },
            {
                "candidate": _cand("peer_2"),
                "policy_exceptions": 0,
                "wins": 7,
                "losses": 0,
                "iqm_delta": 0.205,
                "winsor_delta": 0.205,
                "wilcoxon_p": 0.0078,
                "reject_reason": "cut_in_select_stage_by_stack",
            },
        ],
    )
    assert best is not None
    assert best["source_peer_id"] == "peer_2"

    # 4. Split-sample S_select runs with record_ticks=True and preserves select_batches & confirm_challenger_eval
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.WAIT\n"
    )
    cand_pol = base_pol.replace("Action.WAIT", "Action.SEARCH")
    sel_flags: list[tuple[str, bool]] = []

    class _SplitPool:
        benchmark = "nethack"

        def evaluate_batch(
            self, batch_id, priority_seeds=None, record_ticks=False, **kwargs
        ):
            sel_flags.append((str(batch_id), bool(record_ticks)))
            s_list = list(priority_seeds or [])
            is_cand = "peer_cand" in str(batch_id)
            is_sel = "sel_" in str(batch_id)
            runs = []
            for idx, s in enumerate(s_list):
                # Wins 6/8 on S_select, ties on S_confirm -> rejected C* with qualifying S_select stone
                if is_cand and is_sel:
                    d = 4 if idx < 6 else 2
                elif is_cand:
                    d = 2
                else:
                    d = 2
                runs.append(
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": d * 100,
                        "turns": 100,
                        "end_reason": "killed",
                        "ticks": [
                            {
                                "turn": 1,
                                "step_idx": 1,
                                "depth": d,
                                "hp": 10,
                                "max_hp": 10,
                            }
                        ],
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    verdict = SwarmReferee().evaluate_tournament(
        generation=67,
        candidates=[_cand("peer_cand", cand_pol)],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_SplitPool(),
        seeds=list(range(100, 116)),
        select_seeds=list(range(100, 108)),
    )
    assert verdict.promoted is False
    assert sel_flags and all(flag for _, flag in sel_flags)
    assert "peer_cand" in verdict.select_batches
    assert verdict.confirm_challenger_eval is not None
    assert verdict.confirm_challenger_eval[0] == "peer_cand"


def test_v3_promotion_gate_ranker_triggered_extension_and_paired_guards():
    """
    Verifies Component 2:
    1. S_select admission floor blocks 1W/0L near-no-op (winsor_delta=+0.0002) and ranker puts 5W/2L above 1W/0L.
    2. Triggered Sparse-Surgical Stage-3b Extension (+64 unseen seeds, max N=128) rescues a clean 3W/0L surgical fix.
    3. OBF/Triggered 128-seed pooled winner_avg_depth & incumbent_avg_depth stay synchronized (no stale 64-seed stats).
    4. Paired McNemar horizon_waste guard tolerates 1 noisy extra max_steps while rejecting systematic regressions.
    5. select_qualifying_stepping_stone prioritizes clean confirm_challenger near-misses and excludes proactive_2.
    """
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import (
        SwarmReferee,
        _meets_select_admission_floor,
        _passes_safety_invariants,
        _selection_rank_key,
        select_qualifying_stepping_stone,
    )

    # 1. Admission floor & ranker fix
    pr_noop = {
        "wins": 1,
        "losses": 0,
        "mean_delta": 0.0002,
        "winsor_delta": 0.0002,
        "sign_perm_p": 0.5,
        "lcb_score": 0.0001,
    }
    pr_real = {
        "wins": 5,
        "losses": 2,
        "mean_delta": 0.25,
        "winsor_delta": 0.22,
        "sign_perm_p": 0.18,
        "lcb_score": -0.05,
    }
    assert _meets_select_admission_floor(pr_noop, has_base=True) is False
    assert _meets_select_admission_floor(pr_real, has_base=True) is True
    assert _selection_rank_key({"ast_nodes": 50}, pr_real) > _selection_rank_key(
        {"ast_nodes": 50}, pr_noop
    )

    # 2. Paired McNemar horizon-waste tolerates 1-seed noise, rejects 6-seed worsening
    ok_noise, _ = _passes_safety_invariants(
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "horizon_waste_rate": 0.10,
            "cvar25_prog": 2.0,
            "run_count": 32,
        },
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "horizon_waste_rate": 0.06,
            "cvar25_prog": 2.0,
            "run_count": 32,
        },
        has_base=True,
        pair_stats={
            "hw_worsened": 1,
            "hw_improved": 0,
            "hw_mcnemar_p": 0.5,
            "paired_cvar25_delta": 0.0,
            "paired_cvar25_se": 0.1,
        },
        stage="select",
    )
    assert ok_noise is True
    bad_hw, bad_reason = _passes_safety_invariants(
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "horizon_waste_rate": 0.25,
            "cvar25_prog": 2.0,
            "run_count": 32,
        },
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "horizon_waste_rate": 0.05,
            "cvar25_prog": 2.0,
            "run_count": 32,
        },
        has_base=True,
        pair_stats={
            "hw_worsened": 6,
            "hw_improved": 0,
            "hw_mcnemar_p": 0.0156,
            "paired_cvar25_delta": 0.0,
            "paired_cvar25_se": 0.1,
        },
        stage="select",
    )
    assert bad_hw is False and bad_reason == "horizon_waste_regression"

    # 3. Triggered Sparse-Surgical Stage-3b Extension (max N=128) rescues clean 3W/0L -> 7W/0L across 128 seeds
    base_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.WAIT\n"
    cand_pol = base_pol.replace("Action.WAIT", "Action.SEARCH")

    class _SparseSurgicalPool:
        benchmark = "nethack"

        def __init__(self):
            self.seen_obf = False

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            is_cand = "proactive_1" in str(batch_id)
            is_obf = "obf_" in str(batch_id)
            if is_obf:
                self.seen_obf = True
            runs = []
            for idx, sd in enumerate(s_list):
                if is_cand:
                    # S_select (32): 3W/0L; S_confirm Look 1 (64): 3W/0L (p=0.125 -> triggers sparse extension!);
                    # Stage-3b (+64): 4W/0L -> pooled 7W/0L (p=0.0078 < 0.032 -> PROMOTED!)
                    win_k = 4 if is_obf else 3
                    d = 5 if idx < win_k else 3
                    sc = 1100 if idx < win_k else 500
                else:
                    d, sc = 3, 500
                runs.append(
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 1000,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

    sp_pool = _SparseSurgicalPool()
    c_p1 = PeerCandidate(
        peer_id="proactive_1",
        claimed_deficit_id="proactive_expert_capability",
        claimed_label="surgical_capability",
        policy_code=cand_pol,
        skills={},
        sandbox_passed=True,
        loc_total=4,
        track="proactive_capability",
    )
    v_ext = SwarmReferee().evaluate_tournament(
        generation=80,
        candidates=[c_p1],
        base_policy=base_pol,
        base_skills={},
        eval_pool=sp_pool,
        select_seeds=list(range(1000, 1032)),
        seeds=list(range(2000, 2064)),
    )
    assert sp_pool.seen_obf is True
    assert v_ext.promoted is True, v_ext.reason
    assert len(v_ext.winner_batch_eval["runs"]) == 128
    # Winner avg_depth synchronized with 128-seed pooled eval: (7*5 + 121*3)/128 = 3.109375
    assert abs(v_ext.winner_avg_depth - (7 * 5 + 121 * 3) / 128.0) < 1e-3

    # 4. select_qualifying_stepping_stone prioritizes clean confirm_challenger & excludes proactive_2
    c_p2 = PeerCandidate(
        peer_id="proactive_2",
        claimed_deficit_id="pb",
        claimed_label="pb",
        policy_code=cand_pol,
        skills={},
        track="proactive_branch",
    )
    c_conf = PeerCandidate(
        peer_id="reactive_1",
        claimed_deficit_id="d1",
        claimed_label="d1",
        policy_code=cand_pol,
        skills={},
        track="reactive",
    )
    c_guard = PeerCandidate(
        peer_id="reactive_2",
        claimed_deficit_id="d2",
        claimed_label="d2",
        policy_code=cand_pol,
        skills={},
        track="reactive",
    )
    picked = select_qualifying_stepping_stone(
        81,
        [
            {
                "candidate": c_p2,
                "policy_exceptions": 0,
                "wins": 10,
                "losses": 0,
                "winsor_delta": 0.5,
                "iqm_delta": 0.5,
                "wilcoxon_p": 0.001,
                "sign_perm_p": 0.001,
                "reject_reason": "below_sign_perm_alpha",
                "source_run_kind": "confirm_challenger",
            },
            {
                "candidate": c_guard,
                "policy_exceptions": 0,
                "wins": 10,
                "losses": 2,
                "winsor_delta": 0.4,
                "iqm_delta": 0.4,
                "wilcoxon_p": 0.02,
                "sign_perm_p": 0.02,
                "reject_reason": "cvar25_tail_regression",
                "source_run_kind": "select",
            },
            {
                "candidate": c_conf,
                "policy_exceptions": 0,
                "wins": 6,
                "losses": 2,
                "winsor_delta": 0.18,
                "iqm_delta": 0.18,
                "wilcoxon_p": 0.12,
                "sign_perm_p": 0.09,
                "reject_reason": "below_sign_perm_alpha",
                "source_run_kind": "confirm_challenger",
            },
        ],
    )
    assert picked is not None
    assert picked["source_peer_id"] == "reactive_1"
