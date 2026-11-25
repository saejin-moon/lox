"""Domain-scoped tests for paired hypothesis testing and exact batch dispatch."""

from __future__ import annotations

from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.runner.pool import PersistentEvalPool


def test_parameter_free_hypothesis_testing_and_early_stopping():
    base_runs = [
        {
            "seed": i,
            "max_depth": 2,
            "final_ac": 5,
            "xp_level": 2,
            "score": 100,
            "turns": 200,
        }
        for i in range(1, 31)
    ]
    cand_runs_good = [
        {
            "seed": i,
            "max_depth": 4 if i <= 22 else 2,
            "final_ac": 1 if i <= 22 else 5,
            "xp_level": 4 if i <= 22 else 2,
            "score": 400 if i <= 22 else 100,
            "turns": 350 if i <= 22 else 200,
        }
        for i in range(1, 31)
    ]
    verdict_good = evaluate_paired_hypothesis(cand_runs_good, base_runs)
    assert verdict_good["passed"] is True
    assert verdict_good["mean_delta"] > 0.5
    assert verdict_good["ci_lower"] > 0.0
    assert verdict_good["wilcoxon_p"] < 0.05
    assert verdict_good["wins"] == 22

    cand_runs_bad = [
        {
            "seed": i,
            "max_depth": 1,
            "final_ac": 8,
            "xp_level": 1,
            "score": 20,
            "turns": 50,
        }
        for i in range(1, 31)
    ]
    verdict_bad = evaluate_paired_hypothesis(cand_runs_bad, base_runs)
    assert verdict_bad["passed"] is False
    assert verdict_bad["ci_upper"] < 0.0
    base_by_seed = {r["seed"]: r for r in base_runs}
    assert (
        should_stop_early_for_futility(cand_runs_bad, base_by_seed, min_pairs=20)
        is True
    )

    cand_runs_refactor_neutral = [dict(base_runs[i]) for i in range(len(base_runs))]
    cand_runs_refactor_neutral[0]["score"] += 10
    cand_runs_refactor_neutral[1]["score"] -= 10
    verdict_func_mod = evaluate_paired_hypothesis(
        cand_runs_refactor_neutral, base_runs, mode="FUNC_MOD_ADD"
    )
    assert verdict_func_mod["passed"] is False

    verdict_refactor = evaluate_paired_hypothesis(
        cand_runs_refactor_neutral, base_runs, mode="STRUCTURAL_REFACTOR"
    )
    assert verdict_refactor["passed"] is True
    assert "Refactor non-regression verified" in verdict_refactor["reason"]

    verdict_bad_refactor = evaluate_paired_hypothesis(
        cand_runs_bad, base_runs, mode="STRUCTURAL_REFACTOR"
    )
    assert verdict_bad_refactor["passed"] is False


def test_two_stage_pilot_sign_test():
    """Verify Exact Sign Test filters out noisy 10W/9L coin flips while passing decisive 12W/5L."""
    import scipy.stats as sp_stats

    p_coin = float(sp_stats.binom.sf(10 - 1, 19, 0.5))
    assert p_coin > 0.15

    p_decisive = float(sp_stats.binom.sf(12 - 1, 17, 0.5))
    assert p_decisive <= 0.15


def test_exact_batch_capacity_dispatch():
    """Verify PersistentEvalPool dispatches exact target_episodes over priority_seeds."""
    pool = PersistentEvalPool(workers=2, max_steps=15)
    try:
        cand_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
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
