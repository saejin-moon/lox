"""
Tests for the 3 Modification Modes (`CREATE`, `WRITE`, `REFACTOR`),
Conditional Episode Trigger Filtering (discarding untriggered episodes),
Unreachable Function Circuit Breaker, Parameter-Free Hypothesis Evaluator
(Confidence Sequences + Wilcoxon Signed-Rank + Early Futility Stopping),
and End-to-End Mock Campaign Orchestration.
"""

from __future__ import annotations

from lox.db.schema import connect_db
from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.engine.orchestrator import CampaignOrchestrator
from lox.engine.scheduler import ModeScheduler
from lox.runner.pool import PersistentEvalPool


def test_conditional_episode_trigger_filtering_and_unreachable_circuit_breaker():
    policy_code = """
class Policy:
    def rare_branch(self, state):
        yield Action.SEARCH

    def never_called(self, state):
        yield Action.WAIT

    def run(self, state):
        while True:
            # Trigger `rare_branch` only when starting x coordinate is even
            if state.x % 2 == 0:
                yield from self.rare_branch(state)
            else:
                yield Action.WAIT
            state = self.state
"""
    pool = PersistentEvalPool(workers=2, max_steps=15)
    try:
        # 1. Target `rare_branch`: only episodes where `rare_branch` is called should be recorded!
        res = pool.evaluate_batch(
            batch_id="test_filter",
            policy_code=policy_code,
            macros_code="",
            target_episodes=6,
            target_functions=["rare_branch"],
            seed_start=500,
            record_ticks=False,
            show_progress=False,
        )
        assert res["episodes_recorded"] == 6
        assert len(res["runs"]) == 6
        # Every single recorded episode MUST have triggered `rare_branch` at least once
        for r in res["runs"]:
            assert r["functions_triggered"].get("rare_branch", 0) >= 1

        # 2. Target `never_called`: should trigger Unreachable Function Circuit Breaker
        unreach = pool.evaluate_batch(
            batch_id="test_unreach",
            policy_code=policy_code,
            macros_code="",
            target_episodes=10,
            target_functions=["never_called"],
            seed_start=600,
            record_ticks=False,
            show_progress=False,
            unreachable_abort_after=8,
        )
        assert unreach["episodes_recorded"] == 0
        assert unreach["unreachable_abort"] is True
        assert unreach["episodes_discarded"] >= 8
    finally:
        pool.close()


def test_parameter_free_hypothesis_testing_and_early_stopping():
    # Clear improvement on paired seeds
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

    # Regression on paired seeds should be rejected and trigger early futility stop
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

    # Neutral/modular refactoring (mostly ties, zero mean delta):
    # Fails under FUNC_MOD_ADD (requires p < 0.05), but passes under STRUCTURAL_REFACTOR
    cand_runs_refactor_neutral = [dict(base_runs[i]) for i in range(len(base_runs))]
    # Small neutral fluctuation on 2 seeds
    cand_runs_refactor_neutral[0]["score"] += 10
    cand_runs_refactor_neutral[1]["score"] -= 10
    verdict_func_mod = evaluate_paired_hypothesis(
        cand_runs_refactor_neutral, base_runs, mode="FUNC_MOD_ADD"
    )
    assert (
        verdict_func_mod["passed"] is False
    )  # p is ~1.0, not statistically significant

    verdict_refactor = evaluate_paired_hypothesis(
        cand_runs_refactor_neutral, base_runs, mode="STRUCTURAL_REFACTOR"
    )
    assert verdict_refactor["passed"] is True
    assert "Refactor non-regression verified" in verdict_refactor["reason"]

    # Truly regressive refactoring fails even under STRUCTURAL_REFACTOR
    verdict_bad_refactor = evaluate_paired_hypothesis(
        cand_runs_bad, base_runs, mode="STRUCTURAL_REFACTOR"
    )
    assert verdict_bad_refactor["passed"] is False


def test_mode_scheduler_refactor_cadence_and_line_count():
    sched = ModeScheduler()
    # Gen 0 bootstraps
    assert sched.select_mode(0, has_policy=False) == "CREATE"

    # Small policy remains WRITE
    assert sched.select_mode(1, has_policy=True, policy_line_count=200) == "WRITE"
    assert sched.select_mode(2, has_policy=True, policy_line_count=350) == "WRITE"

    # Line count >= 450 triggers REFACTOR when gens_since_refactor >= 4
    sched.gens_since_refactor = 4
    assert sched.select_mode(4, has_policy=True, policy_line_count=460) == "REFACTOR"

    # Cooldown prevents immediate consecutive refactor
    assert sched.select_mode(5, has_policy=True, policy_line_count=460) == "WRITE"


def test_mode_scheduler_and_end_to_end_mock_campaign(tmp_path):
    sched = ModeScheduler()
    assert sched.select_mode(0, has_policy=False) == "CREATE"
    assert sched.select_mode(1, has_policy=True) == "WRITE"

    # End-to-end 3-generation mock campaign (`CREATE` -> `WRITE` -> `WRITE`)
    db_path = str(tmp_path / "campaign.duckdb")
    policy_path = str(tmp_path / "policy.py")
    macros_path = str(tmp_path / "macros.py")

    orch = CampaignOrchestrator(
        provider="mock",
        model="mock-model",
        db_path=db_path,
        policy_path=policy_path,
        macros_path=macros_path,
        workers=2,
        max_steps=20,
        full_write_episodes=6,
        func_mod_episodes_per_fn=6,
    )
    res = orch.run_campaign(max_generations=3, target_depth=99.0, seed_base=2000)
    assert res["active_policy_code"] != ""

    con = connect_db(db_path, read_only=True)
    try:
        n_batches = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        n_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_exps = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        assert n_batches >= 2
        assert n_runs >= 6
        assert n_exps == 3
    finally:
        con.close()
