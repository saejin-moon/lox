"""Domain-scoped tests for LOX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.tools import AuthorToolRegistry
from lox.core.contracts import CandidatePackage
from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.engine.orchestrator import CampaignOrchestrator
from lox.runner.pool import PersistentEvalPool


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


def test_two_stage_pilot_sign_test():
    """Verify that Exact Sign Test filters out noisy 10W/9L coin flips while passing decisive 12W/5L."""
    import scipy.stats as sp_stats

    # Coin flip: 10 wins, 9 losses (n=19) -> p ~ 0.50 -> must fail
    p_coin = float(sp_stats.binom.sf(10 - 1, 19, 0.5))
    assert p_coin > 0.15

    # Decisive: 12 wins, 5 losses (n=17) -> p ~ 0.07 -> passes alpha=0.15 hurdle
    p_decisive = float(sp_stats.binom.sf(12 - 1, 17, 0.5))
    assert p_decisive <= 0.15


def test_gen0_full_write_pilot_verification(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.runner.pool import PersistentEvalPool

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=15)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "gen0.duckdb"))
        # In Gen 0 FULL_WRITE, baseline_runs is empty
        tools.configure_generation(
            pool=pool,
            baseline_runs=[],
            max_pilot_runs=3,
            pilot_episodes=2,
        )

        valid_starter_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        msg, status = tools.dry_run_policy(valid_starter_policy)
        assert status.startswith("PASSED")
        assert "Gen 0 Baseline:" in status
        assert "GEN 0 BASELINE VERIFIED" in msg
        assert tools.last_passed_code == valid_starter_policy
        assert tools.last_pilot_improved is True
    finally:
        pool.close()


def test_pilot_fallback_best_does_not_abort_on_non_negative_delta(
    tmp_path, monkeypatch
):
    from lox.runner.pool import PersistentEvalPool

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "fallback.duckdb"),
            pilot_fallback="best",
        )
        neutral_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""

        def mock_call_llm(messages, allow_tools=True):
            return {
                "role": "assistant",
                "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'test'\nmechanism: 'test'\n```\n```python\n{neutral_code}```",
                "tool_calls": [],
            }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)
        # Baseline with equal performance so delta is non-negative (>= 0.0)
        baseline_runs = [
            {"seed": 100, "max_depth": 1, "score": 0},
            {"seed": 101, "max_depth": 1, "score": 0},
        ]
        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy=neutral_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=1,
            pilot_episodes=2,
        )
        assert proposal.aborted is False
        assert proposal.full_policy_code != ""
    finally:
        pool.close()


def test_pilot_stops_immediately_after_pass(tmp_path):
    """Verify that dry_run_policy returns cached pass without burning extra pilot runs."""
    from lox.author.tools import AuthorToolRegistry
    from lox.lang.validator import is_ast_equivalent
    from lox.runner.pool import PersistentEvalPool

    c1 = "class Policy:\n    def run(self, state):\n        while True:\n            state = yield Action.WAIT\n"
    c2 = "class Policy:\n    def run(self, state):\n        # Added comment\n        while True:\n            state = yield Action.WAIT\n"
    assert is_ast_equivalent(c1, c2)

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "stop_after_pass.duckdb"))
        tools.configure_generation(
            pool=pool,
            baseline_runs=[],
            max_pilot_runs=3,
            pilot_episodes=2,
        )

        msg, status = tools.dry_run_policy(c1)
        assert status.startswith("PASSED")
        assert tools.last_pilot_improved is True
        assert tools.pilot_runs_used == 1

        # Calling again with AST-equivalent code must return cached pass without burning worker runs
        msg2, status2 = tools.dry_run_policy(c2)
        assert status2.startswith("PASSED (cached")
        assert tools.pilot_runs_used == 1
    finally:
        pool.close()


def test_pilot_revisions_defaults():
    """Verify that default pilot revisions is capped at 4 across CLI and Orchestrator."""
    import inspect

    sig = inspect.signature(CampaignOrchestrator.__init__)
    assert sig.parameters["pilot_revisions"].default == 4


def test_pilot_fallback_aborts_on_negative_delta(tmp_path, monkeypatch):
    """
    Verify Fix 2: When pilot_fallback="best", if all pilot revisions resulted in
    strictly negative delta (Δ < 0.0), rollout is aborted immediately to avoid
    evaluating known regressions.
    """
    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "fallback_neg.duckdb"),
            pilot_fallback="best",
        )
        base_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        # Candidate code that will be returned
        cand_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""

        # Mock LLM calling dry_run_policy which fails with negative delta
        turn = 0

        def mock_call_llm(messages, allow_tools=True):
            nonlocal turn
            turn += 1
            if turn == 1:
                return {
                    "role": "assistant",
                    "content": "I will test my change.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "dry_run_policy",
                                "arguments": {"code": cand_code},
                            },
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": f"```yaml\nmode: WRITE\ntarget_functions: [run]\nbinding_constraint: 'none'\nverified_c_rule: 'none'\nmechanism: 'test'\n```\n```python\n{cand_code}```",
                "tool_calls": [],
            }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)

        # Baseline runs where baseline achieved depth 5
        baseline_runs = [
            {"seed": 100, "max_depth": 5, "score": 500},
            {"seed": 101, "max_depth": 5, "score": 500},
        ]

        # Force dry_run_policy to record a negative delta candidate
        orig_dry_run = agent.tools.dry_run_policy

        def mock_dry_run(*args, **kwargs):
            out, status = orig_dry_run(*args, **kwargs)
            agent.tools.best_pilot_pkg = CandidatePackage(
                code=cand_code,
                pilot_delta=-0.25,
                pilot_summary="10W/20L/0T, Δ=-0.25",
                improved=False,
            )
            agent.tools.last_pilot_improved = False
            agent.tools.last_pilot_summary = "10W/20L/0T, Δ=-0.25"
            return out, status

        monkeypatch.setattr(agent.tools, "dry_run_policy", mock_dry_run)

        proposal = agent.synthesize(
            mode="WRITE",
            generation=2,
            current_policy=base_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=1,
            pilot_episodes=2,
        )

        assert proposal.aborted is True
        assert "PILOT_GATE_ABORT" in proposal.abort_reason
        assert "regressed" in proposal.abort_reason
        assert proposal.pilot_best_delta == -0.25
    finally:
        pool.close()


def test_pilot_fallback_proceeds_on_non_negative_delta(tmp_path, monkeypatch):
    """
    Verify Fix 2: When pilot_fallback="best", if candidate has non-negative delta
    (Δ >= 0.0) despite not passing the strict sign gate, rollout proceeds.
    """
    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "fallback_pos.duckdb"),
            pilot_fallback="best",
        )
        base_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        cand_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""

        turn = 0

        def mock_call_llm(messages, allow_tools=True):
            nonlocal turn
            turn += 1
            if turn == 1:
                return {
                    "role": "assistant",
                    "content": "I will test my change.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "dry_run_policy",
                                "arguments": {"code": cand_code},
                            },
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": f"```yaml\nmode: WRITE\ntarget_functions: [run]\nbinding_constraint: 'none'\nverified_c_rule: 'none'\nmechanism: 'test'\n```\n```python\n{cand_code}```",
                "tool_calls": [],
            }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)

        baseline_runs = [
            {"seed": 100, "max_depth": 5, "score": 500},
            {"seed": 101, "max_depth": 5, "score": 500},
        ]

        orig_dry_run = agent.tools.dry_run_policy

        def mock_dry_run(*args, **kwargs):
            out, status = orig_dry_run(*args, **kwargs)
            agent.tools.best_pilot_pkg = CandidatePackage(
                code=cand_code,
                pilot_delta=0.05,
                pilot_summary="15W/12L/3T, Δ=+0.05",
                improved=False,
            )
            agent.tools.last_pilot_improved = False
            agent.tools.last_pilot_summary = "15W/12L/3T, Δ=+0.05"
            return out, status

        monkeypatch.setattr(agent.tools, "dry_run_policy", mock_dry_run)

        proposal = agent.synthesize(
            mode="WRITE",
            generation=2,
            current_policy=base_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=1,
            pilot_episodes=2,
        )

        assert proposal.aborted is False
        assert proposal.full_policy_code != ""
    finally:
        pool.close()


def test_mutation_scope_rejection_in_write_mode(tmp_path):
    """
    Verify Fix 3: In WRITE mode, modifying/adding > 2 functions triggers MUTATION_SCOPE_ERROR.
    Modifying <= 2 functions is allowed.
    """
    tools = AuthorToolRegistry(db_path=str(tmp_path / "scope.duckdb"))
    base_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
    def fn_a(self):
        return 1
    def fn_b(self):
        return 2
    def fn_c(self):
        return 3
"""
    tools.current_policy_code = base_policy
    tools.mode = "WRITE"

    # Patch modifying 3 functions (fn_a, fn_b, fn_c)
    patch_3 = """class Policy:
    def fn_a(self):
        return 10
    def fn_b(self):
        return 20
    def fn_c(self):
        return 30
"""
    diag, status = tools.dry_run_policy(patch_3, smoke_only=True)
    assert status == "MUTATION_SCOPE_ERROR"
    assert "Mutation Scope Error" in diag
    assert "modified/added 3 functions" in diag

    # Patch modifying 2 functions (fn_a and run)
    patch_2_wired = """class Policy:
    def fn_a(self):
        return 10
    def run(self, state):
        while True:
            self.fn_a()
            state = yield Action.WAIT
"""
    diag2, status2 = tools.dry_run_policy(patch_2_wired, smoke_only=True)
    assert status2 != "MUTATION_SCOPE_ERROR"
    assert status2.startswith("VALID") or status2.startswith("PASSED")


def test_mutation_scope_permitted_in_refactor_and_create(tmp_path):
    """
    Verify Fix 3: In REFACTOR or CREATE modes, multi-function restructuring is permitted.
    """
    tools = AuthorToolRegistry(db_path=str(tmp_path / "refactor_scope.duckdb"))
    base_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
    def fn_a(self):
        return 1
    def fn_b(self):
        return 2
    def fn_c(self):
        return 3
"""
    tools.current_policy_code = base_policy

    # 1. In REFACTOR mode: modifying 3 functions is allowed
    tools.mode = "REFACTOR"
    patch_3_wired = """class Policy:
    def fn_a(self):
        return 10
    def fn_b(self):
        return 20
    def run(self, state):
        while True:
            self.fn_a()
            self.fn_b()
            state = yield Action.WAIT
    def fn_c(self):
        return 30
"""
    diag, status = tools.dry_run_policy(patch_3_wired, smoke_only=True)
    assert status != "MUTATION_SCOPE_ERROR"

    # 2. In CREATE mode: writing full class with many functions is allowed
    tools.mode = "CREATE"
    tools.current_policy_code = ""
    gen0_code = """class Policy:
    def __init__(self):
        pass
    def fn_1(self):
        return 1
    def fn_2(self):
        return 2
    def fn_3(self):
        return 3
    def run(self, state):
        while True:
            self.fn_1()
            self.fn_2()
            self.fn_3()
            state = yield Action.WAIT
"""
    diag_create, status_create = tools.dry_run_policy(gen0_code, smoke_only=True)
    assert status_create != "MUTATION_SCOPE_ERROR"
    assert status_create.startswith("VALID") or status_create.startswith("PASSED")


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
