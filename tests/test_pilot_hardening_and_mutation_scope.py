"""
Tests for Pilot Hardening (Fixes 1 & 2) and Mutation Scope Guards (Fix 3).
"""

from __future__ import annotations

from lox.author.agent import AuthorAgent
from lox.author.tools import AuthorToolRegistry
from lox.core.contracts import CandidatePackage
from lox.engine.orchestrator import CampaignOrchestrator
from lox.runner.pool import PersistentEvalPool


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
