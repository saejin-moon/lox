"""Domain-scoped tests for NetHack adapters, scenarios, and contracts."""

from __future__ import annotations

from lox.core.contracts import EpisodeRun
from lox.engine.hypothesis import (
    progression_metric,
    should_stop_early_for_futility,
)
from lox.env import (
    NetHackHarness,
    get_adapter,
)
from lox.env.actions import Action
from lox.env.base import BenchmarkAdapter, EnvHarness
from lox.env.benchmarks.nethack_scenarios import (
    ALL_SCENARIOS,
    PROMPT_TORTURE_SUITES,
    SCENARIO_ARCHETYPES,
    NetHackScenarioHarness,
    run_scenario_gate,
)
from lox.env.nethack.adapter import NetHackAdapter
from lox.lang.compiler import compile_policy_package
from lox.lang.contracts import GoalIntent, SafetyReport, validate_contract_instance
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    smart_dedent,
)


def test_benchmark_adapter_protocol_and_nethack():
    adapter = get_adapter("nethack")
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "nethack"

    harness = adapter.create_harness(max_steps=50)
    assert isinstance(harness, EnvHarness)
    assert isinstance(harness, NetHackHarness)

    state = harness.reset(seed=123)
    assert state is not None
    next_s, done, info = harness.step(Action.SEARCH)
    assert next_s is not None
    assert harness.step_count == 1

    docs = adapter.get_api_docs()
    assert isinstance(docs, str) and len(docs) > 100


def test_nethack_adapter_methods() -> None:
    """NetHackAdapter extract_milestones_and_stalls & search_knowledge_base."""
    adapter = NetHackAdapter()

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

    kb_res = adapter.search_knowledge_base("door")
    assert isinstance(kb_res, str)
    assert len(kb_res) > 0


def test_nethack_scenario_contracts_and_ast_fixes():
    ep = EpisodeRun(
        {"seed": 1, "hp": 12, "ac": 5, "milestones": None, "summary_digest": None}
    )
    assert ep["final_hp"] == 12 and ep["final_ac"] == 5
    assert isinstance(progression_metric(ep), float)
    raw_bad = {
        "seed": 1,
        "is_shard": True,
        "shard_id": "s1",
        "milestones": None,
        "score": None,
        "final_ac": None,
    }
    assert isinstance(progression_metric(raw_bad), float)
    assert should_stop_early_for_futility([raw_bad], [raw_bad], min_pairs=5) is False

    # Scenario harness reset/replay_bytes + run_scenario_gate
    sh = NetHackScenarioHarness(scenario="prompt_torture_fountain", max_steps=15)
    try:
        sh.reset(seed=11)
        sh.step(Action.ESC)
        s1 = sh.replay_bytes(seed=11, recorded_bytes=sh.action_bytes)
        assert s1 is not None
        assert len(sh.mem.pos_history) >= 1
    finally:
        sh.close()

    sg = run_scenario_gate(
        "class Policy:\n    def step(self, s):\n        return Action.ESC if (s.in_yn_prompt or s.in_text_prompt) else Action.WAIT\n",
        scenarios=["prompt_torture_fountain"],
        max_steps=15,
    )
    assert sg["total_count"] == 1
    assert sg["scenario_results"][0]["end_reason"] != "running"

    # AST validator fixes
    assert "\t" not in smart_dedent("\tdef f():\n\t\treturn 1\n")
    hyg = audit_policy_ast_hygiene(
        "X: int = 1\nX: int = 2\nasync def foo(): pass\nasync def foo(): pass\n"
    )
    assert len(hyg) >= 2


def test_phase5_scenarios_and_contracts():
    assert len(SCENARIO_ARCHETYPES) == 7
    assert len(PROMPT_TORTURE_SUITES) == 4
    assert len(ALL_SCENARIOS) == 11

    adapter = get_adapter("nethack_scenarios")
    assert adapter.name == "nethack_scenarios"

    # Prompt-aware policy resolves text/yn/getobj/getdir/menu prompts cleanly with 0 aborts/exceptions
    clean_pol = """class Policy:
    def run(self, state):
        while True:
            if (
                state.in_text_prompt
                or state.in_yn_prompt
                or state.in_getobj_prompt
                or state.in_getdir_prompt
                or state.in_menu
            ):
                state = yield Action.ESC
            else:
                state = yield Action.S
"""
    res = run_scenario_gate(clean_pol, max_steps=25)
    assert res["exceptions"] == 0
    assert res["prompt_livelocks"] == 0
    assert res["passed_count"] == len(ALL_SCENARIOS)

    # Contracts check
    gi = GoalIntent(goal_id="descend", priority=1.0, target_pos=(4, 5))
    sr = SafetyReport(safe_to_continue=True, risk_score=0.1)
    assert gi.goal_id == "descend" and sr.safe_to_continue
    assert validate_contract_instance(object()) == []

    # Policy compiling & importing contracts inside LOX sandbox
    sandbox_pol = """from lox.lang.contracts import GoalIntent
class Policy:
    def __init__(self):
        self.intent = GoalIntent(goal_id="explore", priority=1.0)
    def step(self, state):
        return Action.WAIT
"""
    compiled = compile_policy_package(sandbox_pol)
    runner = compiled.create_runner()
    assert runner is not None
