"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.core.contracts import EpisodeRun
from lox.engine.hypothesis import (
    progression_metric,
    should_stop_early_for_futility,
)
from lox.engine.scheduler import ModeScheduler
from lox.env import (
    CraftaxAction,
    CraftaxHarness,
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
from lox.lang.bt import (
    Blackboard,
)
from lox.lang.compiler import compile_policy_package
from lox.lang.contracts import GoalIntent, SafetyReport, validate_contract_instance
from lox.lang.strategy import (
    StrategyFrame,
    StrategyStack,
    preempt_if,
    until,
    with_cleanup,
)
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    check_macro_contracts,
    smart_dedent,
    three_way_ast_merge,
)
from lox.runner.pool import PersistentEvalPool
from scripts.calibrate_pilot import parse_pilot_delta


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


def test_craftax_benchmark_adapter_and_deterministic_replay():
    adapter = get_adapter("craftax")
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "craftax"

    harness = adapter.create_harness(max_steps=100)
    assert isinstance(harness, EnvHarness)
    assert isinstance(harness, CraftaxHarness)

    state = harness.reset(seed=777)
    assert state.y == 24 and state.x == 24
    assert state.health == 9

    # Step: move right, then action DO
    s1, _, _ = harness.step(CraftaxAction.RIGHT)
    assert s1.x == 25
    s2, _, _ = harness.step(CraftaxAction.DO)

    # Replay bytes and verify bit-level identical state
    replayed = harness.replay_bytes(
        seed=777, recorded_bytes=bytes(harness.action_bytes)
    )
    assert replayed.y == s2.y and replayed.x == s2.x
    assert replayed.step_count == 2
    assert replayed.health == s2.health

    # Check knowledge base query
    kb_res = adapter.search_knowledge_base("wood pickaxe")
    assert "Wood" in kb_res or "Pickaxe" in kb_res


def test_nethack_adapter_methods() -> None:
    """1. NetHackAdapter extract_milestones_and_stalls & search_knowledge_base."""
    adapter = NetHackAdapter()

    # extract_milestones_and_stalls
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

    # search_knowledge_base
    kb_res = adapter.search_knowledge_base("door")
    assert isinstance(kb_res, str)
    assert len(kb_res) > 0


def test_craftax_benchmark_and_achievements() -> None:
    """2. Real Craftax benchmark with >= 4 achievements across worker pool."""
    # Test directly on harness first
    harness = CraftaxHarness(max_steps=50)
    st = harness.reset(seed=42)
    st.inventory["wood"] = 3
    # Step 1: move East to (24, 25) facing East
    st, _, _ = harness.step(CraftaxAction.RIGHT)
    # Step 2: mine wood at (24, 26)
    st, _, _ = harness.step(CraftaxAction.DO)
    assert "collect_wood" in st.achievements
    # Step 3: place table at (24, 26)
    st, _, _ = harness.step(CraftaxAction.PLACE_TABLE)
    assert "place_table" in st.achievements
    # Step 4: craft wood pickaxe near table
    st, _, _ = harness.step(CraftaxAction.MAKE_WOOD_PICKAXE)
    assert "make_wood_pickaxe" in st.achievements
    # Step 5: move back West to (24, 24)
    st, _, _ = harness.step(CraftaxAction.LEFT)
    # Step 6: move North to (23, 24) facing North
    st, _, _ = harness.step(CraftaxAction.UP)
    # Step 7: mine stone at (22, 24) with wood pickaxe
    st, _, _ = harness.step(CraftaxAction.DO)
    assert "collect_stone" in st.achievements
    # Step 8: sleep
    st, _, _ = harness.step(CraftaxAction.SLEEP)
    assert "wake_up" in st.achievements
    assert len(st.achievements) >= 5

    # Now test through PersistentEvalPool with workers=2 on benchmark="craftax"
    pool = PersistentEvalPool(workers=2, max_steps=60, benchmark="craftax")
    test_policy = """
class Policy:
    def __init__(self):
        self.step_idx = 0

    def run(self, state):
        self.step_idx += 1
        if self.step_idx == 1:
            state.inventory["wood"] = 3
            return 2  # RIGHT (move to 24, 25 facing East)
        elif self.step_idx == 2:
            return 5  # DO (chop tree at 24, 26 -> collect_wood)
        elif self.step_idx == 3:
            return 8  # PLACE_TABLE (place table at 24, 26 -> place_table)
        elif self.step_idx == 4:
            return 11 # MAKE_WOOD_PICKAXE (craft pickaxe -> make_wood_pickaxe)
        elif self.step_idx == 5:
            return 1  # LEFT (move to 24, 24)
        elif self.step_idx == 6:
            return 3  # UP (move to 23, 24 facing North)
        elif self.step_idx == 7:
            return 5  # DO (mine stone at 22, 24 -> collect_stone)
        elif self.step_idx == 8:
            return 6  # SLEEP (sleep -> wake_up)
        return 0
"""
    try:
        eval_res = pool.evaluate_batch(
            batch_id="test_craftax_batch",
            policy_code=test_policy,
            macros_code="",
            target_episodes=4,
            benchmark="craftax",
            record_ticks=True,
            show_progress=False,
        )
        assert eval_res["episodes_recorded"] == 4
        assert len(eval_res["runs"]) == 4
        for r in eval_res["runs"]:
            assert r["achievements"] >= 4
    finally:
        pool.close()


def test_nethack_scenario_craftax_contracts_and_ast_fixes():
    # Craftax WAITING + None episode fields
    ch = CraftaxHarness(max_steps=20)
    ch.reset(seed=7)
    st, done, info = ch.step(Action.WAIT)
    assert st.step_count == 1
    assert info["action_name"] == "NOOP"

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
    assert (
        check_macro_contracts(
            "def f(): return m_ok(1, 2, 3)",
            "def m_ok(a, *args):\n    def inner(): yield 1\n    return a\n",
        )
        == []
    )

    # 3-way PrioritySelector overlap check (insertion inside replacement must conflict)
    base_ps = "def build_tree():\n    return PrioritySelector([A(), B(), C()])\n"
    branch_a = "def build_tree():\n    return PrioritySelector([X()])\n"
    branch_b = (
        "def build_tree():\n    return PrioritySelector([A(), INS(), B(), C()])\n"
    )
    assert three_way_ast_merge(base_ps, branch_a, branch_b) is None

    # Scheduler 320 threshold + calibrate_pilot delta regex
    assert ModeScheduler().refactor_line_threshold == 320
    assert parse_pilot_delta("Δ=+0.35 wins=10") == 0.35
    assert parse_pilot_delta(r"\Delta=-0.12") == -0.12


def test_phase5_scenarios_contracts_and_strategy_stack():
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

    # StrategyStack preemption & cleanup unwind
    cleanup_log: list[str] = []
    bb = Blackboard()
    stack = StrategyStack()
    danger = {"now": False}

    def low_gen(_st, _bb):
        yield Action.N
        yield Action.N

    def hi_gen(_st, _bb):
        yield Action.ESC

    stack.register(
        StrategyFrame(
            name="high_escape",
            priority=100,
            factory=hi_gen,
            preempt_if=preempt_if(lambda _s, _b: danger["now"]),
            until=until(lambda _s, _b: not danger["now"]),
            cleanup=with_cleanup(lambda _s, _b: cleanup_log.append("hi_done")),
        )
    )
    stack.push(
        StrategyFrame(
            name="low_explore",
            priority=10,
            factory=low_gen,
            cleanup=with_cleanup(lambda _s, _b: cleanup_log.append("low_cleaned")),
        )
    )
    assert stack.tick(None, bb) == Action.N
    danger["now"] = True
    assert stack.tick(None, bb) == Action.ESC
    assert stack.preemptions == 1
    assert "low_cleaned" in cleanup_log

    # Contracts check
    gi = GoalIntent(goal_id="descend", priority=1.0, target_pos=(4, 5))
    sr = SafetyReport(safe_to_continue=True, risk_score=0.1)
    assert gi.goal_id == "descend" and sr.safe_to_continue
    assert validate_contract_instance(object()) == []

    # Policy compiling & importing contracts + strategy inside LOX sandbox
    sandbox_pol = """from lox.lang.strategy import StrategyStack, StrategyFrame
from lox.lang.contracts import GoalIntent
class Policy:
    def __init__(self):
        self.stack = StrategyStack()
        self.intent = GoalIntent(goal_id="explore", priority=1.0)
    def step(self, state):
        return Action.WAIT
"""
    compiled = compile_policy_package(sandbox_pol)
    runner = compiled.create_runner()
    assert runner is not None
