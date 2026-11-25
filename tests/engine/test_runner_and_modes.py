"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.lang.compiler import compile_policy_and_macros
from lox.runner.pool import PersistentEvalPool, run_single_episode


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


def test_pool_unique_tiles_floor_aware():
    pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    res = run_single_episode(
        {
            "seed": 42,
            "run_id": "test_floor_tiles",
            "policy_code": pol,
            "max_steps": 10,
            "record_ticks": False,
        }
    )
    assert res["triggered"] is True
    assert res["unique_tiles"] >= 1
    assert res["dungeons_visited"] >= 1


def test_pool_preserves_last_depth_on_terminal_death():
    from lox.runner.pool import run_single_episode

    suicidal_pol = (
        "from lox.env.actions import Action, extcmd, key\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        if state.in_yn_prompt:\n"
        "            return key('y')\n"
        "        return extcmd('pray')\n"
    )
    res = run_single_episode(
        {
            "seed": 12304,
            "run_id": "test_death_depth",
            "role": "val-hum-law-fem",
            "max_steps": 120,
            "policy_code": suicidal_pol,
            "macros_code": "",
            "skills": {},
            "record_ticks": False,
        }
    )
    assert res["depth"] >= 1
    assert res["max_depth"] >= 1


def test_selection_bias_elimination_priority_seeds():
    # Policy with an avoided low-HP emergency state
    policy_code = """
class Policy:
    def handle_emergency(self, state):
        yield Action.WAIT

    def run(self, state):
        while True:
            # Avoid ever calling handle_emergency
            state = yield Action.SEARCH
"""
    _ = compile_policy_and_macros(policy_code, "")

    # Priority seed task (baseline called handle_emergency, candidate avoids it)
    task_priority = {
        "seed": 101,
        "run_id": "p_task_01",
        "role": "val-hum-law-fem",
        "max_steps": 10,
        "policy_code": policy_code,
        "macros_code": "",
        "target_functions": ["handle_emergency"],
        "is_priority_seed": True,
        "enforce_trigger": False,
        "record_ticks": False,
    }
    res_priority = run_single_episode(task_priority)
    # Must NOT be discarded
    assert res_priority.get("triggered") is not False
    assert res_priority.get("steps") is not None

    # Random fresh seed task (untriggered should be discarded)
    task_fresh = {
        "seed": 102,
        "run_id": "fresh_task_01",
        "role": "val-hum-law-fem",
        "max_steps": 10,
        "policy_code": policy_code,
        "macros_code": "",
        "target_functions": ["handle_emergency"],
        "is_priority_seed": False,
        "enforce_trigger": True,
        "record_ticks": False,
    }
    res_fresh = run_single_episode(task_fresh)
    assert res_fresh.get("triggered") is False
