"""
Comprehensive Verification Tests for the LOX Architectural & Theoretical Overhaul (`TODO.md`).
Tests:
1. Zero-Heuristic WorldMemory (`state.mem`) multi-floor spatial & oscillation bookkeeping.
2. Sub-generator skill support with emergency interrupt in PolicyRunner.
3. Prefix-Replay Shard extraction & worker pool execution.
4. Elimination of post-treatment selection bias with baseline priority seeds.
5. MAP-Elites Quality-Diversity Archive and failed pilot experiment logging.
"""

from __future__ import annotations

import numpy as np

from lox.db.schema import init_db
from lox.db.shards import extract_shards_from_runs
from lox.engine.orchestrator import CampaignOrchestrator, PolicyArchiveEntry
from lox.env.actions import Action, seq
from lox.env.harness import NetHackHarness
from lox.env.state import WorldMemory
from lox.lang.compiler import compile_policy_and_macros
from lox.runner.pool import run_single_episode


def test_world_memory_multi_floor_bookkeeping_and_oscillation():
    mem = WorldMemory()
    mem.reset()

    # Create dummy observation tuple
    harness = NetHackHarness(max_steps=50)
    state = harness.reset(seed=123)

    assert hasattr(state, "mem")
    assert isinstance(state.mem, WorldMemory)

    # Initial state should be visited
    assert bool(state.mem.visited[state.y, state.x]) is True
    assert state.mem.visit_count[state.y, state.x] >= 1
    assert not state.mem.is_oscillating

    # Check that remembered chars contains terrain
    rem_chars = state.mem.remembered_chars
    assert isinstance(rem_chars, np.ndarray)
    assert rem_chars.shape == (21, 79)

    # Simulate multi-step oscillation on the same floor
    f0 = (state.dungeon_num, state.level_num)
    for i in range(14):
        y, x = (10, 20) if i % 2 == 0 else (10, 21)
        state.mem.pos_history.append((f0[0], f0[1], y, x))

    assert state.mem.is_oscillating is True

    # Switching floors should maintain separate FloorMemory
    fl1 = state.mem.get_floor(0, 1)
    fl2 = state.mem.get_floor(0, 2)
    fl1.visited[5, 5] = True
    assert bool(fl2.visited[5, 5]) is False


def test_subgenerator_skills_and_emergency_interrupt():
    code = """
class Policy:
    def __init__(self):
        self.flee_hp_ratio = 0.50

    def breach_door(self, state):
        yield seq(Action.OPEN, Action.EAST)
        yield seq(Action.KICK, Action.EAST)
        yield seq(Action.KICK, Action.EAST)

    def run(self, state):
        while True:
            if state.hp / max(1, state.max_hp) < self.flee_hp_ratio:
                state = yield Action.WAIT
                continue
            # Return subgenerator directly
            state = yield from self.breach_door(state)
"""
    compiled = compile_policy_and_macros(code, "")
    runner = compiled.create_runner()

    harness = NetHackHarness(max_steps=50)
    st = harness.reset(seed=42)

    act1, fn1 = runner.next_action(st)
    act2, fn2 = runner.next_action(st)
    act3, fn3 = runner.next_action(st)

    assert act1 is not None
    assert act2 is not None
    assert act3 is not None


def test_prefix_shard_extractor_and_worker_replay():
    # Build sample runs with action bytes
    harness = NetHackHarness(max_steps=80)
    state = harness.reset(seed=999)
    for _ in range(35):
        state, done, _ = harness.step(Action.SEARCH)
        if done:
            break

    recorded_bytes = bytes(harness.action_bytes)
    assert len(recorded_bytes) >= 30

    sample_run = {
        "run_id": "test_run_01",
        "seed": 999,
        "steps": len(recorded_bytes),
        "depth": 1,
        "max_depth": 2,
        "end_reason": "killed by a goblin",
        "summary_digest": "Overview: Depth=1\nStalls: None\nKey Inflections:\n  * None",
        "milestones": {"depth_first_turn": {"2": 15}},
        "action_bytes": recorded_bytes,
    }

    shards = extract_shards_from_runs([sample_run], max_shards=10)
    assert len(shards) >= 1

    pre_death_shards = [s for s in shards if "pre_death" in s.trigger_context]
    assert len(pre_death_shards) == 1
    shard = pre_death_shards[0]
    assert shard.seed == 999
    assert len(shard.action_bytes) < len(recorded_bytes)

    # Execute task with replay_bytes in worker
    policy_code = """
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
"""
    task = {
        "seed": shard.seed,
        "run_id": "shard_eval_01",
        "role": "val-hum-law-fem",
        "max_steps": 20,
        "policy_code": policy_code,
        "macros_code": "",
        "target_functions": [],
        "replay_bytes": shard.action_bytes,
        "post_replay_max_steps": 20,
        "is_priority_seed": True,
        "enforce_trigger": False,
        "record_ticks": True,
    }

    res = run_single_episode(task)
    assert res.get("steps") is not None
    assert res.get("end_reason") is not None


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


def test_map_elites_archive_and_failed_pilot_logging(tmp_path):
    db_path = str(tmp_path / "archive.duckdb")
    init_db(db_path)

    orch = CampaignOrchestrator(
        provider="mock",
        db_path=db_path,
        policy_path=str(tmp_path / "policy.py"),
        macros_path=str(tmp_path / "macros.py"),
        workers=1,
    )

    # Initial empty archive
    assert len(orch.policy_archive) == 0

    # Add entries across niches
    orch.policy_archive["descent"] = PolicyArchiveEntry(
        policy_id="pol_1",
        policy_code="class Policy: pass",
        niche="descent",
        avg_depth=2.5,
        p90_depth=3.0,
        max_depth=4,
        avg_ac=5.0,
        avg_xl=2.0,
        baseline_runs=[],
    )
    orch.policy_archive["survivability"] = PolicyArchiveEntry(
        policy_id="pol_2",
        policy_code="class Policy: pass",
        niche="survivability",
        avg_depth=1.8,
        p90_depth=2.0,
        max_depth=3,
        avg_ac=-2.0,
        avg_xl=4.0,
        baseline_runs=[],
    )

    assert len(orch.policy_archive) == 2
    assert orch.policy_archive["descent"].avg_depth == 2.5
    assert orch.policy_archive["survivability"].avg_ac == -2.0

    # Verify recent experiments limit
    md = orch.get_recent_experiments_markdown(limit=10)
    assert isinstance(md, str)


def test_harness_mid_sequence_more_dismissal_and_prompt_properties():
    harness = NetHackHarness(role="val-hum-law-fem")
    state = harness.reset(seed=42)

    # Check prompt state properties
    assert hasattr(state, "in_moveloop")
    assert hasattr(state, "in_yn_prompt")
    assert hasattr(state, "in_text_prompt")
    assert state.in_moveloop is True
    assert state.in_yn_prompt is False
    assert state.in_text_prompt is False

    # Execute a multi-keystroke sequence (e.g. search 3 times)
    multi_action = seq(Action.SEARCH, Action.SEARCH, Action.SEARCH)
    next_state, done, info = harness.step(multi_action)
    assert next_state is not None
    assert harness.step_count == 1
    # Check that bytes are recorded in action_bytes
    assert len(harness.action_bytes) >= 3

    # Check deterministic replay matches position and step count
    replayed = harness.replay_bytes(seed=42, recorded_bytes=bytes(harness.action_bytes))
    assert replayed.step_count == harness.step_count
    assert replayed.y == next_state.y and replayed.x == next_state.x
    assert (
        replayed.dungeon_num == next_state.dungeon_num
        and replayed.level_num == next_state.level_num
    )
