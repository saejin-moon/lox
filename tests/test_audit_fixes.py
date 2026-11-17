"""
Targeted regression tests for bugs uncovered during the full file-by-file codebase audit.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from lox.author.agent import AuthorAgent
from lox.author.swarm_deficits import DeficitCluster
from lox.datasets.symbols import SYMBOL_REGISTRY
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import init_db
from lox.db.summarizer import build_batch_contrast_digest, summarize_episode_ticks
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.env.state import FloorMemory, WorldMemory
from lox.lang.bt import ActionLeaf, Blackboard, ConditionLeaf, Sequence
from lox.runner.pool import run_single_episode


def test_symbols_registry_real_fields():
    food = SYMBOL_REGISTRY.query_item("food ration")
    assert food is not None
    assert "nutrition" not in food
    assert food["delay"] > 0
    assert food["weight"] > 0
    assert "oprop" in food and "descr" in food

    eye = SYMBOL_REGISTRY.query_monster("floating eye")
    assert eye is not None
    assert "attacks" not in eye
    assert "flags1" in eye and "flags2" in eye and "flags3" in eye
    assert "geno" in eye and "msound" in eye and "mlet" in eye


def test_bt_bound_method_leaves():
    class Helper:
        def is_ok(self, state) -> bool:
            return bool(state.hp > 0)

        def act_step(self, state):
            return Action.WAIT

    h = Helper()
    bb = Blackboard()

    class DummyState:
        hp = 10
        depth = 1
        dungeon_num = 0
        y = 5
        x = 5

    seq_node = Sequence([ConditionLeaf(h.is_ok), ActionLeaf(h.act_step)])
    status, act = seq_node.tick(DummyState(), bb)
    assert act == Action.WAIT
    assert status.name == "SUCCESS"
    assert not bb.leaf_errors


def test_floor_aware_summarizer_and_killers():
    ticks = []
    # Walk on floor 1 then descend to floor 2 and revisit same (10, 20) coordinates
    for i in range(1, 31):
        d = 1 if i <= 15 else 2
        y = 10
        x = 20 + (i % 4)
        ticks.append(
            {
                "turn": i,
                "step_idx": i,
                "depth": d,
                "dungeon_num": 0,
                "hp": 10,
                "max_hp": 10,
                "ac": 10,
                "xp_level": 1,
                "score": i,
                "y": y,
                "x": x,
                "action_name": "EAST",
                "active_function": "explore",
                "message": "",
            }
        )
    digest, _log, ms = summarize_episode_ticks(ticks)
    assert "CONFINEMENT_STALL" not in digest
    assert ms["max_depth"] == 2

    contrast = build_batch_contrast_digest(
        {"batch_id": "b1", "avg_depth": 1.0, "p90_depth": 1.0, "max_depth": 1},
        [
            {
                "run_id": "r_kill_1",
                "seed": 42,
                "max_depth": 1,
                "score": 10,
                "turns": 20,
                "end_reason": "killed by a jackal",
                "death_message": "Killed by a jackal.",
                "summary_digest": "",
            }
        ],
        [],
    )
    assert "Top Lethal Monsters/Hazards" in contrast
    assert "Killed by a jackal." in contrast


def test_state_masks_readonly_and_object_occlusion_cleared():
    fm = FloorMemory()
    assert fm.passable_mask.flags.writeable is False
    assert fm.safe_passable_mask.flags.writeable is False

    h = NetHackHarness(max_steps=50)
    try:
        st = h.reset(seed=42)

        class FakeMsgState:
            y = st.y
            x = st.x
            depth = st.depth
            level_num = st.level_num
            dungeon_num = st.dungeon_num
            turn = st.turn + 1
            step_count = st.step_count + 1
            chars = st.chars
            glyphs = st.glyphs
            message = "There is nothing here to pick up."

        wm2 = WorldMemory()
        wm2.remembered_obj_glyphs[5, 5] = 123
        fms = FakeMsgState()
        fms.y = 5
        fms.x = 5
        fms.chars = np.full((21, 79), ord("."), dtype=np.uint8)
        fms.glyphs = np.zeros((21, 79), dtype=np.int16)
        wm2.update(fms, action_name="PICKUP")
        assert int(wm2.remembered_obj_glyphs[5, 5]) == 0
    finally:
        h.close()


def test_logger_baseline_preserves_oprop_and_branches(tmp_path):
    db_path = str(tmp_path / "test_logger.duckdb")
    init_db(db_path).close()
    logger = BatchDatabaseLogger(db_path=db_path)
    pol = "class Policy:\n    def step(self, s):\n        return None\n"
    logger.save_policy("pol_g1", 1, "SWARM", pol, "", {}, 2.0, 2, is_promoted=True)
    run_row = {
        "run_id": "b1_r1",
        "seed": 42,
        "role": "val-hum-law-fem",
        "depth": 2,
        "max_depth": 2,
        "score": 50,
        "turns": 100,
        "steps": 100,
        "xp_level": 2,
        "final_hp": 12,
        "max_hp": 15,
        "final_ac": 8,
        "gold": 10,
        "end_reason": "died",
        "death_message": "died",
        "summary_digest": "",
        "game_log": "",
        "milestones": {},
        "worn_oprop_count": 3,
        "dungeons_visited": 2,
        "triggered_functions": {"step": 100},
        "fn_turns": {"step": 100},
        "last_active_function": "step",
        "wall_ms": 5.0,
        "ticks": [],
    }
    logger.save_batch(
        batch_id="b1",
        generation=1,
        mode="SWARM",
        parent_policy_id="none",
        candidate_policy_id="pol_g1",
        target_functions=["step"],
        runs=[run_row],
        episodes_discarded=0,
        batch_metrics={
            "avg_depth": 2.0,
            "p90_depth": 2.0,
            "max_depth": 2,
            "avg_score": 50.0,
            "max_score": 50,
            "avg_turns": 100.0,
        },
        verdict={"passed": True, "reason": "ok"},
    )
    loaded = logger.load_champion_baseline()
    assert loaded is not None
    r0 = loaded["baseline_runs"][0]
    assert int(r0["worn_oprop_count"]) == 3
    assert int(r0["dungeons_visited"]) == 2


def test_tools_read_policy_current_and_synthesize_preserves_peer_context(tmp_path):
    db_path = str(tmp_path / "peer_ctx.duckdb")
    init_db(db_path).close()
    agent = AuthorAgent(provider="openrouter", db_path=db_path, api_key="test-key")
    cur_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    deficit = DeficitCluster(
        cluster_id="c1",
        kind="early_trajectory",
        label="test",
    )
    agent.tools.configure_generation(
        pool=None,
        baseline_runs=[],
        generation=2,
        mode="SWARM_PEER",
        claimed_deficit=deficit,
        scenario_gate="strict",
    )
    agent.tools.enforce_peer_research_gate = True
    agent.tools.current_policy_code = cur_pol

    txt, sum_s = agent.tools.read_policy()
    assert "class Policy:" in txt
    assert "current" in sum_s

    with patch.object(
        agent,
        "_call_llm_api",
        return_value={
            "content": f"```python\n{cur_pol}\n```",
            "_reasoning": "",
            "_usage": {},
        },
    ):
        agent.synthesize(
            mode="SWARM_PEER",
            generation=2,
            current_policy=cur_pol,
            max_tool_turns=1,
        )
    assert agent.tools.claimed_deficit is deficit
    assert agent.tools.scenario_gate == "strict"
    assert agent.tools.enforce_peer_research_gate is True


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


def test_agent_64k_max_tokens_and_wait_dominant_rejection():
    from lox.author.swarm_deficits import DeficitReport
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate

    agent = AuthorAgent(provider="mock")
    assert agent.max_tokens == 65536

    pol_zombie = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.did = False\n"
        "    def step(self, state):\n"
        "        if not self.did:\n"
        "            self.did = True\n"
        "            return Action.SEARCH\n"
        "        return Action.WAIT\n"
    )
    cand = incubate_candidate(
        PeerCandidate(
            peer_id="p_zombie",
            claimed_deficit_id="c1",
            claimed_label="zombie wait",
            policy_code=pol_zombie,
        ),
        deficit_report=DeficitReport(generation=1, regime="BOOTSTRAP"),
        claimed_deficit=DeficitCluster(cluster_id="c1", kind="test", label="test"),
    )
    assert cand.sandbox_passed is False
    assert "WAIT-dominant deadlock" in cand.error
