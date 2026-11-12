"""
Tests for Phase 2 Architectural Implementations:
1. Reactive Behavior Tree Engine (`lox.lang.bt`) & `PolicyRunner` native BT integration.
2. Modular starter skills (`skills/bt.py`, `skills/navigation.py`, `skills/memory.py`, `skills/pagination.py`, `skills/prompts.py`).
3. C-level symbol ground truth (`lox.datasets.symbols`) & author tools (`query_monster`, `query_item`).
4. Quality-Diversity MAP-Elites Behavioral Archive (`lox.engine.map_elites`).
5. Go-Explore Frontier Checkpoints (`lox.engine.frontier`).
6. Environment & execution purification ("Hard Code Nothing" invariant verification).
"""

from __future__ import annotations

from typing import Any

from lox.author.tools import AuthorToolRegistry
from lox.datasets.symbols import NetHackSymbolRegistry
from lox.db.schema import init_db
from lox.engine.frontier import FrontierArchive
from lox.engine.map_elites import MAPElitesArchive
from lox.env.actions import Action
from lox.lang.bt import (
    ActionLeaf,
    BehaviorTree,
    Blackboard,
    ConditionLeaf,
    Inverter,
    NodeStatus,
    PrioritySelector,
    Sequence,
)
from lox.lang.compiler import compile_policy_package


# ---------------------------------------------------------------------------
# 1. Behavior Tree Engine Unit Tests
# ---------------------------------------------------------------------------
def test_behavior_tree_composites_and_leaves():
    bb = Blackboard()
    bb.set_var("danger", True)

    def is_in_danger(st: Any, b: Blackboard) -> bool:
        return b.get_var("danger", False)

    def flee_action(st: Any, b: Blackboard) -> Action:
        return Action.WAIT

    def explore_action(st: Any) -> Action:
        return Action.SEARCH

    cond_danger = ConditionLeaf(is_in_danger, name="DangerCheck")
    act_flee = ActionLeaf(flee_action, name="FleeAct")
    flee_branch = Sequence([cond_danger, act_flee], name="FleeBranch")

    act_explore = ActionLeaf(explore_action, name="ExploreAct")

    root = PrioritySelector([flee_branch, act_explore], name="RootSelector")
    tree = BehaviorTree(root=root, blackboard=bb)

    # When danger is True, flee_branch executes and returns Action.WAIT
    act1 = tree.tick(state=None)
    assert act1 == Action.WAIT

    # When danger is False, flee_branch fails, fallback to act_explore (Action.SEARCH)
    bb.set_var("danger", False)
    act2 = tree.tick(state=None)
    assert act2 == Action.SEARCH


def test_behavior_tree_inverter():
    bb = Blackboard()
    cond = ConditionLeaf(lambda st: False)
    inv = Inverter(cond)
    status, _ = inv.tick(None, bb)
    assert status == NodeStatus.SUCCESS


def test_policy_runner_native_bt_support():
    pol_code = """
from lox.lang.bt import BehaviorTree, PrioritySelector, ActionLeaf
from lox.env.actions import Action

class Policy:
    def __init__(self):
        leaf1 = ActionLeaf(lambda st: Action.SEARCH)
        self.tree = BehaviorTree(root=leaf1)
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()
    assert runner._mode == "bt"

    class DummyState:
        hp = 20
        max_hp = 20
        turn = 1

    st = DummyState()
    act, fn = runner.next_action(st)
    assert act == Action.SEARCH


# ---------------------------------------------------------------------------
# 2. C Engine Symbol Ground Truth Tests
# ---------------------------------------------------------------------------
def test_c_symbol_ground_truth_registry():
    reg = NetHackSymbolRegistry()

    # Query monster
    jackal = reg.query_monster("jackal")
    assert jackal is not None
    assert jackal["name"] == "jackal"
    assert "level" in jackal
    assert "ac" in jackal
    assert "speed" in jackal

    # Query item
    food = reg.query_item("food ration")
    assert food is not None
    assert food["name"] == "food ration"
    assert food["class"] == "FOOD"
    assert "cost" in food
    assert "weight" in food

    # Search items
    potions = reg.search_items("potion", limit=3)
    assert len(potions) >= 1
    assert any("potion" in p["name"] for p in potions)


def test_author_tools_symbol_queries(tmp_path):
    db_path = str(tmp_path / "test_sym.duckdb")
    con = init_db(db_path)
    con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    m_out, m_stat = tools.query_monster("floating eye")
    assert "floating eye" in m_out
    assert m_stat == "found"

    i_out, i_stat = tools.query_item("potion of extra healing")
    assert "potion of extra healing" in i_out
    assert i_stat == "found"

    # Test query_db does NOT rewrite SELECT MAX(batch_id)
    # The literal query string should be executed as-is
    res, _ = tools.query_db("SELECT 1 AS num WHERE 1 = (SELECT 1)")
    assert "num: 1" in res


# ---------------------------------------------------------------------------
# 3. MAP-Elites 3D Quality-Diversity Archive Tests
# ---------------------------------------------------------------------------
def test_map_elites_archive(tmp_path):
    db_path = str(tmp_path / "test_map.duckdb")
    con = init_db(db_path)

    archive = MAPElitesArchive()
    assert archive.coverage() == 0.0

    # Policy A: Dlvl 1, low caution (rush), low resource
    metrics_a = {
        "avg_depth": 1.0,
        "max_depth": 1,
        "pct_wait": 0.01,
        "pct_search": 0.01,
        "avg_score": 10.0,
        "avg_turns": 100.0,
    }
    added_a, cell_a, elite_a = archive.add_candidate(
        policy_code="class Policy: pass # A",
        metrics=metrics_a,
        batch_id="b_001",
        generation=1,
        con=con,
    )
    assert added_a is True
    assert cell_a == "cell_d0_c0_r0"
    assert archive.coverage() > 0.0

    # Policy B: Dlvl 2, high caution (searching), medium resource
    metrics_b = {
        "avg_depth": 2.0,
        "max_depth": 2,
        "pct_wait": 0.20,
        "pct_search": 0.20,
        "avg_score": 150.0,
        "avg_turns": 200.0,
    }
    added_b, cell_b, elite_b = archive.add_candidate(
        policy_code="class Policy: pass # B",
        metrics=metrics_b,
        batch_id="b_002",
        generation=2,
        con=con,
    )
    assert added_b is True
    assert cell_b == "cell_d1_c2_r2"

    # Policy A_worse: same cell as A, lower fitness -> rejected
    metrics_a_worse = {
        "avg_depth": 1.0,
        "max_depth": 1,
        "pct_wait": 0.01,
        "pct_search": 0.01,
        "avg_score": 0.0,
        "avg_turns": 20.0,
    }
    added_aw, cell_aw, _ = archive.add_candidate(
        policy_code="class Policy: pass # Aw",
        metrics=metrics_a_worse,
        batch_id="b_003",
        generation=3,
        con=con,
    )
    assert added_aw is False
    assert cell_aw == "cell_d0_c0_r0"

    # Sampling
    best_sample = archive.sample_elite(strategy="best")
    assert best_sample is not None
    assert best_sample.cell_id == cell_b  # Policy B reached Dlvl 2, higher fitness

    curiosity_sample = archive.sample_elite(strategy="curiosity")
    assert curiosity_sample is not None

    # Load from DB
    new_archive = MAPElitesArchive()
    loaded_count = new_archive.load_from_db(con)
    assert loaded_count == 2
    assert "cell_d0_c0_r0" in new_archive.cells
    assert "cell_d1_c2_r2" in new_archive.cells

    con.close()


# ---------------------------------------------------------------------------
# 4. Go-Explore Frontier Checkpoints Tests
# ---------------------------------------------------------------------------
def test_frontier_archive_save_and_sample(tmp_path):
    frontier_dir = tmp_path / "frontier"
    archive = FrontierArchive(frontier_dir=str(frontier_dir))

    # Add checkpoint
    dummy_bytes = bytes([ord("k"), ord("k"), ord("l"), ord("h")])
    cp = archive.add_checkpoint(
        seed=12345,
        depth=4,
        milestone="DLVL_4",
        turns=650,
        action_bytes=dummy_bytes,
        game_log="You enter level 4.",
    )
    assert cp.checkpoint_id.startswith("fc_dlvl_4_")
    assert len(archive.checkpoints) == 1

    # Reload from disk
    new_archive = FrontierArchive(frontier_dir=str(frontier_dir))
    assert len(new_archive.checkpoints) == 1
    sampled = new_archive.sample_checkpoint(min_depth=3)
    assert sampled is not None
    assert sampled.depth == 4
    assert sampled.action_bytes == dummy_bytes


# ---------------------------------------------------------------------------
# 5. Environment & Execution Purification ("Hard Code Nothing") Tests
# ---------------------------------------------------------------------------
def test_environment_purification_no_auto_esc():
    """PolicyRunner must NOT silently replace actions with ESC when in a text prompt."""
    pol_code = """
class Policy:
    def step(self, state):
        return Action.WAIT
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()

    class DummyPromptState:
        hp = 20
        max_hp = 20
        turn = 1
        in_text_prompt = True
        in_yn_prompt = False

    st = DummyPromptState()

    # Call next_action multiple times: Action.WAIT must NOT be rewritten to ESC
    for _ in range(5):
        act, _ = runner.next_action(st)
        assert act == Action.WAIT
    assert runner.auto_esc_count == 0


def test_environment_purification_no_low_hp_preemption():
    """PolicyRunner must NOT cancel active subgenerators solely because HP is low."""
    pol_code = """
class Policy:
    def subgen(self, state):
        while True:
            yield Action.SEARCH

    def run(self, state):
        yield from self.subgen(state)
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()

    class DummyLowHpState:
        hp = 1
        max_hp = 20
        turn = 1
        in_text_prompt = False
        in_yn_prompt = False

    st = DummyLowHpState()

    act1, fn1 = runner.next_action(st)
    assert act1 == Action.SEARCH

    act2, fn2 = runner.next_action(st)
    assert act2 == Action.SEARCH
    assert runner.subgen_preemptions == 0
