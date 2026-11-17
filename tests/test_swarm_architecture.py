"""
Automated Test Suite for the Hierarchical Multi-Agent Swarm:
1. TheoryDocumentStore (LanceDB BM25 Tantivy FTS).
2. Exact LLM cost and token accounting in DuckDB.
3. Frontier checkpoint bootstrapping from recorded runs.
4. Category SwarmWorker execution cycle and report generation.
5. Meta-Agent master Behavior Tree policy assembly and live execution.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_worker import SwarmWorker
from lox.datasets.bootstrap_frontier import bootstrap_frontier_from_duckdb
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import compute_llm_cost, init_db
from lox.db.theories import TheoryDocumentStore
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_package


def test_theory_document_store_lancedb(tmp_path):
    p = tmp_path / "test_theories"
    store = TheoryDocumentStore(db_path=str(p))
    assert store.count() == 1  # Seed row

    # 1. Save document
    doc_id = store.save_document(
        generation=1,
        agent_role="threat",
        doc_type="REFLECTION_REPORT",
        title="Tactical Ant Kiting",
        content="Soldier ants have poisonous bites. In narrow corridors, only one ant can attack at a time.",
        tags=["combat", "ants", "poison"],
        verified_c_rule="corridor choke prevents surrounding",
    )
    assert doc_id.startswith("doc_threat_")
    assert store.count() == 2

    # 2. Search document
    hits = store.search_theories("ants poison", limit=3)
    assert len(hits) >= 1
    assert "Tactical Ant Kiting" in hits[0]["title"]

    # 3. Retrieve latest reports
    reports = store.get_latest_reports(generation=1)
    assert "threat" in reports
    assert reports["threat"]["title"] == "Tactical Ant Kiting"


def test_duckdb_llm_cost_and_token_tracking(tmp_path):
    db_path = str(tmp_path / "test_cost.duckdb")
    con = init_db(db_path)
    con.close()

    # 1. Verify cost formula
    cost = compute_llm_cost(
        "z-ai/glm-5.3-flash:nitro", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert abs(cost - 0.46) < 1e-4

    # 2. Save log with cost
    logger = BatchDatabaseLogger(db_path=db_path)
    logger.save_synthesis_log(
        log_id="log_swarm_01",
        generation=1,
        batch_id="b_swarm_01",
        mode="WRITE",
        turn_index=1,
        turn_type="PILOT",
        model="z-ai/glm-5.3-flash:nitro",
        prompt_tokens=50000,
        completion_tokens=2000,
        reasoning_tokens=500,
        agent_role="threat",
    )

    con = duckdb.connect(db_path, read_only=True)
    row = con.execute(
        "SELECT agent_role, cost_usd, prompt_tokens, completion_tokens FROM synthesis_logs"
    ).fetchone()
    assert row[0] == "threat"
    assert row[1] > 0.0

    rates_count = con.execute("SELECT COUNT(*) FROM llm_cost_rates").fetchone()[0]
    assert rates_count >= 8
    con.close()


def test_frontier_bootstrapping_and_checkpoints(tmp_path):
    frontier_dir = str(tmp_path / "frontier")
    # Bootstrap from current data/lox.duckdb if available
    db_file = Path("data/lox.duckdb")
    if db_file.exists():
        count = bootstrap_frontier_from_duckdb(
            db_path=str(db_file),
            frontier_dir=frontier_dir,
            max_checkpoints_per_depth=2,
        )
        assert count > 0


def test_swarm_worker_and_meta_agent_end_to_end(tmp_path):
    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_swarm.duckdb")
    con = init_db(db_path)
    con.close()

    # 1. Worker runs cycle
    worker = SwarmWorker(
        role="threat",
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
        max_turns=2,
    )
    res = worker.run_cycle(generation=1)
    assert res["role"] == "threat"
    assert "report_id" in res

    # 2. Meta-Agent assembles policy
    meta = SwarmMetaAgent(
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
    )
    master_code = meta.assemble_policy(generation=1)
    assert "class Policy" in master_code
    assert "BehaviorTree" in master_code

    # 3. Compile and execute policy in live harness
    compiled = compile_policy_package(master_code)
    runner = compiled.create_runner()
    assert runner._mode in ("bt", "step")

    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        act, fn = runner.next_action(st)
        assert act in Action
    finally:
        h.close()


def test_dynamic_macro_category_discovery(tmp_path):
    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_dyn.duckdb")
    con = init_db(db_path)
    con.close()

    meta = SwarmMetaAgent(
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
    )
    categories = meta.discover_macro_categories(generation=1, n_categories=4)
    assert len(categories) >= 3
    for cat in categories:
        assert "category_id" in cat
        assert "name" in cat
        assert "target_depth_tier" in cat
        assert "target_metrics" in cat

    # Verify category decomposition stored in LanceDB
    docs = meta.theories.search_theories("decomposition", limit=5)
    assert len(docs) >= 1
    assert docs[0]["doc_type"] == "CATEGORY_DECOMPOSITION"


def test_swarm_campaign_orchestrator_end_to_end(tmp_path):
    workspace = tmp_path / "workspace"
    frontier = tmp_path / "frontier"
    db_path = str(tmp_path / "swarm_test.duckdb")

    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator

    orchestrator = SwarmCampaignOrchestrator(
        db_path=db_path,
        workspace_dir=str(workspace),
        frontier_dir=str(frontier),
        provider="mock",
        num_workers=2,
        turns_per_worker=1,
        seeds_per_batch=2,
    )
    try:
        history = orchestrator.run(generations=1)
        assert len(history) == 1
        assert history[0]["generation"] == 1
        assert history[0]["avg_depth"] >= 1.0

        con = duckdb.connect(db_path, read_only=True)
        batches_cnt = con.execute("SELECT count(*) FROM batches").fetchone()[0]
        runs_cnt = con.execute("SELECT count(*) FROM runs").fetchone()[0]
        assert batches_cnt == 1
        assert runs_cnt == 2
        con.close()
    finally:
        orchestrator.close()


def test_swarm_self_repair_mechanisms(tmp_path):
    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_repair.duckdb")
    con = init_db(db_path)
    con.close()

    worker = SwarmWorker(
        role="threat",
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
    )

    # 1. Test worker candidate evaluation: clean code succeeds
    clean_code = """
from lox.lang.bt import ActionLeaf
from lox.env.actions import Action
threat_node = ActionLeaf(lambda st: Action.SEARCH)
"""
    obs, err = worker.evaluate_candidate(clean_code, steps=5)
    assert err == ""
    assert obs.survived is True
    assert obs.steps_taken > 0
    assert "SEARCH" in obs.actions_taken

    # 2. Test worker candidate evaluation: faulty runtime code returns diagnostic traceback
    buggy_code = """
from lox.lang.bt import ActionLeaf
def faulty_action(st):
    return 1 / 0  # ZeroDivisionError
buggy_node = ActionLeaf(faulty_action)
"""
    obs, err = worker.evaluate_candidate(buggy_code, steps=5)
    assert "ZeroDivisionError" in err

    # 3. Test Meta-Agent policy verification: valid BehaviorTree code passes
    meta = SwarmMetaAgent(
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
    )
    valid_code = meta.assemble_policy(generation=1)
    is_valid, diag = meta.verify_policy(valid_code, steps=5)
    assert is_valid is True
    assert diag == ""

    # 4. Test Meta-Agent policy verification: invalid code returns failure diagnostic
    syntax_error_code = "class Policy:\n    def step(self): return Action.WAIT"
    is_valid, diag = meta.verify_policy(syntax_error_code, steps=5)
    assert is_valid is False
    assert "PolicyValidationError" in diag or "TypeError" in diag

    # 5. Test Meta-Agent policy verification: WAIT-only stalled fallback is rejected
    wait_only_code = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state): return Action.WAIT\n"
    )
    is_valid, diag = meta.verify_policy(wait_only_code, steps=10)
    assert is_valid is False
    assert "Action.WAIT" in diag


def test_swarm_worker_and_meta_dry_run_modes(tmp_path):
    from lox.author.prompts import build_user_prompt
    from lox.author.tools import AuthorToolRegistry

    db_path = str(tmp_path / "swarm_modes.duckdb")
    con = init_db(db_path)
    con.close()

    worker_prompt = build_user_prompt(
        mode="SWARM_WORKER",
        generation=1,
        current_policy="class Policy:\n    pass\n",
        batch_digest="Role: survival",
        skills={"nav": "def move(): return None"},
    )
    assert "PrioritySelector" in worker_prompt
    assert "SWARM_WORKER" in worker_prompt
    assert "no 2-function limit" in worker_prompt
    assert "Hierarchical Skill Registry" in worker_prompt
    assert "Current Live `policy.py` Reference" in worker_prompt

    from lox.db.schema import get_schema_markdown

    schema_md = get_schema_markdown(db_path)
    assert "`policies`" in schema_md
    assert "`discoveries`" in schema_md
    assert "`latest_pilot_ticks`" in schema_md
    assert "`llm_cost_rates`" in schema_md

    from lox.author.bt_eval import evaluate_bt_subtree

    obs, err = evaluate_bt_subtree(
        "def noop(st, bb): return None\nsubtree = ActionLeaf(noop, name='noop')",
        seed=42,
        steps=5,
        baseline_policy_code=(
            "from skills.nav import fallback_search\n"
            "class Policy:\n"
            "    def step(self, state): return fallback_search(state)\n"
        ),
        skills={
            "nav": "from lox.env.actions import Action\ndef fallback_search(state): return Action.SEARCH\n"
        },
    )
    assert err == ""
    assert obs.actions_taken.get("SEARCH", 0) == 5
    assert "RLE Trace (tail):" in obs.summary()

    from lox.author.agent import AuthorAgent

    ag = AuthorAgent(provider="mock", db_path=db_path)
    calls = {"n": 0}

    def fake_llm(messages, allow_tools=True):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "query_db",
                            "arguments": '{"sql": "SELECT COUNT(*) FROM runs"}',
                        },
                    }
                ],
            }
        return {"content": '[{"category_id": "test_cat"}]', "tool_calls": []}

    ag._call_llm_api = fake_llm
    tool_sess_out = ag.run_tool_session("sys", "usr", max_tool_turns=3)
    assert "test_cat" in tool_sess_out

    meta_prompt = build_user_prompt(
        mode="SWARM_META",
        generation=1,
        current_policy="",
        batch_digest="Reports",
    )
    assert "SWARM_META" in meta_prompt
    assert "BehaviorTree" in meta_prompt
    assert "high turn-share alone is NOT hijacking" in meta_prompt

    tools = AuthorToolRegistry(db_path=db_path)
    tools.configure_generation(
        pool=None,
        baseline_runs=[],
        max_pilot_runs=0,
        pilot_episodes=0,
        generation=1,
        mode="SWARM_WORKER",
    )

    # Subtree with 4 helper functions (would trigger MUTATION_SCOPE_ERROR in legacy WRITE mode)
    # and imports `seq` BEFORE the root PrioritySelector.
    subtree_code = """
from lox.env.actions import Action, seq
from lox.lang.bt import ActionLeaf, ConditionLeaf, PrioritySelector, Sequence

def helper_a(st):
    return getattr(st, "waiting_for_space", False)

def helper_b(st, bb):
    return Action.SPACE

def helper_c(st):
    return st.hp > 0

def helper_d(st, bb):
    bb.set_var("searched", True)
    return Action.SEARCH

subtree = PrioritySelector([
    Sequence([ConditionLeaf(helper_a), ActionLeaf(helper_b)]),
    ActionLeaf(helper_d),
])
"""
    msg, status = tools.dry_run_policy(subtree_code, seed=42, max_turns=15)
    assert status.startswith("PASSED"), msg
    assert tools.last_pilot_improved is True
    assert "Triggered 15/15" in msg

    # Now test SWARM_META with non-empty current_policy_code (must NOT call splice or hit MUTATION_SCOPE_ERROR)
    tools.configure_generation(
        pool=None,
        baseline_runs=[],
        max_pilot_runs=0,
        pilot_episodes=0,
        generation=1,
        mode="SWARM_META",
    )
    tools.current_policy_code = (
        "class Policy:\n    def run(self, state):\n        yield Action.WAIT\n"
    )
    meta_code = """
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector

def fn1(st):
    return Action.SPACE if getattr(st, "waiting_for_space", False) else None

def fn2(st):
    return Action.SEARCH

def fn3(st):
    return Action.WAIT

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        self.tree = BehaviorTree(
            PrioritySelector([ActionLeaf(fn1), ActionLeaf(fn2), ActionLeaf(fn3)]),
            blackboard=self.blackboard,
        )

    def step(self, state):
        return self.tree.tick(state) or Action.SEARCH
"""
    m_msg, m_status = tools.dry_run_policy(meta_code, seed=42, max_turns=15)
    assert m_status.startswith("PASSED"), m_msg
    assert tools.last_pilot_improved is False


def test_run_swarm_clean_resets_campaign_while_preserving_experts(tmp_path):
    import subprocess
    import sys

    db_path = tmp_path / "clean_test.duckdb"
    workspace = tmp_path / "workspace"
    frontier = tmp_path / "frontier"
    (workspace / "skills").mkdir(parents=True)
    (workspace / "skills" / "legacy.py").write_text("x = 1\n")
    (workspace / "policy.py").write_text("from skills.legacy import x\n")
    (workspace / "macros.py").write_text("def m(): pass\n")
    (workspace / "STATUS.md").write_text("stale status\n")
    frontier.mkdir(parents=True)
    (frontier / "stale_cp.json").write_text("{}")

    store = TheoryDocumentStore(db_path=str(workspace / "theories.lance"))
    store.save_document(
        generation=99,
        agent_role="stale_worker",
        doc_type="THEORY",
        title="Stale Theory",
        content="Should be removed by --clean",
    )

    con = init_db(str(db_path))
    con.execute(
        "INSERT INTO policies (policy_id, generation, mode, policy_code, is_promoted) "
        "VALUES ('pol_stale', 99, 'WRITE', 'stale', TRUE)"
    )
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, episodes_recorded, avg_depth, max_depth) "
        "VALUES ('b_stale', 99, 'WRITE', 'pol_stale', 1, 1.0, 1)"
    )
    con.execute(
        "INSERT INTO expert_runs (run_id, source, role, max_depth, turns) "
        "VALUES ('exp_keep_1', 'nao', 'Val', 20, 5000)"
    )
    con.close()

    repo_root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [
            sys.executable,
            "scripts/run_swarm.py",
            "--clean",
            "--provider",
            "mock",
            "--generations",
            "1",
            "--workers",
            "1",
            "--turns-per-worker",
            "1",
            "--seeds",
            "1",
            "--db-path",
            str(db_path),
            "--workspace-dir",
            str(workspace),
            "--frontier-dir",
            str(frontier),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert (workspace / "STATUS.md").exists()
    status_txt = (workspace / "STATUS.md").read_text(encoding="utf-8")
    assert "stale status" not in status_txt
    assert "LOX Campaign Status Dashboard" in status_txt
    assert not (workspace / "skills" / "legacy.py").exists()
    assert not (frontier / "stale_cp.json").exists()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        stale_pol = con.execute(
            "SELECT COUNT(*) FROM policies WHERE policy_id = 'pol_stale'"
        ).fetchone()[0]
        stale_batch = con.execute(
            "SELECT COUNT(*) FROM batches WHERE batch_id = 'b_stale'"
        ).fetchone()[0]
        kept_expert = con.execute(
            "SELECT COUNT(*) FROM expert_runs WHERE run_id = 'exp_keep_1'"
        ).fetchone()[0]
        fresh_batches = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        fresh_policies = con.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        assert stale_pol == 0
        assert stale_batch == 0
        assert kept_expert == 1
        assert fresh_batches == 1
        assert fresh_policies == 1
    finally:
        con.close()


def test_swarm_gen2_with_frontier_checkpoint_brief_and_replay(tmp_path):
    from lox.engine.frontier import FrontierArchive

    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_cp_gen2.duckdb")
    con = init_db(db_path)
    con.close()

    # Build a real action trace and save as FrontierCheckpoint (mimics Gen 1 extraction)
    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=42)
        for _ in range(10):
            st, done, _ = h.step(Action.SEARCH)
            if done:
                break
        act_bytes = bytes(h.action_bytes)
        archive = FrontierArchive(frontier_dir=frontier_dir)
        cp = archive.add_checkpoint(
            seed=42,
            depth=2,
            milestone="dlvl_2",
            turns=int(st.turn),
            action_bytes=act_bytes,
            game_log="Explored room.",
            checkpoint_id="fc_dlvl_2_seed42_g1",
        )
    finally:
        h.close()

    worker = SwarmWorker(
        role="threat",
        provider="mock",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
        max_turns=1,
    )
    brief = worker._build_worker_brief(
        generation=2,
        checkpoint=cp,
    )
    assert "fc_dlvl_2_seed42_g1" in brief
    assert "Dlvl 2" in brief

    res = worker.run_cycle(generation=2, checkpoint=cp)
    assert res["role"] == "threat"


def test_swarm_scaffold_trigger_attribution(tmp_path):
    from lox.author.bt_eval import evaluate_bt_subtree

    scaffold_code = """
from lox.env.actions import Action

class Policy:
    def step(self, state):
        return Action.WAIT
"""
    noop_subtree = """
def _leaf_noop(state, bb):
    return None

subtree = ActionLeaf(_leaf_noop, name="noop")
"""
    obs_noop, err_noop = evaluate_bt_subtree(
        noop_subtree,
        seed=42,
        steps=10,
        baseline_policy_code=scaffold_code,
    )
    assert not err_noop
    assert obs_noop.triggered_steps == 0
    assert obs_noop.total_eval_steps == 10
    assert obs_noop.actions_taken.get("WAIT", 0) == 10

    conditional_subtree = """
def _leaf_even_search(state, bb):
    if state.step_count % 2 == 0:
        return Action.SEARCH
    return None

subtree = ActionLeaf(_leaf_even_search, name="even_search")
"""
    obs_cond, err_cond = evaluate_bt_subtree(
        conditional_subtree,
        seed=42,
        steps=10,
        baseline_policy_code=scaffold_code,
    )
    assert not err_cond
    assert 0 < obs_cond.triggered_steps < obs_cond.total_eval_steps
    assert obs_cond.actions_taken.get("SEARCH", 0) == obs_cond.triggered_steps
    assert obs_cond.actions_taken.get("WAIT", 0) == (
        obs_cond.total_eval_steps - obs_cond.triggered_steps
    )


def test_swarm_worker_multi_hypothesis_session(tmp_path):
    import json

    theories_path = str(tmp_path / "theories.lance")
    frontier_dir = str(tmp_path / "frontier")
    db_path = str(tmp_path / "test_warm_worker.duckdb")
    con = init_db(db_path)
    con.close()

    worker = SwarmWorker(
        role="navigation",
        provider="openrouter",
        db_path=db_path,
        theories_path=theories_path,
        frontier_dir=frontier_dir,
        max_turns=3,
    )

    h1_code = (
        "def h1_noop(state, bb):\n"
        "    return None\n"
        "subtree = ActionLeaf(h1_noop, name='h1_noop')\n"
    )
    h2_broken = (
        "def h2_broken(state, bb):\n"
        "    return undefined_var\n"
        "subtree = ActionLeaf(h2_broken, name='h2_broken')\n"
    )
    h2_repaired = (
        "def h2_east(state, bb):\n"
        "    return Action.WAIT\n"
        "subtree = ActionLeaf(h2_east, name='h2_wait')\n"
    )
    h3_winner = (
        "def h3_explore(state, bb):\n"
        "    seen = bb.get_var('seen') or set()\n"
        "    seen.add((state.y, state.x))\n"
        "    bb.set_var('seen', seen)\n"
        "    passable = state.passable_mask\n"
        "    for act, (dy, dx) in [(Action.E, (0, 1)), (Action.S, (1, 0)), (Action.W, (0, -1)), (Action.N, (-1, 0)), (Action.NE, (-1, 1)), (Action.SE, (1, 1)), (Action.NW, (-1, -1)), (Action.SW, (1, -1))]:\n"
        "        ny, nx = state.y + dy, state.x + dx\n"
        "        if 0 <= ny < 21 and 0 <= nx < 79 and passable[ny, nx] and (ny, nx) not in seen:\n"
        "            return act\n"
        "    return Action.SEARCH\n"
        "subtree = ActionLeaf(h3_explore, name='h3_explore')\n"
    )

    msg_history_lengths: list[int] = []
    call_idx = 0

    def fake_llm_call(messages, allow_tools=True):
        nonlocal call_idx
        call_idx += 1
        msg_history_lengths.append(len(messages))
        if call_idx == 1:
            return {
                "content": (
                    "<hypotheses>\n"
                    "- id: H1\n  mechanism: No-op guard check\n"
                    "- id: H2\n  mechanism: Stationary wait test\n"
                    "- id: H3\n  mechanism: Unvisited neighbor walk\n"
                    "</hypotheses>"
                ),
                "tool_calls": [
                    {
                        "id": "call_1a",
                        "function": {
                            "name": "query_wiki",
                            "arguments": json.dumps({"query": "dungeon exploration"}),
                        },
                    },
                    {
                        "id": "call_1b",
                        "function": {
                            "name": "query_db",
                            "arguments": json.dumps(
                                {"sql": "SELECT COUNT(*) AS n FROM expert_runs"}
                            ),
                        },
                    },
                ],
            }
        if call_idx == 2:
            return {
                "content": "Testing H1.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "dry_run_policy",
                            "arguments": json.dumps(
                                {"code": h1_code, "hypothesis": "H1: noop probe"}
                            ),
                        },
                    }
                ],
            }
        if call_idx == 3:
            return {
                "content": "Testing H2 (initial broken draft).",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "function": {
                            "name": "dry_run_policy",
                            "arguments": json.dumps(
                                {"code": h2_broken, "hypothesis": "H2: wait test"}
                            ),
                        },
                    }
                ],
            }
        if call_idx == 4:
            # Verify runtime error from H2 was fed back in the SAME warm conversation
            last_tool_msg = str(messages[-1].get("content", ""))
            assert "RUNTIME_ERROR" in last_tool_msg or "NameError" in last_tool_msg
            return {
                "content": "Repairing H2 after runtime error.",
                "tool_calls": [
                    {
                        "id": "call_4",
                        "function": {
                            "name": "dry_run_policy",
                            "arguments": json.dumps(
                                {"code": h2_repaired, "hypothesis": "H2: wait test"}
                            ),
                        },
                    }
                ],
            }
        if call_idx == 5:
            return {
                "content": "Testing H3 unvisited exploration.",
                "tool_calls": [
                    {
                        "id": "call_5",
                        "function": {
                            "name": "dry_run_policy",
                            "arguments": json.dumps(
                                {
                                    "code": h3_winner,
                                    "hypothesis": "H3: neighbor walk",
                                }
                            ),
                        },
                    }
                ],
            }
        return {
            "content": (
                "<reflection>\n"
                "winner: H3\n"
                "comparison:\n"
                "  - H1: never triggered\n"
                "  - H2: repaired NameError but gained 1 tile only\n"
                "  - H3: explored multiple new tiles cleanly\n"
                "ruled_out: undefined globals and noop fallbacks\n"
                "</reflection>\n"
                f"<policy_code>\n{h3_winner}\n</policy_code>"
            ),
            "tool_calls": [],
        }

    worker.author._call_llm_api = fake_llm_call  # type: ignore[method-assign]
    res = worker.run_cycle(generation=1)

    # Single warm session: conversation strictly accumulates instead of resetting
    assert msg_history_lengths == sorted(msg_history_lengths)
    assert len(msg_history_lengths) == 6
    assert len(res["experiments"]) == 3
    assert "h3_explore" in res["best_code"]

    docs = worker.theories.list_documents()
    assert len(docs) >= 4
    theory_docs = [d for d in docs if d["doc_type"] == "THEORY"]
    ref_docs = [d for d in docs if d["doc_type"] == "REFLECTION_REPORT"]
    assert len(theory_docs) == 3
    assert len(ref_docs) == 1
    assert "winner: H3" in ref_docs[0]["content"]
    assert "H1: noop probe" in ref_docs[0]["content"]
    latest_reports = worker.theories.get_latest_reports(generation=1)
    assert "navigation" in latest_reports
    assert res["observation"].steps_taken > 0


def test_two_pass_advisory_critic_meta_revision_and_diagnostics(tmp_path):
    import pytest

    from lox.author.swarm_critic import (
        READ_ONLY_CRITIC_TOOLS,
        SwarmCriticAgent,
        build_swarm_diagnostic_digest,
        extract_priority_selector_branches,
    )
    from lox.author.swarm_meta import SwarmMetaAgent
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.lang.bt import (
        ActionLeaf,
        Blackboard,
        ConditionLeaf,
        NodeStatus,
        extract_runner_leaf_errors,
    )
    from lox.lang.compiler import compile_policy_package

    # 1. Blackboard records swallowed leaf exceptions & verify_policy catches them
    bb = Blackboard(strict_errors=False)

    def broken_leaf(state, blackboard):
        raise ZeroDivisionError("boom")

    st, act = ActionLeaf(broken_leaf, name="BrokenAct").tick(None, bb)
    assert st == NodeStatus.FAILURE and act is None
    st2, _ = ConditionLeaf(broken_leaf, name="BrokenCond").tick(None, bb)
    assert st2 == NodeStatus.FAILURE
    assert len(bb.leaf_errors) == 2
    assert bb.leaf_errors[0][0] == "BrokenAct"

    # 2. _build_skill_modules rejects broken runtime skill & compiler preserves skills.<name>.__name__
    skills_dir = tmp_path / "skills"
    worker_reports = {
        "combat_tactics": {
            "best_code": (
                "from lox_policy import missing_symbol\n"
                "subtree = ActionLeaf(missing_symbol, name='broken')\n"
            )
        },
        "good_skill": {
            "best_code": (
                "def good_act(state, bb):\n"
                "    return Action.SEARCH\n"
                "subtree = ActionLeaf(good_act, name='good')\n"
            )
        },
    }
    built = SwarmMetaAgent._build_skill_modules(worker_reports, skills_dir=skills_dir)
    assert "combat_tactics" not in built
    assert "good_skill" in built
    assert not (skills_dir / "combat_tactics.py").exists()
    assert (skills_dir / "good_skill.py").exists()

    from lox.lang.validator import PolicyValidationError

    with pytest.raises(
        (ImportError, PolicyValidationError), match="skills.nonexistent"
    ):
        compile_policy_package(
            "import skills.nonexistent\nclass Policy:\n    def step(self, state): return Action.WAIT\n",
            skills=built,
        ).create_runner()

    # 3. Verify swallowed leaf errors fail verify_policy and dry_run_policy
    db_path = str(tmp_path / "critic.duckdb")
    init_db(db_path).close()
    meta = SwarmMetaAgent(
        provider="mock",
        db_path=db_path,
        theories_path=str(tmp_path / "theories.lance"),
        frontier_dir=str(tmp_path / "frontier"),
    )
    crashing_bt_policy = """
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector

def bad_branch(state, bb):
    return 1 / 0

def fallback_branch(state, bb):
    return Action.SEARCH

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        root = PrioritySelector([
            ActionLeaf(bad_branch, name="BadLeaf"),
            ActionLeaf(fallback_branch, name="FallbackLeaf"),
        ], name="MasterSelector")
        self.tree = BehaviorTree(root=root, blackboard=self.blackboard)

    def step(self, state):
        return self.tree.tick(state)
"""
    ok, diag = meta.verify_policy(crashing_bt_policy, steps=5)
    assert ok is False
    assert "BadLeaf" in diag and "ZeroDivisionError" in diag

    # 4. PrioritySelector AST branch extraction & cross-gen diagnostic digest
    sample_policy = """
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector
from skills.good_skill import good_act

def guard_act(state, bb):
    return None

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        root = PrioritySelector([
            ActionLeaf(guard_act, name="GuardLeaf"),
            ActionLeaf(good_act, name="GoodSkillLeaf"),
        ], name="MasterSelector")
        self.tree = BehaviorTree(root=root, blackboard=self.blackboard)

    def step(self, state):
        return self.tree.tick(state) or Action.SEARCH
"""
    branches = extract_priority_selector_branches(sample_policy)
    assert len(branches) == 2
    assert branches[0]["leaf_name"] == "GuardLeaf"
    assert branches[0]["fn_name"] == "guard_act"
    assert branches[1]["leaf_name"] == "GoodSkillLeaf"
    assert branches[1]["fn_name"] == "good_act"

    # 5. Populate 2 generations in DuckDB and test Two-Pass Critic + Historical Skill Lookup
    logger = BatchDatabaseLogger(db_path=db_path)
    logger.save_policy(
        policy_id="policy_swarm_g1",
        generation=1,
        mode="SWARM",
        policy_code=sample_policy,
        macros_code="",
        params={},
        avg_depth=2.5,
        max_depth=5,
        is_promoted=True,
        skills={"good_skill": built["good_skill"]},
    )
    con = connect_db(db_path, read_only=False)
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, avg_depth, p90_depth, max_depth, avg_score, max_score, avg_turns, hypothesis_passed, created_at)
        VALUES ('b1', 1, 'SWARM', 'policy_swarm_g1', 2.5, 4.0, 5, 300.0, 500, 1200.0, TRUE, CURRENT_TIMESTAMP)
        """
    )
    con.execute(
        """
        INSERT INTO function_stats (batch_id, generation, function_name, episodes_triggered, total_calls, total_turns_active, stagnation_share, livelock_score, liblit_failure_score, ochiai_good_run_score)
        VALUES ('b1', 1, 'good_act', 8, 500, 500, 0.05, 0.01, 0.02, 0.80),
               ('b1', 1, 'doors_open', 6, 40, 40, 0.02, 0.0, 0.0, 0.65)
        """
    )
    logger.save_policy(
        policy_id="policy_swarm_g2",
        generation=2,
        mode="SWARM",
        policy_code=sample_policy,
        macros_code="",
        params={},
        avg_depth=1.4,
        max_depth=2,
        is_promoted=True,
        skills={"good_skill": "# modified in g2\n"},
    )
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, avg_depth, p90_depth, max_depth, avg_score, max_score, avg_turns, hypothesis_passed, created_at)
        VALUES ('b2', 2, 'SWARM', 'policy_swarm_g2', 1.4, 2.0, 2, 110.0, 200, 900.0, TRUE, CURRENT_TIMESTAMP)
        """
    )
    con.execute(
        """
        INSERT INTO function_stats (batch_id, generation, function_name, episodes_triggered, total_calls, total_turns_active, stagnation_share, livelock_score, liblit_failure_score, ochiai_good_run_score)
        VALUES ('b2', 2, 'guard_act', 8, 850, 850, 0.92, 0.45, 0.70, 0.10),
               ('b2', 2, 'good_act', 8, 50, 0, 0.0, 0.0, 0.0, 0.0)
        """
    )
    con.close()

    digest = build_swarm_diagnostic_digest(
        db_path, current_policy=sample_policy, target_generation=2
    )
    assert "PEAK_DEPTH" in digest
    assert "HIGH_STAGNATION" in digest
    assert "0-TURN" in digest
    assert "doors_open" in digest

    critic = SwarmCriticAgent(
        provider="mock",
        db_path=db_path,
        theories_store=meta.theories,
    )
    assert "dry_run_policy" not in READ_ONLY_CRITIC_TOOLS
    assert not hasattr(critic, "evaluate_and_gate")

    pre_guide = critic.guide_before_generation(
        generation=2,
        current_policy=sample_policy,
        skills=built,
    )
    assert pre_guide.generation == 2
    assert "Critic Pre-Generation High-Level Guide" in pre_guide.to_prompt_block()

    post_review = critic.review_after_assembly(
        generation=2,
        batch_id="b2",
        candidate_policy_id="policy_swarm_g2",
        current_policy=sample_policy,
        skills=built,
    )
    assert post_review.generation == 2
    assert post_review.candidate_policy_id == "policy_swarm_g2"
    assert "Critic Post-Assembly Comparative Review" in post_review.to_prompt_block()

    # Historical skill source lookup via policy_id
    hist_skill, summary = critic.author.tools.read_skill(
        "good_skill", policy_id="policy_swarm_g1"
    )
    assert "def good_act" in hist_skill
    assert "policy_swarm_g1" in summary

    diff_txt, _ = critic.author.tools.diff_policies(
        "policy_swarm_g1", "policy_swarm_g2"
    )
    assert "skills/good_skill.py" in diff_txt

    # Verify Meta-Agent post-critic revision returns unchanged draft when no changes needed
    rev_code, rev_skills, changed = meta.revise_policy_with_critic(
        generation=2,
        draft_policy=sample_policy,
        skills_dir=skills_dir,
        skills=built,
        post_critique=post_review,
        pre_gen_guide=pre_guide,
    )
    assert changed is False
    assert rev_code == sample_policy
    assert extract_runner_leaf_errors(None) == []

    # 6. Verify Gen 1 empty-DB check allows `AS runs` / `AS ticks` aliases on expert_ticks while blocking `FROM runs`
    empty_db = str(tmp_path / "empty_gen1.duckdb")
    init_db(empty_db).close()
    critic_g1 = SwarmCriticAgent(provider="mock", db_path=empty_db)
    critic_g1.author.tools.generation = 1
    out_ok, sum_ok = critic_g1.author.tools.query_db(
        "SELECT depth, COUNT(DISTINCT run_id) AS runs, COUNT(*) AS ticks FROM expert_ticks GROUP BY depth"
    )
    assert sum_ok != "error", out_ok
    out_err, sum_err = critic_g1.author.tools.query_db(
        "SELECT end_reason, COUNT(*) AS n FROM runs GROUP BY end_reason"
    )
    assert sum_err == "error"
    assert "disallowed" in out_err

    # 7. Verify require_json nudges past Turn-1 plain-text planning in run_tool_session
    nudge_calls = {"n": 0}

    def fake_nudge_llm(messages, allow_tools=True, tool_specs=None):
        nudge_calls["n"] += 1
        if nudge_calls["n"] == 1:
            return {
                "content": "Investigate: diff g8 vs g9, look at runs, function stats.",
                "tool_calls": [],
            }
        return {
            "content": '{"needs_revision": false, "comparison_to_prev_gens": "ok", "identified_issues": [], "recommendations_for_meta": []}',
            "tool_calls": [],
        }

    critic_g1.author._call_llm_api = fake_nudge_llm  # type: ignore[method-assign]
    nudged_out = critic_g1.author.run_tool_session(
        "sys",
        "usr",
        max_tool_turns=3,
        require_json=True,
        expected_json_type="object",
    )
    assert nudge_calls["n"] == 2
    assert '"needs_revision": false' in nudged_out

    # 8. Verify prune_unused_skills removes orphaned files while preserving transitive imports
    prune_dir = tmp_path / "prune_skills"
    prune_dir.mkdir(parents=True)
    for nm, code_s in {
        "used_a": "from skills.used_helper import h\n",
        "used_helper": "h = 1\n",
        "orphaned_old": "x = 2\n",
    }.items():
        (prune_dir / f"{nm}.py").write_text(code_s, encoding="utf-8")
    kept = SwarmMetaAgent.prune_unused_skills(
        "from skills.used_a import h\nclass Policy:\n    pass\n",
        {
            "used_a": "from skills.used_helper import h\n",
            "used_helper": "h = 1\n",
            "orphaned_old": "x = 2\n",
        },
        skills_dir=prune_dir,
    )
    assert sorted(kept.keys()) == ["used_a", "used_helper"]
    assert not (prune_dir / "orphaned_old.py").exists()
    assert (prune_dir / "used_helper.py").exists()


def test_democratized_peer_swarm_incubation_and_crn_tournament(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate
    from lox.db.logger import BatchDatabaseLogger
    from lox.engine.swarm_referee import SwarmReferee

    db_path = str(tmp_path / "peer_swarm.duckdb")
    con = init_db(db_path)
    con.close()

    base_pol = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector\n"
        "from skills.skill_a import fn_a\n"
        "from skills.skill_b import fn_b\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(\n"
        "            PrioritySelector([\n"
        "                ActionLeaf(fn_a, name='leaf_a'),\n"
        "                ActionLeaf(fn_b, name='leaf_b'),\n"
        "            ]),\n"
        "            blackboard=self.blackboard,\n"
        "        )\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    base_skills = {
        "skill_a": "from lox.env.actions import Action\ndef fn_a(st, bb):\n    return None\n",
        "skill_b": "from lox.env.actions import Action\ndef fn_b(st, bb):\n    return Action.SEARCH\n",
    }

    # Log batch with 1 deep outlier run using leaf_b + 3 shallow stalled runs (leaf_a unused -> dormant)
    logger = BatchDatabaseLogger(db_path)
    logger.save_batch(
        batch_id="b_g1",
        generation=1,
        mode="SWARM",
        parent_policy_id="policy_swarm_g0",
        candidate_policy_id="policy_swarm_g1",
        target_functions=["leaf_b"],
        runs=[
            {
                "run_id": "r_good",
                "seed": 1000,
                "depth": 3,
                "max_depth": 3,
                "score": 450,
                "turns": 600,
                "steps": 600,
                "end_reason": "killed by grid bug",
                "unique_tiles": 140,
                "pct_search": 0.10,
                "triggered_functions": {"leaf_b": 25},
                "fn_turns": {"leaf_b": 600},
                "summary_digest": "Overview: Depth=3",
                "milestones": {"stall_count": 0, "max_depth": 3},
            },
            *[
                {
                    "run_id": f"r_stall_{i}",
                    "seed": 1001 + i,
                    "depth": 1,
                    "max_depth": 1,
                    "score": 15,
                    "turns": 500,
                    "steps": 500,
                    "end_reason": "max_steps",
                    "unique_tiles": 22,
                    "pct_search": 0.85,
                    "triggered_functions": {},
                    "fn_turns": {},
                    "summary_digest": "CONFINEMENT_STALL on Dlvl 1",
                    "milestones": {"stall_count": 1, "max_depth": 1},
                }
                for i in range(3)
            ],
        ],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.5, "max_depth": 3, "avg_score": 120.0},
        verdict={"passed": True},
    )

    analyzer = DeficitAnalyzer(db_path)
    report = analyzer.analyze(generation=2, current_policy=base_pol, skills=base_skills)
    assert report.fail_clusters
    assert "leaf_a" in report.dormant_branches
    assert any(b["fn"] == "leaf_b" for b in report.protected_branches)

    # Bad candidate with leaf exception is rejected in isolated sandbox incubation
    bad_cand = PeerCandidate(
        peer_id="peer_bad",
        claimed_deficit_id="confinement_stall",
        claimed_label="stall",
        policy_code=base_pol,
        skills={
            "skill_a": "def fn_a(st, bb):\n    raise RuntimeError('boom')\n",
            "skill_b": base_skills["skill_b"],
        },
    )
    bad_res = incubate_candidate(
        bad_cand, deficit_report=report, claimed_deficit=DeficitCluster("c", "k", "l")
    )
    assert bad_res.sandbox_passed is False
    assert "Swallowed leaf error" in bad_res.error

    # Disjoint skill updates merge cleanly via SwarmReferee.build_orthogonal_pair
    cand_a = PeerCandidate(
        peer_id="peer_1",
        claimed_deficit_id="c1",
        claimed_label="c1",
        policy_code=base_pol,
        skills={
            "skill_a": "from lox.env.actions import Action\ndef fn_a(st, bb):\n    return Action.EAST\n",
            "skill_b": base_skills["skill_b"],
        },
        touched_skills={"skill_a"},
        touched_branches=set(),
        sandbox_passed=True,
        sandbox_metrics={"depth_delta": 0.2, "score_delta": 10.0, "wrapper_ratio": 0.0},
    )
    cand_b = PeerCandidate(
        peer_id="peer_2",
        claimed_deficit_id="c2",
        claimed_label="c2",
        policy_code=base_pol,
        skills={
            "skill_a": base_skills["skill_a"],
            "skill_b": "from lox.env.actions import Action\ndef fn_b(st, bb):\n    return Action.WAIT\n",
        },
        touched_skills={"skill_b"},
        touched_branches=set(),
        sandbox_passed=True,
        sandbox_metrics={"depth_delta": 0.1, "score_delta": 5.0, "wrapper_ratio": 0.0},
    )
    referee = SwarmReferee()
    ortho = referee.build_orthogonal_pair([cand_a, cand_b], base_pol, base_skills)
    assert ortho is not None
    assert "Action.EAST" in ortho.skills["skill_a"]
    assert "Action.WAIT" in ortho.skills["skill_b"]

    # Fake pool verifying tournament rejects strictly regressive candidate and keeps incumbent
    class _FakePool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            is_inc = "Action.SEARCH" in (skills or {}).get("skill_b", "")
            depth = 2 if is_inc else 1
            score = 50.0 if is_inc else 5.0
            return {
                "batch_id": batch_id,
                "avg_depth": float(depth),
                "max_depth": depth,
                "avg_score": score,
                "p90_depth": float(depth),
                "max_score": int(score),
                "avg_turns": 10.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": depth,
                        "max_depth": depth,
                        "score": int(score),
                        "turns": 10,
                        "steps": 10,
                        "end_reason": "max_steps",
                        "fn_turns": {"leaf_b": 10},
                        "summary_digest": "ok",
                    }
                    for sd in priority_seeds
                ],
            }

    verdict = referee.evaluate_tournament(
        generation=2,
        candidates=[cand_b],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_FakePool(),
        seeds=[1000, 1001],
    )
    assert verdict.promoted is False
    assert verdict.winner_id == "incumbent"
    assert verdict.winner_policy_code == base_pol

    # Verify score-collapse guard: +0.25 depth gain (2.75 -> 3.00) with >30% score loss (700 -> 400) is rejected
    class _ScoreCollapsePool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            is_inc = "Action.SEARCH" in (skills or {}).get("skill_b", "")
            avg_d = 2.75 if is_inc else 3.00
            avg_s = 700.0 if is_inc else 400.0
            return {
                "batch_id": batch_id,
                "avg_depth": avg_d,
                "max_depth": 3,
                "avg_score": avg_s,
                "p90_depth": 3.0,
                "max_score": int(avg_s),
                "avg_turns": 100.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": 3 if (not is_inc or idx < 3) else 2,
                        "max_depth": 3 if (not is_inc or idx < 3) else 2,
                        "score": int(avg_s),
                        "turns": 100,
                        "steps": 100,
                        "end_reason": "max_steps",
                        "fn_turns": {"leaf_b": 100},
                        "summary_digest": "ok",
                    }
                    for idx, sd in enumerate(priority_seeds)
                ],
            }

    collapse_verdict = referee.evaluate_tournament(
        generation=3,
        candidates=[cand_b],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_ScoreCollapsePool(),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert collapse_verdict.promoted is False
    assert collapse_verdict.winner_id == "incumbent"

    # Verify incumbent CRN cache avoids re-evaluating identical (pool, policy, seeds) across gens
    eval_calls: list[str] = []

    class _CountingPool:
        def evaluate_batch(
            self,
            batch_id,
            policy_code,
            macros_code,
            skills,
            target_episodes,
            priority_seeds,
            record_ticks=True,
            max_steps=None,
        ):
            eval_calls.append(batch_id)
            return {
                "batch_id": batch_id,
                "avg_depth": 2.0,
                "max_depth": 2,
                "avg_score": 100.0,
                "p90_depth": 2.0,
                "max_score": 100,
                "avg_turns": 50.0,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{sd}",
                        "seed": sd,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 100,
                        "turns": 50,
                        "steps": 50,
                        "end_reason": "max_steps",
                        "fn_turns": {"leaf_b": 50},
                        "summary_digest": "ok",
                    }
                    for sd in priority_seeds
                ],
            }

    c_pool = _CountingPool()
    referee.evaluate_tournament(
        generation=4,
        candidates=[cand_a],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
    )
    assert len(eval_calls) == 2  # Incumbent once + cand_a once
    eval_calls.clear()
    referee.evaluate_tournament(
        generation=5,
        candidates=[cand_a],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
    )
    assert len(eval_calls) == 1  # Incumbent served from CRN cache!

    # Increasing max_steps tier invalidates cached incumbent cleanly
    eval_calls.clear()
    referee.evaluate_tournament(
        generation=6,
        candidates=[cand_a],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=c_pool,
        seeds=[1000, 1001],
        max_steps=10000,
    )
    assert len(eval_calls) == 2

    # Verify FrontierCheckpoint replay during incubate_candidate catches deep-state errors
    from lox.engine.frontier import FrontierCheckpoint

    deep_cp = FrontierCheckpoint(
        checkpoint_id="fc_test_d3",
        seed=1000,
        depth=3,
        milestone="dlvl_3",
        turns=50,
        action_bytes=bytes([ord("s")] * 20),
    )
    ok_res = incubate_candidate(
        cand_a,
        deficit_report=report,
        claimed_deficit=DeficitCluster("c", "k", "l"),
        frontier_checkpoints=[deep_cp],
    )
    assert ok_res.sandbox_passed is True
    assert "frontier_survived" in ok_res.sandbox_metrics

    # Verify bloat_warning flags large policy_loc WITHOUT polluting main peer fail_clusters
    large_pol = base_pol + "\n" + "\n".join(f"# pad {i}" for i in range(410))
    big_report = analyzer.analyze(
        generation=6, current_policy=large_pol, skills=base_skills
    )
    assert big_report.bloat_warning is True
    assert not any(c.cluster_id == "slice_modularize" for c in big_report.fail_clusters)
    assert "DITCH MONOLITH" in big_report.format_brief()


def test_swarm_peer_multi_skill_surgical_splice_and_placeholder_guard(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.author.tools import AuthorToolRegistry

    base_policy = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n\n"
        "def leaf_wait(state, bb):\n"
        "    return Action.WAIT\n\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(leaf_wait, name='wait')], name='root')\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(root=build_tree(self.blackboard), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "t.duckdb"))
    tools.configure_generation(mode="SWARM_PEER", generation=2)
    tools.current_policy_code = base_policy

    # 1. Surgical patch (omits class Policy) splices cleanly into current_policy_code
    res_sp, sum_sp = tools.dry_run_policy(
        code="def leaf_wait(state, bb):\n    return Action.SEARCH\n",
        max_turns=20,
        smoke_only=True,
    )
    assert sum_sp.startswith("PASSED"), res_sp
    assert "class Policy:" in tools.last_passed_code
    assert "Action.SEARCH" in tools.last_passed_code

    # 2. Multi-skill validation (`skills={"mod_a": ..., "mod_b": ...}`) compiles and runs cleanly
    thin_policy = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n"
        "from skills.mod_a import fn_a\n"
        "from skills.mod_b import fn_b\n\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(fn_a, name='a'), ActionLeaf(fn_b, name='b')], name='root')\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(root=build_tree(self.blackboard), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    skills_bundle = {
        "mod_a": "def fn_a(state, bb):\n    return None\n",
        "mod_b": "from lox.env.actions import Action\ndef fn_b(state, bb):\n    return Action.SEARCH\n",
    }
    res_ms, sum_ms = tools.dry_run_policy(
        code=thin_policy,
        skills=skills_bundle,
        max_turns=20,
        smoke_only=True,
    )
    assert sum_ms.startswith("PASSED"), res_ms
    assert set(tools.last_passed_skills.keys()) >= {"mod_a", "mod_b"}

    # 3. Comment-only closing placeholders must NOT pass _inspect_python_fragment
    has_stmts, has_cls = AuthorAgent._inspect_python_fragment(
        "# The complete class Policy: source is already validated above"
    )
    assert has_stmts is False and has_cls is False


def test_prune_empty_skills_unlinks_disk_and_deficit_pct_scale(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_meta import SwarmMetaAgent
    from lox.db.schema import connect_db, init_db

    # 1. prune_unused_skills must unlink orphaned disk files even when skills={}
    sk_dir = tmp_path / "skills"
    sk_dir.mkdir()
    stale = sk_dir / "swarm_core.py"
    stale.write_text("x = 1\n", encoding="utf-8")
    kept = SwarmMetaAgent.prune_unused_skills(
        "class Policy:\n    pass\n", {}, skills_dir=sk_dir
    )
    assert kept == {}
    assert not stale.exists()

    # 2. pct_wait=20.0 (20%) must NOT trip slice_stalled (0..100 scale, threshold >= 65.0)
    db_p = str(tmp_path / "deficits.duckdb")
    init_db(db_p)
    con = connect_db(db_p, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, avg_depth, max_depth) VALUES ('b1', 1, 'SWARM', 3.0, 6)"
        )
        for idx in range(6):
            er = "killed" if idx < 4 else "stalled_zero_turn"
            pw = 20.0 if idx < 4 else 70.0
            fn = "fight_adjacent" if idx < 2 else "explore_frontier"
            con.execute(
                """
                INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, steps,
                                  end_reason, pct_wait, pct_search, last_active_function)
                VALUES (?, 'b1', 1, ?, 'val', ?, ?, 200, ?, 300, ?, ?, 5.0, ?)
                """,
                [
                    f"r{idx}",
                    1000 + idx,
                    4 if idx < 2 else 1,
                    4 if idx < 2 else 1,
                    500 if idx < 2 else 120,
                    er,
                    pw,
                    fn,
                ],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(db_p).analyze(2)
    cids = {c.cluster_id for c in rep.fail_clusters}
    assert "slice_stalled" in cids
    assert "slice_frontier_failures" in cids or "slice_early_terminations" in cids
    stalled = next(c for c in rep.fail_clusters if c.cluster_id == "slice_stalled")
    assert stalled.run_count == 2


def test_peer_swarm_records_theories_to_lancedb(tmp_path):
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import SwarmPeerAgent
    from lox.db.theories import TheoryDocumentStore

    store = TheoryDocumentStore(str(tmp_path / "theories.lance"))
    peer = SwarmPeerAgent(
        peer_id="peer_1",
        provider="mock",
        db_path=str(tmp_path / "lox.duckdb"),
        theories_store=store,
    )
    rep = DeficitReport(generation=1, regime="BOOTSTRAP")
    cluster = DeficitCluster(
        cluster_id="slice_stalled_wait_fallback",
        kind="stalled_trajectory",
        label="Stall in wait_fallback",
        evidence="wait_fallback starvation faint loop",
    )
    peer.run_cycle(
        generation=1,
        deficit_report=rep,
        claimed_deficit=cluster,
        current_policy="",
        current_skills={},
    )
    hits = store.search_theories("wait_fallback starvation", doc_type="THEORY")
    assert len(hits) >= 1
    assert hits[0]["agent_role"] == "peer_1"
    assert "slice_stalled_wait_fallback" in hits[0]["content"]


def test_ast_modularizer_hysteresis_incremental_and_orchestrator(tmp_path):
    import json

    from lox.db.schema import connect_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.engine.swarm_referee import _package_fingerprint
    from lox.lang.modularize import (
        maybe_auto_modularize,
        modularize_policy_source,
        verify_bit_identical,
    )

    monolith = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n\n"
        "SHARED_CONST = 1\n\n"
        "def _shared_ok(state):\n"
        "    return state.hp >= SHARED_CONST\n\n"
        "def leaf_one(state, bb):\n"
        "    if not _shared_ok(state):\n"
        "        return Action.WAIT\n"
        "    return None\n\n"
        "def leaf_two(state, bb):\n"
        "    if _shared_ok(state) and state.step_count % 2 == 0:\n"
        "        return Action.SEARCH\n"
        "    return None\n\n"
        "def leaf_three(state, bb):\n"
        "    return Action.WAIT if _shared_ok(state) else Action.SEARCH\n\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([\n"
        "        ActionLeaf(leaf_one, name='l1'),\n"
        "        ActionLeaf(leaf_two, name='l2'),\n"
        "        ActionLeaf(leaf_three, name='l3'),\n"
        "    ], name='root')\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(root=build_tree(self.blackboard), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    # 1. Fresh monolith split + bit-identical check
    new_pol, new_skills = modularize_policy_source(monolith, num_modules=2)
    assert len(new_pol.splitlines()) < len(monolith.splitlines())
    assert "shared" in new_skills
    assert len(new_skills) >= 2
    verify_bit_identical(
        monolith, {}, new_pol, new_skills, seeds=[1000, 1001], steps=50
    )

    # 2. Below threshold hysteresis check skips splitting
    ch_skip, _, _, _ = maybe_auto_modularize(
        monolith, {}, verify_seeds=[1000], verify_steps=50
    )
    assert ch_skip is False

    # 3. Incremental split when existing `skills/` + imports are already present
    augmented = new_pol.replace(
        "def build_tree(bb):",
        "EXTRA_CONST = 2\n\n"
        "def _extra_helper(state):\n    return state.step_count >= EXTRA_CONST\n\n"
        "def leaf_four(state, bb):\n    return Action.SEARCH if _extra_helper(state) else None\n\n"
        "def leaf_five(state, bb):\n    return Action.WAIT if _extra_helper(state) else None\n\n"
        "def build_tree(bb):",
    ).replace(
        "ActionLeaf(leaf_three, name='l3'),",
        "ActionLeaf(leaf_three, name='l3'),\n        ActionLeaf(leaf_four, name='l4'),\n        ActionLeaf(leaf_five, name='l5'),",
    )
    inc_pol, inc_skills = modularize_policy_source(
        augmented, num_modules=2, existing_skills=new_skills
    )
    assert set(new_skills.keys()).issubset(set(inc_skills.keys()))
    assert inc_skills["shared"] == new_skills["shared"]
    assert len(inc_skills) > len(new_skills)
    verify_bit_identical(
        augmented, new_skills, inc_pol, inc_skills, seeds=[1000, 1001], steps=50
    )

    # 4. SwarmCampaignOrchestrator auto-modularizes a bloated monolith in _run_generation_peer
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    padded_helpers = "\n".join(
        f"def _noop_helper_{i}(state):\n    return state.hp > {i}\n" for i in range(135)
    )
    bloated_monolith = monolith.replace(
        "def build_tree(bb):", f"{padded_helpers}\n\ndef build_tree(bb):"
    )
    (ws_dir / "policy.py").write_text(bloated_monolith, encoding="utf-8")

    db_path = str(tmp_path / "swarm_mod.duckdb")
    orch = SwarmCampaignOrchestrator(
        db_path=db_path,
        workspace_dir=str(ws_dir),
        frontier_dir=str(tmp_path / "frontier"),
        provider="mock",
        num_workers=2,
        eval_workers=2,
        seeds_per_batch=2,
        swarm_architecture="peer",
    )
    try:
        res = orch.run_generation(1)
        assert res["promoted"] is True
        written_pol = (ws_dir / "policy.py").read_text(encoding="utf-8")
        assert len(written_pol.splitlines()) < 60
        skill_files = {
            p.stem: p.read_text(encoding="utf-8")
            for p in (ws_dir / "skills").glob("*.py")
        }
        assert "shared" in skill_files and len(skill_files) >= 2
        con = connect_db(db_path, read_only=True)
        try:
            row = con.execute(
                "SELECT skills_json FROM policies WHERE generation=1"
            ).fetchone()
            assert row is not None and len(json.loads(row[0])) == len(skill_files)
        finally:
            con.close()
        fp = _package_fingerprint(written_pol, skill_files)
        assert fp in orch._attempted_ast_modular_fps
    finally:
        orch.close()


def test_gen41_xml_bundle_and_partial_skill_patch_preservation(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.author.tools import (
        OPENAI_TOOL_SPECS,
        AuthorToolRegistry,
        merge_skill_module_update,
    )

    # 1. OpenAI schema omits untyped additionalProperties `skills` dict
    dry_run_spec = next(
        s for s in OPENAI_TOOL_SPECS if s["function"]["name"] == "dry_run_policy"
    )
    props = dry_run_spec["function"]["parameters"]["properties"]
    assert "skills" not in props
    assert "<skill_code" in props["code"]["description"]

    # 2. Partial # PATCH update to existing skill preserves untouched exports (_delta_of)
    existing_shared = (
        "from lox.env.actions import Action\n\n"
        "def _delta_of(a, b):\n"
        "    return (b[0] - a[0], b[1] - a[1])\n\n"
        "def choose_move(state, bb):\n"
        "    return Action.WAIT\n"
    )
    partial_patch = (
        "# PATCH\n"
        "def choose_move(state, bb):\n"
        "    dx, dy = _delta_of((0, 0), (1, 0))\n"
        "    return Action.SEARCH if dx == 1 and dy == 0 else Action.WAIT\n"
    )
    merged = merge_skill_module_update(existing_shared, partial_patch)
    assert "def _delta_of(a, b):" in merged
    assert "return Action.SEARCH" in merged

    # 3. dry_run_policy unpacks XML bundle (<skill_code> + <policy_code>) and keeps _delta_of
    base_policy = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n"
        "from skills.shared import _delta_of, choose_move\n\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(choose_move, name='mv')], name='root')\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(root=build_tree(self.blackboard), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        _ = _delta_of((0, 0), (0, 1))\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "xml.duckdb"))
    tools.configure_generation(
        mode="SWARM_PEER",
        generation=41,
        skills={"shared": existing_shared},
    )
    tools.current_policy_code = base_policy

    xml_skill_only = (
        '<skill_code module="shared">\n'
        "# PATCH: update choose_move without deleting _delta_of\n"
        "def choose_move(state, bb):\n"
        "    dx, _ = _delta_of((0, 0), (1, 0))\n"
        "    return Action.SEARCH if dx == 1 else Action.WAIT\n"
        "</skill_code>\n"
    )
    res_x, sum_x = tools.dry_run_policy(
        code=xml_skill_only,
        max_turns=20,
        smoke_only=True,
    )
    assert sum_x.startswith("PASSED"), res_x
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]
    assert "Action.SEARCH" in tools.last_passed_skills["shared"]

    # 4. XML bundle with both <skill_code> and <policy_code> splice
    xml_both = (
        '<skill_code module="shared">\n'
        "def choose_move(state, bb):\n"
        "    return Action.SEARCH\n"
        "</skill_code>\n"
        "<policy_code>\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(choose_move, name='mv2')], name='root')\n"
        "</policy_code>\n"
    )
    res_b, sum_b = tools.dry_run_policy(code=xml_both, max_turns=20, smoke_only=True)
    assert sum_b.startswith("PASSED"), res_b
    assert "mv2" in tools.last_passed_code
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]

    # 5. AuthorAgent._execute_tool_call rescues <skill_code> emitted in assistant_text
    agent = AuthorAgent(db_path=str(tmp_path / "xml.duckdb"), provider="mock")
    agent.tools = tools
    out = agent._execute_tool_call(
        turn_idx=1,
        name="dry_run_policy",
        args={"code": "", "max_turns": 20},
        assistant_text=xml_skill_only,
    )
    assert "SMOKE" in out or "Compile" in out or "PASSED" in out or "Pilot" in out
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]


def test_gen41_zero_row_sql_hints_and_deficit_telemetry_window(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
    from lox.author.tools import AuthorToolRegistry

    db_path = str(tmp_path / "hints.duckdb")
    con = init_db(db_path)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, seeds_evaluated, avg_score, avg_depth, p90_depth, max_depth, avg_turns)
            VALUES ('b14', 14, 'SWARM', 'pol_14', 4, 10.0, 1.0, 1, 1, 50.0),
                   ('b40', 40, 'SWARM', 'pol_40', 4, 50.0, 2.0, 2, 2, 120.0)
            """
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, turns, end_reason)
            VALUES ('run_g14_s1', 'b14', 14, 1, 'val-hum-law-fem', 1, 1, 50, 'died'),
                   ('run_g40_s1', 'b40', 40, 1, 'val-hum-law-fem', 2, 2, 120, 'died')
            """
        )
        con.execute(
            """
            INSERT INTO ticks (run_id, step_idx, turn, depth, hp, max_hp, action_name, active_function)
            VALUES ('run_g40_s1', 0, 1, 1, 12, 12, 'WAIT', 'leaf_a')
            """
        )
    finally:
        con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    tools.configure_generation(mode="SWARM_PEER", generation=41)

    q_future, sum_future = tools.query_db(
        "SELECT run_id FROM runs WHERE generation = 41"
    )
    assert sum_future == "0 rows" and "results: []" in q_future
    assert "40" in q_future and "champion_runs" in q_future

    q_pruned, sum_pruned = tools.query_db(
        "SELECT * FROM ticks WHERE run_id = 'run_g14_s1'"
    )
    assert sum_pruned == "0 rows" and "results: []" in q_pruned
    assert "champion_ticks" in q_pruned and "40" in q_pruned

    analyzer = DeficitAnalyzer(db_path=db_path)
    report = analyzer.analyze(generation=41, current_policy="")
    assert report.latest_completed_gen == 40
    assert report.ticks_min_gen == 40
    report.fail_clusters = [
        DeficitCluster(
            cluster_id="c_test",
            kind="early_death",
            label="Early combat death",
            run_count=2,
            avg_depth=1.5,
            avg_turns=80.0,
            sample_run_ids=["run_g40_s1"],
            dominant_fns=["leaf_a"],
            evidence="hp dropped",
        )
    ]
    brief = report.format_brief()
    assert "latest_completed_gen=40" in brief
    assert "Gen 41 is currently running" in brief
    assert "run_ids=['run_g40_s1']" in brief


def test_gen41_prompt_grounding_and_xml_instructions():
    from lox.author.prompts import build_system_prompt, build_user_prompt

    sys_prompt = build_system_prompt(is_swarm=True)
    assert "champion_runs" in sys_prompt
    assert "champion_ticks" in sys_prompt
    assert "champion_function_stats" in sys_prompt

    usr_prompt = build_user_prompt(
        mode="SWARM_PEER",
        generation=41,
        current_policy="class Policy:\n    pass\n",
        batch_digest="Gen 40 summary",
    )
    assert "query_wiki" in usr_prompt
    assert "query_theories" in usr_prompt
    assert "Recent Experiment & Rejection Memory" in usr_prompt
    assert '<skill_code module="name">' in usr_prompt
    assert "<policy_code>" in usr_prompt
    assert "skills={{" not in usr_prompt


def test_topological_skill_import_order_before_shared():
    from lox.lang.compiler import compile_bt_subtree, compile_policy_package

    skills = {
        "aaa_guard": (
            "from skills.equip import pick_action\n\n"
            "def guard_fn(state, bb):\n"
            "    return pick_action(state, bb)\n"
        ),
        "equip": (
            "from lox.env.actions import Action\n"
            "from skills.shared import _delta_of\n\n"
            "def pick_action(state, bb):\n"
            "    return Action.SEARCH if _delta_of(Action.N) == (0, -1) else Action.WAIT\n"
        ),
        "shared": (
            "from lox.env.actions import DIR_DELTAS\n\n"
            "def _delta_of(act):\n"
            "    return DIR_DELTAS.get(act)\n"
        ),
    }
    policy_src = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n"
        "from skills.aaa_guard import guard_fn\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(\n"
        "            PrioritySelector([ActionLeaf(guard_fn, name='g')]),\n"
        "            blackboard=self.blackboard,\n"
        "        )\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    compiled = compile_policy_package(policy_src, "", skills=skills)
    assert compiled._skill_order == ["shared", "equip", "aaa_guard"]
    runner = compiled.create_runner()
    assert runner is not None

    subtree = compile_bt_subtree(
        "from lox.lang.bt import ActionLeaf\n"
        "from skills.aaa_guard import guard_fn\n"
        "subtree = ActionLeaf(guard_fn, name='g_sub')\n",
        skills=skills,
    )
    assert subtree is not None


def test_swarm_referee_blocks_hard_stall_regression_on_minor_depth_tick():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )

    class _StubPool:
        def __init__(self, responses):
            self.responses = responses

        def evaluate_batch(self, batch_id, policy_code, **kwargs):
            for key, val in self.responses.items():
                if key in batch_id:
                    return val
            return self.responses["inc"]

    inc_runs = [
        {
            "run_id": f"inc_{i}",
            "seed": 1000 + i,
            "depth": 4 if i < 3 else 5,
            "max_depth": 4 if i < 3 else 5,
            "score": 600,
            "turns": 1500,
            "end_reason": "killed by grid bug",
            "summary_digest": "Overview: Depth=4",
        }
        for i in range(4)
    ]
    stalled_minor_runs = [
        {
            "run_id": f"st_{i}",
            "seed": 1000 + i,
            "depth": 5 if i < 2 else (3 if i == 2 else 4),
            "max_depth": 5 if i < 2 else (3 if i == 2 else 4),
            "score": 560,
            "turns": 400 if i == 0 else 1400,
            "end_reason": "stalled_zero_turn" if i == 0 else "killed by gnome",
            "summary_digest": "Overview: Stalled",
        }
        for i in range(4)
    ]
    clean_major_runs = [
        {
            "run_id": f"mj_{i}",
            "seed": 1000 + i,
            "depth": 5,
            "max_depth": 5,
            "score": 700,
            "turns": 1600,
            "end_reason": "killed by gnome",
            "summary_digest": "Overview: Depth=5",
        }
        for i in range(4)
    ]

    cand_stalled = PeerCandidate(
        peer_id="peer_stalled",
        claimed_deficit_id="c1",
        claimed_label="minor tick + stall",
        policy_code=base_pol + "\ndef _stalled_marker():\n    return 1\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )
    cand_clean = PeerCandidate(
        peer_id="peer_clean",
        claimed_deficit_id="c2",
        claimed_label="clean breakthrough",
        policy_code=base_pol + "\ndef _clean_marker():\n    return 2\n",
        skills={},
        sandbox_passed=True,
        loc_total=5,
    )

    referee = SwarmReferee()
    v1 = referee.evaluate_tournament(
        generation=52,
        candidates=[cand_stalled],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_StubPool(
            {"_inc_": {"runs": inc_runs}, "peer_stalled": {"runs": stalled_minor_runs}}
        ),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert v1.promoted is False

    v2 = SwarmReferee()
    v2_res = v2.evaluate_tournament(
        generation=52,
        candidates=[cand_stalled, cand_clean],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_StubPool(
            {
                "_inc_": {"runs": inc_runs},
                "peer_stalled": {"runs": stalled_minor_runs},
                "peer_clean": {"runs": clean_major_runs},
            }
        ),
        seeds=[1000, 1001, 1002, 1003],
    )
    assert v2_res.promoted is True
    assert v2_res.winner_id == "peer_clean"


def test_swarm_referee_no_overwrite_noop_rejected_filter_and_parallel_tasks():
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_referee import SwarmReferee

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    noop_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        # comment only\n"
        "        return Action.WAIT\n"
    )
    cand_noop = PeerCandidate(
        peer_id="peer_noop",
        claimed_deficit_id="c1",
        claimed_label="noop",
        policy_code=noop_pol,
        skills={},
        sandbox_passed=True,
        loc_total=4,
    )
    cand_worse = PeerCandidate(
        peer_id="peer_worse",
        claimed_deficit_id="c2",
        claimed_label="worse",
        policy_code=base_pol + "\ndef _extra(s):\n    return None\n",
        skills={},
        sandbox_passed=True,
        loc_total=6,
    )
    cand_better = PeerCandidate(
        peer_id="peer_better",
        claimed_deficit_id="c3",
        claimed_label="better",
        policy_code=base_pol.replace("Action.WAIT", "Action.SEARCH"),
        skills={},
        sandbox_passed=True,
        loc_total=4,
    )

    referee = SwarmReferee()
    # 1. No-op candidate is filtered even when pool has size 1
    assert referee.select_finalists([cand_noop], base_pol, {}) == []

    # 2. Evaluate incumbent + 2 challengers via parallel `evaluate_tasks`
    class _ParallelStubPool:
        role = "val-hum-law-fem"
        max_steps = 100
        benchmark = "nethack"

        def __init__(self):
            self.task_batches = []

        def evaluate_tasks(self, tasks):
            self.task_batches.append(len(tasks))
            out = []
            for tk in tasks:
                rid = tk["run_id"]
                sd = tk["seed"]
                if "peer_better" in rid:
                    d, sc = 5, 500
                elif "peer_worse" in rid:
                    d, sc = 1, 10
                else:
                    d, sc = 3, 200
                out.append(
                    {
                        "run_id": rid,
                        "seed": sd,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 100,
                        "end_reason": "died",
                        "summary_digest": f"Overview: Depth={d}",
                    }
                )
            return out

    par_pool = _ParallelStubPool()
    verdict_rej = referee.evaluate_tournament(
        generation=55,
        candidates=[cand_worse],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert par_pool.task_batches == [
        4
    ]  # incumbent (2) + peer_worse (2) dispatched together
    assert verdict_rej.promoted is False
    assert "inc_retained" in verdict_rej.winner_batch_eval["batch_id"]
    assert verdict_rej.winner_batch_eval["avg_depth"] == 3.0

    # Rejected fingerprint is never re-admitted against the same incumbent
    assert referee.select_finalists([cand_worse], base_pol, {}) == []

    # When 0 finalists remain, retained incumbent uses a fresh cloned batch_id (never overwrites prior batch)
    v_empty = referee.evaluate_tournament(
        generation=56,
        candidates=[cand_worse],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert v_empty.promoted is False
    assert (
        v_empty.winner_batch_eval["batch_id"]
        != verdict_rej.winner_batch_eval["batch_id"]
    )
    assert "g56_inc_retained" in v_empty.winner_batch_eval["batch_id"]

    # Clean superior challenger wins
    v_win = referee.evaluate_tournament(
        generation=57,
        candidates=[cand_better],
        base_policy=base_pol,
        base_skills={},
        eval_pool=par_pool,
        seeds=[1000, 1001],
    )
    assert v_win.promoted is True
    assert v_win.winner_id == "peer_better"


def test_deficit_analyzer_filters_rejected_renders_memory_and_vitals(tmp_path):
    from lox.author.swarm_deficits import DeficitAnalyzer

    db_path = str(tmp_path / "def_clean.duckdb")
    con = init_db(db_path)
    try:
        con.execute(
            """
            INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, hypothesis_passed, seeds_evaluated, avg_score, avg_depth, p90_depth, max_depth, avg_turns)
            VALUES ('b_prom_58', 58, 'SWARM', 'policy_swarm_g58', TRUE, 2, 650.0, 4.5, 5, 5, 1800.0),
                   ('b_rej_60', 60, 'SWARM', 'policy_swarm_g60', FALSE, 2, 100.0, 1.0, 1, 1, 50.0)
            """
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, final_ac, xp_level, end_reason, died_unarmored, milestones_json)
            VALUES ('r_prom', 'b_prom_58', 58, 1000, 'val-hum-law-fem', 5, 5, 650, 1800, 6, 4, 'killed by fire ant', FALSE, '{"unworn_armor_in_pack": true}'),
                   ('r_rej', 'b_rej_60', 60, 1000, 'val-hum-law-fem', 1, 1, 100, 50, 10, 1, 'stalled_bad_candidate', TRUE, '{}')
            """
        )
        con.execute(
            """
            INSERT INTO synthesis_logs (log_id, generation, mode, turn_index, turn_type, agent_role, hypothesis_yaml, status)
            VALUES ('s60', 60, 'SWARM_PEER', 8, 'SMOKE_CHECK', 'peer_4', 'hypothesis: H-wf-stall-fix failed', 'PASSED (smoke check)')
            """
        )
    finally:
        con.close()

    thin_pol = "\n".join(f"# l{i}" for i in range(45)) + "\nclass Policy:\n    pass\n"
    five_skills = {f"sk_{i}": "def fn():\n    return None\n" for i in range(5)}
    report = DeficitAnalyzer(db_path).analyze(
        generation=61, current_policy=thin_pol, skills=five_skills
    )
    assert report.bloat_warning is False
    assert "DITCH MONOLITH" not in report.format_brief()
    assert report.recent_attempts and "REJECTED" in report.recent_attempts[0]
    assert "H-wf-stall-fix" in report.recent_attempts[0]
    assert "avg_ac=6.0" in report.vital_summary
    brief = report.format_brief()
    assert "Recent Experiment & Rejection Memory" in brief
    assert "stalled_bad_candidate" not in brief
    assert any(
        c.cluster_id == "slice_unequipped_progression" for c in report.fail_clusters
    )


def test_prompt_docs_split_schema_noop_diff_and_map_elites(tmp_path):
    from lox.author.prompts import build_system_prompt
    from lox.author.tools import AuthorToolRegistry
    from lox.engine.map_elites import MAPElitesArchive

    sys_swarm = build_system_prompt(is_swarm=True)
    assert "3 Strict Generator Rules" not in sys_swarm
    assert "state.in_menu" in sys_swarm
    assert "no_diag=state.mem.door_mask" in sys_swarm
    assert "`experiments`" not in sys_swarm
    assert 'ActionLeaf(fn, name="...", min_commit=0)' in sys_swarm

    base_pol = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n\n"
        "def leaf_wait(state, bb):\n"
        "    return Action.WAIT\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(PrioritySelector([ActionLeaf(leaf_wait, name='w')]), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "noop.duckdb"))
    tools.configure_generation(mode="SWARM_PEER", generation=62, skills={})
    tools.current_policy_code = base_pol

    # <hypothesis> wrapper is stripped and AST-identical candidate triggers NOOP_DIFF
    msg_noop, st_noop = tools.dry_run_policy(
        code="<hypothesis>\nhypothesis: test\n</hypothesis>\n<policy_code>\n"
        + base_pol
        + "\n</policy_code>\n",
        smoke_only=False,
    )
    assert st_noop == "NOOP_DIFF", msg_noop

    # MAPElitesArchive v2 isolates avg_depth & armor/XL vitals without max_depth aliasing
    cid_pct, d_b, v_b, t_b = MAPElitesArchive.compute_cell_id(
        {
            "avg_depth": 4.5,
            "max_depth": 9,
            "avg_ac": 5.0,
            "avg_xp_level": 3.0,
            "avg_unique_tiles": 900.0,
            "avg_turns": 1500.0,
        }
    )
    assert cid_pct.startswith("cell_v2_")
    assert (
        d_b == 3
    )  # avg_depth 4.5 in [4.0, 6.5) -> d3 (ignores single outlier max_depth 9)
    assert v_b == 2  # (10-5) + 1.5*(3-1) = 8.0 in [7.0, 12.0) -> v2
    assert t_b == 1  # 900/1500 = 0.60 in [0.30, 0.70) -> t1


def test_gamestate_cursed_bug_traps_menu_and_guarded_range():
    from lox.env.actions import Action
    from lox.env.harness import NetHackHarness
    from lox.env.state import _FLOOR_GLYPH_FLAGS_LUT, InventoryItem
    from lox.lang.compiler import compile_policy_package
    from lox.lang.instrument import PolicyLoopError

    # 1. "uncursed" must NEVER match is_cursed
    itm_un = InventoryItem(
        letter="c",
        name="an uncursed +3 small shield (being worn)",
        glyph=1,
        oclass=3,
        char="[",
        quantity=1,
        info=None,
        corpse_monster=None,
    )
    assert itm_un.is_uncursed is True
    assert itm_un.is_cursed is False
    assert itm_un.is_worn is True
    assert itm_un.enchantment == 3

    itm_cu = InventoryItem(
        letter="d",
        name="a cursed ring of hunger (on right hand)",
        glyph=2,
        oclass=4,
        char="=",
        quantity=1,
        info=None,
        corpse_monster=None,
    )
    assert itm_cu.is_cursed is True
    assert itm_cu.is_worn is True

    # 2. Interactive menu is preserved for at least 1 step instead of auto-dismissed
    h = NetHackHarness(max_steps=30)
    try:
        st0 = h.reset(42)
        assert st0.mem.remembered_trap_glyphs.shape == (21, 79)
        assert st0.mem.remembered_obj_glyphs.shape == (21, 79)
        assert st0.trap_mask.shape == (21, 79)
        assert st0.is_trap(0, 0) is False
        assert st0.trap_at(0, 0) is None
        st1, done1, _ = h.step(Action.INVENTORY)
        assert not done1
        assert st1.in_menu is True
        st2, done2, _ = h.step(Action.ESC)
        assert not done2
        assert st2.in_menu is False
        assert _FLOOR_GLYPH_FLAGS_LUT.shape[0] > 0
    finally:
        h.close()

    # 3. guarded_range supports huge range with early break AND catches real zero-yield loops
    ok_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        for i in range(10**9):\n"
        "            if i >= 5:\n"
        "                break\n"
        "        return Action.WAIT\n"
    )
    compiled_ok = compile_policy_package(ok_pol)
    runner_ok = compiled_ok.create_runner()
    act, _ = runner_ok.next_action(st0)
    assert act == Action.WAIT

    bad_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        tot = 0\n"
        "        for i in range(300000):\n"
        "            tot += i\n"
        "        return Action.WAIT\n"
    )
    compiled_bad = compile_policy_package(bad_pol, max_loop_ticks=50000)
    runner_bad = compiled_bad.create_runner()
    try:
        runner_bad.next_action(st0)
        raise AssertionError("Expected PolicyLoopError")
    except PolicyLoopError:
        pass


def test_more_yn_collision_opt_in_menu_linger_and_hypothesis_summary(tmp_path):
    import numpy as np

    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.env.actions import Action
    from lox.env.harness import NetHackHarness, policy_handles_menus
    from lox.env.state import GameState

    # 1. Opt-in menu linger detection
    assert policy_handles_menus("def step(self, s): return Action.WAIT", {}) is False
    assert policy_handles_menus("def step(self, s): return s.in_menu", {}) is True
    assert (
        policy_handles_menus("class Policy: pass", {"m": "if s.in_menu: pass"}) is True
    )

    h = NetHackHarness(max_steps=20, allow_menu_linger=False)
    try:
        st0 = h.reset(42)
        raw = list(h._last_raw_tuple)
        obs_keys = list(h._obs_keys)
        idx_map = dict(h._idx_map)
        assert st0.depth == 1
        st1, done1, _ = h.step(Action.INVENTORY)
        assert not done1
        # Unhandled menus are cleanly auto-dismissed when allow_menu_linger=False
        assert st1.in_menu is False
    finally:
        h.close()

    # 2. in_yn_prompt must yield False when `--More--` is on row 0 while waiting_for_space
    idx_int = idx_map["internal"]
    idx_tty = idx_map["tty_chars"]
    raw[idx_int] = raw[idx_int].copy()
    raw[idx_int][1] = 1
    raw[idx_int][3] = 1
    raw[idx_tty] = raw[idx_tty].copy()
    banner = b"You open the tin. It smells like orcs.--More--"
    raw[idx_tty][0, :] = ord(" ")
    raw[idx_tty][0, : len(banner)] = np.frombuffer(banner, dtype=np.uint8)
    collided = GameState(tuple(raw), obs_keys=obs_keys, obs_idx_map=idx_map)
    assert collided.in_yn_prompt is False

    # 3. DeficitAnalyzer._populate_recent_attempts surfaces hypothesis/cluster over binding_constraint
    dbp = str(tmp_path / "rec.duckdb")
    con = init_db(dbp)
    con.execute(
        "INSERT INTO policies (policy_id, generation, mode, policy_code, avg_depth, max_depth, is_promoted) VALUES ('p1', 1, 'SWARM', 'class Policy: pass', 4.5, 9, TRUE)"
    )
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, parent_policy_id, candidate_policy_id, episodes_recorded, episodes_discarded, avg_depth, max_depth, avg_score, hypothesis_passed) VALUES ('b1', 2, 'SWARM', 'p1', 'p2', 24, 0, 4.75, 11, 700.0, TRUE)"
    )
    con.execute(
        "INSERT INTO synthesis_logs (log_id, generation, agent_role, turn_index, turn_type, hypothesis_yaml) VALUES ('s1', 2, 'peer_1', 3, 'SMOKE_CHECK', 'binding_constraint: long boilerplate\ncluster: slice_stalled\nhypothesis: H-fix-more-yn-loop')"
    )
    rep = DeficitAnalyzer(dbp).analyze(
        generation=3, current_policy="class Policy: pass"
    )
    con.close()
    assert rep.recent_attempts
    assert "H-fix-more-yn-loop" in rep.recent_attempts[0]
    assert "binding_constraint" not in rep.recent_attempts[0]

    # 4. enforce_peer_research_gate requires BOTH query_wiki and query_theories before dry_run_policy
    from lox.author.tools import AuthorToolRegistry

    class _FakeTheories:
        def search_theories(self, query, limit=6, doc_type=None):
            return []

    reg = AuthorToolRegistry(db_path=dbp)
    reg.theories_store = _FakeTheories()
    reg.configure_generation(generation=3, mode="SWARM_PEER")
    reg.enforce_peer_research_gate = True
    pol_code = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    msg_r1, st_r1 = reg.dry_run_policy(pol_code, seed=42, max_turns=15)
    assert st_r1 == "RESEARCH_REQUIRED"
    assert "query_wiki" in msg_r1 and "query_theories" in msg_r1

    # Querying only theories is not enough; query_wiki is still required
    reg.query_theories("armor equip")
    msg_r2, st_r2 = reg.dry_run_policy(pol_code, seed=42, max_turns=15)
    assert st_r2 == "RESEARCH_REQUIRED"
    assert "query_wiki" in msg_r2

    # Once BOTH query_theories and query_wiki ran, dry_run_policy executes
    reg.query_wiki("armor slot weight")
    _msg_r3, st_r3 = reg.dry_run_policy(pol_code, seed=42, max_turns=15)
    assert st_r3 != "RESEARCH_REQUIRED"


def test_phase4_and_5_organic_scenarios_referee_composition_holdout_and_scoped_ochiai(
    tmp_path,
):
    import json

    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitReport
    from lox.author.swarm_peer import (
        PeerCandidate,
        compute_candidate_skill_symbols,
        incubate_candidate,
    )
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.db.shards import mine_organic_scenarios
    from lox.db.summarizer import compute_function_stats
    from lox.engine.frontier import FrontierArchive
    from lox.engine.swarm_referee import SwarmReferee

    db_path = str(tmp_path / "p45.duckdb")
    f_dir = str(tmp_path / "frontier")
    init_db(db_path).close()

    # 1. FrontierArchive inverse-visit visit_count persistence
    fa = FrontierArchive(frontier_dir=f_dir)
    fa.add_checkpoint(
        seed=10,
        depth=6,
        milestone="dlvl_6",
        turns=500,
        action_bytes=b"llllkjjjjhhh" * 2,
    )
    fa.add_checkpoint(
        seed=11,
        depth=6,
        milestone="dlvl_6",
        turns=520,
        action_bytes=b"kkkkhhhhllll" * 2,
    )
    cp = fa.sample_checkpoint(min_depth=4)
    assert cp is not None and cp.visit_count == 1
    fa_reloaded = FrontierArchive(frontier_dir=f_dir)
    assert any(c.visit_count == 1 for c in fa_reloaded.checkpoints.values())

    # 2. mine_organic_scenarios across structural buckets
    con = connect_db(db_path, read_only=False)
    ab_long = b"l" * 120
    con.execute(
        """
        INSERT INTO runs (
            run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, steps,
            end_reason, auto_esc_count, starved_with_food, died_unarmored, is_shard, action_bytes, milestones_json
        ) VALUES
        ('r_deep', 'b1', 1, 2001, 'val', 5, 5, 400, 1200, 1200, 'killed by ant', 0, FALSE, FALSE, FALSE, ?, '{"depth_first_turn":{"2":200}}'),
        ('r_stall', 'b1', 1, 2002, 'val', 1, 1, 10, 80, 200, 'stalled_zero_turn', 3, FALSE, FALSE, FALSE, ?, '{}'),
        ('r_unarm', 'b1', 1, 2003, 'val', 2, 2, 80, 400, 400, 'killed by orc', 0, FALSE, TRUE, FALSE, ?, '{"unworn_armor_in_pack":true,"depth_first_turn":{"2":250}}'),
        ('r_ill', 'b1', 1, 2004, 'val', 2, 2, 70, 500, 500, 'died of terminal illness', 0, TRUE, FALSE, FALSE, ?, '{}')
        """,
        [ab_long, ab_long, ab_long, ab_long],
    )
    con.close()
    mined = mine_organic_scenarios(db_path, frontier_dir=f_dir, max_total=10)
    buckets = {sh.trigger_context for sh in mined}
    assert "frontier_deep" in buckets
    assert "stall_or_prompt" in buckets
    assert "unequipped_crisis" in buckets
    assert "resource_or_mortality" in buckets

    # 3. Hard exception rejection during incubate_candidate
    base_pol = """
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector
from skills.sub import h_base

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        self.tree = BehaviorTree(PrioritySelector([ActionLeaf(h_base, name='base')]), blackboard=self.blackboard)
    def step(self, state):
        return self.tree.tick(state) or Action.WAIT
"""
    base_skills = {
        "sub": "from lox.env.actions import Action\ndef h_base(s, bb):\n    return Action.SEARCH\n"
    }
    cand_exc = PeerCandidate(
        peer_id="p_exc",
        claimed_deficit_id="c1",
        claimed_label="raises in pool",
        policy_code=base_pol,
        skills=base_skills,
    )

    class _ExcPool:
        def evaluate_batch(self, **kwargs):
            return {
                "runs": [
                    {
                        "run_id": "x1",
                        "seed": 42,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 50,
                        "turns": 100,
                        "policy_exceptions": 2,
                        "end_reason": "policy_error:KeyError",
                        "milestones_json": json.dumps(
                            {"exception_tracebacks": ["sub.py:5: KeyError: 'k'"]}
                        ),
                    }
                ]
            }

    rep_inc = DeficitReport(generation=2, regime="PLATEAU")
    inc_out = incubate_candidate(
        cand_exc,
        deficit_report=rep_inc,
        claimed_deficit=DeficitAnalyzer._bootstrap_clusters()[0],
        eval_pool=_ExcPool(),
        shard_seeds=[42],
    )
    assert inc_out.sandbox_passed is False
    assert "exception" in inc_out.error.lower()

    # 4. Top-K 3-way AST composition inside the SAME skill module + Hidden Holdout audit cancellation
    sk_a = base_skills["sub"] + "\ndef h_alpha(s, bb):\n    return Action.EAST\n"
    sk_b = base_skills["sub"] + "\ndef h_beta(s, bb):\n    return Action.WEST\n"
    pol_a = base_pol.replace(
        "from skills.sub import h_base", "from skills.sub import h_alpha, h_base"
    ).replace(
        "ActionLeaf(h_base, name='base')",
        "ActionLeaf(h_alpha, name='alpha'), ActionLeaf(h_base, name='base')",
    )
    pol_b = base_pol.replace(
        "from skills.sub import h_base", "from skills.sub import h_base, h_beta"
    ).replace(
        "ActionLeaf(h_base, name='base')",
        "ActionLeaf(h_base, name='base'), ActionLeaf(h_beta, name='beta')",
    )
    ca = PeerCandidate(
        peer_id="pA",
        claimed_deficit_id="da",
        claimed_label="alpha",
        policy_code=pol_a,
        skills={"sub": sk_a},
        touched_skills={"sub"},
        touched_branches={"alpha"},
        touched_skill_symbols=compute_candidate_skill_symbols(
            {"sub": sk_a}, base_skills
        ),
        sandbox_passed=True,
    )
    cb = PeerCandidate(
        peer_id="pB",
        claimed_deficit_id="db",
        claimed_label="beta",
        policy_code=pol_b,
        skills={"sub": sk_b},
        touched_skills={"sub"},
        touched_branches={"beta"},
        touched_skill_symbols=compute_candidate_skill_symbols(
            {"sub": sk_b}, base_skills
        ),
        sandbox_passed=True,
    )
    combos = SwarmReferee.build_orthogonal_candidates([ca, cb], base_pol, base_skills)
    assert len(combos) == 1
    assert "h_alpha" in combos[0].skills["sub"] and "h_beta" in combos[0].skills["sub"]
    assert "h_alpha" in combos[0].policy_code and "h_beta" in combos[0].policy_code

    # Holdout cancellation check
    class _HoldoutFailPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [1, 2])
            if batch_id.startswith("holdout_") and "_inc_" not in batch_id:
                return {
                    "runs": [
                        {
                            "run_id": f"hr_{s}",
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 0,
                            "turns": 50,
                            "final_ac": 10,
                            "xp_level": 1,
                        }
                        for s in s_list
                    ]
                }
            if "_inc_" in batch_id:
                return {
                    "runs": [
                        {
                            "run_id": f"ir_{s}",
                            "seed": s,
                            "depth": 2,
                            "max_depth": 2,
                            "score": 100,
                            "turns": 200,
                            "final_ac": 9,
                            "xp_level": 2,
                        }
                        for s in s_list
                    ]
                }
            return {
                "runs": [
                    {
                        "run_id": f"cr_{s}",
                        "seed": s,
                        "depth": 4,
                        "max_depth": 4,
                        "score": 300,
                        "turns": 500,
                        "final_ac": 6,
                        "xp_level": 3,
                    }
                    for s in s_list
                ]
            }

    ref = SwarmReferee()
    hv = ref.evaluate_tournament(
        generation=10,
        candidates=[ca],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_HoldoutFailPool(),
        seeds=[1001, 1002, 1003, 1004],
        holdout_seeds=[100000, 100001, 100002, 100003],
    )
    assert hv.promoted is False
    assert hv.holdout_iqm_delta is not None and hv.holdout_iqm_delta < -0.15
    assert len(hv.attempts_summary) >= 1

    # 5. Stagnation-discounted Ochiai + tournament_attempts surfaced by DeficitAnalyzer
    fn_stats = compute_function_stats(
        "b_stag",
        10,
        [
            {
                "run_id": "r_g",
                "is_outlier_good": True,
                "max_depth": 5,
                "turns": 100,
                "end_reason": "killed",
                "triggered_functions": json.dumps({"good_fn": 20, "wait_stag": 10}),
                "milestones": {"stagnation_fn_turns": {"wait_stag": 85}},
                "summary_digest": "Stalls: wait_stag stalled (T5..T90 (85 turns))",
            }
        ],
    )
    by_fn_st = {row["function_name"]: row for row in fn_stats}
    assert (
        by_fn_st["wait_stag"]["ochiai_good_run_score"]
        < by_fn_st["good_fn"]["ochiai_good_run_score"]
    )

    BatchDatabaseLogger(db_path=db_path).save_tournament_attempts(hv.attempts_summary)
    rep_after = DeficitAnalyzer(db_path).analyze(generation=11, current_policy=base_pol)
    assert any(
        "REJECTED" in line and "pA" in line for line in rep_after.recent_attempts
    )
