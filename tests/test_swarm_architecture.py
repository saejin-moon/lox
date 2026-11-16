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

    assert worker.theories._table is not None
    docs = worker.theories._table.to_arrow().to_pylist()
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

    with pytest.raises(ImportError, match="skills.nonexistent"):
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
