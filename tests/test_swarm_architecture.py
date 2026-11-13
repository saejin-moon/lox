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
    assert tools.last_pilot_improved is True


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
    assert not (workspace / "STATUS.md").exists()
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
                        "id": "call_1",
                        "function": {
                            "name": "query_db",
                            "arguments": json.dumps(
                                {"sql": "SELECT COUNT(*) AS n FROM runs"}
                            ),
                        },
                    }
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
