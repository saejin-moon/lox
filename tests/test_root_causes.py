"""
Targeted structural contract & regression tests for RC1..RC5 root causes.
"""

from __future__ import annotations

import ast
import concurrent.futures

from lox.author.tools import AuthorToolRegistry, merge_skill_module_update
from lox.core.contracts import EpisodeRun
from lox.datasets.bootstrap_frontier import bootstrap_frontier_from_duckdb
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, get_schema_markdown, init_db
from lox.engine.frontier import FrontierArchive
from lox.engine.hypothesis import progression_metric, should_stop_early_for_futility
from lox.engine.scheduler import ModeScheduler
from lox.env.actions import Action
from lox.env.benchmarks.nethack_scenarios import (
    NetHackScenarioHarness,
    run_scenario_gate,
)
from lox.env.craftax.harness import CraftaxHarness
from lox.lang.bundle import (
    extract_json_payload,
    extract_skill_imports,
    load_skills_dir,
    unpack_code_bundle,
    write_skills_dir,
)
from lox.lang.compiler import compile_policy_package
from lox.lang.modularize import _modularize_skill_module
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    check_macro_contracts,
    smart_dedent,
    three_way_ast_merge,
)
from scripts.calibrate_pilot import parse_pilot_delta


def test_bundle_ssot_unpack_skills_dir_and_imports(tmp_path):
    raw_verbatim = "def f():\n    return 1\n"
    pol_v, sk_v = unpack_code_bundle(raw_verbatim, preserve_raw_verbatim=True)
    assert pol_v == raw_verbatim
    assert sk_v == {}

    bundle_src = """<think>thinking</think>
<arg_key>code</arg_key><arg_value>ignore</arg_value>
<skill_code module="nav/core">
```python
import math
CONST = 1
def a():
    return 1
def c():
    return 3
```
</skill_code>
<skill_code module="nav/core">
def b():
    return 2
</skill_code>
### skill: survival.food
```python
def eat():
    return 3
```
<policy_code>
from skills import nav
from skills.survival import *
class Policy:
    def step(self, state):
        return Action.WAIT
</policy_code>
"""
    pol, skills = unpack_code_bundle(
        bundle_src, merge_fn=merge_skill_module_update, include_markdown_skills=True
    )
    assert "class Policy:" in pol
    assert "nav/core" in skills
    assert "def a()" in skills["nav/core"]
    assert "def b()" in skills["nav/core"]
    assert "survival/food" in skills

    s_dir = tmp_path / "skills"
    write_skills_dir(s_dir, skills, sync=True)
    loaded = load_skills_dir(s_dir)
    assert set(loaded.keys()) == {"nav/core", "survival/food"}
    assert (s_dir / "nav" / "__init__.py").exists()

    imp_info = extract_skill_imports(ast.parse(pol), set(loaded.keys()))
    assert "nav/core" in imp_info.referenced_keys
    assert "survival/food" in imp_info.referenced_keys
    assert extract_json_payload('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_concurrent_scoped_modules_and_threadsafe_db_writes(tmp_path):
    pol_src = """
def get_magic():
    import skills.helper as h
    return h.MAGIC

class Policy:
    def step(self, state):
        if get_magic() > 0:
            return Action.WAIT
        return Action.WAIT
"""
    db_path = str(tmp_path / "conc.duckdb")
    init_db(db_path).close()
    logger = BatchDatabaseLogger(db_path)

    def _worker(idx: int) -> tuple[int, int]:
        compiled = compile_policy_package(
            pol_src,
            skills={"helper": f"MAGIC = {100 + idx}\n"},
        )
        runner = compiled.create_runner()
        logger.save_synthesis_log(
            log_id=f"log_{idx}",
            generation=1,
            batch_id="b_conc",
            mode="SWARM_PEER",
            turn_index=idx,
            turn_type="TEST",
            agent_role=f"peer_{idx}",
        )
        return idx, runner.instance.__class__.__module__ == "lox_policy" and int(
            compiled.create_runner().next_action.__name__ != ""
        )

    # Execute compile + runtime import + DB write across parallel threads
    c1 = compile_policy_package(pol_src, skills={"helper": "MAGIC = 111\n"})
    c2 = compile_policy_package(pol_src, skills={"helper": "MAGIC = 222\n"})
    r1 = c1.create_runner()
    r2 = c2.create_runner()
    # Both runners execute after each other compiled without overwriting each other's scoped skills
    assert r1.instance is not None and r2.instance is not None

    def _run_magic(val: int) -> int:
        comp = compile_policy_package(
            "from skills import helper\nclass Policy:\n    def step(self, s):\n        import skills.helper as h\n        return Action.N if h.MAGIC == "
            + str(val)
            + " else Action.WAIT\n",
            skills={"helper": f"MAGIC = {val}\n"},
        )
        run = comp.create_runner()
        # Simulate delay so concurrent creates interleave
        act, _ = run.next_action(type("S", (), {})())
        return 1 if act == Action.N else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs_magic = [ex.submit(_run_magic, 10 + i) for i in range(12)]
        futs_db = [ex.submit(_worker, i) for i in range(12)]
        assert all(f.result() == 1 for f in futs_magic)
        for _ in futs_db:
            _.result()

    con = connect_db(db_path, read_only=True)
    try:
        cnt = con.execute("SELECT COUNT(*) FROM synthesis_logs").fetchone()[0]
        assert cnt == 12
    finally:
        con.close()


def test_schema_markdown_parity_and_non_shard_isolation(tmp_path):
    db_path = str(tmp_path / "schema_test.duckdb")
    init_db(db_path).close()

    md_std = get_schema_markdown(db_path, is_swarm=False)
    for tbl in (
        "runs",
        "ticks",
        "batches",
        "discoveries",
        "synthesis_logs",
        "llm_cost_rates",
    ):
        assert tbl in md_std

    con = connect_db(db_path, read_only=False)
    try:
        cols_runs = [r[1] for r in con.execute("PRAGMA table_info('runs')").fetchall()]
        for c in cols_runs:
            assert c in md_std
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b_p', 1, 'PILOT', FALSE)"
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, max_depth, turns, action_bytes, is_shard)
            VALUES
              ('r_full', 'b_p', 1, 1, 2, 50, ?::BLOB, FALSE),
              ('shard_99', 'b_p', 1, 2, 5, 900, ?::BLOB, TRUE)
            """,
            [b"abcd", b"efgh"],
        )
        pilot_ids = [
            r[0] for r in con.execute("SELECT run_id FROM latest_pilot_runs").fetchall()
        ]
        assert pilot_ids == ["r_full"]
    finally:
        con.close()

    added = bootstrap_frontier_from_duckdb(
        db_path=db_path,
        frontier_dir=str(tmp_path / "frontier"),
        max_checkpoints_per_depth=5,
    )
    assert added == 1


def test_benchmark_expert_parity_populated_db(tmp_path):
    db_path = str(tmp_path / "parity.duckdb")
    init_db(db_path).close()
    con = connect_db(db_path, read_only=False)
    try:
        con.execute(
            """
            INSERT INTO expert_runs (run_id, source, player, role, depth, max_depth, score, turns, summary_digest)
            VALUES ('exp_1', 'nao', 'alice', 'Val', 12, 12, 50000, 12000, 'Ascended cleanly')
            """
        )
    finally:
        con.close()

    reg = AuthorToolRegistry(db_path=db_path)
    out, status = reg.benchmark_expert_parity(top_n=5)
    assert status == "parity ok"
    assert "exp_1" in out
    assert "Ascended cleanly" in out


def test_frontier_archive_cross_instance_lock(tmp_path):
    fdir = str(tmp_path / "shared_frontier")

    def _add(i: int) -> str:
        arch = FrontierArchive(frontier_dir=fdir)
        cp = arch.add_checkpoint(
            seed=100 + i,
            depth=2,
            milestone=f"M{i}",
            turns=50 + i,
            action_bytes=bytes(f"act_{i}", "ascii"),
            checkpoint_id=f"cp_{i}",
        )
        return cp.checkpoint_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ids = list(ex.map(_add, range(10)))
    assert len(ids) == 10
    reader = FrontierArchive(frontier_dir=fdir)
    assert len(reader.checkpoints) == 10


def test_modularize_two_level_skill_dotted_imports():
    big_skill = "\n".join(
        [
            "SHARED_CONST = 7",
            "def _helper(x): return x + SHARED_CONST",
            "def alpha(s): return _helper(1)",
            "def beta(s): return _helper(2)",
            "def gamma(s): return _helper(3)",
            "def delta(s): return _helper(4)",
        ]
        + [
            f"def pad_{i}(x):\n    y = _helper(x)\n    return y + {i}\n"
            for i in range(45)
        ]
    )
    res = _modularize_skill_module("nav/core", big_skill, reserved_names=set())
    assert res is not None
    facade, submods = res
    assert "nav/core_shared" in submods
    assert "from skills.nav.core_shared import" in facade
    assert "/" not in "\n".join(facade.splitlines()[1:])
    ast.parse(facade)
    for k, src in submods.items():
        assert "/" not in "\n".join(src.splitlines()[1:])
        ast.parse(src)


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
