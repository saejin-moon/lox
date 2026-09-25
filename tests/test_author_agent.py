"""
S0 tests: evidence tools, agentic author sessions, goal-failure counters,
prompt program loader (AGENT_PLAN §2-S0, §1.3, §1.6).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile

import duckdb
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lox.deliberative.providers.mock_provider import MockProvider
from lox.policy.author_agent import AgenticAuthor, author_session
from lox.policy.author_tools import (
    check_sql_readonly, execute_tool_call, parse_tool_calls, query_duckdb,
    read_env_schema, read_program_tree, read_trajectory, tool_docs, wiki_search,
)
from lox.policy.failure_counters import counters_ctx, load_failure_counters
from lox.policy.manifest import build_manifest
from lox.policy.predicates import evaluate, nethack_bindings, default_ctx
from lox.policy.program import PolicyProgram
from lox.policy.prompts import (
    BUNDLED_DEFAULT_PROMPT, list_prompt_versions, load_prompt_text,
    render_system_prompt,
)
from lox.policy.report import build_failure_counters_section, build_report_bundle
from lox.policy.validator import validate_diff, ValidatorHooks

# ===========================================================================
# Fixtures: deterministic mini telemetry DB
# ===========================================================================

EPISODES_SCHEMA = ("episode_id VARCHAR, run_id VARCHAR, seed BIGINT, mode VARCHAR, "
                   "role VARCHAR, total_turns BIGINT, max_depth INTEGER, "
                   "final_score BIGINT, death_message VARCHAR, death_category VARCHAR, "
                   "is_ascended BOOLEAN, timestamp VARCHAR")
TICKS_SCHEMA = ("episode_id VARCHAR, run_id VARCHAR, step BIGINT, turn BIGINT, "
                "depth INTEGER, dungeon_number INTEGER, x INTEGER, y INTEGER, "
                "hp INTEGER, max_hp INTEGER, ac INTEGER, hunger_state INTEGER, "
                "condition_bits BIGINT, action_name VARCHAR")
GOALS_SCHEMA = ("ts DOUBLE, run_id VARCHAR, episode_id VARCHAR, depth INTEGER, "
                "dnum INTEGER, turn INTEGER, goal VARCHAR, event VARCHAR, "
                "steps_in_goal INTEGER")


@pytest.fixture
def mini_db(tmp_path):
    db = str(tmp_path / "mini.duckdb")
    con = duckdb.connect(db)
    con.execute(f"CREATE TABLE episodes ({EPISODES_SCHEMA})")
    con.execute(f"CREATE TABLE ticks ({TICKS_SCHEMA})")
    con.execute(f"CREATE TABLE goal_events ({GOALS_SCHEMA})")
    con.execute(
        "INSERT INTO episodes VALUES ('ep_1', 'r1', 0, 'eval', 'valkyrie', 100, 3, "
        "42, 'killed by a jackal', 'combat', false, '2026-01-01')")
    con.execute(
        "INSERT INTO episodes VALUES ('ep_2', 'r1', 0, 'eval', 'valkyrie', 50, 1, "
        "7, 'You faint from lack of food', 'starvation', false, '2026-01-02')")
    # ep_2 ticks: 5 turns descending hp, then death
    for i in range(1, 6):
        con.execute(
            "INSERT INTO ticks VALUES ('ep_2', 'r1', ?, ?, 1, 0, 10, 10, ?, 16, 9, "
            "?, 0, 'STEP')", (i, i + 1, 16 - i, 1))
    # goal_events: descend failed twice, explore ok
    con.execute("INSERT INTO goal_events VALUES (1.0, 'r1', 'ep_2', 1, 0, 1, "
                "'descend', 'skipped_budget', 300)")
    con.execute("INSERT INTO goal_events VALUES (2.0, 'r1', 'ep_2', 1, 0, 2, "
                "'descend', 'skipped_when', 0)")
    con.execute("INSERT INTO goal_events VALUES (3.0, 'r1', 'ep_2', 1, 0, 3, "
                "'explore_floor', 'completed', 50)")
    con.close()
    return db


# ===========================================================================
# query_duckdb: read-only guardrails
# ===========================================================================

class TestQueryDuckdb:
    def test_select_is_allowed(self, mini_db):
        out = query_duckdb("SELECT COUNT(*) AS n FROM episodes", db_path=mini_db)
        assert out.startswith("|") and "2" in out

    @pytest.mark.parametrize("sql", [
        "DROP TABLE ticks",
        "INSERT INTO ticks VALUES (1)",
        "UPDATE episodes SET role = 'x'",
        "DELETE FROM ticks",
        "CREATE TABLE evil (a INT)",
        "ATTACH 'other.db' AS other",
        "COPY episodes TO '/tmp/evil.csv'",
        "SELECT 1; DROP TABLE ticks",
    ])
    def test_write_statements_rejected(self, mini_db, sql):
        out = query_duckdb(sql, db_path=mini_db)
        assert out.startswith("ERROR")
        # the table still exists
        assert query_duckdb("SELECT COUNT(*) FROM ticks", db_path=mini_db).startswith("|")

    def test_author_sql_errors_become_results(self, mini_db):
        out = query_duckdb("SELECT * FROM no_such_table", db_path=mini_db)
        assert out.startswith("ERROR")

    def test_check_sql_readonly_direct(self):
        assert check_sql_readonly("WITH t AS (SELECT 1) SELECT * FROM t") == ""
        assert check_sql_readonly("DESCRIBE episodes") == ""
        assert check_sql_readonly("PRAGMA force_checkpoint")


# ===========================================================================
# wiki_search / read_trajectory / env schema / program tree
# ===========================================================================

class TestEvidenceTools:
    @pytest.mark.skipif(not os.path.exists("data/wiki_index.db"),
                        reason="wiki index not built")
    def test_wiki_search_hits(self):
        out = wiki_search("Elbereth", top_k=2)
        assert "Elbereth" in out and not out.startswith("ERROR")

    def test_read_trajectory_slice(self, mini_db):
        out = read_trajectory("ep_2", 0, 10, db_path=mini_db)
        assert "TRAJECTORY ep_2" in out
        assert "faint from lack of food" in out
        assert "explore_floor" in out  # goal events included

    def test_read_trajectory_empty_window(self, mini_db):
        out = read_trajectory("ep_2", 5000, 6000, db_path=mini_db)
        assert out.startswith("OK")

    def test_env_schema_content(self):
        out = read_env_schema()
        assert "QUAFF" in out and "PRAY" in out and "condition_bits" in out

    def test_program_tree_content(self):
        program = PolicyProgram.from_dict(PolicyProgram.load().to_dict())
        out = read_program_tree(program)
        assert f"PROGRAM v{program.version}" in out
        assert "descend" in out and "strategy_plan" in out

    def test_tool_docs_lists_all_six(self):
        docs = tool_docs()
        for t in ("query_duckdb", "wiki_search", "read_trajectory",
                  "read_env_schema", "read_manifest", "read_program_tree"):
            assert f"{t}(" in docs


class TestToolDispatch:
    def test_parse_tool_calls(self):
        text = ('prose first\n(tool query_duckdb "SELECT 1")\n'
                '(tool wiki_search "gold dragon")\ntrailing prose')
        calls = parse_tool_calls(text)
        assert calls == [("query_duckdb", ["SELECT 1"]),
                         ("wiki_search", ["gold dragon"])]

    def test_parse_tool_calls_malformed(self):
        assert parse_tool_calls("(tool query_duckdb \"unclosed") == []

    def test_unknown_tool_and_arity(self, mini_db):
        assert execute_tool_call("rm_rf", [], db_path=mini_db).startswith("ERROR")
        assert execute_tool_call("query_duckdb", [], db_path=mini_db).startswith("ERROR")
        out = execute_tool_call("query_duckdb", ["SELECT 1"], db_path=mini_db)
        assert out.startswith("|")

    def test_execute_tool_call_routes(self, mini_db):
        out = execute_tool_call("read_trajectory", ["ep_2", 0, 10], db_path=mini_db)
        assert "TRAJECTORY ep_2" in out


# ===========================================================================
# Failure counters: goal_events → condition vocabulary
# ===========================================================================

class TestFailureCounters:
    def test_load_counts_skips_only(self, mini_db):
        counters = load_failure_counters(mini_db, refresh=True)
        assert counters == {"descend": 2}   # completed explore_floor is NOT a failure

    def test_counters_ctx_shape(self):
        ctx = counters_ctx({"descend": 3})
        assert ctx["failures_total"] == 3 and ctx["failures"]["descend"] == 3

    def test_predicate_evaluates_loaded_counters(self, mini_db):
        counters = load_failure_counters(mini_db, refresh=True)
        ctx = {**default_ctx(), **counters_ctx(counters)}
        bindings = nethack_bindings(ctx)
        assert evaluate(("call", "failures_in_10_episodes_ge", [2, "descend"]), bindings)
        assert not evaluate(("call", "failures_in_10_episodes_ge", [3, "descend"]), bindings)
        # per-goal scoping: explore has no failures
        assert not evaluate(("call", "failures_in_10_episodes_ge", [1, "explore_floor"]),
                            bindings)

    def test_goal_interpreter_wires_counters(self, mini_db):
        from lox.policy.goal_interpreter import GoalInterpreter
        from tests.test_shop_and_dungeon_graph import make_blstats  # reuse builder

        interp = GoalInterpreter(failure_counters={"descend": 2})
        ctx = interp._ctx(make_blstats(), "valkyrie")
        assert ctx["failures"] == {"descend": 2} and ctx["failures_total"] == 2
        assert evaluate(("call", "failures_in_10_episodes_ge", [2, "descend"]),
                        nethack_bindings(ctx))

    def test_bundle_carries_failure_counters_and_slices(self, mini_db):
        section = build_failure_counters_section(mini_db)
        assert section == [{"goal": "descend", "failures": 2}]
        bundle = build_report_bundle(mini_db, last_n_episodes=10)
        assert bundle["failure_counters"] == section
        assert len(bundle["trajectory_slices"]) >= 1
        assert "faint from lack of food" in bundle["trajectory_slices"][0]["slice"]


# ===========================================================================
# Prompt program loader
# ===========================================================================

class TestPromptProgram:
    def test_versions_discovered(self):
        versions = list_prompt_versions()
        assert len(versions) >= 5  # S0 campaign seeds v1..v5
        assert versions[0] == "v1"

    def test_load_by_name(self):
        assert "DESCEND BEFORE DETOURS" in load_prompt_text("v3")

    def test_load_by_path(self):
        path = os.path.join("data", "prompts", "v1.md")
        assert load_prompt_text(path) == open(path, encoding="utf-8").read()

    def test_unknown_falls_back_to_newest(self):
        text = load_prompt_text("v_does_not_exist")
        versions = list_prompt_versions()
        assert text == load_prompt_text(versions[-1])

    def test_missing_dir_falls_back_to_bundled(self, tmp_path):
        text = load_prompt_text(None, prompts_dir=str(tmp_path / "empty"))
        assert text == BUNDLED_DEFAULT_PROMPT

    def test_render_injects_tool_docs(self):
        rendered = render_system_prompt("PROMPT {tool_docs} END", "TOOLDOCS")
        assert "PROMPT TOOLDOCS END" == rendered

    def test_render_appends_when_no_placeholder(self):
        rendered = render_system_prompt("BASE", "TOOLDOCS")
        assert rendered.startswith("BASE") and "TOOLDOCS" in rendered


# ===========================================================================
# Agentic author sessions (mock provider, deterministic)
# ===========================================================================

@pytest.fixture
def session_env(tmp_path):
    program_path = str(tmp_path / "program.json")
    PolicyProgram.load().save(program_path)
    ledger_path = str(tmp_path / "ledger.jsonl")
    program = PolicyProgram.load(program_path)
    manifest = build_manifest(program.version, program.domain,
                              live_macros={m["name"]: m["body"] for m in program.macros})
    return program_path, ledger_path, program, manifest


def _hooks() -> ValidatorHooks:
    return ValidatorHooks(fixture_states=[
        default_ctx(),
        {**default_ctx(), "failures": {"descend": 3}, "failures_total": 3},
    ])


class TestAgenticAuthor:
    def test_mock_agentic_session_accepts(self, session_env, mini_db):
        program_path, ledger_path, program, manifest = session_env
        out = asyncio.run(author_session(
            MockProvider(), program, program_path,
            ledger_path=ledger_path, author_mode="agentic",
            db_path=mini_db, hooks=_hooks(), manifest=manifest,
        ))
        assert out.accepted
        assert out.candidate.version == program.version + 1
        # tool loop actually ran: the mock gathers evidence before the diff
        assert out.meta.get("tool_log")
        assert any(t["tool"] == "query_duckdb" for t in out.meta["tool_log"])
        # committed (validator = only writer)
        assert PolicyProgram.load(program_path).version == program.version + 1
        # ledger records the acceptance
        ledger_entries = [json.loads(line) for line in open(ledger_path)]
        assert len(ledger_entries) == 1
        assert ledger_entries[0]["accepted"] is True
        assert ledger_entries[0]["prompt_id"].startswith("prompt:")

    def test_rejected_diff_never_touches_program(self, session_env, mini_db):
        program_path, ledger_path, program, manifest = session_env
        provider = MockProvider(canned_response=(
            '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) '
            '(reason "mock: unknown param probe"))\n'
            '(set policy_params.survival.no_such_param 0.5)'))
        out = asyncio.run(author_session(
            provider, program, program_path,
            ledger_path=ledger_path, author_mode="agentic",
            db_path=mini_db, hooks=_hooks(), manifest=manifest,
        ))
        assert not out.accepted
        assert out.error_code == "ERR_BOUNDS"
        assert PolicyProgram.load(program_path).version == program.version  # untouched
        ledger_entries = [json.loads(line) for line in open(ledger_path)]
        assert ledger_entries[0]["accepted"] is False
        assert ledger_entries[0]["reject_code"] == "ERR_BOUNDS"

    def test_bundle_mode_baseline_session(self, session_env, mini_db):
        program_path, ledger_path, program, manifest = session_env
        out = asyncio.run(author_session(
            MockProvider(), program, program_path,
            ledger_path=ledger_path, author_mode="bundle",
            db_path=mini_db, hooks=_hooks(), manifest=manifest,
        ))
        # bundle arm: no tool log (single-shot prompt)
        assert out.accepted and not out.meta.get("tool_log")

    def test_agentic_author_tool_loop_transcript(self, mini_db):
        """Direct AgenticAuthor: tool results appear in the follow-up turn."""
        program = PolicyProgram.load()
        manifest = build_manifest(program.version, program.domain)
        author = AgenticAuthor(MockProvider(), db_path=mini_db)
        diff = asyncio.run(author.propose_diff(program, manifest, {}))
        assert diff.startswith("(revision")
        assert author.last_meta["prompt_version"]
        assert len(author.last_meta["tool_log"]) == 2


# ===========================================================================
# Live validator compatibility: an accepted agentic diff satisfies the gates
# ===========================================================================

def test_validated_agentic_candidate_keeps_invariants(session_env):
    program_path, ledger_path, program, manifest = session_env
    out = asyncio.run(author_session(
        MockProvider(), program, program_path,
        ledger_path=ledger_path, author_mode="agentic", hooks=_hooks(),
        manifest=manifest, commit=False,
    ))
    assert out.accepted
    cand = out.candidate
    assert {g.goal for g in cand.strategy_plan} >= {g.goal for g in program.strategy_plan}
    assert cand.nogoods == program.nogoods


# ===========================================================================
# S0/Transfer Token Optimization & Domain Isolation Tests
# ===========================================================================

class TestTokenOptimizationAndDomainScoping:
    def test_minihack_manifest_compact_and_isolated(self):
        m = build_manifest(1, domain="minihack")
        rendered = m.render(compact=True)
        assert len(rendered) < 1000
        assert "explore.stuck_patience" in rendered
        assert "has_excalibur" not in rendered
        assert "castle_done" not in rendered

    def test_tool_docs_domain_filtering(self):
        nh_docs = tool_docs(domain="nethack")
        mh_docs = tool_docs(domain="minihack")
        assert "wiki_search" in nh_docs
        assert "wiki_search" not in mh_docs

    def test_bundle_formatting_compactness(self, mini_db):
        from lox.policy.author_agent import format_bundle_summary
        bundle = build_report_bundle(mini_db, domain="nethack")
        summary = format_bundle_summary(bundle)
        assert len(summary) < 1000
        assert "Domain: nethack" in summary

    def test_minihack_report_does_not_leak_nethack_runs(self, mini_db):
        mh_bundle = build_report_bundle(mini_db, domain="minihack")
        assert mh_bundle["domain"] == "minihack"
        assert len(mh_bundle["death_taxonomy"]) == 0
        assert mh_bundle["batch"]["episodes"] == 0
