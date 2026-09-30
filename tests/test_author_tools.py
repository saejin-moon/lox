import os
import duckdb
import pytest
from lox.author.tools import DuckDBToolRegistry
from lox.author.agent import AuthorAgent
from lox.telemetry.consolidator import init_db
from lox.telemetry.tokens import get_token_usage_summary


@pytest.fixture
def populated_db(tmp_path):
    db_file = str(tmp_path / "test_lox.duckdb")
    con = init_db(db_file)
    con.execute("""
        INSERT INTO episodes (run_id, episode_id, depth, score, turns, death_reason, solved, wall_sec, max_depth)
        VALUES 
            ('run_1', 'ep_1', 1, 100, 80, 'starved to death', false, 0.5, 1),
            ('run_1', 'ep_2', 2, 250, 150, 'killed by newt', false, 0.8, 2),
            ('run_1', 'ep_3', 1, 120, 85, 'starved to death', false, 0.5, 1);
        INSERT INTO ticks (run_id, episode_id, turn, depth, hp, max_hp, hunger, y, x, action, message, reward)
        VALUES 
            ('run_1', 'ep_1', 1, 1, 16, 16, 'NORMAL', 5, 5, 'step', '', 0.0),
            ('run_1', 'ep_1', 2, 1, 16, 16, 'NORMAL', 5, 6, 'step', '', 0.0),
            ('run_1', 'ep_1', 3, 1, 16, 16, 'NORMAL', 5, 6, 'search', '', 0.0);
    """)
    con.close()
    return db_file


def test_duckdb_tool_registry(populated_db):
    registry = DuckDBToolRegistry(db_path=populated_db)

    # 1. Test get_duckdb_schema
    schema = registry.get_duckdb_schema()
    assert "episodes" in schema
    assert "ticks" in schema
    assert "3 rows" in schema

    # 2. Test query_duckdb with safe SELECT
    res = registry.query_duckdb("SELECT death_reason, count(*) as count FROM episodes GROUP BY 1")
    assert "starved to death" in res
    assert "2" in res

    # 3. Test query_duckdb rejects mutative statement
    err_drop = registry.query_duckdb("DROP TABLE episodes")
    assert "forbidden" in err_drop.lower()

    err_delete = registry.query_duckdb("DELETE FROM episodes WHERE depth = 1")
    assert "forbidden" in err_delete.lower()

    # 4. Test death taxonomy
    tax = registry.get_death_taxonomy(window=5)
    assert "starved to death" in tax
    assert "killed by newt" in tax

    # 5. Test pacing stats
    pacing = registry.get_floor_pacing_stats(depth=1)
    assert "total_episodes" in pacing

    # 6. Test action distribution
    dist = registry.get_action_distribution()
    assert "step" in dist
    assert "search" in dist


def test_author_agent_synthesis_and_token_tracking(populated_db):
    agent = AuthorAgent(provider="mock", db_path=populated_db)
    
    current_code = """
def explore():
    step_to_frontier()

plan = [explore]
"""
    new_code, tree, error = agent.synthesize_policy(
        current_policy=current_code,
        trigger_reason="Floor pacing stall: 80 turns on Depth 1",
        status_report="Incident on DL 1, turn 80",
        run_id="eval_run_99",
    )

    assert error is None
    assert tree is not None
    assert "def emergency_recovery():" in new_code

    # Verify token usage was recorded in DuckDB
    tokens = get_token_usage_summary(run_id="eval_run_99", db_path=populated_db)
    assert tokens["total_calls"] == 1
    assert tokens["prompt_tokens"] > 0
    assert tokens["completion_tokens"] > 0
    assert tokens["total_tokens"] == tokens["prompt_tokens"] + tokens["completion_tokens"]
    assert tokens["total_cost_usd"] > 0
