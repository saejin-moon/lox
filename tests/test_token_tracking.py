import duckdb

from lox.telemetry.tokens import get_token_usage_summary, log_token_usage


def test_token_usage_logging(tmp_path):
    db_file = str(tmp_path / "test_tokens.duckdb")

    # Log two sessions
    res1 = log_token_usage(
        run_id="run_01",
        session_id="sess_01",
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=320,
        completion_tokens=180,
        trigger_reason="Floor pacing stall",
        tools_called=["query_duckdb"],
        db_path=db_file,
    )
    assert res1["total_tokens"] == 500
    assert res1["estimated_cost_usd"] > 0

    res2 = log_token_usage(
        run_id="run_01",
        session_id="sess_02",
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=400,
        completion_tokens=200,
        trigger_reason="Death cluster detected",
        tools_called=["get_death_taxonomy"],
        db_path=db_file,
    )
    assert res2["total_tokens"] == 600

    # Retrieve summary for run_01
    summary = get_token_usage_summary(run_id="run_01", db_path=db_file)
    assert summary["total_calls"] == 2
    assert summary["prompt_tokens"] == 720
    assert summary["completion_tokens"] == 380
    assert summary["total_tokens"] == 1100
    assert summary["total_cost_usd"] > 0

    # Verify DuckDB direct query
    con = duckdb.connect(db_file, read_only=True)
    rows = con.execute(
        "SELECT trigger_reason, total_tokens, tools_called FROM token_usage WHERE run_id = 'run_01'"
    ).fetchall()
    con.close()
    assert len(rows) == 2
    assert rows[0][0] == "Floor pacing stall"
    assert rows[0][1] == 500
    assert '["query_duckdb"]' in rows[0][2]
