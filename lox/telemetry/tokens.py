"""
LOX Token Usage Tracking.
Records LLM prompt, completion, and total token usage directly into DuckDB.
Computes estimated costs and maintains empirical audit trails of synthesis spend.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

import duckdb

from lox.telemetry.consolidator import init_db, safe_duckdb_connect

# Estimated cost in USD per 1,000,000 tokens
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_substring -> (prompt_cost_per_m, completion_cost_per_m)
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimates dollar cost for a query based on model pricing."""
    rate = (0.10, 0.40)  # default fallback
    for name, pr in MODEL_PRICING.items():
        if name in model.lower():
            rate = pr
            break
    prompt_cost = (prompt_tokens / 1_000_000.0) * rate[0]
    comp_cost = (completion_tokens / 1_000_000.0) * rate[1]
    return float(prompt_cost + comp_cost)


def log_token_usage(
    run_id: str,
    session_id: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    trigger_reason: str,
    tools_called: list[str] | None = None,
    db_path: str = "data/lox.duckdb",
) -> dict[str, Any]:
    """
    Inserts a token usage record into the token_usage table in DuckDB.
    """
    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    ts = datetime.datetime.now()
    tools_str = json.dumps(tools_called or [])

    con = init_db(db_path)
    con.execute(
        """
        INSERT INTO token_usage (
            run_id, session_id, timestamp, provider, model,
            prompt_tokens, completion_tokens, total_tokens,
            estimated_cost_usd, trigger_reason, tools_called
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            session_id,
            ts,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
            trigger_reason,
            tools_str,
        ],
    )
    con.close()

    return {
        "run_id": run_id,
        "session_id": session_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }


def get_token_usage_summary(
    run_id: str | None = None, db_path: str = "data/lox.duckdb"
) -> dict[str, Any]:
    """Retrieves cumulative token stats and cost from DuckDB."""
    if not os.path.exists(db_path):
        return {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

    con = safe_duckdb_connect(db_path, read_only=True)
    if run_id:
        res = con.execute(
            """
            SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens), SUM(estimated_cost_usd)
            FROM token_usage WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
    else:
        res = con.execute(
            """
            SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens), SUM(estimated_cost_usd)
            FROM token_usage
            """
        ).fetchone()
    con.close()

    if not res or res[0] == 0:
        return {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

    return {
        "total_calls": int(res[0]),
        "prompt_tokens": int(res[1] or 0),
        "completion_tokens": int(res[2] or 0),
        "total_tokens": int(res[3] or 0),
        "total_cost_usd": float(res[4] or 0.0),
    }
