"""
LOX 2.0 Macro Request Queue.
Allows the LLM Author Agent to formally request new atomic primitives and macros
that are not yet implemented in the engine, logging them to DuckDB and a markdown queue
for human developer implementation post-run.
"""
from __future__ import annotations

import datetime
import os
import uuid
from typing import Any
import duckdb

from lox.telemetry.consolidator import init_db


QUEUE_FILE = "data/macro_requests.md"


def init_macro_request_table(db_path: str = "data/lox.duckdb") -> None:
    """Ensures macro_requests table exists in DuckDB."""
    con = init_db(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_requests (
            request_id VARCHAR,
            run_id VARCHAR,
            timestamp TIMESTAMP,
            macro_name VARCHAR,
            rationale VARCHAR,
            proposed_interface VARCHAR,
            priority VARCHAR,
            status VARCHAR
        );
    """)
    con.close()


def queue_macro_request(
    macro_name: str,
    rationale: str,
    proposed_interface: str = "",
    priority: str = "medium",
    run_id: str = "synth_session",
    db_path: str = "data/lox.duckdb",
    queue_file: str = QUEUE_FILE,
) -> dict[str, Any]:
    """
    Records a macro request in DuckDB and appends to the post-run human development queue.
    """
    init_macro_request_table(db_path)
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    ts = datetime.datetime.now()
    clean_priority = priority.upper()
    if clean_priority not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        clean_priority = "MEDIUM"

    con = duckdb.connect(db_path)
    con.execute(
        """
        INSERT INTO macro_requests (
            request_id, run_id, timestamp, macro_name,
            rationale, proposed_interface, priority, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        [
            req_id,
            run_id,
            ts,
            macro_name,
            rationale,
            proposed_interface,
            clean_priority,
        ],
    )
    con.close()

    # Append to markdown queue
    os.makedirs(os.path.dirname(os.path.abspath(queue_file)), exist_ok=True)
    md_entry = f"""
### [MACRO REQUEST] `{macro_name}` (Priority: {clean_priority})
- **Request ID**: `{req_id}` | **Run ID**: `{run_id}` | **Timestamp**: {ts.strftime('%Y-%m-%d %H:%M:%S')}
- **Rationale**: {rationale}
- **Proposed Interface**: `{proposed_interface or 'N/A'}`
- **Status**: `pending`

---
"""
    with open(queue_file, "a", encoding="utf-8") as f:
        f.write(md_entry)

    return {
        "request_id": req_id,
        "macro_name": macro_name,
        "priority": clean_priority,
        "status": "pending",
        "message": (
            f"Macro '{macro_name}' queued for post-run human implementation (ID: {req_id}). "
            f"For this run, compose a policy using currently approved primitives."
        ),
    }


def list_macro_requests(status: str = "pending", db_path: str = "data/lox.duckdb") -> list[dict[str, Any]]:
    """Returns all queued macro requests matching status."""
    if not os.path.exists(db_path):
        return []

    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT request_id, timestamp, macro_name, rationale, proposed_interface, priority, status
        FROM macro_requests
        WHERE status = ?
        ORDER BY timestamp DESC
        """,
        [status],
    ).fetchall()
    con.close()

    return [
        {
            "request_id": r[0],
            "timestamp": str(r[1]),
            "macro_name": r[2],
            "rationale": r[3],
            "proposed_interface": r[4],
            "priority": r[5],
            "status": r[6],
        }
        for r in rows
    ]
