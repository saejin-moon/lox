import os

from lox.author.requests import list_macro_requests, queue_macro_request
from lox.author.tools import DuckDBToolRegistry


def test_macro_request_queue(tmp_path):
    db_file = str(tmp_path / "test_macro.duckdb")
    queue_file = str(tmp_path / "test_requests.md")

    # 1. Queue a request directly
    res = queue_macro_request(
        macro_name="solve_sokoban_boulder",
        rationale="Need algorithmic boulder pushing to solve Sokoban DL 7-9.",
        proposed_interface="def solve_sokoban(): step_to_boulder(); push_boulder()",
        priority="high",
        run_id="eval_run_42",
        db_path=db_file,
        queue_file=queue_file,
    )
    assert res["status"] == "pending"
    assert "solve_sokoban_boulder" in res["message"]

    # 2. Queue another request via DuckDBToolRegistry
    reg = DuckDBToolRegistry(db_path=db_file)
    msg = reg.request_macro(
        macro_name="dip_for_excalibur",
        rationale="Level 5 Valkyrie needs artifact weapon from fountain.",
        priority="critical",
        run_id="eval_run_42",
    )
    assert "dip_for_excalibur" in msg

    # 3. Verify in DuckDB
    pending = list_macro_requests(status="pending", db_path=db_file)
    assert len(pending) == 2
    names = [r["macro_name"] for r in pending]
    assert "solve_sokoban_boulder" in names
    assert "dip_for_excalibur" in names

    # 4. Verify markdown file was written
    assert os.path.exists(queue_file)
    with open(queue_file, "r") as f:
        content = f.read()
    assert "solve_sokoban_boulder" in content
    assert "Priority: HIGH" in content
