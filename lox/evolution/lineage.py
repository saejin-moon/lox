"""
LOX-ψ Evolution (S2 scaffolding): run ↔ candidate lineage registry.

Per-candidate evidence tagging: an evaluation batch's telemetry is written under a
run_id, but the author needs evidence *for a specific population candidate*. This
registry maps a batch's base run id to its lineage (population, candidate,
generation, parent, program version). `run_parallel_batch` registers a batch; the
report builder resolves a candidate to its run group and scopes the evidence to it.

Additive and backward-compatible: a table in the telemetry DuckDB, no changes to the
per-tick/episode schemas, existing untagged runs resolve to no lineage.
"""
from __future__ import annotations

import os
import time

DEFAULT_DB_PATH = "data/lox_telemetry.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_lineage (
    run_id VARCHAR PRIMARY KEY,
    base_run_id VARCHAR,
    domain VARCHAR,
    program_version INTEGER,
    population_id VARCHAR,
    candidate_id VARCHAR,
    generation INTEGER,
    parent_id VARCHAR,
    created_ts DOUBLE,
    notes VARCHAR
)
"""


class RunLineage:
    """DuckDB-backed run → candidate lineage registry."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def _connect(self, read_only: bool = False):
        import duckdb
        if not read_only:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        return duckdb.connect(self.db_path, read_only=read_only)

    def register(self, *, run_id: str, domain: str = "nethack",
                 program_version: int | None = None, population_id: str | None = None,
                 candidate_id: str | None = None, generation: int | None = None,
                 parent_id: str | None = None, notes: str | None = None) -> None:
        """Upserts lineage for a batch base run id. Idempotent."""
        con = self._connect()
        try:
            con.execute(_SCHEMA)
            con.execute(
                "DELETE FROM run_lineage WHERE run_id = ?", (run_id,))
            con.execute(
                "INSERT INTO run_lineage (run_id, base_run_id, domain, program_version,"
                " population_id, candidate_id, generation, parent_id, created_ts, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, run_id, domain, program_version, population_id, candidate_id,
                 generation, parent_id, time.time(), notes))
        finally:
            con.close()

    def lookup(self, run_id: str) -> dict | None:
        """Lineage for an exact run id or a '<base>_w<n>' worker run id."""
        base = _strip_worker(run_id)
        con = self._connect(read_only=True)
        try:
            row = con.execute(
                "SELECT run_id, base_run_id, domain, program_version, population_id,"
                " candidate_id, generation, parent_id, created_ts, notes"
                " FROM run_lineage WHERE run_id = ?", (base,)).fetchone()
        except Exception:  # noqa: BLE001 — table may not exist
            return None
        finally:
            con.close()
        if not row:
            return None
        keys = ("run_id", "base_run_id", "domain", "program_version", "population_id",
                "candidate_id", "generation", "parent_id", "created_ts", "notes")
        return dict(zip(keys, row))

    def runs_for_candidate(self, candidate_id: str,
                           population_id: str | None = None) -> list[str]:
        """Base run ids evaluated for a candidate (newest first)."""
        con = self._connect(read_only=True)
        try:
            if population_id is not None:
                rows = con.execute(
                    "SELECT run_id FROM run_lineage WHERE candidate_id = ?"
                    " AND population_id = ? ORDER BY created_ts DESC",
                    (candidate_id, population_id)).fetchall()
            else:
                rows = con.execute(
                    "SELECT run_id FROM run_lineage WHERE candidate_id = ?"
                    " ORDER BY created_ts DESC", (candidate_id,)).fetchall()
            return [r[0] for r in rows]
        except Exception:  # noqa: BLE001
            return []
        finally:
            con.close()

    def candidates(self, population_id: str | None = None) -> list[dict]:
        """All registered lineage rows (optionally scoped to a population)."""
        con = self._connect(read_only=True)
        try:
            keys = ("run_id", "base_run_id", "domain", "program_version", "population_id",
                    "candidate_id", "generation", "parent_id", "created_ts", "notes")
            if population_id is not None:
                rows = con.execute(
                    f"SELECT {', '.join(keys)} FROM run_lineage WHERE population_id = ?"
                    " ORDER BY generation, candidate_id", (population_id,)).fetchall()
            else:
                rows = con.execute(
                    f"SELECT {', '.join(keys)} FROM run_lineage"
                    " ORDER BY created_ts DESC").fetchall()
            return [dict(zip(keys, r)) for r in rows]
        except Exception:  # noqa: BLE001
            return []
        finally:
            con.close()


def _strip_worker(run_id: str) -> str:
    import re
    return re.sub(r"_w\d+$", "", run_id or "")