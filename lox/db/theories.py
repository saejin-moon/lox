"""
Unified DuckDB Theory & Reflection Store for LOX.
Replaces the external LanceDB store so theories, peer hypotheses, and tournament verdicts
live inside the single canonical DuckDB file (`theories` table) and are queryable both via
`query_theories` (tokenized parameterized SQL ranking) and `query_db`.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from lox.db.schema import _INITIALIZED_DBS, DEFAULT_DB_PATH, connect_db, init_db

DEFAULT_THEORIES_PATH = DEFAULT_DB_PATH
_THEORY_LOCK = threading.RLock()
_DOC_SEQ = itertools.count(1)


def _resolve_theory_db_path(db_path: str) -> str:
    raw_p = str(db_path or DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH
    if raw_p.endswith(".lance"):
        return f"{raw_p[:-6]}.duckdb"
    if not raw_p.endswith((".duckdb", ".db")):
        Path(raw_p).mkdir(parents=True, exist_ok=True)
        return str(Path(raw_p) / "theories.duckdb")
    return raw_p


class TheoryDocumentStore:
    """
    DuckDB-backed document database (`theories` table) for qualitative hypotheses,
    failure post-mortems, and agent reflection reports.
    """

    TABLE_NAME = "theories"

    def __init__(self, db_path: str = DEFAULT_THEORIES_PATH) -> None:
        self.raw_db_path = str(db_path)
        self.db_path = _resolve_theory_db_path(self.raw_db_path)
        self._fts_dirty = False
        self._init_table()

    def _init_table(self) -> None:
        with _THEORY_LOCK:
            abs_p = os.path.abspath(self.db_path)
            if abs_p not in _INITIALIZED_DBS or not os.path.exists(self.db_path):
                init_db(self.db_path).close()
            con = connect_db(self.db_path, read_only=False)
            try:
                cnt_row = con.execute("SELECT COUNT(*) FROM theories").fetchone()
                cnt = int(cnt_row[0]) if cnt_row else 0
                if cnt == 0:
                    con.execute(
                        """
                        INSERT OR IGNORE INTO theories (
                            doc_id, generation, agent_role, doc_type, title,
                            content, tags, verified_c_rule, metrics_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            "seed_000",
                            0,
                            "system",
                            "META_SYNTHESIS",
                            "System Initialization",
                            "Theory document store initialized for LOX multi-agent swarm.",
                            "init,system",
                            "",
                            "{}",
                            float(time.time()),
                        ],
                    )
            finally:
                con.close()

    def save_document(
        self,
        generation: int,
        agent_role: str,
        doc_type: str,
        title: str,
        content: str,
        tags: str | list[str] = "",
        verified_c_rule: str = "",
        metrics: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Saves a qualitative theory, failure report, or synthesis document into DuckDB `theories`.
        """
        if not doc_id:
            doc_id = f"doc_{agent_role}_{int(time.time() * 1000)}_{next(_DOC_SEQ)}"

        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        metrics_str = json.dumps(metrics) if metrics else "{}"

        with _THEORY_LOCK:
            con = connect_db(self.db_path, read_only=False)
            try:
                con.execute(
                    """
                    INSERT OR REPLACE INTO theories (
                        doc_id, generation, agent_role, doc_type, title,
                        content, tags, verified_c_rule, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(doc_id),
                        int(generation),
                        str(agent_role),
                        str(doc_type),
                        str(title),
                        str(content),
                        tags_str,
                        str(verified_c_rule),
                        metrics_str,
                        float(time.time()),
                    ],
                )
            finally:
                con.close()
        return doc_id

    def search_theories(
        self,
        query: str,
        limit: int = 5,
        agent_role: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Searches DuckDB `theories` using parameterized multi-term keyword scoring
        across `title`, `tags`, `verified_c_rule`, and `content`, weighted by generation recency.
        """
        clean_q = str(query or "").strip()
        lim = max(1, int(limit))
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9_]{2,}", clean_q.lower())
            if t not in {"the", "and", "for", "with", "from", "into", "that", "this"}
        ][:8]

        where_parts: list[str] = []
        where_params: list[Any] = []
        if agent_role:
            where_parts.append("agent_role = ?")
            where_params.append(str(agent_role))
        if doc_type:
            where_parts.append("doc_type = ?")
            where_params.append(str(doc_type))

        if tokens:
            token_clauses: list[str] = []
            score_exprs: list[str] = []
            score_params: list[Any] = []
            for tok in tokens:
                pat = f"%{tok}%"
                token_clauses.append(
                    "(title ILIKE ? OR content ILIKE ? OR tags ILIKE ? OR verified_c_rule ILIKE ?)"
                )
                where_params.extend([pat, pat, pat, pat])
                score_exprs.append(
                    "(CASE WHEN title ILIKE ? THEN 3 ELSE 0 END"
                    " + CASE WHEN tags ILIKE ? THEN 2 ELSE 0 END"
                    " + CASE WHEN verified_c_rule ILIKE ? THEN 2 ELSE 0 END"
                    " + CASE WHEN content ILIKE ? THEN 1 ELSE 0 END)"
                )
                score_params.extend([pat, pat, pat, pat])
            where_parts.append(f"({' OR '.join(token_clauses)})")
            score_sql = " + ".join(score_exprs)
        else:
            score_sql = "0"
            score_params = []

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        sql = f"""
            SELECT doc_id, generation, agent_role, doc_type, title, content,
                   tags, verified_c_rule, metrics_json, created_at
            FROM theories
            {where_sql}
            ORDER BY ({score_sql}) DESC, generation DESC, created_at DESC
            LIMIT ?
        """
        params = [*where_params, *score_params, lim]

        with _THEORY_LOCK:
            con = connect_db(self.db_path, read_only=True)
            try:
                cur = con.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except Exception:
                return []
            finally:
                con.close()

    def list_documents(self, limit: int = 500) -> list[dict[str, Any]]:
        """Returns recent documents stored in the `theories` table."""
        with _THEORY_LOCK:
            con = connect_db(self.db_path, read_only=True)
            try:
                cur = con.execute(
                    """
                    SELECT doc_id, generation, agent_role, doc_type, title, content,
                           tags, verified_c_rule, metrics_json, created_at
                    FROM theories
                    ORDER BY generation DESC, created_at DESC
                    LIMIT ?
                    """,
                    [max(1, int(limit))],
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except Exception:
                return []
            finally:
                con.close()

    def get_latest_reports(
        self,
        generation: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Retrieves the latest REFLECTION_REPORT for each agent_role.
        Returns a mapping of `{agent_role: doc_dict}`.
        """
        with _THEORY_LOCK:
            con = connect_db(self.db_path, read_only=True)
            try:
                rows: list[tuple[Any, ...]] = []
                cols: list[str] = []
                if generation is not None:
                    cur = con.execute(
                        """
                        SELECT doc_id, generation, agent_role, doc_type, title, content,
                               tags, verified_c_rule, metrics_json, created_at
                        FROM theories
                        WHERE doc_type = 'REFLECTION_REPORT' AND generation = ?
                        ORDER BY created_at DESC
                        """,
                        [int(generation)],
                    )
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                if not rows:
                    cur = con.execute(
                        """
                        SELECT doc_id, generation, agent_role, doc_type, title, content,
                               tags, verified_c_rule, metrics_json, created_at
                        FROM theories
                        WHERE doc_type = 'REFLECTION_REPORT'
                        ORDER BY created_at DESC
                        """
                    )
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                latest_by_role: dict[str, dict[str, Any]] = {}
                for row in rows:
                    r = dict(zip(cols, row))
                    role = str(r.get("agent_role", ""))
                    if role and role not in latest_by_role:
                        latest_by_role[role] = r
                return latest_by_role
            except Exception:
                return {}
            finally:
                con.close()

    def count(self) -> int:
        with _THEORY_LOCK:
            con = connect_db(self.db_path, read_only=True)
            try:
                row = con.execute("SELECT COUNT(*) FROM theories").fetchone()
                return int(row[0]) if row else 0
            except Exception:
                return 0
            finally:
                con.close()
