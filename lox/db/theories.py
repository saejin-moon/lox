"""
Embedded Document DB for LOX using LanceDB.
Stores qualitative hypotheses, mechanistic theories, evaluation summaries, and reflection reports
with native Tantivy BM25 full-text keyword search and vector embedding support.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lancedb
from lancedb.index import FTS

DEFAULT_THEORIES_PATH = "data/workspace/theories.lance"


class TheoryDocumentStore:
    """
    Embedded document database for qualitative theories, failure post-mortems,
    and agent reflection reports.
    """

    TABLE_NAME = "theories"

    def __init__(self, db_path: str = DEFAULT_THEORIES_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self._table = None
        self._init_table()

    def _init_table(self) -> None:
        tables = (
            list(self.db.list_tables())
            if hasattr(self.db, "list_tables")
            else self.db.table_names()
        )
        if self.TABLE_NAME in tables:
            self._table = self.db.open_table(self.TABLE_NAME)
        else:
            # Initialize with seed schema record
            seed_data = [
                {
                    "doc_id": "seed_000",
                    "generation": 0,
                    "agent_role": "system",
                    "doc_type": "META_SYNTHESIS",
                    "title": "System Initialization",
                    "content": "Theory document store initialized for LOX multi-agent swarm.",
                    "tags": "init,system",
                    "verified_c_rule": "",
                    "metrics_json": "{}",
                    "created_at": time.time(),
                }
            ]
            self._table = self.db.create_table(self.TABLE_NAME, data=seed_data)
            try:
                self._table.create_index("content", config=FTS())
            except Exception:
                pass

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
        Saves a qualitative theory, failure report, or synthesis document.
        Automatically indexes content for BM25 full-text search.
        """
        if not doc_id:
            doc_id = f"doc_{agent_role}_{int(time.time() * 1000)}"

        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        metrics_str = json.dumps(metrics) if metrics else "{}"

        record = {
            "doc_id": str(doc_id),
            "generation": int(generation),
            "agent_role": str(agent_role),
            "doc_type": str(doc_type),
            "title": str(title),
            "content": str(content),
            "tags": tags_str,
            "verified_c_rule": str(verified_c_rule),
            "metrics_json": metrics_str,
            "created_at": float(time.time()),
        }

        if self._table is None:
            self._init_table()

        assert self._table is not None
        self._table.add([record])
        return doc_id

    def search_theories(
        self,
        query: str,
        limit: int = 5,
        agent_role: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Performs BM25 full-text search over document contents using Tantivy FTS.
        Filters by agent_role and/or doc_type if provided.
        """
        if self._table is None:
            return []

        clean_q = str(query).strip()
        if not clean_q:
            return []

        try:
            builder = self._table.search(clean_q, query_type="fts")
            filter_clauses = []
            if agent_role:
                filter_clauses.append(f"agent_role = '{agent_role}'")
            if doc_type:
                filter_clauses.append(f"doc_type = '{doc_type}'")

            if filter_clauses:
                builder = builder.where(" AND ".join(filter_clauses))

            raw_hits = builder.limit(limit).to_list()
            # Clean up internal scoring fields
            hits = []
            for h in raw_hits:
                d = dict(h)
                d.pop("_score", None)
                hits.append(d)
            return hits
        except Exception:
            # Fallback to in-memory substring filtering if FTS index needs sync
            try:
                all_docs = self._table.to_arrow().to_pylist()
                filtered = []
                for d in all_docs:
                    if agent_role and d.get("agent_role") != agent_role:
                        continue
                    if doc_type and d.get("doc_type") != doc_type:
                        continue
                    if any(
                        term.lower() in str(d.get("content", "")).lower()
                        for term in clean_q.split()
                    ):
                        filtered.append(d)
                return filtered[:limit]
            except Exception:
                return []

    def get_latest_reports(
        self,
        generation: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Retrieves the latest REFLECTION_REPORT for each agent_role.
        Returns a mapping of {agent_role: doc_dict}.
        """
        if self._table is None:
            return {}

        try:
            records = self._table.to_arrow().to_pylist()
            if not records:
                return {}

            reports = [r for r in records if r.get("doc_type") == "REFLECTION_REPORT"]
            if generation is not None:
                gen_reports = [r for r in reports if r.get("generation") == generation]
                if gen_reports:
                    reports = gen_reports

            if not reports:
                return {}

            # Sort descending by created_at
            reports.sort(key=lambda r: float(r.get("created_at", 0.0)), reverse=True)
            latest_by_role: dict[str, dict[str, Any]] = {}
            for r in reports:
                role = str(r.get("agent_role", ""))
                if role and role not in latest_by_role:
                    latest_by_role[role] = r

            return latest_by_role
        except Exception:
            return {}

    def count(self) -> int:
        if self._table is None:
            return 0
        try:
            return len(self._table)
        except Exception:
            return 0
