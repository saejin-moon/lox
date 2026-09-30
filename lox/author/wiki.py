"""
LOX 2.0 Offline NetHack 3.6.6 Wiki Knowledge Engine.
Performs sub-5ms BM25 full-text search with match highlighting,
alias redirect resolution, and exact article lookups against data/wiki_index.db.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any


DEFAULT_WIKI_DB = "data/wiki_index.db"


class WikiEngine:
    """Offline search engine for NetHack 3.6.6 encyclopedia."""

    def __init__(self, db_path: str = DEFAULT_WIKI_DB):
        self.db_path = db_path

    def _resolve_redirect(self, cur: sqlite3.Cursor, query: str) -> str:
        """Checks if query or title is an alias (e.g. 'BoH' -> 'Bag of holding')."""
        cur.execute("SELECT target_title FROM redirects WHERE alias = ? COLLATE NOCASE LIMIT 1;", (query,))
        row = cur.fetchone()
        return row[0] if row else query

    def exact_title_lookup(self, cur: sqlite3.Cursor, title: str) -> dict[str, Any] | None:
        """Performs an O(1) exact article lookup by title or alias."""
        resolved_title = self._resolve_redirect(cur, title)
        cur.execute("SELECT title, content FROM articles WHERE title = ? COLLATE NOCASE LIMIT 1;", (resolved_title,))
        row = cur.fetchone()
        if row:
            return {"title": row[0], "text": row[1], "score": 0.0, "is_exact": True}
        return None

    def search_wiki(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """
        Executes sub-5ms BM25 search across NetHack articles.
        Returns top matching articles with preview snippets.
        """
        if not os.path.exists(self.db_path):
            return []

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        results: list[dict[str, Any]] = []

        # 1. Exact match check
        exact_hit = self.exact_title_lookup(cur, query.strip())
        if exact_hit:
            results.append(exact_hit)
            if len(results) >= top_k:
                con.close()
                return results

        # 2. Tokenize and sanitize for FTS5
        clean_query = query.replace('"', '""')
        tokens = [t for t in clean_query.split() if t.isalnum()]
        if not tokens:
            con.close()
            return results

        fts_query = f'"{clean_query}" OR (' + " AND ".join(tokens) + ")"

        try:
            cur.execute(
                """
                SELECT 
                    a.title,
                    snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                    bm25(wiki_fts, 10.0, 1.0) AS score,
                    substr(a.content, 1, 1200) AS preview
                FROM wiki_fts
                JOIN articles a ON a.id = wiki_fts.rowid
                WHERE wiki_fts MATCH ?
                  AND a.title NOT LIKE 'Forum:%'
                  AND a.title NOT LIKE '%/%'
                ORDER BY score ASC
                LIMIT ?;
                """,
                (fts_query, top_k + 3),
            )
            for r in cur.fetchall():
                if exact_hit and r[0].lower() == exact_hit["title"].lower():
                    continue
                results.append({
                    "title": r[0],
                    "snippet": r[1],
                    "score": r[2],
                    "preview": r[3],
                    "is_exact": False,
                })
                if len(results) >= top_k:
                    break

        except sqlite3.OperationalError:
            # Fallback simple prefix query
            prefix_query = " ".join(f"{t}*" for t in tokens)
            cur.execute(
                """
                SELECT 
                    a.title,
                    snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                    bm25(wiki_fts, 10.0, 1.0) AS score,
                    substr(a.content, 1, 1200) AS preview
                FROM wiki_fts
                JOIN articles a ON a.id = wiki_fts.rowid
                WHERE wiki_fts MATCH ?
                  AND a.title NOT LIKE '%/%'
                ORDER BY score ASC
                LIMIT ?;
                """,
                (prefix_query, top_k),
            )
            for r in cur.fetchall():
                results.append({
                    "title": r[0],
                    "snippet": r[1],
                    "score": r[2],
                    "preview": r[3],
                    "is_exact": False,
                })

        con.close()
        return results

    def query(self, query_str: str, top_k: int = 2) -> str:
        """User/LLM-facing formatted markdown search result."""
        hits = self.search_wiki(query_str, top_k=top_k)
        if not hits:
            return f"No NetHack wiki articles found matching '{query_str}'."

        lines = [f"### NetHack Wiki Results for '{query_str}':"]
        for idx, h in enumerate(hits, 1):
            tag = "[Exact Match]" if h.get("is_exact") else f"[Score: {h.get('score', 0):.2f}]"
            lines.append(f"#### {idx}. {h['title']} {tag}")
            if h.get("snippet"):
                lines.append(f"> {h['snippet']}")
            preview_clean = (h.get("preview") or h.get("text") or "")[:800].strip()
            # Clean up excessive mediawiki templates for LLM readability
            preview_clean = preview_clean.replace("{{", "").replace("}}", "")
            lines.append(f"```\n{preview_clean}\n```\n")

        return "\n".join(lines)
