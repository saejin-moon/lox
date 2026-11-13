"""
LOX Offline NetHack 3.6.6 Wiki Knowledge Engine.
Performs sub-5ms BM25 full-text search with match highlighting,
alias redirect resolution, and exact article lookups against data/wiki_index.db.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from typing import Any

DEFAULT_WIKI_DB = "data/wiki_index.db"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
    "dont",
    "have",
    "go",
    "get",
    "got",
    "can",
    "cant",
    "cannot",
    "do",
    "does",
    "did",
}

VARIANT_KEYWORDS = (
    "slash'em",
    "slashem",
    "dnethack",
    "unnethack",
    "fiqhack",
    "sporkhack",
    "grunthack",
    "xnethack",
    "evilhack",
    "dynahack",
    "nh4",
    "fourk",
)


def ensure_wiki_ready(db_path: str = DEFAULT_WIKI_DB) -> bool:
    """
    Ensures that data/wiki_index.db exists and is populated.
    If missing, automatically builds it using scripts/build_wiki_db.py
    (pure NetHack 3.6.6 encyclopedia without hardcoded strategy guides).
    """
    if os.path.exists(db_path) and os.path.getsize(db_path) > 100000:
        return True

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    build_script = os.path.join(repo_root, "scripts", "build_wiki_db.py")

    if os.path.exists(build_script):
        try:
            print(
                "[LOX] `data/wiki_index.db` not found. Automatically building offline NetHack Wiki database..."
            )
            subprocess.run([sys.executable, build_script], check=True, cwd=repo_root)
            return True
        except Exception as exc:
            print(f"[LOX] Warning: Could not auto-build full wiki index ({exc}).")
            return False
    return False


class WikiEngine:
    """Offline search engine for NetHack 3.6.6 encyclopedia."""

    def __init__(self, db_path: str = DEFAULT_WIKI_DB):
        self.db_path = db_path
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) < 10000:
            ensure_wiki_ready(self.db_path)

    def _resolve_redirect(self, cur: sqlite3.Cursor, query: str) -> str:
        cur.execute(
            "SELECT target_title FROM redirects WHERE alias = ? COLLATE NOCASE LIMIT 1;",
            (query,),
        )
        row = cur.fetchone()
        return row[0] if row else query

    def exact_title_lookup(
        self, cur: sqlite3.Cursor, title: str
    ) -> dict[str, Any] | None:
        resolved_title = self._resolve_redirect(cur, title)
        cur.execute(
            "SELECT title, content FROM articles WHERE title = ? COLLATE NOCASE LIMIT 1;",
            (resolved_title,),
        )
        row = cur.fetchone()
        if row:
            return {"title": row[0], "text": row[1], "score": 0.0, "is_exact": True}
        return None

    def search_wiki(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        results: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        # 1. Exact title lookup or alias redirect
        exact_hit = self.exact_title_lookup(cur, query.strip())
        if exact_hit:
            results.append(exact_hit)
            seen_titles.add(exact_hit["title"].lower())
            if len(results) >= top_k:
                con.close()
                return results

        # 2. Tokenize and filter low-information stopwords
        raw_tokens = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in query.split()]
        tokens = [t for t in raw_tokens if len(t) >= 2]
        content_tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
        if not content_tokens:
            content_tokens = tokens
        if not content_tokens:
            con.close()
            return results

        # Deduplicate content tokens while preserving order
        unique_tokens = list(dict.fromkeys(content_tokens))

        # Strategy A: Exact phrase match
        clean_phrase = " ".join(content_tokens)
        try:
            cur.execute(
                """
                SELECT
                    a.title,
                    snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                    bm25(wiki_fts, 10.0, 1.0) AS score,
                    substr(a.content, 1, 1400) AS preview
                FROM wiki_fts
                JOIN articles a ON a.id = wiki_fts.rowid
                WHERE wiki_fts MATCH ?
                  AND a.title NOT LIKE 'Forum:%'
                  AND a.title NOT LIKE '%/%'
                ORDER BY score ASC
                LIMIT ?;
                """,
                (f'"{clean_phrase}"', top_k),
            )
            for r in cur.fetchall():
                if r[0].lower() not in seen_titles:
                    results.append(
                        {
                            "title": r[0],
                            "snippet": r[1],
                            "score": r[2],
                            "preview": r[3],
                            "is_exact": False,
                        }
                    )
                    seen_titles.add(r[0].lower())
        except sqlite3.OperationalError:
            pass

        # Strategy B: BM25 ranked OR search with prefix wildcards (scores by term relevance across all articles)
        if len(results) < top_k and unique_tokens:
            or_query = " OR ".join(f"{t}*" for t in unique_tokens)
            try:
                cur.execute(
                    """
                    SELECT
                        a.title,
                        snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                        bm25(wiki_fts, 10.0, 1.0) AS score,
                        substr(a.content, 1, 1400) AS preview
                    FROM wiki_fts
                    JOIN articles a ON a.id = wiki_fts.rowid
                    WHERE wiki_fts MATCH ?
                      AND a.title NOT LIKE 'Forum:%'
                      AND a.title NOT LIKE '%/%'
                    ORDER BY score ASC
                    LIMIT ?;
                    """,
                    (or_query, top_k * 8),
                )
                rows = cur.fetchall()
                # Prioritize canonical vanilla articles over variant forks
                rows.sort(
                    key=lambda r: (
                        1 if any(v in r[0].lower() for v in VARIANT_KEYWORDS) else 0,
                        r[2],
                    )
                )
                for r in rows:
                    if r[0].lower() not in seen_titles:
                        results.append(
                            {
                                "title": r[0],
                                "snippet": r[1],
                                "score": r[2],
                                "preview": r[3],
                                "is_exact": False,
                            }
                        )
                        seen_titles.add(r[0].lower())
                        if len(results) >= top_k:
                            break
            except sqlite3.OperationalError:
                pass

        con.close()
        return results

    def query(self, query_str: str, top_k: int = 3) -> str:
        hits = self.search_wiki(query_str, top_k=top_k)
        if not hits:
            return f"No NetHack wiki articles found matching '{query_str}'."

        lines = [f"### NetHack Wiki Results for '{query_str}':"]
        for idx, h in enumerate(hits, 1):
            tag = (
                "[Exact Match]"
                if h.get("is_exact")
                else f"[BM25: {h.get('score', 0):.2f}]"
            )
            lines.append(f"#### {idx}. {h['title']} {tag}")
            if h.get("snippet"):
                lines.append(f"> {h['snippet']}")
            preview_clean = (h.get("preview") or h.get("text") or "")[:1000].strip()
            preview_clean = preview_clean.replace("{{", "").replace("}}", "")
            lines.append(f"```text\n{preview_clean}\n```")

        return "\n".join(lines)
