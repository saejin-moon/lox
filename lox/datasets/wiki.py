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
    "slashthem",
    "splicehack",
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
    "nethack: the next generation",
    "notdnethack",
    "hack'em",
)


def _is_variant_title(title: str) -> bool:
    tl = title.lower()
    return any(v in tl for v in VARIANT_KEYWORDS)


def _clean_wiki_markup(text: str) -> str:
    txt = str(text or "").strip()
    txt = (
        txt.replace("BULLET::::-", "- ")
        .replace("codice_1", "`...`")
        .replace("codice_2", "`...`")
    )
    return txt.replace("{{", "").replace("}}", "")


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
            subprocess.run(
                [sys.executable, build_script, "--db-path", str(db_path)],
                check=True,
                cwd=repo_root,
            )
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

    def _resolve_redirect(
        self, cur: sqlite3.Cursor, query: str, max_hops: int = 5
    ) -> str:
        current = query
        seen: set[str] = {current.lower()}
        for _ in range(max_hops):
            cur.execute(
                "SELECT target_title FROM redirects WHERE alias = ? COLLATE NOCASE LIMIT 1;",
                (current,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                break
            nxt = str(row[0])
            if nxt.lower() in seen:
                break
            seen.add(nxt.lower())
            current = nxt
        return current

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
        results: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        try:
            cur = con.cursor()

            # 1. Exact title lookup or alias redirect
            exact_hit = self.exact_title_lookup(cur, query.strip())
            if exact_hit:
                results.append(exact_hit)
                seen_titles.add(exact_hit["title"].lower())
                if len(results) >= top_k:
                    return results

            # 2. Tokenize and filter low-information stopwords
            raw_tokens = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in query.split()]
            tokens = [t for t in raw_tokens if len(t) >= 2]
            content_tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
            if not content_tokens:
                content_tokens = tokens
            if not content_tokens:
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
                        a.content AS full_text
                    FROM wiki_fts
                    JOIN articles a ON a.id = wiki_fts.rowid
                    WHERE wiki_fts MATCH ?
                      AND a.title NOT LIKE 'Forum:%'
                    ORDER BY score ASC
                    LIMIT ?;
                    """,
                    (f'"{clean_phrase}"', top_k * 6),
                )
                phrase_rows = cur.fetchall()
                phrase_rows.sort(
                    key=lambda r: (
                        1 if _is_variant_title(r[0]) else 0,
                        1 if r[0].lower().startswith("strategy:") else 0,
                        r[2],
                    )
                )
                for r in phrase_rows:
                    if r[0].lower() not in seen_titles:
                        results.append(
                            {
                                "title": r[0],
                                "snippet": r[1],
                                "score": r[2],
                                "text": r[3],
                                "is_exact": False,
                            }
                        )
                        seen_titles.add(r[0].lower())
                        if len(results) >= top_k:
                            break
            except sqlite3.OperationalError:
                pass

            # Strategy B: All-terms AND match with prefix wildcards
            if len(results) < top_k and len(unique_tokens) > 1:
                and_query = " AND ".join(f"{t}*" for t in unique_tokens)
                try:
                    cur.execute(
                        """
                        SELECT
                            a.title,
                            snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                            bm25(wiki_fts, 10.0, 1.0) AS score,
                            a.content AS full_text
                        FROM wiki_fts
                        JOIN articles a ON a.id = wiki_fts.rowid
                        WHERE wiki_fts MATCH ?
                          AND a.title NOT LIKE 'Forum:%'
                        ORDER BY score ASC
                        LIMIT ?;
                        """,
                        (and_query, top_k * 8),
                    )
                    and_rows = cur.fetchall()
                    and_rows.sort(
                        key=lambda r: (
                            1 if _is_variant_title(r[0]) else 0,
                            1 if r[0].lower().startswith("strategy:") else 0,
                            r[2],
                        )
                    )
                    for r in and_rows:
                        if r[0].lower() not in seen_titles:
                            results.append(
                                {
                                    "title": r[0],
                                    "snippet": r[1],
                                    "score": r[2],
                                    "text": r[3],
                                    "is_exact": False,
                                }
                            )
                            seen_titles.add(r[0].lower())
                            if len(results) >= top_k:
                                break
                except sqlite3.OperationalError:
                    pass

            # Strategy C: BM25 ranked OR search with prefix wildcards
            if len(results) < top_k and unique_tokens:
                or_query = " OR ".join(f"{t}*" for t in unique_tokens)
                try:
                    cur.execute(
                        """
                        SELECT
                            a.title,
                            snippet(wiki_fts, 1, '«', '»', '...', 25) AS match_snippet,
                            bm25(wiki_fts, 10.0, 1.0) AS score,
                            a.content AS full_text
                        FROM wiki_fts
                        JOIN articles a ON a.id = wiki_fts.rowid
                        WHERE wiki_fts MATCH ?
                          AND a.title NOT LIKE 'Forum:%'
                        ORDER BY score ASC
                        LIMIT ?;
                        """,
                        (or_query, top_k * 8),
                    )
                    rows = cur.fetchall()
                    # Prioritize canonical vanilla articles over variant forks
                    rows.sort(
                        key=lambda r: (
                            1 if _is_variant_title(r[0]) else 0,
                            1 if r[0].lower().startswith("strategy:") else 0,
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
                                    "text": r[3],
                                    "is_exact": False,
                                }
                            )
                            seen_titles.add(r[0].lower())
                            if len(results) >= top_k:
                                break
                except sqlite3.OperationalError:
                    pass
        finally:
            con.close()
        return results

    def read_article(
        self,
        title: str,
        section: str = "",
        offset: int = 0,
        max_chars: int = 0,
    ) -> str:
        """
        Reads the full untruncated NetHack wiki/guidebook/spoiler article for `title`
        (resolving redirects), with optional section filtering or offset/max_chars windowing.
        """
        if not os.path.exists(self.db_path):
            return f"Wiki database not found at '{self.db_path}'."
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            hit = self.exact_title_lookup(cur, title.strip())
            if hit is None:
                cur.execute(
                    "SELECT title, content FROM articles WHERE title LIKE ? COLLATE NOCASE ORDER BY LENGTH(title) ASC LIMIT 1;",
                    (f"%{title.strip()}%",),
                )
                row = cur.fetchone()
                if not row:
                    return f"No NetHack wiki article found matching '{title}'."
                hit = {"title": row[0], "text": row[1]}
        finally:
            con.close()

        art_title = str(hit["title"])
        body = _clean_wiki_markup(str(hit.get("text") or hit.get("preview") or ""))
        if section.strip():
            sec_l = section.strip().lower()
            lines = body.splitlines()
            uses_mediawiki_headings = any(
                ln.strip().startswith("==") and ln.strip().endswith("==")
                for ln in lines
            )
            start_idx = None
            end_idx = len(lines)
            sec_level = 0
            for idx_l, ln in enumerate(lines):
                s = ln.strip()
                is_heading = False
                lvl = 0
                if s.startswith("==") and s.endswith("=="):
                    lvl = len(s) - len(s.lstrip("="))
                    hdr_txt = s.strip("= ").lower()
                    is_heading = True
                elif not uses_mediawiki_headings and s.startswith("#"):
                    lvl = len(s) - len(s.lstrip("#"))
                    hdr_txt = s.lstrip("# ").lower()
                    is_heading = True
                else:
                    hdr_txt = s.lower()
                if start_idx is None:
                    if (is_heading and sec_l in hdr_txt) or (
                        not is_heading and sec_l == hdr_txt
                    ):
                        start_idx = idx_l
                        sec_level = lvl or 2
                elif is_heading and lvl <= sec_level:
                    end_idx = idx_l
                    break
            if start_idx is not None:
                body = "\n".join(lines[start_idx:end_idx]).strip()

        off = max(0, int(offset or 0))
        if off > 0:
            body = body[off:]
        limit = max(0, int(max_chars or 0))
        if limit > 0 and len(body) > limit:
            body = (
                body[:limit]
                + f"\n... [TRUNCATED: {len(body) - limit} chars remaining; call read_wiki_article(title='{art_title}', offset={off + limit})]"
            )
        return f"### {art_title}\n```text\n{body}\n```"

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
            raw_body = h.get("text") or h.get("preview") or ""
            cleaned = _clean_wiki_markup(raw_body)
            # Deliver exact matches and primary (#1) hit completely untruncated;
            # secondary hits preserve clean depth with explicit pointer to read_wiki_article.
            if h.get("is_exact") or idx == 1 or len(cleaned) <= 4000:
                lines.append(f"```text\n{cleaned}\n```")
            else:
                lines.append(
                    f"```text\n{cleaned[:3000]}\n... [Showing 3000/{len(cleaned)} chars; call read_wiki_article('{h['title']}') for full article]\n```"
                )

        return "\n".join(lines)
