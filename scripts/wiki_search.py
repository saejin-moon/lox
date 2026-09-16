#!/usr/bin/env python3
"""
CLI Query Tool: Offline NetHack 3.6.6 Knowledge Engine.
Performs sub-5ms BM25 full-text search with match highlighting,
alias redirect resolution, and exact article lookups.
"""

import os
import sys
import time
import argparse
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/wiki_index.db")


def resolve_redirect(cur: sqlite3.Cursor, query: str) -> str:
    """Checks if query or title is an alias (e.g. 'BoH' -> 'Bag of holding')."""
    cur.execute("SELECT target_title FROM redirects WHERE alias = ? COLLATE NOCASE LIMIT 1;", (query,))
    row = cur.fetchone()
    return row[0] if row else query


def exact_title_lookup(cur: sqlite3.Cursor, title: str) -> dict | None:
    """Performs an O(1) exact article lookup by title or alias."""
    resolved_title = resolve_redirect(cur, title)
    cur.execute("SELECT title, content FROM articles WHERE title = ? COLLATE NOCASE LIMIT 1;", (resolved_title,))
    row = cur.fetchone()
    if row:
        return {"title": row[0], "text": row[1], "score": 0.0, "is_exact": True}
    return None


def search_wiki(query: str, top_k: int = 3, db_path: str = DB_PATH) -> list[dict]:
    if not os.path.exists(db_path):
        print(f"[!] Database not found at {db_path}. Run 'uv run python scripts/etl/build_wiki_index.py' first.")
        sys.exit(1)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    results = []
    
    # 1. Check exact match or redirect first
    exact_hit = exact_title_lookup(cur, query)
    if exact_hit:
        results.append(exact_hit)
        if len(results) >= top_k:
            con.close()
            return results

    # 2. Sanitize query for FTS5 syntax
    clean_query = query.replace('"', '""')
    tokens = [t for t in clean_query.split() if t.isalnum()]
    if not tokens:
        con.close()
        return results

    # Formulate FTS5 expression: prioritize phrase + individual tokens
    fts_query = f'"{clean_query}" OR (' + " AND ".join(tokens) + ")"

    try:
        cur.execute("""
        SELECT 
            a.title,
            snippet(wiki_fts, 1, '«', '»', '...', 30) AS match_snippet,
            bm25(wiki_fts, 10.0, 1.0) AS score,
            substr(a.content, 1, 1500) AS preview
        FROM wiki_fts
        JOIN articles a ON a.id = wiki_fts.rowid
        WHERE wiki_fts MATCH ?
          AND a.title NOT LIKE 'Forum:%'
          AND a.title NOT LIKE '%/%'
        ORDER BY score ASC
        LIMIT ?;
        """, (fts_query, top_k + 5))

        rows = cur.fetchall()
        for r in rows:
            if exact_hit and r[0].lower() == exact_hit["title"].lower():
                continue
            results.append({
                "title": r[0],
                "snippet": r[1],
                "score": r[2],
                "preview": r[3],
                "is_exact": False
            })
            if len(results) >= top_k:
                break

    except sqlite3.OperationalError:
        # Fallback to simple token prefix match
        prefix_query = " ".join(f"{t}*" for t in tokens)
        cur.execute("""
        SELECT 
            a.title,
            snippet(wiki_fts, 1, '«', '»', '...', 30) AS match_snippet,
            bm25(wiki_fts, 10.0, 1.0) AS score,
            substr(a.content, 1, 1500) AS preview
        FROM wiki_fts
        JOIN articles a ON a.id = wiki_fts.rowid
        WHERE wiki_fts MATCH ?
          AND a.title NOT LIKE '%/ko'
          AND a.title NOT LIKE '%/zh'
          AND a.title NOT LIKE '%/ru'
          AND a.title NOT LIKE '%/de'
          AND a.title NOT LIKE '%/fr'
          AND a.title NOT LIKE '%/es'
          AND a.title NOT LIKE '%/ja'
        ORDER BY score ASC
        LIMIT ?;
        """, (prefix_query, top_k + 5))
        for r in cur.fetchall():
            results.append({
                "title": r[0],
                "snippet": r[1],
                "score": r[2],
                "preview": r[3],
                "is_exact": False
            })

    con.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Query offline NetHack 3.6.6 Knowledge Engine")
    parser.add_argument("query", type=str, help="Search query (e.g. 'wand of wishing', 'cockatrice', 'PYEC')")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return (default: 3)")
    parser.add_argument("--full", action="store_true", help="Print full article text if exact match found")
    args = parser.parse_args()

    t0 = time.time()
    hits = search_wiki(args.query, top_k=args.top_k)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\n================================================================================")
    print(f" NetHack 3.6.6 Knowledge Retrieval: '{args.query}' [{elapsed_ms:.1f} ms]")
    print(f"================================================================================\n")

    if not hits:
        print("[!] No matching articles found.")
        return

    for idx, hit in enumerate(hits, 1):
        tag = "[EXACT MATCH]" if hit.get("is_exact") else f"[BM25: {hit.get('score', 0.0):.3f}]"
        print(f"### Result {idx}: {hit['title']} {tag}")
        
        if hit.get("snippet"):
            print(f"> **Snippet**: {hit['snippet']}\n")
        
        if args.full and hit.get("text"):
            print(hit["text"])
        elif hit.get("preview"):
            paragraphs = [p.strip() for p in hit["preview"].split("\n\n") if p.strip()]
            preview_text = "\n\n".join(paragraphs[:3])
            print(preview_text)
        elif hit.get("text"):
            paragraphs = [p.strip() for p in hit["text"].split("\n\n") if p.strip()]
            preview_text = "\n\n".join(paragraphs[:3])
            print(preview_text)
            
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
