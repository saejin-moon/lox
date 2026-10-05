#!/usr/bin/env python3
"""
Downloads MiniHack's NetHack wiki dataset and builds data/wiki_index.db
with full SQLite FTS5 BM25 full-text search and alias redirects.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess

DATA_DIR = "data"
WIKI_DB_PATH = os.path.join(DATA_DIR, "wiki_index.db")
WIKI_JSON_PATH = os.path.join(DATA_DIR, "nethackwikidata.json")
DROPBOX_URL = "https://www.dropbox.com/s/6qbfmsr3l89sip0/nethackwikidata.json?dl=1"


def download_wiki_json() -> str:
    """Downloads nethackwikidata.json via curl if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(WIKI_JSON_PATH) and os.path.getsize(WIKI_JSON_PATH) > 1000000:
        print(
            f"[WIKI BUILD] Found existing {WIKI_JSON_PATH} ({os.path.getsize(WIKI_JSON_PATH):,} bytes)"
        )
        return WIKI_JSON_PATH

    print(f"[WIKI BUILD] Curling NetHack wiki dataset from {DROPBOX_URL}...")
    cmd = [
        "curl",
        "-L",
        "-s",
        "-S",
        "--max-time",
        "120",
        "-o",
        WIKI_JSON_PATH,
        DROPBOX_URL,
    ]
    subprocess.run(cmd, check=True)
    print(
        f"[WIKI BUILD] Successfully curled {WIKI_JSON_PATH} ({os.path.getsize(WIKI_JSON_PATH):,} bytes)"
    )
    return WIKI_JSON_PATH


def build_database(json_path: str, db_path: str) -> None:
    """Populates SQLite FTS5 database from wiki JSON."""
    print(f"[WIKI BUILD] Building SQLite FTS5 index at {db_path}...")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    # Schema matches lox.author.wiki.WikiEngine
    cur.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT UNIQUE COLLATE NOCASE, content TEXT);"
    )
    cur.execute(
        "CREATE TABLE redirects (alias TEXT PRIMARY KEY COLLATE NOCASE, target_title TEXT COLLATE NOCASE);"
    )
    cur.execute(
        "CREATE VIRTUAL TABLE wiki_fts USING fts5(title, content, content='articles', content_rowid='id');"
    )

    articles_inserted = 0
    redirects_inserted = 0

    common_aliases = {
        "boh": "Bag of holding",
        "bag of holding": "Bag of holding",
        "excal": "Excalibur",
        "excalibur": "Excalibur",
        "mr": "Magic resistance",
        "magic resistance": "Magic resistance",
        "poison resistance": "Poison resistance",
        "poison res": "Poison resistance",
        "teleport": "Teleportation",
        "teleportation": "Teleportation",
        "elbereth": "Elbereth",
        "floating eye": "Floating eye",
        "gas spore": "Gas spore",
        "cockatrice": "Cockatrice",
        "werejackal": "Werejackal",
        "soldier ant": "Soldier ant",
        "killer bee": "Killer bee",
        "gray dragon scale mail": "Gray dragon scale mail",
        "silver dragon scale mail": "Silver dragon scale mail",
        "sokoban": "Sokoban",
        "mines": "Gnomish Mines",
        "gnomish mines": "Gnomish Mines",
    }
    for alias, target in common_aliases.items():
        cur.execute(
            "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
            (alias, target),
        )
        redirects_inserted += 1

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = (
                item.get("wikipedia_title", "").strip() or item.get("title", "").strip()
            )
            text_data = item.get("text", "")
            if isinstance(text_data, list):
                content = "".join(text_data)
            elif isinstance(text_data, str):
                content = text_data
            else:
                content = ""

            if not title or not content:
                continue

            # Check for MediaWiki redirects: #REDIRECT [[Target]]
            if content.strip().upper().startswith("#REDIRECT"):
                start = content.find("[[")
                end = content.find("]]")
                if start != -1 and end != -1 and end > start + 2:
                    target = content[start + 2 : end].strip()
                    cur.execute(
                        "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
                        (title, target),
                    )
                    redirects_inserted += 1
                continue

            try:
                cur.execute(
                    "INSERT OR IGNORE INTO articles (title, content) VALUES (?, ?);",
                    (title, content),
                )
                if cur.rowcount > 0:
                    articles_inserted += 1
            except sqlite3.Error:
                pass

    print(
        f"[WIKI BUILD] Inserted {articles_inserted:,} articles and {redirects_inserted:,} redirects."
    )
    print("[WIKI BUILD] Populating FTS5 full-text index...")
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('rebuild');")
    con.commit()

    # Optimize index
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('optimize');")
    con.commit()
    con.close()

    # VACUUM in autocommit mode
    vac_con = sqlite3.connect(db_path, isolation_level=None)
    vac_con.execute("VACUUM;")
    vac_con.close()

    db_size = os.path.getsize(db_path)
    print(
        f"[WIKI BUILD] Complete! {db_path} created successfully ({db_size / (1024 * 1024):.1f} MB)."
    )


def main():
    json_path = download_wiki_json()
    build_database(json_path, WIKI_DB_PATH)


if __name__ == "__main__":
    main()
