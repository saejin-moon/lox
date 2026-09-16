#!/usr/bin/env python3
"""
ETL Pipeline: NetHack 3.6.6 Wiki Indexer.
Streams MediaWiki XML dump, sanitizes NetHack 3.7 tags,
resolves redirects, and indexes into SQLite FTS5 with BM25.
"""

import os
import re
import sys
import time
import gzip
import sqlite3
import xml.etree.ElementTree as ET
import mwparserfromhell

RAW_DUMP_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/nethackwiki_current.xml.gz")
DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/wiki_index.db")

SKIP_PREFIXES = (
    "Talk:", "User:", "User talk:", "File:", "File talk:",
    "Template:", "Template talk:", "Category talk:", "NetHackWiki:",
    "NetHackWiki talk:", "Help:", "MediaWiki:", "MediaWiki talk:",
    "Module:", "Module talk:"
)

REDIRECT_REGEX = re.compile(r"#REDIRECT\s*\[\[(.*?)\]\]", re.IGNORECASE)
SECTION_37_REGEX = re.compile(r"==+\s*(?:In\s+)?NetHack\s+3\.7(?:\.\d+)?\s*==+.*?(?===+|$)", re.IGNORECASE | re.DOTALL)
TEMPLATE_37_REGEX = re.compile(r"\{\{in37\d?\}\}", re.IGNORECASE)


def sanitize_nethack_366(raw_text: str) -> str:
    """
    Strips NetHack 3.7 development sections and templates to preserve pure 3.6.6 rules.
    """
    # 1. Remove sections specifically titled NetHack 3.7
    text = SECTION_37_REGEX.sub("", raw_text)
    # 2. Remove in370 inline templates
    text = TEMPLATE_37_REGEX.sub("", text)
    
    # 3. Strip MediaWiki markup to clean text
    try:
        parsed = mwparserfromhell.parse(text)
        # Strip code/templates into clean text
        clean_text = parsed.strip_code(normalize_whitespace=True)
    except Exception:
        # Fallback regex cleaner if mwparser fails on strange syntax
        clean_text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        clean_text = re.sub(r"\{\{.*?\}\}", "", clean_text)
        clean_text = re.sub(r"'{2,5}", "", clean_text)
    
    # Normalize excessive newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return clean_text


def build_database(dump_path: str = RAW_DUMP_PATH, db_path: str = DB_PATH):
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"Missing raw dump: {dump_path}. Run download_wiki.py first.")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        print(f"[*] Removing existing database: {db_path}")
        os.remove(db_path)

    print(f"[*] Initializing SQLite database at {db_path}...")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Fast ingest pragmas
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
    cur.execute("PRAGMA cache_size = 100000;")

    # Schema creation
    cur.execute("""
    CREATE TABLE articles (
        id INTEGER PRIMARY KEY,
        title TEXT UNIQUE COLLATE NOCASE,
        content TEXT,
        length INTEGER
    );
    """)

    cur.execute("""
    CREATE VIRTUAL TABLE wiki_fts USING fts5(
        title,
        content,
        content='articles',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    """)

    cur.execute("""
    CREATE TABLE redirects (
        alias TEXT PRIMARY KEY COLLATE NOCASE,
        target_title TEXT COLLATE NOCASE
    );
    """)

    con.commit()

    print(f"[*] Streaming XML from {dump_path}...")
    start_time = time.time()
    total_pages = 0
    article_count = 0
    redirect_count = 0
    skipped_count = 0

    cur.execute("BEGIN TRANSACTION;")

    with gzip.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag.endswith("page"):
                total_pages += 1
                title_el = elem.find("{*}title")
                text_el = elem.find(".//{*}text")

                if title_el is not None and title_el.text:
                    title = title_el.text.strip()
                    raw_text = text_el.text if (text_el is not None and text_el.text) else ""

                    # Skip internal/meta namespaces
                    if title.startswith(SKIP_PREFIXES):
                        skipped_count += 1
                    else:
                        # Check for redirects
                        redirect_match = REDIRECT_REGEX.search(raw_text)
                        if redirect_match:
                            target = redirect_match.group(1).split("#")[0].strip()
                            try:
                                cur.execute("INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);", (title, target))
                                redirect_count += 1
                            except sqlite3.Error:
                                pass
                        elif raw_text:
                            # Standard article -> Clean and index
                            clean = sanitize_nethack_366(raw_text)
                            if len(clean) > 20:  # Skip stub/empty pages
                                try:
                                    cur.execute("INSERT INTO articles (title, content, length) VALUES (?, ?, ?);",
                                                (title, clean, len(clean)))
                                    article_id = cur.lastrowid
                                    cur.execute("INSERT INTO wiki_fts (rowid, title, content) VALUES (?, ?, ?);",
                                                (article_id, title, clean))
                                    article_count += 1
                                except sqlite3.IntegrityError:
                                    pass

                # Memory management: Free parsed XML node
                elem.clear()

                # Periodic progress logging
                if total_pages % 2000 == 0:
                    con.commit()
                    cur.execute("BEGIN TRANSACTION;")
                    elapsed = time.time() - start_time
                    rate = total_pages / (elapsed + 1e-5)
                    print(f"[*] Processed {total_pages:,} pages | {article_count:,} articles | {redirect_count:,} redirects ({rate:.0f} p/s)...", end="\r")

    con.commit()
    
    print("\n[*] Optimizing FTS5 index and building B-tree indices...")
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('optimize');")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_redirects_alias ON redirects(alias);")
    con.commit()
    con.close()

    elapsed = time.time() - start_time
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n[✓] Knowledge Engine Built Successfully!")
    print(f"    - Total Pages Processed: {total_pages:,}")
    print(f"    - Clean Articles Indexed: {article_count:,}")
    print(f"    - Redirects Mapped: {redirect_count:,}")
    print(f"    - Skipped Meta Pages: {skipped_count:,}")
    print(f"    - Database Size: {db_size_mb:.1f} MB")
    print(f"    - Build Time: {elapsed:.1f}s")


if __name__ == "__main__":
    build_database()
