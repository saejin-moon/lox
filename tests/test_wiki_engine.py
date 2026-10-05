import os
import sqlite3
import tempfile

import pytest

from lox.author.wiki import WikiEngine


@pytest.fixture
def wiki_engine():
    """Provides a WikiEngine with either real data or a fast hermetic mock SQLite FTS5 database."""
    if os.path.exists("data/wiki_index.db"):
        yield WikiEngine(db_path="data/wiki_index.db")
        return

    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    con = sqlite3.connect(tmp_db)
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, content TEXT)"
    )
    cur.execute("CREATE TABLE redirects (alias TEXT, target_title TEXT)")
    cur.execute(
        "CREATE VIRTUAL TABLE wiki_fts USING fts5(title, content, content='articles', content_rowid='id')"
    )

    cur.execute(
        "INSERT INTO articles VALUES (1, 'Excalibur', 'Excalibur is a lawful artifact weapon based on a long sword.')"
    )
    cur.execute(
        "INSERT INTO articles VALUES (2, 'Cockatrice', 'A cockatrice is a monster that can inflict stoning.')"
    )
    cur.execute(
        "INSERT INTO articles VALUES (3, 'Bag of holding', 'A bag of holding is a magical container.')"
    )
    cur.execute(
        "INSERT INTO articles VALUES (4, 'Poison resistance', 'Poison resistance can be gained by eating a poison resistance corpse such as a killer bee.')"
    )

    cur.execute("INSERT INTO redirects VALUES ('BoH', 'Bag of holding')")
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('rebuild')")
    con.commit()
    con.close()

    try:
        yield WikiEngine(db_path=tmp_db)
    finally:
        if os.path.exists(tmp_db):
            os.remove(tmp_db)


def test_wiki_engine_lookups(wiki_engine):
    wiki = wiki_engine

    # 1. Exact match lookup
    res_excal = wiki.query("Excalibur", top_k=1)
    assert "Excalibur" in res_excal
    assert "artifact weapon" in res_excal or "long sword" in res_excal

    # 2. Exact monster lookup
    res_cockatrice = wiki.query("Cockatrice", top_k=1)
    assert "Cockatrice" in res_cockatrice
    assert "stoning" in res_cockatrice.lower() or "monster" in res_cockatrice.lower() or "stone" in res_cockatrice.lower()

    # 3. Alias redirect lookup
    res_boh = wiki.query("BoH", top_k=1)
    assert "bag of holding" in res_boh.lower() or "abbreviations" in res_boh.lower()

    # 4. BM25 full-text query
    res_poison = wiki.query("poison resistance corpse", top_k=2)
    assert len(res_poison) > 100
    assert "poison resistance" in res_poison.lower()
