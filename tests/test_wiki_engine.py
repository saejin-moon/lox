import os
import pytest
from lox.author.wiki import WikiEngine


def test_wiki_engine_lookups():
    wiki = WikiEngine(db_path="data/wiki_index.db")
    if not os.path.exists("data/wiki_index.db"):
        pytest.skip("data/wiki_index.db not present locally")

    # 1. Exact match lookup
    res_excal = wiki.query("Excalibur", top_k=1)
    assert "Excalibur" in res_excal
    assert "artifact weapon" in res_excal or "long sword" in res_excal

    # 2. Exact monster lookup
    res_cockatrice = wiki.query("Cockatrice", top_k=1)
    assert "Cockatrice" in res_cockatrice
    assert "stoning" in res_cockatrice or "monster" in res_cockatrice

    # 3. Alias redirect lookup
    res_boh = wiki.query("BoH", top_k=1)
    assert "bag of holding" in res_boh.lower() or "abbreviations" in res_boh.lower()

    # 4. BM25 full-text query
    res_poison = wiki.query("poison resistance corpse", top_k=2)
    assert len(res_poison) > 100
    assert "poison resistance" in res_poison.lower()
