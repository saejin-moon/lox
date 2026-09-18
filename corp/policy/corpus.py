"""
CORP-Ω Policy Layer: per-domain RAG corpus (R5, AGENT_PLAN step 14).

Each domain gets an FTS5 corpus at data/corpus/<domain>/corpus.db built from its docs
(NetHack: the wiki_index.db article set; Craftax: official docs + paper + docstrings).
`rag_slices()` returns k chunks for the author prompt. Corpus coverage is checked
before author prompts go live (RAG on/off is a paper ablation).
"""
from __future__ import annotations

import os
import sqlite3

CORPUS_ROOT = "data/corpus"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150


def corpus_path(domain: str) -> str:
    return os.path.join(CORPUS_ROOT, domain, "corpus.db")


def build_corpus(domain: str, documents: list[tuple[str, str]],
                 replace: bool = True) -> int:
    """Builds (or extends) the domain corpus from (title, text) documents.
    Texts are chunked with overlap for RAG slicing. Returns the number of chunks."""
    path = corpus_path(domain)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if replace and os.path.exists(path):
        os.unlink(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, title TEXT, chunk_idx INTEGER, text TEXT)")
    cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, text,
                   content='docs', content_rowid='id')""")
    n = 0
    for title, text in documents:
        for idx, chunk in enumerate(_chunks(text)):
            if len(chunk.strip()) < 40:
                continue
            cur.execute("INSERT INTO docs (title, chunk_idx, text) VALUES (?, ?, ?)",
                        (title, idx, chunk))
            cur.execute("INSERT INTO docs_fts (rowid, title, text) VALUES (?, ?, ?)",
                        (cur.lastrowid, title, chunk))
            n += 1
    con.commit()
    con.close()
    return n


def _chunks(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        yield text
        return
    start = 0
    while start < len(text):
        yield text[start:start + size]
        start += size - overlap


def rag_slices(domain: str, query: str, k: int = 4) -> list[str]:
    """Returns up to k matched chunk slices for the author prompt; [] when the corpus
    is absent or the query is empty (RAG-off ablation)."""
    path = corpus_path(domain)
    if not query or not os.path.exists(path):
        return []
    con = sqlite3.connect(path)
    cur = con.cursor()
    try:
        # Sanitize: FTS5 MATCH treats quotes/punctuation as syntax; keep word chars
        safe = " ".join(w for w in query.replace('"', " ").replace("'", " ").split()
                        if w.isalnum())
        if not safe:
            return []
        rows = cur.execute(
            """SELECT title, text FROM docs_fts WHERE docs_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (safe, k),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return [f"[{title}] {text}" for title, text in rows]


def corpus_coverage(domain: str) -> dict:
    """Coverage check (§7 verification discipline): doc count, chunk count, sample
    query smoke. Runs before author prompts go live."""
    path = corpus_path(domain)
    if not os.path.exists(path):
        return {"domain": domain, "exists": False}
    con = sqlite3.connect(path)
    cur = con.cursor()
    docs = cur.execute("SELECT COUNT(DISTINCT title), COUNT(*) FROM docs").fetchone()
    probe = []
    for term in ("survival", "monster", "staircase", "the"):
        probe = rag_slices(domain, term, k=1)
        if probe:
            break
    con.close()
    return {"domain": domain, "exists": True, "documents": docs[0], "chunks": docs[1],
            "probe_query_nonempty": bool(probe)}


# ---------------------------------------------------------------------------
# Domain corpus builders
# ---------------------------------------------------------------------------

def build_nethack_corpus(top_k_articles: int = 40) -> int:
    """Extracts the highest-value NetHack wiki articles from data/wiki_index.db into
    the RAG corpus (survival/tactics/combat mechanics the author LLM needs)."""
    wiki = "data/wiki_index.db"
    if not os.path.exists(wiki):
        raise FileNotFoundError("data/wiki_index.db missing — cannot build nethack corpus")
    con = sqlite3.connect(wiki)
    cur = con.cursor()
    cur.execute("SELECT title, content FROM articles ORDER BY LENGTH(content) DESC")
    rows = cur.fetchall()
    con.close()

    # Prioritize mechanics-bearing articles by title keywords, then longest articles
    keywords = ("monster", "attack", "weapon", "armor", "potion", "scroll", "wand",
                "ring", "spell", "prayer", "sacrifice", "elbereth", "poison",
                "resistance", "hunger", "stairs", "dungeon", "branch", "luck")
    scored = []
    for title, content in rows:
        score = sum(1 for kw in keywords if kw in title.lower()) * 100 + min(len(content), 20000) / 200
        scored.append((score, title, content))
    scored.sort(reverse=True)
    docs = [(t, c) for _, t, c in scored[:top_k_articles]]
    return build_corpus("nethack", docs)


def build_minihack_corpus() -> int:
    """MiniHack = NetHack wiki + MiniHack task docs (the obs/action interface is NLE's,
    so NetHack knowledge transfers; the task docs add reward structure)."""
    docs: list[tuple[str, str]] = [
        ("MiniHack: tasks", (
            "MiniHack is a framework of easily customizable NetHack-based reinforcement "
            "learning environments built on the NetHack Learning Environment (NLE). "
            "Observations include glyphs, chars (character map), blstats (bottom line "
            "stats: hit points, hunger, position), and message. Episodes end on death, "
            "task completion, or step limit. ExploreMaze tasks reward the agent for "
            "reaching the staircase down (the '>' tile). Mapped variants reveal the "
            "full map; unmapped require exploration. Actions are the NLE compass moves "
            "plus interactions. Seeds fully control dungeon generation."
        )),
        ("MiniHack: ExploreMaze", (
            "The ExploreMaze task family requires navigating a procedural maze to reach "
            "the exit staircase ('>'). The agent spawns away from the stairs; walls are "
            "stone '#', walkable floor is '.', corridors '#' may connect rooms, and "
            "doors '+' block until opened. Reward is granted on reaching the staircase. "
            "Hard variants use larger mazes with sparse connectivity; mapped variants "
            "reveal the whole dungeon on the character screen from turn one."
        )),
    ]
    # NetHack wiki knowledge transfers (same engine)
    try:
        wiki = "data/wiki_index.db"
        if os.path.exists(wiki):
            con = sqlite3.connect(wiki)
            cur = con.cursor()
            for kw in ("stairs", "door", "Elbereth", "hunger", "monster"):
                cur.execute("SELECT title, content FROM articles WHERE title LIKE ? LIMIT 3",
                            (f"%{kw}%",))
                docs.extend(cur.fetchall())
            con.close()
    except Exception:
        pass
    return build_corpus("minihack", docs)


def build_craftax_corpus() -> int:
    """Craftax = official docs + paper + source docstrings + achievement table
    (self-built and load-bearing — no community wiki exists). Requires the craftax
    package installed; raises FileNotFoundError otherwise."""
    try:
        import craftax  # noqa: F401,PLC0415
    except ImportError:
        raise FileNotFoundError(
            "craftax not installed — install it (uv add craftax) before building its corpus")
    import craftax, inspect, os  # noqa: PLC0415
    root = os.path.dirname(inspect.getfile(craftax))
    docs: list[tuple[str, str]] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                try:
                    src = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                doc = " ".join(l.strip()[3:] for l in src.splitlines()
                               if l.strip().startswith('"""') or (l.strip().startswith("#") and "achievement" in l.lower()))
                if doc:
                    docs.append((os.path.relpath(p, root), doc[:6000]))
    return build_corpus("craftax", docs)


if __name__ == "__main__":
    import json
    for domain, builder in (("nethack", build_nethack_corpus), ("minihack", build_minihack_corpus)):
        try:
            n = builder()
            print(domain, builder and "corpus:", n, "chunks |", json.dumps(corpus_coverage(domain)))
        except FileNotFoundError as e:
            print(domain, "skipped:", e)