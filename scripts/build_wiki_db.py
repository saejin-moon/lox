#!/usr/bin/env python3
"""
Builds data/wiki_index.db with full SQLite FTS5 BM25 full-text search and alias redirects
from authoritative NetHack 3.6.6 sources:
1. Official NetHackWiki XML dump (`data/nethackwiki_current.xml.gz`, preserving ASCII maps,
   tables, templates, `#REDIRECT`s, and excluding non-vanilla variant namespaces/titles).
2. Official NetHack 3.6.6 Guidebook (`https://www.nethack.org/v366/Guidebook.html`).
3. Plain-text reference spoilers (`pray.txt`, `wand_id.txt`, `conditions.txt`) optional cache.
4. Falls back cleanly to `data/nethackwikidata.json` if XML dump is unavailable offline.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

DATA_DIR = "data"
WIKI_DB_PATH = os.path.join(DATA_DIR, "wiki_index.db")
WIKI_XML_PATH = os.path.join(DATA_DIR, "nethackwiki_current.xml.gz")
WIKI_JSON_PATH = os.path.join(DATA_DIR, "nethackwikidata.json")
WIKI_XML_URL = "https://archive.alt.org/nethackwiki/nethackwiki_current.xml.gz"
GUIDEBOOK_366_URL = "https://www.nethack.org/v366/Guidebook.html"
DROPBOX_URL = "https://www.dropbox.com/s/6qbfmsr3l89sip0/nethackwikidata.json?dl=1"

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

COMMON_ALIASES: dict[str, str] = {
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
    "rodney": "Wizard of Yendor",
    "two-weapon": "Twoweapon",
    "two-weapon combat": "Twoweapon",
    "intrinsics": "Intrinsic",
    "food poisoning": "Sickness",
    "beatitude": "BUC",
    "oracle": "The Oracle",
    "vlad": "Vlad's Tower",
    "medusa": "Medusa's Island",
    "sanctum": "Moloch's Sanctum",
    "wand of wishing": "Wand of wishing",
    "ascension": "Ascension",
    "standard ascension kit": "Ascension kit",
}

_RE_REPLACECHARS = re.compile(
    r"<replacecharsblock>(.*?)</replacecharsblock>", re.DOTALL | re.IGNORECASE
)
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
_RE_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_MULTI_NL = re.compile(r"\n{3,}")


def _is_variant_title(title: str) -> bool:
    tl = title.lower()
    if tl.startswith("strategy:"):
        return False
    return any(v in tl for v in VARIANT_KEYWORDS)


def _clean_replacechars(match: re.Match[str]) -> str:
    inner = html.unescape(match.group(1))
    out_lines: list[str] = []
    in_map = False
    for ln in inner.splitlines():
        s = ln.rstrip()
        if not in_map:
            if s.strip().startswith("newline="):
                in_map = True
            continue
        cleaned = _RE_BR.sub("\n", s)
        cleaned = _RE_HTML_TAGS.sub("", cleaned)
        out_lines.append(cleaned)
    if not out_lines:
        # Fallback if newline= header absent
        inner_clean = _RE_HTML_TAGS.sub("", _RE_BR.sub("\n", inner))
        return "\n" + inner_clean.strip() + "\n"
    return "\n" + "\n".join(out_lines).strip() + "\n"


def _render_template(tpl: Any) -> str:
    name = str(tpl.name).strip().lower()
    params = [
        html.unescape(str(p.value).strip())
        for p in tpl.params
        if str(p.name).strip().isdigit() is not False or str(p.name).strip().isdigit()
    ]
    pos_params = [
        html.unescape(str(p.value).strip()) for p in tpl.params if not p.showkey
    ]
    if name in ("languages", "tocright", "stub", "nethack36", "shortdescription"):
        return ""
    if name in ("frac", "sfrac") and len(pos_params) >= 2:
        return f"{pos_params[0]}/{pos_params[1]}"
    if name == "monsym" and pos_params:
        return pos_params[-1] or pos_params[0]
    if (
        name in ("item", "monster", "mon", "spell", "object", "obj", "w", "wp")
        and pos_params
    ):
        return pos_params[-1]
    if name in ("refsrc", "ref") and pos_params:
        return f"[{pos_params[0]}]"
    if (
        name
        in (
            "gray",
            "white",
            "lightgray",
            "red",
            "green",
            "blue",
            "yellow",
            "cyan",
            "magenta",
            "brown",
            "orange",
            "purple",
        )
        and pos_params
    ):
        return pos_params[0]
    if name == "nethack":
        return "NetHack"
    if pos_params:
        return " ".join(p for p in pos_params[:3] if p)
    return " ".join(p for p in params[:2] if p)


def clean_mediawiki_text(raw_text: str) -> str:
    """Converts MediaWiki markup to clean readable text preserving ASCII maps & tables."""
    import mwparserfromhell

    txt = _RE_COMMENTS.sub("", raw_text)
    txt = _RE_REF.sub("", txt)
    txt = _RE_REPLACECHARS.sub(_clean_replacechars, txt)
    txt = _RE_BR.sub("\n", txt)

    parsed = mwparserfromhell.parse(txt)
    for tpl in list(parsed.filter_templates(recursive=True)):
        try:
            rep = _render_template(tpl)
            parsed.replace(tpl, rep)
        except Exception:
            pass

    for link in list(parsed.filter_wikilinks(recursive=True)):
        try:
            tgt = str(link.title).strip()
            disp = str(link.text).strip() if link.text else tgt
            if tgt.lower().startswith(("file:", "image:", "category:")):
                parsed.replace(link, "")
            else:
                parsed.replace(link, disp)
        except Exception:
            pass

    out = str(parsed)
    out = html.unescape(out)
    out = re.sub(
        r"</?(?:div|span|tt|code|pre|nowiki|small|big|sup|sub|center|font)[^>]*>",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = out.replace("'''", "").replace("''", "")
    out = _RE_MULTI_NL.sub("\n\n", out)
    return out.strip()


def download_file(url: str, dest_path: str, timeout: int = 120) -> bool:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    cmd = ["curl", "-fsSL", "--max-time", str(timeout), "-o", dest_path, url]
    res = subprocess.run(cmd, check=False)
    return (
        res.returncode == 0
        and os.path.exists(dest_path)
        and os.path.getsize(dest_path) > 1024
    )


def _html_to_text(raw_html: str) -> str:
    txt = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE
    )
    txt = re.sub(r"<(h[1-6]|p|li|tr|br|dt|dd)[^>]*>", "\n", txt, flags=re.IGNORECASE)
    txt = _RE_HTML_TAGS.sub("", txt)
    txt = html.unescape(txt)
    return _RE_MULTI_NL.sub("\n\n", txt).strip()


def _init_sqlite_wiki(db_path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT UNIQUE COLLATE NOCASE, content TEXT);"
    )
    cur.execute(
        "CREATE TABLE redirects (alias TEXT PRIMARY KEY COLLATE NOCASE, target_title TEXT COLLATE NOCASE);"
    )
    cur.execute(
        "CREATE VIRTUAL TABLE wiki_fts USING fts5(title, content, content='articles', content_rowid='id');"
    )
    return con, cur


def _ingest_xml_dump(xml_path: str, cur: sqlite3.Cursor) -> tuple[int, int]:
    articles_inserted = 0
    redirects_inserted = 0
    opener = gzip.open if xml_path.endswith(".gz") else open
    with opener(xml_path, "rb") as f:
        for _ev, el in ET.iterparse(f, events=("end",)):
            if not el.tag.endswith("page"):
                continue
            title_el = next((c for c in el if c.tag.endswith("title")), None)
            ns_el = next((c for c in el if c.tag.endswith("ns")), None)
            text_el = el.find(".//{*}text")
            redir_el = next((c for c in el if c.tag.endswith("redirect")), None)

            title = (title_el.text or "").strip() if title_el is not None else ""
            ns = (ns_el.text or "").strip() if ns_el is not None else "0"
            raw_text = (
                (text_el.text or "") if text_el is not None and text_el.text else ""
            )
            el.clear()

            # Restrict cleanly to main namespace (0) & exclude non-vanilla variants
            if ns != "0" or not title or _is_variant_title(title):
                continue

            if redir_el is not None:
                tgt = redir_el.attrib.get("title", "").strip()
                if tgt and not _is_variant_title(tgt):
                    cur.execute(
                        "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
                        (title, tgt),
                    )
                    redirects_inserted += 1
                continue

            if raw_text.strip().upper().startswith("#REDIRECT"):
                start = raw_text.find("[[")
                end = raw_text.find("]]")
                if start != -1 and end != -1 and end > start + 2:
                    tgt = (
                        raw_text[start + 2 : end]
                        .split("|", 1)[0]
                        .split("#", 1)[0]
                        .strip()
                    )
                    if tgt and not _is_variant_title(tgt):
                        cur.execute(
                            "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
                            (title, tgt),
                        )
                        redirects_inserted += 1
                continue

            if not raw_text.strip():
                continue

            cleaned = clean_mediawiki_text(raw_text)
            if not cleaned:
                continue

            try:
                cur.execute(
                    "INSERT OR REPLACE INTO articles (title, content) VALUES (?, ?);",
                    (title, cleaned),
                )
                if cur.rowcount > 0:
                    articles_inserted += 1
            except sqlite3.Error:
                pass
    return articles_inserted, redirects_inserted


def _ingest_json_dump(json_path: str, cur: sqlite3.Cursor) -> tuple[int, int]:
    articles_inserted = 0
    redirects_inserted = 0
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
            if not title or _is_variant_title(title):
                continue
            text_data = item.get("text", "")
            content = (
                "".join(text_data)
                if isinstance(text_data, list)
                else str(text_data or "")
            )
            if not content:
                continue

            for alias in item.get("aliases") or []:
                al_str = str(alias or "").strip()
                if al_str and al_str.lower() != title.lower():
                    cur.execute(
                        "INSERT OR IGNORE INTO redirects (alias, target_title) VALUES (?, ?);",
                        (al_str, title),
                    )
                    redirects_inserted += 1

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
    return articles_inserted, redirects_inserted


def _ingest_extra_docs(cur: sqlite3.Cursor) -> int:
    added = 0
    guide_cache = os.path.join(DATA_DIR, "guidebook_366.html")
    if os.path.exists(guide_cache) or download_file(
        GUIDEBOOK_366_URL, guide_cache, timeout=20
    ):
        try:
            raw = open(guide_cache, "r", encoding="utf-8", errors="replace").read()
            txt = _html_to_text(raw)
            if len(txt) > 5000:
                cur.execute(
                    "INSERT OR REPLACE INTO articles (title, content) VALUES (?, ?);",
                    ("NetHack 3.6.6 Guidebook", txt),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
                    ("Guidebook", "NetHack 3.6.6 Guidebook"),
                )
                added += 1
        except Exception:
            pass
    return added


def build_database(
    source_path: str,
    db_path: str = WIKI_DB_PATH,
    include_extras: bool = True,
) -> None:
    """Populates SQLite FTS5 database cleanly from XML or JSON NetHack Wiki dump."""
    print(f"[WIKI BUILD] Building SQLite FTS5 index at {db_path} from {source_path}...")
    con, cur = _init_sqlite_wiki(db_path)

    redirects_inserted = 0
    for alias, target in COMMON_ALIASES.items():
        cur.execute(
            "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
            (alias, target),
        )
        redirects_inserted += 1

    if source_path.endswith((".xml", ".xml.gz")):
        art_n, red_n = _ingest_xml_dump(source_path, cur)
    else:
        art_n, red_n = _ingest_json_dump(source_path, cur)

    extra_n = _ingest_extra_docs(cur) if include_extras else 0
    articles_inserted = art_n + extra_n
    redirects_inserted += red_n

    print(
        f"[WIKI BUILD] Inserted {articles_inserted:,} articles (+{extra_n} 3.6.6 docs) "
        f"and {redirects_inserted:,} redirects."
    )
    print("[WIKI BUILD] Populating FTS5 full-text index...")
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('rebuild');")
    con.commit()
    cur.execute("INSERT INTO wiki_fts(wiki_fts) VALUES('optimize');")
    con.commit()
    con.close()

    vac_con = sqlite3.connect(db_path, isolation_level=None)
    vac_con.execute("VACUUM;")
    vac_con.close()

    db_size = os.path.getsize(db_path)
    print(
        f"[WIKI BUILD] Complete! {db_path} created successfully ({db_size / (1024 * 1024):.1f} MB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LOX offline NetHack 3.6.6 Wiki index."
    )
    parser.add_argument(
        "--source",
        choices=["auto", "xml", "json"],
        default="auto",
        help="Input dump format (default: auto, prefers nethackwiki_current.xml.gz).",
    )
    parser.add_argument("--db-path", default=WIKI_DB_PATH)
    parser.add_argument(
        "--download-xml",
        action="store_true",
        help="Download fresh nethackwiki_current.xml.gz from archive.alt.org if missing.",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="Skip fetching NetHack 3.6.6 Guidebook HTML.",
    )
    args = parser.parse_args()

    src = ""
    if args.source in ("auto", "xml"):
        if os.path.exists(WIKI_XML_PATH) and os.path.getsize(WIKI_XML_PATH) > 1_000_000:
            src = WIKI_XML_PATH
        elif args.download_xml and download_file(WIKI_XML_URL, WIKI_XML_PATH):
            src = WIKI_XML_PATH

    if not src and args.source in ("auto", "json"):
        if (
            os.path.exists(WIKI_JSON_PATH)
            and os.path.getsize(WIKI_JSON_PATH) > 1_000_000
        ):
            src = WIKI_JSON_PATH
        elif args.source == "json" and download_file(DROPBOX_URL, WIKI_JSON_PATH):
            src = WIKI_JSON_PATH

    if not src:
        raise FileNotFoundError(
            f"No NetHack wiki dump found at {WIKI_XML_PATH} or {WIKI_JSON_PATH}. Run with --download-xml."
        )

    build_database(src, db_path=args.db_path, include_extras=not args.no_extras)


if __name__ == "__main__":
    main()
