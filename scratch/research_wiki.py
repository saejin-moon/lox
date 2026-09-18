import sqlite3
import re

con = sqlite3.connect('data/wiki_index.db')
cur = con.cursor()

def get_article(title):
    # Resolve redirect
    cur.execute("SELECT target_title FROM redirects WHERE alias = ? COLLATE NOCASE;", (title,))
    row = cur.fetchone()
    target = row[0] if row else title
    cur.execute("SELECT title, content FROM articles WHERE title = ? COLLATE NOCASE;", (target,))
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    return None, None

def search_snippets(term, limit=5):
    cur.execute("""
        SELECT a.title, snippet(wiki_fts, 1, '«', '»', '...', 25)
        FROM wiki_fts
        JOIN articles a ON a.id = wiki_fts.rowid
        WHERE wiki_fts MATCH ?
          AND a.title NOT LIKE 'Forum:%' AND a.title NOT LIKE '%/%'
        LIMIT ?;
    """, (term, limit))
    return cur.fetchall()

print("--- Checking 'Standard strategy' ---")
t, c = get_article('Standard strategy')
if c:
    print(f"Title: {t}, Length: {len(c)}")
    # Print headings
    headings = re.findall(r'(==+[^=]+==+)', c)
    print("Headings:", headings[:15])

print("\n--- Checking 'Experience level' ---")
t, c = get_article('Experience level')
if c:
    print(f"Title: {t}, Length: {len(c)}")
    headings = re.findall(r'(==+[^=]+==+)', c)
    print("Headings:", headings[:15])

print("\n--- Checking 'Skill' ---")
t, c = get_article('Skill')
if c:
    print(f"Title: {t}, Length: {len(c)}")
    headings = re.findall(r'(==+[^=]+==+)', c)
    print("Headings:", headings[:15])

print("\n--- Checking 'Protection racket' ---")
t, c = get_article('Protection racket')
if c:
    print(f"Title: {t}, Length: {len(c)}")
    headings = re.findall(r'(==+[^=]+==+)', c)
    print("Headings:", headings[:15])
