"""
Read-only DuckDB, NetHack Wiki, C-Symbol Table, Theory Store, and Expert Parity tools.
Includes a strict single-pass SQL lexer and DuckDB `enable_external_access = false`
isolation to prevent SQL injection, DDL/DML mutations, and arbitrary host file reads.
"""

from __future__ import annotations

import os
import re
from typing import Any

from lox.datasets.symbols import SYMBOL_REGISTRY
from lox.db.schema import connect_db

_ALLOWED_LEADING_KEYWORDS = frozenset({"select", "with", "describe", "show"})

_FORBIDDEN_SQL_KEYWORDS = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "attach",
        "detach",
        "copy",
        "merge",
        "export",
        "import",
        "install",
        "load",
        "set",
        "pragma",
        "call",
        "vacuum",
        "checkpoint",
        "prepare",
        "execute",
        "deallocate",
        "use",
    }
)

_FORBIDDEN_TABLE_FUNCTIONS = frozenset(
    {
        "read_text",
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_blob",
        "read_xlsx",
        "glob",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "parquet_file_metadata",
        "parquet_kv_metadata",
        "sniff_csv",
        "sqlite_attach",
        "sqlite_scan",
        "postgres_attach",
        "postgres_scan",
        "mysql_attach",
        "iceberg_scan",
        "delta_scan",
        "http_get",
        "st_read",
        "duckdb_secrets",
        "duckdb_settings",
        "duckdb_databases",
        "duckdb_extensions",
        "pragma_database_list",
        "current_setting",
        "getenv",
    }
)


def strip_sql_comments_and_literals(sql: str) -> str:
    """
    Single-pass SQL lexer that strips `--` line comments and `/* ... */` block comments
    (handling nested block comments) and replaces single-quoted `'...'` (with `''` escapes)
    and double-quoted `"..."` identifiers with neutral placeholders so keyword and semicolon
    validation cannot be bypassed via comments or escaped quotes.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        # Single-line comment -- ... \n
        if ch == "-" and nxt == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue
        # Block comment /* ... */ (supports nesting as DuckDB does)
        if ch == "/" and nxt == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and sql[i] == "/" and sql[i + 1] == "*":
                    depth += 1
                    i += 2
                elif i + 1 < n and sql[i] == "*" and sql[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            out.append(" ")
            continue
        # Single-quoted string literal '...' with '' escape
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("''")
            continue
        # Double-quoted identifier "..." with "" escape.
        # Preserve forbidden system/table-function names (e.g. "read_text"(...))
        # so quoting cannot bypass _FORBIDDEN_TABLE_FUNCTIONS checks, while keeping
        # ordinary quoted column identifiers harmless.
        if ch == '"':
            i += 1
            ident_chars: list[str] = []
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        ident_chars.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                ident_chars.append(sql[i])
                i += 1
            ident_val = "".join(ident_chars)
            if ident_val.lower() in _FORBIDDEN_TABLE_FUNCTIONS:
                out.append(f" {ident_val} ")
            else:
                out.append('""')
            continue
        # Dollar-quoted string $$...$$ or $tag$...$tag$
        if ch == "$":
            m_tag = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
            if m_tag:
                tag = m_tag.group(0)
                end_idx = sql.find(tag, i + len(tag))
                if end_idx == -1:
                    i = n
                else:
                    i = end_idx + len(tag)
                out.append("''")
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def validate_readonly_sql(sql: str) -> tuple[str | None, str, str]:
    """
    Validates that `sql` is a single read-only query.
    Returns `(error_message_or_None, cleaned_sql, leading_keyword)`.
    """
    raw = str(sql or "").strip()
    if not raw:
        return "Error: Empty SQL query. Provide a single SELECT statement.", "", ""

    stripped_for_check = strip_sql_comments_and_literals(raw).strip()
    # Allow a single trailing semicolon, reject any interior semicolon
    stripped_no_trailing = stripped_for_check.rstrip()
    if stripped_no_trailing.endswith(";"):
        stripped_no_trailing = stripped_no_trailing[:-1]
    if ";" in stripped_no_trailing:
        return (
            "Error: Multiple SQL statements in a single query_db call are not allowed. "
            "Pass exactly ONE SELECT statement per query_db call.",
            "",
            "",
        )

    low_sanitized = stripped_no_trailing.lower()
    m_lead = re.match(r"^\s*\(*\s*([a-z_][a-z0-9_]*)", low_sanitized)
    lead_kw = m_lead.group(1) if m_lead else ""
    if lead_kw not in _ALLOWED_LEADING_KEYWORDS:
        return (
            f"Error: Only read-only queries starting with SELECT/WITH/DESCRIBE/SHOW are permitted (got '{lead_kw or 'invalid'}').",
            "",
            "",
        )

    for kw in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", low_sanitized):
            return (
                f"Error: Mutative or administrative SQL keyword '{kw.upper()}' is forbidden. Use SELECT only.",
                "",
                "",
            )

    if re.search(r"\breplace\b(?!\s*\()", low_sanitized):
        return (
            "Error: Mutative keyword 'REPLACE' is forbidden. Use SELECT only.",
            "",
            "",
        )

    for fn in _FORBIDDEN_TABLE_FUNCTIONS:
        if re.search(rf"\b{fn}\s*\(", low_sanitized):
            return (
                f"Error: External/system table function '{fn}()' is forbidden in read-only query_db.",
                "",
                "",
            )

    # Trim trailing whitespace/semicolon safely from raw query
    clean_exec = raw.rstrip().rstrip(";").rstrip()
    return None, clean_exec, lead_kw


def format_cursor_yaml(cur: Any, max_rows: int = 40) -> tuple[str, int]:
    if not cur.description:
        return "status: ok", 0
    headers = [d[0] for d in cur.description]
    all_rows = cur.fetchall()
    rows_truncated = len(all_rows) > max_rows
    rows = all_rows[:max_rows]
    if not rows:
        return "results: []", 0
    max_chars = (
        16000
        if len(rows) <= 3
        else (6000 if len(rows) <= 8 else (2500 if len(rows) <= 15 else 900))
    )
    truncated_fields = 0
    lines = ["results:"]
    for r in rows:
        parts = []
        for h, v in zip(headers, r):
            if v is None:
                parts.append(f"{h}: null")
            elif isinstance(v, (int, float, bool)):
                parts.append(f"{h}: {v}")
            elif isinstance(v, (bytes, bytearray)):
                parts.append(f'{h}: "<{len(v)} bytes>"')
            else:
                s = str(v)
                if len(s) > max_chars:
                    truncated_fields += 1
                    s = (
                        s[: max_chars // 2]
                        + f" ... [TRUNCATED {len(s) - max_chars} chars; query single row for more] ... "
                        + s[-max_chars // 2 :]
                    )
                s_esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                parts.append(f'{h}: "{s_esc}"')
        lines.append("  - {" + ", ".join(parts) + "}")
    note = ""
    if rows_truncated:
        note += f"\n# NOTE: Showing {len(rows)} of {len(all_rows)} rows (cap={max_rows}). Use LIMIT/OFFSET or narrower WHERE."
    if truncated_fields:
        note += (
            f"\n# NOTE: {truncated_fields} long field(s) middle-truncated at {max_chars} chars "
            f"(select 1-3 rows for up to 16000 chars, or use SUBSTR(col, offset, 8000) to paginate)."
        )
    return "```yaml\n" + "\n".join(lines) + "\n```" + note, len(rows)


class DBToolsMixin:
    """Read-only database, Wiki, symbol-table, theory, and expert-parity tool methods."""

    db_path: str
    wiki: Any
    theories_store: Any
    generation: int
    wiki_queries_run: int
    theories_queries_run: int
    db_queries_run: int

    def _campaign_tables_empty(self) -> bool:
        """Returns True if both `batches` and `runs` have 0 rows in DuckDB."""
        if not os.path.exists(self.db_path):
            return True
        con = connect_db(self.db_path, read_only=True)
        try:
            b_cnt = int(
                (con.execute("SELECT COUNT(*) FROM batches").fetchone() or [0])[0]
            )
            r_cnt = int((con.execute("SELECT COUNT(*) FROM runs").fetchone() or [0])[0])
            return b_cnt == 0 and r_cnt == 0
        except Exception:
            return True
        finally:
            con.close()

    def query_db(self, sql: str) -> tuple[str, str]:
        """Executes a strictly validated read-only SQL query against DuckDB."""
        if not os.path.exists(self.db_path):
            return "Database not found.", "0 rows"

        err, clean, lead_kw = validate_readonly_sql(sql)
        if err is not None:
            return err, "error"

        low_no_literals = strip_sql_comments_and_literals(clean).lower()
        if self.generation == 1 and self._campaign_tables_empty():
            forbidden_gen1 = re.search(
                r"\b(?:from|join)\s+(runs|ticks|batches|function_stats|policies|experiments|discoveries|champion_runs|champion_ticks|champion_function_stats|latest_pilot_runs|latest_pilot_ticks)\b",
                low_no_literals,
            )
            if forbidden_gen1:
                tbl = forbidden_gen1.group(1)
                return (
                    f"Error: In Generation 1 (before any batch is evaluated), campaign table '{tbl}' is empty (0 rows) and disallowed. "
                    "Query ONLY `expert_runs` and `expert_ticks` (and use `query_wiki` / `query_monster` / `query_item`).",
                    "error",
                )

        auto_limited = lead_kw in ("select", "with") and not re.search(
            r"\blimit\b", low_no_literals
        )
        max_rows_cap = 40
        if auto_limited:
            clean += " LIMIT 30"
        else:
            m_lim = re.search(r"\blimit\s+(\d+)\b", low_no_literals)
            if m_lim:
                max_rows_cap = max(40, min(200, int(m_lim.group(1))))

        con = connect_db(self.db_path, read_only=True)
        try:
            # Defense-in-depth at C engine level: disable all external file/network access
            try:
                con.execute("SET enable_external_access = false")
            except Exception as sec_exc:
                return (
                    f"Error: Failed to enforce DuckDB external-access sandbox: {sec_exc}",
                    "error",
                )
            cur = con.execute(clean)
            self.db_queries_run += 1
            text, count = format_cursor_yaml(cur, max_rows=max_rows_cap)
            if count == 0:
                hint = self._build_zero_row_hint(con, clean, low_no_literals)
                if hint:
                    text += f"\n# NOTE: {hint}"
            elif auto_limited and count >= 30:
                text += "\n# NOTE: Implicit LIMIT 30 applied; specify explicit LIMIT/OFFSET to paginate."
            return text, f"{count} rows"
        except Exception as exc:
            err_msg = f"SQL Error: {exc}"
            try:
                err_low = str(exc).lower()
                if "binder error" in err_low or "not found" in err_low:
                    tbl_matches = re.findall(
                        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", low_no_literals
                    )
                    col_hints: list[str] = []
                    for tbl in list(dict.fromkeys(tbl_matches))[:3]:
                        try:
                            cols = [
                                str(r[1])
                                for r in con.execute(
                                    f"PRAGMA table_info('{tbl}')"
                                ).fetchall()
                            ]
                            if cols:
                                col_hints.append(f"{tbl}({', '.join(cols)})")
                        except Exception:
                            pass
                    if col_hints:
                        err_msg += "\nValid schema columns: " + " | ".join(col_hints)
                elif "catalog error" in err_low:
                    tbls = [str(r[0]) for r in con.execute("SHOW TABLES").fetchall()]
                    if tbls:
                        err_msg += f"\nAvailable tables: {', '.join(tbls)}"
            except Exception:
                pass
            return err_msg, "sql_error"
        finally:
            con.close()

    def _build_zero_row_hint(self, con: Any, raw_sql: str, low_no_literals: str) -> str:
        """Explains common 0-row causes (in-progress generation filter or pruned old ticks)."""
        try:
            max_gen_row = con.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM runs"
            ).fetchone()
            latest_gen = int(max_gen_row[0] or 0) if max_gen_row else 0
            if self.generation > 0 and latest_gen < self.generation:
                cur_gen = int(self.generation)
                if (
                    re.search(rf"\bgeneration\s*=\s*{cur_gen}\b", low_no_literals)
                    or f"g{cur_gen}" in raw_sql.lower()
                ):
                    return (
                        f"Gen {cur_gen} has not completed yet (latest completed generation is Gen {latest_gen}). "
                        f"Query `champion_runs` / `champion_ticks` / `champion_function_stats` or `WHERE generation = {latest_gen}`."
                    )
            if re.search(r"\bfrom\s+ticks\b", low_no_literals):
                rid_m = re.search(r"run_id\s*=\s*'([^']+)'", raw_sql)
                if rid_m:
                    rid = rid_m.group(1)
                    r_gen = con.execute(
                        "SELECT generation FROM runs WHERE run_id = ? LIMIT 1", [rid]
                    ).fetchone()
                    min_t = con.execute(
                        "SELECT MIN(r.generation) FROM ticks t JOIN runs r ON t.run_id = r.run_id"
                    ).fetchone()
                    min_t_gen = (
                        int(min_t[0] or 0) if min_t and min_t[0] is not None else 0
                    )
                    if r_gen and min_t_gen and int(r_gen[0]) < min_t_gen:
                        return (
                            f"`{rid}` is from Gen {int(r_gen[0])}, whose raw ticks were pruned "
                            f"(ticks retained for Gen >= {min_t_gen}). Query `champion_ticks` or runs from Gen {latest_gen}."
                        )
        except Exception:
            pass
        return ""

    def query_wiki(self, query: str, top_k: int = 3) -> tuple[str, str]:
        """Searches the offline NetHack 3.6.6 Wiki."""
        self.wiki_queries_run += 1
        k = max(1, min(6, int(top_k or 3)))
        hits = self.wiki.search_wiki(query, top_k=k)
        return self.wiki.query(query, top_k=k), f"{len(hits)} articles"

    def read_wiki_article(
        self, title: str, section: str = "", offset: int = 0, max_chars: int = 0
    ) -> tuple[str, str]:
        """Reads bounded article or section slices from the offline NetHackWiki + Guidebook index."""
        self.wiki_queries_run += 1
        req_cap = int(max_chars or 0)
        safe_cap = 16000 if req_cap <= 0 else min(24000, max(500, req_cap))
        out = self.wiki.read_article(
            title=title, section=section, offset=offset, max_chars=safe_cap
        )
        return out, f"article:{title[:32]}"

    def query_monster(self, name: str) -> tuple[str, str]:
        """Queries C-level permonst ground truth struct for a monster."""
        import yaml

        self.wiki_queries_run += 1
        data = SYMBOL_REGISTRY.query_monster(name)
        if data:
            return (
                "```yaml\n" + yaml.dump(data, default_flow_style=False) + "\n```",
                "found",
            )
        matches = SYMBOL_REGISTRY.search_monsters(name, limit=5)
        if matches:
            return (
                "Monster not found exact. Closest matches:\n```yaml\n"
                + yaml.dump(matches, default_flow_style=False)
                + "\n```",
                f"{len(matches)} matches",
            )
        return (
            f"Monster '{name}' not found in NetHack 3.6.6 permonst symbol table.",
            "not_found",
        )

    def query_item(self, name: str) -> tuple[str, str]:
        """Queries C-level objclass ground truth struct for an item."""
        import yaml

        self.wiki_queries_run += 1
        data = SYMBOL_REGISTRY.query_item(name)
        if data:
            return (
                "```yaml\n" + yaml.dump(data, default_flow_style=False) + "\n```",
                "found",
            )
        matches = SYMBOL_REGISTRY.search_items(name, limit=5)
        if matches:
            return (
                "Item not found exact. Closest matches:\n```yaml\n"
                + yaml.dump(matches, default_flow_style=False)
                + "\n```",
                f"{len(matches)} matches",
            )
        return (
            f"Item '{name}' not found in NetHack 3.6.6 objclass symbol table.",
            "not_found",
        )

    def query_theories(
        self, query: str = "", doc_type: str = "", limit: int = 6
    ) -> tuple[str, str]:
        """Searches DuckDB `theories` knowledge store."""
        self.theories_queries_run += 1
        if self.theories_store is None:
            return "Theories store not attached in this session.", "unavailable"
        try:
            lim = max(1, min(15, int(limit or 6)))
            dt = str(doc_type or "").strip() or None
            q = str(query or "").strip() or "failure stall depth"
            docs = self.theories_store.search_theories(q, limit=lim, doc_type=dt)
            if not docs:
                return "No matching theory documents found.", "0 docs"
            blocks = []
            for d in docs:
                blocks.append(
                    f"### [Gen {d.get('generation', '?')} | {d.get('doc_type', 'DOC')} | {d.get('agent_role', '')}] "
                    f"{d.get('title', '')}\n{str(d.get('content', '')).strip()}"
                )
            return "\n\n---\n\n".join(blocks), f"{len(docs)} docs"
        except Exception as exc:
            return f"Theories query error: {exc}", "error"

    def benchmark_expert_parity(self, top_n: int = 10) -> tuple[str, str]:
        """Compares champion `runs` vs `expert_runs` across progression, survival, and telemetry metrics."""
        if not os.path.exists(self.db_path):
            return "Database not found.", "not_found"
        con = connect_db(self.db_path, read_only=True)
        try:
            try:
                champ = con.execute(
                    """
                    SELECT COUNT(*) AS n,
                           ROUND(AVG(max_depth), 2) AS avg_depth,
                           MAX(max_depth) AS max_depth,
                           ROUND(AVG(turns), 1) AS avg_turns,
                           ROUND(AVG(score), 1) AS avg_score,
                           ROUND(AVG(final_ac), 2) AS avg_ac,
                           ROUND(AVG(xp_level), 2) AS avg_xl,
                           ROUND(AVG(worn_oprop_count), 2) AS avg_oprop,
                           ROUND(AVG(dungeons_visited), 2) AS avg_branches,
                           ROUND(AVG(unique_tiles), 1) AS avg_tiles
                    FROM champion_runs
                    """
                ).fetchall()
            except Exception:
                champ = []
            try:
                exp = con.execute(
                    """
                    SELECT COUNT(*) AS n,
                           ROUND(AVG(max_depth), 2) AS avg_depth,
                           MAX(max_depth) AS max_depth,
                           ROUND(AVG(turns), 1) AS avg_turns,
                           ROUND(AVG(score), 1) AS avg_score
                    FROM expert_runs
                    """
                ).fetchall()
            except Exception:
                exp = []
            try:
                lim = max(1, min(30, int(top_n or 10)))
                early_expert = con.execute(
                    """
                    SELECT run_id, role, max_depth, turns, score, summary_digest
                    FROM expert_runs
                    ORDER BY turns DESC
                    LIMIT ?
                    """,
                    [lim],
                ).fetchall()
            except Exception:
                early_expert = []
        finally:
            con.close()
        out_parts = ["### Champion vs Expert Parity Snapshot"]
        if champ and champ[0][0]:
            r = champ[0]
            out_parts.append(
                f"- Champion (`n={r[0]}`): avg_depth={r[1]}, max_depth={r[2]}, avg_turns={r[3]}, "
                f"avg_score={r[4]}, avg_ac={r[5]}, avg_xl={r[6]}, avg_oprop={r[7]}, "
                f"avg_branches={r[8]}, avg_tiles={r[9]}"
            )
        else:
            out_parts.append("- Champion: no `champion_runs` recorded yet.")
        if exp and exp[0][0]:
            er = exp[0]
            out_parts.append(
                f"- Human Experts (`n={er[0]}`): avg_depth={er[1]}, max_depth={er[2]}, avg_turns={er[3]}, avg_score={er[4]}"
            )
        if early_expert:
            out_parts.append(f"- Top {len(early_expert)} Expert Reference Runs:")
            for row in early_expert[:10]:
                digest_line = str(row[5] or "").replace("\n", " ")[:180]
                out_parts.append(
                    f"  * `{row[0]}` ({row[1]}, Dlvl {row[2]}, turns={row[3]}, score={row[4]}): {digest_line}"
                )
        return "\n".join(out_parts), "parity ok"
