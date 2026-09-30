"""
LOX 2.0 DuckDB Analytical Tooling for LLM Policy Authoring.
Provides safe, read-only analytical functions allowing the LLM to inspect historical
gameplay data, test failure hypotheses, and query death/action statistics.
"""
from __future__ import annotations

import os
import re
from typing import Any
import duckdb


def _format_table(cursor) -> str:
    """Formats DuckDB cursor results into a clean markdown table with zero pandas dependency."""
    if not cursor.description:
        return "Query executed successfully."
    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return "Query returned 0 rows."
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(val) if val is not None else "-" for val in r) + " |" for r in rows]
    return "\n".join([header_line, separator] + row_lines)


class DuckDBToolRegistry:
    """Registry of safe, read-only analytical tools for the Author Agent."""

    def __init__(self, db_path: str = "data/lox.duckdb"):
        self.db_path = db_path

    def get_duckdb_schema(self) -> str:
        """
        Returns the database tables, column names, types, and current row counts.
        """
        if not os.path.exists(self.db_path):
            return "DuckDB database not found. No telemetry has been consolidated yet."

        con = duckdb.connect(self.db_path, read_only=True)
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            con.close()
            return "DuckDB database is empty (no tables found)."

        lines = ["### DuckDB Telemetry Schema:"]
        for (tname,) in tables:
            count = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            cols = con.execute(f"DESCRIBE {tname}").fetchall()
            col_desc = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
            lines.append(f"- **`{tname}`** ({count} rows): {col_desc}")
        con.close()
        return "\n".join(lines)

    def query_duckdb(self, sql: str) -> str:
        """
        Executes a read-only SQL query against the DuckDB database and returns markdown results.
        Mutative statements (INSERT, UPDATE, DELETE, DROP, ALTER) are strictly rejected.
        """
        if not os.path.exists(self.db_path):
            return "DuckDB database does not exist yet."

        clean_sql = sql.strip().rstrip(";")
        low = clean_sql.lower()
        disallowed = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "replace", "execute"]
        for word in disallowed:
            if re.search(rf"\b{word}\b", low):
                return f"Error: Mutative statement '{word.upper()}' is forbidden. Only SELECT queries are permitted."

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            if "limit" not in low:
                clean_sql += " LIMIT 50"
            cur = con.execute(clean_sql)
            res = _format_table(cur)
            con.close()
            return res
        except Exception as e:
            con.close()
            return f"SQL Execution Error: {e}"

    def get_death_taxonomy(self, window: int = 20) -> str:
        """
        Summarizes the top death reasons, frequencies, average depth, and average turns survived.
        """
        if not os.path.exists(self.db_path):
            return "No database found."

        con = duckdb.connect(self.db_path, read_only=True)
        query = f"""
            SELECT death_reason, COUNT(*) as fatalities, ROUND(AVG(depth), 2) as avg_depth, ROUND(AVG(turns), 1) as avg_turns
            FROM episodes
            GROUP BY death_reason
            ORDER BY fatalities DESC
            LIMIT {max(1, window)}
        """
        try:
            cur = con.execute(query)
            res = _format_table(cur)
            con.close()
            return "### Top Fatalities Taxonomy:\n" + res
        except Exception as e:
            con.close()
            return f"Error querying death taxonomy: {e}"

    def get_floor_pacing_stats(self, depth: int = 1) -> str:
        """
        Returns floor pacing statistics for a given dungeon depth: turn count distribution and survival rates.
        """
        if not os.path.exists(self.db_path):
            return "No database found."

        con = duckdb.connect(self.db_path, read_only=True)
        query = f"""
            SELECT 
                COUNT(*) as total_episodes,
                ROUND(AVG(turns), 1) as avg_total_turns,
                MAX(turns) as max_turns,
                SUM(CASE WHEN depth > {depth} THEN 1 ELSE 0 END) as descended_past_floor,
                ROUND(AVG(score), 1) as avg_score
            FROM episodes
            WHERE max_depth >= {depth}
        """
        try:
            cur = con.execute(query)
            res = _format_table(cur)
            con.close()
            return f"### Floor Pacing Stats (Depth >= {depth}):\n" + res
        except Exception as e:
            con.close()
            return f"Error querying pacing stats: {e}"

    def get_action_distribution(self, run_id: str = "") -> str:
        """
        Returns action distribution and percentage of turns spent searching vs moving vs attacking.
        """
        if not os.path.exists(self.db_path):
            return "No database found."

        con = duckdb.connect(self.db_path, read_only=True)
        where = f"WHERE run_id = '{run_id}'" if run_id else ""
        query = f"""
            SELECT 
                action, 
                COUNT(*) as count, 
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
            FROM ticks
            {where}
            GROUP BY action
            ORDER BY count DESC
        """
        try:
            cur = con.execute(query)
            res = _format_table(cur)
            con.close()
            return "### Action Distribution:\n" + res
        except Exception as e:
            con.close()
            return f"Error querying action distribution: {e}"


# Tool Schema for OpenAI / OpenRouter function calling
OPENAI_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "query_duckdb",
            "description": "Execute a read-only SQL query against data/lox.duckdb (tables: episodes, ticks, events).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute, e.g. 'SELECT death_reason, count(*) FROM episodes GROUP BY 1 ORDER BY 2 DESC LIMIT 10'",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_duckdb_schema",
            "description": "View tables, columns, and data types in data/lox.duckdb.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_death_taxonomy",
            "description": "Get summary table of top death causes, frequencies, and avg depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "integer",
                        "description": "Maximum number of distinct causes to return (default 20).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_action_distribution",
            "description": "Get action frequencies and percentage distribution across ticks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Optional run_id to filter actions by. If omitted, aggregates across all runs.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_floor_pacing_stats",
            "description": "Get pacing statistics for a given dungeon floor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Dungeon depth (default 1).",
                    }
                },
            },
        },
    },
]
