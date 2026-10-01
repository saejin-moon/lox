"""
LOX 2.0 DuckDB Analytical Tooling and Knowledge Retrieval for LLM Policy Authoring.
Provides safe analytical functions (DuckDB SQL, death taxonomy, pacing stats),
offline NetHack 3.6.6 Wiki retrieval (BM25 search), and a macro request queue
for human post-run implementation.
"""
from __future__ import annotations

import os
import re
from typing import Any
import duckdb

from lox.author.wiki import WikiEngine
from lox.author.requests import queue_macro_request


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
    """Registry of analytical tools, wiki retrieval, and macro requests for the Author Agent."""

    def __init__(self, db_path: str = "data/lox.duckdb", wiki_db_path: str = "data/wiki_index.db"):
        self.db_path = db_path
        self.wiki = WikiEngine(db_path=wiki_db_path)

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

        try:
            window_int = int(window)
        except Exception:
            window_int = 20

        con = duckdb.connect(self.db_path, read_only=True)
        query = f"""
            SELECT death_reason, COUNT(*) as fatalities, ROUND(AVG(depth), 2) as avg_depth, ROUND(AVG(turns), 1) as avg_turns
            FROM episodes
            GROUP BY death_reason
            ORDER BY fatalities DESC
            LIMIT {max(1, window_int)}
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

        try:
            depth_int = int(depth)
        except Exception:
            depth_int = 1

        con = duckdb.connect(self.db_path, read_only=True)
        query = f"""
            SELECT 
                COUNT(*) as total_episodes,
                ROUND(AVG(turns), 1) as avg_total_turns,
                MAX(turns) as max_turns,
                SUM(CASE WHEN depth > {depth_int} THEN 1 ELSE 0 END) as descended_past_floor,
                ROUND(AVG(score), 1) as avg_score
            FROM episodes
            WHERE max_depth >= {depth_int}
        """
        try:
            cur = con.execute(query)
            res = _format_table(cur)
            con.close()
            return f"### Floor Pacing Stats (Depth >= {depth_int}):\n" + res
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

    def query_wiki(self, query: str, top_k: int = 2) -> str:
        """
        Performs sub-5ms BM25 full-text search across the offline NetHack 3.6.6 encyclopedia.
        Use to research monster traits, corpses conveying intrinsics, item properties, or dungeon mechanics.
        """
        try:
            top_k_int = int(top_k)
        except Exception:
            top_k_int = 2
        return self.wiki.query(query, top_k=top_k_int)

    def request_macro(
        self,
        macro_name: str,
        rationale: str,
        proposed_interface: str = "",
        priority: str = "medium",
        run_id: str = "synth_session",
    ) -> str:
        """
        Queues an unimplemented action or macro request for human developers to build post-run.
        Returns confirmation message.
        """
        res = queue_macro_request(
            macro_name=macro_name,
            rationale=rationale,
            proposed_interface=proposed_interface,
            priority=priority,
            run_id=run_id,
            db_path=self.db_path,
        )
        return res["message"]

    def get_dungeon_topology(self, depth: int = 1) -> str:
        """
        Renders a visual 21x79 ASCII map of visited tiles for a given dungeon level
        showing corridors (#), rooms (.), doors (+), stairs (> <), fountains ({), and traps (^).
        """
        if not os.path.exists(self.db_path):
            return "No database found."
        try:
            depth_int = int(depth)
        except Exception:
            depth_int = 1

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute(f"""
                SELECT DISTINCT y, x, tile_type
                FROM ticks
                WHERE depth = {depth_int} AND y >= 0 AND y < 21 AND x >= 0 AND x < 79
            """).fetchall()
            con.close()
        except Exception as e:
            con.close()
            return f"Error retrieving dungeon topology: {e}"

        if not rows:
            return f"No visited tiles found for depth {depth_int}."

        char_map = {"room": ".", "corridor": "#", "doorway": "+", "fountain": "{", "altar": "_", "trap": "^", "stairs_down": ">", "stairs_up": "<"}
        grid = [[" " for _ in range(79)] for _ in range(21)]
        for y, x, ttype in rows:
            grid[y][x] = char_map.get(ttype, ".")

        lines = [f"### Dungeon Topology for Depth {depth_int} (Visited Footprint):", "```"]
        for row in grid:
            lines.append("".join(row).rstrip())
        lines.append("```")
        return "\n".join(lines)

    def get_hazard_map(self, depth: int = 1) -> str:
        """
        Returns coordinates and details of discovered traps, sleeping hostiles, and floating eyes.
        """
        if not os.path.exists(self.db_path):
            return "No database found."
        try:
            depth_int = int(depth)
        except Exception:
            depth_int = 1

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            cur = con.execute(f"""
                SELECT y, x, tile_type, closest_hostile_name, COUNT(*) as occurrences
                FROM ticks
                WHERE depth = {depth_int} AND (tile_type = 'trap' OR closest_hostile_name IN ('floating eye', 'soldier ant', 'mimic'))
                GROUP BY y, x, tile_type, closest_hostile_name
                ORDER BY occurrences DESC
                LIMIT 20
            """)
            res = _format_table(cur)
            con.close()
            return f"### Hazard Map (Depth {depth_int}):\n" + res
        except Exception as e:
            con.close()
            return f"Error retrieving hazard map: {e}"

    def get_floor_stash_report(self) -> str:
        """
        Returns list of fountains, altars, and items discovered across all explored dungeon levels.
        """
        if not os.path.exists(self.db_path):
            return "No database found."

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            cur = con.execute("""
                SELECT depth, tile_type, COUNT(DISTINCT (y || ',' || x)) as distinct_features
                FROM ticks
                WHERE tile_type IN ('fountain', 'altar', 'doorway')
                GROUP BY depth, tile_type
                ORDER BY depth ASC, tile_type ASC
            """)
            res = _format_table(cur)
            con.close()
            return "### Dungeon Stash & Features Report:\n" + res
        except Exception as e:
            con.close()
            return f"Error querying floor stash report: {e}"

    def get_death_autopsy_trace(self, episode_id: str = "") -> str:
        """
        Returns the granular tick-by-tick flight trace of the final 15 ticks of an episode.
        """
        if not os.path.exists(self.db_path):
            return "No database found."

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            where = f"WHERE episode_id = '{episode_id}'" if episode_id else "WHERE episode_id = (SELECT episode_id FROM episodes ORDER BY rowid DESC LIMIT 1)"
            cur = con.execute(f"""
                SELECT turn, depth, hp, max_hp, hunger, tile_type, action, closest_hostile_name, closest_hostile_dist, message
                FROM ticks
                {where}
                ORDER BY turn DESC
                LIMIT 15
            """)
            res = _format_table(cur)
            con.close()
            return f"### Final 15 Ticks Autopsy Trace:\n" + res
        except Exception as e:
            con.close()
            return f"Error querying autopsy trace: {e}"


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
    {
        "type": "function",
        "function": {
            "name": "query_wiki",
            "description": "Search the offline NetHack 3.6.6 knowledge base (monsters, items, intrinsics, rituals).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term, e.g. 'floating eye', 'poison resistance corpse', 'Excalibur'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of articles to return (default 2).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_macro",
            "description": "Queue an unimplemented macro or primitive for human developers to build post-run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "macro_name": {
                        "type": "string",
                        "description": "Identifier name for the desired macro or primitive.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this capability is needed to progress or survive.",
                    },
                    "proposed_interface": {
                        "type": "string",
                        "description": "Optional Python signature or pseudo-code.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                        "description": "Priority level.",
                    },
                },
                "required": ["macro_name", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dungeon_topology",
            "description": "Renders visual 21x79 ASCII map of visited tiles for a given dungeon level showing rooms (.), corridors (#), doors (+), and stairs (> <).",
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
    {
        "type": "function",
        "function": {
            "name": "get_hazard_map",
            "description": "Returns coordinates and occurrences of traps, floating eyes, and dangerous monsters on that level.",
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
    {
        "type": "function",
        "function": {
            "name": "get_floor_stash_report",
            "description": "Returns summary of altars, fountains, and features discovered across all explored dungeon levels.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_death_autopsy_trace",
            "description": "Returns granular tick-by-tick flight trace of the final 15 ticks leading up to fatal termination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "Optional episode ID. If omitted, returns trace of the most recent death.",
                    }
                },
            },
        },
    },
]
