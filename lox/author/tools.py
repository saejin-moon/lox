"""
Minimal, High-Signal Tool Suite for the Author Agent:
1. `query_db(sql)`: Read-only SQL over all DuckDB tables (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`)
2. `query_wiki(query, top_k)`: Offline NetHack 3.6.6 Wiki BM25 search
3. `query_policy_docs(topic)`: Policy language specification, `GameState`/`Action` docs, and current `macros.py` signatures & source
4. `dry_run_policy(code, is_macro, seed, max_turns)`: Compiles and runs candidate code in a 20ms live NetHack sandbox to catch any syntax/runtime/loop bugs before submission.
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter
from typing import Any

from lox.author.prompts import POLICY_LANG_DOCS
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_and_macros
from lox.lang.macros import DEFAULT_MACROS_PATH, MacroLibrary
from lox.lang.validator import splice_functions_into_policy


def _format_cursor_yaml(cur: Any, max_rows: int = 40) -> tuple[str, int]:
    if not cur.description:
        return "status: ok", 0
    headers = [d[0] for d in cur.description]
    rows = cur.fetchall()[:max_rows]
    if not rows:
        return "results: []", 0
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
                if len(s) > 700:
                    s = s[:340] + " ... " + s[-340:]
                s_esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                parts.append(f'{h}: "{s_esc}"')
        lines.append("  - {" + ", ".join(parts) + "}")
    return "```yaml\n" + "\n".join(lines) + "\n```", len(rows)


class AuthorToolRegistry:
    """Executes tool calls from the Author Agent."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        wiki_db_path: str = DEFAULT_WIKI_DB,
        macros_path: str = DEFAULT_MACROS_PATH,
    ):
        self.db_path = db_path
        self.wiki = WikiEngine(db_path=wiki_db_path)
        self.macro_lib = MacroLibrary(path=macros_path)
        self.current_policy_code: str = ""
        self.last_passed_code: str = ""

    def query_db(self, sql: str) -> tuple[str, str]:
        """Executes a read-only SQL query against DuckDB."""
        if not os.path.exists(self.db_path):
            return "Database not found.", "0 rows"

        clean = sql.strip().rstrip(";")
        low = clean.lower()
        # Strip single-quoted string literals before checking mutative SQL keywords
        low_no_literals = re.sub(r"'[^']*'", "''", low)
        for word in (
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "truncate",
            "create",
            "attach",
            "copy",
        ):
            if re.search(rf"\b{word}\b", low_no_literals):
                return (
                    f"Error: Mutative keyword '{word.upper()}' is forbidden. Use SELECT only.",
                    "error",
                )
        if re.search(r"\breplace\b(?!\s*\()", low_no_literals):
            return (
                "Error: Mutative keyword 'REPLACE' is forbidden. Use SELECT only.",
                "error",
            )

        if "limit" not in low_no_literals:
            clean += " LIMIT 30"

        con = connect_db(self.db_path, read_only=True)
        try:
            cur = con.execute(clean)
            text, count = _format_cursor_yaml(cur)
            return text, f"{count} rows"
        except Exception as exc:
            return f"SQL Error: {exc}", "sql_error"
        finally:
            con.close()

    def query_wiki(self, query: str, top_k: int = 3) -> tuple[str, str]:
        """Searches the offline NetHack 3.6.6 Wiki."""
        k = max(1, min(6, int(top_k or 3)))
        hits = self.wiki.search_wiki(query, top_k=k)
        return self.wiki.query(query, top_k=k), f"{len(hits)} articles"

    def query_policy_docs(self, topic: str = "all") -> tuple[str, str]:
        """Returns policy language specs and registered macro definitions/docstrings."""
        macros_code = self.macro_lib.load_code()
        macros_section = (
            f"\n### Current `macros.py` Source:\n```python\n{macros_code}\n```"
            if macros_code
            else "\n### Current `macros.py`: (Empty - no custom macros defined yet)"
        )
        t = (topic or "all").lower()
        if "macro" in t:
            return self.macro_lib.format_docs() + macros_section, "macros docs"
        return (
            POLICY_LANG_DOCS + "\n" + self.macro_lib.format_docs() + macros_section,
            "full docs",
        )

    def dry_run_policy(
        self,
        code: str,
        is_macro: bool = False,
        seed: int = 42,
        max_turns: int = 60,
    ) -> tuple[str, str]:
        """
        Compiles and executes candidate `code` in a live NetHack sandbox for `max_turns` steps
        to verify syntax, sandbox rules, attribute access, and action yielding.
        """
        macros_code = self.macro_lib.load_code()
        policy_code = self.current_policy_code

        try:
            if is_macro:
                # Test compiling the candidate macro code merged with existing macros
                from lox.lang.validator import validate_python_code

                validate_python_code(code, filename="macros_candidate.py")
                merged_macros = (
                    (macros_code + "\n\n" + code).strip() if macros_code else code
                )
                if policy_code.strip():
                    compiled = compile_policy_and_macros(policy_code, merged_macros)
                else:
                    self.last_passed_code = code
                    return (
                        "Macro syntax and AST validation PASSED. (No active policy yet to run steps).",
                        "VALID (macro)",
                    )
            else:
                if policy_code.strip():
                    spliced_policy, changed_fns = splice_functions_into_policy(
                        policy_code, code
                    )
                    # Static reachability check: ensure any non-root method in changed_fns is referenced in spliced_policy
                    spliced_tree = ast.parse(spliced_policy)
                    called_attrs: set[str] = set()
                    for node in ast.walk(spliced_tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                called_attrs.add(node.func.attr)
                            elif isinstance(node.func, ast.Name):
                                called_attrs.add(node.func.id)
                    unreferenced = [
                        fn
                        for fn in changed_fns
                        if fn not in ("run", "step", "act", "__init__")
                        and fn not in called_attrs
                    ]
                    if unreferenced:
                        return (
                            f"Unreachable Function Error: Method(s) {unreferenced} were spliced into `class Policy`, "
                            f"but are NEVER called by `run` or any other method in `policy.py`! "
                            f"Please include the updated caller method (e.g. `def run(self, state):` or dispatcher) "
                            f"in your code block so {unreferenced} is actually wired in and called.",
                            "UNREACHABLE_ERROR",
                        )
                else:
                    spliced_policy = code
                compiled = compile_policy_and_macros(spliced_policy, macros_code)
        except Exception as exc:
            return (
                f"Compilation / Validation Error:\n{type(exc).__name__}: {exc}",
                "COMPILE_ERROR",
            )

        harness = NetHackHarness(max_steps=max(10, min(200, int(max_turns))))
        try:
            runner = compiled.create_runner()
            state = harness.reset(seed=int(seed))
            actions_taken: list[str] = []
            raw_act_names: list[str] = []
            unique_positions: set[tuple[int, int]] = {(state.y, state.x)}
            for _ in range(max(10, min(200, int(max_turns)))):
                raw_act, active_fn = runner.next_action(state)
                state, done, info = harness.step(raw_act)
                unique_positions.add((state.y, state.x))
                act_name = info["action_name"]
                raw_act_names.append(act_name)
                actions_taken.append(f"{active_fn}:{act_name}")
                if done:
                    break
            fn_summary = ", ".join(
                f"{k}:{v}" for k, v in runner.tracer.fn_calls.items()
            )
            act_counts = ", ".join(
                f"{k}:{v}" for k, v in Counter(raw_act_names).most_common(6)
            )
            params_summary = (
                ", ".join(
                    f"{k}={v.value}" for k, v in runner.tracer.registered_params.items()
                )
                or "none"
            )
            sample_acts = " -> ".join(actions_taken[:10])
            warn_line = ""
            if len(unique_positions) == 1 and harness.step_count >= 15:
                warn_line = (
                    "\n- WARNING: Hero stayed on 1 tile ((y,x) did not change) across all steps! "
                    "Verify your passable mask and movement logic."
                )
            self.last_passed_code = code
            return (
                f"Dry-Run PASSED ({harness.step_count} steps on seed {seed}).\n"
                f"- Final State: Dlvl={state.depth}, Pos=({state.y},{state.x}), UniqueTiles={len(unique_positions)}, HP={state.hp}/{state.max_hp}, Turn={state.turn}\n"
                f"- Functions Called: {{{fn_summary}}}\n"
                f"- Action Breakdown: {{{act_counts}}}\n"
                f"- Registered Params: {{{params_summary}}}\n"
                f"- First 10 Actions: {sample_acts}{warn_line}",
                f"PASSED ({harness.step_count} steps)",
            )
        except Exception as exc:
            import traceback

            tb = traceback.format_exc(limit=4)
            return (
                f"Runtime Error during Dry-Run on seed {seed}:\n{type(exc).__name__}: {exc}\n{tb}",
                "RUNTIME_ERROR",
            )
        finally:
            harness.close()


OPENAI_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Execute a read-only SELECT SQL query on the DuckDB database "
                "(tables: runs, ticks, batches, function_stats, experiments, expert_runs, expert_ticks)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Read-only SQL SELECT query.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_wiki",
            "description": "Search the offline NetHack 3.6.6 Wiki for monster stats, items, corpses, intrinsics, or branch mechanics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search topic or entity name (e.g., 'floating eye', 'Excalibur', 'hungry', 'altar').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of wiki articles to return (1-5).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_policy_docs",
            "description": "Retrieve Policy Language documentation, GameState fields, Action names, and all macros currently in macros.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "'all', 'macros', 'state', or 'actions'.",
                        "default": "all",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dry_run_policy",
            "description": "Compile and test candidate Python policy or macro code for 60 steps in a live NetHack sandbox to catch any syntax or runtime errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code (class Policy, method definitions, or macro functions) to validate and dry-run.",
                    },
                    "is_macro": {
                        "type": "boolean",
                        "description": "Set to true if testing macro functions for macros.py; false if testing Policy code.",
                        "default": False,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "NetHack seed to test on.",
                        "default": 42,
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Number of steps to simulate (10-200).",
                        "default": 60,
                    },
                },
                "required": ["code"],
            },
        },
    },
]
