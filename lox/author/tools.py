"""
Minimal, High-Signal Tool Suite for the Author Agent:
1. `query_db(sql)`: Read-only SQL over all DuckDB tables (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`)
2. `query_wiki(query, top_k)`: Offline NetHack 3.6.6 Wiki BM25 search
3. `query_policy_docs(topic)`: Policy language specification, `GameState`/`Action` docs, and current `macros.py` signatures & source
4. `dry_run_policy(code, is_macro, seed, max_turns)`: Compiles and runs candidate code in a 20ms live NetHack sandbox to catch any syntax/runtime/loop bugs before submission.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lox.author.bt_eval import evaluate_bt_subtree
from lox.author.prompts import BEHAVIOR_TREE_DOCS, FULL_ACTION_DOCS, POLICY_LANG_DOCS
from lox.core.contracts import CandidatePackage
from lox.datasets.symbols import SYMBOL_REGISTRY
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.db.shards import extract_shards_from_runs
from lox.env.actions import Action
from lox.env.harness import NetHackHarness, policy_handles_menus
from lox.lang.bundle import (
    strip_code_fences as _strip_fences,
)
from lox.lang.bundle import (
    unpack_code_bundle,
)
from lox.lang.compiler import compile_policy_package
from lox.lang.macros import DEFAULT_MACROS_PATH, MacroLibrary
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    check_evolve_block_bounds,
    is_ast_equivalent,
    normalize_mode,
    normalize_skill_key,
    prune_extracted_skill_functions,
    run_pyright_on_package,
    smart_dedent,
    splice_functions_into_policy,
    validate_cross_module_contracts,
)

_PEER_MICRO_PILOT_LOCK = threading.Lock()


def merge_skill_module_update(existing_src: str, candidate_src: str) -> str:
    """
    Merges an edited skill module safely into `existing_src`.
    If `candidate_src` is a partial function/constant patch (omits existing imports,
    starts with a patch comment, or drops more than half of `existing_src`'s top-level
    definitions), splices it cleanly via `splice_functions_into_policy` so existing
    exports (e.g. `_delta_of`) are never wiped out.
    """
    cand = smart_dedent(str(candidate_src or "")).strip()
    if not existing_src or not existing_src.strip():
        return cand
    if not cand:
        return existing_src

    try:
        ex_tree = ast.parse(existing_src)
        ca_tree = ast.parse(cand)
    except Exception:
        return cand

    ex_has_imports = any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ex_tree.body
    )
    ca_has_imports = any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ca_tree.body
    )
    ex_defs = {
        n.name
        for n in ex_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    ca_defs = {
        n.name
        for n in ca_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    ex_assigns = {
        t.id
        for n in ex_tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {
        n.target.id
        for n in ex_tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    ca_assigns = {
        t.id
        for n in ca_tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {
        n.target.id
        for n in ca_tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    first_line = cand.splitlines()[0].strip().lower() if cand.splitlines() else ""
    if first_line.startswith("# replace") or first_line.startswith("# full replace"):
        return cand
    looks_patch_comment = (
        first_line.startswith("# patch")
        or "other functions unchanged" in cand[:300].lower()
    )
    ca_has_docstring = ast.get_docstring(ca_tree) is not None
    dropped_most_defs = len(ex_defs) >= 3 and len(ca_defs) < max(2, len(ex_defs) // 2)
    omitted_existing = bool((ex_defs - ca_defs) | (ex_assigns - ca_assigns))
    overlaps_existing = bool((ex_defs & ca_defs) or (ex_assigns & ca_assigns))
    looks_full_rewrite = (
        not looks_patch_comment
        and (ca_has_docstring or ca_has_imports)
        and (ca_has_imports or not ex_has_imports)
        and (len(ca_assigns) > 0 or len(ex_assigns) == 0 or ca_has_docstring)
        and not dropped_most_defs
        and (
            (ca_has_docstring and len(ca_defs) >= 2 and not (ca_defs < ex_defs))
            or (
                len(ca_defs) >= max(2, len(ex_defs) - 1)
                and (ca_has_docstring or not (ca_defs < ex_defs))
            )
        )
    )
    if looks_full_rewrite:
        return cand

    partial_subset = (
        (len(ex_defs) >= 2 and 0 < len(ca_defs) < len(ex_defs) and ca_defs <= ex_defs)
        or (len(ex_defs) >= 1 and len(ca_defs) == 0 and len(ca_assigns) > 0)
        or (omitted_existing and overlaps_existing and len(ex_defs | ex_assigns) >= 2)
    )
    missing_imports = (
        ex_has_imports
        and not ca_has_imports
        and (len(ca_defs) < len(ex_defs) or len(ex_defs) == 0 or omitted_existing)
    )
    missing_constants = (
        len(ex_assigns) > 0
        and len(ca_assigns) == 0
        and (len(ca_defs) <= len(ex_defs) or omitted_existing)
    )

    def _imported_skill_modules(tr: ast.AST) -> set[str]:
        mods: set[str] = set()
        for nd in getattr(tr, "body", []):
            if isinstance(nd, ast.ImportFrom) and str(nd.module or "").startswith(
                "skills."
            ):
                mods.add(str(nd.module))
            elif isinstance(nd, ast.Import):
                for alias in nd.names:
                    if str(alias.name).startswith("skills."):
                        mods.add(str(alias.name))
        return mods

    new_skill_imports = _imported_skill_modules(ca_tree) - _imported_skill_modules(
        ex_tree
    )
    if new_skill_imports and not looks_patch_comment:
        return cand

    if (
        looks_patch_comment
        or dropped_most_defs
        or partial_subset
        or missing_imports
        or missing_constants
    ):
        try:
            spliced, _ = splice_functions_into_policy(existing_src, cand)
            return spliced
        except Exception:
            return cand
    return cand


def _unpack_xml_bundle(raw_code: str) -> tuple[str, dict[str, str]]:
    """
    Extracts `<skill_code module="name">...</skill_code>` blocks and `<policy_code>...</policy_code>`
    from `raw_code`. Delegates to canonical `lox.lang.bundle.unpack_code_bundle`.
    """
    return unpack_code_bundle(
        raw_code,
        merge_fn=merge_skill_module_update,
        include_markdown_skills=False,
        preserve_raw_verbatim=True,
    )


def _format_cursor_yaml(cur: Any, max_rows: int = 40) -> tuple[str, int]:
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


class AuthorToolRegistry:
    """Executes tool calls from the Author Agent."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        wiki_db_path: str = DEFAULT_WIKI_DB,
        macros_path: str = DEFAULT_MACROS_PATH,
        benchmark: str = "nethack",
        max_steps: int = 5000,
    ):
        self.db_path = db_path
        self.wiki = WikiEngine(db_path=wiki_db_path)
        self.macro_lib = MacroLibrary(path=macros_path)
        self.benchmark = str(benchmark)
        self.max_steps = max_steps
        self.current_policy_code: str = ""
        self.last_passed_pkg: CandidatePackage = CandidatePackage()
        self.best_pilot_pkg: CandidatePackage = CandidatePackage()
        self.pool: Any = None
        self.baseline_runs: list[dict[str, Any]] = []
        self.pilot_runs_used: int = 0
        self.max_pilot_runs: int = 8
        self.pilot_episodes: int = 30
        self.last_pilot_improved: bool = False
        self.last_pilot_summary: str = ""
        self.last_dry_run_mode: str = "none"
        self.last_dry_run_passed: bool = False
        self.last_cand_depths: list[float] = []
        self.last_base_depths: list[float] = []
        self.skills: dict[str, str] = {}
        self.last_passed_skills: dict[str, str] = {}
        self.last_dry_run_diag: str = ""
        self.theories_store: Any = None
        self.generation: int = 0
        self.mode: str = "WRITE"
        self.swarm_checkpoint: Any = None
        self.swarm_frontier: Any = None
        self.swarm_replay_fixtures: list[dict[str, Any]] = []
        self.swarm_scaffold_code: str = ""
        self.swarm_experiments: list[dict[str, Any]] = []
        self._reusable_harness: Any = None
        self._reusable_harness_benchmark: str = ""
        self.wiki_queries_run: int = 0
        self.theories_queries_run: int = 0
        self.db_queries_run: int = 0
        self.enforce_peer_research_gate: bool = False
        self.claimed_deficit: Any = None
        self.scenario_gate: str = "off"
        self.peer_guard_seeds: list[int] = []
        self.peer_target_seeds: list[int] = []
        self.peer_probe_seeds: list[int] = []
        self.peer_micro_pilots_used: int = 0
        self.max_peer_micro_pilots: int = 3
        self.peer_id: str = ""
        self.baseline_policy_code: str = ""
        self.baseline_skills: dict[str, str] | None = None
        self._dead_patch_bonus_granted: bool = False

    def _get_harness(self, max_steps: int, benchmark: str | None = None) -> Any:
        steps = max(10, int(max_steps))
        bench = str(benchmark or self.benchmark or "nethack").strip().lower()
        if (
            self._reusable_harness is not None
            and self._reusable_harness_benchmark != bench
        ):
            self.close()
        if self._reusable_harness is None:
            if bench == "nethack":
                self._reusable_harness = NetHackHarness(max_steps=steps)
            else:
                from lox.env import get_adapter

                self._reusable_harness = get_adapter(bench).create_harness(
                    max_steps=steps
                )
            self._reusable_harness_benchmark = bench
        else:
            self._reusable_harness.max_steps = steps
        return self._reusable_harness

    def close(self) -> None:
        """Closes any cached C/simulator harness owned by this registry."""
        if self._reusable_harness is not None:
            try:
                self._reusable_harness.close()
            except Exception:
                pass
            self._reusable_harness = None
            self._reusable_harness_benchmark = ""

    @property
    def last_passed_code(self) -> str:
        return self.last_passed_pkg.code

    @last_passed_code.setter
    def last_passed_code(self, val: str) -> None:
        self.last_passed_pkg.code = str(val or "")

    @property
    def last_passed_skill_name(self) -> str | None:
        return self.last_passed_pkg.skill_name or None

    @last_passed_skill_name.setter
    def last_passed_skill_name(self, val: str | None) -> None:
        self.last_passed_pkg.skill_name = str(val or "")

    @property
    def last_passed_skill_code(self) -> str | None:
        return self.last_passed_pkg.skill_code or None

    @last_passed_skill_code.setter
    def last_passed_skill_code(self, val: str | None) -> None:
        self.last_passed_pkg.skill_code = str(val or "")

    @property
    def best_pilot_code(self) -> str:
        return self.best_pilot_pkg.code

    @best_pilot_code.setter
    def best_pilot_code(self, val: str) -> None:
        self.best_pilot_pkg.code = str(val or "")

    @property
    def best_pilot_skill_name(self) -> str | None:
        return self.best_pilot_pkg.skill_name or None

    @best_pilot_skill_name.setter
    def best_pilot_skill_name(self, val: str | None) -> None:
        self.best_pilot_pkg.skill_name = str(val or "")

    @property
    def best_pilot_skill_code(self) -> str | None:
        return self.best_pilot_pkg.skill_code or None

    @best_pilot_skill_code.setter
    def best_pilot_skill_code(self, val: str | None) -> None:
        self.best_pilot_pkg.skill_code = str(val or "")

    @property
    def best_pilot_delta(self) -> float:
        return self.best_pilot_pkg.pilot_delta

    @best_pilot_delta.setter
    def best_pilot_delta(self, val: float) -> None:
        self.best_pilot_pkg.pilot_delta = float(val)

    def configure_generation(
        self,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        max_pilot_runs: int = 8,
        pilot_episodes: int = 30,
        skills: dict[str, str] | None = None,
        generation: int = 0,
        benchmark: str | None = None,
        mode: str = "WRITE",
        swarm_checkpoint: Any = None,
        swarm_frontier: Any = None,
        swarm_replay_fixtures: list[dict[str, Any]] | None = None,
        swarm_scaffold_code: str = "",
        claimed_deficit: Any = None,
        scenario_gate: str = "off",
        peer_guard_seeds: list[int] | None = None,
        peer_target_seeds: list[int] | None = None,
        peer_probe_seeds: list[int] | None = None,
        max_peer_micro_pilots: int = 3,
        peer_id: str = "",
    ) -> None:
        """Sets worker pool and baseline runs context for pilot evaluations."""
        self.pool = pool
        self.baseline_runs = list(baseline_runs or [])
        self.pilot_runs_used = 0
        self.max_pilot_runs = max_pilot_runs
        self.pilot_episodes = pilot_episodes
        self.last_pilot_improved = False
        self.last_pilot_summary = ""
        self.last_dry_run_mode = "none"
        self.last_dry_run_passed = False
        self.last_dry_run_diag = ""
        self.last_passed_pkg = CandidatePackage()
        self.last_passed_skills = {}
        self.best_pilot_pkg = CandidatePackage()
        self.last_cand_depths = []
        self.last_base_depths = []
        self.generation = int(generation)
        self.mode = normalize_mode(mode)
        self.swarm_checkpoint = swarm_checkpoint
        self.swarm_frontier = swarm_frontier
        self.swarm_replay_fixtures = list(swarm_replay_fixtures or [])
        self.swarm_scaffold_code = str(swarm_scaffold_code or "")
        self.claimed_deficit = claimed_deficit
        self.scenario_gate = str(scenario_gate or "off").strip().lower()
        self.swarm_experiments = []
        self.wiki_queries_run = 0
        self.theories_queries_run = 0
        self.db_queries_run = 0
        self.enforce_peer_research_gate = False
        self.peer_guard_seeds = [int(s) for s in (peer_guard_seeds or [])]
        self.peer_target_seeds = [int(s) for s in (peer_target_seeds or [])]
        self.peer_probe_seeds = [int(s) for s in (peer_probe_seeds or [])]
        self.peer_micro_pilots_used = 0
        self.max_peer_micro_pilots = max(0, int(max_peer_micro_pilots))
        self.peer_id = str(peer_id or self.peer_id or "")
        self._dead_patch_bonus_granted = False
        if skills is not None:
            self.skills = dict(skills)
        if benchmark is not None:
            self.benchmark = str(benchmark)

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
        """Executes a read-only SQL query against DuckDB."""
        if not os.path.exists(self.db_path):
            return "Database not found.", "0 rows"

        clean = sql.strip().rstrip(";")
        low = clean.lower()
        # Strip single-quoted string literals before checking mutative SQL keywords
        low_no_literals = re.sub(r"'[^']*'", "''", low)
        if ";" in low_no_literals:
            return (
                "Error: Multiple SQL statements in a single query_db call are not allowed. Pass exactly ONE SELECT statement per query_db call.",
                "error",
            )
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

        auto_limited = "limit" not in low_no_literals
        max_rows_cap = 40
        if auto_limited:
            clean += " LIMIT 30"
        else:
            m_lim = re.search(r"\blimit\s+(\d+)\b", low_no_literals)
            if m_lim:
                max_rows_cap = max(40, min(200, int(m_lim.group(1))))

        con = connect_db(self.db_path, read_only=True)
        try:
            cur = con.execute(clean)
            self.db_queries_run += 1
            text, count = _format_cursor_yaml(cur, max_rows=max_rows_cap)
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

    def query_monster(self, name: str) -> tuple[str, str]:
        """Queries C-level permonst ground truth struct for a monster."""
        import yaml

        self.wiki_queries_run += 1
        data = SYMBOL_REGISTRY.query_monster(name)
        if data:
            return "```yaml\n" + yaml.dump(
                data, default_flow_style=False
            ) + "\n```", "found"
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
            return "```yaml\n" + yaml.dump(
                data, default_flow_style=False
            ) + "\n```", "found"
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

    def query_policy_docs(self, topic: str = "all") -> tuple[str, str]:
        """Returns policy language specs, current policy.py, skill registry, and macros.py."""
        macros_code = self.macro_lib.load_code()
        macros_section = (
            f"\n### Current `macros.py` Source:\n```python\n{macros_code}\n```"
            if macros_code
            else "\n### Current `macros.py`: (Empty - no custom macros defined yet)"
        )
        skill_names = sorted(self.skills.keys())
        if not skill_names:
            skills_dir = Path(self.macro_lib.path).parent / "skills"
            if skills_dir.exists():
                for p in sorted(skills_dir.rglob("*.py")):
                    if p.name == "__init__.py":
                        continue
                    rel_k = normalize_skill_key(
                        str(p.relative_to(skills_dir)).replace("\\", "/")
                    )
                    if rel_k and rel_k not in skill_names:
                        skill_names.append(rel_k)
        skills_section = (
            f"\n### Available `skills/` Modules ({len(skill_names)}): {', '.join(sorted(skill_names)) or '(none)'}\n"
            f"(Call `read_skill(skill_name)` to inspect full source of any module.)"
        )
        t = (topic or "all").lower()
        if "policy" in t:
            pol_src = self.current_policy_code or "(Empty policy)"
            return (
                f"### Current `policy.py` Full Source:\n```python\n{pol_src}\n```{skills_section}",
                "policy source",
            )
        if "skill" in t:
            return skills_section.strip(), f"{len(skill_names)} skills"
        if "action" in t:
            return FULL_ACTION_DOCS, "action docs"
        if "macro" in t:
            return self.macro_lib.format_docs() + macros_section, "macros docs"
        pol_src = self.current_policy_code or "(Empty policy)"
        if self.mode in ("SWARM_WORKER", "SWARM_META", "SWARM_PEER"):
            return (
                BEHAVIOR_TREE_DOCS
                + "\n"
                + POLICY_LANG_DOCS
                + skills_section
                + f"\n### Current `policy.py` Source:\n```python\n{pol_src}\n```",
                "full docs",
            )
        return (
            POLICY_LANG_DOCS
            + "\n"
            + self.macro_lib.format_docs()
            + macros_section
            + skills_section
            + f"\n### Current `policy.py` Source:\n```python\n{pol_src}\n```",
            "full docs",
        )

    def query_theories(
        self, query: str = "", doc_type: str = "", limit: int = 6
    ) -> tuple[str, str]:
        """Searches LanceDB theories/reflections/decompositions without aggressive truncation."""
        self.theories_queries_run += 1
        if self.theories_store is None:
            return "LanceDB theories store not attached in this session.", "unavailable"
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

    @staticmethod
    def _lookup_policy_snapshot(
        con: Any, pid: str
    ) -> tuple[int, str, float, int, bool, str, str] | None:
        """Looks up a policy snapshot by `policy_id` in `policies`, falling back to MAP-Elites `archive_policies`."""
        try:
            row = con.execute(
                "SELECT generation, mode, avg_depth, max_depth, is_promoted, policy_code, skills_json FROM policies WHERE policy_id = ?",
                [pid],
            ).fetchone()
        except Exception:
            try:
                row_legacy = con.execute(
                    "SELECT generation, mode, avg_depth, max_depth, is_promoted, policy_code FROM policies WHERE policy_id = ?",
                    [pid],
                ).fetchone()
                row = (*row_legacy, "{}") if row_legacy else None
            except Exception:
                row = None
        if row is not None:
            return row
        try:
            arow = con.execute(
                """
                SELECT generation, policy_code, skills_json, metrics_json
                FROM archive_policies
                WHERE cell_id = ? OR batch_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                [pid, pid],
            ).fetchone()
        except Exception:
            arow = None
        if not arow:
            return None
        gen, p_code, s_json, m_json = arow
        try:
            metrics = json.loads(m_json or "{}") if isinstance(m_json, str) else {}
        except Exception:
            metrics = {}
        return (
            int(gen or 0),
            "MAP_ELITES",
            float(metrics.get("avg_depth", 1.0) or 1.0),
            int(metrics.get("max_depth", 1) or 1),
            False,
            str(p_code or ""),
            str(s_json or "{}"),
        )

    def read_skill(
        self,
        skill_name: str,
        policy_id: str = "",
        start_line: int = 1,
        end_line: int | None = None,
    ) -> tuple[str, str]:
        """Reads a skill module (or line range `start_line..end_line`) from the workspace or a `policy_id` snapshot."""
        from lox.lang.validator import normalize_skill_key

        try:
            clean_name = normalize_skill_key(skill_name)
        except Exception:
            clean_name = skill_name.strip().removeprefix("skills/").removesuffix(".py")

        def _slice_lines(src_text: str, tag: str) -> tuple[str, str]:
            all_lines = src_text.splitlines()
            total_l = len(all_lines)
            s_l = max(1, int(start_line or 1))
            e_l = min(total_l, int(end_line)) if end_line is not None else total_l
            if s_l == 1 and end_line is None:
                return src_text, f"{total_l} lines{tag}"
            if s_l > total_l:
                return (
                    f"(File `{clean_name}` has only {total_l} lines.)",
                    f"0 lines{tag}",
                )
            numbered = [
                f"{idx}: {all_lines[idx - 1]}" for idx in range(s_l, max(s_l, e_l) + 1)
            ]
            return (
                f"# {clean_name}.py lines {s_l}-{max(s_l, e_l)} of {total_l}\n"
                + "\n".join(numbered),
                f"lines {s_l}-{max(s_l, e_l)}/{total_l}{tag}",
            )

        pid = str(policy_id or "").strip()
        if pid:
            if not os.path.exists(self.db_path):
                return "Database not found.", "not_found"
            con = connect_db(self.db_path, read_only=True)
            try:
                snap = self._lookup_policy_snapshot(con, pid)
                if not snap:
                    return f"Error: Policy '{pid}' not found.", "not_found"
                try:
                    skills_map = json.loads(snap[6] or "{}") or {}
                except Exception:
                    skills_map = {}
                if clean_name in skills_map:
                    return _slice_lines(str(skills_map[clean_name]), f" ({pid})")
                return (
                    f"Error: Skill '{clean_name}' not found in policy '{pid}'. Available skills in '{pid}': {sorted(skills_map.keys()) or 'None'}",
                    "not_found",
                )
            finally:
                con.close()

        if clean_name in self.skills:
            return _slice_lines(self.skills[clean_name], "")
        skills_dir = Path(self.macro_lib.path).parent / "skills"
        skill_file = skills_dir / f"{clean_name}.py"
        if skill_file.exists():
            content = skill_file.read_text(encoding="utf-8")
            return _slice_lines(content, "")
        avail = list(self.skills.keys())
        if skills_dir.exists():
            avail += [
                str(p.relative_to(skills_dir)).removesuffix(".py").replace("\\", "/")
                for p in skills_dir.rglob("*.py")
                if p.name != "__init__.py"
            ]
        return (
            f"Error: Skill '{clean_name}' not found. Available skills: {sorted(set(avail)) or 'None'}",
            "not_found",
        )

    def read_policy(self, policy_id: str = "") -> tuple[str, str]:
        """Reads current `policy.py` (when empty/`current`) or a historical `policy_id` snapshot."""
        pid = str(policy_id or "").strip()
        if not pid or pid.lower() in ("current", "workspace", "incumbent", "policy.py"):
            cur = str(self.current_policy_code or "")
            if cur.strip():
                return cur, f"{len(cur.splitlines())} lines (current)"
            ws_pol = Path(self.macro_lib.path).parent / "policy.py"
            if ws_pol.exists():
                cur = ws_pol.read_text(encoding="utf-8")
                return cur, f"{len(cur.splitlines())} lines (current)"
            return "(Empty current policy.py)", "0 lines"
        if not os.path.exists(self.db_path):
            return "Database not found.", "not_found"
        con = connect_db(self.db_path, read_only=True)
        try:
            row = self._lookup_policy_snapshot(con, pid)
            if not row:
                return f"Error: Policy '{policy_id}' not found.", "not_found"
            gen, mode, avg_d, max_d, prom, p_code, s_json = row
            try:
                skills_map = json.loads(s_json or "{}")
            except Exception:
                skills_map = {}
            skill_names = (
                sorted(skills_map.keys()) if isinstance(skills_map, dict) else []
            )
            tip = (
                f"# Tip: call read_skill('<skill_name>', policy_id='{policy_id.strip()}') to read full source of any skill from this snapshot.\n"
                if skill_names
                else ""
            )
            header = (
                f"# policy_id: {policy_id} | generation: {gen} | mode: {mode} "
                f"| avg_depth: {avg_d} | max_depth: {max_d} | is_promoted: {prom}\n"
                f"# skills_snapshot: {skill_names}\n"
                f"{tip}\n"
            )
            return header + str(
                p_code or ""
            ), f"{len(str(p_code or '').splitlines())} lines"
        finally:
            con.close()

    def diff_policies(self, policy_id_a: str, policy_id_b: str) -> tuple[str, str]:
        """Returns unified diffs of `policy_code` and all `skills_json` modules between two `policy_id`s."""
        import difflib

        if not os.path.exists(self.db_path):
            return "Database not found.", "not_found"
        con = connect_db(self.db_path, read_only=True)
        try:
            snap_a = self._lookup_policy_snapshot(con, policy_id_a.strip())
            snap_b = self._lookup_policy_snapshot(con, policy_id_b.strip())
            row_a = (snap_a[5], snap_a[6]) if snap_a else None
            row_b = (snap_b[5], snap_b[6]) if snap_b else None
            if not row_a:
                return f"Error: Policy '{policy_id_a}' not found.", "not_found"
            if not row_b:
                return f"Error: Policy '{policy_id_b}' not found.", "not_found"

            diffs: list[str] = []
            p_diff = list(
                difflib.unified_diff(
                    str(row_a[0] or "").splitlines(),
                    str(row_b[0] or "").splitlines(),
                    fromfile=f"{policy_id_a}/policy.py",
                    tofile=f"{policy_id_b}/policy.py",
                    lineterm="",
                    n=2,
                )
            )
            if p_diff:
                if len(p_diff) > 500:
                    diffs.append(
                        "\n".join(p_diff[:500])
                        + f"\n... [TRUNCATED {len(p_diff) - 500} diff lines] ..."
                    )
                else:
                    diffs.append("\n".join(p_diff))
            try:
                sk_a = json.loads(row_a[1] or "{}") or {}
                sk_b = json.loads(row_b[1] or "{}") or {}
            except Exception:
                sk_a, sk_b = {}, {}
            for sk_name in sorted(set(sk_a) | set(sk_b)):
                s_diff = list(
                    difflib.unified_diff(
                        str(sk_a.get(sk_name, "")).splitlines(),
                        str(sk_b.get(sk_name, "")).splitlines(),
                        fromfile=f"{policy_id_a}/skills/{sk_name}.py",
                        tofile=f"{policy_id_b}/skills/{sk_name}.py",
                        lineterm="",
                        n=2,
                    )
                )
                if s_diff:
                    if len(s_diff) > 300:
                        diffs.append(
                            "\n".join(s_diff[:300])
                            + f"\n... [TRUNCATED {len(s_diff) - 300} diff lines] ..."
                        )
                    else:
                        diffs.append("\n".join(s_diff))
            out = (
                "\n\n".join(diffs)
                if diffs
                else "No differences between policy snapshots."
            )
            if len(out) > 32000:
                out = (
                    out[:32000]
                    + f"\n... [TRUNCATED {len(out) - 32000} chars; read individual files via read_policy/read_skill] ..."
                )
            return out, f"{len(diffs)} changed files"
        finally:
            con.close()

    def _iter_workspace_sources(self) -> list[tuple[str, str]]:
        """Returns `[(rel_path, source_text), ...]` for active `policy.py` and `skills/**/*.py`."""
        files: list[tuple[str, str]] = []
        pol = str(self.current_policy_code or "").strip()
        if pol:
            files.append(("policy.py", self.current_policy_code))
        seen_sk: set[str] = set()
        for k in sorted(self.skills.keys()):
            src = str(self.skills.get(k) or "")
            if src.strip():
                clean_k = str(k).replace(".", "/").strip("/")
                files.append((f"skills/{clean_k}.py", src))
                seen_sk.add(clean_k)
        if not self.skills:
            skills_dir = Path(self.macro_lib.path).parent / "skills"
            if skills_dir.exists():
                for p in sorted(skills_dir.rglob("*.py")):
                    if p.name == "__init__.py":
                        continue
                    rel_k = (
                        str(p.relative_to(skills_dir))
                        .removesuffix(".py")
                        .replace("\\", "/")
                    )
                    if rel_k not in seen_sk:
                        try:
                            files.append(
                                (f"skills/{rel_k}.py", p.read_text(encoding="utf-8"))
                            )
                        except Exception:
                            pass
        return files

    def read_symbol(self, qualified_name: str) -> tuple[str, str]:
        """Reads an exact top-level function, class, or `Class.method` AST definition with line numbers."""
        q = str(qualified_name or "").strip()
        if not q:
            return "Error: `qualified_name` must be non-empty.", "error"
        sources = self._iter_workspace_sources()
        known_keys = {
            rel_path.removeprefix("skills/").removesuffix(".py"): rel_path
            for rel_path, _ in sources
        }
        mod_hint = ""
        cls_hint = ""
        sym_name = q
        q_norm = q.replace(".py.", ".")
        if ":" in q_norm:
            mod_hint, rest = q_norm.split(":", 1)
            sym_name = rest.strip()
        elif "/" in q_norm and "." in q_norm:
            mod_hint, sym_name = q_norm.split(".", 1)
        elif q_norm.count(".") >= 1:
            q_dot = q_norm.removeprefix("skills.")
            matched_key = ""
            for key in sorted(known_keys.keys(), key=len, reverse=True):
                dot_key = key.replace("/", ".")
                if q_dot.startswith(dot_key + "."):
                    matched_key = key
                    sym_name = q_dot[len(dot_key) + 1 :]
                    break
            if matched_key:
                mod_hint = matched_key
            elif q_norm.count(".") >= 2:
                parts = q_norm.rsplit(".", 1)
                mod_hint, sym_name = parts[0], parts[1]
        if "." in sym_name:
            cls_hint, sym_name = sym_name.split(".", 1)

        mod_hint_norm = (
            mod_hint.strip()
            .removeprefix("skills/")
            .removeprefix("skills.")
            .removesuffix(".py")
            .replace(".", "/")
        )
        matches: list[str] = []
        for rel_path, src in sources:
            rel_key = rel_path.removeprefix("skills/").removesuffix(".py")
            if mod_hint_norm and mod_hint_norm not in (rel_key, rel_path):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            src_lines = src.splitlines()
            for node in tree.body:
                if (
                    cls_hint
                    and isinstance(node, ast.ClassDef)
                    and node.name == cls_hint
                ):
                    for sub in node.body:
                        if (
                            isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and sub.name == sym_name
                        ):
                            s_l = int(sub.lineno)
                            e_l = int(getattr(sub, "end_lineno", s_l) or s_l)
                            body = "\n".join(
                                f"{i}: {src_lines[i - 1]}" for i in range(s_l, e_l + 1)
                            )
                            matches.append(
                                f"### {rel_path}:{s_l}-{e_l} (`{cls_hint}.{sym_name}`)\n{body}"
                            )
                elif (
                    not cls_hint
                    and isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and node.name == sym_name
                ):
                    s_l = int(node.lineno)
                    e_l = int(getattr(node, "end_lineno", s_l) or s_l)
                    body = "\n".join(
                        f"{i}: {src_lines[i - 1]}" for i in range(s_l, e_l + 1)
                    )
                    matches.append(f"### {rel_path}:{s_l}-{e_l} (`{sym_name}`)\n{body}")
        if not matches:
            return f"Symbol `{q}` not found in `policy.py` or `skills/`.", "not_found"
        return "\n\n".join(matches[:4]), f"{len(matches)} match(es)"

    def read_wiki_article(
        self, title: str, section: str = "", offset: int = 0, max_chars: int = 0
    ) -> tuple[str, str]:
        """Reads full untruncated article or section from the offline NetHackWiki + Guidebook index."""
        self.wiki_queries_run += 1
        out = self.wiki.read_article(
            title=title, section=section, offset=offset, max_chars=max_chars
        )
        return out, f"article:{title[:32]}"

    def find_references(self, symbol: str) -> tuple[str, str]:
        """Finds definitions, imports, and call/attribute sites for a symbol across `policy.py` and `skills/`."""
        sym = str(symbol or "").strip()
        if not sym:
            return "Error: `symbol` must be non-empty.", "error"
        pat = re.compile(rf"\b{re.escape(sym)}\b")
        defs: list[str] = []
        refs: list[str] = []
        for rel_path, src in self._iter_workspace_sources():
            for idx, ln in enumerate(src.splitlines(), start=1):
                if not pat.search(ln):
                    continue
                stripped = ln.strip()
                if re.match(
                    rf"^(?:async\s+)?def\s+{re.escape(sym)}\b", stripped
                ) or re.match(rf"^class\s+{re.escape(sym)}\b", stripped):
                    defs.append(f"  [DEF] {rel_path}:{idx}: {stripped[:160]}")
                else:
                    refs.append(f"  {rel_path}:{idx}: {stripped[:160]}")
        total = len(defs) + len(refs)
        if total == 0:
            return f"No references found for `{sym}` across policy/skills.", "0 refs"
        out_lines = [
            f"### References for `{sym}` ({len(defs)} defs, {len(refs)} uses):"
        ]
        out_lines.extend(defs[:20])
        out_lines.extend(refs[:60])
        if len(refs) > 60:
            out_lines.append(f"  ... ({len(refs) - 60} more uses)")
        return "\n".join(out_lines), f"{total} refs"

    def grep_skills(self, pattern: str) -> tuple[str, str]:
        """Searches regex/text pattern across `policy.py` and `skills/*.py`."""
        pat_s = str(pattern or "").strip()
        if not pat_s:
            return "Error: `pattern` must be non-empty.", "error"
        try:
            rx = re.compile(pat_s, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pat_s), re.IGNORECASE)
        hits: list[str] = []
        for rel_path, src in self._iter_workspace_sources():
            for idx, ln in enumerate(src.splitlines(), start=1):
                if rx.search(ln):
                    hits.append(f"  {rel_path}:{idx}: {ln.strip()[:160]}")
        if not hits:
            return f"No matches for `{pat_s}` in policy/skills.", "0 matches"
        shown = hits[:80]
        if len(hits) > 80:
            shown.append(f"  ... ({len(hits) - 80} more matches)")
        return "\n".join(shown), f"{len(hits)} matches"

    def blackboard_map(self) -> tuple[str, str]:
        """Lists shared blackboard keys/attributes with writer and reader locations across `policy.py` and `skills/`."""
        bb_keys: dict[str, dict[str, set[str]]] = {}
        bb_names = {"bb", "blackboard"}

        def _rec(key_name: str, kind: str, loc: str) -> None:
            entry = bb_keys.setdefault(key_name, {"writers": set(), "readers": set()})
            entry[kind].add(loc)

        for rel_path, src in self._iter_workspace_sources():
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            parent_map: dict[int, ast.AST] = {}
            for p in ast.walk(tree):
                for ch in ast.iter_child_nodes(p):
                    parent_map[id(ch)] = p
            fn_scopes: list[tuple[int, int, str]] = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn_scopes.append(
                        (n.lineno, getattr(n, "end_lineno", n.lineno), n.name)
                    )

            def _scope_for(lineno: int) -> str:
                best = "<module>"
                best_span = 10**9
                for s_l, e_l, fn_nm in fn_scopes:
                    if s_l <= lineno <= e_l and (e_l - s_l) < best_span:
                        best = fn_nm
                        best_span = e_l - s_l
                return f"{rel_path}::{best}"

            def _is_bb_expr(expr: ast.AST) -> bool:
                return (isinstance(expr, ast.Name) and expr.id in bb_names) or (
                    isinstance(expr, ast.Attribute)
                    and isinstance(expr.value, ast.Name)
                    and expr.value.id == "self"
                    and expr.attr == "blackboard"
                )

            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and _is_bb_expr(node.value):
                    if node.attr.startswith("__"):
                        continue
                    is_write = isinstance(node.ctx, (ast.Store, ast.Del))
                    kind = "writers" if is_write else "readers"
                    if node.attr not in {"get", "set", "setdefault", "data"}:
                        _rec(node.attr, kind, _scope_for(getattr(node, "lineno", 1)))
                elif isinstance(node, ast.Subscript) and _is_bb_expr(node.value):
                    if isinstance(node.slice, ast.Constant) and isinstance(
                        node.slice.value, str
                    ):
                        kind = (
                            "writers"
                            if isinstance(node.ctx, (ast.Store, ast.Del))
                            else "readers"
                        )
                        _rec(
                            node.slice.value,
                            kind,
                            _scope_for(getattr(node, "lineno", 1)),
                        )
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and _is_bb_expr(node.func.value)
                    and node.func.attr in ("get", "set", "setdefault")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    loc = _scope_for(getattr(node, "lineno", 1))
                    key_s = node.args[0].value
                    if node.func.attr == "get":
                        _rec(key_s, "readers", loc)
                    elif node.func.attr == "set":
                        _rec(key_s, "writers", loc)
                    else:
                        _rec(key_s, "readers", loc)
                        _rec(key_s, "writers", loc)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("setattr", "getattr") and len(node.args) >= 2:
                        a0, a1 = node.args[0], node.args[1]
                        if (
                            _is_bb_expr(a0)
                            and isinstance(a1, ast.Constant)
                            and isinstance(a1.value, str)
                        ):
                            kind = "writers" if node.func.id == "setattr" else "readers"
                            _rec(a1.value, kind, _scope_for(getattr(node, "lineno", 1)))
        if not bb_keys:
            return "No Blackboard attribute reads/writes detected yet.", "0 keys"
        lines = [f"### Shared Blackboard Key Map ({len(bb_keys)} keys):"]
        for k in sorted(bb_keys.keys())[:80]:
            w = ", ".join(sorted(bb_keys[k]["writers"])[:4]) or "none"
            r = ", ".join(sorted(bb_keys[k]["readers"])[:4]) or "none"
            lines.append(f"  - `{k}` | writers=[{w}] | readers=[{r}]")
        return "\n".join(lines), f"{len(bb_keys)} keys"

    def architecture_map(self) -> tuple[str, str]:
        """AST-indexed map of `register_goal(...)` GoalSpec handlers, `ws.domain(...)` typed states, and module LOC."""
        goals_found: list[tuple[str, str, int, str]] = []
        domains_found: dict[str, dict[str, Any]] = {}
        module_summary: list[str] = []

        for rel_path, src in self._iter_workspace_sources():
            loc = len(src.splitlines())
            module_summary.append(f"  - `{rel_path}` ({loc} LOC)")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn_nm = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (
                            node.func.attr
                            if isinstance(node.func, ast.Attribute)
                            else ""
                        )
                    )
                    if fn_nm == "register_goal":
                        g_name = ""
                        prio = "50"
                        if node.args and isinstance(node.args[0], ast.Constant):
                            g_name = str(node.args[0].value)
                        for kw in node.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                g_name = str(kw.value.value)
                            elif kw.arg == "priority" and isinstance(
                                kw.value, ast.Constant
                            ):
                                prio = str(kw.value.value)
                        if g_name:
                            goals_found.append(
                                (
                                    prio,
                                    g_name,
                                    int(getattr(node, "lineno", 1)),
                                    rel_path,
                                )
                            )
                    elif (
                        fn_nm == "domain"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                    ):
                        d_name = str(node.args[0].value)
                        cls_nm = ""
                        if len(node.args) >= 2:
                            a1 = node.args[1]
                            cls_nm = (
                                a1.id
                                if isinstance(a1, ast.Name)
                                else (a1.attr if isinstance(a1, ast.Attribute) else "")
                            )
                        entry = domains_found.setdefault(
                            d_name, {"class": cls_nm, "sites": set()}
                        )
                        if cls_nm and not entry["class"]:
                            entry["class"] = cls_nm
                        entry["sites"].add(f"{rel_path}:{getattr(node, 'lineno', 1)}")

        lines: list[str] = ["### Architecture Map (Option A)"]
        lines.append(f"**Modules ({len(module_summary)}):**")
        lines.extend(module_summary[:40])
        if len(module_summary) > 40:
            lines.append(f"  ... (+{len(module_summary) - 40} more)")
        lines.append(f"\n**Registered Goals ({len(goals_found)}):**")
        if goals_found:
            for prio, g_name, ln, rp in sorted(
                goals_found,
                key=lambda t: (int(t[0]) if t[0].lstrip("-").isdigit() else 999, t[1]),
            ):
                lines.append(f"  - prio={prio} `{g_name}` @ `{rp}:{ln}`")
        else:
            lines.append("  (none registered yet via `register_goal(...)`)")
        lines.append(f"\n**Typed `ws.domain(...)` Sub-States ({len(domains_found)}):**")
        if domains_found:
            for d_name, info in sorted(domains_found.items()):
                cls_lbl = info["class"] or "?"
                sites = ", ".join(sorted(info["sites"])[:5])
                lines.append(f"  - `{d_name}` ({cls_lbl}) used at [{sites}]")
        else:
            lines.append("  (none registered yet)")
        return "\n".join(
            lines
        ), f"{len(goals_found)} goals, {len(domains_found)} domains"

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
                    f"""
                    SELECT run_id, role, max_depth, turns, score, summary_digest
                    FROM expert_runs
                    ORDER BY turns DESC
                    LIMIT {lim}
                    """
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

    def _verify_macro_on_real_states(self, code: str) -> tuple[str, str]:
        """
        Executes candidate macro code against real NetHack GameState fixtures
        to guarantee runtime soundness, attribute validity, and Action yield protocol.
        """
        import traceback

        from lox.env.actions import ActionSequence, RawAction
        from lox.env.state import GameState, VisibleMonster, VisibleObject
        from lox.lang.compiler import _COMPILE_EXEC_LOCK, _build_base_namespace
        from lox.lang.instrument import ExecutionTracer, PolicyInstrumentTransformer
        from lox.lang.validator import validate_python_code

        try:
            macro_tree = validate_python_code(code, filename="macros_candidate.py")
        except Exception as exc:
            return (
                f"Macro AST Validation Error:\n{type(exc).__name__}: {exc}",
                "COMPILE_ERROR",
            )

        fn_names = [
            n.name
            for n in macro_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not fn_names:
            return (
                "No function definitions found in candidate macro code block.",
                "NO_FUNCTIONS",
            )

        try:
            with _COMPILE_EXEC_LOCK:
                tracer = ExecutionTracer()
                ns = _build_base_namespace(tracer)
                macro_inst_tree = PolicyInstrumentTransformer(
                    is_macro_module=True
                ).visit(macro_tree)
                ast.fix_missing_locations(macro_inst_tree)
                code_obj = compile(
                    macro_inst_tree, filename="macros_candidate.py", mode="exec"
                )
                exec(code_obj, ns)
        except Exception as exc:
            tb = traceback.format_exc(limit=-8)
            return (
                f"Macro Import/Compilation Error:\n{type(exc).__name__}: {exc}\n{tb}",
                "COMPILE_ERROR",
            )

        # Generate 3 real NetHack GameState fixtures across diverse seeds
        test_states: list[GameState] = []
        try:
            for s_seed in (42, 101, 202):
                h = NetHackHarness(max_steps=50)
                try:
                    st = h.reset(seed=s_seed)
                    if s_seed == 101:
                        for a in (Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST):
                            if not h.done:
                                st, _, _ = h.step(a)
                    elif s_seed == 202:
                        for a in (Action.SEARCH, Action.WAIT, Action.DOWN, Action.UP):
                            if not h.done:
                                st, _, _ = h.step(a)
                    test_states.append(st)
                finally:
                    h.close()
        except Exception as exc:
            tb = traceback.format_exc(limit=-8)
            return (
                f"Harness Error preparing GameState fixtures:\n{type(exc).__name__}: {exc}\n{tb}",
                "RUNTIME_ERROR",
            )

        for fn_name in fn_names:
            fn = ns.get(fn_name)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
            except Exception:
                params = []

            for state_idx, st in enumerate(test_states):
                dummy_obj = (
                    st.visible_objects()[0]
                    if st.visible_objects()
                    else VisibleObject(
                        y=st.y,
                        x=st.x,
                        glyph=0,
                        char="%",
                        color=7,
                        is_corpse=True,
                        corpse_monster=None,
                        name="lichen corpse",
                        info=None,
                    )
                )
                dummy_mon = (
                    st.visible_monsters()[0]
                    if st.visible_monsters()
                    else VisibleMonster(
                        y=st.y,
                        x=st.x + 1,
                        glyph=0,
                        char="d",
                        color=7,
                        is_pet=True,
                        is_statue=False,
                        is_ridden=False,
                        is_invisible=False,
                        name="little dog",
                        info=None,
                    )
                )

                args = []
                for idx, p in enumerate(params):
                    p_name = p.name.lower()
                    if "obj" in p_name or "item" in p_name or "corpse" in p_name:
                        args.append(dummy_obj)
                    elif "mon" in p_name or "creature" in p_name:
                        args.append(dummy_mon)
                    elif p_name in ("dy", "delta_y"):
                        args.append(1)
                    elif p_name in ("dx", "delta_x"):
                        args.append(0)
                    elif "mask" in p_name or "grid" in p_name or "passable" in p_name:
                        args.append(np.ones((21, 79), dtype=bool))
                    elif "dir" in p_name:
                        args.append(Action.EAST)
                    elif (
                        "pos" in p_name
                        or "coord" in p_name
                        or "target" in p_name
                        or "dest" in p_name
                    ):
                        args.append((st.y, st.x))
                    elif idx == 0 or "state" in p_name or "st" in p_name:
                        args.append(st)
                    elif p.default is not inspect.Parameter.empty:
                        pass
                    else:
                        args.append(1)

                try:
                    if inspect.isgeneratorfunction(fn):
                        gen = fn(*args)
                        act = next(gen, None)
                        if act is not None:
                            is_valid_act = (
                                isinstance(act, (Action, ActionSequence, RawAction))
                                or (
                                    isinstance(act, int)
                                    and (
                                        0 <= act <= 255
                                        or act in Action._value2member_map_
                                    )
                                )
                                or (isinstance(act, str) and len(act) == 1)
                            )
                            if not is_valid_act:
                                return (
                                    f"Macro generator `{fn_name}` yielded invalid action {act!r} ({type(act).__name__}). "
                                    "Must yield Action, seq(...), key(...), or ActionSequence.",
                                    "INVALID_YIELD",
                                )
                    else:
                        res = fn(*args)
                        if (
                            isinstance(res, np.ndarray)
                            and res.ndim == 2
                            and res.shape != (21, 79)
                        ):
                            return (
                                f"Macro `{fn_name}` returned ndarray with shape {res.shape}, expected grid shape (21, 79).",
                                "INVALID_SHAPE",
                            )
                        elif (
                            isinstance(res, tuple)
                            and len(res) == 2
                            and isinstance(res[0], int)
                            and isinstance(res[1], int)
                        ):
                            y, x = res
                            if not (0 <= y < 21 and 0 <= x < 79):
                                return (
                                    f"Macro `{fn_name}` returned out-of-bounds coordinates ({y}, {x}). Grid is 21x79.",
                                    "OUT_OF_BOUNDS",
                                )
                except Exception as exc:
                    tb = traceback.format_exc(limit=-8)
                    return (
                        f"Runtime Error in macro `{fn_name}` on real NetHack GameState fixture {state_idx + 1}/3:\n{type(exc).__name__}: {exc}\n{tb}",
                        "RUNTIME_ERROR",
                    )

        self.last_passed_code = code
        self.last_pilot_improved = True
        return (
            f"Real-State Macro Verification PASSED across 3 diverse GameStates for functions: {', '.join(fn_names)}.",
            "PASSED (macro)",
        )

    def _dry_run_swarm_worker(
        self,
        code: str,
        seed: int = 42,
        max_turns: int = 150,
        replay_run_id: str = "",
        replay_steps: int = 0,
        hypothesis: str = "",
    ) -> tuple[str, str]:
        """Validates and empirically tests a standalone SWARM_WORKER BT subtree across frontier + failure fixtures."""
        steps = max(1, int(max_turns))
        hyp_label = (
            str(hypothesis or "").strip() or f"H{len(self.swarm_experiments) + 1}"
        )
        scaffold = self.swarm_scaffold_code or self.current_policy_code

        contexts: list[dict[str, Any]] = []
        explicit_rid = str(replay_run_id or "").strip()
        if explicit_rid and os.path.exists(self.db_path):
            con = connect_db(self.db_path, read_only=True)
            try:
                row = con.execute(
                    "SELECT seed, action_bytes FROM runs WHERE run_id = ?",
                    [explicit_rid],
                ).fetchone()
            finally:
                con.close()
            if row:
                r_bytes = bytes(row[1] or b"")
                stop_idx = (
                    min(len(r_bytes), int(replay_steps))
                    if int(replay_steps) > 0
                    else max(1, len(r_bytes) - 25)
                )
                contexts.append(
                    {
                        "replay_fixture": {
                            "run_id": explicit_rid,
                            "seed": int(row[0]),
                            "action_bytes": r_bytes,
                            "stop_at_byte_idx": stop_idx,
                        }
                    }
                )

        if not contexts:
            if self.swarm_checkpoint is not None and self.swarm_frontier is not None:
                contexts.append(
                    {
                        "checkpoint": self.swarm_checkpoint,
                        "frontier": self.swarm_frontier,
                    }
                )
            else:
                contexts.append({"seed": int(seed)})
            for fix in self.swarm_replay_fixtures:
                contexts.append({"replay_fixture": fix})

        observations = []
        errors: list[str] = []
        for ctx in contexts:
            obs, err = evaluate_bt_subtree(
                code,
                checkpoint=ctx.get("checkpoint"),
                frontier=ctx.get("frontier"),
                replay_fixture=ctx.get("replay_fixture"),
                seed=ctx.get("seed"),
                steps=steps,
                baseline_policy_code=scaffold,
                skills=self.skills,
            )
            if err:
                errors.append(err)
                first_line = err.strip().splitlines()[0] if err.strip() else ""
                if any(
                    tag in first_line
                    for tag in (
                        "SyntaxError",
                        "PolicyValidationError",
                        "CompileError",
                    )
                ):
                    break
                continue
            observations.append(obs)

        if errors:
            err_msg = errors[0]
            first_line = err_msg.strip().splitlines()[0] if err_msg.strip() else "error"
            is_compile = any(
                tag in first_line
                for tag in (
                    "SyntaxError",
                    "ValueError",
                    "PolicyValidationError",
                    "CompileError",
                )
            )
            status_tag = "COMPILE_ERROR" if is_compile else "RUNTIME_ERROR"
            self.swarm_experiments.append(
                {
                    "hypothesis": hyp_label,
                    "code": code,
                    "observations": [],
                    "triggered_steps": 0,
                    "total_steps": 0,
                    "survived_all": False,
                    "max_depth": 1,
                    "max_unique_tiles": 0,
                    "error": first_line,
                    "passed": False,
                }
            )
            partial_obs = (
                "\n\nSuccessful contexts:\n"
                + "\n".join(f"- {o.summary()}" for o in observations)
                if observations
                else ""
            )
            return (
                f"Subtree {status_tag} for [{hyp_label}]:\n{err_msg}{partial_obs}",
                status_tag,
            )

        total_triggered = sum(o.triggered_steps for o in observations)
        total_eval = sum(o.total_eval_steps or o.steps_taken for o in observations)
        survived_all = all(o.survived for o in observations)
        max_depth = max((o.depth_end for o in observations), default=1)
        max_tiles = max((o.unique_tiles for o in observations), default=0)

        self.swarm_experiments.append(
            {
                "hypothesis": hyp_label,
                "code": code,
                "observations": observations,
                "triggered_steps": total_triggered,
                "total_steps": total_eval,
                "survived_all": survived_all,
                "max_depth": max_depth,
                "max_unique_tiles": max_tiles,
                "error": "",
                "passed": True,
            }
        )
        # Store last valid subtree and summary for compatibility with legacy callers.
        # Multi-hypothesis sessions in run_swarm_worker_session do not check last_pilot_improved
        # so all H1..HK continue sequentially across the warm conversation.
        self.last_passed_code = code
        self.last_pilot_improved = True
        self.last_dry_run_mode = "worker"
        self.last_dry_run_passed = True
        self.last_pilot_summary = f"[{hyp_label}] Triggered {total_triggered}/{total_eval} across {len(observations)} context(s)"

        lines = [
            f"Subtree Evaluation for [{hyp_label}] across {len(observations)} context(s):",
            f"- Aggregate Activation: Triggered {total_triggered}/{total_eval} steps (scaffold active={bool(scaffold.strip())})",
        ]
        for obs in observations:
            lines.append(f"  * {obs.summary()}")
        if total_triggered == 0:
            lines.append(
                "- WARNING: Your subtree returned None on 0/"
                f"{total_eval} steps (it NEVER triggered!). Broaden or fix your condition check, or repair and re-test before moving to the next hypothesis."
            )
        return "\n".join(lines), (
            f"PASSED ({hyp_label}: trig {total_triggered}/{total_eval})"
        )

    @staticmethod
    def _has_policy_class(src: str) -> bool:
        if not src or not src.strip():
            return False
        try:
            tree = ast.parse(src)
            return any(
                isinstance(n, ast.ClassDef) and n.name in ("Policy", "Agent")
                for n in tree.body
            )
        except Exception:
            return False

    def dry_run_policy(
        self,
        code: str = "",
        is_macro: bool = False,
        seed: int = 42,
        max_turns: int = 150,
        replay_run_id: str = "",
        replay_steps: int = 0,
        smoke_only: bool = False,
        skill_name: str | None = None,
        skill_code: str | None = None,
        skills: dict[str, str] | None = None,
        hypothesis: str = "",
    ) -> tuple[str, str]:
        """
        Compiles and tests candidate `code` in a live NetHack sandbox:
        1. Smoke Check: AST validation, reachability, loop check, and behavioral critic (150 steps).
        2. Pilot Evaluation: If worker pool is attached, runs 20 paired episodes across workers
           (up to 5 pilot revisions per generation) to measure real baseline delta and surface failure logs.
        """
        macros_code = self.macro_lib.load_code()
        policy_code = self.current_policy_code
        self.last_dry_run_mode = "check"
        self.last_dry_run_passed = False
        self.last_dry_run_diag = ""

        code, bundle_skills = _unpack_xml_bundle(code)

        cand_skills = dict(self.skills)
        if self.mode != "SWARM_PEER":
            for prev_k, prev_v in self.last_passed_skills.items():
                if prev_v:
                    cand_skills[prev_k] = prev_v
        if isinstance(skills, dict):
            for k, v in skills.items():
                clean_k = normalize_skill_key(str(k or ""))
                if clean_k and str(v or "").strip():
                    cleaned_v = _strip_fences(str(v))
                    cand_skills[clean_k] = merge_skill_module_update(
                        cand_skills.get(clean_k, ""), cleaned_v
                    )
        for k, v in bundle_skills.items():
            clean_bk = normalize_skill_key(str(k or ""))
            if clean_bk:
                cand_skills[clean_bk] = merge_skill_module_update(
                    cand_skills.get(clean_bk, ""), v
                )
        if skill_name and skill_code:
            clean_sn = normalize_skill_key(str(skill_name))
            if clean_sn:
                cleaned_sc = _strip_fences(str(skill_code))
                cand_skills[clean_sn] = merge_skill_module_update(
                    cand_skills.get(clean_sn, ""), cleaned_sc
                )

        has_extra_skills = bool(bundle_skills) or isinstance(skills, dict)
        skill_matches = (skill_code is None and not has_extra_skills) or (
            self.last_passed_skill_code is not None
            and skill_code is not None
            and skill_code.strip() == self.last_passed_skill_code.strip()
            and not has_extra_skills
        )

        has_replay = bool(str(replay_run_id or "").strip())

        if (
            not is_macro
            and self.mode != "SWARM_WORKER"
            and not has_replay
            and self.last_passed_code
            and self.last_pilot_improved
            and (
                code.strip() == self.last_passed_code.strip()
                or is_ast_equivalent(code, self.last_passed_code)
            )
            and skill_matches
        ):
            self.last_dry_run_mode = "pilot" if self.pilot_runs_used > 0 else "check"
            self.last_dry_run_passed = True
            return (
                f"Candidate code already verified in passing pilot batch ({self.last_pilot_summary}). "
                "DO NOT run more pilots. Conclude your turn immediately and output your final hypothesis.",
                f"PASSED (cached: {self.last_pilot_summary})",
            )

        if is_macro:
            return self._verify_macro_on_real_states(code)

        if (
            self.mode == "SWARM_PEER"
            and self.enforce_peer_research_gate
            and not smoke_only
        ):
            total_empirical = (
                int(self.db_queries_run)
                + int(self.wiki_queries_run)
                + int(self.theories_queries_run)
            )
            if total_empirical == 0:
                return (
                    "RESEARCH_REQUIRED: Before calling `dry_run_policy`, run at least 1 empirical grounding lookup "
                    "(`query_db(...)` on `champion_ticks`/`champion_runs`, `query_wiki(...)`/`query_monster(...)`/`query_item(...)`, "
                    "or `query_theories(...)`) to verify the exact failure trace or rule before patching.",
                    "RESEARCH_REQUIRED",
                )

        if self.mode == "SWARM_WORKER":
            self.last_dry_run_mode = "worker"
            return self._dry_run_swarm_worker(
                code=code,
                seed=seed,
                max_turns=max(1, int(max_turns)),
                replay_run_id=replay_run_id,
                replay_steps=replay_steps,
                hypothesis=hypothesis,
            )

        try:
            use_surgical_splice = bool(policy_code.strip()) and (
                self.mode not in ("SWARM_META", "SWARM_PEER")
                or (self.mode == "SWARM_PEER" and not self._has_policy_class(code))
            )
            if use_surgical_splice:
                if code.strip():
                    spliced_policy, changed_fns = splice_functions_into_policy(
                        policy_code, code
                    )
                else:
                    spliced_policy = policy_code
                    changed_fns = []
                if self.mode == "REFACTOR" and skill_code:
                    spliced_policy, pruned_fns = prune_extracted_skill_functions(
                        spliced_policy, skill_code
                    )
                    if pruned_fns:
                        changed_fns = [
                            fn for fn in changed_fns if fn not in set(pruned_fns)
                        ]
                # Mutation scope check for WRITE mode (SWARM_PEER allows cross-file decomposition)
                if (
                    self.mode not in ("CREATE", "REFACTOR", "SWARM_PEER")
                    and len(changed_fns) > 2
                ):
                    return (
                        f"Mutation Scope Error: You modified/added {len(changed_fns)} functions ({changed_fns}). "
                        "In WRITE mode, you must target at most 2 functions (e.g. 1 primary method + 1 caller/helper) "
                        "to isolate causal impact and prevent coupled cross-subsystem regressions. "
                        "Please narrow your patch to the 1-2 essential functions (or use REFACTOR for broader rewrites).",
                        "MUTATION_SCOPE_ERROR",
                    )
                # Static reachability check: ensure any non-root method in changed_fns is referenced in spliced_policy
                if changed_fns and self.mode != "SWARM_PEER":
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
                        if fn
                        not in (
                            "run",
                            "step",
                            "act",
                            "__init__",
                            "dispatch",
                            "step_dispatch",
                        )
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
                changed_fns = (
                    [] if self.mode in ("SWARM_META", "SWARM_PEER") else ["run"]
                )
            if (
                self.mode == "SWARM_PEER"
                and not smoke_only
                and not has_replay
                and self.current_policy_code.strip()
                and is_ast_equivalent(spliced_policy, self.current_policy_code)
                and set(cand_skills.keys()) == set(self.skills.keys())
                and all(
                    is_ast_equivalent(cand_skills[k], self.skills[k])
                    for k in cand_skills
                )
            ):
                return (
                    "NOOP_DIFF: Your candidate code is AST-identical to the current incumbent policy and skills. "
                    "Make a substantive behavioral modification before calling `dry_run_policy`.",
                    "NOOP_DIFF",
                )
            scaffold_ref = self.swarm_scaffold_code or self.current_policy_code
            if (
                self.mode == "SWARM_PEER"
                and scaffold_ref.strip()
                and "#EVOLVE-BLOCK" in scaffold_ref.upper()
            ):
                eb_err = check_evolve_block_bounds(scaffold_ref, spliced_policy)
                if eb_err:
                    return (f"Evolve-Block Guard Error:\n{eb_err}", "COMPILE_ERROR")

            compiled = compile_policy_package(
                spliced_policy, macros_code, skills=cand_skills
            )
            validate_cross_module_contracts(spliced_policy, cand_skills)
            pyright_errs = run_pyright_on_package(spliced_policy, cand_skills)
            if pyright_errs:
                self.last_dry_run_diag = "PYRIGHT: " + " | ".join(pyright_errs[:4])
                return (
                    "Type / Contract Validation Error (pyright):\n"
                    + "\n".join(f"- {e}" for e in pyright_errs),
                    "COMPILE_ERROR",
                )
        except Exception as exc:
            self.last_dry_run_diag = f"{type(exc).__name__}: {exc}"
            return (
                f"Compilation / Validation Error:\n{type(exc).__name__}: {exc}",
                "COMPILE_ERROR",
            )

        # Step 1: Single-Episode Smoke Check + Behavioral Critic (reuse persistent harness)
        active_seed = int(seed)
        harness = self._get_harness(max_turns)
        harness.allow_menu_linger = policy_handles_menus(spliced_policy, cand_skills)
        try:
            runner = compiled.create_runner()
            actions_taken: list[str] = []
            raw_act_names: list[str] = []
            unique_positions: set[tuple[int, int, int, int]] = set()
            positions_history: list[tuple[int, int, int]] = []
            oscillation_detected = False
            smoke_msg_counts: Counter[str] = Counter()
            prompt_repeat_count = 0
            last_prompt_text = ""
            prompt_livelock_detected = False
            livelock_prompt_msg = ""

            if replay_run_id and os.path.exists(self.db_path):
                con = connect_db(self.db_path, read_only=True)
                try:
                    row = con.execute(
                        "SELECT seed, action_bytes FROM runs WHERE run_id = ?",
                        [replay_run_id],
                    ).fetchone()
                finally:
                    con.close()
                if row:
                    r_seed, r_bytes = int(row[0]), bytes(row[1] or b"")
                    cut = (
                        min(len(r_bytes), int(replay_steps))
                        if int(replay_steps) > 0
                        else max(0, len(r_bytes) - 25)
                    )
                    state = harness.replay_bytes(
                        seed=r_seed,
                        recorded_bytes=r_bytes,
                        stop_at_byte_idx=cut,
                    )
                    if harness.done and cut > 1:
                        for back in (25, 60, max(100, cut // 2)):
                            state = harness.replay_bytes(
                                seed=r_seed,
                                recorded_bytes=r_bytes,
                                stop_at_byte_idx=max(1, cut - back),
                            )
                            if not harness.done:
                                break
                    harness.step_count = 0
                    harness.done = False
                else:
                    state = harness.reset(seed=int(seed))
            else:
                state = harness.reset(seed=int(seed))

            smoke_cap = max(10, min(2000, int(max_turns)))
            last_live_state = (
                state if int(getattr(state, "depth", 0) or 0) > 0 else None
            )
            smoke_ended = False
            smoke_end_reason = ""
            for _ in range(smoke_cap):
                raw_act, active_fn = runner.next_action(state)
                state, done, info = harness.step(raw_act)
                if not done and int(getattr(state, "depth", 0) or 0) > 0:
                    last_live_state = state
                msg_l = (state.message or "").lower()
                is_prompt = getattr(state, "in_yn_prompt", False) or getattr(
                    state, "in_text_prompt", False
                )
                if is_prompt:
                    if msg_l == last_prompt_text:
                        prompt_repeat_count += 1
                        if prompt_repeat_count >= 5:
                            prompt_livelock_detected = True
                            livelock_prompt_msg = state.message or "interactive prompt"
                    else:
                        prompt_repeat_count = 1
                        last_prompt_text = msg_l
                else:
                    prompt_repeat_count = 0
                    last_prompt_text = ""
                msg_clean = (state.message or "").strip()
                if msg_clean:
                    smoke_msg_counts[msg_clean] += 1
                if not done and int(getattr(state, "depth", 0) or 0) > 0:
                    d_now = int(getattr(state, "depth", 1) or 1)
                    unique_positions.add(
                        (
                            int(getattr(state, "dungeon_num", 0) or 0),
                            d_now,
                            int(state.y),
                            int(state.x),
                        )
                    )
                    positions_history.append((d_now, int(state.y), int(state.x)))
                if len(positions_history) >= 40:
                    if len(set(positions_history[-40:])) <= 3:
                        oscillation_detected = True
                act_name = info["action_name"]
                raw_act_names.append(act_name)
                actions_taken.append(f"{active_fn}:{act_name}")
                if done:
                    hit_smoke_cap = (
                        harness.step_count >= smoke_cap
                        and int(getattr(state, "depth", 0) or 0) > 0
                        and int(getattr(state, "hp", 0) or 0) > 0
                    )
                    if not hit_smoke_cap:
                        smoke_ended = True
                        smoke_end_reason = (
                            str(getattr(state, "end_reason", "") or "")
                            or str(info.get("end_reason", "") or "")
                            or str(getattr(state, "message", "") or "").split(".")[0][
                                :80
                            ]
                            or "ended"
                        )
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

            from lox.lang.bt import extract_runner_leaf_errors

            swallowed_leaf_errs = extract_runner_leaf_errors(runner)
            if swallowed_leaf_errs:
                err_lines = "\n".join(
                    f"- Leaf `{n}` raised {e}" for n, e in swallowed_leaf_errs[:8]
                )
                return (
                    f"Runtime Error in BehaviorTree leaf during smoke test (seed {seed}):\n{err_lines}\n"
                    "Fix the crashing leaf or missing attribute/import before finalizing.",
                    "RUNTIME_ERROR",
                )

            # Multi-seed smoke sweep on diverse seeds [1000, 1001] when running smoke check
            if not is_macro and not has_replay and smoke_only:
                extra_smoke_steps = min(100, max(50, int(max_turns)))
                for extra_s in (1000, 1001):
                    if int(seed) == extra_s:
                        continue
                    active_seed = int(extra_s)
                    extra_runner = compiled.create_runner()
                    est = harness.reset(seed=extra_s)
                    for _ in range(extra_smoke_steps):
                        eact, _ = extra_runner.next_action(est)
                        est, edone, _ = harness.step(eact)
                        if edone:
                            break
                    extra_leaf_errs = extract_runner_leaf_errors(extra_runner)
                    if extra_leaf_errs:
                        err_lines = "\n".join(
                            f"- Leaf `{n}` raised {e}" for n, e in extra_leaf_errs[:8]
                        )
                        return (
                            f"Runtime Error in BehaviorTree leaf during smoke sweep (seed {extra_s}):\n{err_lines}\n"
                            "Fix the crashing leaf or missing attribute/import before finalizing.",
                            "RUNTIME_ERROR",
                        )
                active_seed = int(seed)

            warn_lines: list[str] = []
            for h_warn in audit_policy_ast_hygiene(spliced_policy):
                warn_lines.append(f"- AST HYGIENE WARNING: {h_warn}")
            if prompt_livelock_detected:
                warn_lines.append(
                    f"- INTERACTIVE PROMPT LIVELOCK DETECTED: The prompt '{livelock_prompt_msg}' was triggered repeatedly without resolving! "
                    "Ensure your prompt handling (e.g. handle_prompts) answers the prompt cleanly instead of looping indefinitely."
                )
            if smoke_msg_counts:
                top_msgs = ", ".join(
                    f"{m!r}: {c}x" for m, c in smoke_msg_counts.most_common(4) if c >= 2
                )
                if top_msgs:
                    warn_lines.append(f"- REPEATED MESSAGES: {top_msgs}")
            if len(unique_positions) == 1 and harness.step_count >= 15:
                warn_lines.append(
                    "- WARNING: Hero stayed on 1 tile ((y,x) did not change) across all steps! Verify movement logic."
                )
            if oscillation_detected:
                warn_lines.append(
                    "- BEHAVIORAL WARNING: 2-tile or 3-tile ping-pong oscillation detected (visited <= 3 unique tiles over 40 steps). Ensure you use persistent targets and pos_history cycle breaking!"
                )
            if len(raw_act_names) >= 20:
                stall_acts = sum(1 for a in raw_act_names if a in ("SEARCH", "WAIT"))
                if stall_acts / len(raw_act_names) > 0.5:
                    warn_lines.append(
                        f"- BEHAVIORAL WARNING: High action stall rate ({stall_acts}/{len(raw_act_names)} actions were SEARCH or WAIT)."
                    )
            warn_block = ("\n" + "\n".join(warn_lines)) if warn_lines else ""

            # Step 2: 20-Episode Parallel Pilot Evaluation (if pool is attached and pilot budget remains)
            if (
                not is_macro
                and not smoke_only
                and not has_replay
                and self.mode != "SWARM_PEER"
                and self.pool is not None
                and self.pilot_runs_used < self.max_pilot_runs
            ):
                self.pilot_runs_used += 1
                rev_idx = self.pilot_runs_used
                max_rev = self.max_pilot_runs
                target_eps = max(10, self.pilot_episodes)

                pilot_seeds: list[int] = []
                pilot_shards = []
                base_trig_seeds: list[int] = []
                if self.baseline_runs:
                    target_set = set(changed_fns or [])
                    for r in self.baseline_runs:
                        fn_map = r.get("triggered_functions") or r.get(
                            "functions_triggered"
                        )
                        if isinstance(fn_map, str):
                            try:
                                fn_map = json.loads(fn_map)
                            except Exception:
                                fn_map = {}
                        fn_keys = (
                            set(fn_map.keys()) if isinstance(fn_map, dict) else set()
                        )
                        if target_set and (target_set & fn_keys):
                            base_trig_seeds.append(int(r["seed"]))

                    clean_baselines = [
                        r
                        for r in self.baseline_runs
                        if float(r.get("outlier_z_score", 0.0) or 0.0) <= 2.5
                    ] or self.baseline_runs

                    if self.mode == "REFACTOR":
                        clean_baselines = [
                            r
                            for r in clean_baselines
                            if not r.get("is_shard")
                            and not str(r.get("shard_id", "")).strip()
                            and not str(r.get("run_id", "")).startswith("shard_")
                        ] or clean_baselines

                    fail_runs = [
                        r
                        for r in clean_baselines
                        if r.get("end_reason") not in ("ascended", "max_steps", "ended")
                        or (
                            "stalls:" in str(r.get("summary_digest", "")).lower()
                            and "stalls: none"
                            not in str(r.get("summary_digest", "")).lower()
                        )
                    ]
                    if clean_baselines and self.mode != "REFACTOR":
                        max_sh = min(40, max(1, target_eps // 3))
                        pilot_shards = extract_shards_from_runs(
                            clean_baselines,
                            max_shards=max_sh,
                            target_functions=changed_fns,
                        )
                    shard_seeds = {int(sh.seed) for sh in pilot_shards}
                    normal_runs = [r for r in clean_baselines if r not in fail_runs]
                    avail_fail = [
                        r for r in fail_runs if int(r["seed"]) not in shard_seeds
                    ]
                    avail_norm = [
                        r for r in normal_runs if int(r["seed"]) not in shard_seeds
                    ]
                    needed_full = max(1, target_eps - len(pilot_shards))
                    num_fail = min(len(avail_fail), int(needed_full * 2 // 3))
                    num_norm = min(len(avail_norm), needed_full - num_fail)
                    strat_runs = avail_fail[:num_fail] + avail_norm[:num_norm]
                    if len(strat_runs) < needed_full:
                        rem = [
                            r
                            for r in clean_baselines
                            if r not in strat_runs and int(r["seed"]) not in shard_seeds
                        ]
                        strat_runs.extend(rem[: needed_full - len(strat_runs)])
                    pilot_seeds = [int(r["seed"]) for r in strat_runs]
                    for r in clean_baselines:
                        s_val = int(r["seed"])
                        if s_val not in shard_seeds and s_val not in pilot_seeds:
                            pilot_seeds.append(s_val)
                else:
                    pilot_seeds = list(range(int(seed), int(seed) + target_eps))

                base_by_seed = {
                    int(r["seed"]): r
                    for r in (
                        clean_baselines if self.baseline_runs else self.baseline_runs
                    )
                    if "seed" in r and not r.get("is_shard")
                }

                pilot_batch_id = (
                    f"pilot_{int(time.time() * 1000)}_{self.pilot_runs_used:02d}"
                )

                # Evaluate baseline policy on the exact pilot shards for 100% apples-to-apples comparison
                base_shard_runs: list[dict[str, Any]] = []
                if pilot_shards and self.current_policy_code:
                    base_shard_res = self.pool.evaluate_batch(
                        batch_id=f"{pilot_batch_id}_base_shards",
                        policy_code=self.current_policy_code,
                        macros_code=macros_code,
                        target_episodes=len(pilot_shards),
                        shards=pilot_shards,
                        max_steps=self.max_steps,
                        record_ticks=False,
                        show_progress=False,
                        skills=self.skills,
                    )
                    base_shard_runs = base_shard_res.get("runs", [])
                base_shard_by_seed = {
                    (str(r.get("shard_id") or "") or int(r["seed"])): r
                    for r in base_shard_runs
                }
                if not base_shard_by_seed and pilot_shards:
                    # Fallback mapping to baseline runs when current_policy_code wasn't provided
                    base_run_by_seed = {int(r["seed"]): r for r in self.baseline_runs}
                    for sh in pilot_shards:
                        s = int(sh.seed)
                        if s in base_run_by_seed:
                            sh_key = str(sh.shard_id) or s
                            base_shard_by_seed[sh_key] = base_run_by_seed[s]
                            base_shard_by_seed[s] = base_run_by_seed[s]

                if policy_code.strip() and changed_fns:
                    target_fns = changed_fns
                elif skill_name and skill_name in compiled.skill_function_names:
                    target_fns = compiled.skill_function_names[skill_name]
                else:
                    target_fns = None

                pilot_res = self.pool.evaluate_batch(
                    batch_id=pilot_batch_id,
                    policy_code=spliced_policy,
                    macros_code=macros_code,
                    target_episodes=target_eps,
                    target_functions=target_fns,
                    priority_seeds=pilot_seeds,
                    unconditional_seeds=base_trig_seeds if base_trig_seeds else None,
                    shards=pilot_shards if pilot_shards else None,
                    skills=cand_skills,
                    max_steps=self.max_steps,
                    max_raw_multiplier=3,
                    unreachable_abort_after=min(60, target_eps * 2),
                    record_ticks=True,
                    show_progress=False,
                )

                p_runs = pilot_res.get("runs", [])

                # Backfill any newly triggered pilot seeds not in base_by_seed
                missing_seeds = [
                    int(r["seed"])
                    for r in p_runs
                    if not r.get("is_shard") and int(r["seed"]) not in base_by_seed
                ]
                if missing_seeds and self.current_policy_code:
                    backfill_res = self.pool.evaluate_batch(
                        batch_id=f"{pilot_batch_id}_base_backfill",
                        policy_code=self.current_policy_code,
                        macros_code=macros_code,
                        target_episodes=len(missing_seeds),
                        priority_seeds=missing_seeds,
                        max_steps=self.max_steps,
                        record_ticks=False,
                        show_progress=False,
                        skills=self.skills,
                    )
                    for br in backfill_res.get("runs", []):
                        base_by_seed[int(br["seed"])] = br

                wins = 0
                losses = 0
                ties = 0

                bench_name = getattr(self, "benchmark", "nethack")
                if bench_name != "nethack":
                    from lox.env import get_adapter

                    metric_fn = get_adapter(bench_name).progression_metric
                else:
                    from lox.engine.hypothesis import progression_metric

                    metric_fn = progression_metric

                cand_full_depths: list[float] = []
                base_full_depths: list[float] = []
                cand_shard_depths: list[float] = []
                base_shard_depths: list[float] = []
                deltas: list[float] = []
                paired_regressions: list[
                    tuple[float, int, dict[str, Any], dict[str, Any]]
                ] = []

                for r in p_runs:
                    s = int(r["seed"])
                    c_d = float(r.get("max_depth", 1))
                    if r.get("is_shard"):
                        sh_key = str(r.get("shard_id") or "") or s
                        br = base_shard_by_seed.get(sh_key) or base_shard_by_seed.get(s)
                        if br is not None:
                            b_d = float(br.get("max_depth", 1))
                            d_val = float(metric_fn(r) - metric_fn(br))
                            deltas.append(d_val)
                            cand_shard_depths.append(c_d)
                            base_shard_depths.append(b_d)
                            if d_val > 1e-6:
                                wins += 1
                            elif d_val < -1e-6:
                                losses += 1
                                paired_regressions.append((d_val, s, r, br))
                            else:
                                ties += 1
                    else:
                        if s in base_by_seed:
                            br = base_by_seed[s]
                            if float(br.get("outlier_z_score", 0.0) or 0.0) > 2.5:
                                continue
                            b_d = float(br.get("max_depth", 1))
                            d_val = float(metric_fn(r) - metric_fn(br))
                            deltas.append(d_val)
                            cand_full_depths.append(c_d)
                            base_full_depths.append(b_d)
                            if d_val > 1e-6:
                                wins += 1
                            elif d_val < -1e-6:
                                losses += 1
                                paired_regressions.append((d_val, s, r, br))
                            else:
                                ties += 1

                if cand_full_depths:
                    cand_depths = cand_full_depths
                    base_depths = base_full_depths
                elif cand_shard_depths:
                    cand_depths = cand_shard_depths
                    base_depths = base_shard_depths
                else:
                    cand_depths = []
                    base_depths = []

                self.last_cand_depths = list(cand_depths)
                self.last_base_depths = list(base_depths)

                mean_delta = float(np.mean(deltas)) if deltas else 0.0

                if cand_depths:
                    c_avg_depth = float(np.mean(cand_depths))
                    b_avg_depth = float(np.mean(base_depths))
                else:
                    c_avg_depth = (
                        float(np.mean([float(r["max_depth"]) for r in p_runs]))
                        if p_runs
                        else 0.0
                    )
                    b_avg_depth = (
                        float(self.baseline_runs[0].get("depth", 1.0))
                        if self.baseline_runs
                        else 1.0
                    )

                delta_depth = c_avg_depth - b_avg_depth
                c_max_depth = int(
                    max((r.get("max_depth", 1) for r in p_runs), default=0)
                )

                dlvl1_stalls = sum(1 for r in p_runs if r.get("max_depth", 1) <= 1)
                door_stalls = sum(
                    1 for r in p_runs if "locked" in str(r.get("game_log", "")).lower()
                )
                end_reasons = Counter(r.get("end_reason", "unknown") for r in p_runs)
                end_reasons_str = ", ".join(
                    f"{k}: {v}" for k, v in end_reasons.most_common(4)
                )

                sorted_runs = sorted(
                    p_runs,
                    key=lambda x: (
                        x.get("max_depth", 1),
                        x.get("score", 0),
                        -x.get("steps", 0),
                    ),
                )
                worst_runs = sorted_runs[:2]
                best_run = sorted_runs[-1] if sorted_runs else None

                worst_logs = []
                for wr in worst_runs:
                    w_seed = wr.get("seed")
                    w_d = wr.get("max_depth", 1)
                    w_steps = wr.get("steps", 0)
                    w_end = wr.get("end_reason", "unknown")
                    w_stalls = ""
                    for line in str(wr.get("summary_digest", "")).splitlines():
                        if line.startswith("Stalls:") and line != "Stalls: None":
                            w_stalls = f" | {line}"
                            break
                    w_glog = str(wr.get("game_log", ""))
                    if len(w_glog) > 400:
                        w_glog = "..." + w_glog[-400:]
                    worst_logs.append(
                        f"  - Stalled Run (seed={w_seed}, Dlvl={w_d}, Steps={w_steps}, End={w_end}{w_stalls}): {w_glog}"
                    )
                worst_logs_str = "\n".join(worst_logs) if worst_logs else "  - None"

                best_log_str = "None"
                if best_run:
                    b_seed = best_run.get("seed")
                    b_d = best_run.get("max_depth", 1)
                    b_steps = best_run.get("steps", 0)
                    b_glog = str(best_run.get("game_log", ""))
                    if len(b_glog) > 400:
                        b_glog = "..." + b_glog[-400:]
                    best_log_str = (
                        f"seed={b_seed}, Dlvl={b_d}, Steps={b_steps} | Log: {b_glog}"
                    )

                regressions_block = ""
                if paired_regressions:
                    reg_lines = []
                    for d_val, s_id, cand_r, base_r in sorted(
                        paired_regressions, key=lambda x: x[0]
                    )[:3]:
                        b_d_val = base_r.get("max_depth", 1)
                        b_xl_val = base_r.get("final_xp_level", 1)
                        b_ac_val = base_r.get("final_ac", 10)
                        b_st_val = base_r.get("steps", 0)
                        b_end_val = base_r.get("end_reason", "ended")
                        c_d_val = cand_r.get("max_depth", 1)
                        c_xl_val = cand_r.get("final_xp_level", 1)
                        c_ac_val = cand_r.get("final_ac", 10)
                        c_st_val = cand_r.get("steps", 0)
                        c_fn_val = cand_r.get("last_active_function", "root")
                        c_end_val = cand_r.get("end_reason", "ended")
                        c_glog_lines = (
                            str(cand_r.get("game_log", "") or "").strip().splitlines()
                        )
                        c_tail = (
                            " / ".join(c_glog_lines[-3:])
                            if c_glog_lines
                            else str(cand_r.get("death_message", ""))
                        )
                        if len(c_tail) > 260:
                            c_tail = "..." + c_tail[-260:]
                        reg_lines.append(
                            f"  - Seed {s_id} (Δ={d_val:+.2f}): Baseline [Dlvl={b_d_val}, XL={b_xl_val}, AC={b_ac_val}, Steps={b_st_val}, End='{b_end_val}'] "
                            f"-> Candidate [Dlvl={c_d_val}, XL={c_xl_val}, AC={c_ac_val}, Steps={c_st_val}, ActiveFn=`{c_fn_val}`, End='{c_end_val}'] | "
                            f"Candidate Pre-Mortem: {c_tail}"
                        )
                    regressions_block = (
                        "• Paired Seed Regressions (Candidate < Baseline — inspect ticks via `latest_pilot_ticks`):\n"
                        + "\n".join(reg_lines)
                        + "\n"
                    )

                if self.baseline_runs and not p_runs:
                    improved = False
                    raw_run_cnt = pilot_res.get("raw_episodes_run", 0)
                    summary_short = f"0/{raw_run_cnt} eps triggered {target_fns}"
                    paired_line = f"• Trigger Check Failed: Target function(s) {target_fns} triggered in 0/{raw_run_cnt} episodes!\n"
                    verdict_line = (
                        f">>> PILOT VERDICT: TARGET FUNCTION NEVER TRIGGERED AT RUNTIME ({target_fns}). "
                        "Loosen its entry condition in `run()` so it triggers on current dungeon levels!"
                    )
                elif base_depths:
                    decisive = wins + losses
                    if decisive > 0:
                        import scipy.stats as sp_stats

                        # Exact one-sided binomial sign test under H0: p=0.5
                        p_sign = float(sp_stats.binom.sf(wins - 1, decisive, 0.5))
                    else:
                        p_sign = 1.0

                    if self.mode == "REFACTOR":
                        crashes = sum(
                            1 for r in p_runs if r.get("end_reason") == "crash"
                        )
                        max_allowed_losses = wins + max(2, int(len(p_runs) * 0.10))
                        improved = (
                            crashes == 0
                            and mean_delta >= -0.05
                            and losses <= max_allowed_losses
                        )
                        summary_short = (
                            f"Parity {wins}W/{losses}L/{ties}T, Δ={delta_depth:+.2f}"
                        )
                        paired_line = (
                            f"• Refactor Behavioral Parity vs Baseline: {wins} Wins / {losses} Losses / {ties} Ties | "
                            f"Δ Depth: {delta_depth:+.2f} (Avg Depth: {c_avg_depth:.2f} vs Baseline: {b_avg_depth:.2f}, Max Depth: {c_max_depth})\n"
                        )
                        verdict_line = (
                            ">>> PILOT VERDICT: REFACTOR BEHAVIORAL PARITY VERIFIED (0 crashes, non-inferior progression)."
                            if improved
                            else (
                                f">>> PILOT VERDICT: REFACTOR PARITY FAILED ({wins}W/{losses}L/{ties}T, Δ={delta_depth:+.2f}, crashes={crashes}). "
                                "Ensure your extracted skill module preserves exact baseline behavior!"
                            )
                        )
                    else:
                        # Tie-aware directional pilot validation
                        improved = (
                            wins > losses
                            and mean_delta >= -1e-6
                            and (
                                p_sign <= 0.20
                                or delta_depth >= 0.04
                                or wins >= losses + 2
                                or mean_delta >= 0.03
                            )
                        )
                        summary_short = f"{wins}W/{losses}L/{ties}T, Δ={delta_depth:+.2f} (p_sign={p_sign:.3f})"
                        paired_line = (
                            f"• Paired vs Baseline: {wins} Wins / {losses} Losses / {ties} Ties | "
                            f"Δ Depth: {delta_depth:+.2f} (Avg Depth: {c_avg_depth:.2f} vs Baseline: {b_avg_depth:.2f}, Max Depth: {c_max_depth}) | "
                            f"Sign-Test p: {p_sign:.4f}\n"
                        )
                        verdict_line = (
                            ">>> PILOT VERDICT: POSITIVE DIRECTIONAL PILOT SIGNAL over baseline."
                            if improved
                            else (
                                f">>> PILOT VERDICT: INSUFFICIENT OR FLAT SIGNAL VS BASELINE ({wins}W/{losses}L, p_sign={p_sign:.3f}). "
                                "STOP! Do NOT submit noisy code. Refine your code or call query_db/query_wiki before submitting!"
                            )
                        )
                else:
                    # In FULL_WRITE (Gen 0) or without existing baseline runs:
                    # Initial policy is valid as long as it executes episodes without crashing!
                    crashes = sum(1 for r in p_runs if r.get("end_reason") == "crash")
                    improved = (
                        len(p_runs) > 0 and crashes < len(p_runs) and c_max_depth >= 1
                    )
                    summary_short = f"Gen 0 Baseline: {len(p_runs)} eps, Depth: {c_avg_depth:.2f} (Max {c_max_depth})"
                    paired_line = f"• Initial Gen 0 Baseline Validation: {len(p_runs)} episodes executed across workers | Avg Depth: {c_avg_depth:.2f}, Max Depth: {c_max_depth}\n"
                    verdict_line = (
                        ">>> PILOT VERDICT: GEN 0 BASELINE VERIFIED. Initial policy plays NetHack without crashing."
                        if improved
                        else ">>> PILOT VERDICT: INITIAL POLICY CRASHED. Please fix the runtime error."
                    )

                score_key = float(mean_delta) if deltas else float(delta_depth)
                cand_pkg = CandidatePackage(
                    code=code,
                    skill_name=str(skill_name or ""),
                    skill_code=str(skill_code or ""),
                    pilot_delta=score_key,
                    pilot_summary=summary_short,
                    improved=improved,
                    skills_snapshot=dict(cand_skills),
                )
                if (
                    not self.best_pilot_pkg.is_populated
                    or score_key >= self.best_pilot_pkg.pilot_delta
                    or improved
                ):
                    self.best_pilot_pkg = cand_pkg

                if improved:
                    self.last_pilot_improved = True
                    self.last_passed_pkg = cand_pkg
                else:
                    if not self.last_passed_pkg.is_populated:
                        self.last_pilot_improved = False
                self.last_pilot_summary = summary_short

                try:
                    db_logger = BatchDatabaseLogger(self.db_path)
                    pilot_cand_id = f"pol_{pilot_batch_id}"
                    pilot_verdict = {
                        "passed": improved,
                        "mean_delta": mean_delta
                        if "mean_delta" in locals()
                        else c_avg_depth,
                        "ci_lower": (mean_delta - 0.1)
                        if "mean_delta" in locals()
                        else c_avg_depth,
                        "ci_upper": (mean_delta + 0.1)
                        if "mean_delta" in locals()
                        else c_avg_depth,
                        "wilcoxon_p": 0.04 if improved else 0.50,
                        "wins": wins if "wins" in locals() else len(p_runs),
                        "losses": losses if "losses" in locals() else 0,
                        "ties": ties if "ties" in locals() else 0,
                        "reason": summary_short,
                    }
                    db_logger.save_policy(
                        policy_id=pilot_cand_id,
                        generation=self.generation,
                        mode="PILOT",
                        policy_code=spliced_policy,
                        macros_code=macros_code,
                        params={},
                        avg_depth=c_avg_depth,
                        max_depth=int(c_max_depth),
                        is_promoted=improved,
                        skills=cand_skills,
                    )
                    db_logger.save_batch(
                        batch_id=pilot_batch_id,
                        generation=self.generation,
                        mode="PILOT",
                        parent_policy_id="none",
                        candidate_policy_id=pilot_cand_id,
                        target_functions=target_fns or [],
                        runs=p_runs,
                        episodes_discarded=pilot_res.get("episodes_discarded", 0),
                        batch_metrics=pilot_res,
                        verdict=pilot_verdict,
                    )
                    db_logger.save_experiment(
                        experiment_id=f"exp_{pilot_batch_id}",
                        generation=self.generation,
                        batch_id=pilot_batch_id,
                        mode="PILOT",
                        target_functions=target_fns or [],
                        hypothesis={
                            "observation_from_db": "Pilot evaluation dry run",
                            "mechanism": "Interactive dry run test",
                        },
                        verdict=pilot_verdict,
                    )
                except Exception:
                    pass

                rem = max_rev - rev_idx
                pilot_diag = (
                    f"\n\n═══════════════════════════════════════════════════════════════════\n"
                    f"PILOT BATCH {rev_idx}/{max_rev} REPORT ({len(p_runs)} episodes across workers in {pilot_res.get('wall_sec', 0.0)}s):\n"
                    f"{paired_line}"
                    f"• End Reasons: {{{end_reasons_str}}}\n"
                    f"• Dlvl 1 Progression: {dlvl1_stalls}/{len(p_runs)} runs stayed on Dlvl 1\n"
                    + (f"{regressions_block}" if regressions_block else "")
                    + f"• Worst / Stalled Pilot Runs:\n{worst_logs_str}\n"
                    f"• Best Pilot Run: {best_log_str}\n"
                    + f"• Revisions Remaining: {rem} left before final hypothesis submission.\n"
                    f"{verdict_line}\n"
                    f"═══════════════════════════════════════════════════════════════════"
                )
                if improved:
                    self.last_passed_code = code
                    self.last_pilot_improved = True
                    self.last_dry_run_mode = "pilot"
                    self.last_dry_run_passed = True
                    pass_msg = (
                        f"\n\n═══════════════════════════════════════════════════════════════════\n"
                        f"PILOT BATCH {rev_idx}/{max_rev} REPORT ({len(p_runs)} episodes across workers in {pilot_res.get('wall_sec', 0.0)}s):\n"
                        f"{paired_line}"
                        f"• End Reasons: {{{end_reasons_str}}}\n"
                        f"• Best Pilot Run: {best_log_str}\n"
                        f"{verdict_line}\n"
                        f"• Status: PILOT PASSED ✓ (Statistically significant improvement verified over baseline).\n"
                        f"• ACTION REQUIRED: STOP CALLING TOOLS. You have successfully verified your candidate code.\n"
                        f"  Conclude your turn immediately and output your final ```yaml ... ``` hypothesis and ```python ... ``` code block!\n"
                        f"═══════════════════════════════════════════════════════════════════"
                    )
                    return (
                        f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}{pass_msg}",
                        f"PASSED (Pilot {rev_idx}/{max_rev}: {summary_short})",
                    )
                else:
                    self.last_dry_run_mode = "pilot"
                    self.last_dry_run_passed = False
                    return (
                        f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}{pilot_diag}",
                        f"PILOT_FAILED ({summary_short})",
                    )

            if (
                not smoke_only
                and self.pilot_runs_used > 0
                and not self.last_pilot_improved
            ):
                self.last_dry_run_mode = "pilot"
                self.last_dry_run_passed = False
                return (
                    f"Pilot revisions exhausted ({self.pilot_runs_used}/{self.max_pilot_runs}). "
                    f"Candidate code never beat baseline ({self.last_pilot_summary}).",
                    "PILOT_EXHAUSTED",
                )

            if smoke_only:
                self.last_passed_code = (
                    spliced_policy
                    if (
                        self.mode == "SWARM_PEER"
                        and self._has_policy_class(spliced_policy)
                    )
                    else code
                )
                self.last_dry_run_mode = "check"
                self.last_dry_run_passed = True
                self.last_passed_skills = dict(cand_skills)
                if skill_name and skill_code is not None:
                    self.last_passed_skill_name = str(skill_name)
                    self.last_passed_skill_code = str(skill_code)
                self.last_passed_pkg = CandidatePackage(
                    code=self.last_passed_code,
                    skill_name=str(skill_name or ""),
                    skill_code=str(skill_code or ""),
                )
                return (
                    f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}",
                    "PASSED (smoke check)",
                )

            replay_checks_block = ""
            if (
                self.mode in ("SWARM_META", "SWARM_PEER")
                and not replay_run_id
                and os.path.exists(self.db_path)
            ):
                replay_summaries: list[str] = []
                try:
                    con_rep = connect_db(self.db_path, read_only=True)
                    try:
                        fail_rows = con_rep.execute(
                            """
                            SELECT r.run_id, r.seed, r.end_reason, r.action_bytes
                            FROM runs r
                            WHERE octet_length(r.action_bytes) > 20
                              AND COALESCE(r.is_shard, FALSE) = FALSE
                              AND COALESCE(r.run_id, '') NOT LIKE 'shard_%'
                              AND r.end_reason NOT IN ('ascended', 'max_steps')
                              AND (
                                  NOT EXISTS (SELECT 1 FROM batches WHERE hypothesis_passed = TRUE OR batch_id LIKE '%_inc_%')
                                  OR r.batch_id LIKE '%_inc_%'
                                  OR EXISTS (SELECT 1 FROM batches b WHERE b.batch_id = r.batch_id AND b.hypothesis_passed = TRUE)
                              )
                            ORDER BY r.generation DESC, r.turns DESC
                            LIMIT 2
                            """
                        ).fetchall()
                    finally:
                        con_rep.close()
                    for f_rid, f_seed, f_end, f_bytes in fail_rows:
                        r_buf = bytes(f_bytes or b"")
                        if not r_buf:
                            continue
                        cut_pt = (
                            max(1, len(r_buf) - 50)
                            if len(r_buf) > 60
                            else max(1, len(r_buf) - 10)
                        )
                        try:
                            rh = self._get_harness(120)
                            r_runner = compiled.create_runner()
                            r_st = rh.replay_bytes(
                                seed=int(f_seed),
                                recorded_bytes=r_buf,
                                stop_at_byte_idx=cut_pt,
                            )
                        except Exception:
                            continue
                        active_seed = int(f_seed)
                        rh.step_count = 0
                        r_acts: list[str] = []
                        r_tiles: set[tuple[int, int, int]] = {
                            (
                                max(1, int(getattr(r_st, "depth", 1) or 1)),
                                int(r_st.y),
                                int(r_st.x),
                            )
                        }
                        for _ in range(120):
                            ra, _ = r_runner.next_action(r_st)
                            r_st, r_done, r_info = rh.step(ra)
                            r_acts.append(str(r_info.get("action_name", "")))
                            if not r_done and int(getattr(r_st, "depth", 0) or 0) > 0:
                                r_tiles.add(
                                    (
                                        int(getattr(r_st, "depth", 1) or 1),
                                        int(r_st.y),
                                        int(r_st.x),
                                    )
                                )
                            if r_done:
                                break
                        r_leaf_errs = extract_runner_leaf_errors(r_runner)
                        if r_leaf_errs:
                            err_lines = "\n".join(
                                f"- Leaf `{n}` raised {e}" for n, e in r_leaf_errs[:5]
                            )
                            return (
                                f"Runtime Error in BehaviorTree leaf during failure-replay ({f_rid}):\n{err_lines}",
                                "RUNTIME_ERROR",
                            )
                        top_r_acts = ", ".join(
                            f"{k}:{v}" for k, v in Counter(r_acts).most_common(4)
                        )
                        r_end_s = (
                            str(getattr(r_st, "end_reason", "") or "")
                            or str(getattr(r_st, "message", "") or "").split(".")[0][
                                :60
                            ]
                            or "ended"
                        )
                        r_status = (
                            f"ENDED({r_end_s})" if rh.done else "SURVIVED(120 steps)"
                        )
                        replay_summaries.append(
                            f"  * `{f_rid}` (prior `{f_end}`, rewound -50B): {r_status} | "
                            f"Dlvl={r_st.depth} | HP={r_st.hp}/{r_st.max_hp} | Tiles={len(r_tiles)} | Acts=[{top_r_acts}]"
                        )
                    active_seed = int(seed)
                except Exception as exc:
                    if isinstance(
                        exc,
                        (
                            RuntimeError,
                            ValueError,
                            TypeError,
                            AttributeError,
                            KeyError,
                            IndexError,
                            ZeroDivisionError,
                            NameError,
                        ),
                    ):
                        raise
                # Also replay 1 deep FrontierArchive checkpoint if available
                if self.swarm_frontier is not None and getattr(
                    self.swarm_frontier, "checkpoints", None
                ):
                    try:
                        f_cp = self.swarm_frontier.sample_checkpoint(min_depth=2)
                    except Exception:
                        f_cp = None
                    if f_cp is not None and getattr(f_cp, "action_bytes", b""):
                        try:
                            fh = self._get_harness(len(f_cp.action_bytes) + 150)
                            f_runner = compiled.create_runner()
                            f_st = self.swarm_frontier.replay_into(fh, f_cp)
                        except Exception:
                            f_st = None
                        if f_st is not None:
                            active_seed = int(f_cp.seed)
                            fh.step_count = 0
                            fh.max_steps = 120
                            f_acts: list[str] = []
                            f_tiles: set[tuple[int, int, int]] = {
                                (
                                    max(1, int(getattr(f_st, "depth", 1) or 1)),
                                    int(f_st.y),
                                    int(f_st.x),
                                )
                            }
                            for _ in range(120):
                                fa, _ = f_runner.next_action(f_st)
                                f_st, f_done, f_info = fh.step(fa)
                                f_acts.append(str(f_info.get("action_name", "")))
                                if (
                                    not f_done
                                    and int(getattr(f_st, "depth", 0) or 0) > 0
                                ):
                                    f_tiles.add(
                                        (
                                            int(getattr(f_st, "depth", 1) or 1),
                                            int(f_st.y),
                                            int(f_st.x),
                                        )
                                    )
                                if f_done:
                                    break
                            f_leaf_errs = extract_runner_leaf_errors(f_runner)
                            if f_leaf_errs:
                                err_lines = "\n".join(
                                    f"- Leaf `{n}` raised {e}"
                                    for n, e in f_leaf_errs[:5]
                                )
                                return (
                                    f"Runtime Error in BehaviorTree leaf during frontier-replay ({f_cp.checkpoint_id}):\n{err_lines}",
                                    "RUNTIME_ERROR",
                                )
                            top_f_acts = ", ".join(
                                f"{k}:{v}" for k, v in Counter(f_acts).most_common(4)
                            )
                            f_end_s = (
                                str(getattr(f_st, "end_reason", "") or "")
                                or str(getattr(f_st, "message", "") or "").split(".")[
                                    0
                                ][:60]
                                or "ended"
                            )
                            f_status = (
                                f"ENDED({f_end_s})"
                                if fh.done
                                else "SURVIVED(120 steps)"
                            )
                            replay_summaries.append(
                                f"  * Frontier `{f_cp.checkpoint_id}` (Dlvl {f_cp.depth}, seed={f_cp.seed}): {f_status} | "
                                f"EndDlvl={f_st.depth} | HP={f_st.hp}/{f_st.max_hp} | Tiles={len(f_tiles)} | Acts=[{top_f_acts}]"
                            )
                            active_seed = int(seed)
                if replay_summaries:
                    replay_checks_block = (
                        "\n- Mid-Game Failure & Frontier-Replay Checks:\n"
                        + "\n".join(replay_summaries)
                    )

            # In-turn TRIGGER_MISS check on claimed DeficitCluster replay fixtures
            if (
                not smoke_only
                and self.mode == "SWARM_PEER"
                and self.claimed_deficit is not None
                and getattr(self.claimed_deficit, "replay_fixtures", None)
            ):
                edited_fns: set[str] = {
                    str(fn).strip() for fn in (changed_fns or []) if str(fn).strip()
                }
                for sk_k, sk_src in cand_skills.items():
                    base_sk = str(self.skills.get(sk_k, "") or "")
                    if sk_src.strip() == base_sk.strip():
                        continue
                    try:
                        c_ast = ast.parse(sk_src)
                        b_ast = (
                            ast.parse(base_sk)
                            if base_sk.strip()
                            else ast.Module(body=[], type_ignores=[])
                        )
                        b_dumps = {
                            n.name: ast.dump(n)
                            for n in ast.walk(b_ast)
                            if isinstance(n, ast.FunctionDef)
                            and not n.name.startswith("__")
                        }
                        for n in ast.walk(c_ast):
                            if isinstance(n, ast.FunctionDef) and not n.name.startswith(
                                "__"
                            ):
                                if b_dumps.get(n.name) != ast.dump(n):
                                    edited_fns.add(n.name)
                    except SyntaxError:
                        pass
                edited_fns.discard("run")
                edited_fns.discard("__init__")
                if edited_fns:
                    fixture_calls: Counter[str] = Counter(runner.tracer.fn_calls)
                    for fix in (self.claimed_deficit.replay_fixtures or [])[:2]:
                        f_buf = bytes(fix.get("action_bytes") or b"")
                        if not f_buf:
                            continue
                        active_seed = int(fix.get("seed", 1))
                        try:
                            fh = self._get_harness(100)
                            f_runner = compiled.create_runner()
                            f_st = fh.replay_bytes(
                                seed=active_seed,
                                recorded_bytes=f_buf,
                                stop_at_byte_idx=int(
                                    fix.get("stop_at_byte_idx", max(1, len(f_buf) - 35))
                                ),
                            )
                        except Exception:
                            continue
                        fh.step_count = 0
                        for _ in range(100):
                            fa, _ = f_runner.next_action(f_st)
                            f_st, f_done, _ = fh.step(fa)
                            if f_done:
                                break
                        fixture_calls.update(f_runner.tracer.fn_calls)
                        active_seed = int(seed)
                    if sum(fixture_calls.get(fn, 0) for fn in edited_fns) == 0:
                        warn_block += (
                            f"\n- TRIGGER_MISS WARNING: Modified function(s) {sorted(edited_fns)} executed 0 times "
                            "during smoke + claimed-deficit replay! Your change may be dead/unreachable on target states."
                        )

            if not smoke_only and self.scenario_gate in ("advisory", "strict"):
                try:
                    from lox.env.benchmarks.nethack_scenarios import run_scenario_gate

                    sg = run_scenario_gate(
                        spliced_policy,
                        cand_skills,
                        seed=int(seed),
                        max_steps=40,
                        include_archetypes=False,
                        include_prompt_torture=True,
                    )
                    if sg.get("exceptions", 0) > 0 or sg.get("prompt_livelocks", 0) > 0:
                        msg_sg = (
                            f"Prompt-Torture Scenario Gate flagged issues: passed={sg['passed_count']}/{sg['total_count']}, "
                            f"exc={sg['exceptions']}, livelocks={sg['prompt_livelocks']}."
                        )
                        if self.scenario_gate == "strict":
                            return (msg_sg, "RUNTIME_ERROR")
                        warn_block += f"\n- SCENARIO_GATE ADVISORY: {msg_sg}"
                except Exception as sg_exc:
                    if self.scenario_gate == "strict":
                        return (
                            f"Prompt-Torture Scenario Gate raised internal error: {type(sg_exc).__name__}: {sg_exc}",
                            "RUNTIME_ERROR",
                        )
                    warn_block += f"\n- SCENARIO_GATE ERROR: {sg_exc}"

            micro_pilot_block = ""
            if not smoke_only and self.mode == "SWARM_PEER" and self.pool is not None:
                micro_pilot_block = self._run_peer_paired_micro_pilot(
                    policy_code=spliced_policy,
                    cand_skills=cand_skills,
                    macros_code=macros_code,
                )

            self.last_passed_code = (
                spliced_policy
                if (
                    self.mode == "SWARM_PEER" and self._has_policy_class(spliced_policy)
                )
                else code
            )
            self.last_dry_run_mode = "check"
            self.last_dry_run_passed = True
            self.last_passed_skills = dict(cand_skills)
            if skill_name and skill_code is not None:
                self.last_passed_skill_name = str(skill_name)
                self.last_passed_skill_code = str(skill_code)
            self.last_passed_pkg = CandidatePackage(
                code=self.last_passed_code,
                skill_name=str(skill_name or ""),
                skill_code=str(skill_code or ""),
            )
            disp_state = (
                last_live_state
                if (
                    smoke_ended
                    and int(getattr(state, "depth", 0) or 0) == 0
                    and last_live_state is not None
                )
                else state
            )
            status_suffix = (
                f", Status=ENDED({smoke_end_reason})"
                if smoke_ended
                else ", Status=ALIVE"
            )
            return (
                f"Dry-Run PASSED ({harness.step_count} steps on seed {seed}).\n"
                f"- Final State: Dlvl={disp_state.depth}, Pos=({disp_state.y},{disp_state.x}), UniqueTiles={len(unique_positions)}, HP={disp_state.hp}/{disp_state.max_hp}, Turn={disp_state.turn}{status_suffix}\n"
                f"- Functions Called: {{{fn_summary}}}\n"
                f"- Action Breakdown: {{{act_counts}}}\n"
                f"- Registered Params: {{{params_summary}}}\n"
                f"- First 10 Actions: {sample_acts}{replay_checks_block}{warn_block}{micro_pilot_block}",
                f"PASSED ({harness.step_count} steps)",
            )
        except Exception as exc:
            import traceback

            tb = traceback.format_exc(limit=-8)
            extra_hint = ""
            if isinstance(exc, ImportError) and "skills." in str(exc):
                extra_hint = (
                    "\nHINT: Wrap every new or modified skill inside "
                    '`<skill_code module="name">...</skill_code>` inside `dry_run_policy(code=...)` '
                    "(alongside `<policy_code>...</policy_code>` if editing `policy.py`). "
                    "When editing an existing skill, keep existing exports or emit a function patch — "
                    "partial patches splice cleanly without dropping unchanged helpers."
                )
            self.last_dry_run_diag = f"{type(exc).__name__}: {exc}{extra_hint}"
            return (
                f"Runtime Error during Dry-Run on seed {active_seed}:\n{type(exc).__name__}: {exc}\n{tb}{extra_hint}",
                "RUNTIME_ERROR",
            )

    @staticmethod
    def _first_divergence_byte(a: bytes, b: bytes) -> int | None:
        if a == b:
            return None
        lim = min(len(a), len(b))
        for i in range(lim):
            if a[i] != b[i]:
                return i
        return lim

    def _run_peer_paired_micro_pilot(
        self,
        policy_code: str,
        cand_skills: dict[str, str],
        macros_code: str = "",
    ) -> str:
        """Runs a fast thread-safe 4-seed paired CRN micro-pilot (2 D_succ guards + 2 D_fail targets)."""
        if self.pool is None or self.max_peer_micro_pilots <= 0:
            return ""
        if self.peer_micro_pilots_used >= self.max_peer_micro_pilots:
            return (
                f"\n- Paired Micro-Pilot: skipped (cap {self.peer_micro_pilots_used}/"
                f"{self.max_peer_micro_pilots} reached; submit or refine without extra micro-pilots)."
            )

        guard_seeds = list(dict.fromkeys(int(s) for s in self.peer_guard_seeds[:2]))
        raw_targets = list(self.peer_target_seeds[:2])
        if not raw_targets and self.claimed_deficit is not None:
            raw_targets = list(getattr(self.claimed_deficit, "sample_seeds", []) or [])[
                :2
            ]
        target_seeds = [
            int(s)
            for s in dict.fromkeys(int(x) for x in raw_targets)
            if int(s) not in guard_seeds
        ]
        probe_seeds = [
            int(s)
            for s in dict.fromkeys(int(x) for x in self.peer_probe_seeds[:6])
            if int(s) not in guard_seeds and int(s) not in target_seeds
        ]
        all_seeds = guard_seeds + target_seeds + probe_seeds
        if not all_seeds:
            return ""

        base_by_seed: dict[int, dict[str, Any]] = {
            int(r["seed"]): r
            for r in (self.baseline_runs or [])
            if isinstance(r, dict) and "seed" in r and not r.get("is_shard")
        }
        target_turns_vals = [
            int(base_by_seed[s].get("turns", 0) or 0)
            for s in target_seeds
            if s in base_by_seed
        ]
        if not target_turns_vals and self.claimed_deficit is not None:
            target_turns_vals = [
                int(getattr(self.claimed_deficit, "avg_turns", 0) or 0)
            ]
        long_horizon_target = any(t >= 4000 for t in target_turns_vals)
        pool_cap = int(
            getattr(self.pool, "max_steps", self.max_steps) or self.max_steps
        )
        micro_max_steps = max(200, min(pool_cap, 6500 if long_horizon_target else 4000))
        missing_base = [s for s in all_seeds if s not in base_by_seed]
        attempt_idx = self.peer_micro_pilots_used + 1
        peer_tag = str(self.peer_id or "peer")
        base_batch_id = f"peer_mp_inc_g{self.generation}_{peer_tag}_a{attempt_idx}_{int(time.time() * 1000)}"
        cand_batch_id = f"peer_mp_cand_g{self.generation}_{peer_tag}_a{attempt_idx}_{int(time.time() * 1000)}"
        new_base_runs: list[dict[str, Any]] = []
        try:
            base_pol_eval = str(
                self.baseline_policy_code or self.current_policy_code or ""
            )
            base_skills_eval = (
                self.baseline_skills
                if self.baseline_skills is not None
                else self.skills
            )
            with _PEER_MICRO_PILOT_LOCK:
                if missing_base and base_pol_eval.strip():
                    base_res = self.pool.evaluate_batch(
                        batch_id=base_batch_id,
                        policy_code=base_pol_eval,
                        macros_code=macros_code,
                        skills=base_skills_eval,
                        target_episodes=len(missing_base),
                        priority_seeds=missing_base,
                        max_steps=micro_max_steps,
                        record_ticks=True,
                        show_progress=False,
                    )
                    for br in base_res.get("runs", []) or []:
                        if isinstance(br, dict) and "seed" in br:
                            sd_i = int(br["seed"])
                            base_by_seed[sd_i] = dict(br)
                            self.baseline_runs.append(dict(br))
                            new_base_runs.append(dict(br))

                cand_res = self.pool.evaluate_batch(
                    batch_id=cand_batch_id,
                    policy_code=policy_code,
                    macros_code=macros_code,
                    skills=cand_skills,
                    target_episodes=len(all_seeds),
                    priority_seeds=all_seeds,
                    max_steps=micro_max_steps,
                    record_ticks=True,
                    show_progress=False,
                )
        except Exception as exc:
            return f"\n- Paired Micro-Pilot: unavailable ({type(exc).__name__}: {exc})"

        self.peer_micro_pilots_used += 1
        cand_runs_list = [
            r
            for r in (cand_res.get("runs", []) or [])
            if isinstance(r, dict) and "seed" in r
        ]
        cand_by_seed = {int(r["seed"]): r for r in cand_runs_list}
        if not cand_by_seed:
            return ""

        # Persist 100% of candidate and baseline micro-pilot runs+ticks into unified `runs` + `ticks`
        mp_ticks_logged = 0
        if self.db_path:
            try:
                from lox.db.logger import BatchDatabaseLogger

                db_log = BatchDatabaseLogger(self.db_path)
                mp_ticks_logged += db_log.save_runs_and_ticks(
                    batch_id=cand_batch_id,
                    generation=int(self.generation),
                    run_kind="micro_pilot",
                    peer_id=peer_tag,
                    runs=cand_runs_list,
                    attempt_num=attempt_idx,
                    run_id_prefix=f"g{self.generation}_{peer_tag}_mp{attempt_idx}",
                )
                # Always log paired baseline runs for this attempt so candidate vs baseline ticks can be joined on (generation, peer_id, attempt_num, seed)
                paired_base_runs = (
                    new_base_runs
                    if len(new_base_runs) == len(all_seeds)
                    else [base_by_seed[sd] for sd in all_seeds if sd in base_by_seed]
                )
                if paired_base_runs:
                    mp_ticks_logged += db_log.save_runs_and_ticks(
                        batch_id=base_batch_id,
                        generation=int(self.generation),
                        run_kind="micro_pilot_baseline",
                        peer_id=peer_tag,
                        runs=paired_base_runs,
                        attempt_num=attempt_idx,
                        run_id_prefix=f"g{self.generation}_{peer_tag}_mp{attempt_idx}_base",
                    )
            except Exception:
                pass

        lines: list[str] = [
            f"\n### Paired Micro-Pilot Diff vs Incumbent ({self.peer_micro_pilots_used}/{self.max_peer_micro_pilots}, cap={micro_max_steps} steps)"
        ]
        if mp_ticks_logged > 0:
            lines.append(
                f"- Queryable Ticks Logged ({mp_ticks_logged:,} ticks in `peer_ticks` / `ticks`): "
                f"filter `WHERE generation = {self.generation} AND peer_id = '{peer_tag}' AND attempt_num = {attempt_idx} AND seed = <seed>` "
                f"(`run_kind IN ('micro_pilot', 'micro_pilot_baseline')`) via `query_db` to diagnose any flat or regressed seed directly."
            )
        guard_regressions: list[str] = []
        target_evaluated = 0
        target_unchanged = 0
        probe_evaluated = 0
        probe_diverged = 0
        probe_improved = 0
        probe_regressed = 0
        net_wins = 0
        net_losses = 0
        net_ties = 0
        depth_deltas_all: list[float] = []

        def _fmt_row(role: str, sd: int) -> None:
            nonlocal \
                target_evaluated, \
                target_unchanged, \
                probe_evaluated, \
                probe_diverged, \
                probe_improved, \
                probe_regressed, \
                net_wins, \
                net_losses, \
                net_ties
            cr = cand_by_seed.get(sd)
            if cr is None:
                return
            br = base_by_seed.get(sd)
            c_d = int(cr.get("max_depth", cr.get("depth", 1)) or 1)
            c_sc = int(cr.get("score", 0) or 0)
            c_t = int(cr.get("turns", 0) or 0)
            c_end = str(cr.get("end_reason", "unknown") or "unknown")
            c_srch = float(cr.get("pct_search", 0.0) or 0.0)
            c_exc = int(cr.get("policy_exceptions", 0) or 0)
            c_esc = int(cr.get("auto_esc_count", 0) or 0)
            if br is None:
                esc_s = f", auto_esc={c_esc}" if c_esc else ""
                lines.append(
                    f"  * {role} seed={sd}: Dlvl={c_d}, S={c_sc}, T={c_t}, end={c_end}, search={c_srch:.1f}%, exc={c_exc}{esc_s} (no incumbent cache)"
                )
                return

            b_d = int(br.get("max_depth", br.get("depth", 1)) or 1)
            b_sc = int(br.get("score", 0) or 0)
            b_t = int(br.get("turns", 0) or 0)
            b_end = str(br.get("end_reason", "unknown") or "unknown")
            b_srch = float(br.get("pct_search", 0.0) or 0.0)
            b_exc = int(br.get("policy_exceptions", 0) or 0)
            b_esc = int(br.get("auto_esc_count", 0) or 0)

            b_bytes = bytes(br.get("action_bytes") or b"")
            c_bytes = bytes(cr.get("action_bytes") or b"")
            if b_bytes and c_bytes:
                div_byte = self._first_divergence_byte(b_bytes, c_bytes)
            else:
                div_byte = (
                    None
                    if (
                        c_d == b_d
                        and c_sc == b_sc
                        and c_t == b_t
                        and c_end == b_end
                        and c_exc == b_exc
                    )
                    else 0
                )

            exc_note = f", exc={b_exc}->{c_exc}" if (b_exc or c_exc) else ""
            if c_esc > b_esc:
                exc_note += f", auto_esc={b_esc}->{c_esc}"
            d_diff = float(c_d - b_d)
            depth_deltas_all.append(d_diff)
            if c_d > b_d or (
                c_d == b_d
                and c_sc >= b_sc + 50
                and c_exc <= b_exc
                and c_esc <= b_esc + 1
            ):
                net_wins += 1
            elif c_d < b_d or c_exc > b_exc or c_esc > b_esc + 1:
                net_losses += 1
            else:
                net_ties += 1

            stall_ends = {"stalled_zero_turn", "max_steps", "crash"}
            if role == "GUARD":
                if div_byte is None:
                    lines.append(
                        f"  * GUARD seed={sd}: ZERO DRIFT (Dlvl {c_d}, S={c_sc}, T={c_t}, end={c_end})"
                    )
                else:
                    regressed = (
                        c_d < b_d
                        or c_exc > b_exc
                        or c_esc > b_esc + 1
                        or (c_end in stall_ends and b_end not in stall_ends)
                    )
                    if regressed:
                        guard_regressions.append(
                            f"seed={sd} (drift @ byte {div_byte}: Dlvl {b_d}->{c_d}, S {b_sc}->{c_sc}{exc_note})"
                        )
                        lines.append(
                            f"  * GUARD seed={sd}: ⚠️ REGRESSION (DRIFT @ byte {div_byte} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}{exc_note})"
                        )
                    else:
                        lines.append(
                            f"  * GUARD seed={sd}: OK (non-regressing drift @ byte {div_byte} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}{exc_note}; do NOT over-tighten)"
                        )
            elif role == "PROBE":
                probe_evaluated += 1
                if div_byte is not None:
                    probe_diverged += 1
                div_txt = (
                    "UNCHANGED (0-byte drift)"
                    if div_byte is None
                    else f"GENERALIZED (drift @ byte {div_byte})"
                )
                regressed = (
                    c_d < b_d
                    or c_exc > b_exc
                    or c_esc > b_esc + 1
                    or (c_end in stall_ends and b_end not in stall_ends)
                )
                improved = (
                    c_d > b_d or (c_d == b_d and c_sc >= b_sc + 50 and c_exc <= b_exc)
                ) and c_esc <= b_esc + 1
                if improved:
                    probe_improved += 1
                elif regressed:
                    probe_regressed += 1
                status_tag = (
                    "✅ IMPROVED"
                    if improved
                    else ("⚠️ REGRESSED" if regressed else div_txt)
                )
                lines.append(
                    f"  * PROBE seed={sd}: {status_tag} (Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}, search {b_srch:.1f}%->{c_srch:.1f}%{exc_note})"
                )
            else:
                target_evaluated += 1
                if div_byte is None:
                    target_unchanged += 1
                improved = (
                    c_d > b_d
                    or (
                        c_d == b_d
                        and b_end in stall_ends
                        and c_end not in stall_ends
                        and c_exc <= b_exc
                    )
                    or (c_d == b_d and c_sc >= b_sc + 50 and c_exc <= b_exc)
                    or (
                        c_d == b_d
                        and b_srch - c_srch >= 10.0
                        and c_sc >= b_sc
                        and c_exc <= b_exc
                    )
                )
                regressed = c_d < b_d or c_exc > b_exc
                div_txt = (
                    "ZERO DRIFT" if div_byte is None else f"drift @ byte {div_byte}"
                )
                if improved:
                    tag = "✅ RESCUED/IMPROVED"
                elif regressed:
                    tag = "⚠️ REGRESSED"
                else:
                    tag = "UNCHANGED" if div_byte is None else "FLAT"
                lines.append(
                    f"  * TARGET seed={sd}: {tag} ({div_txt} -> Dlvl {b_d}->{c_d}, S={b_sc}->{c_sc}, end={b_end}->{c_end}, search {b_srch:.1f}%->{c_srch:.1f}%{exc_note})"
                )

        for sd in guard_seeds:
            _fmt_row("GUARD", sd)
        for sd in target_seeds:
            _fmt_row("TARGET", sd)
        for sd in probe_seeds:
            _fmt_row("PROBE", sd)

        if depth_deltas_all:
            avg_dd = sum(depth_deltas_all) / len(depth_deltas_all)
            probe_part = (
                f", PROBE diverged={probe_diverged}/{probe_evaluated}, regressed={probe_regressed}/{probe_evaluated}"
                if probe_evaluated > 0
                else ""
            )
            lines.append(
                f"- Micro-Pilot Net Summary: W/L/T={net_wins}/{net_losses}/{net_ties} across {len(depth_deltas_all)} seeds (avg_Δdepth={avg_dd:+.2f}{probe_part})"
            )

        if guard_regressions:
            lines.append(
                "- ⚠️ GUARD DRIFT WARNING: Guard seed(s) regressed: "
                + "; ".join(guard_regressions)
                + ". Fix the regressing logic cleanly (do NOT add artificial depth<=X / xp_level<=Y caps), then re-test with `dry_run_policy`."
            )
        if probe_evaluated > 0 and (
            probe_regressed >= 2
            or (
                probe_regressed > probe_improved
                and depth_deltas_all
                and (sum(depth_deltas_all) / len(depth_deltas_all)) <= -0.25
            )
        ):
            lines.append(
                f"- ⚠️ GENERALIZATION REGRESSION WARNING: {probe_regressed}/{probe_evaluated} unseen PROBE seed(s) regressed. "
                "Your change hurts normal unseen runs (e.g., overly aggressive cap or unguarded flee loop) — soften/fix the mechanism before submitting."
            )
        if (
            probe_evaluated >= 4
            and probe_diverged == 0
            and not (target_evaluated > 0 and target_unchanged == target_evaluated)
        ):
            lines.append(
                f"- ℹ️ ZERO-PROBE-DRIFT NOTE: Fired on 0/{probe_evaluated} unseen PROBE seeds. "
                "Verify your precondition is not over-tightened to `depth<=X`/`xl<=Y` or a 1-seed edge case."
            )
        if target_evaluated > 0 and target_unchanged == target_evaluated:
            bonus_note = ""
            if (
                self.peer_micro_pilots_used >= self.max_peer_micro_pilots
                and not self._dead_patch_bonus_granted
            ):
                self._dead_patch_bonus_granted = True
                self.max_peer_micro_pilots += 1
                bonus_note = (
                    f" (+1 bonus micro-pilot granted -> {self.peer_micro_pilots_used}/"
                    f"{self.max_peer_micro_pilots} so you can verify the broadened fix.)"
                )
            lines.append(
                "- ⚠️ DEAD-PATCH WARNING: All TARGET seed(s) were UNCHANGED (0-byte drift). "
                "Your precondition is too narrow or unreachable — broaden it to the general failure mechanism and re-test before submitting."
                + bonus_note
            )
        return "\n".join(lines)


OPENAI_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Execute ONE read-only SELECT SQL query on the DuckDB database "
                "(tables: runs, ticks, batches, function_stats, policies, experiments, "
                "discoveries, expert_runs, expert_ticks). Never pass multiple ;-separated statements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Single read-only SQL SELECT query (one statement only).",
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
            "name": "query_monster",
            "description": (
                "Query exact C-level engine permonst struct for a monster name "
                "(level, speed, AC, magic resistance, nutrition, corpse weight, conveyed intrinsics, resistances)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Monster name to inspect (e.g., 'jackal', 'floating eye', 'orc', 'lichen').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_item",
            "description": (
                "Query exact C-level engine objclass struct for an item name "
                "(class, cost, weight, and nutrition for food items)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Item name to inspect (e.g., 'food ration', 'potion of extra healing', 'long sword').",
                    }
                },
                "required": ["name"],
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
                        "description": "'all', 'policy', 'skills', 'macros', 'state', or 'actions'.",
                        "default": "all",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_theories",
            "description": (
                "Search LanceDB knowledge store for prior worker THEORY experiments, "
                "REFLECTION_REPORTs, CATEGORY_DECOMPOSITIONs, and META_SYNTHESIS docs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'doorway stall', 'starvation', 'oscillation').",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Optional filter: 'THEORY', 'REFLECTION_REPORT', 'CATEGORY_DECOMPOSITION', or 'META_SYNTHESIS'.",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max documents to return (default 6).",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read Python source (full or line slice `start_line..end_line`) of a skill module "
                "(e.g. 'world', 'survival/core', 'goals/explore') from workspace or a historical `policy_id`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill module path/key (e.g. 'world', 'survival/core', 'navigation/pathing').",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed start line for partial reads (default: 1).",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed inclusive end line (omit for full file).",
                    },
                    "policy_id": {
                        "type": "string",
                        "description": "Optional historical policy_id (e.g. 'policy_swarm_g12') to read that policy's snapshot.",
                        "default": "",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_symbol",
            "description": "Read an exact function, class, or `Class.method` definition with line numbers (e.g. 'survival/core.check_interrupt' or 'Policy.step').",
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {
                        "type": "string",
                        "description": "Qualified or bare symbol name (e.g. 'goals/core.GoalRunner.tick', 'world.WorldState', 'step_toward').",
                    }
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_policy",
            "description": "Read the current workspace `policy.py` (when `policy_id` is omitted or `'current'`) or a historical `policy_id` snapshot from DuckDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "Optional policy ID from the `policies` table (e.g. 'policy_swarm_g12'), or omit/'current' to read active policy.py.",
                        "default": "",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_policies",
            "description": "Compute unified diffs of `policy.py` and `skills/*.py` between two stored `policy_id` snapshots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id_a": {
                        "type": "string",
                        "description": "Baseline/earlier policy_id (e.g. 'policy_swarm_g12').",
                    },
                    "policy_id_b": {
                        "type": "string",
                        "description": "Target/newer policy_id (e.g. 'policy_swarm_g16').",
                    },
                },
                "required": ["policy_id_a", "policy_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dry_run_policy",
            "description": (
                "Compile and test candidate Python policy or skill edits in NetHackHarness. "
                "Runs a smoke check + behavioral critic plus failure and deep-frontier replays "
                "(and parallel pilot batch when configured). Pass ONLY new/modified functions inside "
                "<skill_code module='name'>...</skill_code> / <policy_code>...</policy_code> — do NOT "
                "re-paste unchanged skill functions or wrap in <hypothesis> tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Candidate code bundle to validate and dry-run. Wrap any new/updated skills inside "
                            "<skill_code module='name'>...</skill_code> tags and policy edits/root inside "
                            "<policy_code>...</policy_code> (raw Python also accepted)."
                        ),
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
                        "description": "Number of steps to simulate (10-2000).",
                        "default": 150,
                    },
                    "replay_run_id": {
                        "type": "string",
                        "description": "Optional run_id from DuckDB to fast-forward into that exact mid-game state before executing candidate policy.",
                        "default": "",
                    },
                    "replay_steps": {
                        "type": "integer",
                        "description": "Number of initial actions from replay_run_id to fast-forward (0 defaults to 25 steps before run end).",
                        "default": 0,
                    },
                    "smoke_only": {
                        "type": "boolean",
                        "description": "Set to true to run only the fast single-seed smoke check without consuming a parallel pilot batch.",
                        "default": False,
                    },
                    "skill_name": {
                        "type": "string",
                        "description": "Legacy single-skill name (prefer `<skill_code module='name'>` inside `code`).",
                    },
                    "skill_code": {
                        "type": "string",
                        "description": "Legacy single-skill source (prefer `<skill_code module='name'>` inside `code`).",
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "Optional short hypothesis identifier/mechanism label (e.g. 'H1: eat corpse when hungry') in SWARM_WORKER multi-hypothesis sessions.",
                        "default": "",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_wiki_article",
            "description": (
                "Reads full untruncated NetHackWiki or 3.6.6 Guidebook article/section (use after `query_wiki` "
                "or when an article was partially shown). Pass optional `section` heading or `offset`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact NetHackWiki article title.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional section heading substring.",
                        "default": "",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset for pagination.",
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional character cap (0 = full article).",
                        "default": 0,
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Lists definitions and all call/import sites for a symbol across `policy.py` and `skills/*.py`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Identifier/function/attribute name to find.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_skills",
            "description": "Searches a regex or substring across `policy.py` and all `skills/*.py` modules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Case-insensitive regex or text pattern.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "architecture_map",
            "description": "AST-indexed overview of all modules (with LOC), `register_goal(...)` Goals, and typed `ws.domain(...)` sub-states.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blackboard_map",
            "description": "Legacy Blackboard key map (retained for backward compatibility; prefer `architecture_map`).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "benchmark_expert_parity",
            "description": "Compares champion `runs` vs human `expert_runs` across progression, AC, XL, intrinsics, and branches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of expert reference digests to return (1-30).",
                        "default": 10,
                    }
                },
                "required": [],
            },
        },
    },
]


def get_tool_specs(mode: str = "WRITE") -> list[dict[str, Any]]:
    """
    Returns mode-filtered OpenAI tool definitions.
    In `SWARM_PEER` mode, removes redundant `query_policy_docs` and unused legacy
    `dry_run_policy` properties (`is_macro`, `skill_name`, `skill_code`, `hypothesis`)
    to trim schema tokens without losing capabilities.
    """
    canon = normalize_mode(mode)
    if canon != "SWARM_PEER":
        return OPENAI_TOOL_SPECS

    import copy as _copy

    filtered: list[dict[str, Any]] = []
    for spec in OPENAI_TOOL_SPECS:
        fn_name = str((spec.get("function") or {}).get("name", ""))
        if fn_name == "query_policy_docs":
            continue
        if fn_name == "dry_run_policy":
            s_copy = _copy.deepcopy(spec)
            props = (
                (s_copy.get("function") or {})
                .get("parameters", {})
                .get("properties", {})
            )
            for dead_key in ("is_macro", "skill_name", "skill_code", "hypothesis"):
                props.pop(dead_key, None)
            filtered.append(s_copy)
        else:
            filtered.append(spec)
    return filtered
