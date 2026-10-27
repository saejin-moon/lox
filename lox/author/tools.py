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
import time
from collections import Counter
from typing import Any

import numpy as np

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
    max_chars = 4500 if len(rows) <= 5 else (1800 if len(rows) <= 15 else 700)
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
                    s = s[: max_chars // 2] + " ... " + s[-max_chars // 2 :]
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
        self.pool: Any = None
        self.baseline_runs: list[dict[str, Any]] = []
        self.pilot_runs_used: int = 0
        self.max_pilot_runs: int = 5
        self.pilot_episodes: int = 20
        self.last_pilot_improved: bool = False
        self.last_pilot_summary: str = ""

    def configure_generation(
        self,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        max_pilot_runs: int = 5,
        pilot_episodes: int = 20,
    ) -> None:
        """Sets worker pool and baseline runs context for pilot evaluations."""
        self.pool = pool
        self.baseline_runs = list(baseline_runs or [])
        self.pilot_runs_used = 0
        self.max_pilot_runs = max_pilot_runs
        self.pilot_episodes = pilot_episodes
        self.last_pilot_improved = False
        self.last_pilot_summary = ""

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
        max_turns: int = 150,
        replay_run_id: str = "",
        replay_steps: int = 0,
    ) -> tuple[str, str]:
        """
        Compiles and tests candidate `code` in a live NetHack sandbox:
        1. Smoke Check: AST validation, reachability, loop check, and behavioral critic (150 steps).
        2. Pilot Evaluation: If worker pool is attached, runs 20 paired episodes across workers
           (up to 5 pilot revisions per generation) to measure real baseline delta and surface failure logs.
        """
        macros_code = self.macro_lib.load_code()
        policy_code = self.current_policy_code

        try:
            if is_macro:
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
                    changed_fns = ["run"]
                compiled = compile_policy_and_macros(spliced_policy, macros_code)
        except Exception as exc:
            return (
                f"Compilation / Validation Error:\n{type(exc).__name__}: {exc}",
                "COMPILE_ERROR",
            )

        # Step 1: Single-Episode Smoke Check + Behavioral Critic (with optional prefix replay)
        harness = NetHackHarness(max_steps=max(10, min(200, int(max_turns))))
        try:
            runner = compiled.create_runner()
            actions_taken: list[str] = []
            raw_act_names: list[str] = []
            unique_positions: set[tuple[int, int]] = set()
            positions_history: list[tuple[int, int]] = []
            oscillation_detected = False

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
                    state = harness.reset(seed=r_seed)
                    cut = (
                        min(len(r_bytes), int(replay_steps))
                        if int(replay_steps) > 0
                        else max(0, len(r_bytes) - 25)
                    )
                    for b in r_bytes[:cut]:
                        state, done, _ = harness.step(b)
                        if done:
                            break
                else:
                    state = harness.reset(seed=int(seed))
            else:
                state = harness.reset(seed=int(seed))

            for _ in range(max(10, min(200, int(max_turns)))):
                raw_act, active_fn = runner.next_action(state)
                state, done, info = harness.step(raw_act)
                pos = (state.y, state.x)
                unique_positions.add(pos)
                positions_history.append(pos)
                if len(positions_history) >= 40:
                    if len(set(positions_history[-40:])) <= 3:
                        oscillation_detected = True
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

            warn_lines: list[str] = []
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
                and self.pool is not None
                and self.pilot_runs_used < self.max_pilot_runs
            ):
                self.pilot_runs_used += 1
                rev_idx = self.pilot_runs_used
                max_rev = self.max_pilot_runs
                target_eps = min(self.pilot_episodes, 30)

                pilot_seeds: list[int] = []
                if self.baseline_runs:
                    pilot_seeds = [
                        int(r["seed"]) for r in self.baseline_runs[:target_eps]
                    ]
                else:
                    pilot_seeds = list(range(int(seed), int(seed) + target_eps))

                base_by_seed = {
                    int(r["seed"]): r for r in self.baseline_runs if "seed" in r
                }

                pilot_batch_id = f"pilot_{rev_idx}_{int(time.time())}"
                target_fns = changed_fns if policy_code.strip() else None

                pilot_res = self.pool.evaluate_batch(
                    batch_id=pilot_batch_id,
                    policy_code=spliced_policy,
                    macros_code=macros_code,
                    target_episodes=target_eps,
                    target_functions=target_fns,
                    priority_seeds=pilot_seeds,
                    record_ticks=True,
                    show_progress=False,
                )

                p_runs = pilot_res.get("runs", [])
                wins = 0
                losses = 0
                ties = 0
                cand_depths = [float(r["max_depth"]) for r in p_runs]
                base_depths: list[float] = []

                for r in p_runs:
                    s = int(r["seed"])
                    if s in base_by_seed:
                        br = base_by_seed[s]
                        b_d = float(br.get("max_depth", 1))
                        b_s = float(br.get("score", 0))
                        base_depths.append(b_d)
                        c_d = float(r.get("max_depth", 1))
                        c_s = float(r.get("score", 0))
                        if c_d > b_d or (c_d == b_d and c_s > b_s):
                            wins += 1
                        elif c_d < b_d or (c_d == b_d and c_s < b_s):
                            losses += 1
                        else:
                            ties += 1

                c_avg_depth = float(np.mean(cand_depths)) if cand_depths else 0.0
                b_avg_depth = (
                    float(np.mean(base_depths))
                    if base_depths
                    else (
                        float(self.baseline_runs[0].get("depth", 1.0))
                        if self.baseline_runs
                        else 1.0
                    )
                )
                delta_depth = c_avg_depth - b_avg_depth
                c_max_depth = int(max(cand_depths)) if cand_depths else 0

                dlvl1_stalls = sum(1 for r in p_runs if r.get("max_depth", 1) <= 1)
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
                    w_glog = str(wr.get("game_log", ""))
                    if len(w_glog) > 400:
                        w_glog = "..." + w_glog[-400:]
                    worst_logs.append(
                        f"  - Stalled Run (seed={w_seed}, Dlvl={w_d}, Steps={w_steps}, End={w_end}): {w_glog}"
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

                if base_depths:
                    improved = wins > losses and delta_depth >= 0.0
                else:
                    improved = c_avg_depth > 1.0 or c_max_depth > 1

                self.last_pilot_improved = improved
                summary_short = f"{wins}W/{losses}L/{ties}T, Δ={delta_depth:+.2f}"
                self.last_pilot_summary = summary_short

                rem = max_rev - rev_idx
                pilot_diag = (
                    f"\n\n═══════════════════════════════════════════════════════════════════\n"
                    f"PILOT BATCH {rev_idx}/{max_rev} REPORT ({len(p_runs)} episodes across workers in {pilot_res.get('wall_sec', 0.0)}s):\n"
                    f"• Paired vs Baseline: {wins} Wins / {losses} Losses / {ties} Ties | Δ Depth: {delta_depth:+.2f} (Avg Depth: {c_avg_depth:.2f} vs Baseline: {b_avg_depth:.2f}, Max Depth: {c_max_depth})\n"
                    f"• End Reasons: {{{end_reasons_str}}}\n"
                    f"• Dlvl 1 Progression: {dlvl1_stalls}/{len(p_runs)} runs stayed on Dlvl 1\n"
                    f"• Worst / Stalled Pilot Runs:\n{worst_logs_str}\n"
                    f"• Best Pilot Run: {best_log_str}\n"
                    f"• Revisions Remaining: {rem} left before final hypothesis submission.\n"
                    f"{'>>> PILOT VERDICT: IMPROVEMENT DETECTED over baseline.' if improved else '>>> PILOT VERDICT: FLAT OR REGRESSED VS BASELINE. Please inspect the worst run logs above and revise your skill before finalizing!'}\n"
                    f"═══════════════════════════════════════════════════════════════════"
                )
                self.last_passed_code = code
                return (
                    f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}{pilot_diag}",
                    f"PASSED (Pilot {rev_idx}/{max_rev}: {summary_short})",
                )

            self.last_passed_code = code
            return (
                f"Dry-Run PASSED ({harness.step_count} steps on seed {seed}).\n"
                f"- Final State: Dlvl={state.depth}, Pos=({state.y},{state.x}), UniqueTiles={len(unique_positions)}, HP={state.hp}/{state.max_hp}, Turn={state.turn}\n"
                f"- Functions Called: {{{fn_summary}}}\n"
                f"- Action Breakdown: {{{act_counts}}}\n"
                f"- Registered Params: {{{params_summary}}}\n"
                f"- First 10 Actions: {sample_acts}{warn_block}",
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
            "description": (
                "Compile and test candidate Python policy or macro code. "
                "Performs a 150-step smoke check and behavioral critic analysis, plus a 20-episode "
                "parallel pilot batch on the worker pool (up to 5 pilot revisions per generation) "
                "to compare against the current baseline and surface concrete failure logs."
            ),
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
                },
                "required": ["code"],
            },
        },
    },
]
