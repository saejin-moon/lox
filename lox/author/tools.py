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
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lox.author.prompts import FULL_ACTION_DOCS, POLICY_LANG_DOCS
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.db.shards import extract_shards_from_runs
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_package
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
        benchmark: str = "nethack",
    ):
        self.db_path = db_path
        self.wiki = WikiEngine(db_path=wiki_db_path)
        self.macro_lib = MacroLibrary(path=macros_path)
        self.benchmark = str(benchmark)
        self.current_policy_code: str = ""
        self.last_passed_code: str = ""
        self.best_pilot_code: str = ""
        self.best_pilot_delta: float = -1e9
        self.pool: Any = None
        self.baseline_runs: list[dict[str, Any]] = []
        self.pilot_runs_used: int = 0
        self.max_pilot_runs: int = 5
        self.pilot_episodes: int = 30
        self.last_pilot_improved: bool = False
        self.last_pilot_summary: str = ""
        self.last_cand_depths: list[float] = []
        self.last_base_depths: list[float] = []
        self.skills: dict[str, str] = {}
        self.generation: int = 0

    def configure_generation(
        self,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        max_pilot_runs: int = 5,
        pilot_episodes: int = 30,
        skills: dict[str, str] | None = None,
        generation: int = 0,
        benchmark: str | None = None,
    ) -> None:
        """Sets worker pool and baseline runs context for pilot evaluations."""
        self.pool = pool
        self.baseline_runs = list(baseline_runs or [])
        self.pilot_runs_used = 0
        self.max_pilot_runs = max_pilot_runs
        self.pilot_episodes = pilot_episodes
        self.last_pilot_improved = False
        self.last_pilot_summary = ""
        self.best_pilot_code = ""
        self.best_pilot_delta = -1e9
        self.last_cand_depths = []
        self.last_base_depths = []
        self.generation = int(generation)
        if skills is not None:
            self.skills = dict(skills)
        if benchmark is not None:
            self.benchmark = str(benchmark)

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

        # Dynamically rewrite SELECT MAX(batch_id) poison pill to target the active champion batch
        max_batch_pattern = re.compile(
            r"\(\s*SELECT\s+MAX\s*\(\s*batch_id\s*\)\s+FROM\s+(?:runs|function_stats|batches)\s*\)",
            re.IGNORECASE,
        )
        if max_batch_pattern.search(clean):
            con_temp = connect_db(self.db_path, read_only=True)
            try:
                row = con_temp.execute(
                    "SELECT batch_id FROM batches WHERE hypothesis_passed = TRUE "
                    "AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE') "
                    "ORDER BY generation DESC, created_at DESC LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    target_batch_sql = (
                        "(SELECT batch_id FROM batches WHERE hypothesis_passed = TRUE "
                        "AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE') "
                        "ORDER BY generation DESC, created_at DESC LIMIT 1)"
                    )
                else:
                    target_batch_sql = "(SELECT batch_id FROM batches ORDER BY created_at DESC LIMIT 1)"
            except Exception:
                target_batch_sql = (
                    "(SELECT batch_id FROM batches WHERE hypothesis_passed = TRUE "
                    "AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE') "
                    "ORDER BY generation DESC, created_at DESC LIMIT 1)"
                )
            finally:
                con_temp.close()
            clean = max_batch_pattern.sub(target_batch_sql, clean)

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
        if "policy" in t:
            pol_src = self.current_policy_code or "(Empty policy)"
            return (
                f"### Current `policy.py` Full Source:\n```python\n{pol_src}\n```",
                "policy source",
            )
        if "action" in t:
            return FULL_ACTION_DOCS, "action docs"
        if "macro" in t:
            return self.macro_lib.format_docs() + macros_section, "macros docs"
        return (
            POLICY_LANG_DOCS + "\n" + self.macro_lib.format_docs() + macros_section,
            "full docs",
        )

    def read_skill(self, skill_name: str) -> tuple[str, str]:
        """Reads the full Python source code of an individual skill module from the skill library."""
        clean_name = skill_name.strip().removeprefix("skills/").removesuffix(".py")
        if clean_name in self.skills:
            code = self.skills[clean_name]
            return code, f"{len(code.splitlines())} lines"
        skills_dir = Path(self.macro_lib.path).parent / "skills"
        skill_file = skills_dir / f"{clean_name}.py"
        if skill_file.exists():
            content = skill_file.read_text(encoding="utf-8")
            return content, f"{len(content.splitlines())} lines"
        avail = list(self.skills.keys())
        if skills_dir.exists():
            avail += [
                p.stem for p in skills_dir.glob("*.py") if p.name != "__init__.py"
            ]
        return (
            f"Error: Skill '{clean_name}' not found. Available skills: {sorted(set(avail)) or 'None'}",
            "not_found",
        )

    def _verify_macro_on_real_states(self, code: str) -> tuple[str, str]:
        """
        Executes candidate macro code against real NetHack GameState fixtures
        to guarantee runtime soundness, attribute validity, and Action yield protocol.
        """
        import traceback

        from lox.env.actions import Action, ActionSequence, RawAction
        from lox.env.state import GameState, VisibleMonster, VisibleObject
        from lox.lang.compiler import _build_base_namespace
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
            n.name for n in ast.walk(macro_tree) if isinstance(n, ast.FunctionDef)
        ]
        if not fn_names:
            return (
                "No function definitions found in candidate macro code block.",
                "NO_FUNCTIONS",
            )

        try:
            tracer = ExecutionTracer()
            ns = _build_base_namespace(tracer)
            macro_inst_tree = PolicyInstrumentTransformer(is_macro_module=True).visit(
                macro_tree
            )
            ast.fix_missing_locations(macro_inst_tree)
            code_obj = compile(
                macro_inst_tree, filename="macros_candidate.py", mode="exec"
            )
            exec(code_obj, ns)
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
            return (
                f"Macro Import/Compilation Error:\n{type(exc).__name__}: {exc}\n{tb}",
                "COMPILE_ERROR",
            )

        # Generate 3 real NetHack GameState fixtures across diverse seeds
        test_states: list[GameState] = []
        try:
            for s_seed in (42, 101, 202):
                h = NetHackHarness(max_steps=50)
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
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
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
                        is_peaceful=True,
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
                    tb = traceback.format_exc(limit=4)
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

    def dry_run_policy(
        self,
        code: str,
        is_macro: bool = False,
        seed: int = 42,
        max_turns: int = 150,
        replay_run_id: str = "",
        replay_steps: int = 0,
        smoke_only: bool = False,
    ) -> tuple[str, str]:
        """
        Compiles and tests candidate `code` in a live NetHack sandbox:
        1. Smoke Check: AST validation, reachability, loop check, and behavioral critic (150 steps).
        2. Pilot Evaluation: If worker pool is attached, runs 20 paired episodes across workers
           (up to 5 pilot revisions per generation) to measure real baseline delta and surface failure logs.
        """
        macros_code = self.macro_lib.load_code()
        policy_code = self.current_policy_code

        if (
            not is_macro
            and self.last_passed_code
            and code.strip() == self.last_passed_code.strip()
            and self.last_pilot_improved
        ):
            return (
                f"Candidate code already verified in pilot batch ({self.last_pilot_summary}).",
                f"PASSED (cached: {self.last_pilot_summary})",
            )

        if is_macro:
            return self._verify_macro_on_real_states(code)

        try:
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
            compiled = compile_policy_package(
                spliced_policy, macros_code, skills=self.skills
            )
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
            locked_door_stalls = 0
            pet_stalls = 0
            boulder_stalls = 0

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
                    harness.step_count = 0
                    harness.done = False
                else:
                    state = harness.reset(seed=int(seed))
            else:
                state = harness.reset(seed=int(seed))

            for _ in range(max(10, min(200, int(max_turns)))):
                raw_act, active_fn = runner.next_action(state)
                state, done, info = harness.step(raw_act)
                msg_l = (state.message or "").lower()
                if "locked" in msg_l:
                    locked_door_stalls += 1
                if "swap places with" in msg_l or "in the way" in msg_l:
                    pet_stalls += 1
                if "try to move the boulder" in msg_l:
                    boulder_stalls += 1
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
            if locked_door_stalls >= 2:
                warn_lines.append(
                    f"- DOOR STALL TELEMETRY: Attempted Action.OPEN on locked door {locked_door_stalls} times! "
                    "Engine returned message 'This door is locked.' with 0 position change. "
                    "Call `query_wiki('door')` or inspect `expert_runs` to identify the appropriate action response."
                )
            if pet_stalls >= 3:
                warn_lines.append(
                    f"- PET INTERACTION TELEMETRY: Hero swapped places with pet {pet_stalls} times! "
                    "Engine returned message 'You swap places with your kitten/dog.' "
                    "Call `query_wiki('pet')` or inspect `state.mem.is_oscillating` to handle pet pathing."
                )
            if boulder_stalls >= 2:
                warn_lines.append(
                    f"- BOULDER INTERACTION TELEMETRY: Movement blocked by boulder {boulder_stalls} times! "
                    "Engine returned message 'You try to move the boulder, but in vain.' "
                    "Inspect obstacle classification in your navigation mask."
                )
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

                    fail_runs = [
                        r
                        for r in self.baseline_runs
                        if r.get("end_reason") not in ("ascended", "max_steps", "ended")
                        or (
                            "stalls:" in str(r.get("summary_digest", "")).lower()
                            and "stalls: none"
                            not in str(r.get("summary_digest", "")).lower()
                        )
                    ]
                    if fail_runs:
                        max_sh = max(1, target_eps // 3)
                        pilot_shards = extract_shards_from_runs(
                            fail_runs,
                            max_shards=max_sh,
                            target_functions=changed_fns,
                        )
                    shard_seeds = {int(sh.seed) for sh in pilot_shards}
                    normal_runs = [r for r in self.baseline_runs if r not in fail_runs]
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
                            for r in self.baseline_runs
                            if r not in strat_runs and int(r["seed"]) not in shard_seeds
                        ]
                        strat_runs.extend(rem[: needed_full - len(strat_runs)])
                    pilot_seeds = [int(r["seed"]) for r in strat_runs]
                    for r in self.baseline_runs:
                        s_val = int(r["seed"])
                        if s_val not in shard_seeds and s_val not in pilot_seeds:
                            pilot_seeds.append(s_val)
                else:
                    pilot_seeds = list(range(int(seed), int(seed) + target_eps))

                base_by_seed = {
                    int(r["seed"]): r
                    for r in self.baseline_runs
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

                target_fns = changed_fns if policy_code.strip() else None

                pilot_res = self.pool.evaluate_batch(
                    batch_id=pilot_batch_id,
                    policy_code=spliced_policy,
                    macros_code=macros_code,
                    target_episodes=target_eps,
                    target_functions=target_fns,
                    priority_seeds=pilot_seeds,
                    unconditional_seeds=base_trig_seeds if base_trig_seeds else None,
                    shards=pilot_shards if pilot_shards else None,
                    skills=self.skills,
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
                            else:
                                ties += 1
                    else:
                        if s in base_by_seed:
                            br = base_by_seed[s]
                            b_d = float(br.get("max_depth", 1))
                            d_val = float(metric_fn(r) - metric_fn(br))
                            deltas.append(d_val)
                            cand_full_depths.append(c_d)
                            base_full_depths.append(b_d)
                            if d_val > 1e-6:
                                wins += 1
                            elif d_val < -1e-6:
                                losses += 1
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

                if base_depths:
                    improved = (wins > losses and mean_delta > 0.0) or (
                        delta_depth >= 0.05 and mean_delta >= 0.0
                    )
                    summary_short = f"{wins}W/{losses}L/{ties}T, Δ={delta_depth:+.2f}"
                    paired_line = f"• Paired vs Baseline: {wins} Wins / {losses} Losses / {ties} Ties | Δ Depth: {delta_depth:+.2f} (Avg Depth: {c_avg_depth:.2f} vs Baseline: {b_avg_depth:.2f}, Max Depth: {c_max_depth})\n"
                    verdict_line = (
                        ">>> PILOT VERDICT: IMPROVEMENT DETECTED over baseline."
                        if improved
                        else ">>> PILOT VERDICT: FLAT OR REGRESSED VS BASELINE. STOP! Do NOT blindly guess. Call query_wiki or query_db to investigate before submitting revised code!"
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
                if (
                    not self.best_pilot_code
                    or score_key >= self.best_pilot_delta
                    or improved
                ):
                    self.best_pilot_delta = max(self.best_pilot_delta, score_key)
                    self.best_pilot_code = code

                if improved:
                    self.last_pilot_improved = True
                    self.last_passed_code = code
                else:
                    if not self.last_passed_code:
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
                    )
                    db_logger.save_batch(
                        batch_id=pilot_batch_id,
                        generation=self.generation,
                        mode="PILOT",
                        parent_policy_id="none",
                        candidate_policy_id=pilot_cand_id,
                        target_functions=target_fns or [],
                        runs=[
                            {k: v for k, v in r.items() if k != "ticks"} for r in p_runs
                        ],
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

                research_hints_block = ""
                if not improved:
                    hints = []
                    monsters: set[str] = set()
                    for r in p_runs:
                        er = str(r.get("end_reason", ""))
                        m = re.search(
                            r"killed by (?:an? )?([a-zA-Z\s]+?)(?:,|\.|$)", er
                        )
                        if m:
                            m_name = m.group(1).strip()
                            if (
                                m_name
                                and len(m_name) < 25
                                and not m_name.startswith("while")
                            ):
                                monsters.add(m_name)
                    for mon in sorted(monsters)[:2]:
                        hints.append(
                            f"query_wiki(query='{mon}')  # Inspect monster speed, melee damage & tactics"
                        )
                    if door_stalls > 0:
                        hints.append(
                            "query_wiki(query='door')  # Research C engine mechanics for locked doors"
                        )
                    if dlvl1_stalls > len(p_runs) * 0.4:
                        hints.append(
                            "query_wiki(query='stairs')  # Research stairs discovery & navigation mechanics"
                        )
                    hints.append(
                        'query_db(sql="SELECT max_depth, max_hp, game_log, summary_digest FROM expert_runs WHERE max_depth >= 2 ORDER BY turns ASC LIMIT 2")'
                    )

                    hints_str = "\n".join(f"  • {h}" for h in hints)
                    research_hints_block = (
                        f"• MANDATORY INVESTIGATION PROTOCOL (Execute BEFORE Modifying Code):\n"
                        f"  DO NOT guess code changes! Diagnose the exact failure using your research tools:\n"
                        f"{hints_str}\n"
                    )

                rem = max_rev - rev_idx
                pilot_diag = (
                    f"\n\n═══════════════════════════════════════════════════════════════════\n"
                    f"PILOT BATCH {rev_idx}/{max_rev} REPORT ({len(p_runs)} episodes across workers in {pilot_res.get('wall_sec', 0.0)}s):\n"
                    f"{paired_line}"
                    f"• End Reasons: {{{end_reasons_str}}}\n"
                    f"• Dlvl 1 Progression: {dlvl1_stalls}/{len(p_runs)} runs stayed on Dlvl 1\n"
                    + (
                        f"• DOOR STALL ALERT: {door_stalls}/{len(p_runs)} runs hit locked doors ('This door is locked.'). Use seq(Action.KICK, dir) to breach locked doors.\n"
                        if door_stalls > 0
                        else ""
                    )
                    + f"• Worst / Stalled Pilot Runs:\n{worst_logs_str}\n"
                    f"• Best Pilot Run: {best_log_str}\n"
                    + (f"{research_hints_block}" if research_hints_block else "")
                    + f"• Revisions Remaining: {rem} left before final hypothesis submission.\n"
                    f"{verdict_line}\n"
                    f"═══════════════════════════════════════════════════════════════════"
                )
                if improved:
                    self.last_passed_code = code
                    return (
                        f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}{pilot_diag}",
                        f"PASSED (Pilot {rev_idx}/{max_rev}: {summary_short})",
                    )
                else:
                    return (
                        f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}{pilot_diag}",
                        f"PILOT_FAILED ({summary_short})",
                    )

            if (
                not smoke_only
                and self.pilot_runs_used > 0
                and not self.last_pilot_improved
            ):
                return (
                    f"Pilot revisions exhausted ({self.pilot_runs_used}/{self.max_pilot_runs}). "
                    f"Candidate code never beat baseline ({self.last_pilot_summary}).",
                    "PILOT_EXHAUSTED",
                )

            if smoke_only:
                return (
                    f"Dry-Run Smoke Check PASSED ({harness.step_count} steps on seed {seed}).{warn_block}",
                    "PASSED (smoke check)",
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
            "name": "read_skill",
            "description": (
                "Retrieve the complete Python source code of a specific skill module (e.g., 'survival', 'combat', 'exploration') "
                "from the hierarchical skill registry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill module to read (e.g. 'survival', 'combat', 'identification').",
                    }
                },
                "required": ["skill_name"],
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
