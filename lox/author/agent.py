"""
Stateless Per-Generation Author Agent with Compiler-in-the-Loop Self-Correction.
Supports OpenRouter (including Gemma 4 31B with thinking, GLM-5, DeepSeek, Qwen, Claude),
Gemini, OpenAI-compatible endpoints, and deterministic Mock mode for testing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.tools import OPENAI_TOOL_SPECS, AuthorToolRegistry
from lox.db.schema import DEFAULT_DB_PATH
from lox.lang.macros import DEFAULT_MACROS_PATH
from lox.lang.validator import splice_functions_into_policy
from lox.ui.console import ui

load_dotenv()


@dataclass(slots=True)
class SynthesisProposal:
    """Validated output from an Author Agent synthesis session."""

    mode: str
    hypothesis: dict[str, Any]
    raw_code_patch: str
    full_policy_code: str
    full_macros_code: str
    target_functions: list[str]
    tool_calls_made: int = 0
    skill_code: str = ""


class AuthorAgent:
    """
    Stateless Author Agent.
    Starts a fresh conversation each generation with a static cached System Prompt,
    executes tool calls (`query_db`, `query_wiki`, `query_policy_docs`, `dry_run_policy`),
    and runs a compiler-in-the-loop gate to ensure emitted code compiles and runs cleanly.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "google/gemma-4-31b-it",
        api_key: str | None = None,
        base_url: str | None = None,
        db_path: str = DEFAULT_DB_PATH,
        macros_path: str = DEFAULT_MACROS_PATH,
        max_tokens: int = 16384,
        reasoning_effort: str = "medium",
        routing: str = "floor",
        max_repair_turns: int = 5,
        benchmark: str = "nethack",
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = base_url
        self.db_path = db_path
        self.macros_path = macros_path
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.routing = routing
        self.max_repair_turns = max_repair_turns
        self.benchmark = str(benchmark)
        self.latest_attempted_hyp: dict[str, Any] = {}
        self.tools = AuthorToolRegistry(
            db_path=db_path, macros_path=macros_path, benchmark=benchmark
        )

    @staticmethod
    def extract_yaml_and_python(content: str) -> tuple[dict[str, Any], str]:
        """Extracts the YAML hypothesis dict and Python code block from an LLM response."""
        text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()

        yaml_dict: dict[str, Any] = {}
        yaml_blocks = re.findall(
            r"```(?:yaml|yml)\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        for yb in yaml_blocks:
            try:
                parsed = yaml.safe_load(yb)
                if isinstance(parsed, dict):
                    yaml_dict = parsed
                    break
            except Exception:
                continue

        if yaml_dict:
            if "target_functions" not in yaml_dict:
                t_funcs = (
                    yaml_dict.get("target_function") or yaml_dict.get("functions") or []
                )
                if isinstance(t_funcs, str):
                    t_funcs = [t_funcs]
                yaml_dict["target_functions"] = list(t_funcs)
            elif isinstance(yaml_dict["target_functions"], str):
                yaml_dict["target_functions"] = [yaml_dict["target_functions"]]

            if "binding_constraint" not in yaml_dict:
                yaml_dict["binding_constraint"] = str(
                    yaml_dict.get("observation_from_db")
                    or yaml_dict.get("observation")
                    or ""
                )

            if "verified_c_rule" not in yaml_dict:
                yaml_dict["verified_c_rule"] = str(
                    yaml_dict.get("c_rule") or yaml_dict.get("wiki_rule") or ""
                )

            if "mechanism" not in yaml_dict:
                yaml_dict["mechanism"] = str(
                    yaml_dict.get("description") or yaml_dict.get("hypothesis") or ""
                )

            if "state_persistence" not in yaml_dict:
                yaml_dict["state_persistence"] = str(
                    yaml_dict.get("persistence") or yaml_dict.get("memory") or ""
                )

        py_blocks = re.findall(
            r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if not py_blocks:
            py_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
            py_blocks = [b for b in py_blocks if "def " in b or "class " in b]

        code_str = py_blocks[-1].strip() if py_blocks else ""
        return yaml_dict, code_str

    @staticmethod
    def extract_skill_code(content: str, hyp: dict[str, Any]) -> str:
        """Extracts external skill module code if specified in hypothesis or markdown."""
        skill_mod = hyp.get("skill_module")
        if not skill_mod:
            return ""
        if hyp.get("skill_code"):
            return str(hyp["skill_code"]).strip()

        text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
        m = re.search(
            rf"```(?:python:)?(?:skill|{re.escape(str(skill_mod))})\s*\n(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        py_blocks = re.findall(
            r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if len(py_blocks) >= 2:
            for b in py_blocks:
                if "class Policy" not in b and ("def " in b or "class " in b):
                    return b.strip()

        return ""

    def _execute_tool_call(self, turn_idx: int, name: str, args: dict[str, Any]) -> str:
        t0 = time.perf_counter()
        if name == "query_db":
            sql = str(args.get("sql", ""))
            out, summary = self.tools.query_db(sql)
            arg_disp = sql
        elif name == "query_wiki":
            q = str(args.get("query", ""))
            tk = int(args.get("top_k", 3))
            out, summary = self.tools.query_wiki(q, top_k=tk)
            arg_disp = q
        elif name == "query_policy_docs":
            topic = str(args.get("topic", "all"))
            out, summary = self.tools.query_policy_docs(topic)
            arg_disp = topic
        elif name == "read_skill":
            skill_name = str(args.get("skill_name", ""))
            out, summary = self.tools.read_skill(skill_name)
            arg_disp = skill_name
        elif name == "dry_run_policy":
            code = str(args.get("code", "")).strip()
            if not code:
                out = "Error: `dry_run_policy` requires a non-empty `code` string containing Python code."
                summary = "MISSING_CODE"
                arg_disp = "empty code"
            else:
                is_macro = bool(args.get("is_macro", False))
                seed = int(args.get("seed", 42))
                mt = int(args.get("max_turns", 150))
                replay_id = str(args.get("replay_run_id", ""))
                replay_steps = int(args.get("replay_steps", 0))
                out, summary = self.tools.dry_run_policy(
                    code=code,
                    is_macro=is_macro,
                    seed=seed,
                    max_turns=mt,
                    replay_run_id=replay_id,
                    replay_steps=replay_steps,
                )
                arg_disp = f"{len(code.splitlines())} lines, seed={seed}"
                if replay_id:
                    arg_disp += f", replay={replay_id[:10]}..."
        else:
            out, summary = f"Unknown tool '{name}'.", "unknown"
            arg_disp = str(args)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ui.tool_call(turn_idx, name, arg_disp, summary, elapsed_ms)
        return out

    def _call_llm_api(
        self,
        messages: list[dict[str, Any]],
        allow_tools: bool = True,
    ) -> dict[str, Any]:
        """Dispatches a single chat completion request to OpenRouter, OpenAI, or Gemini."""
        if self.provider == "openrouter":
            url = self.base_url or "https://openrouter.ai/api/v1/chat/completions"
        elif self.provider == "gemini":
            url = (
                self.base_url
                or "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            )
        else:
            url = self.base_url or "https://api.openai.com/v1/chat/completions"

        model_id = self.model
        if (
            self.provider == "openrouter"
            and self.routing == "floor"
            and ":" not in model_id
        ):
            model_id = f"{model_id}:floor"

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.4,
        }
        if allow_tools:
            payload["tools"] = OPENAI_TOOL_SPECS
            payload["tool_choice"] = "auto"
        if self.provider == "openrouter" and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if not self.api_key or self.api_key in ("your-api-key", "sk-or-v1-..."):
            raise PermissionError(
                "Missing or placeholder API key. Set a valid OPENROUTER_API_KEY in `.env` or pass `--api-key`."
            )
        for attempt in range(3):
            try:
                with httpx.Client(timeout=180.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code in (401, 402, 403):
                        err_detail = resp.text[:200]
                        try:
                            err_detail = (
                                resp.json().get("error", {}).get("message")
                                or err_detail
                            )
                        except Exception:
                            pass
                        raise PermissionError(
                            f"LLM API HTTP {resp.status_code}: {err_detail} "
                            "(Check your OPENROUTER_API_KEY in `.env` or environment)"
                        )
                    if resp.status_code != 200 and "reasoning" in payload:
                        # Retry without reasoning field if endpoint does not support it
                        payload.pop("reasoning", None)
                        resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]
            except PermissionError:
                raise
            except (httpx.RequestError, httpx.HTTPStatusError):
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("Failed to obtain response from LLM API after retries.")

    def _synthesize_mock(
        self,
        mode: str,
        current_policy: str,
    ) -> SynthesisProposal:
        """Deterministic mock synthesizer for automated tests (`provider='mock'`)."""
        self.tools.current_policy_code = current_policy
        # Exercise tools so tool pipeline is verified in tests
        self._execute_tool_call(
            1, "query_db", {"sql": "SELECT COUNT(*) AS n FROM expert_runs"}
        )
        self._execute_tool_call(1, "query_wiki", {"query": "Excalibur", "top_k": 1})

        if mode == "FULL_WRITE" or not current_policy.strip():
            code = """class Policy:
    def __init__(self):
        self.visited = set()

    def explore_step(self, state):
        self.visited.add((state.y, state.x))
        if state.char_at(state.y, state.x) == '>':
            return Action.DOWN
        stairs = np.argwhere(state.chars == ord('>'))
        passable = (
            (state.chars == ord('.'))
            | (state.chars == ord('#'))
            | (state.chars == ord('>'))
            | (state.chars == ord('<'))
            | (state.chars == ord('+'))
        )
        if len(stairs) > 0:
            dy, dx = state.path_step(passable, (int(stairs[0][0]), int(stairs[0][1])))
            if (dy, dx) != (0, 0):
                if state.char_at(state.y + dy, state.x + dx) == '+':
                    return seq(Action.OPEN, DIR_DELTAS[(dy, dx)])
                return DIR_DELTAS[(dy, dx)]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = state.y + dy, state.x + dx
            if 0 <= ny < 21 and 0 <= nx < 79 and passable[ny, nx] and (ny, nx) not in self.visited:
                if state.char_at(ny, nx) == '+':
                    return seq(Action.OPEN, DIR_DELTAS[(dy, dx)])
                return DIR_DELTAS[(dy, dx)]
        return Action.SEARCH

    def handle_combat(self, state):
        hp_thresh = self.param("retreat_hp", 0.25, 0.1, 0.6)
        mons = [m for m in state.visible_monsters() if not m.is_pet and not m.is_peaceful]
        for m in mons:
            if abs(m.y - state.y) <= 1 and abs(m.x - state.x) <= 1:
                return DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
        return self.explore_step(state)

    def run(self, state):
        while True:
            mons = [m for m in state.visible_monsters() if not m.is_pet and not m.is_peaceful]
            if any(abs(m.y - state.y) <= 1 and abs(m.x - state.x) <= 1 for m in mons):
                act = self.handle_combat(state)
            else:
                act = self.explore_step(state)
            state = yield act
"""
            hyp = {
                "mode": "FULL_WRITE",
                "target_functions": ["run", "explore_step", "handle_combat"],
                "observation_from_db": "Initial baseline synthesis from expert_runs and wiki.",
                "mechanism": "Builds modular Policy with explore_step and handle_combat.",
            }
            return SynthesisProposal(
                mode="FULL_WRITE",
                hypothesis=hyp,
                raw_code_patch=code,
                full_policy_code=code,
                full_macros_code=self.tools.macro_lib.load_code(),
                target_functions=["run", "explore_step", "handle_combat"],
                tool_calls_made=2,
            )

        if mode == "MACRO_WRITE":
            macro_code = '''def build_walkable_mask(state):
    """Returns a boolean (21, 79) mask of standard walkable floor, corridor, stair, and door tiles."""
    return (
        (state.chars == ord('.'))
        | (state.chars == ord('#'))
        | (state.chars == ord('>'))
        | (state.chars == ord('<'))
        | (state.chars == ord('+'))
        | (state.chars == ord('_'))
        | (state.chars == ord('{'))
    )
'''
            merged_macros, names = self.tools.macro_lib.merge_macros(macro_code)
            hyp = {
                "mode": "MACRO_WRITE",
                "target_functions": names,
                "observation_from_db": "Extracted reusable walkable mask builder into macros.py.",
                "mechanism": "Adds build_walkable_mask(state) to macros.py.",
            }
            return SynthesisProposal(
                mode="MACRO_WRITE",
                hypothesis=hyp,
                raw_code_patch=macro_code,
                full_policy_code=current_policy,
                full_macros_code=merged_macros,
                target_functions=names,
                tool_calls_made=2,
            )

        # FUNC_MOD_ADD mock
        patch = """def handle_combat(self, state):
    hp_thresh = self.param("retreat_hp", 0.30, 0.1, 0.6)
    if state.hp / max(1, state.max_hp) < hp_thresh and state.turn > 10:
        for item in state.inventory:
            if "healing" in item.name.lower() and item.char == "!":
                return seq(Action.QUAFF, item.letter)
    mons = [m for m in state.visible_monsters() if not m.is_pet and not m.is_peaceful]
    for m in mons:
        if abs(m.y - state.y) <= 1 and abs(m.x - state.x) <= 1:
            return DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
    return self.explore_step(state)
"""
        spliced, changed = splice_functions_into_policy(current_policy, patch)
        hyp = {
            "mode": "FUNC_MOD_ADD",
            "target_functions": changed,
            "observation_from_db": "Outlier good runs survived melee by quaffing healing potions at low HP.",
            "mechanism": "Adds healing potion quaff check to handle_combat.",
        }
        return SynthesisProposal(
            mode="FUNC_MOD_ADD",
            hypothesis=hyp,
            raw_code_patch=patch,
            full_policy_code=spliced,
            full_macros_code=self.tools.macro_lib.load_code(),
            target_functions=changed,
            tool_calls_made=2,
        )

    def synthesize(
        self,
        mode: str,
        generation: int,
        current_policy: str,
        batch_digest: str = "",
        recent_experiments: str = "",
        max_tool_turns: int | None = None,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        max_pilot_runs: int = 5,
        pilot_episodes: int = 30,
        skills: dict[str, str] | None = None,
    ) -> SynthesisProposal:
        """
        Runs a fresh, stateless multi-turn tool-calling session for `generation` and `mode`.
        Supports multi-turn pilot evaluations against `baseline_runs` using `pool` inside
        `dry_run_policy` (up to `max_pilot_runs` revisions), plus compiler-in-the-loop self-repair.
        """
        if self.provider == "mock":
            return self._synthesize_mock(mode=mode, current_policy=current_policy)

        self.tools.current_policy_code = current_policy
        self.tools.last_passed_code = ""
        self.tools.skills = dict(skills) if skills else {}
        self.tools.configure_generation(
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=max_pilot_runs,
            pilot_episodes=pilot_episodes,
            skills=skills,
            generation=generation,
            benchmark=self.benchmark,
        )
        if max_tool_turns is None:
            max_tool_turns = 14 if mode == "FULL_WRITE" else 12

        sys_prompt = build_system_prompt(self.db_path)
        user_prompt = build_user_prompt(
            mode=mode,
            generation=generation,
            current_policy=current_policy,
            macros_docs=self.tools.macro_lib.format_docs(),
            batch_digest=batch_digest,
            recent_experiments=recent_experiments,
            skills=skills,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_tool_calls = 0
        research_tools_called: set[str] = set()
        final_content = ""
        latest_hyp: dict[str, Any] = {}
        last_attempted_code = ""

        for turn_idx in range(1, max_tool_turns + 2):
            allow_tools = turn_idx <= max_tool_turns
            ui.status(
                f"Author Agent querying {self.model} ({'initial reasoning & tool generation' if turn_idx == 1 else f'tool synthesis turn {turn_idx}'})..."
            )
            msg = self._call_llm_api(messages, allow_tools=allow_tools)
            messages.append(msg)

            msg_text = msg.get("content") or ""
            if msg_text:
                inter_hyp, _ = self.extract_yaml_and_python(msg_text)
                if inter_hyp:
                    latest_hyp = inter_hyp
                    self.latest_attempted_hyp = latest_hyp

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                _, found_code = self.extract_yaml_and_python(msg_text)

                # Turn 1 Mandatory Investigation Enforcement:
                # In FUNC_MOD_ADD and STRUCTURAL_REFACTOR modes, Turn 1 is strictly reserved for Empirical Research.
                # If candidate code is output on Turn 1 without having called query_db or query_wiki, block auto-pilot and enforce research.
                if (
                    turn_idx == 1
                    and mode in ("FUNC_MOD_ADD", "STRUCTURAL_REFACTOR")
                    and found_code
                    and not ({"query_db", "query_wiki"} & research_tools_called)
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "🛑 PHASE 1 ENFORCEMENT: Turn 1 is strictly reserved for Empirical Research.\n"
                                "DO NOT output code yet. You must first investigate the previous batch's failure modes:\n"
                                "1. Call `query_db(sql=...)` to compare `champion_runs` / `champion_function_stats` against `expert_runs` or `expert_ticks`.\n"
                                "2. Call `query_wiki(query=...)` on the exact message strings from the top failure causes.\n"
                                "Once you have gathered empirical C engine facts, formulate your hypothesis on Turn 2."
                            ),
                        }
                    )
                    continue

                # Check if candidate code is present and has not yet passed a pilot evaluation
                is_new_code = bool(
                    found_code
                    and (
                        not self.tools.last_passed_code
                        or found_code.strip() != self.tools.last_passed_code.strip()
                    )
                )
                needs_pilot = is_new_code or not self.tools.last_pilot_improved

                # Auto-pilot dispatch: if candidate code was emitted in text without running dry_run_policy
                if (
                    found_code
                    and needs_pilot
                    and self.tools.pilot_runs_used < self.tools.max_pilot_runs
                    and turn_idx < max_tool_turns
                    and mode != "MACRO_WRITE"
                ):
                    total_tool_calls += 1
                    t0 = time.perf_counter()
                    pilot_res, pilot_summary = self.tools.dry_run_policy(
                        code=found_code
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    arg_disp = f"auto-pilot ({len(found_code.splitlines())} lines)"
                    ui.tool_call(
                        turn_idx,
                        "dry_run_policy",
                        arg_disp,
                        pilot_summary,
                        elapsed_ms,
                    )

                    if (
                        not self.tools.last_pilot_improved
                        and self.tools.pilot_runs_used < self.tools.max_pilot_runs
                    ):
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"We executed your candidate code in {self.tools.pilot_episodes}-episode Pilot Revision {self.tools.pilot_runs_used}/{self.tools.max_pilot_runs}:\n"
                                    f"{pilot_res}\n\n"
                                    f"🛑 PILOT REVISION FAILED ({self.tools.last_pilot_summary}). "
                                    f"You have {self.tools.max_pilot_runs - self.tools.pilot_runs_used} pilot revision(s) remaining.\n\n"
                                    "MANDATORY SCIENTIFIC INVESTIGATION PROTOCOL:\n"
                                    "DO NOT BLINDLY GUESS CODE EDITS!\n"
                                    "1. Inspect the failed run logs and behavioral critic warnings above (look at the exact end_reason, HP, Dlvl, turn, and last actions).\n"
                                    "2. Use your research tools right now before modifying code:\n"
                                    "   - Call `query_wiki(query=...)` to check C engine mechanics for the failure cause (e.g. monster attacks, locked doors, trap effects, hunger).\n"
                                    "   - Call `query_db(sql=...)` to inspect expert runs (`expert_runs` / `expert_ticks`) to see how human ascensions handled this scenario.\n"
                                    "3. Once you have identified the concrete mechanism from the wiki/DB, revise your method and test again with `dry_run_policy(code=...)`."
                                ),
                            }
                        )
                        continue
                    elif "COMPILE_ERROR" in pilot_res or "RUNTIME_ERROR" in pilot_res:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your candidate code resulted in an error:\n{pilot_res}\n"
                                    "Please fix the error and test with `dry_run_policy`."
                                ),
                            }
                        )
                        continue

                # Early stop guard: if the model emitted a plan on an early turn without calling tools or testing code
                if (
                    not found_code
                    and not self.tools.last_passed_code
                    and turn_idx < max_tool_turns
                    and mode != "MACRO_WRITE"
                ):
                    if (
                        self.tools.pilot_runs_used > 0
                        and not self.tools.last_pilot_improved
                    ):
                        nudge_msg = (
                            f"Pilot revision {self.tools.pilot_runs_used}/{self.tools.max_pilot_runs} failed ({self.tools.last_pilot_summary}). "
                            f"You have {self.tools.max_pilot_runs - self.tools.pilot_runs_used} revision(s) remaining.\n"
                            "Please call `query_wiki` or `query_db` to investigate the failure mechanism, then test your revised code with `dry_run_policy(code=...)`."
                        )
                    elif total_tool_calls == 0 and turn_idx == 1:
                        nudge_msg = (
                            "Please begin your empirical investigation by calling `query_db` or `query_wiki` to examine the previous batch's failure modes and expert runs, "
                            "then test your candidate method with `dry_run_policy(code=...)`."
                        )
                    else:
                        nudge_msg = (
                            "Please test your candidate code with `dry_run_policy(code=...)` before finalizing. "
                            f"This executes a {self.tools.pilot_episodes}-episode pilot test against the current baseline and reports where your code succeeds or fails."
                        )
                    messages.append({"role": "user", "content": nudge_msg})
                    continue

                # Iterative pilot revision nudge: if the pilot didn't beat baseline and revisions remain
                if (
                    self.tools.pilot_runs_used > 0
                    and not self.tools.last_pilot_improved
                    and self.tools.pilot_runs_used < self.tools.max_pilot_runs
                    and turn_idx < max_tool_turns - 1
                    and mode == "FUNC_MOD_ADD"
                    and (
                        not found_code
                        or found_code.strip() != last_attempted_code.strip()
                    )
                ):
                    last_attempted_code = found_code
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your latest {self.tools.pilot_episodes}-episode pilot did not beat the baseline ({self.tools.last_pilot_summary}). "
                                f"You have {self.tools.max_pilot_runs - self.tools.pilot_runs_used} pilot revision(s) remaining.\n"
                                "🛑 Do NOT guess! Call `query_wiki` or `query_db` to investigate the failure mechanism, then revise your code and call `dry_run_policy(code=...)` to verify your fix."
                            ),
                        }
                    )
                    continue

                final_content = msg_text
                break

            for tc in tool_calls:
                total_tool_calls += 1
                fn_Spec = tc.get("function") or {}
                t_name = fn_Spec.get("name", "")
                if t_name in ("query_db", "query_wiki"):
                    research_tools_called.add(t_name)
                raw_args = fn_Spec.get("arguments", "{}")
                try:
                    t_args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else dict(raw_args)
                    )
                except Exception:
                    t_args = {}
                tool_out = self._execute_tool_call(turn_idx, t_name, t_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                        "name": t_name,
                        "content": tool_out,
                    }
                )

        # Compiler-in-the-loop verification & sliding-window self-repair
        is_macro_mode = mode == "MACRO_WRITE"
        base_msg_count = len(messages)

        for repair_idx in range(self.max_repair_turns + 1):
            hyp, code_patch = self.extract_yaml_and_python(final_content)
            if hyp:
                latest_hyp = hyp
            elif latest_hyp:
                hyp = latest_hyp

            # If the model omitted the code block in its final summary after already passing dry_run_policy in a tool call, use that verified code!
            if not code_patch:
                code_patch = self.tools.last_passed_code or self.tools.best_pilot_code

            if not code_patch:
                diag = "No ```python ... ``` code block found in your response. Please emit your YAML hypothesis and Python code block."
                status = "MISSING_CODE"
            else:
                diag, status = self.tools.dry_run_policy(
                    code=code_patch,
                    is_macro=is_macro_mode,
                    seed=42,
                    max_turns=50,
                    smoke_only=True,
                )

            if status.startswith("PASSED") or status.startswith("VALID"):
                skill_code = self.extract_skill_code(final_content, hyp)
                if is_macro_mode:
                    merged_macros, macro_names = self.tools.macro_lib.merge_macros(
                        code_patch
                    )
                    targets = hyp.get("target_functions") or macro_names
                    if isinstance(targets, str):
                        targets = [targets]
                    return SynthesisProposal(
                        mode=mode,
                        hypothesis=hyp,
                        raw_code_patch=code_patch,
                        full_policy_code=current_policy,
                        full_macros_code=merged_macros,
                        target_functions=list(targets),
                        tool_calls_made=total_tool_calls,
                        skill_code=skill_code,
                    )
                else:
                    if (
                        mode in ("FULL_WRITE", "STRUCTURAL_REFACTOR")
                        or not current_policy.strip()
                    ):
                        spliced_policy = code_patch
                        changed_fns = hyp.get("target_functions") or ["run"]
                    else:
                        spliced_policy, changed_fns = splice_functions_into_policy(
                            current_policy, code_patch
                        )
                    if isinstance(changed_fns, str):
                        changed_fns = [changed_fns]
                    return SynthesisProposal(
                        mode=mode,
                        hypothesis=hyp,
                        raw_code_patch=code_patch,
                        full_policy_code=spliced_policy,
                        full_macros_code=self.tools.macro_lib.load_code(),
                        target_functions=list(changed_fns),
                        tool_calls_made=total_tool_calls,
                        skill_code=skill_code,
                    )

            if repair_idx < self.max_repair_turns:
                ui.warn(
                    f"Sandbox compiler gate caught {status}; requesting in-turn repair ({repair_idx + 1}/{self.max_repair_turns})..."
                )
                # Sliding-window repair context: keep only the latest attempt + traceback so failed attempts don't poison context
                messages = messages[:base_msg_count] + [
                    {
                        "role": "user",
                        "content": (
                            f"Your submitted Python code failed the sandbox dry-run check:\n\n{diag}\n\n"
                            "Please fix the error and output the corrected ```yaml ... ``` and ```python ... ``` blocks directly in your response."
                        ),
                    }
                ]
                repair_msg = self._call_llm_api(messages, allow_tools=True)
                messages.append(repair_msg)
                r_text = repair_msg.get("content") or ""
                if r_text:
                    r_hyp, _ = self.extract_yaml_and_python(r_text)
                    if r_hyp:
                        latest_hyp = r_hyp
                r_tools = repair_msg.get("tool_calls") or []
                if r_tools:
                    updated_via_dry_run = False
                    for tc in r_tools:
                        total_tool_calls += 1
                        fn_spec = tc.get("function") or {}
                        t_name = fn_spec.get("name", "")
                        raw_args = fn_spec.get("arguments", "{}")
                        try:
                            t_args = (
                                json.loads(raw_args)
                                if isinstance(raw_args, str)
                                else dict(raw_args)
                            )
                        except Exception:
                            t_args = {}
                        tool_out = self._execute_tool_call(
                            max_tool_turns + repair_idx + 1, t_name, t_args
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get(
                                    "id", f"call_{total_tool_calls}"
                                ),
                                "name": t_name,
                                "content": tool_out,
                            }
                        )
                        if t_name == "dry_run_policy" and t_args.get("code"):
                            final_content = f"```python\n{t_args['code']}\n```"
                            updated_via_dry_run = True
                    if not updated_via_dry_run:
                        messages.append(
                            {
                                "role": "user",
                                "content": "Now that you have consulted your research tools, please provide your revised implementation in a single ```python ... ``` block.",
                            }
                        )
                        follow_msg = self._call_llm_api(messages, allow_tools=False)
                        messages.append(follow_msg)
                        f_text = follow_msg.get("content") or ""
                        _, f_code = self.extract_yaml_and_python(f_text)
                        _, r_code = self.extract_yaml_and_python(r_text)
                        if f_code:
                            final_content = f_text
                        elif r_code:
                            final_content = r_text
                        else:
                            final_content = f_text
                else:
                    final_content = r_text

        # Final fallback: if the model already passed dry_run_policy during the tool phase, use that verified code!
        best_code = self.tools.last_passed_code or self.tools.best_pilot_code
        if best_code:
            ui.info(
                "Using verified code from earlier passing/best dry_run_policy call."
            )
            code_patch = best_code
            skill_code = self.extract_skill_code(final_content, latest_hyp)
            if is_macro_mode:
                merged_macros, macro_names = self.tools.macro_lib.merge_macros(
                    code_patch
                )
                return SynthesisProposal(
                    mode=mode,
                    hypothesis=latest_hyp,
                    raw_code_patch=code_patch,
                    full_policy_code=current_policy,
                    full_macros_code=merged_macros,
                    target_functions=list(macro_names),
                    tool_calls_made=total_tool_calls,
                    skill_code=skill_code,
                )
            if (
                mode in ("FULL_WRITE", "STRUCTURAL_REFACTOR")
                or not current_policy.strip()
            ):
                spliced_policy = code_patch
                changed_fns = latest_hyp.get("target_functions") or ["run"]
            else:
                spliced_policy, changed_fns = splice_functions_into_policy(
                    current_policy, code_patch
                )
            if isinstance(changed_fns, str):
                changed_fns = [changed_fns]
            return SynthesisProposal(
                mode=mode,
                hypothesis=latest_hyp,
                raw_code_patch=code_patch,
                full_policy_code=spliced_policy,
                full_macros_code=self.tools.macro_lib.load_code(),
                target_functions=list(changed_fns),
                tool_calls_made=total_tool_calls,
                skill_code=skill_code,
            )

        raise RuntimeError(
            f"Author Agent failed to produce valid compiling code after {self.max_repair_turns} repair attempts. Last error: {diag}"
        )
