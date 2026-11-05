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
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH
from lox.lang.macros import DEFAULT_MACROS_PATH
from lox.lang.validator import is_ast_equivalent, splice_functions_into_policy
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
    aborted: bool = False
    abort_reason: str = ""


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
        model: str = "z-ai/glm-5.3-flash:nitro",
        api_key: str | None = None,
        base_url: str | None = None,
        db_path: str = DEFAULT_DB_PATH,
        macros_path: str = DEFAULT_MACROS_PATH,
        max_tokens: int = 16384,
        reasoning_effort: str = "medium",
        routing: str = "nitro",
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
        self.db_logger = BatchDatabaseLogger(db_path=db_path)

    def build_repair_prompt(
        self,
        status: str,
        diag: str,
        mode: str,
        repair_idx: int,
        max_repair_turns: int,
        candidate_code: str = "",
    ) -> str:
        """Builds a precise, category-specific, actionable repair prompt for the Author Agent."""
        hdr = (
            f"Your candidate code failed the compiler/sandbox verification gate "
            f"(Attempt {repair_idx}/{max_repair_turns}):"
        )
        if "UNREACHABLE" in status or "Unreachable Function" in diag:
            guidance = (
                "🚨 PROBLEM: UNREACHABLE METHOD IN POLICY\n"
                "LOX's static reachability analysis found that your newly added/modified method is NEVER invoked "
                "by `run(self, state)` or any other active method in `policy.py`.\n"
                "NetHack policy execution enters via `def run(self, state):` which yields actions.\n\n"
                "REQUIRED FIX:\n"
                "In your ```python ... ``` block, you MUST provide BOTH:\n"
                "1. Your helper method definition (e.g. `def your_method(self, state): ...`)\n"
                "2. The updated caller method (e.g. `def run(self, state): ...` or parent dispatcher) "
                "that explicitly invokes `self.your_method(state)`.\n"
                "Both methods must be included in the SAME ```python ... ``` block."
            )
        elif "COMPILE" in status or "SyntaxError" in diag or "IndentationError" in diag:
            guidance = (
                "🚨 PROBLEM: PYTHON SYNTAX OR INDENTATION ERROR\n"
                "Your code could not be compiled into Python bytecode.\n"
                "REQUIRED FIX:\n"
                "1. Look at the line pointer `>` and column caret `^` in the snippet below.\n"
                "2. Ensure all opening brackets `(`, `[`, `{` have matching closing brackets.\n"
                "3. Ensure strictly 4 spaces per indentation level.\n"
                "4. Use only permitted modules and builtins (e.g. `Action`, `DIR_DELTAS`, `key`, `seq`, `GameState`, `np`).\n"
                "5. Output the complete, self-contained method/class in a single ```python ... ``` block."
            )
        elif "MISSING_CODE" in status:
            guidance = (
                "🚨 PROBLEM: NO PYTHON CODE BLOCK FOUND\n"
                "Your previous response did not contain a valid ```python ... ``` block.\n"
                "REQUIRED FIX:\n"
                "1. Emit your hypothesis in a ```yaml ... ``` block.\n"
                "2. Emit your Python implementation in a ```python ... ``` block.\n"
                "Ensure standard triple backticks (```python) are used."
            )
        elif "RUNTIME" in status:
            guidance = (
                "🚨 PROBLEM: RUNTIME EXCEPTION IN SMOKE TEST\n"
                "Your code crashed during the initial single-episode NetHack smoke test:\n"
                "REQUIRED FIX:\n"
                "1. Guard against unexpected state or empty collections (check `if monsters:`, `if (dy, dx) != (0, 0):`, etc.).\n"
                "2. Ensure you only call valid attributes/methods on `state` (e.g. `state.y`, `state.x`, `state.char_at(y, x)`, `state.visible_monsters()`, `state.path_step(...)`).\n"
                "3. If using `yield`, ensure the generator yields valid `Action` or `seq(...)` actions.\n"
                "4. Output the complete corrected ```python ... ``` block."
            )
        else:
            guidance = (
                "🚨 PROBLEM: VERIFICATION FAILURE\n"
                "Your candidate code did not pass the sandbox compiler verification check.\n"
                "REQUIRED FIX:\n"
                "Review the error diagnostic below, fix the issue, and output the corrected ```python ... ``` block."
            )

        return (
            f"{hdr}\n\n"
            f"```\n{diag.strip()}\n```\n\n"
            f"{guidance}\n\n"
            "Please fix the error and output the corrected ```yaml ... ``` and ```python ... ``` blocks directly in your response."
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

        code_str = ""
        if py_blocks:
            policy_candidates = [
                b for b in py_blocks if "class Policy" in b or "class Agent" in b
            ]
            if policy_candidates:
                code_str = policy_candidates[-1].strip()
            else:
                code_str = py_blocks[-1].strip()
        return yaml_dict, code_str

    @staticmethod
    def extract_skill_code(
        content: str,
        hyp: dict[str, Any],
        fallback_skill_code: str = "",
        fallback_skill_name: str | None = None,
    ) -> str:
        """Extracts external skill module code if specified in hypothesis or markdown."""
        skill_mod = hyp.get("skill_module") or fallback_skill_name
        if skill_mod and "skill_module" not in hyp:
            hyp["skill_module"] = skill_mod
        if not skill_mod:
            return fallback_skill_code

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
                if (
                    f"module: {skill_mod}" in b.lower()
                    or f"skills/{skill_mod}" in b.lower()
                ):
                    return b.strip()
            for b in py_blocks:
                if "class Policy" not in b and ("def " in b or "class " in b):
                    return b.strip()

        return fallback_skill_code

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
            s_name = args.get("skill_name")
            s_code = args.get("skill_code")
            if not code and not (s_name and s_code):
                out = (
                    "Error: `dry_run_policy` requires non-empty `code` or `skill_code`."
                )
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
                    skill_name=str(s_name) if s_name else None,
                    skill_code=str(s_code) if s_code else None,
                )
                arg_disp = f"{len(code.splitlines())} lines, seed={seed}"
                if s_name:
                    arg_disp += f", skill={s_name}"
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
        if self.provider == "openrouter" and ":" not in model_id:
            if self.routing in ("nitro", "floor"):
                model_id = f"{model_id}:{self.routing}"

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
        for attempt in range(5):
            if attempt >= 2:
                for suffix in (":nitro", ":floor"):
                    if payload.get("model", "").endswith(suffix):
                        payload["model"] = payload["model"][: -len(suffix)]
                        break
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
                    if isinstance(data, dict) and "error" in data:
                        err_obj = data["error"]
                        err_code = (
                            err_obj.get("code") if isinstance(err_obj, dict) else None
                        )
                        err_msg = (
                            err_obj.get("message", str(err_obj))
                            if isinstance(err_obj, dict)
                            else str(err_obj)
                        )
                        if err_code in (401, 402, 403):
                            raise PermissionError(
                                f"LLM API Error {err_code}: {err_msg}"
                            )
                        raise ValueError(f"Upstream provider error in JSON: {err_msg}")
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not choices:
                        raise ValueError(
                            f"Empty choices in LLM response: {str(data)[:200]}"
                        )
                    msg_obj = choices[0].get("message") or {}
                    usage = data.get("usage") or {}
                    msg_obj["_usage"] = usage
                    reasoning = (
                        msg_obj.get("reasoning")
                        or msg_obj.get("reasoning_content")
                        or ""
                    )
                    content = msg_obj.get("content") or ""
                    if not reasoning and content:
                        m_think = re.findall(
                            r"<think>(.*?)</think>", content, flags=re.DOTALL
                        )
                        if m_think:
                            reasoning = "\n\n".join(t.strip() for t in m_think)
                    msg_obj["_reasoning"] = reasoning

                    if not content and not msg_obj.get("tool_calls"):
                        if reasoning:
                            msg_obj["content"] = reasoning
                        else:
                            raise ValueError(
                                "LLM returned empty message content and empty tool_calls"
                            )
                    return msg_obj
            except PermissionError:
                raise
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
                KeyError,
                IndexError,
            ):
                if attempt == 4:
                    raise
                time.sleep(min(16.0, 2.0 * (2**attempt)))
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
            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_mock_{int(time.time() * 1000)}",
                    generation=0,
                    batch_id="batch_mock",
                    mode=mode,
                    turn_index=1,
                    turn_type="MOCK_SYNTHESIS",
                    repair_idx=0,
                    model="mock",
                    thinking_content="Mock reasoning step for testing telemetry persistence.",
                    message_content="Mock proposal content",
                    tool_calls_json="[]",
                    tool_results_json="[]",
                    candidate_code=code,
                    skill_name="",
                    skill_code="",
                    hypothesis_yaml=yaml.dump(hyp),
                    status="PASSED",
                    error_diagnostic="",
                    prompt_sent="Mock prompt",
                    prompt_tokens=100,
                    completion_tokens=200,
                    reasoning_tokens=50,
                    wall_ms=1.5,
                )
            except Exception:
                pass
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
        recent_discoveries: list[dict[str, Any]] | None = None,
        recent_failures: list[dict[str, Any]] | None = None,
        max_steps: int = 5000,
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
        self.tools.max_steps = max_steps
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
            recent_discoveries=recent_discoveries,
            recent_failures=recent_failures,
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
        batch_id = f"gen_{generation:03d}_{mode.lower()}"

        for turn_idx in range(1, max_tool_turns + 2):
            allow_tools = turn_idx <= max_tool_turns and not getattr(
                self.tools, "last_pilot_improved", False
            )
            ui.status(
                f"Author Agent querying {self.model} ({'initial reasoning & tool generation' if turn_idx == 1 else f'tool synthesis turn {turn_idx}'})..."
            )
            t0_turn = time.perf_counter()
            msg = self._call_llm_api(messages, allow_tools=allow_tools)
            turn_ms = (time.perf_counter() - t0_turn) * 1000.0
            msg_text = msg.get("content") or ""
            reasoning_text = msg.get("_reasoning") or ""
            usage = msg.get("_usage") or {}
            prompt_toks = int(usage.get("prompt_tokens", 0))
            comp_toks = int(usage.get("completion_tokens", 0))
            reasoning_toks = int(
                usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            )

            clean_text = re.sub(
                r"<think>.*?</think>", "", msg_text, flags=re.DOTALL
            ).strip()
            clean_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
            if msg.get("tool_calls"):
                clean_msg["tool_calls"] = msg["tool_calls"]
            messages.append(clean_msg)
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
                if getattr(self.tools, "last_pilot_improved", False):
                    # Pilot has already passed in this generation; do NOT dispatch further pilots
                    needs_pilot = False
                else:
                    is_new_code = bool(
                        found_code
                        and (
                            not self.tools.last_passed_code
                            or not is_ast_equivalent(
                                found_code, self.tools.last_passed_code
                            )
                        )
                    )
                    needs_pilot = is_new_code

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
                    try:
                        self.db_logger.save_synthesis_log(
                            log_id=f"syn_{generation}_{turn_idx}_autopilot_{int(time.time() * 1000)}",
                            generation=generation,
                            batch_id=batch_id,
                            mode=mode,
                            turn_index=turn_idx,
                            turn_type="PILOT_EXECUTION",
                            repair_idx=0,
                            model=self.model,
                            thinking_content=reasoning_text,
                            message_content=clean_text,
                            tool_calls_json="[]",
                            tool_results_json=json.dumps(
                                [{"pilot_summary": pilot_summary}]
                            ),
                            candidate_code=found_code,
                            skill_name=str((latest_hyp or {}).get("skill_module", "")),
                            skill_code=self.extract_skill_code(clean_text, latest_hyp),
                            hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                            status="PILOT_IMPROVED"
                            if self.tools.last_pilot_improved
                            else (
                                "COMPILE_ERROR"
                                if "COMPILE_ERROR" in pilot_res
                                else (
                                    "RUNTIME_ERROR"
                                    if "RUNTIME_ERROR" in pilot_res
                                    else "PILOT_FAILED"
                                )
                            ),
                            error_diagnostic=pilot_res[:4000],
                            prompt_sent="",
                            prompt_tokens=prompt_toks,
                            completion_tokens=comp_toks,
                            reasoning_tokens=reasoning_toks,
                            wall_ms=turn_ms,
                        )
                    except Exception:
                        pass
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
                        "COMPILE_ERROR" in pilot_res
                        or "RUNTIME_ERROR" in pilot_res
                        or "UNREACHABLE_ERROR" in pilot_res
                    ):
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
                    elif (
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

            tool_results_list: list[dict[str, Any]] = []
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
                tool_results_list.append(
                    {"name": t_name, "args": t_args, "output": str(tool_out)[:2000]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                        "name": t_name,
                        "content": tool_out,
                    }
                )

            try:
                has_pilot = any(
                    (tc.get("function") or {}).get("name") == "dry_run_policy"
                    for tc in tool_calls
                )
                t_type = "PILOT_EXECUTION" if has_pilot else "RESEARCH_TOOL"
                t_stat = (
                    "PILOT_IMPROVED"
                    if getattr(self.tools, "last_pilot_improved", False)
                    else ("PILOT_FAILED" if has_pilot else "TOOL_CALLS")
                )
                _, cand_code_turn = self.extract_yaml_and_python(msg_text)
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_{generation}_{turn_idx}_0_{int(time.time() * 1000)}",
                    generation=generation,
                    batch_id=batch_id,
                    mode=mode,
                    turn_index=turn_idx,
                    turn_type=t_type,
                    repair_idx=0,
                    model=self.model,
                    thinking_content=reasoning_text,
                    message_content=clean_text,
                    tool_calls_json=json.dumps(tool_calls),
                    tool_results_json=json.dumps(tool_results_list),
                    candidate_code=cand_code_turn,
                    skill_name=str((latest_hyp or {}).get("skill_module", "")),
                    skill_code=self.extract_skill_code(clean_text, latest_hyp),
                    hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                    status=t_stat,
                    error_diagnostic=getattr(self.tools, "last_pilot_summary", ""),
                    prompt_sent=str(messages[-len(tool_calls) - 2].get("content", ""))
                    if len(messages) >= len(tool_calls) + 2
                    else "",
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    reasoning_tokens=reasoning_toks,
                    wall_ms=turn_ms,
                )
            except Exception:
                pass

        # Pilot Gate Abort: If pilots were run during the tool phase and none improved over baseline,
        # abort immediately instead of falling through to smoke check and running 150-seed rollout.
        if (
            self.tools.pilot_runs_used > 0
            and not self.tools.last_pilot_improved
            and mode not in ("MACRO_WRITE", "FULL_WRITE")
        ):
            summary = self.tools.last_pilot_summary or "insufficient signal"
            ui.warn(
                f"Pilot Gate: Candidate failed all {self.tools.pilot_runs_used} pilot revisions ({summary}). Aborting rollout."
            )
            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_{generation}_{total_tool_calls}_pilot_abort_{int(time.time() * 1000)}",
                    generation=generation,
                    batch_id=batch_id,
                    mode=mode,
                    turn_index=total_tool_calls,
                    turn_type="PILOT_ABORT",
                    repair_idx=0,
                    model=self.model,
                    thinking_content="",
                    message_content=summary,
                    tool_calls_json="[]",
                    tool_results_json="[]",
                    candidate_code=self.tools.last_attempted_pilot_code or "",
                    skill_name=str((latest_hyp or {}).get("skill_module", "")),
                    skill_code=getattr(self.tools, "last_passed_skill_code", ""),
                    hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                    status="PILOT_FAILED",
                    error_diagnostic=summary,
                    prompt_sent="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    wall_ms=0.0,
                )
            except Exception:
                pass

            return SynthesisProposal(
                mode=mode,
                hypothesis=latest_hyp,
                raw_code_patch="",
                full_policy_code=current_policy,
                full_macros_code=self.tools.macro_lib.load_code(),
                target_functions=[],
                tool_calls_made=total_tool_calls,
                skill_code="",
                aborted=True,
                abort_reason=f"PILOT_GATE_ABORT: {summary}",
            )

        # Compiler-in-the-loop verification & sequential self-repair
        is_macro_mode = mode == "MACRO_WRITE"

        for repair_idx in range(self.max_repair_turns + 1):
            hyp, code_patch = self.extract_yaml_and_python(final_content)
            if hyp:
                latest_hyp = hyp
            elif latest_hyp:
                hyp = latest_hyp

            # If the model passed a pilot, ensure we use that verified code (avoiding slight markdown re-formatting errors)
            if self.tools.last_pilot_improved and self.tools.last_passed_code:
                if not code_patch or is_ast_equivalent(
                    code_patch, self.tools.last_passed_code
                ):
                    code_patch = self.tools.last_passed_code
            elif not code_patch and self.tools.last_passed_code:
                code_patch = self.tools.last_passed_code
            elif not code_patch:
                code_patch = (
                    self.tools.last_passed_code
                    or self.tools.best_pilot_code
                    or code_patch
                )

            if not code_patch:
                diag = "No ```python ... ``` code block found in your response. Please emit your YAML hypothesis and Python code block."
                status = "MISSING_CODE"
            else:
                s_name = (hyp or latest_hyp).get("skill_module") or getattr(
                    self.tools, "last_passed_skill_name", None
                )
                s_code = getattr(self.tools, "last_passed_skill_code", None)
                diag, status = self.tools.dry_run_policy(
                    code=code_patch,
                    is_macro=is_macro_mode,
                    seed=42,
                    max_turns=50,
                    smoke_only=True,
                    skill_name=s_name,
                    skill_code=s_code,
                )

            # Persist verification check to synthesis_logs
            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_{generation}_{max_tool_turns + repair_idx}_{repair_idx}_{int(time.time() * 1000)}",
                    generation=generation,
                    batch_id=batch_id,
                    mode=mode,
                    turn_index=max_tool_turns + repair_idx,
                    turn_type="SMOKE_CHECK" if repair_idx == 0 else "REPAIR_VERIFY",
                    repair_idx=repair_idx,
                    model=self.model,
                    thinking_content="",
                    message_content=final_content[:2000],
                    tool_calls_json="[]",
                    tool_results_json="[]",
                    candidate_code=code_patch,
                    skill_name=str(s_name or ""),
                    skill_code=str(s_code or ""),
                    hypothesis_yaml=yaml.dump(hyp or latest_hyp)
                    if (hyp or latest_hyp)
                    else "",
                    status=status,
                    error_diagnostic=diag,
                    prompt_sent="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    wall_ms=0.0,
                )
            except Exception:
                pass

            if status.startswith("PASSED") or status.startswith("VALID"):
                skill_code = self.extract_skill_code(
                    final_content,
                    hyp,
                    fallback_skill_code=getattr(
                        self.tools, "last_passed_skill_code", ""
                    )
                    or "",
                    fallback_skill_name=getattr(
                        self.tools, "last_passed_skill_name", None
                    ),
                )
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
                    if mode == "FULL_WRITE" or not current_policy.strip():
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
                repair_prompt = self.build_repair_prompt(
                    status=status,
                    diag=diag,
                    mode=mode,
                    repair_idx=repair_idx + 1,
                    max_repair_turns=self.max_repair_turns,
                    candidate_code=code_patch,
                )

                # Ensure previous proposal is present in history before the repair prompt
                if not messages or messages[-1].get("role") != "assistant":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": final_content
                            or (
                                f"```python\n{code_patch}\n```"
                                if code_patch
                                else "I will implement the policy changes."
                            ),
                        }
                    )
                messages.append({"role": "user", "content": repair_prompt})

                t0_rep = time.perf_counter()
                repair_msg = self._call_llm_api(messages, allow_tools=False)
                rep_ms = (time.perf_counter() - t0_rep) * 1000.0

                rep_text = repair_msg.get("content") or ""
                rep_reasoning = repair_msg.get("_reasoning") or ""
                rep_usage = repair_msg.get("_usage") or {}
                clean_rep = re.sub(
                    r"<think>.*?</think>", "", rep_text, flags=re.DOTALL
                ).strip()
                messages.append({"role": "assistant", "content": clean_rep})

                r_hyp, r_code = self.extract_yaml_and_python(clean_rep)
                if r_hyp:
                    latest_hyp = r_hyp
                if r_code:
                    code_patch = r_code
                    final_content = clean_rep
                elif clean_rep:
                    final_content = clean_rep

                # Persist repair turn to synthesis_logs
                try:
                    self.db_logger.save_synthesis_log(
                        log_id=f"syn_{generation}_{max_tool_turns + repair_idx + 1}_rep_{repair_idx + 1}_{int(time.time() * 1000)}",
                        generation=generation,
                        batch_id=batch_id,
                        mode=mode,
                        turn_index=max_tool_turns + repair_idx + 1,
                        turn_type="REPAIR_TURN",
                        repair_idx=repair_idx + 1,
                        model=self.model,
                        thinking_content=rep_reasoning,
                        message_content=clean_rep,
                        tool_calls_json="[]",
                        tool_results_json="[]",
                        candidate_code=r_code or code_patch,
                        skill_name=str((r_hyp or latest_hyp).get("skill_module", "")),
                        skill_code=self.extract_skill_code(
                            clean_rep, r_hyp or latest_hyp
                        ),
                        hypothesis_yaml=yaml.dump(r_hyp or latest_hyp)
                        if (r_hyp or latest_hyp)
                        else "",
                        status="REPAIR_SUBMITTED",
                        error_diagnostic="",
                        prompt_sent=repair_prompt,
                        prompt_tokens=int(rep_usage.get("prompt_tokens", 0)),
                        completion_tokens=int(rep_usage.get("completion_tokens", 0)),
                        reasoning_tokens=int(
                            rep_usage.get("completion_tokens_details", {}).get(
                                "reasoning_tokens", 0
                            )
                        ),
                        wall_ms=rep_ms,
                    )
                except Exception:
                    pass

        # Final fallback: if the model already passed dry_run_policy during the tool phase, use that verified code!
        best_code = self.tools.last_passed_code or self.tools.best_pilot_code
        if best_code:
            ui.info(
                "Using verified code from earlier passing/best dry_run_policy call."
            )
            code_patch = best_code
            skill_code = self.extract_skill_code(
                final_content,
                latest_hyp,
                fallback_skill_code=getattr(self.tools, "last_passed_skill_code", "")
                or "",
                fallback_skill_name=getattr(self.tools, "last_passed_skill_name", None),
            )
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
            if mode == "FULL_WRITE" or not current_policy.strip():
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
