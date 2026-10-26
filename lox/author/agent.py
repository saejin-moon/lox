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
    tool_calls_made: int


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
        self.tools = AuthorToolRegistry(db_path=db_path, macros_path=macros_path)

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

        py_blocks = re.findall(
            r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if not py_blocks:
            py_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
            py_blocks = [b for b in py_blocks if "def " in b or "class " in b]

        code_str = py_blocks[-1].strip() if py_blocks else ""
        return yaml_dict, code_str

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
        elif name == "dry_run_policy":
            code = str(args.get("code", ""))
            is_macro = bool(args.get("is_macro", False))
            seed = int(args.get("seed", 42))
            mt = int(args.get("max_turns", 60))
            out, summary = self.tools.dry_run_policy(
                code=code, is_macro=is_macro, seed=seed, max_turns=mt
            )
            arg_disp = f"{len(code.splitlines())} lines, seed={seed}"
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
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200 and "reasoning" in payload:
                # Retry without reasoning field if endpoint does not support it
                payload.pop("reasoning", None)
                resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]

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
    ) -> SynthesisProposal:
        """
        Runs a fresh, stateless multi-turn tool-calling session for `generation` and `mode`.
        Includes a compiler-in-the-loop validation gate (up to 2 repair turns) so models
        like Gemma 4 31B self-correct any syntax or attribute errors before returning.
        """
        if self.provider == "mock":
            return self._synthesize_mock(mode=mode, current_policy=current_policy)

        self.tools.current_policy_code = current_policy
        self.tools.last_passed_code = ""
        if max_tool_turns is None:
            max_tool_turns = 12 if mode == "FULL_WRITE" else 6

        sys_prompt = build_system_prompt(self.db_path)
        user_prompt = build_user_prompt(
            mode=mode,
            generation=generation,
            current_policy=current_policy,
            macros_docs=self.tools.macro_lib.format_docs(),
            batch_digest=batch_digest,
            recent_experiments=recent_experiments,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_tool_calls = 0
        final_content = ""

        for turn_idx in range(1, max_tool_turns + 2):
            allow_tools = turn_idx <= max_tool_turns
            msg = self._call_llm_api(messages, allow_tools=allow_tools)
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                final_content = msg.get("content") or ""
                break

            for tc in tool_calls:
                total_tool_calls += 1
                fn_Spec = tc.get("function") or {}
                t_name = fn_Spec.get("name", "")
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
        latest_hyp: dict[str, Any] = {}

        for repair_idx in range(self.max_repair_turns + 1):
            hyp, code_patch = self.extract_yaml_and_python(final_content)
            if hyp:
                latest_hyp = hyp
            elif latest_hyp:
                hyp = latest_hyp

            # If the model omitted the code block in its final summary after already passing dry_run_policy in a tool call, use that verified code!
            if not code_patch and self.tools.last_passed_code:
                code_patch = self.tools.last_passed_code

            if not code_patch:
                diag = "No ```python ... ``` code block found in your response. Please emit your YAML hypothesis and Python code block."
                status = "MISSING_CODE"
            else:
                diag, status = self.tools.dry_run_policy(
                    code=code_patch, is_macro=is_macro_mode, seed=42, max_turns=50
                )

            if status.startswith("PASSED") or status.startswith("VALID"):
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
                r_tools = repair_msg.get("tool_calls") or []
                if r_tools:
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
                else:
                    final_content = repair_msg.get("content") or ""

        # Final fallback: if the model already passed dry_run_policy during the tool phase, use that verified code!
        if self.tools.last_passed_code:
            ui.info("Using verified code from earlier passing dry_run_policy call.")
            code_patch = self.tools.last_passed_code
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
            )

        raise RuntimeError(
            f"Author Agent failed to produce valid compiling code after {self.max_repair_turns} repair attempts. Last error: {diag}"
        )
