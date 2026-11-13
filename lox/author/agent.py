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
from lox.lang.validator import (
    is_ast_equivalent,
    normalize_mode,
    prune_extracted_skill_functions,
    smart_dedent,
    splice_functions_into_policy,
    validate_python_code,
)
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
    pilot_best_delta: float = 0.0


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
        pilot_fallback: str = "best",
        agent_role: str = "author",
    ):
        self.provider = provider.lower()
        self.model = model
        self.agent_role = str(agent_role)
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
        self.pilot_fallback = str(pilot_fallback)
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
        canon_mode = normalize_mode(mode)
        hdr = (
            f"Your candidate code failed the compiler/sandbox verification gate "
            f"(Attempt {repair_idx}/{max_repair_turns}):"
        )
        if canon_mode == "SWARM_WORKER":
            return (
                f"{hdr}\n\n"
                f"```\n{diag.strip()}\n```\n\n"
                "🚨 PROBLEM: SWARM WORKER BEHAVIOR TREE SUBTREE FAILED VERIFICATION\n"
                "REQUIRED FIX:\n"
                "1. Emit a valid, standalone Behavior Tree subtree assigned to `subtree` "
                "(e.g. `subtree = PrioritySelector([...])` or `subtree = ActionLeaf(...)`).\n"
                "2. Use only permitted imports (`from lox.lang.bt import ...`, `from lox.env.actions import Action, DIR_DELTAS, seq`). "
                "NEVER import `macros`, `skills_common`, or unapproved modules.\n"
                "3. Do NOT wrap the subtree in monolithic `class Policy` with `def run(self, state): yield ...`.\n"
                "4. Fix any syntax, indentation, or runtime errors and output the complete corrected subtree inside a single ```python ... ``` block."
            )
        if canon_mode == "SWARM_META":
            return (
                f"{hdr}\n\n"
                f"```\n{diag.strip()}\n```\n\n"
                "🚨 PROBLEM: MASTER BEHAVIOR TREE POLICY FAILED VERIFICATION\n"
                "REQUIRED FIX:\n"
                "1. Output a complete, self-contained `class Policy:` defining `def __init__(self):` "
                "(with `self.blackboard = Blackboard()` and `self.tree = BehaviorTree(...)`) and "
                "`def step(self, state): return self.tree.tick(state) or Action.SEARCH`.\n"
                "2. Do NOT reference undeclared worker helper names or import `macros` / `skills_common`.\n"
                "3. Output the complete corrected `class Policy:` inside a single ```python ... ``` block."
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
    def _sanitize_skill_code(code: str) -> str:
        """Strips leading non-python markdown headers (e.g. `**Block 2 — skills/...**:`) and artifacts."""
        lines = code.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if (
                stripped.startswith("**")
                or stripped.startswith("##")
                or stripped.startswith("Block ")
            ):
                continue
            start_idx = i
            break
        return "\n".join(lines[start_idx:])

    @staticmethod
    def extract_yaml_and_python(content: str) -> tuple[dict[str, Any], str]:
        """Extracts the YAML hypothesis dict and Python code block from an LLM response."""
        text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()

        yaml_dict: dict[str, Any] = {}
        # 1. Prefer explicit <hypothesis> ... </hypothesis> XML tags
        hyp_match = re.search(
            r"<hypothesis>(.*?)</hypothesis>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if hyp_match:
            try:
                parsed = yaml.safe_load(hyp_match.group(1).strip())
                if isinstance(parsed, dict):
                    yaml_dict = parsed
            except Exception:
                pass

        if not yaml_dict:
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

        code_str = ""
        # 2. Prefer explicit <policy_code> ... </policy_code> or <policy> ... </policy> XML tags
        pol_match = re.search(
            r"<(?:policy_code|policy)>(.*?)</(?:policy_code|policy)>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if pol_match:
            code_str = pol_match.group(1).strip()
            fence_m = re.search(
                r"```(?:python)?\s*\n(.*?)```",
                code_str,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if fence_m:
                code_str = fence_m.group(1).strip()

        if not code_str:
            py_blocks = re.findall(
                r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
            )
            if not py_blocks:
                py_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
                py_blocks = [
                    b
                    for b in py_blocks
                    if any(
                        tok in b
                        for tok in (
                            "def ",
                            "class ",
                            "ActionLeaf",
                            "ConditionLeaf",
                            "PrioritySelector",
                            "Sequence",
                            "BehaviorTree",
                            "subtree",
                        )
                    )
                ]

            if py_blocks:
                policy_candidates = [
                    b for b in py_blocks if "class Policy" in b or "class Agent" in b
                ]
                if policy_candidates:
                    code_str = policy_candidates[-1].strip()
                else:
                    code_str = py_blocks[-1].strip()

        if code_str:
            code_str = smart_dedent(code_str)

        return yaml_dict, code_str

    @staticmethod
    def extract_skill_code(
        content: str,
        hyp: dict[str, Any],
        fallback_skill_code: str = "",
        fallback_skill_name: str | None = None,
    ) -> str:
        """Extracts external skill module code if specified in hypothesis, XML tags, or markdown."""
        text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)

        # 1. Prefer explicit <skill_code ...> ... </skill_code> or <skill ...> ... </skill> XML tags
        skill_tag_m = re.search(
            r"<(?:skill_code|skill)(?:\s+(?:module|name)=[\"'](?P<mod>[^\"']+)[\"'])?[^>]*>(?P<code>.*?)</(?:skill_code|skill)>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if skill_tag_m:
            mod_attr = skill_tag_m.group("mod")
            if mod_attr and not hyp.get("skill_module"):
                hyp["skill_module"] = mod_attr.strip()
            s_raw = skill_tag_m.group("code").strip()
            fence_m = re.search(
                r"```(?:python)?\s*\n(.*?)```", s_raw, flags=re.DOTALL | re.IGNORECASE
            )
            if fence_m:
                s_raw = fence_m.group(1).strip()
            s_clean = AuthorAgent._sanitize_skill_code(s_raw)
            if s_clean:
                return smart_dedent(s_clean)

        skill_mod = hyp.get("skill_module") or fallback_skill_name
        if skill_mod and "skill_module" not in hyp:
            hyp["skill_module"] = skill_mod
        if not skill_mod:
            return fallback_skill_code

        if hyp.get("skill_code"):
            s_clean = AuthorAgent._sanitize_skill_code(str(hyp["skill_code"]).strip())
            return smart_dedent(s_clean)

        m = re.search(
            rf"```(?:python:)?(?:skill|{re.escape(str(skill_mod))})\s*\n(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            s_clean = AuthorAgent._sanitize_skill_code(m.group(1).strip())
            return smart_dedent(s_clean)

        py_blocks = re.findall(
            r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if len(py_blocks) >= 2:
            for b in py_blocks:
                if (
                    f"module: {skill_mod}" in b.lower()
                    or f"skills/{skill_mod}" in b.lower()
                ):
                    s_clean = AuthorAgent._sanitize_skill_code(b.strip())
                    return smart_dedent(s_clean)
            for b in py_blocks:
                if "class Policy" not in b and ("def " in b or "class " in b):
                    s_clean = AuthorAgent._sanitize_skill_code(b.strip())
                    return smart_dedent(s_clean)

        return fallback_skill_code

    @staticmethod
    def extract_hypotheses_and_reflection(
        text: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Extracts structured <hypotheses> list and <reflection> summary from swarm worker output."""
        hypotheses: list[dict[str, Any]] = []
        reflection = ""

        hyp_m = re.search(
            r"<hypotheses>(.*?)</hypotheses>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if hyp_m:
            raw_h = hyp_m.group(1).strip()
            try:
                parsed = yaml.safe_load(raw_h)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            hypotheses.append(dict(item))
                        else:
                            hypotheses.append({"mechanism": str(item)})
                elif isinstance(parsed, dict):
                    hypotheses.append(dict(parsed))
            except Exception:
                for line in raw_h.splitlines():
                    clean_line = line.strip().lstrip("-* ").strip()
                    if clean_line:
                        hypotheses.append({"mechanism": clean_line})

        ref_m = re.search(
            r"<reflection>(.*?)</reflection>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if ref_m:
            reflection = ref_m.group(1).strip()

        return hypotheses, reflection

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
        elif name == "query_monster":
            m_name = str(args.get("name", ""))
            out, summary = self.tools.query_monster(m_name)
            arg_disp = m_name
        elif name == "query_item":
            i_name = str(args.get("name", ""))
            out, summary = self.tools.query_item(i_name)
            arg_disp = i_name
        elif name == "query_policy_docs":
            topic = str(args.get("topic", "all"))
            out, summary = self.tools.query_policy_docs(topic)
            arg_disp = topic
        elif name == "read_skill":
            skill_name = str(args.get("skill_name", ""))
            out, summary = self.tools.read_skill(skill_name)
            arg_disp = skill_name
        elif name == "query_theories":
            q = str(args.get("query", ""))
            dt = str(args.get("doc_type", ""))
            lim = int(args.get("limit", 6) or 6)
            out, summary = self.tools.query_theories(query=q, doc_type=dt, limit=lim)
            arg_disp = q[:40]
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
                hyp_label = str(args.get("hypothesis", ""))
                out, summary = self.tools.dry_run_policy(
                    code=code,
                    is_macro=is_macro,
                    seed=seed,
                    max_turns=mt,
                    replay_run_id=replay_id,
                    replay_steps=replay_steps,
                    skill_name=str(s_name) if s_name else None,
                    skill_code=str(s_code) if s_code else None,
                    hypothesis=hyp_label,
                )
                arg_disp = f"{len(code.splitlines())} lines, seed={seed}"
                if hyp_label:
                    arg_disp += f", {hyp_label[:28]}"
                if s_name:
                    arg_disp += f", skill={s_name}"
                if replay_id:
                    arg_disp += f", replay={replay_id[:10]}..."
        else:
            out, summary = f"Unknown tool '{name}'.", "unknown"
            arg_disp = str(args)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ui.tool_call(
            turn_idx,
            name,
            arg_disp,
            summary,
            elapsed_ms,
            agent_role=self.agent_role,
        )
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
                with httpx.Client(timeout=60.0) as client:
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
            ) as exc:
                if attempt == 4:
                    raise
                ui.warn(
                    f"[{self.agent_role}] LLM API attempt {attempt + 1}/5 failed "
                    f"({type(exc).__name__}: {str(exc)[:120]}); retrying..."
                )
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

        canon_mode = normalize_mode(mode)
        if canon_mode == "CREATE" or not current_policy.strip():
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
        adj_mons = state.adjacent_monsters(exclude_pets=True)
        if adj_mons:
            m = adj_mons[0]
            return DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
        return self.explore_step(state)

    def run(self, state):
        while True:
            if state.adjacent_monsters(exclude_pets=True):
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
                    agent_role=self.agent_role,
                    wall_ms=1.5,
                )
            except Exception:
                pass
            return SynthesisProposal(
                mode=mode,
                hypothesis=hyp,
                raw_code_patch=code,
                full_policy_code=code,
                full_macros_code=self.tools.macro_lib.load_code(),
                target_functions=["run", "explore_step", "handle_combat"],
                tool_calls_made=2,
            )

        # WRITE / REFACTOR mock
        patch = """def handle_combat(self, state):
    hp_thresh = self.param("retreat_hp", 0.30, 0.1, 0.6)
    if state.hp / max(1, state.max_hp) < hp_thresh and state.turn > 10:
        for item in state.inventory:
            if "healing" in item.name.lower() and item.char == "!":
                return seq(Action.QUAFF, item.letter)
    adj_mons = state.adjacent_monsters(exclude_pets=True)
    if adj_mons:
        m = adj_mons[0]
        return DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
    return self.explore_step(state)
"""
        spliced, changed = splice_functions_into_policy(current_policy, patch)
        hyp = {
            "mode": mode,
            "target_functions": changed,
            "observation_from_db": "Outlier good runs survived melee by quaffing healing potions at low HP.",
            "mechanism": "Adds healing potion quaff check to handle_combat.",
        }
        return SynthesisProposal(
            mode=mode,
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
        max_pilot_runs: int = 8,
        pilot_episodes: int = 30,
        skills: dict[str, str] | None = None,
        recent_discoveries: list[dict[str, Any]] | None = None,
        recent_failures: list[dict[str, Any]] | None = None,
        max_steps: int = 5000,
        stagnation_function: str = "",
    ) -> SynthesisProposal:
        """
        Runs a fresh, stateless multi-turn tool-calling session for `generation` and `mode`.
        Supports multi-turn pilot evaluations against `baseline_runs` using `pool` inside
        `dry_run_policy` (up to `max_pilot_runs` revisions), plus compiler-in-the-loop self-repair.
        """
        if self.provider == "mock":
            return self._synthesize_mock(mode=mode, current_policy=current_policy)

        canon_mode = normalize_mode(mode)

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
            mode=canon_mode,
        )
        if max_tool_turns is None:
            max_tool_turns = (
                24 if canon_mode == "CREATE" else max(18, max_pilot_runs + 10)
            )

        sys_prompt = build_system_prompt(self.db_path)
        user_prompt = build_user_prompt(
            mode=mode,
            generation=generation,
            current_policy=current_policy,
            macros_docs="",
            batch_digest=batch_digest,
            recent_experiments=recent_experiments,
            skills=skills,
            recent_discoveries=recent_discoveries,
            recent_failures=recent_failures,
            stagnation_function=stagnation_function,
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
            if self.agent_role in ("", "author"):
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
                # In WRITE and REFACTOR modes, Turn 1 is strictly reserved for Empirical Research.
                # If candidate code is output on Turn 1 without having called query_db or query_wiki, block auto-pilot and enforce research.
                if (
                    turn_idx == 1
                    and canon_mode in ("WRITE", "REFACTOR")
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
                ):
                    total_tool_calls += 1
                    t0 = time.perf_counter()
                    auto_s_name = (
                        str((latest_hyp or {}).get("skill_module", ""))
                        or getattr(self.tools, "last_passed_skill_name", None)
                        or getattr(self.tools, "best_pilot_skill_name", None)
                    )
                    auto_s_code = self.extract_skill_code(
                        clean_text,
                        latest_hyp or {},
                        fallback_skill_code=(
                            getattr(self.tools, "last_passed_skill_code", None)
                            or getattr(self.tools, "best_pilot_skill_code", None)
                            or ""
                        ),
                        fallback_skill_name=auto_s_name,
                    )
                    pilot_res, pilot_summary = self.tools.dry_run_policy(
                        code=found_code,
                        skill_name=auto_s_name if auto_s_name else None,
                        skill_code=auto_s_code if auto_s_code else None,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    arg_disp = f"auto-pilot ({len(found_code.splitlines())} lines)"
                    if auto_s_name:
                        arg_disp += f", skill={auto_s_name}"
                    ui.tool_call(
                        turn_idx,
                        "dry_run_policy",
                        arg_disp,
                        pilot_summary,
                        elapsed_ms,
                        agent_role=self.agent_role,
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
                            skill_name=str(auto_s_name or ""),
                            skill_code=str(auto_s_code or ""),
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
                            agent_role=self.agent_role,
                            wall_ms=turn_ms,
                        )
                    except Exception:
                        pass

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
                    and canon_mode == "WRITE"
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
                if t_name in ("query_db", "query_wiki", "query_monster", "query_item"):
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

            if (
                turn_idx >= 4
                and self.tools.pilot_runs_used == 0
                and not any(
                    (tc.get("function") or {}).get("name") == "dry_run_policy"
                    for tc in tool_calls
                )
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"⚠️ Budget Notice: You are on Turn {turn_idx}/{max_tool_turns}. "
                            "You have completed empirical research. Please synthesize your candidate code now and call `dry_run_policy(code=...)` "
                            "to verify your hypothesis before tool turns run out."
                        ),
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
                    agent_role=self.agent_role,
                    wall_ms=turn_ms,
                )
            except Exception:
                pass

        # Pilot Gate: If pilots were run during the tool phase and none passed the strict sign gate:
        if (
            self.tools.pilot_runs_used > 0
            and not self.tools.last_pilot_improved
            and canon_mode not in ("CREATE", "SWARM_WORKER", "SWARM_META")
        ):
            summary = self.tools.last_pilot_summary or "insufficient signal"
            best_delta = getattr(self.tools, "best_pilot_delta", -1.0)
            if (
                getattr(self, "pilot_fallback", "best") == "best"
                and self.tools.best_pilot_code
                and best_delta >= 0.0
            ):
                ui.info(
                    f"Pilot Gate: Revisions exhausted without strict sign pass ({summary}). "
                    f"Proceeding to full evaluation with best non-negative pilot candidate (Δ={best_delta:+.2f})."
                )
            else:
                abort_detail = (
                    f"all {self.tools.pilot_runs_used} pilot revisions regressed (best Δ={best_delta:+.2f} < 0.0)"
                    if self.tools.best_pilot_code and best_delta < 0.0
                    else f"failed all {self.tools.pilot_runs_used} pilot revisions ({summary})"
                )
                ui.warn(f"Pilot Gate: Candidate {abort_detail}. Aborting rollout.")
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
                        message_content=abort_detail,
                        tool_calls_json="[]",
                        tool_results_json="[]",
                        candidate_code=self.tools.last_attempted_pilot_code or "",
                        skill_name=str((latest_hyp or {}).get("skill_module", "")),
                        skill_code=getattr(self.tools, "last_passed_skill_code", ""),
                        hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                        status="PILOT_FAILED",
                        error_diagnostic=abort_detail,
                        prompt_sent="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        reasoning_tokens=0,
                        agent_role=self.agent_role,
                        wall_ms=0.0,
                    )
                except Exception:
                    pass

                return SynthesisProposal(
                    mode=mode,
                    hypothesis=latest_hyp,
                    raw_code_patch="",
                    full_policy_code=current_policy,
                    full_macros_code="",
                    target_functions=[],
                    tool_calls_made=total_tool_calls,
                    skill_code="",
                    aborted=True,
                    abort_reason=f"PILOT_GATE_ABORT: {abort_detail}",
                    pilot_best_delta=best_delta,
                )

        # Compiler-in-the-loop verification & sequential self-repair
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
            elif (
                self.tools.best_pilot_code
                and (not code_patch or not self.tools.last_pilot_improved)
                and getattr(self.tools, "best_pilot_delta", -1.0) >= 0.0
            ):
                code_patch = self.tools.best_pilot_code

            if not code_patch:
                diag = "No <policy_code> or ```python ... ``` code block found in your response. Please emit your YAML hypothesis and Python code block."
                status = "MISSING_CODE"
            else:
                s_name = (
                    (hyp or latest_hyp).get("skill_module")
                    or getattr(self.tools, "last_passed_skill_name", None)
                    or getattr(self.tools, "best_pilot_skill_name", None)
                )
                s_code = self.extract_skill_code(
                    final_content,
                    hyp or latest_hyp,
                    fallback_skill_code=(
                        getattr(self.tools, "last_passed_skill_code", None)
                        or getattr(self.tools, "best_pilot_skill_code", None)
                        or ""
                    ),
                    fallback_skill_name=s_name,
                )
                diag, status = self.tools.dry_run_policy(
                    code=code_patch,
                    is_macro=False,
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
                    agent_role=self.agent_role,
                    wall_ms=0.0,
                )
            except Exception:
                pass

            if status.startswith("PASSED") or status.startswith("VALID"):
                skill_code = self.extract_skill_code(
                    final_content,
                    hyp,
                    fallback_skill_code=(
                        getattr(self.tools, "last_passed_skill_code", None)
                        or getattr(self.tools, "best_pilot_skill_code", None)
                        or ""
                    ),
                    fallback_skill_name=(
                        getattr(self.tools, "last_passed_skill_name", None)
                        or getattr(self.tools, "best_pilot_skill_name", None)
                    ),
                )
                if skill_code and skill_code.strip():
                    try:
                        validate_python_code(
                            skill_code,
                            filename=f"skills/{hyp.get('skill_module', 'skill')}.py",
                        )
                    except Exception as val_err:
                        diag = f"SyntaxError in skill code: {val_err}"
                        status = "SYNTAX_ERROR"
                        if repair_idx < self.max_repair_turns:
                            continue
                        return SynthesisProposal(
                            mode=mode,
                            hypothesis=hyp,
                            raw_code_patch=code_patch,
                            full_policy_code=current_policy,
                            full_macros_code="",
                            target_functions=hyp.get("target_functions") or ["run"],
                            tool_calls_made=total_tool_calls,
                            skill_code="",
                            aborted=True,
                            abort_reason=f"SKILL_SYNTAX_ERROR: {val_err}",
                            pilot_best_delta=getattr(
                                self.tools, "best_pilot_delta", 0.0
                            ),
                        )
                if (
                    canon_mode in ("CREATE", "SWARM_WORKER", "SWARM_META")
                    or not current_policy.strip()
                ):
                    spliced_policy = code_patch
                    default_target = (
                        ["step"]
                        if canon_mode == "SWARM_META"
                        else (["subtree"] if canon_mode == "SWARM_WORKER" else ["run"])
                    )
                    changed_fns = hyp.get("target_functions") or default_target
                else:
                    spliced_policy, changed_fns = splice_functions_into_policy(
                        current_policy, code_patch
                    )
                    if canon_mode == "REFACTOR" and skill_code:
                        spliced_policy, pruned_fns = prune_extracted_skill_functions(
                            spliced_policy, skill_code
                        )
                        if pruned_fns:
                            changed_fns = [
                                fn for fn in changed_fns if fn not in set(pruned_fns)
                            ] or ["run"]
                if isinstance(changed_fns, str):
                    changed_fns = [changed_fns]
                if (
                    canon_mode
                    not in ("CREATE", "REFACTOR", "SWARM_WORKER", "SWARM_META")
                    and len(changed_fns) > 2
                ):
                    ui.warn(
                        f"Mutation Scope: Candidate modified {len(changed_fns)} functions ({changed_fns}), exceeding limit of 2. Aborting rollout."
                    )
                    return SynthesisProposal(
                        mode=mode,
                        hypothesis=hyp,
                        raw_code_patch=code_patch,
                        full_policy_code=current_policy,
                        full_macros_code="",
                        target_functions=list(changed_fns),
                        tool_calls_made=total_tool_calls,
                        skill_code=skill_code,
                        aborted=True,
                        abort_reason=f"MUTATION_SCOPE_EXCEEDED: modified {len(changed_fns)} functions ({changed_fns}), limit is 2",
                        pilot_best_delta=getattr(self.tools, "best_pilot_delta", 0.0),
                    )
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
                repair_msg = self._call_llm_api(messages, allow_tools=True)
                if repair_msg.get("tool_calls"):
                    tcs = repair_msg.get("tool_calls") or []
                    messages.append(
                        {
                            "role": "assistant",
                            "content": repair_msg.get("content") or "",
                            "tool_calls": tcs,
                        }
                    )
                    for idx_tc, tc in enumerate(tcs, start=1):
                        res_out = self._dispatch_raw_tool_call(
                            max_tool_turns + repair_idx + 1, tc
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", f"rep_tc_{idx_tc}"),
                                "content": res_out,
                            }
                        )
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
                        agent_role=self.agent_role,
                        wall_ms=rep_ms,
                    )
                except Exception:
                    pass

        # Final fallback: if the model already passed dry_run_policy during the tool phase, use that verified code!
        best_code = self.tools.last_passed_code or (
            self.tools.best_pilot_code
            if getattr(self.tools, "best_pilot_delta", -1.0) >= 0.0
            else None
        )
        if best_code:
            ui.info(
                "Using verified code from earlier passing/best dry_run_policy call."
            )
            code_patch = best_code
            skill_code = self.extract_skill_code(
                final_content,
                latest_hyp,
                fallback_skill_code=(
                    getattr(self.tools, "last_passed_skill_code", None)
                    or getattr(self.tools, "best_pilot_skill_code", None)
                    or ""
                ),
                fallback_skill_name=(
                    getattr(self.tools, "last_passed_skill_name", None)
                    or getattr(self.tools, "best_pilot_skill_name", None)
                ),
            )
            if (
                canon_mode in ("CREATE", "SWARM_WORKER", "SWARM_META")
                or not current_policy.strip()
            ):
                spliced_policy = code_patch
                default_target = (
                    ["step"]
                    if canon_mode == "SWARM_META"
                    else (["subtree"] if canon_mode == "SWARM_WORKER" else ["run"])
                )
                changed_fns = latest_hyp.get("target_functions") or default_target
            else:
                spliced_policy, changed_fns = splice_functions_into_policy(
                    current_policy, code_patch
                )
                if canon_mode == "REFACTOR" and skill_code:
                    spliced_policy, pruned_fns = prune_extracted_skill_functions(
                        spliced_policy, skill_code
                    )
                    if pruned_fns:
                        changed_fns = [
                            fn for fn in changed_fns if fn not in set(pruned_fns)
                        ] or ["run"]
            if isinstance(changed_fns, str):
                changed_fns = [changed_fns]
            best_delta = (
                float(self.tools.best_pilot_delta)
                if getattr(self, "tools", None) and self.tools.best_pilot_delta > -1e8
                else 0.0
            )
            if (
                canon_mode not in ("CREATE", "REFACTOR", "SWARM_WORKER", "SWARM_META")
                and len(changed_fns) > 2
            ):
                ui.warn(
                    f"Mutation Scope: Candidate modified {len(changed_fns)} functions ({changed_fns}), exceeding limit of 2. Aborting rollout."
                )
                return SynthesisProposal(
                    mode=mode,
                    hypothesis=latest_hyp,
                    raw_code_patch=code_patch,
                    full_policy_code=current_policy,
                    full_macros_code="",
                    target_functions=list(changed_fns),
                    tool_calls_made=total_tool_calls,
                    skill_code=skill_code,
                    aborted=True,
                    abort_reason=f"MUTATION_SCOPE_EXCEEDED: modified {len(changed_fns)} functions ({changed_fns}), limit is 2",
                    pilot_best_delta=best_delta,
                )
            return SynthesisProposal(
                mode=mode,
                hypothesis=latest_hyp,
                raw_code_patch=code_patch,
                full_policy_code=spliced_policy,
                full_macros_code=self.tools.macro_lib.load_code(),
                target_functions=list(changed_fns),
                tool_calls_made=total_tool_calls,
                skill_code=skill_code,
                pilot_best_delta=best_delta,
            )

        raise RuntimeError(
            f"Author Agent failed to produce valid compiling code after {self.max_repair_turns} repair attempts. Last error: {diag}"
        )

    def _dispatch_raw_tool_call(self, turn_idx: int, tc: dict[str, Any]) -> str:
        """Parses an OpenAI-format tool call dict and executes it via `_execute_tool_call`."""
        fn_spec = tc.get("function") or {}
        t_name = str(fn_spec.get("name") or "")
        raw_args = fn_spec.get("arguments", "{}")
        try:
            t_args = (
                json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            )
        except Exception:
            t_args = {}
        return self._execute_tool_call(turn_idx, t_name, t_args)

    def run_tool_session(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tool_turns: int = 5,
        required_tools: list[str] | None = None,
    ) -> str:
        """Runs a lightweight tool-enabled conversation and returns the final assistant text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_content = ""
        called_tools: set[str] = set()
        req = [str(t) for t in (required_tools or []) if str(t).strip()]
        for turn_idx in range(1, max(1, int(max_tool_turns)) + 2):
            allow_tools = turn_idx <= max_tool_turns
            msg = self._call_llm_api(messages, allow_tools=allow_tools)
            content = str(msg.get("content") or "").strip()
            if content:
                last_content = content
            tcs = msg.get("tool_calls") or []
            if not tcs:
                missing = [t for t in req if t not in called_tools]
                if missing and turn_idx <= max_tool_turns:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Required tool(s) not yet called: {', '.join(missing)}. "
                                f"You MUST call `{missing[0]}` first to research authoritative "
                                "NetHack mechanics before returning your final JSON response."
                            ),
                        }
                    )
                    continue
                return last_content
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tcs,
                }
            )
            for idx_tc, tc in enumerate(tcs, start=1):
                fn_name = str((tc.get("function") or {}).get("name") or "")
                if fn_name:
                    called_tools.add(fn_name)
                res_out = self._dispatch_raw_tool_call(turn_idx, tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"tc_{turn_idx}_{idx_tc}"),
                        "content": res_out,
                    }
                )
        return last_content

    def run_swarm_worker_session(
        self,
        generation: int,
        brief: str,
        current_policy: str = "",
        num_hypotheses: int = 3,
        checkpoint: Any = None,
        frontier: Any = None,
        replay_fixtures: list[dict[str, Any]] | None = None,
        scaffold_code: str = "",
        skills: dict[str, str] | None = None,
        max_tool_turns: int | None = None,
    ) -> dict[str, Any]:
        """
        Runs a single warm conversation for a SwarmWorker:
        1. Empirical audit (tools stay active across turns)
        2. Brainstorm `num_hypotheses` distinct mechanisms in <hypotheses>
        3. Sequentially translate & test each hypothesis via `dry_run_policy(..., hypothesis=...)`
        4. Emit <reflection> comparing tested candidates + winning `<policy_code>` subtree
        """
        k_hyp = max(1, int(num_hypotheses))
        self.tools.configure_generation(
            pool=None,
            baseline_runs=[],
            max_pilot_runs=max(12, k_hyp * 4),
            pilot_episodes=1,
            skills=skills,
            generation=generation,
            benchmark=self.benchmark,
            mode="SWARM_WORKER",
            swarm_checkpoint=checkpoint,
            swarm_frontier=frontier,
            swarm_replay_fixtures=replay_fixtures,
            swarm_scaffold_code=scaffold_code,
        )
        self.tools.current_policy_code = scaffold_code or current_policy

        if self.provider == "mock":
            self.tools.query_db("SELECT COUNT(*) FROM runs")
            mock_hyps = [
                {
                    "id": "H1",
                    "mechanism": "ActionLeaf wait fallback probe (non-triggering baseline check)",
                    "trigger_condition": "Never fires",
                },
                {
                    "id": "H2",
                    "mechanism": "Move step right along passable terrain",
                    "trigger_condition": "Adjacent east tile passable",
                },
                {
                    "id": "H3",
                    "mechanism": "Frontier BFS exploration step toward unexplored tiles",
                    "trigger_condition": "Unvisited walkable tile reachable",
                },
            ][:k_hyp]
            mock_codes = [
                "def leaf_noop(state, bb):\n    return None\nsubtree = ActionLeaf(leaf_noop, name='noop')",
                "def leaf_east(state, bb):\n    return Action.E if state.passable_mask[state.y, min(78, state.x + 1)] else None\nsubtree = ActionLeaf(leaf_east, name='east')",
                "def leaf_explore(state, bb):\n    if not hasattr(bb, 'seen'):\n        bb.seen = set()\n    bb.seen.add((state.y, state.x))\n    for d, (dy, dx) in [(Action.E, (0, 1)), (Action.S, (1, 0)), (Action.W, (0, -1)), (Action.N, (-1, 0))]:\n        ny, nx = state.y + dy, state.x + dx\n        if 0 <= ny < 21 and 0 <= nx < 79 and state.passable_mask[ny, nx] and (ny, nx) not in bb.seen:\n            return d\n    return None\nsubtree = ActionLeaf(leaf_explore, name='explore')",
            ][:k_hyp]
            for h_dict, h_code in zip(mock_hyps, mock_codes):
                self.tools.dry_run_policy(
                    code=h_code,
                    hypothesis=f"{h_dict['id']}: {h_dict['mechanism']}",
                )
            winning_code = mock_codes[-1]
            reflection = (
                f"winner: {mock_hyps[-1]['id']}\n"
                f"tested: {len(mock_hyps)} hypotheses cleanly in warm mock session."
            )
            return {
                "hypotheses": mock_hyps,
                "reflection": reflection,
                "experiments": list(self.tools.swarm_experiments),
                "final_code": winning_code,
                "raw_response": f"<reflection>\n{reflection}\n</reflection>\n<policy_code>\n{winning_code}\n</policy_code>",
                "tool_calls_made": len(mock_hyps) + 1,
            }

        sys_prompt = build_system_prompt(self.db_path)
        user_prompt = build_user_prompt(
            mode="SWARM_WORKER",
            generation=generation,
            current_policy=current_policy or scaffold_code,
            batch_digest=brief,
            skills=skills,
            num_hypotheses=k_hyp,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        turn_limit = max_tool_turns or max(6, k_hyp * 2 + 2)
        total_tool_calls = 0
        parsed_hypotheses: list[dict[str, Any]] = []
        reflection_text = ""
        final_content = ""
        last_valid_code = ""
        nudges_used = 0
        batch_id = f"swarm_g{generation}_{self.agent_role}"

        for turn_idx in range(1, turn_limit + 2):
            allow_tools = turn_idx <= turn_limit
            t0_turn = time.perf_counter()
            msg = self._call_llm_api(messages, allow_tools=allow_tools)
            turn_ms = (time.perf_counter() - t0_turn) * 1000.0
            msg_text = msg.get("content") or ""
            reasoning_text = msg.get("_reasoning") or ""
            usage = msg.get("_usage") or {}

            clean_text = re.sub(
                r"<think>.*?</think>", "", msg_text, flags=re.DOTALL
            ).strip()
            clean_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
            if msg.get("tool_calls"):
                clean_msg["tool_calls"] = msg["tool_calls"]
            messages.append(clean_msg)

            if clean_text:
                final_content = clean_text
                turn_hyps, turn_ref = self.extract_hypotheses_and_reflection(clean_text)
                if turn_hyps:
                    parsed_hypotheses = turn_hyps
                if turn_ref:
                    reflection_text = turn_ref
                _, extracted_code = self.extract_yaml_and_python(clean_text)
                if extracted_code:
                    last_valid_code = extracted_code

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                _, extracted_code = self.extract_yaml_and_python(clean_text)
                distinct_hyp_count = len(
                    {
                        str(e.get("hypothesis", f"exp_{i}"))
                        for i, e in enumerate(self.tools.swarm_experiments)
                    }
                )
                if (
                    extracted_code
                    and not self.tools.swarm_experiments
                    and turn_idx <= turn_limit
                ):
                    total_tool_calls += 1
                    t0 = time.perf_counter()
                    pilot_res, pilot_summary = self.tools.dry_run_policy(
                        code=extracted_code,
                        hypothesis="H1: emitted subtree",
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    ui.tool_call(
                        turn_idx,
                        "dry_run_policy",
                        f"auto-eval ({len(extracted_code.splitlines())} lines)",
                        pilot_summary,
                        elapsed_ms,
                        agent_role=self.agent_role,
                    )
                    distinct_hyp_count = len(self.tools.swarm_experiments)

                min_required = min(k_hyp, 2) if k_hyp >= 2 else 1
                if (
                    distinct_hyp_count < min_required
                    and nudges_used < 2
                    and turn_idx < turn_limit
                ):
                    nudges_used += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You have empirically tested {distinct_hyp_count} hypothesis/hypotheses via `dry_run_policy` so far, "
                                f"but this session expects {k_hyp} competing hypotheses (`H1`..`H{k_hyp}`). "
                                "Please formulate and test the remaining distinct hypothesis/hypotheses via "
                                "`dry_run_policy(code=..., hypothesis='H...: ...')` before emitting `<reflection>` and the winning `<policy_code>`."
                            ),
                        }
                    )
                    continue
                break

            tool_results_list: list[dict[str, str]] = []
            for tc in tool_calls:
                total_tool_calls += 1
                fn_obj = tc.get("function", {})
                t_name = fn_obj.get("name", "")
                raw_args = fn_obj.get("arguments", "{}")
                try:
                    t_args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except Exception:
                    t_args = {}
                t_out = self._execute_tool_call(turn_idx, t_name, t_args or {})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                        "content": t_out,
                    }
                )
                tool_results_list.append({"name": t_name, "result": t_out[:2000]})

            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_sw_{generation}_{self.agent_role}_{turn_idx}_{int(time.time() * 1000)}",
                    generation=generation,
                    batch_id=batch_id,
                    mode="SWARM_WORKER",
                    turn_index=turn_idx,
                    turn_type="WARM_HYPOTHESIS_TURN",
                    repair_idx=0,
                    model=self.model,
                    thinking_content=reasoning_text,
                    message_content=clean_text,
                    tool_calls_json=json.dumps(tool_calls),
                    tool_results_json=json.dumps(tool_results_list),
                    candidate_code=last_valid_code,
                    status="TOOL_CALLS",
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    reasoning_tokens=int(
                        usage.get("completion_tokens_details", {}).get(
                            "reasoning_tokens", 0
                        )
                    ),
                    agent_role=self.agent_role,
                    wall_ms=turn_ms,
                )
            except Exception:
                pass

        chosen_code = last_valid_code or self.tools.last_passed_code or ""
        return {
            "hypotheses": parsed_hypotheses,
            "reflection": reflection_text,
            "experiments": list(self.tools.swarm_experiments),
            "final_code": chosen_code,
            "raw_response": final_content,
            "tool_calls_made": total_tool_calls,
        }
