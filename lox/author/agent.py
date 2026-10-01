"""
LOX 2.0 Author Agent: Asynchronous LLM policy synthesis engine with DuckDB analytical tooling.
Queries LLMs across multiple providers (Gemini, OpenRouter, local vLLM, or Mock),
enables interactive empirical data investigation (ReAct),
logs token usage directly into DuckDB, and validates proposed policies against AST invariants.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.tools import DuckDBToolRegistry, OPENAI_TOOL_SPECS
from lox.telemetry.tokens import log_token_usage
from lox.dsl.compiler import compile_policy
from lox.core.tree import BehaviorTree


class AuthorAgent:
    """Manages LLM policy authoring sessions with DuckDB analytical tools and token tracking."""

    def __init__(
        self,
        provider: str = "mock",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        db_path: str = "data/lox.duckdb",
        max_tool_turns: int = 50,
    ):
        self.provider = provider
        self.model = model
        if api_key:
            self.api_key = api_key
        elif provider == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        elif provider == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY")
        else:
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.base_url = base_url
        self.db_path = db_path
        self.max_tool_turns = max_tool_turns
        self.tools = DuckDBToolRegistry(db_path=db_path)

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatches tool execution against the DuckDBToolRegistry with robust error shielding."""
        try:
            if tool_name == "query_duckdb":
                return self.tools.query_duckdb(args.get("sql", ""))
            elif tool_name == "get_duckdb_schema":
                return self.tools.get_duckdb_schema()
            elif tool_name == "get_death_taxonomy":
                return self.tools.get_death_taxonomy(window=args.get("window", 20))
            elif tool_name == "get_floor_pacing_stats":
                return self.tools.get_floor_pacing_stats(depth=args.get("depth", 1))
            elif tool_name == "get_action_distribution":
                return self.tools.get_action_distribution(run_id=args.get("run_id", ""))
            elif tool_name == "query_wiki":
                return self.tools.query_wiki(args.get("query", ""), top_k=args.get("top_k", 2))
            elif tool_name == "request_macro":
                run_id = getattr(self, "_current_run_id", "synth_session")
                return self.tools.request_macro(
                    macro_name=args.get("macro_name", ""),
                    rationale=args.get("rationale", ""),
                    proposed_interface=args.get("proposed_interface", ""),
                    priority=args.get("priority", "medium"),
                    run_id=run_id,
                )
            return f"Unknown tool '{tool_name}'."
        except Exception as e:
            return f"Tool execution error for '{tool_name}': {e}"

    def _call_mock(
        self,
        current_policy: str,
        trigger_reason: str,
        run_id: str,
        session_id: str,
    ) -> str:
        """Deterministic mock provider with token tracking and tool simulation."""
        # Simulate tool execution for testing
        tools_called = ["query_duckdb", "query_wiki"]
        self._execute_tool("get_duckdb_schema", {})
        self._execute_tool("query_wiki", {"query": "Excalibur"})

        log_token_usage(
            run_id=run_id,
            session_id=session_id,
            provider="mock",
            model="mock-author-v2",
            prompt_tokens=260,
            completion_tokens=140,
            trigger_reason=trigger_reason,
            tools_called=tools_called,
            db_path=self.db_path,
        )

        return """```python
def emergency_recovery():
    if hp_frac < 0.25:
        pray()
    if hp_frac < 0.50:
        quaff_healing()
    if hunger_state >= HUNGRY:
        eat_carried_food()

def combat():
    if adjacent_hostile:
        melee_attack_hostile()

def descend():
    if stairs_down_known:
        step_to_stairs_down()

def explore():
    if has_unvisited_frontier:
        step_to_frontier()
    else:
        search()

plan = [
    emergency_recovery,
    combat,
    descend,
    explore,
]
```"""

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        run_id: str,
        session_id: str,
        trigger_reason: str,
    ) -> str:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        model_name = self.model or "gemini-2.5-flash"

        # Provide python functions to Gemini for automatic tool execution
        tool_callables = [
            self.tools.query_duckdb,
            self.tools.get_duckdb_schema,
            self.tools.get_death_taxonomy,
            self.tools.get_floor_pacing_stats,
            self.tools.get_action_distribution,
            self.tools.query_wiki,
            self.tools.request_macro,
        ]

        # Use chats for automatic multi-turn tool calling
        chat = client.chats.create(
            model=model_name,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.2,
                "tools": tool_callables,
            },
        )
        response = chat.send_message(user_prompt)

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        log_token_usage(
            run_id=run_id,
            session_id=session_id,
            provider="gemini",
            model=model_name,
            prompt_tokens=prompt_tokens or 350,
            completion_tokens=completion_tokens or 200,
            trigger_reason=trigger_reason,
            db_path=self.db_path,
        )

        return response.text or ""

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        run_id: str,
        session_id: str,
        trigger_reason: str,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                f"Missing API key for provider '{self.provider}'. "
                f"Please pass --api-key or set {self.provider.upper()}_API_KEY in your environment or .env file."
            )

        url = (self.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "HTTP-Referer": "https://github.com/saejin-moon/lox",
            "X-Title": "LOX 2.0 Policy Synthesis",
        }
        model_name = self.model or "google/gemini-2.5-flash"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        tools_invoked = []

        with httpx.Client(timeout=120.0) as client:
            use_tools = True
            max_tool_turns = self.max_tool_turns  # Configurable ceiling (default 50) allowing extensive empirical exploration
            for turn in range(max_tool_turns):
                payload: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if use_tools:
                    payload["tools"] = OPENAI_TOOL_SPECS

                resp = client.post(url, headers=headers, json=payload)
                if resp.is_error:
                    err_msg = resp.text
                    try:
                        err_json = resp.json()
                        if "error" in err_json:
                            err_info = err_json["error"]
                            if isinstance(err_info, dict):
                                err_msg = err_info.get("message", err_msg)
                            else:
                                err_msg = str(err_info)
                    except Exception:
                        pass

                    # If model doesn't support tools, fallback to no-tools
                    if use_tools and resp.status_code == 400 and ("tool" in err_msg.lower() or "not supported" in err_msg.lower()):
                        use_tools = False
                        continue

                    raise RuntimeError(
                        f"OpenRouter API error ({resp.status_code}): {err_msg}"
                    )

                data = resp.json()

                usage = data.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

                choice = data["choices"][0]
                msg = choice["message"]
                messages.append(msg)

                # Check if tool was called
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn_name = tc["function"]["name"]
                        fn_args = json.loads(tc["function"].get("arguments", "{}"))
                        tools_invoked.append(fn_name)
                        result_text = self._execute_tool(fn_name, fn_args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_text,
                        })
                else:
                    # Model provided text/code without requesting more tools
                    break
            else:
                # If tool budget ceiling was reached without dynamic exit, prompt for final code synthesis
                messages.append({
                    "role": "user",
                    "content": (
                        "You have concluded your empirical tool investigation. "
                        "Based on all telemetry, incident logs, and knowledge retrieved above, "
                        "synthesize your complete revised policy program now in a single ```python ... ``` block."
                    ),
                })
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                }
                resp = client.post(url, headers=headers, json=payload)
                if not resp.is_error:
                    data = resp.json()
                    usage = data.get("usage", {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    choice = data["choices"][0]
                    msg = choice["message"]
                    messages.append(msg)

        log_token_usage(
            run_id=run_id,
            session_id=session_id,
            provider="openrouter",
            model=model_name,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            trigger_reason=trigger_reason,
            tools_called=tools_invoked,
            db_path=self.db_path,
        )

        # Store context messages for potential repair turns
        self._last_messages = messages
        self._last_model_name = model_name

        final_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                final_content = m["content"]
                break
        return str(final_content)

    def _repair_code(
        self,
        candidate_code: str,
        compile_err: str,
        run_id: str,
        session_id: str,
        trigger_reason: str,
    ) -> str:
        """Prompts the LLM to fix syntax or vocabulary violations with AST feedback."""
        repair_user_msg = (
            f"The candidate policy program failed AST validation and compilation with error:\n"
            f"{compile_err}\n\n"
            f"Candidate Code:\n```python\n{candidate_code.strip()}\n```\n\n"
            "Please fix all syntax and vocabulary violations to adhere strictly to the Approved Vocabulary and grammar rules. "
            "Output ONLY the complete corrected policy code in a single ```python ... ``` block."
        )

        if self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            url = (self.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "HTTP-Referer": "https://github.com/saejin-moon/lox",
                "X-Title": "LOX 2.0 Policy Synthesis",
            }
            messages = getattr(self, "_last_messages", [])
            messages.append({"role": "user", "content": repair_user_msg})
            model_name = getattr(self, "_last_model_name", self.model or "google/gemini-2.5-flash")

            with httpx.Client(timeout=120.0) as client:
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                }
                resp = client.post(url, headers=headers, json=payload)
                if not resp.is_error:
                    data = resp.json()
                    usage = data.get("usage", {})
                    log_token_usage(
                        run_id=run_id,
                        session_id=session_id,
                        provider="openrouter",
                        model=model_name,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        trigger_reason=f"repair: {trigger_reason}",
                        db_path=self.db_path,
                    )
                    choice = data["choices"][0]
                    msg = choice["message"]
                    messages.append(msg)
                    return str(msg.get("content") or "")
        return candidate_code

    def extract_code(self, response_text: str) -> str:
        """Robustly extracts the Python policy code block from markdown."""
        # 1. Search all code blocks, prioritizing the ones containing policy structure
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
        for block in reversed(blocks):
            if "def " in block and "plan" in block:
                return block.strip()
        for block in blocks:
            if "def " in block:
                return block.strip()

        # 2. Unclosed python code block containing policy code
        match_unclosed = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", response_text, re.DOTALL)
        if match_unclosed:
            code = match_unclosed.group(1).strip()
            if "def " in code:
                return re.sub(r"```+$", "", code).strip()

        # 3. No backticks: find from first 'def ' to end of 'plan = [...]'
        match_def = re.search(r"(def \w+\(.*?\nplan\s*=\s*\[.*?\])", response_text, re.DOTALL)
        if match_def:
            return match_def.group(1).strip()

        match_any_def = re.search(r"(def \w+\(.*)", response_text, re.DOTALL)
        if match_any_def:
            return match_any_def.group(1).strip()

        return response_text.strip()

    def synthesize_policy(
        self,
        current_policy: str,
        trigger_reason: str,
        status_report: str = "",
        run_id: str = "synth_run",
        action_handlers: dict | None = None,
        max_repairs: int = 2,
    ) -> tuple[str, BehaviorTree | None, str | None]:
        """
        Runs one authoring session with tool calling, AST compilation, and self-repair retries.
        Returns (new_code, compiled_tree_or_None, error_message_or_None).
        """
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self._current_run_id = run_id
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(current_policy, trigger_reason, status_report)

        if self.provider == "gemini":
            raw_response = self._call_gemini(system_prompt, user_prompt, run_id, session_id, trigger_reason)
        elif self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            raw_response = self._call_openai_compatible(system_prompt, user_prompt, run_id, session_id, trigger_reason)
        else:
            raw_response = self._call_mock(current_policy, trigger_reason, run_id, session_id)

        # Compilation and Self-Repair Loop
        candidate_code = self.extract_code(raw_response)
        last_error = None

        for attempt in range(max_repairs + 1):
            if "def " in candidate_code and "plan" in candidate_code:
                try:
                    tree = compile_policy(candidate_code, action_handlers=action_handlers)
                    return candidate_code, tree, None
                except Exception as e:
                    last_error = f"Policy compilation failed: {e}"
            else:
                last_error = "Policy compilation failed: Extracted response does not contain valid policy code structure ('def' and 'plan' missing)."

            # If compilation failed and repair attempts remain, ask the model to self-correct
            if attempt < max_repairs and self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
                print(f"[Author Agent] Compilation issue detected ({last_error}). Triggering self-repair turn {attempt + 1}/{max_repairs}...")
                repaired_response = self._repair_code(
                    candidate_code=candidate_code,
                    compile_err=last_error,
                    run_id=run_id,
                    session_id=session_id,
                    trigger_reason=trigger_reason,
                )
                candidate_code = self.extract_code(repaired_response)

        return candidate_code, None, last_error
