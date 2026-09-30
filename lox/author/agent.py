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
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.db_path = db_path
        self.tools = DuckDBToolRegistry(db_path=db_path)

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatches tool execution against the DuckDBToolRegistry."""
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
        return f"Unknown tool '{tool_name}'."

    def _call_mock(
        self,
        current_policy: str,
        trigger_reason: str,
        run_id: str,
        session_id: str,
    ) -> str:
        """Deterministic mock provider with token tracking and tool simulation."""
        # Simulate tool execution for testing
        tools_called = ["query_duckdb"]
        self._execute_tool("get_duckdb_schema", {})

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
        url = (self.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        model_name = self.model or "google/gemini-2.5-flash"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        tools_invoked = []

        with httpx.Client(timeout=120.0) as client:
            for _ in range(5):  # Max 5 tool turns
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "tools": OPENAI_TOOL_SPECS,
                    "temperature": 0.2,
                }
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
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
                    break

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

        final_content = messages[-1].get("content") or ""
        return str(final_content)

    def extract_code(self, response_text: str) -> str:
        """Extracts the Python code block from markdown."""
        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    def synthesize_policy(
        self,
        current_policy: str,
        trigger_reason: str,
        status_report: str = "",
        run_id: str = "synth_run",
        action_handlers: dict | None = None,
    ) -> tuple[str, BehaviorTree | None, str | None]:
        """
        Runs one authoring session with tool calling and token logging.
        Returns (new_code, compiled_tree_or_None, error_message_or_None).
        """
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(current_policy, trigger_reason, status_report)

        if self.provider == "gemini":
            raw_response = self._call_gemini(system_prompt, user_prompt, run_id, session_id, trigger_reason)
        elif self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            raw_response = self._call_openai_compatible(system_prompt, user_prompt, run_id, session_id, trigger_reason)
        else:
            raw_response = self._call_mock(current_policy, trigger_reason, run_id, session_id)

        new_code = self.extract_code(raw_response)
        try:
            tree = compile_policy(new_code, action_handlers=action_handlers)
            return new_code, tree, None
        except Exception as e:
            return new_code, None, f"Policy compilation failed: {e}"
