"""
LOX Author Agent: Asynchronous LLM policy synthesis engine with DuckDB analytical tooling.
Queries LLMs across multiple providers (Gemini, OpenRouter, local vLLM, or Mock),
enables interactive empirical data investigation (ReAct),
logs token usage directly into DuckDB, and validates proposed policies against AST invariants.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import signal
import threading
import time
import uuid
from typing import Any

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import numpy as np

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.tools import OPENAI_TOOL_SPECS, DuckDBToolRegistry
from lox.core.types import Action, HeroState, Observation
from lox.dsl.compiler import compile_policy
from lox.telemetry.tokens import log_token_usage


class AuthorAgent:
    """Manages LLM policy authoring sessions with DuckDB analytical tools and token tracking."""

    def __init__(
        self,
        provider: str = "mock",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        db_path: str = "data/lox.duckdb",
        max_tool_turns: int = 5,
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
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
        self.base_url = base_url
        self.db_path = db_path
        self.max_tool_turns = max_tool_turns
        self.tools = DuckDBToolRegistry(db_path=db_path)

    @staticmethod
    def splice_policy_methods(base_code: str, patch_code: str) -> str:
        """
        Surgically splices replacement and new methods from patch_code into base_code via AST.
        Supports both standalone `def foo(self, ...):` and `class Agent:` wrapped method definitions.
        If patch_code is a complete standalone policy or base_code is empty, returns patch_code.
        """
        if not base_code or not base_code.strip():
            return patch_code
        if not patch_code or not patch_code.strip():
            return base_code

        import textwrap
        from lox.dsl.parser import repair_python_indentation

        patch_code = patch_code.replace("\t", "    ")
        repaired_patch = repair_python_indentation(patch_code)

        # Multi-candidate AST parsing
        patch_ast = None
        for candidate in (
            repaired_patch,
            textwrap.dedent(repaired_patch),
            patch_code,
            textwrap.dedent(patch_code),
        ):
            try:
                patch_ast = ast.parse(candidate)
                break
            except SyntaxError:
                continue

        # Extract replacement methods from patch_ast
        replacements: dict[str, ast.FunctionDef] = {}
        patch_has_class = False
        if patch_ast is not None:
            for node in patch_ast.body:
                if isinstance(node, ast.FunctionDef):
                    replacements[node.name] = node
                elif isinstance(node, ast.ClassDef):
                    patch_has_class = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            replacements[item.name] = item

        # Robust Fallback: Method Chunk Extraction
        # If whole-block parsing failed or missed methods due to whitespace anomalies, extract each def chunk in isolation
        if not replacements:
            lines = repaired_patch.splitlines()
            chunks = []
            curr = []
            def_pat = re.compile(r"^\s*def\s+\w+\s*\(")
            for line in lines:
                if def_pat.match(line):
                    if curr:
                        chunks.append("\n".join(curr))
                        curr = []
                curr.append(line)
            if curr:
                chunks.append("\n".join(curr))

            for chunk in chunks:
                norm = textwrap.dedent(chunk)
                c_lines = norm.splitlines()
                if c_lines and c_lines[0].startswith("def "):
                    body_indents = [
                        len(l) - len(l.lstrip(" "))
                        for l in c_lines[1:]
                        if l.strip()
                    ]
                    if body_indents and min(body_indents) > 4:
                        excess = min(body_indents) - 4
                        norm = "\n".join(
                            [c_lines[0]]
                            + [
                                (
                                    l[excess:]
                                    if l.startswith(" " * excess)
                                    else l.lstrip()
                                )
                                if l.strip()
                                else ""
                                for l in c_lines[1:]
                            ]
                        )
                for c_cand in (norm, textwrap.dedent(norm), chunk):
                    try:
                        c_ast = ast.parse(c_cand)
                        for node in c_ast.body:
                            if isinstance(node, ast.FunctionDef):
                                replacements[node.name] = node
                        break
                    except SyntaxError:
                        continue

        if not replacements:
            return repaired_patch

        try:
            base_ast = ast.parse(base_code)
        except SyntaxError:
            return patch_code

        # Locate class Agent in base_ast
        agent_node = None
        for node in base_ast.body:
            if isinstance(node, ast.ClassDef) and node.name == "Agent":
                agent_node = node
                break

        if agent_node is None:
            return patch_code

        # If patch defines a class with all core methods (run and __init__), it's a complete replacement
        base_method_names = {
            item.name for item in agent_node.body if isinstance(item, ast.FunctionDef)
        }
        if (
            patch_has_class
            and "run" in replacements
            and len(replacements) >= len(base_method_names)
        ):
            return patch_code

        new_body = []
        for item in agent_node.body:
            if isinstance(item, ast.FunctionDef) and item.name in replacements:
                new_method = replacements.pop(item.name)
                # Ensure 'self' is the first parameter if missing
                if not new_method.args.args or new_method.args.args[0].arg != "self":
                    new_method.args.args.insert(0, ast.arg(arg="self"))
                new_body.append(new_method)
            else:
                new_body.append(item)

        # Append any brand-new helper methods defined in patch
        for new_method in replacements.values():
            if not new_method.args.args or new_method.args.args[0].arg != "self":
                new_method.args.args.insert(0, ast.arg(arg="self"))
            new_body.append(new_method)

        agent_node.body = new_body

        try:
            return ast.unparse(base_ast)
        except Exception:
            return patch_code

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatches tool execution against the DuckDBToolRegistry with robust error shielding."""
        try:
            if tool_name == "query_duckdb":
                return self.tools.query_duckdb(args.get("sql", ""))
            elif tool_name == "get_duckdb_schema":
                return self.tools.get_duckdb_schema()
            elif tool_name == "get_death_taxonomy":
                return self.tools.get_death_taxonomy(window=args.get("window", 20))
            elif tool_name == "query_root_causes":
                return self.tools.query_root_causes(
                    run_id=args.get("run_id", ""),
                    window=args.get("window", 10),
                )
            elif tool_name == "get_floor_pacing_stats":
                return self.tools.get_floor_pacing_stats(depth=args.get("depth", 1))
            elif tool_name == "get_action_distribution":
                return self.tools.get_action_distribution(run_id=args.get("run_id", ""))
            elif tool_name == "query_wiki":
                return self.tools.query_wiki(
                    args.get("query", ""), top_k=args.get("top_k", 2)
                )
            elif tool_name == "query_invariants":
                return self.tools.query_invariants(
                    query=args.get("query", ""),
                    category=args.get("category", ""),
                    top_k=args.get("top_k", 3),
                )
            elif tool_name == "request_macro":
                run_id = getattr(self, "_current_run_id", "synth_session")
                return self.tools.request_macro(
                    macro_name=args.get("macro_name", ""),
                    rationale=args.get("rationale", ""),
                    proposed_interface=args.get("proposed_interface", ""),
                    priority=args.get("priority", "medium"),
                    run_id=run_id,
                )
            elif tool_name == "get_dungeon_topology":
                return self.tools.get_dungeon_topology(depth=args.get("depth", 1))
            elif tool_name == "get_hazard_map":
                return self.tools.get_hazard_map(depth=args.get("depth", 1))
            elif tool_name == "get_floor_stash_report":
                return self.tools.get_floor_stash_report()
            elif tool_name == "get_death_autopsy_trace":
                return self.tools.get_death_autopsy_trace(
                    episode_id=args.get("episode_id", ""),
                    limit=args.get("limit", 30),
                )
            elif tool_name == "get_macro_requests":
                return self.tools.get_macro_requests()
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
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # Emergency: self-monitored prayer and healing
            if obs.hero.hp_frac < 0.20 and (obs.hero.turn - self.last_prayer_turn >= 350):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            elif obs.hero.hp_frac < 0.35 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
            elif obs.hero.hunger_state >= HUNGRY and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # Tactical combat & retreat
            if obs.combat.adjacent_hostile:
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                elif obs.hero.hp_frac < 0.35 and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # Environment & Navigation
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.stairs_down_known and not obs.spatial.has_unvisited_frontier:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                obs = yield wait()
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
            self.tools.query_invariants,
            self.tools.get_dungeon_topology,
            self.tools.get_hazard_map,
            self.tools.get_floor_stash_report,
            self.tools.get_death_autopsy_trace,
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
            prompt_tokens = (
                getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            )
            completion_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

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

        url = (self.base_url or "https://openrouter.ai/api/v1").rstrip(
            "/"
        ) + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "HTTP-Referer": "https://github.com/saejin-moon/lox",
            "X-Title": "LOX Policy Synthesis",
        }
        model_name = self.model or "qwen/qwen3.5-9b"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        tools_invoked = []

        with httpx.Client(timeout=120.0) as client:
            use_tools = True
            max_tool_turns = self.max_tool_turns  # Budget ceiling (default 5 turns)
            for turn in range(max_tool_turns):
                payload: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 3000,
                }
                if use_tools:
                    payload["tools"] = OPENAI_TOOL_SPECS
                    payload["tool_choice"] = "auto"

                max_retries = 6
                resp = None
                for attempt in range(max_retries):
                    try:
                        resp = client.post(url, headers=headers, json=payload)
                        status = getattr(resp, "status_code", 200)
                        if (
                            status in (429, 500, 502, 503, 504)
                            and attempt < max_retries - 1
                        ):
                            backoff = (2**attempt) + random.uniform(1.0, 3.0)
                            time.sleep(backoff)
                            continue
                        break
                    except (
                        httpx.RemoteProtocolError,
                        httpx.NetworkError,
                        httpx.TimeoutException,
                    ) as exc:
                        if attempt < max_retries - 1:
                            backoff = (2**attempt) + random.uniform(1.0, 3.0)
                            time.sleep(backoff)
                            continue
                        raise RuntimeError(
                            f"OpenRouter network error after {max_retries} attempts: {exc}"
                        ) from exc

                if resp is None:
                    raise RuntimeError("No response received from OpenRouter API")

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
                    if (
                        use_tools
                        and resp.status_code == 400
                        and (
                            "tool" in err_msg.lower()
                            or "not supported" in err_msg.lower()
                        )
                    ):
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
                        try:
                            fn_args = json.loads(tc["function"].get("arguments", "{}"))
                        except Exception:
                            fn_args = {}
                        tools_invoked.append(fn_name)
                        result_text = self._execute_tool(fn_name, fn_args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result_text,
                            }
                        )
                else:
                    # Model provided text/code without requesting more tools
                    break
            else:
                # If tool budget ceiling was reached without dynamic exit, prompt for final code synthesis
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have concluded your empirical tool investigation. "
                            "Based on all telemetry, incident logs, and knowledge retrieved above, "
                            "synthesize your complete revised policy program now in a single ```python ... ``` block."
                        ),
                    }
                )
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
            "Output ONLY the corrected method or complete corrected policy code in a single ```python ... ``` block."
        )

        if self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            url = (self.base_url or "https://openrouter.ai/api/v1").rstrip(
                "/"
            ) + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "HTTP-Referer": "https://github.com/saejin-moon/lox",
                "X-Title": "LOX Policy Synthesis",
            }
            messages = getattr(self, "_last_messages", [])
            messages.append({"role": "user", "content": repair_user_msg})
            model_name = getattr(
                self, "_last_model_name", self.model or "google/gemini-2.5-flash"
            )

            try:
                with httpx.Client(timeout=120.0) as client:
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 3000,
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
            except Exception as exc:
                print(f"[Warning] Policy repair LLM call failed: {exc}")
        return candidate_code

    def extract_code(self, response_text: str) -> str:
        """Robustly extracts Python policy code block (classes or generators) from markdown."""
        if not response_text:
            return ""

        def _clean(c: str) -> str:
            if not c:
                return ""
            c = c.replace("\t", "    ")
            return "\n".join(l.rstrip() for l in c.splitlines()).strip()

        # 1. Search all code blocks, prioritizing class definitions or generator/plan structures
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
        for block in reversed(blocks):
            if "class " in block and ("run" in block or "def " in block):
                return _clean(block)
            if "def " in block and ("yield" in block or "plan" in block):
                return _clean(block)
        for block in blocks:
            if "class " in block or "def " in block:
                return _clean(block)

        # 2. Unclosed python code block containing policy code
        match_unclosed = re.search(
            r"```(?:python)?\s*\n(.*?)(?:```|$)", response_text, re.DOTALL
        )
        if match_unclosed:
            code = match_unclosed.group(1).strip()
            if "class " in code or "def " in code:
                return _clean(re.sub(r"```+$", "", code))

        # 3. No backticks: find from first 'class ' or 'def '
        match_class = re.search(r"(class \w+.*)", response_text, re.DOTALL)
        if match_class:
            return _clean(match_class.group(1))

        match_def = re.search(r"(def \w+\(.*)", response_text, re.DOTALL)
        if match_def:
            return _clean(match_def.group(1))

        return ""

    def extract_hypothesis(self, response_text: str) -> dict[str, Any] | None:
        """Extracts structured scientific hypothesis from response text."""
        if not response_text:
            return None
        import yaml

        # 1. Search for ```yaml ... ``` or ```json ... ``` blocks with hypothesis
        blocks = re.findall(
            r"```(?:ya?ml|json)?\s*\n(.*?)```", response_text, re.DOTALL
        )
        for block in blocks:
            if "hypothesis" in block:
                try:
                    data = yaml.safe_load(block)
                    if isinstance(data, dict):
                        return data.get("hypothesis", data)
                except Exception:
                    pass

        # 2. Search unclosed or raw block
        match = re.search(
            r"(?:^|\n)(hypothesis:\s*\n(?:[ \t]+[^\n]+\n?)+)", response_text
        )
        if match:
            try:
                data = yaml.safe_load(match.group(1))
                if isinstance(data, dict):
                    return data.get("hypothesis", data)
            except Exception:
                pass

    def compile_empirical_dossier(
        self,
        batch_summary: Any = None,
        top_k_deep: int = 3,
    ) -> str:
        """
        Synthesizes the Pareto Hybrid Dossier on Turn 0:
        1. The Trustworthy Triad:
           - Factor 1: Macro Archetype Distribution Table (Top 5 causes)
           - Factor 2: Orthogonal Systemic Bottlenecks & Strategic Recommendations
           - Factor 3: Ground-Truth Empirical Telemetry Context (Killers, Depth, Flight Traces)
        2. Deep Mechanics Dossier for the Top 2-3 Dominant Causes:
           - Automatic Wiki Mechanics retrieval
           - Ground-truth Canonical Invariants retrieval
        """
        lines = []
        top_archetypes: list[str] = []

        if batch_summary is not None:
            if hasattr(batch_summary, "format_trustworthy_triad_report"):
                lines.append(batch_summary.format_trustworthy_triad_report())
                if hasattr(batch_summary, "archetype_counts") and batch_summary.archetype_counts:
                    sorted_archs = sorted(
                        batch_summary.archetype_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    top_archetypes = [a[0] for a in sorted_archs[:top_k_deep]]
            elif isinstance(batch_summary, dict):
                import yaml
                lines.append("## THE TRUSTWORTHY TRIAD: EMPIRICAL BATCH AUTOPSY REPORT\n")
                lines.append(yaml.dump(batch_summary, sort_keys=False))
                rc = batch_summary.get("root_causes", {})
                if isinstance(rc, dict):
                    sorted_rc = sorted(
                        rc.items(),
                        key=lambda x: x[1].get("count", 0) if isinstance(x[1], dict) else x[1],
                        reverse=True,
                    )
                    top_archetypes = [a[0] for a in sorted_rc[:top_k_deep]]

        # Deep Mechanics Dossier for Top Dominant Archetypes
        if top_archetypes:
            lines.append("\n## DEEP MECHANICS & INVARIANTS DOSSIER (TOP DOMINANT CAUSES)")
            archetype_to_wiki = {
                "ARMOR_DEFICIT": "Armor class",
                "STALL_SECRET_DOOR": "Secret door",
                "STARVATION_FAINTING": "Hunger",
                "COMBAT_FAST_PREDATOR": "Speed",
                "COMBAT_TACTICAL_SWARM": "Elbereth",
                "PASSIVE_HAZARD_PARALYSIS": "Floating eye",
                "INSTADEATH_POISON": "Poison resistance",
                "PETRIFICATION": "Cockatrice",
                "TRAP_FATALITY": "Traps",
                "DROWNING_OR_LAVA": "Water",
                "ENCUMBRANCE_IMMOBILITY": "Encumbrance",
                "PEACEFUL_NPC_PROVOCATION": "Peaceful monster",
            }
            for arch in top_archetypes:
                wiki_topic = archetype_to_wiki.get(arch, arch.lower().replace("_", " "))
                lines.append(f"\n### Deep Mechanics for `{arch}`:")
                try:
                    wiki_res = self.tools.query_wiki(wiki_topic, top_k=1)
                    if wiki_res and "No NetHack wiki" not in wiki_res:
                        lines.append(wiki_res)
                except Exception:
                    pass
                try:
                    inv_res = self.tools.query_invariants(query=arch, top_k=2)
                    if inv_res and "No matching invariants" not in inv_res:
                        lines.append(inv_res)
                except Exception:
                    pass

        return "\n".join(lines).strip()

    def synthesize_policy(
        self,
        current_policy: str,
        trigger_reason: str,
        status_report: str = "",
        run_id: str = "synth_run",
        action_handlers: dict | None = None,
        max_repairs: int = 2,
        causal_summary: Any = "",
        batch_summary: Any = None,
    ) -> tuple[str, Any, str | None]:
        """
        Runs one authoring session with tool calling, AST compilation, and self-repair retries.
        Returns (new_code, compiled_executable_or_None, error_message_or_None).
        """
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self._current_run_id = run_id
        self.last_hypothesis = None

        # Pre-compile Turn 0 Pareto Hybrid Dossier if batch summary is available
        if batch_summary is not None:
            dossier_text = self.compile_empirical_dossier(batch_summary=batch_summary)
            if dossier_text:
                status_report = f"{dossier_text}\n\n{status_report}".strip()
        elif causal_summary:
            dossier_text = self.compile_empirical_dossier(batch_summary=causal_summary)
            if dossier_text:
                status_report = f"{dossier_text}\n\n{status_report}".strip()

        recent_experiments_str = ""
        if os.path.exists(self.db_path):
            try:
                from lox.telemetry.consolidator import safe_duckdb_connect
                con = safe_duckdb_connect(self.db_path, read_only=True)
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                if "meta_experiments" in tables:
                    rows = con.execute("""
                        SELECT generation, targeted_skill, causal_finding, paired_seed_delta, outcome_validated
                        FROM meta_experiments
                        ORDER BY generation DESC
                        LIMIT 5
                    """).fetchall()
                    if rows:
                        items = []
                        for r in rows:
                            gen_num, skill, finding, delta, val = r
                            status = "VALIDATED" if val else "FALSIFIED"
                            delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
                            items.append(
                                f"- Gen {gen_num} [{status}]: skill='{skill}', delta={delta_str}, finding='{finding}'"
                            )
                        recent_experiments_str = "\n".join(items)
                con.close()
            except Exception:
                pass

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            current_policy,
            trigger_reason,
            status_report,
            causal_summary=causal_summary,
            recent_experiments=recent_experiments_str,
        )

        if self.provider == "gemini":
            raw_response = self._call_gemini(
                system_prompt, user_prompt, run_id, session_id, trigger_reason
            )
        elif self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            raw_response = self._call_openai_compatible(
                system_prompt, user_prompt, run_id, session_id, trigger_reason
            )
        else:
            raw_response = self._call_mock(
                current_policy, trigger_reason, run_id, session_id
            )

        # Compilation and Self-Repair Loop
        self.last_hypothesis = self.extract_hypothesis(raw_response)
        extracted = self.extract_code(raw_response)
        candidate_code = (
            self.splice_policy_methods(current_policy, extracted)
            if (current_policy and extracted)
            else extracted
        )
        last_error = None

        for attempt in range(max_repairs + 1):
            is_valid_structure = (
                "class " in candidate_code
                and ("def " in candidate_code or "run" in candidate_code)
            ) or (
                "def " in candidate_code
                and (
                    "yield" in candidate_code
                    or "plan" in candidate_code
                    or "return" in candidate_code
                )
            )
            if is_valid_structure:
                try:
                    tree = compile_policy(
                        candidate_code, action_handlers=action_handlers
                    )
                    # Multi-scenario dry-run validation: execute test steps across scenarios with timeout shield
                    from lox.core.types import (
                        CombatView,
                        FloorCorpse,
                        HungerState,
                        SpatialView,
                    )

                    test_scenarios = [
                        (
                            "normal",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10, x=10, hp=16, max_hp=16, depth=1, turn=10
                                ),
                            ),
                        ),
                        (
                            "hungry_unsafe_corpse",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10,
                                    x=10,
                                    hp=16,
                                    max_hp=16,
                                    depth=1,
                                    turn=750,
                                    hunger_state=HungerState.HUNGRY,
                                ),
                                corpses=[
                                    FloorCorpse(
                                        name="kobold corpse",
                                        y=11,
                                        x=11,
                                        drop_turn=10,
                                        age_turns=100,
                                        is_poisonous=True,
                                        is_deadly=True,
                                        is_fresh=False,
                                    )
                                ],
                            ),
                        ),
                        (
                            "floating_eye_combat",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10, x=10, hp=16, max_hp=16, depth=1, turn=10
                                ),
                                combat=CombatView(
                                    hostile_count_fov=1,
                                    adjacent_hostile=True,
                                    adjacent_floating_eye=True,
                                    closest_hostile_name="floating eye",
                                    floating_eye_in_fov=True,
                                ),
                            ),
                        ),
                        (
                            "combat",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10, x=10, hp=16, max_hp=16, depth=1, turn=10
                                ),
                                combat=CombatView(
                                    hostile_count_fov=1, adjacent_hostile=True
                                ),
                            ),
                        ),
                        (
                            "combat_distant",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10, x=10, hp=16, max_hp=16, depth=1, turn=10
                                ),
                                combat=CombatView(
                                    hostile_count_fov=1,
                                    adjacent_hostile=False,
                                    closest_hostile_dist=3.0,
                                    closest_hostile_name="jackal",
                                    has_active_hostile=False,
                                ),
                            ),
                        ),
                        (
                            "dead_end",
                            Observation(
                                chars=np.full((21, 79), ord("."), dtype=np.uint8),
                                glyphs=np.zeros((21, 79), dtype=np.int16),
                                hero=HeroState(
                                    y=10, x=10, hp=16, max_hp=16, depth=1, turn=10
                                ),
                                spatial=SpatialView(has_unsearched_dead_end=True),
                            ),
                        ),
                    ]

                    def _step_with_timeout(runner, s_obs, timeout=1.0):
                        if (
                            threading.current_thread() is threading.main_thread()
                            and hasattr(signal, "SIGALRM")
                        ):

                            def _alarm_handler(signum, frame):
                                raise TimeoutError("Step execution timed out")

                            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                            signal.setitimer(signal.ITIMER_REAL, timeout)
                            try:
                                return runner.send(s_obs)
                            finally:
                                signal.setitimer(signal.ITIMER_REAL, 0)
                                signal.signal(signal.SIGALRM, old_handler)
                        else:
                            import multiprocessing as mp

                            mp_ctx = (
                                mp.get_context("fork")
                                if hasattr(mp, "get_context")
                                and "fork" in mp.get_all_start_methods()
                                else mp
                            )
                            q = mp_ctx.Queue()

                            def _sub_worker():
                                try:
                                    res = runner.send(s_obs)
                                    q.put(("OK", res))
                                except Exception as exc:
                                    q.put(("ERR", exc))

                            p = mp_ctx.Process(target=_sub_worker)
                            p.start()
                            p.join(timeout=timeout)
                            if p.is_alive():
                                p.kill()
                                p.join()
                                raise TimeoutError("Step execution timed out")
                            if not q.empty():
                                status, val = q.get()
                                if status == "OK":
                                    return val
                                raise val
                            raise TimeoutError("Step execution timed out")

                    for s_name, s_obs in test_scenarios:
                        runner = tree.create_runner(s_obs)
                        for step_i in range(3):
                            try:
                                act = _step_with_timeout(runner, s_obs, timeout=1.0)
                            except TimeoutError:
                                raise ValueError(
                                    f"Policy entered an infinite loop without yielding under {s_name} scenario (step {step_i + 1}). "
                                    f"Ensure all generator subroutines yield an action (e.g. obs = yield wait()) on all execution paths."
                                )
                            if not isinstance(act, Action):
                                raise ValueError(
                                    f"Policy runner produced non-Action under {s_name} state: {act}"
                                )
                            if (
                                s_name == "hungry_unsafe_corpse"
                                and act.name == "eat_floor_corpse"
                            ):
                                raise ValueError(
                                    "Invariant regression: Policy attempted to eat unsafe/poisonous corpse! Check corpse.is_safe first."
                                )
                            if (
                                s_name == "floating_eye_combat"
                                and act.name == "melee_attack_hostile"
                            ):
                                raise ValueError(
                                    "Invariant regression: Policy attempted melee attack against floating eye! Gaze will paralyze hero."
                                )
                    return candidate_code, tree, None
                except Exception as e:
                    last_error = f"Policy compilation/validation failed: {e}"
            else:
                last_error = "Policy compilation failed: Extracted response does not contain valid policy code structure ('class Agent:' or 'def' missing)."

            # If compilation failed and repair attempts remain, ask the model to self-correct
            if attempt < max_repairs and self.provider in (
                "openrouter",
                "vllm",
                "llama_cpp",
                "openai",
            ):
                print(
                    f"[Author Agent] Compilation issue detected ({last_error}). Triggering self-repair turn {attempt + 1}/{max_repairs}..."
                )
                repaired_response = self._repair_code(
                    candidate_code=candidate_code,
                    compile_err=last_error,
                    run_id=run_id,
                    session_id=session_id,
                    trigger_reason=trigger_reason,
                )
                repaired_extracted = self.extract_code(repaired_response)
                candidate_code = (
                    self.splice_policy_methods(candidate_code, repaired_extracted)
                    if (candidate_code and repaired_extracted)
                    else repaired_extracted
                )

        return candidate_code, None, last_error
