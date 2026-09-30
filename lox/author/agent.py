"""
LOX 2.0 Author Agent: Asynchronous LLM policy synthesis engine.
Queries LLMs across multiple providers (Gemini, OpenRouter, local vLLM, or Mock)
and validates proposed policy programs against formal AST and safety invariants.
"""
from __future__ import annotations

import os
import re
from typing import Any
import httpx

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.dsl.compiler import compile_policy
from lox.core.tree import BehaviorTree


class AuthorAgent:
    """Manages LLM policy authoring sessions."""

    def __init__(
        self,
        provider: str = "mock",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url

    def _call_mock(self, current_policy: str, trigger_reason: str) -> str:
        """Deterministic mock provider for testing and offline verification."""
        return """```python
def emergency_survival():
    if hp_frac <= 0.30 and can_safely_pray:
        pray()
    elif hp_frac <= 0.50 and has_healing:
        quaff_healing()

def handle_nutrition():
    if hunger_state >= HUNGRY and has_carried_food:
        eat_carried_food()

def combat():
    if adjacent_hostile:
        if hp_frac <= 0.35 and can_retreat:
            step_away_from_hostile()
        else:
            melee_attack_hostile()

def floor_progression():
    if stairs_down_known and (floor_explored or turns_on_level > 80):
        step_to_stairs_down()
        descend()
    elif has_unvisited_frontier:
        step_to_frontier()
    elif has_unsearched_dead_end:
        step_to_dead_end()
        search()

plan = [
    emergency_survival,
    handle_nutrition,
    combat,
    floor_progression,
]
```"""

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        model_name = self.model or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={"system_instruction": system_prompt, "temperature": 0.2},
        )
        return response.text or ""

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        url = (self.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model or "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def extract_code(self, response_text: str) -> str:
        """Extracts the Python code block from markdown."""
        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: if no code fences, assume raw code
        return response_text.strip()

    def synthesize_policy(
        self,
        current_policy: str,
        trigger_reason: str,
        autopsy_report: str = "",
        action_handlers: dict | None = None,
    ) -> tuple[str, BehaviorTree | None, str | None]:
        """
        Runs one authoring session.
        Returns (new_code, compiled_tree_or_None, error_message_or_None).
        """
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(current_policy, trigger_reason, autopsy_report)

        if self.provider == "gemini":
            raw_response = self._call_gemini(system_prompt, user_prompt)
        elif self.provider in ("openrouter", "vllm", "llama_cpp", "openai"):
            raw_response = self._call_openai_compatible(system_prompt, user_prompt)
        else:
            raw_response = self._call_mock(current_policy, trigger_reason)

        new_code = self.extract_code(raw_response)
        try:
            tree = compile_policy(new_code, action_handlers=action_handlers)
            return new_code, tree, None
        except Exception as e:
            return new_code, None, f"Policy compilation failed: {e}"
