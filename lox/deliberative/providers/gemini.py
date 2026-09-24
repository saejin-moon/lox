"""
GeminiProvider: Connects to Google AI Studio.

Two paths (R3):
- `generate_reasoning_and_json`: OpenAI-compatible endpoint (existing deliberative path).
- `generate_text`: native `google-genai` SDK with thinking config (default: level=high)
  — the policy-diff author path. Thinking moves the model's reasoning into a separate
  channel so `response.text` is clean diff output. Falls back to the OpenAI-compatible
  endpoint when `google-genai` is unavailable.

Env: GEMINI_API_KEY (both paths) · GEMINI_MODEL (default model) ·
GEMINI_THINKING_LEVEL (minimal|low|medium|high, default high; "off" disables).
"""

import os
import time
from typing import Type, TypeVar
from pydantic import BaseModel

from lox.deliberative.providers.base import LLMProvider, LLMResponse
from lox.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """
    Executes deliberative reasoning via Google AI Studio. The free tier provides up to
    1,500 requests/day, easily covering research runs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout: float = 45.0,
        thinking_level: str | None = None,
        use_native_genai: bool | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
        self.base_url = base_url
        self.timeout = timeout
        self.thinking_level = (thinking_level
                               or os.environ.get("GEMINI_THINKING_LEVEL", "high")).lower()

        # Native google-genai path (author path with thinking); auto-enabled when the
        # SDK is importable, forced off with use_native_genai=False or GEMINI_USE_GENAI=0.
        env_flag = os.environ.get("GEMINI_USE_GENAI", "1") not in ("0", "false", "False")
        self._genai_client = None
        if use_native_genai is None:
            use_native_genai = env_flag
        if use_native_genai:
            try:
                from google import genai as _genai  # noqa: PLC0415
                self._genai_client = _genai.Client(api_key=self.api_key)
            except ImportError:
                self._genai_client = None

        # OpenAI-compatible fallback client (lazy import keeps native path independent)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Native google-genai helpers
    # ------------------------------------------------------------------
    def _genai_config(self, extra: dict | None = None):
        from google.genai import types  # noqa: PLC0415
        cfg_kwargs: dict = {}
        if self.thinking_level not in ("", "off", "none"):
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level.upper())
        if extra:
            cfg_kwargs.update(extra)
        return types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    async def _genai_generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Runs the native SDK call in a worker thread (sync SDK). Returns a dict with
        the answer text, the THOUGHT-channel text (captured for the ledger), and the
        full token breakdown. The system prompt goes through `system_instruction`;
        thinking config moves the model's reasoning out of `response.text`."""
        import asyncio  # noqa: PLC0415
        client = self._genai_client
        config = self._genai_config({"system_instruction": system_prompt})

        def _call():
            resp = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            # Split thought-channel parts from answer parts (candidates[0].content.parts)
            answer_parts: list[str] = []
            thought_parts: list[str] = []
            for cand in getattr(resp, "candidates", None) or []:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    text = getattr(part, "text", None) or ""
                    if not text:
                        continue
                    if getattr(part, "thought", False):
                        thought_parts.append(text)
                    else:
                        answer_parts.append(text)
            answer = "".join(answer_parts) or (resp.text or "")
            thought = "".join(thought_parts)
            um = getattr(resp, "usage_metadata", None)
            usage = {
                "in": int(getattr(um, "prompt_token_count", 0) or 0),
                "out": int(getattr(um, "candidates_token_count", 0) or 0),
                "thought": int(getattr(um, "thoughts_token_count", 0) or 0),
                "total": int(getattr(um, "total_token_count", 0) or 0),
            }
            return {"answer": answer, "thought": thought, "usage": usage}

        return await asyncio.to_thread(_call)

    # ------------------------------------------------------------------
    # JSON deliberative path (unchanged OpenAI-compatible endpoint)
    # ------------------------------------------------------------------
    async def generate_reasoning_and_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> LLMResponse:
        start_time = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        latency = (time.perf_counter() - start_time) * 1000.0

        thinking, parsed = ResponseParser.parse_dual_phase(raw_text, schema)

        return LLMResponse(
            thinking_content=thinking,
            parsed_payload=parsed,
            raw_text=raw_text,
            tokens_consumed=tokens,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # Raw text path — the R3 policy-diff author path (native SDK + thinking)
    # ------------------------------------------------------------------
    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        context: dict | None = None,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        if self._genai_client is not None:
            out = await self._genai_generate(system_prompt, user_prompt)
            raw_text = out["answer"]
            thought_text = out["thought"]
            usage = out["usage"]
            tokens = usage["total"]
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            thought_text = ""
            usage = {"in": 0, "out": 0, "thought": 0}
            tokens = response.usage.total_tokens if response.usage else 0
        latency = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            thinking_content=thought_text,   # native thought-channel output (ledger-captured)
            parsed_payload=raw_text,
            raw_text=raw_text,
            tokens_consumed=tokens,
            latency_ms=latency,
            tokens_in=usage["in"],
            tokens_out=usage["out"],
            tokens_thought=usage["thought"],
        )