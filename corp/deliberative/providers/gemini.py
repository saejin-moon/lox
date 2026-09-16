"""
GeminiProvider: Connects to Google AI Studio Gemini Flash API via OpenAI-compatible endpoint.
"""

import os
import time
from typing import Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """
    Executes deliberative reasoning via Google AI Studio (Gemini 2.5 Flash / 1.5 Flash).
    The free tier provides up to 1,500 requests/day, easily covering research runs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout: float = 45.0,
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "EMPTY")
        self.client = AsyncOpenAI(base_url=base_url, api_key=resolved_key, timeout=timeout)
        self.model = model

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
