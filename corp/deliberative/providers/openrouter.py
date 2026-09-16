"""
OpenRouterProvider: Connects to OpenRouter free/commercial API endpoints.
"""

import os
import time
from typing import Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class OpenRouterProvider(LLMProvider):
    """
    Executes deliberative reasoning via OpenRouter API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 45.0,
    ):
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
        self.client = AsyncOpenAI(base_url=base_url, api_key=resolved_key, timeout=timeout)
        self.model = model or os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")

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
