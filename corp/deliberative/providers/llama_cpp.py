"""
LlamaCppProvider: Connects to local llama-server instance (e.g. QwQ-32B or Qwen-2.5-32B).
Zero cloud egress, 100% offline, deterministic temperature (T=0.0).
"""

import time
from typing import Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class LlamaCppProvider(LLMProvider):
    """
    Executes deliberative reasoning against local llama.cpp / vLLM server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        import os
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model or os.environ.get("LLAMA_CPP_MODEL", "local-model")
        self.temperature = temperature

    async def generate_reasoning_and_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> LLMResponse:
        start_time = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
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
