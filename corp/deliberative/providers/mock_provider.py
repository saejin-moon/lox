"""
MockProvider: Deterministic offline LLM provider for unit tests and local simulation.
"""

from typing import Any, Type, TypeVar
import time
from pydantic import BaseModel

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class MockProvider(LLMProvider):
    """
    Returns scripted or auto-generated responses for offline unit testing.
    """

    def __init__(self, canned_response: str | None = None):
        self.canned_response = canned_response
        self.call_history: list[dict[str, Any]] = []

    def set_canned_response(self, text: str):
        self.canned_response = text

    async def generate_reasoning_and_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> LLMResponse:
        start_time = time.perf_counter()
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema.__name__,
        })

        if not self.canned_response:
            raise ValueError("MockProvider called without canned_response set.")

        thinking, parsed = ResponseParser.parse_dual_phase(self.canned_response, schema)
        latency = (time.perf_counter() - start_time) * 1000.0

        return LLMResponse(
            thinking_content=thinking,
            parsed_payload=parsed,
            raw_text=self.canned_response,
            tokens_consumed=len(self.canned_response.split()),
            latency_ms=latency,
        )
