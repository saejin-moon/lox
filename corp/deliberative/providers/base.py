"""
Base interface and response wrapper for deliberative LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True, frozen=True)
class LLMResponse:
    thinking_content: str
    parsed_payload: Any  # Validated Pydantic model instance
    raw_text: str
    tokens_consumed: int = 0
    latency_ms: float = 0.0


class LLMProvider(ABC):
    """
    Abstract interface for multi-provider deliberative LLM execution.
    """

    @abstractmethod
    async def generate_reasoning_and_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> LLMResponse:
        """
        Executes reasoning call, extracting <think> and validating structured JSON into schema.
        """
        pass

    @abstractmethod
    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        context: dict | None = None,
    ) -> LLMResponse:
        """
        Plain text completion (R3 policy-diff author path, AGENT_PLAN §4.5). No JSON
        parsing — `parsed_payload` carries the raw text. `context` lets deterministic
        providers (Mock) format canned outputs (e.g. current program version); API
        providers ignore it.
        """
        raise NotImplementedError
