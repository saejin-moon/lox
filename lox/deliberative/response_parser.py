"""
Response parser for deliberative LLM outputs: Extracts <think> reasoning
and parses post-think JSON payloads into validated Pydantic models.
"""

import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMParseError(Exception):
    """Raised when LLM output cannot be parsed into the expected Pydantic schema."""
    pass


class ResponseParser:
    """
    Robust parser handling dual-phase output: <think> reasoning + structured JSON.
    """

    @classmethod
    def parse_dual_phase(cls, raw_text: str, schema: Type[T]) -> tuple[str, T]:
        """
        Extracts thinking scratchpad and parses JSON into schema.
        Returns (thinking_content, parsed_model_instance).
        """
        # 1. Extract <think> content
        thinking = ""
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            # Remove the <think> section from payload search
            post_think_text = raw_text[think_match.end():].strip()
        else:
            post_think_text = raw_text.strip()

        # 2. Extract JSON payload
        json_str = cls._extract_json_substring(post_think_text)
        if not json_str:
            # Fallback: search entire raw text if not found post-think
            json_str = cls._extract_json_substring(raw_text)

        if not json_str:
            raise LLMParseError(f"No valid JSON object found in LLM response:\n{raw_text[:300]}...")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise LLMParseError(f"Malformed JSON from LLM: {e}\nRaw JSON text: {json_str[:200]}") from e

        try:
            model_instance = schema.model_validate(data)
        except Exception as e:
            raise LLMParseError(f"Pydantic validation failed for {schema.__name__}: {e}\nData: {data}") from e

        return thinking, model_instance

    @classmethod
    def _extract_json_substring(cls, text: str) -> str:
        """Finds markdown codeblock or outermost matching curly braces."""
        # 1. Check for ```json ... ``` codeblock
        block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if block_match:
            return block_match.group(1).strip()

        # 2. Find outermost '{' and '}'
        start_idx = text.find("{")
        if start_idx == -1:
            return ""

        depth = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            return text[start_idx:end_idx + 1].strip()

        return ""
