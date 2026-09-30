import json
import os
import pytest
from unittest.mock import patch, MagicMock
import httpx

from lox.author.agent import AuthorAgent


def test_openrouter_api_key_resolution(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret-456")

    # OpenRouter provider should specifically select OPENROUTER_API_KEY
    agent_or = AuthorAgent(provider="openrouter")
    assert agent_or.api_key == "openrouter-secret-456"

    # Gemini provider should specifically select GEMINI_API_KEY
    agent_gemini = AuthorAgent(provider="gemini")
    assert agent_gemini.api_key == "gemini-secret-123"

    # Explicit api_key overrides environment
    agent_custom = AuthorAgent(provider="openrouter", api_key="custom-override")
    assert agent_custom.api_key == "custom-override"


def test_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = AuthorAgent(provider="openrouter", api_key=None)
    with pytest.raises(ValueError, match="Missing API key for provider 'openrouter'"):
        agent.synthesize_policy(
            current_policy="plan = []",
            trigger_reason="stall",
            status_report="status",
        )


def test_openrouter_http_error_parsing():
    agent = AuthorAgent(provider="openrouter", api_key="sk-test-key")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.is_error = True
    mock_resp.status_code = 401
    mock_resp.text = json.dumps({
        "error": {
            "message": "API key expired.",
            "code": 401,
        }
    })
    mock_resp.json.return_value = {
        "error": {
            "message": "API key expired.",
            "code": 401,
        }
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="OpenRouter API error \\(401\\): API key expired."):
            agent.synthesize_policy(
                current_policy="plan = []",
                trigger_reason="stall",
                status_report="status",
            )


def test_openrouter_tool_calling_and_code_synthesis():
    agent = AuthorAgent(provider="openrouter", api_key="sk-test-key")

    # Turn 1: Model requests a tool call to query_duckdb
    turn1_resp = MagicMock(spec=httpx.Response)
    turn1_resp.is_error = False
    turn1_resp.json.return_value = {
        "usage": {"prompt_tokens": 120, "completion_tokens": 25},
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_death_taxonomy",
                        "arguments": json.dumps({"window": 5})
                    }
                }]
            }
        }]
    }

    # Turn 2: Model finishes with synthesized Python policy
    turn2_resp = MagicMock(spec=httpx.Response)
    turn2_resp.is_error = False
    turn2_resp.json.return_value = {
        "usage": {"prompt_tokens": 180, "completion_tokens": 50},
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "```python\ndef combat():\n    if adjacent_hostile:\n        melee_attack_hostile()\n\nplan = [combat]\n```"
            }
        }]
    }

    with patch("httpx.Client.post", side_effect=[turn1_resp, turn2_resp]):
        code, tree, error = agent.synthesize_policy(
            current_policy="plan = []",
            trigger_reason="Cluster death",
            status_report="Status report",
            run_id="test_run",
        )

    assert error is None
    assert tree is not None
    assert "melee_attack_hostile" in code
