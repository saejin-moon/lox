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
                "content": "```python\ndef combat():\n    if floating_eye_in_fov:\n        step_away_from_hostile()\n    elif adjacent_hostile:\n        melee_attack_hostile()\n\nplan = [combat]\n```"
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


def test_openrouter_tool_ceiling_wrap_up():
    """Verifies that hitting the tool ceiling triggers the wrap-up prompt with tools disabled."""
    agent = AuthorAgent(provider="openrouter", api_key="sk-test-key", max_tool_turns=15)

    def make_tool_resp(call_id):
        r = MagicMock(spec=httpx.Response)
        r.is_error = False
        r.json.return_value = {
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{call_id}",
                        "type": "function",
                        "function": {"name": "get_death_taxonomy", "arguments": "{}"}
                    }]
                }
            }]
        }
        return r

    tool_responses = [make_tool_resp(i) for i in range(15)]
    final_resp = MagicMock(spec=httpx.Response)
    final_resp.is_error = False
    final_resp.json.return_value = {
        "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "```python\ndef explore():\n    step_to_frontier()\n\nplan = [explore]\n```"
            }
        }]
    }

    with patch("httpx.Client.post", side_effect=tool_responses + [final_resp]) as mock_post:
        code, tree, error = agent.synthesize_policy(
            current_policy="plan = []",
            trigger_reason="Stall",
            status_report="Status",
            run_id="test_ceiling",
        )

    assert error is None
    assert tree is not None
    assert "step_to_frontier" in code
    # Ensure 15 tool turns + 1 final wrap-up turn were executed
    assert mock_post.call_count == 16
    # Verify tools were omitted on the 16th turn
    last_call_payload = mock_post.call_args_list[-1][1]["json"]
    assert "tools" not in last_call_payload


def test_openrouter_ast_self_repair():
    """Verifies that an AST syntax error triggers self-repair feedback to the model."""
    agent = AuthorAgent(provider="openrouter", api_key="sk-test-key")

    # Turn 1: Model outputs code with an illegal predicate 'invalid_pred_xyz'
    bad_resp = MagicMock(spec=httpx.Response)
    bad_resp.is_error = False
    bad_resp.json.return_value = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "```python\ndef combat():\n    if invalid_pred_xyz:\n        melee_attack_hostile()\n\nplan = [combat]\n```"
            }
        }]
    }

    # Repair Turn: Model receives AST validation feedback and fixes the predicate
    repair_resp = MagicMock(spec=httpx.Response)
    repair_resp.is_error = False
    repair_resp.json.return_value = {
        "usage": {"prompt_tokens": 150, "completion_tokens": 30},
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "```python\ndef combat():\n    if floating_eye_in_fov:\n        step_away_from_hostile()\n    elif adjacent_hostile:\n        melee_attack_hostile()\n\nplan = [combat]\n```"
            }
        }]
    }

    with patch("httpx.Client.post", side_effect=[bad_resp, repair_resp]) as mock_post:
        code, tree, error = agent.synthesize_policy(
            current_policy="plan = []",
            trigger_reason="Fix bug",
            status_report="Status",
            run_id="test_repair",
        )

    assert error is None
    assert tree is not None
    assert "adjacent_hostile" in code
    assert mock_post.call_count == 2
