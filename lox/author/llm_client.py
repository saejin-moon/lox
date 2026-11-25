"""
HTTP LLM client, provider-first API key resolution, safe token accounting,
and tool-output context compaction for LOX Author sessions.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from lox.ui.console import ui


def resolve_api_key(provider: str, explicit_key: str | None = None) -> str:
    """
    Resolves the API key matching `provider` first before falling back to generic env vars.
    Prevents an unrelated `OPENROUTER_API_KEY` or `OPENAI_API_KEY` from hijacking another provider.
    """
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()
    prov = str(provider or "").strip().lower()
    env_order_by_provider: dict[str, tuple[str, ...]] = {
        "openrouter": ("OPENROUTER_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }
    primary = env_order_by_provider.get(
        prov, ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
    )
    for var in primary:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    # Secondary fallback only for unknown/custom OpenAI-compatible providers
    if prov not in env_order_by_provider:
        for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
            val = os.environ.get(var)
            if val and val.strip():
                return val.strip()
    return ""


def extract_usage_tokens(usage: Any) -> tuple[int, int, int]:
    """
    Safely extracts `(prompt_tokens, completion_tokens, reasoning_tokens)` from provider
    usage dicts, handling `None`, missing keys, or `null` `completion_tokens_details`.
    """
    if not isinstance(usage, dict):
        return 0, 0, 0
    try:
        p_toks = int(usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        p_toks = 0
    try:
        c_toks = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        c_toks = 0
    details = usage.get("completion_tokens_details")
    r_raw = details.get("reasoning_tokens") if isinstance(details, dict) else 0
    if r_raw is None:
        r_raw = usage.get("reasoning_tokens") or 0
    try:
        r_toks = int(r_raw or 0)
    except (TypeError, ValueError):
        r_toks = 0
    return p_toks, c_toks, r_toks


def compact_tool_messages(
    messages: list[dict[str, Any]],
    keep_recent_turns: int = 4,
    max_old_chars: int = 600,
    head_chars: int = 350,
) -> list[dict[str, Any]]:
    """
    Returns a shallow-copied message list where the last `keep_recent_turns` assistant/tool
    exchanges remain 100% verbatim while older large `role == 'tool'` outputs are compacted
    to prevent quadratic context inflation across long ReAct sessions.
    """
    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(asst_indices) <= keep_recent_turns:
        return messages
    cutoff_idx = asst_indices[-keep_recent_turns]
    compacted: list[dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if idx < cutoff_idx and m.get("role") == "tool":
            raw_c = str(m.get("content", "") or "")
            if len(raw_c) > max_old_chars:
                t_nm = str(m.get("name", "tool") or "tool")
                compacted.append(
                    {
                        **m,
                        "content": (
                            f"{raw_c[:head_chars]}\n"
                            f"... [compacted earlier `{t_nm}` output: {len(raw_c)} chars]"
                        ),
                    }
                )
                continue
        compacted.append(m)
    return compacted


def _raise_if_auth_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 402, 403):
        err_detail = resp.text[:200]
        try:
            err_detail = resp.json().get("error", {}).get("message") or err_detail
        except Exception:
            pass
        raise PermissionError(
            f"LLM API HTTP {resp.status_code}: {err_detail} "
            "(Check your provider API key in `.env` or environment)"
        )


def call_llm_api(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    max_tokens: int,
    reasoning_effort: str,
    routing: str,
    agent_role: str,
    messages: list[dict[str, Any]],
    allow_tools: bool = True,
    tool_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatches a single chat completion request to OpenRouter, OpenAI, or Gemini."""
    prov = str(provider or "openrouter").lower()
    if prov == "openrouter":
        url = base_url or "https://openrouter.ai/api/v1/chat/completions"
    elif prov == "gemini":
        url = (
            base_url
            or "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
    else:
        url = base_url or "https://api.openai.com/v1/chat/completions"

    model_id = str(model)
    if prov == "openrouter" and ":" not in model_id:
        if routing in ("nitro", "floor", "exacto"):
            model_id = f"{model_id}:{routing}"

    api_messages = messages
    if prov in ("openrouter", "anthropic") and messages:
        api_messages = []
        for idx_m, m_item in enumerate(messages):
            if (
                idx_m == 0
                and m_item.get("role") == "system"
                and isinstance(m_item.get("content"), str)
            ):
                api_messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": m_item["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                api_messages.append(m_item)

    eff = str(reasoning_effort or "").lower() if prov == "openrouter" else ""
    effective_max_tokens = (
        min(max_tokens, 32768) if eff in ("low", "medium") else max_tokens
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": api_messages,
        "max_tokens": effective_max_tokens,
        "temperature": 0.4,
    }
    if tool_specs:
        # Keep tool definitions static across turns so provider prefix caches match on wrap-up turns.
        payload["tools"] = tool_specs
        payload["tool_choice"] = "auto" if allow_tools else "none"
    if prov == "openrouter" and eff and eff != "none":
        payload["reasoning"] = {"effort": eff}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if not api_key or api_key in ("your-api-key", "sk-or-v1-..."):
        raise PermissionError(
            f"Missing or placeholder API key for provider '{prov}'. Set the matching API key in `.env` or pass `--api-key`."
        )

    should_strip_routing = False
    for attempt in range(5):
        if attempt >= 2 and should_strip_routing:
            for suffix in (":nitro", ":floor", ":exacto"):
                if str(payload.get("model", "")).endswith(suffix):
                    payload["model"] = str(payload["model"])[: -len(suffix)]
                    break
        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                _raise_if_auth_error(resp)
                # Only downgrade/strip `reasoning` on parameter-rejection 400/422 responses
                if resp.status_code in (400, 422) and "reasoning" in payload:
                    cur_eff = str(payload["reasoning"].get("effort", "")).lower()
                    if cur_eff == "max":
                        payload["reasoning"] = {"effort": "high"}
                        resp = client.post(url, headers=headers, json=payload)
                        _raise_if_auth_error(resp)
                    if resp.status_code in (400, 422):
                        payload.pop("reasoning", None)
                        resp = client.post(url, headers=headers, json=payload)
                        _raise_if_auth_error(resp)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "error" in data:
                    err_obj = data["error"]
                    err_code = (
                        err_obj.get("code") if isinstance(err_obj, dict) else None
                    )
                    err_msg = (
                        err_obj.get("message", str(err_obj))
                        if isinstance(err_obj, dict)
                        else str(err_obj)
                    )
                    if err_code in (401, 402, 403):
                        raise PermissionError(f"LLM API Error {err_code}: {err_msg}")
                    raise ValueError(f"Upstream provider error in JSON: {err_msg}")
                choices = data.get("choices") if isinstance(data, dict) else None
                if not choices:
                    raise ValueError(
                        f"Empty choices in LLM response: {str(data)[:200]}"
                    )
                msg_obj = choices[0].get("message") or {}
                usage = data.get("usage") or {}
                msg_obj["_usage"] = usage
                reasoning = (
                    msg_obj.get("reasoning") or msg_obj.get("reasoning_content") or ""
                )
                content = msg_obj.get("content") or ""
                if not reasoning and content:
                    m_think = re.findall(
                        r"<think>(.*?)</think>", content, flags=re.DOTALL
                    )
                    if m_think:
                        reasoning = "\n\n".join(t.strip() for t in m_think)
                msg_obj["_reasoning"] = reasoning

                if not content and not msg_obj.get("tool_calls"):
                    if reasoning:
                        raise ValueError(
                            "LLM exhausted max_tokens inside reasoning without emitting content or tool_calls"
                        )
                    raise ValueError(
                        "LLM returned empty message content and empty tool_calls"
                    )
                return msg_obj
        except PermissionError:
            raise
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValueError,
            KeyError,
            IndexError,
        ) as exc:
            if isinstance(exc, (httpx.RequestError, httpx.HTTPStatusError)):
                should_strip_routing = True
            if attempt == 4:
                raise
            ui.warn(
                f"[{agent_role}] LLM API attempt {attempt + 1}/5 failed "
                f"({type(exc).__name__}: {str(exc)[:120]}); retrying..."
            )
            time.sleep(min(16.0, 2.0 * (2**attempt)))
    raise RuntimeError("Failed to obtain response from LLM API after retries.")
