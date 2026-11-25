"""
Safe parsing utilities for LLM tool arguments, YAML hypotheses, `<policy_code>`,
and `<skill_code module="...">` payloads.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import yaml

from lox.author.tools.code_tools import merge_skill_module_update
from lox.lang.bundle import (
    extract_fenced_inner,
    extract_json_payload,
    extract_policy_fallback_from_markdown,
    extract_skill_modules,
    strip_think_tags,
    unpack_code_bundle,
)
from lox.lang.validator import (
    PolicyValidationError,
    normalize_skill_key,
    smart_dedent,
)


def safe_as_int(val: Any, default: int) -> int:
    """Safely coerces LLM tool integer arguments, returning `default` on `None`/malformed values."""
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_normalize_skill_key(raw: Any) -> str:
    """Normalizes a skill module key, returning `""` instead of raising on malformed LLM keys."""
    try:
        return normalize_skill_key(str(raw or ""))
    except (PolicyValidationError, ValueError, TypeError):
        return ""


def sanitize_skill_code(code: str) -> str:
    """Strips leading non-python markdown headers (e.g. `**Block 2 — skills/...**:`) and artifacts."""
    lines = code.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith("**")
            or stripped.startswith("##")
            or stripped.startswith("Block ")
        ):
            continue
        start_idx = i
        break
    return "\n".join(lines[start_idx:])


def parse_tool_args(raw_args: Any) -> tuple[dict[str, Any], str | None]:
    """Parses OpenAI tool arguments JSON; returns `(args_dict, error_message)` failing closed on malformed JSON."""
    if raw_args is None or raw_args == "":
        return {}, None
    if isinstance(raw_args, dict):
        return dict(raw_args), None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception as exc:
            return (
                {},
                f"TOOL_ARG_ERROR: Malformed JSON arguments ({exc}). Re-emit valid JSON.",
            )
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "TOOL_ARG_ERROR: Tool arguments must be a JSON object."
    return {}, "TOOL_ARG_ERROR: Unsupported tool arguments type."


def extract_yaml_and_python(content: str) -> tuple[dict[str, Any], str]:
    """Extracts the YAML hypothesis dict and Python code block from an LLM response."""
    text = strip_think_tags(content or "")

    yaml_dict: dict[str, Any] = {}
    hyp_match = re.search(
        r"<hypothesis>(.*?)</hypothesis>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if hyp_match:
        try:
            parsed = yaml.safe_load(hyp_match.group(1).strip())
            if isinstance(parsed, dict):
                yaml_dict = parsed
        except Exception:
            pass

    if not yaml_dict:
        yaml_blocks = re.findall(
            r"```(?:yaml|yml)\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
        )
        for yb in yaml_blocks:
            try:
                parsed = yaml.safe_load(yb)
                if isinstance(parsed, dict):
                    yaml_dict = parsed
                    break
            except Exception:
                continue

    if yaml_dict:
        if "target_functions" not in yaml_dict:
            t_funcs = (
                yaml_dict.get("target_function") or yaml_dict.get("functions") or []
            )
            if isinstance(t_funcs, str):
                t_funcs = [t_funcs]
            yaml_dict["target_functions"] = list(t_funcs)
        elif isinstance(yaml_dict["target_functions"], str):
            yaml_dict["target_functions"] = [yaml_dict["target_functions"]]

        if "binding_constraint" not in yaml_dict:
            yaml_dict["binding_constraint"] = str(
                yaml_dict.get("observation_from_db")
                or yaml_dict.get("observation")
                or ""
            )

        if "verified_c_rule" not in yaml_dict:
            yaml_dict["verified_c_rule"] = str(
                yaml_dict.get("c_rule") or yaml_dict.get("wiki_rule") or ""
            )

        if "mechanism" not in yaml_dict:
            yaml_dict["mechanism"] = str(
                yaml_dict.get("description") or yaml_dict.get("hypothesis") or ""
            )

        if "state_persistence" not in yaml_dict:
            yaml_dict["state_persistence"] = str(
                yaml_dict.get("persistence") or yaml_dict.get("memory") or ""
            )

    code_str = ""
    pol_match = re.search(
        r"<(?:policy_code|policy)(?:\s+[^>]*)?>(.*?)</(?:policy_code|policy)>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if pol_match:
        code_str, _ = unpack_code_bundle(
            pol_match.group(0),
            include_markdown_skills=False,
            preserve_raw_verbatim=False,
        )
    else:
        code_str = extract_policy_fallback_from_markdown(text)

    if code_str:
        code_str = smart_dedent(code_str)

    return yaml_dict, code_str


def extract_all_skill_codes(
    content: str,
    hyp: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Extracts all `<skill_code module="name">...</skill_code>` and `### skill: <name>` blocks."""
    raw_found = extract_skill_modules(
        content or "",
        merge_fn=merge_skill_module_update,
        include_markdown=True,
    )
    out: dict[str, str] = {}
    for k, v in raw_found.items():
        nk = safe_normalize_skill_key(k)
        s_clean = sanitize_skill_code(v.strip())
        if nk and s_clean:
            out[nk] = smart_dedent(s_clean)

    hyp_mod = str((hyp or {}).get("skill_module") or "").strip()
    if hyp_mod:
        text = strip_think_tags(content or "")
        for m in re.finditer(
            r"<(?:skill_code|skill)(?!\s+(?:module|name)=)[^>]*>(?P<code>.*?)</(?:skill_code|skill)>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            s_clean = sanitize_skill_code(extract_fenced_inner(m.group("code")))
            nk = safe_normalize_skill_key(hyp_mod)
            if nk and s_clean:
                out[nk] = smart_dedent(s_clean)
    return out


def extract_skill_code(
    content: str,
    hyp: dict[str, Any],
    fallback_skill_code: str = "",
    fallback_skill_name: str | None = None,
) -> str:
    """Extracts external skill module code if specified in hypothesis, XML tags, or markdown."""
    text = strip_think_tags(content or "")
    extracted = extract_all_skill_codes(text, hyp)

    req_mod = hyp.get("skill_module") or fallback_skill_name
    if req_mod:
        norm_req = safe_normalize_skill_key(req_mod)
        if norm_req and norm_req in extracted:
            if "skill_module" not in hyp:
                hyp["skill_module"] = norm_req
            return extracted[norm_req]
    if extracted:
        first_mod, first_src = next(iter(extracted.items()))
        if not hyp.get("skill_module"):
            hyp["skill_module"] = first_mod
        return first_src

    skill_mod = hyp.get("skill_module") or fallback_skill_name
    if skill_mod and "skill_module" not in hyp:
        hyp["skill_module"] = skill_mod
    if not skill_mod:
        return fallback_skill_code

    if hyp.get("skill_code"):
        s_clean = sanitize_skill_code(str(hyp["skill_code"]).strip())
        return smart_dedent(s_clean)

    m = re.search(
        rf"```(?:python:)?(?:skill|{re.escape(str(skill_mod))})\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        s_clean = sanitize_skill_code(m.group(1).strip())
        return smart_dedent(s_clean)

    py_blocks = re.findall(
        r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if len(py_blocks) >= 2:
        for b in py_blocks:
            if (
                f"module: {skill_mod}" in b.lower()
                or f"skills/{skill_mod}" in b.lower()
            ):
                s_clean = sanitize_skill_code(b.strip())
                return smart_dedent(s_clean)
        for b in py_blocks:
            if "class Policy" not in b and ("def " in b or "class " in b):
                s_clean = sanitize_skill_code(b.strip())
                return smart_dedent(s_clean)

    return fallback_skill_code


def inspect_python_fragment(src: str) -> tuple[bool, bool]:
    """Returns `(has_executable_stmts, has_policy_class)` without false matches inside comments."""
    if not src or not src.strip():
        return False, False
    try:
        tree = ast.parse(src)
    except Exception:
        return False, False
    has_stmts = len(tree.body) > 0
    has_cls = any(
        isinstance(n, ast.ClassDef) and n.name in ("Policy", "Agent") for n in tree.body
    )
    return has_stmts, has_cls


def has_executable_skill_input(skills_arg: Any) -> bool:
    return isinstance(skills_arg, dict) and any(
        str(k or "").strip() and str(v or "").strip() for k, v in skills_arg.items()
    )


def has_valid_json(text: str, expected_type: str = "object") -> bool:
    """Checks if `text` contains a valid JSON object (`{...}`) or array (`[...]`)."""
    parsed = extract_json_payload(text)
    if parsed is None:
        return False
    if expected_type == "array":
        return isinstance(parsed, list) or (
            isinstance(parsed, dict) and isinstance(parsed.get("categories"), list)
        )
    return isinstance(parsed, dict)
