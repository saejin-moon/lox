"""
Single Source of Truth (SSOT) for:
1. Unpacking LLM `<policy_code>`, `<skill_code>`, `### skill:`, and JSON responses cleanly.
2. Reading and writing 1-level / 2-level `skills/` workspace packages cleanly.
3. Canonical AST scanning of `skills.*` imports and dependencies across modules.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lox.lang.validator import (
    normalize_skill_key,
    resolve_skill_import,
    skill_key_to_relpath,
    smart_dedent,
)

_TOOL_XML_ARG_RE = re.compile(
    r"<arg_key>.*?</arg_key>(?:\s*<arg_value>.*?(?:</arg_value>|(?=<(?:skill_code|policy_code|skill|policy|arg_key|tool_call)|$)))?",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_XML_WRAPPER_RE = re.compile(
    r"</?(?:tool_call|arg_key|arg_value)\b[^>]*>",
    flags=re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_META_TAG_RE = re.compile(
    r"<(?P<tag>hypothesis|reflection)>.*?</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SKILL_XML_RE = re.compile(
    r"<(?P<tag>skill_code|skill)(?P<attrs>\s+[^>]*)?>(?P<code>.*?)</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ATTR_MOD_RE = re.compile(
    r"""\b(?:module|name)\s*=\s*["']([^"']+)["']""",
    flags=re.IGNORECASE,
)
_MD_SKILL_RE = re.compile(
    r"###\s*skill:\s*([A-Za-z0-9_./-]+)\s*\n+```(?:python)?\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
_POLICY_XML_RE = re.compile(
    r"<(?P<tag>policy_code|policy)(?:\s+[^>]*)?>(?P<code>.*?)</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)


def strip_think_tags(text: str) -> str:
    """Removes `<think>...</think>` reasoning blocks from raw LLM text."""
    return _THINK_RE.sub("", str(text or "")).strip()


def strip_leaked_tool_xml(text: str) -> str:
    """Strips accidental LLM XML tool-call wrappers (`<arg_key>`, `<arg_value>`, `<tool_call>`)."""
    cleaned = _TOOL_XML_ARG_RE.sub("", str(text or ""))
    cleaned = _TOOL_XML_WRAPPER_RE.sub("", cleaned)
    return cleaned.strip()


def strip_code_fences(src: str) -> str:
    """Strips outer or embedded ```python / ``` markdown fences cleanly."""
    text = str(src or "").strip()
    exact = re.match(
        r"^```(?:python)?\s*\n(.*?)```\s*$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if exact:
        return exact.group(1).strip()
    return text


def extract_fenced_inner(src: str) -> str:
    """Extracts fenced Python inside XML blocks if present, otherwise returns text stripped of full outer fences."""
    text = str(src or "").strip()
    exact = strip_code_fences(text)
    if exact != text:
        return exact
    inner = re.search(
        r"```(?:python)?\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if inner and text.lstrip().startswith("```"):
        return inner.group(1).strip()
    return text


def extract_json_payload(raw_text: str) -> Any | None:
    """Parses a JSON object/array from raw or markdown-fenced LLM output cleanly."""
    txt = strip_think_tags(raw_text)
    if not txt:
        return None
    candidates: list[str] = [txt]
    for blk in re.findall(
        r"```(?:json)?\s*\n(.*?)```", txt, flags=re.DOTALL | re.IGNORECASE
    ):
        candidates.append(blk.strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, txt, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def _normalize_bundle_skill_key(raw_mod: str) -> str:
    try:
        return normalize_skill_key(raw_mod)
    except Exception:
        cleaned = (
            str(raw_mod or "")
            .strip()
            .replace("\\", "/")
            .removeprefix("skills/")
            .removeprefix("skills.")
            .removesuffix(".py")
            .replace(".", "/")
            .strip("/")
        )
        parts = [
            re.sub(r"[^a-zA-Z0-9_]", "_", p.lower()).strip("_")
            for p in cleaned.split("/")
            if p.strip()
        ]
        return "/".join(parts[:2])


def extract_skill_modules(
    raw_text: str,
    *,
    merge_fn: Callable[[str, str], str] | None = None,
    include_markdown: bool = True,
) -> dict[str, str]:
    """
    Extracts all `<skill_code module="...">` and optional `### skill: ...` blocks from `raw_text`.
    Merges multiple blocks targeting the same module via `merge_fn` instead of silently overwriting.
    """
    text = strip_leaked_tool_xml(strip_think_tags(raw_text))
    if not text:
        return {}
    found: dict[str, str] = {}

    def _add_block(raw_mod: str, raw_body: str) -> None:
        mod_name = _normalize_bundle_skill_key(raw_mod)
        inner = smart_dedent(
            extract_fenced_inner(strip_leaked_tool_xml(raw_body))
        ).strip()
        if not mod_name or not inner:
            return
        if mod_name in found and merge_fn is not None:
            found[mod_name] = merge_fn(found[mod_name], inner)
        else:
            found[mod_name] = inner

    for m in _SKILL_XML_RE.finditer(text):
        attrs = m.group("attrs") or ""
        mod_m = _ATTR_MOD_RE.search(attrs)
        raw_mod = mod_m.group(1) if mod_m else ""
        _add_block(raw_mod, m.group("code") or "")

    if include_markdown:
        text_no_xml = _SKILL_XML_RE.sub("", text)
        for m_name, m_body in _MD_SKILL_RE.findall(text_no_xml):
            _add_block(m_name, m_body)

    return found


def unpack_code_bundle(
    raw_code: str,
    *,
    merge_fn: Callable[[str, str], str] | None = None,
    include_markdown_skills: bool = True,
    preserve_raw_verbatim: bool = True,
) -> tuple[str, dict[str, str]]:
    """
    Extracts `(policy_or_patch_code, skill_modules_dict)` cleanly from XML bundles,
    markdown `### skill:` sections, fenced code, or raw tagless Python.
    """
    raw_str = str(raw_code or "")
    if not raw_str.strip():
        return "", {}

    # Preserve exact verbatim string (including trailing whitespace) when no XML/fences/skill headers appear.
    cleaned = _META_TAG_RE.sub("", strip_think_tags(raw_str))
    cleaned_xml = strip_leaked_tool_xml(cleaned)

    skills_found = extract_skill_modules(
        cleaned_xml,
        merge_fn=merge_fn,
        include_markdown=include_markdown_skills,
    )

    pol_matches = list(_POLICY_XML_RE.finditer(cleaned_xml))
    if pol_matches:
        pol_src = smart_dedent(
            extract_fenced_inner(strip_leaked_tool_xml(pol_matches[-1].group("code")))
        ).strip()
        return pol_src, skills_found

    if skills_found:
        rem = _SKILL_XML_RE.sub("", cleaned_xml)
        if include_markdown_skills:
            rem = _MD_SKILL_RE.sub("", rem)
        rem = strip_leaked_tool_xml(rem)
        rem_inner = smart_dedent(strip_code_fences(rem)).strip()
        return rem_inner, skills_found

    stripped_fences = strip_code_fences(cleaned_xml)
    if stripped_fences != cleaned_xml:
        return smart_dedent(stripped_fences).strip(), {}

    if preserve_raw_verbatim and cleaned_xml == raw_str.strip():
        return raw_str, {}
    return cleaned_xml, {}


def extract_policy_fallback_from_markdown(text: str) -> str:
    """
    Extracts fallback ```python block for `policy.py` after removing all `<skill_code>`
    and `### skill:` sections so skill code is never misclassified as `policy.py`.
    """
    stripped = _SKILL_XML_RE.sub("", strip_leaked_tool_xml(strip_think_tags(text)))
    stripped = _MD_SKILL_RE.sub("", stripped)
    py_blocks = re.findall(
        r"```python\s*\n(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not py_blocks:
        raw_blocks = re.findall(r"```\s*\n(.*?)```", stripped, flags=re.DOTALL)
        py_blocks = [
            b
            for b in raw_blocks
            if any(
                tok in b
                for tok in (
                    "def ",
                    "class ",
                    "ActionLeaf",
                    "ConditionLeaf",
                    "PrioritySelector",
                    "Sequence",
                    "BehaviorTree",
                    "subtree",
                )
            )
        ]
    if not py_blocks:
        return ""
    policy_candidates = [
        b for b in py_blocks if "class Policy" in b or "class Agent" in b
    ]
    chosen = policy_candidates[-1] if policy_candidates else py_blocks[-1]
    return smart_dedent(chosen.strip())


def load_skills_dir(skills_dir: str | Path) -> dict[str, str]:
    """Loads all normalized 1-level & 2-level skill modules from `skills_dir`."""
    s_dir = Path(skills_dir)
    if not s_dir.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(s_dir.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        rel = str(p.relative_to(s_dir)).replace("\\", "/")
        try:
            key = normalize_skill_key(rel)
        except Exception:
            continue
        if key:
            out[key] = p.read_text(encoding="utf-8")
    return out


def write_skills_dir(
    skills_dir: str | Path,
    skills: dict[str, str],
    *,
    sync: bool = False,
    ensure_trailing_newline: bool = True,
) -> dict[str, str]:
    """
    Writes normalized `skills` (`domain/mod` and `mod`) into `skills_dir`, creating package
    `__init__.py` files cleanly. If `sync=True`, removes stale files not in `skills`.
    """
    s_dir = Path(skills_dir)
    s_dir.mkdir(parents=True, exist_ok=True)
    root_init = s_dir / "__init__.py"
    if not root_init.exists():
        root_init.write_text("", encoding="utf-8")

    norm_skills: dict[str, str] = {}
    for raw_k, raw_v in (skills or {}).items():
        if not raw_k or not str(raw_v or "").strip():
            continue
        try:
            clean_k = normalize_skill_key(str(raw_k))
        except Exception:
            continue
        if clean_k:
            norm_skills[clean_k] = str(raw_v)

    if sync:
        for p in sorted(s_dir.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            rel = str(p.relative_to(s_dir)).replace("\\", "/")
            try:
                rel_k = normalize_skill_key(rel)
            except Exception:
                rel_k = ""
            if rel_k not in norm_skills:
                try:
                    p.unlink()
                except Exception:
                    pass

    for clean_k, code in norm_skills.items():
        rel_sub = skill_key_to_relpath(clean_k).removeprefix("skills/")
        dst = s_dir / rel_sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        if "/" in clean_k:
            pkg_init = dst.parent / "__init__.py"
            if not pkg_init.exists():
                pkg_init.write_text("", encoding="utf-8")
        txt = code.strip() + "\n" if ensure_trailing_newline else code
        dst.write_text(txt, encoding="utf-8")
    return norm_skills


@dataclass(slots=True)
class SkillImportInfo:
    """Canonical summary of `skills.*` imports inside an AST module."""

    referenced_keys: set[str] = field(default_factory=set)
    pkg_expansions: set[str] = field(default_factory=set)
    module_aliases: dict[str, str] = field(default_factory=dict)
    direct_symbols: dict[str, tuple[str, str]] = field(default_factory=dict)
    star_modules: set[str] = field(default_factory=set)
    missing_refs: list[str] = field(default_factory=list)


def extract_skill_imports(
    tree: ast.AST,
    known_keys: set[str],
    *,
    pkg_names: set[str] | None = None,
) -> SkillImportInfo:
    """
    Single SSOT scanner for `import skills.*` / `from skills... import ...` statements.
    Expands `from skills import <pkg>` and `from skills.<pkg> import *` cleanly to all
    known `<pkg>/*` submodules so pruning, topological compilation, and validation agree.
    """
    pkgs = (
        pkg_names
        if pkg_names is not None
        else {k.split("/", 1)[0] for k in known_keys if "/" in k}
    )
    info = SkillImportInfo()

    def _expand_pkg(pkg: str) -> None:
        info.pkg_expansions.add(pkg)
        for k in known_keys:
            if k.startswith(f"{pkg}/"):
                info.referenced_keys.add(k)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = str(node.module or "")
            if mod == "skills":
                for al in node.names:
                    cand = _normalize_bundle_skill_key(al.name)
                    if cand in known_keys:
                        info.referenced_keys.add(cand)
                        info.module_aliases[al.asname or al.name] = cand
                    elif cand in pkgs:
                        _expand_pkg(cand)
                    elif known_keys:
                        info.missing_refs.append(f"skills.{al.name}")
            elif mod.startswith("skills."):
                tail = mod[len("skills.") :]
                if tail in pkgs and tail not in known_keys:
                    for al in node.names:
                        if al.name == "*":
                            _expand_pkg(tail)
                            continue
                        sub_k = f"{tail}/{al.name}"
                        if sub_k in known_keys:
                            info.referenced_keys.add(sub_k)
                            info.module_aliases[al.asname or al.name] = sub_k
                        else:
                            info.missing_refs.append(f"skills.{tail}.{al.name}")
                    continue
                s_mod = resolve_skill_import(mod, known_keys)
                if s_mod in known_keys or not known_keys:
                    info.referenced_keys.add(s_mod)
                    for al in node.names:
                        if al.name == "*":
                            info.star_modules.add(s_mod)
                        else:
                            info.direct_symbols[al.asname or al.name] = (s_mod, al.name)
                else:
                    info.missing_refs.append(mod)
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.startswith("skills."):
                    tail = al.name[len("skills.") :]
                    if tail in pkgs and tail not in known_keys:
                        _expand_pkg(tail)
                        continue
                    s_mod = resolve_skill_import(al.name, known_keys)
                    if s_mod in known_keys or not known_keys:
                        info.referenced_keys.add(s_mod)
                        if al.asname:
                            info.module_aliases[al.asname] = s_mod
                    else:
                        info.missing_refs.append(al.name)
    return info
