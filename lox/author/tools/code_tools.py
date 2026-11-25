"""
Workspace code inspection, skill reading, AST symbol lookup, diffing, and skill-merge tools.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any

from lox.author.prompts import FULL_ACTION_DOCS, POLICY_LANG_DOCS
from lox.db.schema import connect_db
from lox.lang.bundle import unpack_code_bundle
from lox.lang.validator import (
    PolicyValidationError,
    normalize_skill_key,
    smart_dedent,
    splice_functions_into_policy,
)


def merge_skill_module_update(existing_src: str, candidate_src: str) -> str:
    """
    Merges an edited skill module safely into `existing_src`.
    If `candidate_src` is a partial function/constant patch (omits existing imports,
    starts with a patch comment, or drops more than half of `existing_src`'s top-level
    definitions), splices it cleanly via `splice_functions_into_policy` so existing
    exports are never wiped out.
    """
    cand = smart_dedent(str(candidate_src or "")).strip()
    if not existing_src or not existing_src.strip():
        return cand
    if not cand:
        return existing_src

    try:
        ex_tree = ast.parse(existing_src)
        ca_tree = ast.parse(cand)
    except Exception:
        return cand

    ex_has_imports = any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ex_tree.body
    )
    ca_has_imports = any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ca_tree.body
    )
    ex_defs = {
        n.name
        for n in ex_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    ca_defs = {
        n.name
        for n in ca_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    ex_assigns = {
        t.id
        for n in ex_tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {
        n.target.id
        for n in ex_tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    ca_assigns = {
        t.id
        for n in ca_tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {
        n.target.id
        for n in ca_tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    first_line = cand.splitlines()[0].strip().lower() if cand.splitlines() else ""
    if first_line.startswith("# replace") or first_line.startswith("# full replace"):
        return cand
    looks_patch_comment = (
        first_line.startswith("# patch")
        or "other functions unchanged" in cand[:300].lower()
    )
    ca_has_docstring = ast.get_docstring(ca_tree) is not None
    dropped_most_defs = len(ex_defs) >= 3 and len(ca_defs) < max(2, len(ex_defs) // 2)
    omitted_existing = bool((ex_defs - ca_defs) | (ex_assigns - ca_assigns))
    overlaps_existing = bool((ex_defs & ca_defs) or (ex_assigns & ca_assigns))
    looks_full_rewrite = (
        not looks_patch_comment
        and (ca_has_docstring or ca_has_imports)
        and (ca_has_imports or not ex_has_imports)
        and (len(ca_assigns) > 0 or len(ex_assigns) == 0 or ca_has_docstring)
        and not dropped_most_defs
        and (
            (ca_has_docstring and len(ca_defs) >= 2 and not (ca_defs < ex_defs))
            or (
                len(ca_defs) >= max(2, len(ex_defs) - 1)
                and (ca_has_docstring or not (ca_defs < ex_defs))
            )
        )
    )
    if looks_full_rewrite:
        return cand

    partial_subset = (
        (len(ex_defs) >= 2 and 0 < len(ca_defs) < len(ex_defs) and ca_defs <= ex_defs)
        or (len(ex_defs) >= 1 and len(ca_defs) == 0 and len(ca_assigns) > 0)
        or (omitted_existing and overlaps_existing and len(ex_defs | ex_assigns) >= 2)
    )
    missing_imports = (
        ex_has_imports
        and not ca_has_imports
        and (len(ca_defs) < len(ex_defs) or len(ex_defs) == 0 or omitted_existing)
    )
    missing_constants = (
        len(ex_assigns) > 0
        and len(ca_assigns) == 0
        and (len(ca_defs) <= len(ex_defs) or omitted_existing)
    )

    def _imported_skill_modules(tr: ast.AST) -> set[str]:
        mods: set[str] = set()
        for nd in getattr(tr, "body", []):
            if isinstance(nd, ast.ImportFrom) and str(nd.module or "").startswith(
                "skills."
            ):
                mods.add(str(nd.module))
            elif isinstance(nd, ast.Import):
                for alias in nd.names:
                    if str(alias.name).startswith("skills."):
                        mods.add(str(alias.name))
        return mods

    new_skill_imports = _imported_skill_modules(ca_tree) - _imported_skill_modules(
        ex_tree
    )
    if new_skill_imports and not looks_patch_comment:
        return cand

    if (
        looks_patch_comment
        or dropped_most_defs
        or partial_subset
        or missing_imports
        or missing_constants
    ):
        try:
            spliced, _ = splice_functions_into_policy(existing_src, cand)
            return spliced
        except Exception:
            return cand
    return cand


def unpack_xml_bundle(raw_code: str) -> tuple[str, dict[str, str]]:
    """
    Extracts `<skill_code module="name">...</skill_code>` blocks and `<policy_code>...</policy_code>`
    from `raw_code`. Delegates to canonical `lox.lang.bundle.unpack_code_bundle`.
    """
    return unpack_code_bundle(
        raw_code,
        merge_fn=merge_skill_module_update,
        include_markdown_skills=False,
        preserve_raw_verbatim=True,
    )


class CodeToolsMixin:
    """Code reading, diffing, symbol lookup, grep, and Option-A architecture map tools."""

    db_path: str
    workspace_dir: Path
    current_policy_code: str
    skills: dict[str, str]
    mode: str

    def query_policy_docs(self, topic: str = "all") -> tuple[str, str]:
        """Returns policy language specs, current `policy.py`, and available `skills/`."""
        skill_names = sorted(self.skills.keys())
        if not skill_names:
            skills_dir = self.workspace_dir / "skills"
            if skills_dir.exists():
                for p in sorted(skills_dir.rglob("*.py")):
                    if p.name == "__init__.py":
                        continue
                    try:
                        rel_k = normalize_skill_key(
                            str(p.relative_to(skills_dir)).replace("\\", "/")
                        )
                    except PolicyValidationError:
                        continue
                    if rel_k and rel_k not in skill_names:
                        skill_names.append(rel_k)
        skills_section = (
            f"\n### Available `skills/` Modules ({len(skill_names)}): {', '.join(sorted(skill_names)) or '(none)'}\n"
            f"(Call `read_skill(skill_name)` to inspect full source of any module.)"
        )
        t = (topic or "all").lower()
        if "policy" in t:
            pol_src = self.current_policy_code or "(Empty policy)"
            return (
                f"### Current `policy.py` Full Source:\n```python\n{pol_src}\n```{skills_section}",
                "policy source",
            )
        if "skill" in t:
            return skills_section.strip(), f"{len(skill_names)} skills"
        if "action" in t:
            return FULL_ACTION_DOCS, "action docs"
        pol_src = self.current_policy_code or "(Empty policy)"
        return (
            POLICY_LANG_DOCS
            + skills_section
            + f"\n### Current `policy.py` Source:\n```python\n{pol_src}\n```",
            "full docs",
        )

    @staticmethod
    def _lookup_policy_snapshot(
        con: Any, pid: str
    ) -> tuple[int, str, float, int, bool, str, str] | None:
        """Looks up a policy snapshot by `policy_id` in `policies`, falling back to `archive_policies`."""
        try:
            row = con.execute(
                "SELECT generation, mode, avg_depth, max_depth, is_promoted, policy_code, skills_json FROM policies WHERE policy_id = ?",
                [pid],
            ).fetchone()
        except Exception:
            try:
                row_legacy = con.execute(
                    "SELECT generation, mode, avg_depth, max_depth, is_promoted, policy_code FROM policies WHERE policy_id = ?",
                    [pid],
                ).fetchone()
                row = (*row_legacy, "{}") if row_legacy else None
            except Exception:
                row = None
        if row is not None:
            return row
        try:
            arow = con.execute(
                """
                SELECT generation, policy_code, skills_json, metrics_json
                FROM archive_policies
                WHERE cell_id = ? OR batch_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                [pid, pid],
            ).fetchone()
        except Exception:
            arow = None
        if not arow:
            return None
        gen, p_code, s_json, m_json = arow
        try:
            metrics = json.loads(m_json or "{}") if isinstance(m_json, str) else {}
        except Exception:
            metrics = {}
        return (
            int(gen or 0),
            "MAP_ELITES",
            float(metrics.get("avg_depth", 1.0) or 1.0),
            int(metrics.get("max_depth", 1) or 1),
            False,
            str(p_code or ""),
            str(s_json or "{}"),
        )

    def read_skill(
        self,
        skill_name: str,
        policy_id: str = "",
        start_line: int = 1,
        end_line: int | None = None,
    ) -> tuple[str, str]:
        """Reads a skill module (or line range `start_line..end_line`) from the workspace or a `policy_id` snapshot."""
        try:
            clean_name = normalize_skill_key(str(skill_name or ""))
        except (PolicyValidationError, ValueError, TypeError) as exc:
            return f"Error: Invalid skill name '{skill_name}': {exc}", "error"
        if not clean_name:
            return "Error: `skill_name` must be non-empty.", "error"

        def _slice_lines(src_text: str, tag: str) -> tuple[str, str]:
            all_lines = src_text.splitlines()
            total_l = len(all_lines)
            s_l = max(1, int(start_line or 1))
            e_l = min(total_l, int(end_line)) if end_line is not None else total_l
            if s_l == 1 and end_line is None:
                return src_text, f"{total_l} lines{tag}"
            if s_l > total_l:
                return (
                    f"(File `{clean_name}` has only {total_l} lines.)",
                    f"0 lines{tag}",
                )
            numbered = [
                f"{idx}: {all_lines[idx - 1]}" for idx in range(s_l, max(s_l, e_l) + 1)
            ]
            return (
                f"# {clean_name}.py lines {s_l}-{max(s_l, e_l)} of {total_l}\n"
                + "\n".join(numbered),
                f"lines {s_l}-{max(s_l, e_l)}/{total_l}{tag}",
            )

        pid = str(policy_id or "").strip()
        if pid:
            if not os.path.exists(self.db_path):
                return "Database not found.", "not_found"
            con = connect_db(self.db_path, read_only=True)
            try:
                snap = self._lookup_policy_snapshot(con, pid)
                if not snap:
                    return f"Error: Policy '{pid}' not found.", "not_found"
                try:
                    skills_map = json.loads(snap[6] or "{}") or {}
                except Exception:
                    skills_map = {}
                if clean_name in skills_map:
                    return _slice_lines(str(skills_map[clean_name]), f" ({pid})")
                return (
                    f"Error: Skill '{clean_name}' not found in policy '{pid}'. Available skills in '{pid}': {sorted(skills_map.keys()) or 'None'}",
                    "not_found",
                )
            finally:
                con.close()

        if clean_name in self.skills:
            return _slice_lines(self.skills[clean_name], "")

        # When an isolated in-memory skills dict is attached (e.g. proactive branch or peer snapshot),
        # do NOT fall back to global workspace disk files that might differ or leak out-of-snapshot state.
        if self.skills:
            return (
                f"Error: Skill '{clean_name}' not found. Available skills: {sorted(self.skills.keys())}",
                "not_found",
            )

        skills_dir = (self.workspace_dir / "skills").resolve()
        skill_file = (skills_dir / f"{clean_name}.py").resolve()
        if not skill_file.is_relative_to(skills_dir):
            return (
                f"Error: Skill path '{skill_name}' escapes workspace skills directory.",
                "error",
            )
        if skill_file.exists() and skill_file.is_file():
            content = skill_file.read_text(encoding="utf-8")
            return _slice_lines(content, "")

        avail: list[str] = []
        if skills_dir.exists():
            avail = [
                str(p.relative_to(skills_dir)).removesuffix(".py").replace("\\", "/")
                for p in skills_dir.rglob("*.py")
                if p.name != "__init__.py"
            ]
        return (
            f"Error: Skill '{clean_name}' not found. Available skills: {sorted(set(avail)) or 'None'}",
            "not_found",
        )

    def read_policy(self, policy_id: str = "") -> tuple[str, str]:
        """Reads current `policy.py` (when empty/`current`) or a historical `policy_id` snapshot."""
        pid = str(policy_id or "").strip()
        if not pid or pid.lower() in ("current", "workspace", "incumbent", "policy.py"):
            cur = str(self.current_policy_code or "")
            if cur.strip():
                return cur, f"{len(cur.splitlines())} lines (current)"
            ws_pol = self.workspace_dir / "policy.py"
            if ws_pol.exists():
                cur = ws_pol.read_text(encoding="utf-8")
                return cur, f"{len(cur.splitlines())} lines (current)"
            return "(Empty current policy.py)", "0 lines"
        if not os.path.exists(self.db_path):
            return "Database not found.", "not_found"
        con = connect_db(self.db_path, read_only=True)
        try:
            row = self._lookup_policy_snapshot(con, pid)
            if not row:
                return f"Error: Policy '{policy_id}' not found.", "not_found"
            gen, mode, avg_d, max_d, prom, p_code, s_json = row
            try:
                skills_map = json.loads(s_json or "{}")
            except Exception:
                skills_map = {}
            skill_names = (
                sorted(skills_map.keys()) if isinstance(skills_map, dict) else []
            )
            tip = (
                f"# Tip: call read_skill('<skill_name>', policy_id='{policy_id.strip()}') to read full source of any skill from this snapshot.\n"
                if skill_names
                else ""
            )
            header = (
                f"# policy_id: {policy_id} | generation: {gen} | mode: {mode} "
                f"| avg_depth: {avg_d} | max_depth: {max_d} | is_promoted: {prom}\n"
                f"# skills_snapshot: {skill_names}\n"
                f"{tip}\n"
            )
            return header + str(
                p_code or ""
            ), f"{len(str(p_code or '').splitlines())} lines"
        finally:
            con.close()

    def diff_policies(self, policy_id_a: str, policy_id_b: str) -> tuple[str, str]:
        """Returns unified diffs of `policy_code` and all `skills_json` modules between two `policy_id`s."""
        if not os.path.exists(self.db_path):
            return "Database not found.", "not_found"
        con = connect_db(self.db_path, read_only=True)
        try:
            snap_a = self._lookup_policy_snapshot(con, policy_id_a.strip())
            snap_b = self._lookup_policy_snapshot(con, policy_id_b.strip())
            row_a = (snap_a[5], snap_a[6]) if snap_a else None
            row_b = (snap_b[5], snap_b[6]) if snap_b else None
            if not row_a:
                return f"Error: Policy '{policy_id_a}' not found.", "not_found"
            if not row_b:
                return f"Error: Policy '{policy_id_b}' not found.", "not_found"

            diffs: list[str] = []
            p_diff = list(
                difflib.unified_diff(
                    str(row_a[0] or "").splitlines(),
                    str(row_b[0] or "").splitlines(),
                    fromfile=f"{policy_id_a}/policy.py",
                    tofile=f"{policy_id_b}/policy.py",
                    lineterm="",
                    n=2,
                )
            )
            if p_diff:
                if len(p_diff) > 500:
                    diffs.append(
                        "\n".join(p_diff[:500])
                        + f"\n... [TRUNCATED {len(p_diff) - 500} diff lines] ..."
                    )
                else:
                    diffs.append("\n".join(p_diff))
            try:
                sk_a = json.loads(row_a[1] or "{}") or {}
                sk_b = json.loads(row_b[1] or "{}") or {}
            except Exception:
                sk_a, sk_b = {}, {}
            for sk_name in sorted(set(sk_a) | set(sk_b)):
                s_diff = list(
                    difflib.unified_diff(
                        str(sk_a.get(sk_name, "")).splitlines(),
                        str(sk_b.get(sk_name, "")).splitlines(),
                        fromfile=f"{policy_id_a}/skills/{sk_name}.py",
                        tofile=f"{policy_id_b}/skills/{sk_name}.py",
                        lineterm="",
                        n=2,
                    )
                )
                if s_diff:
                    if len(s_diff) > 300:
                        diffs.append(
                            "\n".join(s_diff[:300])
                            + f"\n... [TRUNCATED {len(s_diff) - 300} diff lines] ..."
                        )
                    else:
                        diffs.append("\n".join(s_diff))
            out = (
                "\n\n".join(diffs)
                if diffs
                else "No differences between policy snapshots."
            )
            if len(out) > 32000:
                out = (
                    out[:32000]
                    + f"\n... [TRUNCATED {len(out) - 32000} chars; read individual files via read_policy/read_skill] ..."
                )
            return out, f"{len(diffs)} changed files"
        finally:
            con.close()

    def _iter_workspace_sources(self) -> list[tuple[str, str]]:
        """Returns `[(rel_path, source_text), ...]` for active `policy.py` and `skills/**/*.py`."""
        files: list[tuple[str, str]] = []
        pol = str(self.current_policy_code or "").strip()
        if pol:
            files.append(("policy.py", self.current_policy_code))
        seen_sk: set[str] = set()
        for k in sorted(self.skills.keys()):
            src = str(self.skills.get(k) or "")
            if src.strip():
                clean_k = str(k).replace(".", "/").strip("/")
                files.append((f"skills/{clean_k}.py", src))
                seen_sk.add(clean_k)
        if not self.skills:
            skills_dir = self.workspace_dir / "skills"
            if skills_dir.exists():
                for p in sorted(skills_dir.rglob("*.py")):
                    if p.name == "__init__.py":
                        continue
                    rel_k = (
                        str(p.relative_to(skills_dir))
                        .removesuffix(".py")
                        .replace("\\", "/")
                    )
                    if rel_k not in seen_sk:
                        try:
                            files.append(
                                (f"skills/{rel_k}.py", p.read_text(encoding="utf-8"))
                            )
                        except Exception:
                            pass
        return files

    def read_symbol(self, qualified_name: str) -> tuple[str, str]:
        """Reads an exact top-level function, class, or `Class.method` AST definition with line numbers."""
        q = str(qualified_name or "").strip()
        if not q:
            return "Error: `qualified_name` must be non-empty.", "error"
        sources = self._iter_workspace_sources()
        known_keys = {
            rel_path.removeprefix("skills/").removesuffix(".py"): rel_path
            for rel_path, _ in sources
        }
        mod_hint = ""
        cls_hint = ""
        sym_name = q
        q_norm = q.replace(".py.", ".")
        if ":" in q_norm:
            mod_hint, rest = q_norm.split(":", 1)
            sym_name = rest.strip()
        elif "/" in q_norm and "." in q_norm:
            mod_hint, sym_name = q_norm.split(".", 1)
        elif q_norm.count(".") >= 1:
            q_dot = q_norm.removeprefix("skills.")
            matched_key = ""
            for key in sorted(known_keys.keys(), key=len, reverse=True):
                dot_key = key.replace("/", ".")
                if q_dot.startswith(dot_key + "."):
                    matched_key = key
                    sym_name = q_dot[len(dot_key) + 1 :]
                    break
            if matched_key:
                mod_hint = matched_key
            elif q_norm.count(".") >= 2:
                parts = q_norm.rsplit(".", 1)
                mod_hint, sym_name = parts[0], parts[1]
        if "." in sym_name:
            cls_hint, sym_name = sym_name.split(".", 1)

        mod_hint_norm = (
            mod_hint.strip()
            .removeprefix("skills/")
            .removeprefix("skills.")
            .removesuffix(".py")
            .replace(".", "/")
        )
        matches: list[str] = []
        for rel_path, src in sources:
            rel_key = rel_path.removeprefix("skills/").removesuffix(".py")
            if mod_hint_norm and mod_hint_norm not in (rel_key, rel_path):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            src_lines = src.splitlines()
            for node in tree.body:
                if (
                    cls_hint
                    and isinstance(node, ast.ClassDef)
                    and node.name == cls_hint
                ):
                    for sub in node.body:
                        if (
                            isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and sub.name == sym_name
                        ):
                            s_l = int(sub.lineno)
                            e_l = int(getattr(sub, "end_lineno", s_l) or s_l)
                            body = "\n".join(
                                f"{i}: {src_lines[i - 1]}" for i in range(s_l, e_l + 1)
                            )
                            matches.append(
                                f"### {rel_path}:{s_l}-{e_l} (`{cls_hint}.{sym_name}`)\n{body}"
                            )
                elif (
                    not cls_hint
                    and isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and node.name == sym_name
                ):
                    s_l = int(node.lineno)
                    e_l = int(getattr(node, "end_lineno", s_l) or s_l)
                    body = "\n".join(
                        f"{i}: {src_lines[i - 1]}" for i in range(s_l, e_l + 1)
                    )
                    matches.append(f"### {rel_path}:{s_l}-{e_l} (`{sym_name}`)\n{body}")
        if not matches:
            return f"Symbol `{q}` not found in `policy.py` or `skills/`.", "not_found"
        return "\n\n".join(matches[:4]), f"{len(matches)} match(es)"

    def find_references(self, symbol: str) -> tuple[str, str]:
        """Finds definitions, imports, and call/attribute sites for a symbol across `policy.py` and `skills/`."""
        sym = str(symbol or "").strip()
        if not sym:
            return "Error: `symbol` must be non-empty.", "error"
        pat = re.compile(rf"\b{re.escape(sym)}\b")
        defs: list[str] = []
        refs: list[str] = []
        for rel_path, src in self._iter_workspace_sources():
            for idx, ln in enumerate(src.splitlines(), start=1):
                if not pat.search(ln):
                    continue
                stripped = ln.strip()
                if re.match(
                    rf"^(?:async\s+)?def\s+{re.escape(sym)}\b", stripped
                ) or re.match(rf"^class\s+{re.escape(sym)}\b", stripped):
                    defs.append(f"  [DEF] {rel_path}:{idx}: {stripped[:160]}")
                else:
                    refs.append(f"  {rel_path}:{idx}: {stripped[:160]}")
        total = len(defs) + len(refs)
        if total == 0:
            return f"No references found for `{sym}` across policy/skills.", "0 refs"
        out_lines = [
            f"### References for `{sym}` ({len(defs)} defs, {len(refs)} uses):"
        ]
        out_lines.extend(defs[:20])
        out_lines.extend(refs[:60])
        if len(refs) > 60:
            out_lines.append(f"  ... ({len(refs) - 60} more uses)")
        return "\n".join(out_lines), f"{total} refs"

    def grep_skills(self, pattern: str) -> tuple[str, str]:
        """Searches regex/text pattern across `policy.py` and `skills/**/*.py`."""
        pat_s = str(pattern or "").strip()
        if not pat_s:
            return "Error: `pattern` must be non-empty.", "error"
        try:
            rx = re.compile(pat_s, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pat_s), re.IGNORECASE)
        hits: list[str] = []
        for rel_path, src in self._iter_workspace_sources():
            for idx, ln in enumerate(src.splitlines(), start=1):
                if rx.search(ln):
                    hits.append(f"  {rel_path}:{idx}: {ln.strip()[:160]}")
        if not hits:
            return f"No matches for `{pat_s}` in policy/skills.", "0 matches"
        shown = hits[:80]
        if len(hits) > 80:
            shown.append(f"  ... ({len(hits) - 80} more matches)")
        return "\n".join(shown), f"{len(hits)} matches"

    def architecture_map(self) -> tuple[str, str]:
        """AST-indexed map of `register_goal(...)` GoalSpec handlers, `ws.domain(...)` typed states, and module LOC."""
        goals_found: list[tuple[str, str, int, str]] = []
        domains_found: dict[str, dict[str, Any]] = {}
        module_summary: list[str] = []

        for rel_path, src in self._iter_workspace_sources():
            loc = len(src.splitlines())
            module_summary.append(f"  - `{rel_path}` ({loc} LOC)")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn_nm = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (
                            node.func.attr
                            if isinstance(node.func, ast.Attribute)
                            else ""
                        )
                    )
                    if fn_nm == "register_goal":
                        g_name = ""
                        prio = "50"
                        if node.args and isinstance(node.args[0], ast.Constant):
                            g_name = str(node.args[0].value)
                        for kw in node.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                g_name = str(kw.value.value)
                            elif kw.arg == "priority" and isinstance(
                                kw.value, ast.Constant
                            ):
                                prio = str(kw.value.value)
                        if g_name:
                            goals_found.append(
                                (
                                    prio,
                                    g_name,
                                    int(getattr(node, "lineno", 1)),
                                    rel_path,
                                )
                            )
                    elif (
                        fn_nm == "domain"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                    ):
                        d_name = str(node.args[0].value)
                        cls_nm = ""
                        if len(node.args) >= 2:
                            a1 = node.args[1]
                            cls_nm = (
                                a1.id
                                if isinstance(a1, ast.Name)
                                else (a1.attr if isinstance(a1, ast.Attribute) else "")
                            )
                        entry = domains_found.setdefault(
                            d_name, {"class": cls_nm, "sites": set()}
                        )
                        if cls_nm and not entry["class"]:
                            entry["class"] = cls_nm
                        entry["sites"].add(f"{rel_path}:{getattr(node, 'lineno', 1)}")

        lines: list[str] = ["### Architecture Map (Option A)"]
        lines.append(f"**Modules ({len(module_summary)}):**")
        lines.extend(module_summary[:40])
        if len(module_summary) > 40:
            lines.append(f"  ... (+{len(module_summary) - 40} more)")
        lines.append(f"\n**Registered Goals ({len(goals_found)}):**")
        if goals_found:
            for prio, g_name, ln, rp in sorted(
                goals_found,
                key=lambda t: (int(t[0]) if t[0].lstrip("-").isdigit() else 999, t[1]),
            ):
                lines.append(f"  - prio={prio} `{g_name}` @ `{rp}:{ln}`")
        else:
            lines.append("  (none registered yet via `register_goal(...)`)")
        lines.append(f"\n**Typed `ws.domain(...)` Sub-States ({len(domains_found)}):**")
        if domains_found:
            for d_name, info in sorted(domains_found.items()):
                cls_lbl = info["class"] or "?"
                sites = ", ".join(sorted(info["sites"])[:5])
                lines.append(f"  - `{d_name}` ({cls_lbl}) used at [{sites}]")
        else:
            lines.append("  (none registered yet)")
        return "\n".join(
            lines
        ), f"{len(goals_found)} goals, {len(domains_found)} domains"
