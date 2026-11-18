"""
Deterministic, domain-agnostic AST Behavior-Tree policy modularizer.

Splits a large `policy.py` into:
  - shared helper/constant module (`skills/shared.py` or `skills/shared_ext*.py`)
  - balanced subtree leaf modules (`skills/subtree_<k>_<root>.py`) grouped by call graph
  - thin `policy.py` root (`build_tree` + `class Policy`) importing from `skills.*`

Preserves any existing `skills/*.py` imports and verifies exact bit-identical
`action_bytes` execution on real NetHack seeds before returning/committing.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Any, Callable

from lox.env.harness import NetHackHarness
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package
from lox.lang.validator import (
    normalize_skill_key,
    skill_key_to_module,
    validate_cross_module_contracts,
)


def _extract_names_used(node: ast.AST) -> set[str]:
    """Returns all `ast.Name` identifiers referenced within `node`."""
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _import_bound_names(stmt: ast.Import | ast.ImportFrom) -> set[str]:
    return {alias.asname or alias.name.split(".")[0] for alias in stmt.names}


def _extract_ordered_leaf_fns(build_tree_node: ast.FunctionDef | None) -> list[str]:
    """Extracts callback function names passed to ActionLeaf/ConditionLeaf in `build_tree`."""
    if build_tree_node is None:
        return []
    seen: set[str] = set()
    nodes_with_pos: list[tuple[int, str]] = []
    for sub in ast.walk(build_tree_node):
        if isinstance(sub, ast.Call):
            func_id = (
                sub.func.id
                if isinstance(sub.func, ast.Name)
                else (sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
            )
            if func_id in ("ActionLeaf", "ConditionLeaf") and sub.args:
                arg0 = sub.args[0]
                if isinstance(arg0, ast.Name) and arg0.id not in seen:
                    seen.add(arg0.id)
                    nodes_with_pos.append((getattr(arg0, "lineno", 0), arg0.id))
    nodes_with_pos.sort(key=lambda t: t[0])
    return [name for _, name in nodes_with_pos]


def _pick_unused_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def modularize_policy_source(
    policy_code: str,
    num_modules: int = 4,
    existing_skills: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Deterministically decomposes top-level definitions inside `policy_code` into
    clean `skills/*.py` modules + a thin `policy.py` root, forming an acyclic DAG
    and preserving any pre-existing `existing_skills` modules & imports.
    """
    existing = dict(existing_skills or {})
    lines = policy_code.splitlines()
    tree = ast.parse(policy_code)

    def _slice_stmt(stmt: ast.stmt) -> str:
        start = getattr(stmt, "lineno", 1) - 1
        end = getattr(stmt, "end_lineno", start + 1)
        return "\n".join(lines[start:end])

    import_stmts: list[tuple[ast.Import | ast.ImportFrom, str]] = []
    const_defs: dict[str, tuple[str, set[str]]] = {}
    fn_defs: dict[str, tuple[str, ast.FunctionDef, set[str]]] = {}
    helper_classes: dict[str, tuple[str, set[str]]] = {}
    build_tree_node: ast.FunctionDef | None = None
    build_tree_src: str = ""
    policy_classes: list[tuple[str, ast.ClassDef]] = []

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            import_stmts.append((stmt, _slice_stmt(stmt)))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            src = _slice_stmt(stmt)
            refs = _extract_names_used(stmt)
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        const_defs[tgt.id] = (src, refs - {tgt.id})
            elif isinstance(stmt.target, ast.Name):
                const_defs[stmt.target.id] = (src, refs - {stmt.target.id})
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = _slice_stmt(stmt)
            if stmt.name == "build_tree":
                build_tree_node = stmt  # type: ignore[assignment]
                build_tree_src = src
            else:
                refs = _extract_names_used(stmt) - {stmt.name}
                fn_defs[stmt.name] = (src, stmt, refs)  # type: ignore[assignment]
        elif isinstance(stmt, ast.ClassDef):
            if stmt.name in ("Policy", "Agent"):
                policy_classes.append((_slice_stmt(stmt), stmt))
            else:
                helper_classes[stmt.name] = (
                    _slice_stmt(stmt),
                    _extract_names_used(stmt) - {stmt.name},
                )

    all_defined = set(const_defs) | set(fn_defs) | set(helper_classes)
    if not all_defined or (build_tree_node is None and len(fn_defs) < 3):
        return policy_code, existing

    def _deps_of(sym: str) -> set[str]:
        if sym in const_defs:
            return const_defs[sym][1] & all_defined
        if sym in fn_defs:
            return fn_defs[sym][2] & all_defined
        if sym in helper_classes:
            return helper_classes[sym][1] & all_defined
        return set()

    def _line_count(sym: str) -> int:
        if sym in const_defs:
            return len(const_defs[sym][0].splitlines())
        if sym in fn_defs:
            return len(fn_defs[sym][0].splitlines())
        if sym in helper_classes:
            return len(helper_classes[sym][0].splitlines())
        return 1

    # Partition only leaf functions defined directly inside `policy_code`
    leaf_fns = [n for n in _extract_ordered_leaf_fns(build_tree_node) if n in fn_defs]
    if not leaf_fns:
        leaf_fns = list(fn_defs.keys())

    k = max(1, min(int(num_modules), len(leaf_fns)))
    leaf_weights: list[int] = []
    for lf in leaf_fns:
        seen_w: set[str] = set()
        stack_w = [lf]
        w = 0
        while stack_w:
            cur = stack_w.pop()
            if cur in seen_w:
                continue
            seen_w.add(cur)
            w += _line_count(cur)
            stack_w.extend(_deps_of(cur))
        leaf_weights.append(max(1, w))

    total_w = sum(leaf_weights)
    target_w = max(1.0, total_w / k)
    chunks: list[list[str]] = []
    cur_chunk: list[str] = []
    cur_w = 0
    for idx, (lf, lw) in enumerate(zip(leaf_fns, leaf_weights, strict=True)):
        cur_chunk.append(lf)
        cur_w += lw
        remaining_leaves = len(leaf_fns) - idx - 1
        remaining_slots = k - len(chunks) - 1
        if remaining_slots > 0 and (
            cur_w >= target_w or remaining_leaves <= remaining_slots
        ):
            chunks.append(cur_chunk)
            cur_chunk = []
            cur_w = 0
    if cur_chunk:
        chunks.append(cur_chunk)

    chunk_closures: list[set[str]] = []
    ref_counts: defaultdict[str, int] = defaultdict(int)
    for ch in chunks:
        closure: set[str] = set()
        stack = list(ch)
        while stack:
            cur = stack.pop()
            if cur in closure or cur not in all_defined:
                continue
            closure.add(cur)
            stack.extend(_deps_of(cur))
        chunk_closures.append(closure)
        for sym in closure:
            ref_counts[sym] += 1

    root_direct_refs: set[str] = set()
    if build_tree_node is not None:
        root_direct_refs |= _extract_names_used(build_tree_node) & all_defined
    for _, cls_node in policy_classes:
        root_direct_refs |= _extract_names_used(cls_node) & all_defined
    non_leaf_root_refs = root_direct_refs - set(leaf_fns)

    shared_syms: set[str] = {
        sym for sym, cnt in ref_counts.items() if cnt >= 2
    } | non_leaf_root_refs
    shared_syms -= set(leaf_fns)

    stack_sh = list(shared_syms)
    shared_closed: set[str] = set()
    while stack_sh:
        cur = stack_sh.pop()
        if cur in shared_closed or cur not in all_defined:
            continue
        shared_closed.add(cur)
        stack_sh.extend(_deps_of(cur))
    shared_syms = shared_closed

    changed = True
    while changed:
        changed = False
        rem_counts: defaultdict[str, int] = defaultdict(int)
        for cl in chunk_closures:
            for sym in cl - shared_syms:
                rem_counts[sym] += 1
        extra = {sym for sym, cnt in rem_counts.items() if cnt >= 2}
        if extra:
            changed = True
            stack_ex = list(extra)
            while stack_ex:
                cur = stack_ex.pop()
                if cur in shared_syms or cur not in all_defined:
                    continue
                shared_syms.add(cur)
                stack_ex.extend(_deps_of(cur))

    covered = set(shared_syms)
    for cl in chunk_closures:
        covered |= cl
    shared_syms |= all_defined - covered

    decl_order = [
        name
        for stmt in tree.body
        for name in (
            (
                [stmt.name]
                if isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else []
            )
            + (
                [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                if isinstance(stmt, ast.Assign)
                else (
                    [stmt.target.id]
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    else []
                )
            )
        )
        if name in all_defined
    ]

    def _emit_used_imports(syms: set[str], extra_text: str = "") -> list[str]:
        used_tokens: set[str] = set()
        for sym in syms:
            if sym in const_defs:
                used_tokens |= const_defs[sym][1]
            elif sym in fn_defs:
                used_tokens |= _extract_names_used(fn_defs[sym][1])
            elif sym in helper_classes:
                used_tokens |= helper_classes[sym][1]
        if extra_text:
            try:
                used_tokens |= _extract_names_used(ast.parse(extra_text))
            except Exception:
                pass
        imp_lines: list[str] = []
        for stmt, src in import_stmts:
            if _import_bound_names(stmt) & used_tokens:
                imp_lines.append(src)
        return imp_lines

    def _render_defs(syms: set[str]) -> list[str]:
        blocks: list[str] = []
        seen_const_srcs: set[str] = set()
        for sym in decl_order:
            if sym not in syms:
                continue
            if sym in const_defs:
                src = const_defs[sym][0]
                if src not in seen_const_srcs:
                    seen_const_srcs.add(src)
                    blocks.append(src)
            elif sym in helper_classes:
                blocks.append("\n\n" + helper_classes[sym][0])
            elif sym in fn_defs:
                blocks.append("\n\n" + fn_defs[sym][0])
        return blocks

    reserved_names = set(existing.keys())
    new_skills: dict[str, str] = {}
    shared_mod_name = _pick_unused_name(
        "shared" if "shared" not in reserved_names else "shared_ext", reserved_names
    )
    if shared_syms:
        reserved_names.add(shared_mod_name)
        sh_imports = _emit_used_imports(shared_syms)
        sh_body = _render_defs(shared_syms)
        new_skills[shared_mod_name] = (
            '"""Shared constants and helper routines extracted from policy.py."""\n\n'
            + "\n".join(sh_imports)
            + ("\n\n" if sh_imports else "")
            + "\n".join(sh_body).lstrip()
            + "\n"
        )

    existing_indices = [
        int(m.group(1))
        for k_name in reserved_names
        if (m := re.match(r"^subtree_(\d+)_", k_name))
    ]
    next_idx = max(existing_indices, default=0) + 1

    policy_skill_imports: list[str] = []
    for ch_leaves, cl in zip(chunks, chunk_closures, strict=True):
        mod_syms = cl - shared_syms
        if not mod_syms:
            needed = sorted(set(ch_leaves) & shared_syms)
            if needed:
                policy_skill_imports.append(
                    f"from skills.{shared_mod_name} import {', '.join(needed)}"
                )
            continue
        slug = re.sub(r"[^a-z0-9_]", "_", ch_leaves[0].lower()).strip("_")[:18]
        raw_mod_name = f"subtree_{next_idx}_{slug}" if slug else f"subtree_{next_idx}"
        mod_name = _pick_unused_name(raw_mod_name, reserved_names)
        reserved_names.add(mod_name)
        next_idx += 1

        needed_shared = sorted(
            {dep for sym in mod_syms for dep in _deps_of(sym) if dep in shared_syms}
        )
        m_imports = _emit_used_imports(mod_syms)
        if needed_shared:
            m_imports.append(
                f"from skills.{shared_mod_name} import {', '.join(needed_shared)}"
            )
        m_body = _render_defs(mod_syms)
        leaf_doc = ", ".join(ch_leaves)
        new_skills[mod_name] = (
            f'"""BT subtree module `{mod_name}` (leaves: {leaf_doc})."""\n\n'
            + "\n".join(m_imports)
            + ("\n\n" if m_imports else "")
            + "\n".join(m_body).lstrip()
            + "\n"
        )

        root_needs = sorted((set(ch_leaves) | root_direct_refs) & mod_syms)
        if root_needs:
            policy_skill_imports.append(
                f"from skills.{mod_name} import {', '.join(root_needs)}"
            )

    root_shared_needs = sorted(root_direct_refs & shared_syms)
    if root_shared_needs:
        policy_skill_imports.append(
            f"from skills.{shared_mod_name} import {', '.join(root_shared_needs)}"
        )

    root_text = "\n\n".join(
        ([build_tree_src] if build_tree_src else [])
        + [cls_src for cls_src, _ in policy_classes]
    )
    root_imports = _emit_used_imports(set(), extra_text=root_text)
    new_policy = (
        "\n".join(root_imports + policy_skill_imports) + "\n\n\n" + root_text + "\n"
    )
    merged_skills = {**existing, **new_skills}
    return new_policy, merged_skills


def _modularize_skill_module(
    skill_name: str,
    skill_src: str,
    reserved_names: set[str],
    num_modules: int = 3,
) -> tuple[str, dict[str, str]] | None:
    """
    Deterministically decomposes a bloated `skills/<skill_name>.py` module into:
      - `<skill_name>_shared` (constants, classes, and shared helpers called by >=2 chunks)
      - `<skill_name>_<slug>` cohesive leaf submodules
      - thin `<skill_name>` facade re-exporting all original symbols in order
    Returns `(facade_src, new_submodules)` or `None` if no meaningful split exists.
    """
    lines = skill_src.splitlines()
    tree = ast.parse(skill_src)

    def _slice_stmt(stmt: ast.stmt) -> str:
        start = getattr(stmt, "lineno", 1) - 1
        end = getattr(stmt, "end_lineno", start + 1)
        return "\n".join(lines[start:end])

    import_stmts: list[tuple[ast.Import | ast.ImportFrom, str]] = []
    const_defs: dict[str, tuple[str, set[str]]] = {}
    fn_defs: dict[
        str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, set[str]]
    ] = {}
    helper_classes: dict[str, tuple[str, set[str]]] = {}
    extra_stmts: list[str] = []
    self_mod_full = f"skills.{skill_name}"

    def _collect_refs_and_self_imports(node: ast.AST) -> tuple[set[str], set[str]]:
        refs = _extract_names_used(node)
        self_syms: set[str] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.ImportFrom)
                and str(sub.module or "") == self_mod_full
            ):
                for alias in sub.names:
                    self_syms.add(str(alias.name or "").strip())
        return refs | self_syms, self_syms

    self_imported_syms: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            import_stmts.append((stmt, _slice_stmt(stmt)))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            src = _slice_stmt(stmt)
            refs, s_syms = _collect_refs_and_self_imports(stmt)
            self_imported_syms |= s_syms
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        const_defs[tgt.id] = (src, refs - {tgt.id})
            elif isinstance(stmt.target, ast.Name):
                const_defs[stmt.target.id] = (src, refs - {stmt.target.id})
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            refs, s_syms = _collect_refs_and_self_imports(stmt)
            self_imported_syms |= s_syms
            fn_defs[stmt.name] = (_slice_stmt(stmt), stmt, refs - {stmt.name})
        elif isinstance(stmt, ast.ClassDef):
            refs, s_syms = _collect_refs_and_self_imports(stmt)
            self_imported_syms |= s_syms
            helper_classes[stmt.name] = (_slice_stmt(stmt), refs - {stmt.name})
        elif not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Side-effect top-level statements cannot be safely split across submodules
            extra_stmts.append(_slice_stmt(stmt))

    if extra_stmts or len(fn_defs) < 3:
        return None

    all_defined = set(const_defs) | set(fn_defs) | set(helper_classes)
    decl_order = [
        name
        for stmt in tree.body
        for name in (
            (
                [stmt.name]
                if isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else []
            )
            + (
                [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                if isinstance(stmt, ast.Assign)
                else (
                    [stmt.target.id]
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    else []
                )
            )
        )
        if name in all_defined
    ]

    def _deps_of(sym: str) -> set[str]:
        if sym in const_defs:
            return const_defs[sym][1] & all_defined
        if sym in fn_defs:
            return fn_defs[sym][2] & all_defined
        if sym in helper_classes:
            return helper_classes[sym][1] & all_defined
        return set()

    def _line_count(sym: str) -> int:
        if sym in const_defs:
            return len(const_defs[sym][0].splitlines())
        if sym in fn_defs:
            return len(fn_defs[sym][0].splitlines())
        if sym in helper_classes:
            return len(helper_classes[sym][0].splitlines())
        return 1

    root_fns = [n for n in decl_order if n in fn_defs and not n.startswith("_")]
    if len(root_fns) < 2:
        root_fns = [n for n in decl_order if n in fn_defs]

    k = max(2, min(int(num_modules), len(root_fns)))
    leaf_weights: list[int] = []
    for lf in root_fns:
        seen_w: set[str] = set()
        stack_w = [lf]
        w = 0
        while stack_w:
            cur = stack_w.pop()
            if cur in seen_w:
                continue
            seen_w.add(cur)
            w += _line_count(cur)
            stack_w.extend(_deps_of(cur))
        leaf_weights.append(max(1, w))

    target_w = max(1.0, sum(leaf_weights) / k)
    chunks: list[list[str]] = []
    cur_chunk: list[str] = []
    cur_w = 0
    for idx, (lf, lw) in enumerate(zip(root_fns, leaf_weights, strict=True)):
        cur_chunk.append(lf)
        cur_w += lw
        remaining = len(root_fns) - idx - 1
        rem_slots = k - len(chunks) - 1
        if rem_slots > 0 and (cur_w >= target_w or remaining <= rem_slots):
            chunks.append(cur_chunk)
            cur_chunk = []
            cur_w = 0
    if cur_chunk:
        chunks.append(cur_chunk)

    chunk_closures: list[set[str]] = []
    ref_counts: defaultdict[str, int] = defaultdict(int)
    for ch in chunks:
        closure: set[str] = set()
        stack = list(ch)
        while stack:
            cur = stack.pop()
            if cur in closure or cur not in all_defined:
                continue
            closure.add(cur)
            stack.extend(_deps_of(cur))
        chunk_closures.append(closure)
        for sym in closure:
            ref_counts[sym] += 1

    # Any symbol referenced by >=2 chunks or lazily self-imported moves to `<skill>_shared`
    shared_syms: set[str] = {sym for sym, cnt in ref_counts.items() if cnt >= 2} | (
        self_imported_syms & all_defined
    )
    stack_sh = list(shared_syms)
    shared_closed: set[str] = set()
    while stack_sh:
        cur = stack_sh.pop()
        if cur in shared_closed or cur not in all_defined:
            continue
        shared_closed.add(cur)
        stack_sh.extend(_deps_of(cur))
    shared_syms = shared_closed

    covered = set(shared_syms)
    for cl in chunk_closures:
        covered |= cl
    shared_syms |= all_defined - covered

    def _emit_used_imports(syms: set[str]) -> list[str]:
        used_tokens: set[str] = set()
        for sym in syms:
            if sym in const_defs:
                used_tokens |= const_defs[sym][1]
            elif sym in fn_defs:
                used_tokens |= _extract_names_used(fn_defs[sym][1])
            elif sym in helper_classes:
                used_tokens |= helper_classes[sym][1]
        return [
            src for stmt, src in import_stmts if _import_bound_names(stmt) & used_tokens
        ]

    def _rewrite_self_imports(code_src: str, target_mod: str) -> str:
        """Redirects `from skills.<skill> import ...` inside split modules to `target_mod` to prevent cycles."""
        return re.sub(
            rf"\bfrom\s+skills\.{re.escape(skill_name)}\s+import\b",
            f"from skills.{target_mod} import",
            code_src,
        )

    def _render_defs(syms: set[str], target_mod: str) -> list[str]:
        blocks: list[str] = []
        seen_const_srcs: set[str] = set()
        for sym in decl_order:
            if sym not in syms:
                continue
            if sym in const_defs:
                src = _rewrite_self_imports(const_defs[sym][0], target_mod)
                if src not in seen_const_srcs:
                    seen_const_srcs.add(src)
                    blocks.append(src)
            elif sym in helper_classes:
                blocks.append(
                    "\n\n" + _rewrite_self_imports(helper_classes[sym][0], target_mod)
                )
            elif sym in fn_defs:
                blocks.append(
                    "\n\n" + _rewrite_self_imports(fn_defs[sym][0], target_mod)
                )
        return blocks

    local_reserved = set(reserved_names)
    submodules: dict[str, str] = {}
    reexport_lines: list[str] = []
    exported_order = list(dict.fromkeys(decl_order))

    shared_mod = _pick_unused_name(f"{skill_name}_shared", local_reserved)
    if shared_syms:
        local_reserved.add(shared_mod)
        sh_imports = _emit_used_imports(shared_syms)
        sh_body = _render_defs(shared_syms, shared_mod)
        submodules[shared_mod] = (
            f'"""Shared constants and helpers split from `{skill_name}`."""\n\n'
            + "\n".join(sh_imports)
            + ("\n\n" if sh_imports else "")
            + "\n".join(sh_body).lstrip()
            + "\n"
        )
        sh_exports = [s for s in exported_order if s in shared_syms]
        if sh_exports:
            reexport_lines.append(
                f"from skills.{shared_mod} import {', '.join(sh_exports)}"
            )

    for idx_ch, (ch_leaves, cl) in enumerate(
        zip(chunks, chunk_closures, strict=True), start=1
    ):
        mod_syms = cl - shared_syms
        if not mod_syms:
            continue
        slug = re.sub(r"[^a-z0-9_]", "_", ch_leaves[0].lower()).strip("_")[:18]
        raw_sub = f"{skill_name}_{slug}" if slug else f"{skill_name}_part_{idx_ch}"
        sub_name = _pick_unused_name(raw_sub, local_reserved)
        local_reserved.add(sub_name)

        needed_shared = sorted(
            {dep for sym in mod_syms for dep in _deps_of(sym) if dep in shared_syms}
        )
        m_imports = _emit_used_imports(mod_syms)
        if needed_shared:
            m_imports.append(
                f"from skills.{shared_mod} import {', '.join(needed_shared)}"
            )
        m_body = _render_defs(mod_syms, shared_mod)
        submodules[sub_name] = (
            f'"""Submodule `{sub_name}` split from `{skill_name}`."""\n\n'
            + "\n".join(m_imports)
            + ("\n\n" if m_imports else "")
            + "\n".join(m_body).lstrip()
            + "\n"
        )
        sub_exports = [s for s in exported_order if s in mod_syms]
        if sub_exports:
            reexport_lines.append(
                f"from skills.{sub_name} import {', '.join(sub_exports)}"
            )

    if len(submodules) < 2 or not reexport_lines:
        return None

    facade_src = (
        f'"""Modular facade for `{skill_name}` (re-exports split submodules)."""\n\n'
        + "\n".join(reexport_lines)
        + "\n"
    )
    max_after = max(
        len(facade_src.splitlines()),
        *(len(v.splitlines()) for v in submodules.values()),
    )
    if max_after > len(lines) - 30:
        return None
    reserved_names.update(submodules.keys())
    return facade_src, submodules


def verify_bit_identical(
    orig_policy: str,
    orig_skills: dict[str, str],
    new_policy: str,
    new_skills: dict[str, str],
    seeds: list[int],
    steps: int = 300,
) -> None:
    """Compiles both packages and asserts exact step-by-step action_bytes identity."""
    with _COMPILE_EXEC_LOCK:
        orig_compiled = compile_policy_package(orig_policy, "", skills=orig_skills)
        new_compiled = compile_policy_package(new_policy, "", skills=new_skills)

    for sd in seeds:
        h1 = NetHackHarness(max_steps=steps)
        h2 = NetHackHarness(max_steps=steps)
        try:
            r1 = orig_compiled.create_runner()
            r2 = new_compiled.create_runner()
            s1 = h1.reset(seed=int(sd))
            s2 = h2.reset(seed=int(sd))
            for _ in range(steps):
                a1, _ = r1.next_action(s1)
                a2, _ = r2.next_action(s2)
                s1, d1, _ = h1.step(a1)
                s2, d2, _ = h2.step(a2)
                if bytes(h1.action_bytes) != bytes(h2.action_bytes) or d1 != d2:
                    raise AssertionError(
                        f"Behavioral divergence on seed {sd} at step {h1.step_count}!"
                    )
                if d1:
                    break
        finally:
            h1.close()
            h2.close()


def maybe_auto_modularize(
    policy_code: str,
    skills: dict[str, str] | None = None,
    *,
    loc_threshold: int = 250,
    monolith_threshold: int = 200,
    skill_loc_threshold: int = 450,
    num_modules: int = 4,
    verify_seeds: list[int] | None = None,
    verify_steps: int = 300,
) -> tuple[bool, str, dict[str, str], str]:
    """
    Hysteresis-gated circuit breaker: splits bloated `policy.py` and/or bloated
    `skills/*.py` modules deterministically when LOC exceeds threshold, verifies
    bit-identical NetHack trajectories, and returns `(changed, policy_code, skills, summary)`.
    """
    orig_skills = dict(skills or {})
    if not policy_code.strip():
        return False, policy_code, orig_skills, "Skipped: empty policy."
    loc_before = len(policy_code.splitlines())
    needs_pol_split = loc_before >= loc_threshold or (
        not orig_skills and loc_before >= monolith_threshold
    )

    def _is_nested_submod(sk: str, src: str) -> bool:
        sk_len = len(src.splitlines())
        if sk.endswith("_shared") and sk_len < int(skill_loc_threshold * 1.5):
            return True
        return any(
            k != sk
            and sk.startswith(f"{k}_")
            and "Modular facade" in (orig_skills.get(k) or "")
            for k in orig_skills
        ) and sk_len < int(skill_loc_threshold * 1.5)

    bloated_skills = [
        sk
        for sk, src in sorted(orig_skills.items())
        if src
        and len(src.splitlines()) >= skill_loc_threshold
        and not _is_nested_submod(sk, src)
    ]
    if not needs_pol_split and not bloated_skills:
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: below threshold ({loc_before} LOC).",
        )

    new_pol = policy_code
    new_skills = dict(orig_skills)
    notes: list[str] = []

    if needs_pol_split:
        try:
            split_pol, split_skills = modularize_policy_source(
                new_pol, num_modules=num_modules, existing_skills=new_skills
            )
            loc_after = len(split_pol.splitlines())
            if loc_after <= loc_before - 30 and len(split_skills) > len(new_skills):
                new_pol = split_pol
                new_skills = split_skills
                notes.append(f"policy.py {loc_before}->{loc_after} LOC")
        except Exception as exc:
            if not bloated_skills:
                return False, policy_code, orig_skills, f"AST split error: {exc}"

    reserved_names = set(new_skills.keys())
    for sk in bloated_skills:
        sk_src = new_skills.get(sk, "")
        sk_before = len(sk_src.splitlines())
        if sk_before < skill_loc_threshold:
            continue
        try:
            split_res = _modularize_skill_module(
                sk, sk_src, reserved_names, num_modules=min(4, max(2, sk_before // 200))
            )
        except Exception:
            split_res = None
        if split_res is None:
            continue
        facade_src, submods = split_res
        new_skills[sk] = facade_src
        new_skills.update(submods)
        notes.append(
            f"skill/{sk} {sk_before}->{len(facade_src.splitlines())} LOC (+{len(submods)} submodules)"
        )

    if not notes:
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: insufficient reduction ({loc_before} LOC).",
        )

    seeds = list(verify_seeds) if verify_seeds else [1000 + i for i in range(6)]
    try:
        verify_bit_identical(
            policy_code,
            orig_skills,
            new_pol,
            new_skills,
            seeds=seeds,
            steps=max(50, int(verify_steps)),
        )
    except Exception as exc:
        return (
            False,
            policy_code,
            orig_skills,
            f"Bit-identity check failed: {exc}",
        )

    return (
        True,
        new_pol,
        new_skills,
        f"AST-Modularized: {', '.join(notes)} ({len(new_skills)} skills, {len(seeds)}-seed bit-identical)",
    )


def build_module_symbol_outline(skill_src: str) -> list[dict[str, Any]]:
    """Builds a compact AST outline (`name`, `kind`, `loc`, `calls`, `callers`) for cohesive split planning."""
    try:
        tree = ast.parse(skill_src or "")
    except SyntaxError:
        return []

    defs: list[tuple[str, str, ast.AST, int]] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(stmt, ast.ClassDef) else "fn"
            loc = max(
                1,
                int(getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno)
                - int(stmt.lineno)
                + 1,
            )
            defs.append((stmt.name, kind, stmt, loc))

    def_names = {d[0] for d in defs}
    calls_by: dict[str, set[str]] = {name: set() for name, _, _, _ in defs}
    callers_by: dict[str, set[str]] = {name: set() for name, _, _, _ in defs}
    for name, _, node, _ in defs:
        used = (_extract_names_used(node) & def_names) - {name}
        calls_by[name] = used
        for target in used:
            callers_by[target].add(name)

    return [
        {
            "name": name,
            "kind": kind,
            "loc": loc,
            "calls": sorted(calls_by[name]),
            "callers": sorted(callers_by[name]),
        }
        for name, kind, _, loc in defs
    ]


def _pick_deterministic_split_symbols(
    outline: list[dict[str, Any]],
) -> tuple[str, list[str]] | None:
    """Finds a coherent subset of top-level definitions (~25–55% LOC) with zero mutual cycles."""
    if len(outline) < 4:
        return None
    by_name = {d["name"]: d for d in outline}
    total_loc = sum(int(d["loc"]) for d in outline)

    # Try each public non-underscored entrypoint as a seed and gather its private closure
    candidates: list[tuple[int, str, list[str]]] = []
    for item in outline:
        root_nm = item["name"]
        if root_nm.startswith("_"):
            continue
        # BFS down internal calls that are NOT called by anything outside the seed closure
        closure: set[str] = {root_nm}
        frontier = [root_nm]
        while frontier:
            cur = frontier.pop()
            for dep in by_name[cur]["calls"]:
                if dep in closure:
                    continue
                # Include private helpers or items whose callers stay inside closure
                ext_callers = set(by_name[dep]["callers"]) - closure
                if not ext_callers:
                    closure.add(dep)
                    frontier.append(dep)
        # Verify nothing outside `closure` calls `closure` (or keep public root only if no internal cycle)
        move_loc = sum(int(by_name[n]["loc"]) for n in closure)
        if 40 <= move_loc <= max(60, int(total_loc * 0.70)):
            slug = re.sub(r"_(?:step|goal|handler|check)$", "", root_nm).strip("_")
            slug = re.sub(r"[^a-z0-9_]+", "_", slug.lower()).strip("_") or "extra"
            candidates.append((abs(move_loc - total_loc // 2), slug, sorted(closure)))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    _, slug, syms = candidates[0]
    return slug, syms


def _rewrite_symbol_imports_in_source(
    source: str,
    src_mod_dotted: str,
    dst_mod_dotted: str,
    moved_symbols: set[str],
    *,
    src_renamed_to: str | None = None,
    ensure_dst_imported: bool = False,
) -> str:
    """Rewrites imports and module-attribute references cleanly in callers without facade shims."""
    if not source or not source.strip():
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines()
    old_aliases: set[str] = set()
    used_dst_alias = dst_mod_dotted.rsplit(".", 1)[-1]
    alias_collisions = _extract_names_used(tree)
    if used_dst_alias in alias_collisions:
        used_dst_alias = f"_{dst_mod_dotted.split('.')[-2]}_{used_dst_alias}"

    need_dst_import = False
    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == src_mod_dotted:
            kept: list[ast.alias] = []
            moved_als: list[ast.alias] = []
            for al in stmt.names:
                if al.name in moved_symbols:
                    moved_als.append(al)
                else:
                    kept.append(al)
            if kept:
                mod_out = src_renamed_to or src_mod_dotted
                new_body.append(
                    ast.ImportFrom(module=mod_out, names=kept, level=stmt.level)
                )
            elif src_renamed_to and not moved_als:
                new_body.append(
                    ast.ImportFrom(
                        module=src_renamed_to, names=stmt.names, level=stmt.level
                    )
                )
            if moved_als:
                new_body.append(
                    ast.ImportFrom(
                        module=dst_mod_dotted, names=moved_als, level=stmt.level
                    )
                )
        elif isinstance(stmt, ast.Import):
            new_names: list[ast.alias] = []
            for al in stmt.names:
                if al.name == src_mod_dotted:
                    alias_id = al.asname or src_mod_dotted
                    old_aliases.add(alias_id)
                    new_names.append(
                        ast.alias(
                            name=src_renamed_to or src_mod_dotted, asname=al.asname
                        )
                    )
                else:
                    new_names.append(al)
            if new_names:
                new_body.append(ast.Import(names=new_names))
        else:
            new_body.append(stmt)

    # Check if any attribute accesses `<alias>.<moved_sym>` or `skills.<...>.<moved_sym>` exist
    class _AttrRewriter(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            nonlocal need_dst_import
            self.generic_visit(node)
            if node.attr in moved_symbols and isinstance(node.value, ast.Name):
                if node.value.id in old_aliases:
                    need_dst_import = True
                    return ast.Attribute(
                        value=ast.Name(id=used_dst_alias, ctx=ast.Load()),
                        attr=node.attr,
                        ctx=node.ctx,
                    )
            return node

    tree.body = new_body
    tree = _AttrRewriter().visit(tree)

    if need_dst_import or ensure_dst_imported:
        already_imported = any(
            (
                isinstance(st, ast.Import)
                and any(al.name == dst_mod_dotted for al in st.names)
            )
            or (isinstance(st, ast.ImportFrom) and st.module == dst_mod_dotted)
            for st in tree.body
        )
        if not already_imported:
            imp_stmt = (
                ast.Import(
                    names=[ast.alias(name=dst_mod_dotted, asname=used_dst_alias)]
                )
                if need_dst_import
                else ast.Import(names=[ast.alias(name=dst_mod_dotted, asname=None)])
            )
            insert_at = 0
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                insert_at = 1
            tree.body.insert(insert_at, imp_stmt)

    ast.fix_missing_locations(tree)
    _ = lines
    return ast.unparse(tree)


def _split_module_ast_verbatim(
    src_key: str,
    dst_key: str,
    move_symbols: set[str],
    policy_code: str,
    skills_map: dict[str, str],
) -> tuple[str, dict[str, str]] | None:
    """Deterministically splits `move_symbols` from `src_key` into `dst_key` without facade shims."""
    src_code = skills_map.get(src_key, "")
    if not src_code.strip() or not move_symbols:
        return None
    try:
        src_tree = ast.parse(src_code)
    except SyntaxError:
        return None

    lines = src_code.splitlines()
    imports: list[ast.stmt] = []
    kept_stmts: list[ast.stmt] = []
    moved_stmts: list[ast.stmt] = []
    src_exports: set[str] = set()
    kept_names: set[str] = set()
    moved_names: set[str] = set()

    def _stmt_segment(stmt: ast.stmt) -> str:
        s_line = max(1, int(stmt.lineno) - 1)
        e_line = int(getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno)
        return "\n".join(lines[s_line:e_line])

    def _top_name(stmt: ast.stmt) -> str:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return stmt.name
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.targets[0].id
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return ""

    doc_line = ""
    body_stmts = list(src_tree.body)
    if (
        body_stmts
        and isinstance(body_stmts[0], ast.Expr)
        and isinstance(body_stmts[0].value, ast.Constant)
        and isinstance(body_stmts[0].value.value, str)
    ):
        doc_line = _stmt_segment(body_stmts[0])
        body_stmts = body_stmts[1:]

    for stmt in body_stmts:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(stmt)
            continue
        nm = _top_name(stmt)
        if nm:
            src_exports.add(nm)
        if nm and nm in move_symbols:
            moved_stmts.append(stmt)
            moved_names.add(nm)
        else:
            kept_stmts.append(stmt)
            if nm:
                kept_names.add(nm)

    if not moved_stmts or not kept_stmts:
        return None

    # Acyclicity check: items remaining in `src_key` MUST NOT reference `moved_names`
    kept_refs: set[str] = set()
    for st in kept_stmts:
        kept_refs |= _extract_names_used(st)
    if kept_refs & moved_names:
        return None

    moved_refs: set[str] = set()
    for st in moved_stmts:
        moved_refs |= _extract_names_used(st)

    # Determine whether flat `src_key` must relocate to `<src_key>/core` to avoid Python shadowing
    final_src_key = src_key
    src_renamed_to: str | None = None
    if "/" not in src_key and dst_key.startswith(f"{src_key}/"):
        final_src_key = f"{src_key}/core"
        src_renamed_to = skill_key_to_module(final_src_key)

    src_mod_dotted = skill_key_to_module(src_key)
    final_src_dotted = skill_key_to_module(final_src_key)
    dst_mod_dotted = skill_key_to_module(dst_key)

    # Filter imports needed by kept vs moved statements
    def _imports_for(used: set[str]) -> list[str]:
        out: list[str] = []
        for imp in imports:
            if isinstance(imp, ast.Import):
                kept_als = [
                    al
                    for al in imp.names
                    if (al.asname or al.name.split(".")[0]) in used
                ]
                if kept_als:
                    out.append(ast.unparse(ast.Import(names=kept_als)))
            elif isinstance(imp, ast.ImportFrom):
                if any(al.name == "*" for al in imp.names):
                    out.append(_stmt_segment(imp))
                else:
                    kept_als = [
                        al for al in imp.names if (al.asname or al.name) in used
                    ]
                    if kept_als:
                        out.append(
                            ast.unparse(
                                ast.ImportFrom(
                                    module=imp.module, names=kept_als, level=imp.level
                                )
                            )
                        )
        return out

    needed_from_src = sorted(moved_refs & kept_names)
    dst_lines: list[str] = [
        f'"""Cohesive submodule `{dst_key}` split from `{src_key}`."""',
        "",
    ]
    dst_lines.extend(_imports_for(moved_refs))
    if needed_from_src:
        dst_lines.append(f"from {final_src_dotted} import {', '.join(needed_from_src)}")
    dst_lines.append("")
    for st in moved_stmts:
        dst_lines.append(_stmt_segment(st))
        dst_lines.append("")
    dst_src = "\n".join(dst_lines).strip() + "\n"

    src_lines: list[str] = []
    if doc_line:
        src_lines.extend([doc_line, ""])
    src_lines.extend(_imports_for(kept_refs))
    src_lines.append("")
    for st in kept_stmts:
        src_lines.append(_stmt_segment(st))
        src_lines.append("")
    new_src_code = "\n".join(src_lines).strip() + "\n"

    new_skills = dict(skills_map)
    if final_src_key != src_key:
        new_skills.pop(src_key, None)
    new_skills[final_src_key] = new_src_code
    new_skills[dst_key] = dst_src

    # Rewrite imports directly in policy.py and all other skills (0 facades!)
    new_policy = _rewrite_symbol_imports_in_source(
        policy_code,
        src_mod_dotted,
        dst_mod_dotted,
        moved_names,
        src_renamed_to=src_renamed_to,
        ensure_dst_imported=False,
    )
    for ok_k in list(new_skills.keys()):
        if ok_k in (final_src_key, dst_key):
            continue
        new_skills[ok_k] = _rewrite_symbol_imports_in_source(
            new_skills[ok_k],
            src_mod_dotted,
            dst_mod_dotted,
            moved_names,
            src_renamed_to=src_renamed_to,
            ensure_dst_imported=False,
        )

    return new_policy, new_skills


def run_side_cohesion_refactor(
    policy_code: str,
    skills: dict[str, str] | None = None,
    *,
    max_loc: int = 500,
    split_plan_fn: Callable[
        [str, list[dict[str, Any]], set[str]], dict[str, Any] | None
    ]
    | None = None,
    llm_decompose_fn: Callable[[str, str], str | None] | None = None,
    verify_seeds: list[int] | None = None,
    verify_steps: int = 300,
) -> tuple[bool, str, dict[str, str], str]:
    """
    Ungated hybrid AST + LLM side-channel refactor pass.

    Runs outside tournament promotion whenever any module exceeds `max_loc` (default 500):
    - Case A (`>= 4` top-level defs): AST builds symbol/call outline, LLM selects semantic split
      manifest (or deterministic cluster fallback), and AST executes lossless extraction + direct
      caller import rewriting (0 facade shims).
    - Case B (`1-3` oversized defs): calls `llm_decompose_fn` if supplied to break monoliths into
      helpers inside the same module before Case A splitting.

    Enforces compile, cross-module contract, and `verify_bit_identical` parity before returning.
    """
    orig_skills = {
        normalize_skill_key(k): v
        for k, v in (skills or {}).items()
        if normalize_skill_key(k) and str(v or "").strip()
    }
    if not (policy_code or "").strip():
        return False, policy_code, orig_skills, "Skipped: empty package."

    pol_loc = len((policy_code or "").splitlines())
    needs_pol_split = (not orig_skills and pol_loc >= 200) or pol_loc >= int(max_loc)
    bloated = [
        sk
        for sk, src in sorted(orig_skills.items())
        if len(src.splitlines()) >= int(max_loc)
    ]
    if not needs_pol_split and not bloated:
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: all modules < {max_loc} LOC.",
        )

    cur_pol = policy_code
    cur_skills = dict(orig_skills)
    notes: list[str] = []

    if needs_pol_split:
        try:
            split_pol, split_skills = modularize_policy_source(
                cur_pol, num_modules=4, existing_skills=cur_skills
            )
            loc_after = len(split_pol.splitlines())
            if loc_after <= pol_loc - 30 and len(split_skills) > len(cur_skills):
                cur_pol = split_pol
                cur_skills = {
                    normalize_skill_key(k): v
                    for k, v in split_skills.items()
                    if normalize_skill_key(k) and str(v or "").strip()
                }
                notes.append(f"policy.py {pol_loc}->{loc_after} LOC")
                bloated = [
                    sk
                    for sk, src in sorted(cur_skills.items())
                    if len(src.splitlines()) >= int(max_loc)
                ]
        except Exception:
            pass

    if not cur_skills:
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: no skills to split ({pol_loc} LOC).",
        )

    for sk in bloated:
        src = cur_skills.get(sk, "")
        if len(src.splitlines()) < int(max_loc):
            continue
        outline = build_module_symbol_outline(src)

        # Case B: oversized monolith inside 1-3 functions
        if len(outline) < 4 and llm_decompose_fn is not None:
            try:
                decomposed = llm_decompose_fn(sk, src)
                if (
                    decomposed
                    and decomposed.strip()
                    and decomposed.strip() != src.strip()
                ):
                    test_skills = dict(cur_skills)
                    test_skills[sk] = decomposed.strip() + "\n"
                    validate_cross_module_contracts(cur_pol, test_skills)
                    cur_skills = test_skills
                    src = cur_skills[sk]
                    outline = build_module_symbol_outline(src)
                    notes.append(f"{sk}: decomposed monolith")
            except Exception:
                pass

        # Case A: >= 4 top-level defs -> LLM split plan + deterministic AST extraction
        if len(outline) < 4:
            continue

        domain = sk.split("/", 1)[0]
        move_syms: set[str] = set()
        dst_key = ""

        if split_plan_fn is not None:
            try:
                plan = split_plan_fn(sk, outline, set(cur_skills.keys()))
                if isinstance(plan, dict):
                    raw_target = str(plan.get("target_submodule", "") or "").strip()
                    if raw_target:
                        nk_t = normalize_skill_key(raw_target)
                        if "/" not in nk_t:
                            nk_t = f"{domain}/{nk_t}"
                        if nk_t not in cur_skills and nk_t != sk:
                            dst_key = nk_t
                    raw_syms = plan.get("move_symbols") or []
                    if isinstance(raw_syms, list):
                        move_syms = {str(s).strip() for s in raw_syms if str(s).strip()}
            except Exception:
                dst_key = ""
                move_syms = set()

        if not dst_key or not move_syms:
            det = _pick_deterministic_split_symbols(outline)
            if det is None:
                continue
            slug, det_syms = det
            base_dst = f"{domain}/{slug}"
            cand_dst = base_dst
            idx = 2
            while cand_dst in cur_skills or cand_dst == sk:
                cand_dst = f"{base_dst}_{idx}"
                idx += 1
            dst_key = cand_dst
            move_syms = set(det_syms)

        res = _split_module_ast_verbatim(sk, dst_key, move_syms, cur_pol, cur_skills)
        if res is None:
            continue
        cand_pol, cand_skills = res
        try:
            validate_cross_module_contracts(cand_pol, cand_skills)
            compile_policy_package(cand_pol, "", skills=cand_skills)
        except Exception:
            continue
        cur_pol, cur_skills = cand_pol, cand_skills
        notes.append(f"{sk} -> +{dst_key} ({len(move_syms)} symbols)")

    if not notes:
        return False, policy_code, orig_skills, "Skipped: no cohesive split found."

    seeds = list(verify_seeds) if verify_seeds else [1000 + i for i in range(6)]
    try:
        verify_bit_identical(
            policy_code,
            orig_skills,
            cur_pol,
            cur_skills,
            seeds=seeds,
            steps=max(50, int(verify_steps)),
        )
    except Exception as exc:
        return False, policy_code, orig_skills, f"Refactor parity check failed: {exc}"

    return (
        True,
        cur_pol,
        cur_skills,
        f"Side Cohesion Refactor: {', '.join(notes)} ({len(seeds)}-seed bit-identical)",
    )
