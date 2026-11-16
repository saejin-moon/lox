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

from lox.env.harness import NetHackHarness
from lox.lang.compiler import _COMPILE_EXEC_LOCK, compile_policy_package


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
    loc_threshold: int = 400,
    monolith_threshold: int = 300,
    num_modules: int = 4,
    verify_seeds: list[int] | None = None,
    verify_steps: int = 300,
) -> tuple[bool, str, dict[str, str], str]:
    """
    Hysteresis-gated circuit breaker: splits bloated `policy.py` deterministically
    only when LOC exceeds threshold, verifies bit-identical NetHack trajectories,
    and returns `(changed, policy_code, skills, summary)`.
    """
    orig_skills = dict(skills or {})
    if not policy_code.strip():
        return False, policy_code, orig_skills, "Skipped: empty policy."
    loc_before = len(policy_code.splitlines())
    needs_split = loc_before >= loc_threshold or (
        not orig_skills and loc_before >= monolith_threshold
    )
    if not needs_split:
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: below threshold ({loc_before} LOC).",
        )

    try:
        new_pol, new_skills = modularize_policy_source(
            policy_code, num_modules=num_modules, existing_skills=orig_skills
        )
    except Exception as exc:
        return (
            False,
            policy_code,
            orig_skills,
            f"AST split error: {exc}",
        )

    loc_after = len(new_pol.splitlines())
    if loc_after > loc_before - 30 or len(new_skills) <= len(orig_skills):
        return (
            False,
            policy_code,
            orig_skills,
            f"Skipped: insufficient reduction ({loc_before}->{loc_after} LOC).",
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
        f"AST-Modularized: policy.py {loc_before}->{loc_after} LOC ({len(new_skills)} skills, {len(seeds)}-seed bit-identical)",
    )
