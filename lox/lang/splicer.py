"""
Surgical AST Function Splicer, 3-Way AST Merger, Hygiene Auditor, and `#EVOLVE-BLOCK` Guard.
"""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from typing import Any

from lox.lang.sandbox import smart_dedent, validate_python_code


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in getattr(node, "decorator_list", ()):
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
    return names


def _first_positional_arg(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    if node.args.posonlyargs:
        return node.args.posonlyargs[0].arg
    if node.args.args:
        return node.args.args[0].arg
    return None


def _collect_all_bound_names_in_fn(
    fn_n: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Collects all parameter and local binding names across the entire body of `fn_n`."""
    names: set[str] = {
        a.arg
        for a in (
            list(fn_n.args.posonlyargs)
            + list(fn_n.args.args)
            + list(fn_n.args.kwonlyargs)
            + ([fn_n.args.vararg] if fn_n.args.vararg else [])
            + ([fn_n.args.kwarg] if fn_n.args.kwarg else [])
        )
    }

    def _add_target(tgt: ast.AST) -> None:
        if isinstance(tgt, ast.Name):
            names.add(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for elt in tgt.elts:
                _add_target(elt)
        elif isinstance(tgt, ast.Starred):
            _add_target(tgt.value)

    stack: list[ast.AST] = list(fn_n.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(cur.name)
            continue
        if isinstance(cur, ast.Assign):
            for t in cur.targets:
                _add_target(t)
        elif isinstance(cur, ast.AnnAssign):
            _add_target(cur.target)
        elif isinstance(cur, ast.AugAssign):
            _add_target(cur.target)
        elif isinstance(cur, (ast.For, ast.AsyncFor)):
            _add_target(cur.target)
        elif isinstance(cur, (ast.With, ast.AsyncWith)):
            for it in cur.items:
                if it.optional_vars is not None:
                    _add_target(it.optional_vars)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            names.add(cur.name)
        elif isinstance(cur, ast.NamedExpr):
            _add_target(cur.target)
        elif isinstance(cur, (ast.Import, ast.ImportFrom)):
            for al in cur.names:
                names.add(al.asname or al.name.split(".")[0])
        elif isinstance(cur, ast.MatchAs) and cur.name:
            names.add(cur.name)
        elif isinstance(cur, ast.MatchStar) and cur.name:
            names.add(cur.name)
        elif isinstance(cur, ast.MatchMapping) and cur.rest:
            names.add(cur.rest)
        for ch in ast.iter_child_nodes(cur):
            stack.append(ch)
    return names


def splice_functions_into_policy(
    base_policy_code: str,
    patch_code: str,
) -> tuple[str, list[str]]:
    """
    Splices added or modified functions/methods from `patch_code` into `base_policy_code`.
    Supports:
    - Standalone `def method_name(self, ...):` definitions (spliced onto `class Policy`)
    - Standalone helper functions `def helper_name(...):` without `self` (spliced at module top-level or onto `Policy`)
    - Full `class Policy:` definitions containing one or more methods
    Returns `(new_policy_code, modified_or_added_function_names)`.
    """
    patch_code = smart_dedent(patch_code)
    patch_tree = validate_python_code(patch_code, filename="<patch>")
    if not base_policy_code or not base_policy_code.strip():
        fn_names = [
            n.name
            for n in ast.walk(patch_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name != "__init__"
        ]
        return ast.unparse(patch_tree), fn_names

    base_tree = validate_python_code(base_policy_code, filename="<base_policy>")

    # Locate target class (`Policy` or `Agent` or a stateful class with `self` methods) in base_tree;
    # plain field-only @dataclass structs must NOT act as target_class.
    target_class: ast.ClassDef | None = None
    for node in base_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Policy", "Agent"):
            target_class = node
            break
    if target_class is None:
        for node in base_tree.body:
            if isinstance(node, ast.ClassDef) and any(
                isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _first_positional_arg(m) == "self"
                for m in node.body
            ):
                target_class = node
                break

    # Collect attribute accesses `self.<attr>` outside `__init__` so we only preserve still-needed attrs
    self_attrs_referenced_outside_init: set[str] = set()
    for tree_to_scan in (patch_tree, base_tree):
        for sub in ast.walk(tree_to_scan):
            if (
                isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                and sub.name != "__init__"
            ):
                for inner in ast.walk(sub):
                    if (
                        isinstance(inner, ast.Attribute)
                        and isinstance(inner.value, ast.Name)
                        and inner.value.id == "self"
                    ):
                        self_attrs_referenced_outside_init.add(inner.attr)

    patch_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    patch_class_assigns: dict[str, ast.stmt] = {}
    patch_top_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    patch_top_extras: list[ast.stmt] = []

    def _assign_target_name(stmt: ast.stmt) -> str | None:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if isinstance(t, ast.Name):
                return t.id
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    def _dedup_assigns_keep_last(stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Deduplicates identical-target top-level assigns only when no intervening statement loads the earlier binding."""
        positions_by_name: dict[str, list[int]] = {}
        for idx, st in enumerate(stmts):
            nm = _assign_target_name(st)
            if nm is not None:
                positions_by_name.setdefault(nm, []).append(idx)

        drop_indices: set[int] = set()
        for nm, idxs in positions_by_name.items():
            if len(idxs) <= 1:
                continue
            for pos_i, earlier_idx in enumerate(idxs[:-1]):
                next_idx = idxs[pos_i + 1]
                used_between = False
                for mid_idx in range(earlier_idx + 1, next_idx + 1):
                    mid_st = stmts[mid_idx]
                    # Check RHS of next assign or any expression in between for Load of `nm`
                    scan_node: ast.AST = (
                        mid_st.value
                        if mid_idx == next_idx
                        and isinstance(mid_st, (ast.Assign, ast.AnnAssign))
                        and mid_st.value is not None
                        else mid_st
                    )
                    if any(
                        isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load)
                        and n.id == nm
                        for n in ast.walk(scan_node)
                    ):
                        used_between = True
                        break
                if not used_between:
                    drop_indices.add(earlier_idx)

        seen_imports: set[str] = set()
        cleaned: list[ast.stmt] = []
        for idx, st in enumerate(stmts):
            if idx in drop_indices:
                continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                imp_src = ast.unparse(st).strip()
                if imp_src in seen_imports:
                    continue
                seen_imports.add(imp_src)
            cleaned.append(st)
        return cleaned

    base_tree.body = _dedup_assigns_keep_last(base_tree.body)
    if target_class is not None:
        target_class.body = _dedup_assigns_keep_last(target_class.body)

    def _merge_top_extras(extras: list[ast.stmt]) -> list[str]:
        """Merges imports (unioning ImportFrom aliases), top-level constants, and helper classes."""
        changed_top: list[str] = []
        if not extras:
            return changed_top
        existing_top_src = {
            ast.unparse(stmt).strip()
            for stmt in base_tree.body
            if not isinstance(
                stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
        }
        existing_top_classes = {
            stmt.name: idx
            for idx, stmt in enumerate(base_tree.body)
            if isinstance(stmt, ast.ClassDef)
        }
        existing_top_assigns = {
            _assign_target_name(stmt): idx
            for idx, stmt in enumerate(base_tree.body)
            if _assign_target_name(stmt) is not None
        }

        def _refresh_top_indices() -> None:
            nonlocal existing_top_classes, existing_top_assigns
            existing_top_classes = {
                stmt.name: idx
                for idx, stmt in enumerate(base_tree.body)
                if isinstance(stmt, ast.ClassDef)
            }
            existing_top_assigns = {
                _assign_target_name(stmt): idx
                for idx, stmt in enumerate(base_tree.body)
                if _assign_target_name(stmt) is not None
            }

        def _is_future_import(stmt: ast.stmt) -> bool:
            return isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"

        def _goal_reg_key(stmt: ast.stmt) -> str | None:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                return None
            call = stmt.value
            fn = call.func
            fn_nm = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else "")
            )
            if fn_nm != "register_goal":
                return None
            if (
                call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                return call.args[0].value
            for kw in call.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    return kw.value.value
            return None

        def _insert_pos_for(stmt: ast.stmt) -> int:
            doc_offset = 0
            if (
                base_tree.body
                and isinstance(base_tree.body[0], ast.Expr)
                and isinstance(base_tree.body[0].value, ast.Constant)
                and isinstance(base_tree.body[0].value.value, str)
            ):
                doc_offset = 1
            if _is_future_import(stmt):
                return doc_offset
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                idx = doc_offset
                while idx < len(base_tree.body) and isinstance(
                    base_tree.body[idx], (ast.Import, ast.ImportFrom)
                ):
                    idx += 1
                return idx
            if isinstance(stmt, (ast.Expr, ast.If)):
                return len(base_tree.body)
            # If an assignment references any top-level function/class already in base_tree,
            # place it AFTER the last referenced definition so module execution never hits NameError.
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                rhs_names = {
                    n.id
                    for n in ast.walk(stmt.value)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                last_dep_idx = -1
                for j, st in enumerate(base_tree.body):
                    if (
                        isinstance(
                            st, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                        and st.name in rhs_names
                    ):
                        last_dep_idx = j
                if last_dep_idx >= 0:
                    return last_dep_idx + 1
            for j, st in enumerate(base_tree.body):
                if isinstance(
                    st, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    return j
            return len(base_tree.body)

        for extra in extras:
            if isinstance(extra, ast.ClassDef):
                if extra.name in existing_top_classes:
                    base_tree.body[existing_top_classes[extra.name]] = extra
                else:
                    base_tree.body.insert(_insert_pos_for(extra), extra)
                    _refresh_top_indices()
                changed_top.append(extra.name)
            elif isinstance(extra, ast.ImportFrom):
                matched_idx = None
                for i, st in enumerate(base_tree.body):
                    if (
                        isinstance(st, ast.ImportFrom)
                        and st.module == extra.module
                        and st.level == extra.level
                    ):
                        matched_idx = i
                        break
                if matched_idx is not None:
                    ex_if = base_tree.body[matched_idx]
                    assert isinstance(ex_if, ast.ImportFrom)
                    if any(a.name == "*" for a in extra.names):
                        if not any(a.name == "*" for a in ex_if.names):
                            ex_if.names = [ast.alias(name="*", asname=None)]
                            changed_top.append(f"import:{extra.module or ''}.*")
                    elif not any(a.name == "*" for a in ex_if.names):
                        have = {(a.name, a.asname) for a in ex_if.names}
                        for a in extra.names:
                            if (a.name, a.asname) not in have:
                                ex_if.names.append(copy.deepcopy(a))
                                have.add((a.name, a.asname))
                                changed_top.append(
                                    f"import:{extra.module or ''}.{a.name}"
                                )
                else:
                    base_tree.body.insert(_insert_pos_for(extra), extra)
                    _refresh_top_indices()
                    changed_top.append(f"import:{extra.module or ''}")
            elif isinstance(extra, ast.Import):
                raw = ast.unparse(extra).strip()
                if raw not in existing_top_src:
                    base_tree.body.insert(_insert_pos_for(extra), extra)
                    existing_top_src.add(raw)
                    _refresh_top_indices()
                    changed_top.append(raw)
            else:
                reg_k = _goal_reg_key(extra)
                if reg_k is not None:
                    replaced_reg = False
                    for idx_r, st_r in enumerate(base_tree.body):
                        if _goal_reg_key(st_r) == reg_k:
                            base_tree.body[idx_r] = extra
                            replaced_reg = True
                            changed_top.append(f"register_goal:{reg_k}")
                            break
                    if replaced_reg:
                        continue
                t_nm = _assign_target_name(extra)
                if t_nm is not None and t_nm in existing_top_assigns:
                    old_dump = ast.dump(base_tree.body[existing_top_assigns[t_nm]])
                    base_tree.body[existing_top_assigns[t_nm]] = extra
                    if ast.dump(extra) != old_dump:
                        changed_top.append(t_nm)
                elif ast.unparse(extra).strip() not in existing_top_src:
                    pos = _insert_pos_for(extra)
                    base_tree.body.insert(pos, extra)
                    existing_top_src.add(ast.unparse(extra).strip())
                    if t_nm is not None:
                        changed_top.append(t_nm)
                    _refresh_top_indices()
        return changed_top

    for node in patch_tree.body:
        if isinstance(node, ast.ClassDef):
            if target_class is not None and (
                node.name == target_class.name or node.name in ("Policy", "Agent")
            ):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        patch_methods[item.name] = item
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        c_nm = _assign_target_name(item)
                        if c_nm is not None:
                            patch_class_assigns[c_nm] = item
            else:
                patch_top_extras.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = _decorator_names(node)
            first_arg = _first_positional_arg(node)
            has_self = first_arg == "self"
            if has_self and target_class is not None:
                patch_methods[node.name] = node
            elif target_class is not None:
                existing_m = next(
                    (
                        m
                        for m in target_class.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and m.name == node.name
                    ),
                    None,
                )
                if existing_m is not None:
                    ex_decs = _decorator_names(existing_m)
                    is_static_or_cls = bool(
                        (decs | ex_decs) & {"staticmethod", "classmethod"}
                    )
                    if not is_static_or_cls and first_arg != "self":
                        if node.args.posonlyargs:
                            node.args.posonlyargs.insert(0, ast.arg(arg="self"))
                        else:
                            node.args.args.insert(0, ast.arg(arg="self"))
                    patch_methods[node.name] = node
                else:
                    patch_top_funcs[node.name] = node
            else:
                patch_top_funcs[node.name] = node
        elif isinstance(
            node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.If)
        ):
            patch_top_extras.append(node)
        elif isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            patch_top_extras.append(node)

    top_changed = _merge_top_extras(patch_top_extras)

    def _merge_top_funcs(
        funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[str]:
        changed_top_fns: list[str] = []
        if not funcs:
            return changed_top_fns
        seen_names: set[str] = set()
        deduped_body: list[ast.stmt] = []
        for stmt in base_tree.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name in funcs
            ):
                if stmt.name in seen_names:
                    continue
                seen_names.add(stmt.name)
            deduped_body.append(stmt)
        base_tree.body = deduped_body

        existing_map = {
            n.name: i
            for i, n in enumerate(base_tree.body)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, fn_node in funcs.items():
            if name in existing_map:
                idx = existing_map[name]
                if ast.dump(base_tree.body[idx]) != ast.dump(fn_node):
                    changed_top_fns.append(name)
                base_tree.body[idx] = fn_node
            else:
                changed_top_fns.append(name)
                insert_at = 0
                if (
                    base_tree.body
                    and isinstance(base_tree.body[0], ast.Expr)
                    and isinstance(base_tree.body[0].value, ast.Constant)
                    and isinstance(base_tree.body[0].value.value, str)
                ):
                    insert_at = 1
                while insert_at < len(base_tree.body) and isinstance(
                    base_tree.body[insert_at],
                    (
                        ast.Import,
                        ast.ImportFrom,
                        ast.Assign,
                        ast.AnnAssign,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    insert_at += 1
                base_tree.body.insert(insert_at, fn_node)
                existing_map = {
                    n.name: i
                    for i, n in enumerate(base_tree.body)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
        return changed_top_fns

    if target_class is None:
        changed = list(top_changed)
        changed.extend(_merge_top_funcs({**patch_top_funcs, **patch_methods}))
        return ast.unparse(base_tree), changed

    if patch_class_assigns:
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            or (
                isinstance(d, ast.Call)
                and (
                    (isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                    or (
                        isinstance(d.func, ast.Attribute) and d.func.attr == "dataclass"
                    )
                )
            )
            for d in getattr(target_class, "decorator_list", [])
        )
        cls_doc_offset = (
            1
            if (
                target_class.body
                and isinstance(target_class.body[0], ast.Expr)
                and isinstance(target_class.body[0].value, ast.Constant)
                and isinstance(target_class.body[0].value.value, str)
            )
            else 0
        )
        for c_nm, c_stmt in patch_class_assigns.items():
            repl_idx = next(
                (
                    idx
                    for idx, st in enumerate(target_class.body)
                    if _assign_target_name(st) == c_nm
                ),
                None,
            )
            if repl_idx is not None:
                target_class.body[repl_idx] = c_stmt
            elif is_dataclass:
                ins_dc = cls_doc_offset
                while ins_dc < len(target_class.body) and not isinstance(
                    target_class.body[ins_dc],
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    ins_dc += 1
                target_class.body.insert(ins_dc, c_stmt)
            else:
                target_class.body.insert(cls_doc_offset, c_stmt)

    existing_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        m.name: m
        for m in target_class.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    if "__init__" in existing_methods and "__init__" in patch_methods:
        patch_init = patch_methods["__init__"]
        exist_init = existing_methods["__init__"]

        def _get_self_attrs(stmt: ast.stmt) -> set[str]:
            attrs: set[str] = set()
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                    ):
                        attrs.add(t.attr)
            elif isinstance(stmt, ast.AnnAssign):
                t = stmt.target
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    attrs.add(t.attr)
            return attrs

        patch_assigned: set[str] = set()
        for stmt in patch_init.body:
            patch_assigned.update(_get_self_attrs(stmt))

        exist_only_locals = _collect_all_bound_names_in_fn(
            exist_init
        ) - _collect_all_bound_names_in_fn(patch_init)
        missing_stmts: list[ast.stmt] = []
        for stmt in exist_init.body:
            stmt_attrs = _get_self_attrs(stmt)
            # Only preserve missing `self.<attr>` initializers that are STILL referenced
            # outside `__init__` and don't depend on local variables absent from `patch_init`.
            if (
                stmt_attrs
                and (stmt_attrs & self_attrs_referenced_outside_init)
                and not (stmt_attrs & patch_assigned)
            ):
                used_vars = {
                    n.id
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                if used_vars & exist_only_locals:
                    continue
                missing_stmts.append(copy.deepcopy(stmt))
                patch_assigned.update(stmt_attrs)

        if missing_stmts:
            patch_init.body.extend(missing_stmts)

    changed_names: list[str] = []
    for name, new_fn in list(patch_methods.items()):
        if name in existing_methods:
            old_src = textwrap.dedent(ast.unparse(existing_methods[name])).strip()
            new_src = textwrap.dedent(ast.unparse(new_fn)).strip()
            if old_src != new_src:
                changed_names.append(name)
        else:
            changed_names.append(name)

    new_class_body: list[ast.stmt] = []
    inserted: set[str] = set()
    for item in target_class.body:
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name in patch_methods
        ):
            new_class_body.append(patch_methods[item.name])
            inserted.add(item.name)
        else:
            new_class_body.append(item)

    for name, fn_node in patch_methods.items():
        if name not in inserted:
            new_class_body.append(fn_node)

    target_class.body = new_class_body

    for name in _merge_top_funcs(patch_top_funcs):
        if name not in changed_names:
            changed_names.append(name)

    if not changed_names and patch_methods:
        changed_names = [k for k in patch_methods.keys() if k != "__init__"] or list(
            patch_methods.keys()
        )
    elif not changed_names and top_changed:
        changed_names = [t for t in top_changed if not t.startswith("import")] or list(
            top_changed
        )

    root_entrypoints = {"run", "step", "act", "dispatch", "step_dispatch", "__init__"}
    non_root_changed = [fn for fn in changed_names if fn not in root_entrypoints]
    if non_root_changed:
        changed_names = non_root_changed

    return ast.unparse(base_tree), changed_names


def prune_extracted_skill_functions(
    policy_code: str, skill_code: str
) -> tuple[str, list[str]]:
    """
    During `REFACTOR` mode, removes top-level functions or `Policy` methods from `policy_code`
    that were extracted into `skill_code` and are either imported from `skills.*`
    or no longer called directly in `policy_code`.
    """
    if not policy_code or not skill_code:
        return policy_code, []
    try:
        pol_tree = ast.parse(policy_code)
        skl_tree = ast.parse(skill_code)
    except Exception:
        return policy_code, []

    skill_fn_names = {
        n.name
        for n in skl_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not skill_fn_names:
        return policy_code, []

    imported_names: set[str] = set()
    for node in pol_tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "skills"
        ):
            for alias in node.names:
                imported_names.add(alias.name)
                if alias.asname:
                    imported_names.add(alias.asname)

    ref_bare_names: set[str] = set()
    ref_self_attrs: set[str] = set()
    for node in pol_tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in skill_fn_names
        ):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref_bare_names.add(sub.id)
            elif (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                ref_self_attrs.add(sub.attr)

    pruned_names: list[str] = []
    new_body: list[ast.stmt] = []
    for node in pol_tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in skill_fn_names
        ):
            if node.name in imported_names or node.name not in ref_bare_names:
                pruned_names.append(node.name)
                continue
        elif isinstance(node, ast.ClassDef) and node.name in ("Policy", "Agent"):
            new_cls_body: list[ast.stmt] = []
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in skill_fn_names
                    and item.name
                    not in (
                        "run",
                        "step",
                        "act",
                        "dispatch",
                        "step_dispatch",
                        "__init__",
                    )
                ):
                    if item.name not in ref_self_attrs:
                        pruned_names.append(item.name)
                        continue
                new_cls_body.append(item)
            node.body = new_cls_body or [ast.Pass()]
        new_body.append(node)

    pol_tree.body = new_body
    return ast.unparse(pol_tree), pruned_names


def audit_policy_ast_hygiene(policy_code: str) -> list[str]:
    """Performs a zero-heuristic static AST hygiene audit on `policy_code`."""
    if not policy_code or not policy_code.strip():
        return []
    try:
        tree = ast.parse(policy_code)
    except Exception:
        return []

    warnings: list[str] = []
    top_assign_counts: dict[str, int] = {}
    top_fn_counts: dict[str, int] = {}
    top_fn_arity: dict[str, int] = {}

    policy_cls: ast.ClassDef | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            nm = node.targets[0].id
            top_assign_counts[nm] = top_assign_counts.get(nm, 0) + 1
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nm = node.target.id
            top_assign_counts[nm] = top_assign_counts.get(nm, 0) + 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_fn_counts[node.name] = top_fn_counts.get(node.name, 0) + 1
            top_fn_arity[node.name] = len(node.args.posonlyargs) + len(node.args.args)
        elif isinstance(node, ast.ClassDef) and node.name in ("Policy", "Agent"):
            policy_cls = node

    dup_top = [f"`{k}` ({v}x)" for k, v in top_assign_counts.items() if v > 1]
    if dup_top:
        warnings.append(
            f"Duplicate top-level constant definitions (later overwrites earlier): {', '.join(dup_top)}"
        )

    dup_top_fns = [f"`{k}` ({v}x)" for k, v in top_fn_counts.items() if v > 1]
    if dup_top_fns:
        warnings.append(
            f"Duplicate top-level function definitions: {', '.join(dup_top_fns)}"
        )

    if policy_cls is not None:
        cls_assign_counts: dict[str, int] = {}
        cls_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        cls_method_counts: dict[str, int] = {}
        for item in policy_cls.body:
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
            ):
                nm = item.targets[0].id
                cls_assign_counts[nm] = cls_assign_counts.get(nm, 0) + 1
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                nm = item.target.id
                cls_assign_counts[nm] = cls_assign_counts.get(nm, 0) + 1
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls_methods[item.name] = item
                cls_method_counts[item.name] = cls_method_counts.get(item.name, 0) + 1

        dup_cls_assigns = [
            f"`Policy.{k}` ({v}x)" for k, v in cls_assign_counts.items() if v > 1
        ]
        if dup_cls_assigns:
            warnings.append(
                f"Duplicate class attribute assignments: {', '.join(dup_cls_assigns)}"
            )

        dup_cls_methods = [
            f"`Policy.{k}` ({v}x)" for k, v in cls_method_counts.items() if v > 1
        ]
        if dup_cls_methods:
            warnings.append(
                f"Duplicate class method definitions: {', '.join(dup_cls_methods)}"
            )

        for m_name, m_node in cls_methods.items():
            if m_name in top_fn_arity:
                m_non_self = len([a for a in m_node.args.args if a.arg != "self"])
                t_arity = top_fn_arity[m_name]
                warnings.append(
                    f"Name shadowing: `{m_name}` is defined both as a top-level function ({t_arity} arg(s)) "
                    f"and as `Policy.{m_name}(self, ...)` ({m_non_self} non-self arg(s)). Consolidate to avoid calling the wrong version."
                )

        called_attrs: set[str] = set()
        for sub in ast.walk(policy_cls):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                called_attrs.add(sub.attr)
        dead_methods = [
            m_name
            for m_name in cls_methods
            if m_name not in ("run", "step", "act", "__init__")
            and m_name not in called_attrs
        ]
        if dead_methods:
            warnings.append(
                f"Dead/unreferenced methods on `Policy` (never accessed via `self.<method>`): {', '.join(f'`{m}`' for m in dead_methods)}"
            )

    return warnings


def extract_module_outline(code: str, module_name: str = "") -> str:
    """
    Extracts high-level class/function signatures and one-line docstrings from
    Python module AST for context-scoped prompting without token bloat.
    """
    if not code or not code.strip():
        return ""
    try:
        tree = ast.parse(code)
    except Exception:
        return f"# Module: {module_name} (syntax error in source)"

    mod_lines = len(code.splitlines())
    lines = [f"# Module: {module_name} ({mod_lines} LOC)"] if module_name else []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        first_line = mod_doc.strip().split("\n")[0]
        lines.append(f'"""{first_line}"""')

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.isupper():
                    try:
                        lines.append(f"{tgt.id} = {ast.unparse(node.value)[:80]}")
                    except Exception:
                        pass
        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}:")
            cls_doc = ast.get_docstring(node)
            if cls_doc:
                lines.append(f'    """{cls_doc.strip().split(chr(10))[0]}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args_str = ", ".join(a.arg for a in item.args.args)
                    lines.append(f"    def {item.name}({args_str}):")
                    fn_doc = ast.get_docstring(item)
                    if fn_doc:
                        lines.append(
                            f'        """{fn_doc.strip().split(chr(10))[0]}"""'
                        )
                    lines.append("        ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args_str = ", ".join(a.arg for a in node.args.args)
            lines.append(f"def {node.name}({args_str}):")
            fn_doc = ast.get_docstring(node)
            if fn_doc:
                lines.append(f'    """{fn_doc.strip().split(chr(10))[0]}"""')
            lines.append("    ...")

    return "\n".join(lines)


def _node_dump(node: ast.AST | None) -> str:
    return ast.dump(node, include_attributes=False) if node is not None else ""


def three_way_ast_merge(base_src: str, src_a: str, src_b: str) -> str | None:
    """
    Deterministic AST-level 3-way merge of `(base_src, src_a, src_b)`.
    Supports module-level imports (unioned), constants/assignments, top-level functions,
    and `Policy` / `Agent` class definitions while preserving statement ordering.
    Returns merged source string, or `None` if both branches mutated the same symbol differently.
    """
    try:
        t_base = ast.parse(smart_dedent(base_src).strip() or "pass")
        t_a = ast.parse(smart_dedent(src_a).strip())
        t_b = ast.parse(smart_dedent(src_b).strip())
    except Exception:
        return None

    if _node_dump(t_a) == _node_dump(t_base):
        return smart_dedent(src_b).strip()
    if _node_dump(t_b) == _node_dump(t_base):
        return smart_dedent(src_a).strip()
    if _node_dump(t_a) == _node_dump(t_b):
        return smart_dedent(src_a).strip()

    def _stmt_key(stmt: ast.stmt) -> tuple[str, str] | None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("fn", stmt.name)
        if isinstance(stmt, ast.ClassDef):
            return ("cls", stmt.name)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return ("assign", stmt.targets[0].id)
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return ("assign", stmt.target.id)
        return None

    def _merge_body_lists(
        b_body: list[ast.stmt],
        a_body: list[ast.stmt],
        c_body: list[ast.stmt],
    ) -> list[ast.stmt] | None:
        def _collect_imp_tokens(
            body: list[ast.stmt],
        ) -> tuple[list[tuple[str, Any]], list[ast.stmt]]:
            toks: list[tuple[str, Any]] = []
            rest: list[ast.stmt] = []
            for st in body:
                if isinstance(st, ast.ImportFrom):
                    lvl = int(st.level or 0)
                    for al in st.names:
                        toks.append(("from", (st.module, lvl, al.name, al.asname)))
                elif isinstance(st, ast.Import):
                    for al in st.names:
                        toks.append(("imp", (al.name, al.asname)))
                else:
                    rest.append(st)
            return toks, rest

        base_toks, other_base = _collect_imp_tokens(b_body)
        a_toks, other_a = _collect_imp_tokens(a_body)
        b_toks, other_b = _collect_imp_tokens(c_body)
        base_tset, a_tset, b_tset = set(base_toks), set(a_toks), set(b_toks)
        kept_toks: list[tuple[str, Any]] = []
        seen_toks: set[tuple[str, Any]] = set()
        for tok in a_toks + b_toks:
            if tok in seen_toks:
                continue
            if tok in base_tset and (tok not in a_tset or tok not in b_tset):
                continue
            seen_toks.add(tok)
            kept_toks.append(tok)

        kept_toks.sort(
            key=lambda t: 0 if t[0] == "from" and t[1][0] == "__future__" else 1
        )
        imports_merged: list[ast.stmt] = []
        import_from_map: dict[tuple[str | None, int], ast.ImportFrom] = {}
        for kind, payload in kept_toks:
            if kind == "from":
                mod, lvl, nm, asnm = payload
                key = (mod, lvl)
                if key not in import_from_map:
                    node = ast.ImportFrom(
                        module=mod,
                        names=[ast.alias(name=nm, asname=asnm)],
                        level=lvl,
                    )
                    import_from_map[key] = node
                    imports_merged.append(node)
                else:
                    import_from_map[key].names.append(ast.alias(name=nm, asname=asnm))
            else:
                nm, asnm = payload
                imports_merged.append(
                    ast.Import(names=[ast.alias(name=nm, asname=asnm)])
                )

        def _pop_doc(bucket: list[ast.stmt]) -> ast.stmt | None:
            if (
                bucket
                and isinstance(bucket[0], ast.Expr)
                and isinstance(bucket[0].value, ast.Constant)
                and isinstance(bucket[0].value.value, str)
            ):
                return bucket.pop(0)
            return None

        doc_base = _pop_doc(other_base)
        doc_a = _pop_doc(other_a)
        doc_b = _pop_doc(other_b)
        db_d, da_d, dc_d = _node_dump(doc_base), _node_dump(doc_a), _node_dump(doc_b)
        if da_d == dc_d:
            doc_stmt = copy.deepcopy(doc_a) if doc_a is not None else None
        elif da_d == db_d:
            doc_stmt = copy.deepcopy(doc_b) if doc_b is not None else None
        elif dc_d == db_d:
            doc_stmt = copy.deepcopy(doc_a) if doc_a is not None else None
        else:
            return None

        def _index(
            stmts: list[ast.stmt],
        ) -> tuple[
            dict[tuple[str, str], ast.stmt],
            list[tuple[tuple[str, str] | None, ast.stmt]],
        ]:
            mp: dict[tuple[str, str], ast.stmt] = {}
            seq_items: list[tuple[tuple[str, str] | None, ast.stmt]] = []
            for st in stmts:
                k = _stmt_key(st)
                if k is not None:
                    mp[k] = st
                seq_items.append((k, st))
            return mp, seq_items

        mb, seq_base = _index(other_base)
        ma, seq_a = _index(other_a)
        mc, seq_b = _index(other_b)

        misc_base = [st for k, st in seq_base if k is None]
        misc_a = [st for k, st in seq_a if k is None]
        misc_b = [st for k, st in seq_b if k is None]
        sb, sa, sc = (
            [_node_dump(x) for x in misc_base],
            [_node_dump(x) for x in misc_a],
            [_node_dump(x) for x in misc_b],
        )
        if sa != sb and sc != sb and sa != sc:
            return None

        # Choose the authoritative sequence skeleton for misc placement:
        # if branch A changed misc, follow A's order; else if B changed misc, follow B's order; else base.
        primary_seq = seq_a if sa != sb else (seq_b if sc != sb else seq_base)

        merged_by_key: dict[tuple[str, str], ast.stmt] = {}
        all_keys: list[tuple[str, str]] = []
        seen_k: set[tuple[str, str]] = set()
        for seq_src in (primary_seq, seq_base, seq_a, seq_b):
            for k, _ in seq_src:
                if k is not None and k not in seen_k:
                    seen_k.add(k)
                    all_keys.append(k)

        for k in all_keys:
            nb, na, nc = mb.get(k), ma.get(k), mc.get(k)
            db, da, dc = _node_dump(nb), _node_dump(na), _node_dump(nc)
            if da == dc:
                if na is not None:
                    merged_by_key[k] = copy.deepcopy(na)
            elif da == db:
                if nc is not None:
                    merged_by_key[k] = copy.deepcopy(nc)
            elif dc == db:
                if na is not None:
                    merged_by_key[k] = copy.deepcopy(na)
            elif (
                isinstance(nb, ast.ClassDef)
                and isinstance(na, ast.ClassDef)
                and isinstance(nc, ast.ClassDef)
            ):
                cls_body = _merge_body_lists(nb.body, na.body, nc.body)
                if cls_body is None:
                    return None
                hb_b = [_node_dump(x) for x in nb.bases] + [
                    _node_dump(x) for x in nb.decorator_list
                ]
                hb_a = [_node_dump(x) for x in na.bases] + [
                    _node_dump(x) for x in na.decorator_list
                ]
                hb_c = [_node_dump(x) for x in nc.bases] + [
                    _node_dump(x) for x in nc.decorator_list
                ]
                if hb_a != hb_b and hb_c != hb_b and hb_a != hb_c:
                    return None
                hdr_src = na if hb_a != hb_b else nc
                merged_cls = copy.deepcopy(hdr_src)
                merged_cls.body = cls_body or [ast.Pass()]
                merged_by_key[k] = merged_cls
            else:
                return None

        # Emit statements in primary_seq order so misc statements stay in place, then append newly added symbols.
        emitted_keys: set[tuple[str, str]] = set()
        ordered_stmts: list[ast.stmt] = []
        for k, st in primary_seq:
            if k is None:
                ordered_stmts.append(copy.deepcopy(st))
            elif k in merged_by_key and k not in emitted_keys:
                ordered_stmts.append(merged_by_key[k])
                emitted_keys.add(k)
        for k in all_keys:
            if k in merged_by_key and k not in emitted_keys:
                ordered_stmts.append(merged_by_key[k])
                emitted_keys.add(k)

        out: list[ast.stmt] = []
        if doc_stmt is not None:
            out.append(doc_stmt)
        out.extend(imports_merged)
        out.extend(ordered_stmts)
        return out

    merged_body = _merge_body_lists(t_base.body, t_a.body, t_b.body)
    if merged_body is None:
        return None
    out_mod = ast.Module(body=merged_body, type_ignores=[])
    ast.fix_missing_locations(out_mod)
    return ast.unparse(out_mod)


_EVOLVE_START_RE = re.compile(r"^\s*#\s*EVOLVE-BLOCK\b(.*)$", re.IGNORECASE)
_EVOLVE_END_RE = re.compile(r"^\s*#\s*END-EVOLVE-BLOCK\b", re.IGNORECASE)


def extract_evolve_blocks(source: str) -> list[tuple[int, int, str]]:
    """Returns `[(start_line_0idx, end_line_0idx_excl, label), ...]` for `#EVOLVE-BLOCK` spans."""
    lines = (source or "").splitlines()
    blocks: list[tuple[int, int, str]] = []
    open_idx: int | None = None
    open_label = ""
    for i, ln in enumerate(lines):
        m_start = _EVOLVE_START_RE.match(ln)
        if m_start:
            open_idx = i
            open_label = (m_start.group(1) or "").strip()
        elif _EVOLVE_END_RE.match(ln) and open_idx is not None:
            blocks.append((open_idx, i + 1, open_label))
            open_idx = None
    return blocks


def check_evolve_block_bounds(base_source: str, candidate_source: str) -> str | None:
    """
    If `base_source` defines `#EVOLVE-BLOCK ... #END-EVOLVE-BLOCK` spans, verifies
    that immutable lines outside every `#EVOLVE-BLOCK` span remain unchanged in
    `candidate_source`. Returns an error description if violated, else None.
    """
    blocks = extract_evolve_blocks(base_source)
    if not blocks:
        return None
    cand_blocks = extract_evolve_blocks(candidate_source)
    if len(cand_blocks) != len(blocks):
        return (
            f"#EVOLVE-BLOCK boundary mismatch: expected {len(blocks)} block(s), "
            f"candidate has {len(cand_blocks)}."
        )
    base_lines = (base_source or "").splitlines()
    cand_lines = (candidate_source or "").splitlines()

    def _outside_chunks(
        lines: list[str], spans: list[tuple[int, int, str]]
    ) -> list[str]:
        chunks: list[str] = []
        cursor = 0
        for s_idx, e_idx, _ in spans:
            chunks.append("\n".join(lines[cursor:s_idx]))
            cursor = e_idx
        chunks.append("\n".join(lines[cursor:]))
        return chunks

    b_out = _outside_chunks(base_lines, blocks)
    c_out = _outside_chunks(cand_lines, cand_blocks)
    if b_out != c_out:
        return "Candidate modified frozen scaffold lines outside `#EVOLVE-BLOCK ... #END-EVOLVE-BLOCK`."
    return None
