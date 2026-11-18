"""
Near-Full Python AST Validator & Surgical Function Splicer.
Supports full Python 3.11+ syntax (classes, generators, comprehensions, try/except,
type hints, f-strings, match/case, walrus operator, and safe stdlib imports)
while sandboxing OS, filesystem, subprocess, and dynamic code execution escapes.
"""

from __future__ import annotations

import ast
import copy
import os
import re
import symtable
import textwrap
from typing import Any


class PolicyValidationError(Exception):
    """Raised when policy or macro code contains syntax errors or forbidden sandbox escapes."""


MODE_ALIASES: dict[str, str] = {
    "CREATE": "CREATE",
    "FULL_WRITE": "CREATE",
    "WRITE": "WRITE",
    "FUNC_MOD_ADD": "WRITE",
    "REFACTOR": "REFACTOR",
    "STRUCTURAL_REFACTOR": "REFACTOR",
    "STRUCTURAL_REWRITE": "REFACTOR",
    "SWARM_WORKER": "SWARM_WORKER",
    "SWARM_META": "SWARM_META",
    "SWARM_PEER": "SWARM_PEER",
}


def normalize_mode(mode: str | None) -> str:
    """Normalizes legacy or lowercase mode names to canonical (`CREATE`, `WRITE`, `REFACTOR`)."""
    if not mode:
        return "WRITE"
    u = str(mode).strip().upper()
    return MODE_ALIASES.get(u, u)


ALLOWED_IMPORT_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "collections",
        "heapq",
        "itertools",
        "functools",
        "re",
        "enum",
        "dataclasses",
        "typing",
        "numpy",
        "np",
        "random",
        "operator",
        "bisect",
        "copy",
        "skills",
        "lox",
    }
)

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "exit",
        "quit",
    }
)

FORBIDDEN_DUNDERS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__loader__",
        "__spec__",
        "__import__",
        "__mro__",
        "__bases__",
        "__dict__",
    }
)


ALLOWED_DUNDERS: frozenset[str] = frozenset(
    {
        "__init__",
        "__post_init__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__eq__",
        "__ne__",
        "__hash__",
        "__repr__",
        "__str__",
        "__len__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__call__",
        "__enter__",
        "__exit__",
    }
)


class SandboxASTVisitor(ast.NodeVisitor):
    """Validates an AST for sandbox safety while allowing rich Python constructs."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod not in ALLOWED_IMPORT_MODULES:
                raise PolicyValidationError(
                    f"Import of module '{alias.name}' is forbidden in policy sandbox. "
                    f"Allowed modules: {sorted(ALLOWED_IMPORT_MODULES)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0]
        if mod not in ALLOWED_IMPORT_MODULES:
            raise PolicyValidationError(
                f"Import from '{node.module}' is forbidden in policy sandbox. "
                f"Allowed modules: {sorted(ALLOWED_IMPORT_MODULES)}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise PolicyValidationError(
                    f"Call to builtin '{node.func.id}()' is forbidden in policy sandbox."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_DUNDERS or (
            node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS
        ):
            raise PolicyValidationError(
                f"Access to dunder attribute '{node.attr}' is forbidden in policy sandbox."
            )
        self.generic_visit(node)


def format_code_with_line_numbers(
    code: str,
    highlight_line: int | None = None,
    col_offset: int | None = None,
    window: int = 4,
) -> str:
    """
    Formats code lines with 1-based line numbers, highlighting the specified line and column with a caret.
    Provides clear visual context for compiler and syntax diagnostics sent to the Author Agent.
    """
    lines = code.splitlines()
    if not lines:
        return "(empty)"
    total_lines = len(lines)

    if highlight_line is not None and 1 <= highlight_line <= total_lines:
        start = max(1, highlight_line - window)
        end = min(total_lines, highlight_line + window)
    else:
        start = 1
        end = min(total_lines, window * 2 + 1)

    num_digits = max(2, len(str(end)))
    out: list[str] = []
    for ln in range(start, end + 1):
        line_content = lines[ln - 1]
        is_hl = ln == highlight_line
        marker = "> " if is_hl else "  "
        out.append(f"{marker}{str(ln).rjust(num_digits)} | {line_content}")
        if is_hl and col_offset is not None and col_offset > 0:
            out.append(" " * (2 + num_digits + 3) + " " * (col_offset - 1) + "^")
    return "\n".join(out)


def validate_python_code(code: str, filename: str = "<policy>") -> ast.Module:
    """Parses and validates Python source code, returning the parsed AST Module."""
    if not code or not code.strip():
        raise PolicyValidationError(f"Empty code submitted for {filename}.")
    cleaned = textwrap.dedent(code.replace("\t", "    ")).strip()
    try:
        tree = ast.parse(cleaned, filename=filename)
    except SyntaxError as exc:
        snippet = format_code_with_line_numbers(
            cleaned,
            highlight_line=exc.lineno,
            col_offset=exc.offset,
        )
        raise PolicyValidationError(
            f"SyntaxError in {filename} at line {exc.lineno}: {exc.msg}\n\n{snippet}"
        ) from exc

    visitor = SandboxASTVisitor()
    visitor.visit(tree)
    return tree


def is_ast_equivalent(code1: str, code2: str) -> bool:
    """Returns True if code1 and code2 parse to structurally equivalent ASTs (ignoring comments/whitespace)."""
    if not code1 or not code2:
        return False
    if code1.strip() == code2.strip():
        return True
    try:
        t1 = ast.dump(ast.parse(textwrap.dedent(code1).strip()))
        t2 = ast.dump(ast.parse(textwrap.dedent(code2).strip()))
        return t1 == t2
    except Exception:
        return False


def smart_dedent(code: str) -> str:
    """Smart dedents code even when comments or docstrings appear at column 0."""
    norm = (code or "").replace("\t", "    ")
    lines = norm.splitlines()
    code_lines = [
        line_str
        for line_str in lines
        if line_str.strip() and not line_str.strip().startswith("#")
    ]
    if not code_lines:
        return norm
    common_indent = min(
        len(line_str) - len(line_str.lstrip()) for line_str in code_lines
    )
    if common_indent > 0:
        new_lines = []
        for line_str in lines:
            if not line_str.strip():
                new_lines.append("")
            elif (
                line_str.strip().startswith("#")
                and (len(line_str) - len(line_str.lstrip())) < common_indent
            ):
                new_lines.append(line_str.strip())
            else:
                new_lines.append(
                    line_str[common_indent:]
                    if len(line_str) >= common_indent
                    else line_str.lstrip()
                )
        return "\n".join(new_lines)
    return norm


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
            if isinstance(n, ast.FunctionDef) and n.name != "__init__"
        ]
        return ast.unparse(patch_tree), fn_names

    base_tree = validate_python_code(base_policy_code, filename="<base_policy>")

    # Locate target class (`Policy` or `Agent` or a stateful class with `self` methods) in base_tree;
    # plain field-only @dataclass structs (e.g. ExploreState) must NOT act as target_class.
    target_class: ast.ClassDef | None = None
    for node in base_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Policy", "Agent"):
            target_class = node
            break
    if target_class is None:
        for node in base_tree.body:
            if isinstance(node, ast.ClassDef) and any(
                isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and bool(m.args.args and m.args.args[0].arg == "self")
                for m in node.body
            ):
                target_class = node
                break

    # Collect all attribute accesses `self.<attr>` in patch_tree and base_tree
    self_attrs_referenced: set[str] = set()
    for tree_to_scan in (patch_tree, base_tree):
        for sub in ast.walk(tree_to_scan):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                self_attrs_referenced.add(sub.attr)

    patch_methods: dict[str, ast.FunctionDef] = {}
    patch_class_assigns: dict[str, ast.stmt] = {}
    patch_top_funcs: dict[str, ast.FunctionDef] = {}
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
        last_idx_by_name: dict[str, int] = {}
        seen_imports: set[str] = set()
        for idx, st in enumerate(stmts):
            nm = _assign_target_name(st)
            if nm is not None:
                last_idx_by_name[nm] = idx
        cleaned: list[ast.stmt] = []
        for idx, st in enumerate(stmts):
            nm = _assign_target_name(st)
            if nm is not None and last_idx_by_name.get(nm) != idx:
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
            has_self = bool(node.args.args and node.args.args[0].arg == "self")
            if has_self and target_class is not None:
                patch_methods[node.name] = node
            elif target_class is not None:
                existing_in_class = any(
                    isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and m.name == node.name
                    for m in target_class.body
                )
                if existing_in_class:
                    if not (node.args.args and node.args.args[0].arg == "self"):
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

    # Merge top-level imports, constants, and helper classes BEFORE checking target_class is None
    # so modular skill patches (`skills/*.py`) keep imports & module constants too.
    top_changed = _merge_top_extras(patch_top_extras)

    def _merge_top_funcs(funcs: dict[str, ast.FunctionDef]) -> list[str]:
        """Replaces top-level functions in place (removing duplicates) and inserts new ones after imports."""
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
        # Base has no stateful class; merge top-level functions cleanly after imports/constants
        changed = list(top_changed)
        changed.extend(_merge_top_funcs({**patch_top_funcs, **patch_methods}))
        return ast.unparse(base_tree), changed

    # Merge any class-level assignments in-place onto target_class.body
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

    # Compute which methods in patch_methods are genuinely new or modified compared to target_class
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

        def _local_bound_in(fn_n: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
            names = {
                a.arg
                for a in (
                    list(fn_n.args.posonlyargs)
                    + list(fn_n.args.args)
                    + list(fn_n.args.kwonlyargs)
                    + ([fn_n.args.vararg] if fn_n.args.vararg else [])
                    + ([fn_n.args.kwarg] if fn_n.args.kwarg else [])
                )
            }
            for st in fn_n.body:
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    names.add(st.target.id)
            return names

        exist_only_locals = _local_bound_in(exist_init) - _local_bound_in(patch_init)
        missing_stmts: list[ast.stmt] = []
        for stmt in exist_init.body:
            stmt_attrs = _get_self_attrs(stmt)
            if stmt_attrs and not (stmt_attrs & patch_assigned):
                used_vars = {
                    n.id
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                if used_vars & exist_only_locals:
                    continue
                missing_stmts.append(copy.deepcopy(stmt))
                patch_assigned.update(stmt_attrs)

        insert_idx = 0
        if (
            patch_init.body
            and isinstance(patch_init.body[0], ast.Expr)
            and isinstance(patch_init.body[0].value, ast.Constant)
            and isinstance(patch_init.body[0].value.value, str)
        ):
            insert_idx = 1
        for s in missing_stmts:
            patch_init.body.insert(insert_idx, s)
            insert_idx += 1

    changed_names: list[str] = []

    for name, new_fn in list(patch_methods.items()):
        if name in existing_methods:
            old_src = textwrap.dedent(ast.unparse(existing_methods[name])).strip()
            new_src = textwrap.dedent(ast.unparse(new_fn)).strip()
            if old_src != new_src:
                changed_names.append(name)
        else:
            changed_names.append(name)

    # Replace in-place or append onto target_class.body
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

    # Merge any top-level helper functions cleanly after imports/constants (never at line 0)
    for name in _merge_top_funcs(patch_top_funcs):
        if name not in changed_names:
            changed_names.append(name)

    # If all methods matched identically, still report the method names present in patch_methods
    if not changed_names and patch_methods:
        changed_names = [k for k in patch_methods.keys() if k != "__init__"] or list(
            patch_methods.keys()
        )
    elif not changed_names and top_changed:
        changed_names = [t for t in top_changed if not t.startswith("import")] or list(
            top_changed
        )

    # When a caller entrypoint (`run`/`step`/`act`/`dispatch`) was modified alongside specific subroutines,
    # exclude root entrypoints from `changed_names` so conditional trigger filtering tracks the subroutines!
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
    Returns `(pruned_policy_code, pruned_fn_names)`.
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
    """
    Performs a zero-heuristic static AST hygiene audit on `policy_code`, detecting:
    1. Duplicate top-level variable or function definitions
    2. Duplicate class-level attribute or method definitions
    3. Name shadowing between module-level functions and `Policy` methods with mismatched arity
    4. Dead/unreferenced helper methods on `Policy`
    """
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

        # Check shadowing between top-level functions and Policy methods
        for m_name, m_node in cls_methods.items():
            if m_name in top_fn_arity:
                m_non_self = len([a for a in m_node.args.args if a.arg != "self"])
                t_arity = top_fn_arity[m_name]
                warnings.append(
                    f"Name shadowing: `{m_name}` is defined both as a top-level function ({t_arity} arg(s)) "
                    f"and as `Policy.{m_name}(self, ...)` ({m_non_self} non-self arg(s)). Consolidate to avoid calling the wrong version."
                )

        # Check dead/unreferenced methods on Policy
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


def check_macro_contracts(policy_code: str, macros_code: str) -> list[str]:
    """
    Statically inspects macro call sites in `policy_code` against function signatures
    and generator protocols defined in `macros_code`.
    Returns a list of diagnostic error strings (empty if contracts are satisfied).
    """
    if (
        not macros_code
        or not macros_code.strip()
        or not policy_code
        or not policy_code.strip()
    ):
        return []

    try:
        macro_tree = ast.parse(macros_code)
        policy_tree = ast.parse(policy_code)
    except Exception:
        return []

    def _has_shallow_yield(fn_n: ast.AST) -> bool:
        for ch in ast.iter_child_nodes(fn_n):
            if isinstance(ch, (ast.Yield, ast.YieldFrom)):
                return True
            if not isinstance(
                ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                if _has_shallow_yield(ch):
                    return True
        return False

    macro_specs: dict[str, dict[str, Any]] = {}
    for node in macro_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_gen = _has_shallow_yield(node)
            pos_args = [
                a.arg
                for a in (list(node.args.posonlyargs) + list(node.args.args))
                if a.arg != "self"
            ]
            min_args = max(0, len(pos_args) - len(node.args.defaults))
            max_args = len(pos_args) + len(node.args.kwonlyargs)
            req_kwonly = {
                a.arg
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if d is None
            }
            macro_specs[node.name] = {
                "min_args": min_args,
                "max_args": max_args,
                "has_vararg": node.args.vararg is not None,
                "has_kwarg": node.args.kwarg is not None,
                "req_kwonly": req_kwonly,
                "is_generator": is_gen,
                "arg_names": set(pos_args) | {a.arg for a in node.args.kwonlyargs},
            }

    if not macro_specs:
        return []

    errors: list[str] = []

    # Map each node to its parent to check if a call is inside a yield instead of yield from
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(policy_tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    top_level_fns = {
        n.name
        for n in policy_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for node in ast.walk(policy_tree):
        if isinstance(node, ast.Call):
            fn_name: str | None = None
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in macro_specs
                and node.func.id not in top_level_fns
            ):
                fn_name = node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in macro_specs
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("macros", "m")
            ):
                fn_name = node.func.attr

            if fn_name:
                spec = macro_specs[fn_name]
                has_star = any(isinstance(a, ast.Starred) for a in node.args)
                has_kwstar = any(kw.arg is None for kw in node.keywords)
                passed_args = len(node.args)
                kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                total_passed = passed_args + len(node.keywords)
                if not has_star and not has_kwstar and total_passed < spec["min_args"]:
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' requires at least {spec['min_args']} argument(s), "
                        f"but only {total_passed} were provided."
                    )
                elif (
                    not spec["has_vararg"]
                    and not spec["has_kwarg"]
                    and not has_star
                    and not has_kwstar
                    and total_passed > spec["max_args"]
                ):
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' takes at most {spec['max_args']} argument(s), "
                        f"but {total_passed} were provided."
                    )
                elif not has_kwstar and spec["req_kwonly"] - kw_names:
                    miss_kw = sorted(spec["req_kwonly"] - kw_names)
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' missing required keyword-only argument(s): {miss_kw}."
                    )

                # Check if generator macro is directly yielded (yield macro(...)) instead of (yield from macro(...))
                parent = parent_map.get(node)
                if spec["is_generator"] and isinstance(parent, ast.Yield):
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' is a generator. "
                        f"Use 'yield from {fn_name}(...)' instead of direct 'yield {fn_name}(...)'."
                    )

    return errors


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


def _try_merge_priority_selector_fn(
    fn_base: ast.FunctionDef,
    fn_a: ast.FunctionDef,
    fn_b: ast.FunctionDef,
) -> ast.FunctionDef | None:
    """
    If two branches both edited `build_tree` (or similar method) solely by adding/modifying
    distinct children inside `PrioritySelector([...])` without conflicting non-list lines,
    returns a merged FunctionDef; otherwise returns None.
    """
    if len(fn_base.body) != len(fn_a.body) or len(fn_base.body) != len(fn_b.body):
        return None

    merged_fn = copy.deepcopy(fn_base)

    def _find_ps_lists(tree: ast.AST) -> list[ast.List]:
        found: list[ast.List] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fid = (
                    n.func.id
                    if isinstance(n.func, ast.Name)
                    else (n.func.attr if isinstance(n.func, ast.Attribute) else "")
                )
                if fid in ("PrioritySelector", "BehaviorTree"):
                    if n.args and isinstance(n.args[0], ast.List):
                        found.append(n.args[0])
                    else:
                        for kw in n.keywords:
                            if kw.arg == "children" and isinstance(kw.value, ast.List):
                                found.append(kw.value)
                                break
        return found

    ps_base = _find_ps_lists(fn_base)
    ps_a = _find_ps_lists(fn_a)
    ps_b = _find_ps_lists(fn_b)
    if not ps_base or len(ps_base) != len(ps_a) or len(ps_base) != len(ps_b):
        return None

    # Verify everything OUTSIDE the PrioritySelector list items matches across base, A, B
    def _skeleton(fn: ast.FunctionDef) -> str:
        c = copy.deepcopy(fn)
        for lst in _find_ps_lists(c):
            lst.elts = []
        return ast.dump(c, include_attributes=False)

    if _skeleton(fn_base) != _skeleton(fn_a) or _skeleton(fn_base) != _skeleton(fn_b):
        return None

    ps_merged = _find_ps_lists(merged_fn)
    import difflib

    for lb, la, lc, lm in zip(ps_base, ps_a, ps_b, ps_merged):
        base_dumps = [_node_dump(e) for e in lb.elts]
        a_dumps = [_node_dump(e) for e in la.elts]
        b_dumps = [_node_dump(e) for e in lc.elts]
        # Position-preserving 3-way list merge keyed by base index boundaries.
        sm_a = difflib.SequenceMatcher(None, base_dumps, a_dumps, autojunk=False)
        sm_b = difflib.SequenceMatcher(None, base_dumps, b_dumps, autojunk=False)
        a_ops = [op for op in sm_a.get_opcodes() if op[0] != "equal"]
        b_ops = [op for op in sm_b.get_opcodes() if op[0] != "equal"]

        # Map non-equal base spans and zero-width insertion points
        edits_by_pos: list[tuple[int, int, str, list[ast.expr]]] = []
        a_spans: list[tuple[int, int]] = []
        for tag, i1, i2, j1, j2 in a_ops:
            edits_by_pos.append((i1, i2, "a", list(la.elts[j1:j2])))
            a_spans.append((i1, i2))

        def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
            if s1 == e1 and s2 == e2:
                return s1 == s2
            if s1 == e1:
                return s2 < s1 < e2
            if s2 == e2:
                return s1 < s2 < e1
            return max(s1, s2) < min(e1, e2)

        for tag, i1, i2, j1, j2 in b_ops:
            if any(_spans_overlap(i1, i2, a_i1, a_i2) for a_i1, a_i2 in a_spans):
                # Identical edit in both branches is non-conflicting; divergent/overlapping edits abort.
                matching_a = [
                    e
                    for e in edits_by_pos
                    if e[0] == i1
                    and e[1] == i2
                    and [_node_dump(x) for x in e[3]] == b_dumps[j1:j2]
                ]
                if not matching_a:
                    return None
                continue
            edits_by_pos.append((i1, i2, "b", list(lc.elts[j1:j2])))

        # Reconstruct list in ascending base order preserving relative positions
        edits_by_pos.sort(key=lambda e: (e[0], e[1]))
        new_elts: list[ast.expr] = []
        cursor = 0
        for i1, i2, _src, repl_elts in edits_by_pos:
            while cursor < i1:
                new_elts.append(copy.deepcopy(lb.elts[cursor]))
                cursor += 1
            for elt in repl_elts:
                new_elts.append(copy.deepcopy(elt))
            cursor = max(cursor, i2)
        while cursor < len(lb.elts):
            new_elts.append(copy.deepcopy(lb.elts[cursor]))
            cursor += 1
        lm.elts = new_elts
    return merged_fn


def three_way_ast_merge(base_src: str, src_a: str, src_b: str) -> str | None:
    """
    Deterministic AST-level 3-way merge of `(base_src, src_a, src_b)`.
    Supports module-level imports (unioned), constants/assignments, top-level functions,
    and `Policy` / `Agent` class definitions + disjoint `PrioritySelector([...])` edits.
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
        allow_selector_merge: bool = False,
    ) -> list[ast.stmt] | None:
        # Deletion-aware 3-way merge of Import / ImportFrom statements
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
            # Deletion in one branch when present in base removes it cleanly
            if tok in base_tset and (tok not in a_tset or tok not in b_tset):
                continue
            seen_toks.add(tok)
            kept_toks.append(tok)

        # Keep `__future__` imports pinned first
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

        # 3-way merge module/class docstring cleanly
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
        ) -> tuple[dict[tuple[str, str], ast.stmt], list[ast.stmt]]:
            mp: dict[tuple[str, str], ast.stmt] = {}
            misc: list[ast.stmt] = []
            for st in stmts:
                k = _stmt_key(st)
                if k is not None:
                    mp[k] = st
                else:
                    misc.append(st)
            return mp, misc

        mb, misc_base = _index(other_base)
        ma, misc_a = _index(other_a)
        mc, misc_b = _index(other_b)

        # Verify misc (e.g. top-level calls) didn't diverge incompatibly
        sb, sa, sc = (
            [_node_dump(x) for x in misc_base],
            [_node_dump(x) for x in misc_a],
            [_node_dump(x) for x in misc_b],
        )
        if sa != sb and sc != sb and sa != sc:
            return None
        merged_misc = misc_a if sa != sb else misc_b

        all_keys: list[tuple[str, str]] = []
        seen_k: set[tuple[str, str]] = set()
        for mp in (mb, ma, mc):
            for k in mp:
                if k not in seen_k:
                    seen_k.add(k)
                    all_keys.append(k)

        merged_named: list[ast.stmt] = []
        for k in all_keys:
            nb, na, nc = mb.get(k), ma.get(k), mc.get(k)
            db, da, dc = _node_dump(nb), _node_dump(na), _node_dump(nc)
            if da == dc:
                if na is not None:
                    merged_named.append(copy.deepcopy(na))
            elif da == db:
                if nc is not None:
                    merged_named.append(copy.deepcopy(nc))
            elif dc == db:
                if na is not None:
                    merged_named.append(copy.deepcopy(na))
            elif (
                isinstance(nb, ast.ClassDef)
                and isinstance(na, ast.ClassDef)
                and isinstance(nc, ast.ClassDef)
            ):
                cls_body = _merge_body_lists(
                    nb.body, na.body, nc.body, allow_selector_merge=True
                )
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
                merged_named.append(merged_cls)
            elif (
                allow_selector_merge
                and isinstance(nb, (ast.FunctionDef, ast.AsyncFunctionDef))
                and type(na) is type(nb)
                and type(nc) is type(nb)
            ):
                ps_merged = _try_merge_priority_selector_fn(nb, na, nc)
                if ps_merged is None:
                    return None
                merged_named.append(ps_merged)
            else:
                return None

        out: list[ast.stmt] = []
        if doc_stmt is not None:
            out.append(doc_stmt)
        out.extend(imports_merged)
        out.extend(merged_named)
        out.extend(copy.deepcopy(merged_misc))
        return out

    merged_body = _merge_body_lists(
        t_base.body, t_a.body, t_b.body, allow_selector_merge=True
    )
    if merged_body is None:
        return None
    out_mod = ast.Module(body=merged_body, type_ignores=[])
    ast.fix_missing_locations(out_mod)
    return ast.unparse(out_mod)


def _extract_module_exports(tree: ast.Module) -> tuple[set[str], bool]:
    """Returns `(exported_names, has_star_import)` for a skill module AST."""
    exports: set[str] = set()
    has_star = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    exports.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            exports.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.add(node.target.id)
        elif isinstance(node, ast.Import):
            for al in node.names:
                exports.add(al.asname or al.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                if al.name == "*":
                    has_star = True
                else:
                    exports.add(al.asname or al.name)
    return exports, has_star


def normalize_skill_key(raw: str) -> str:
    """Normalizes `skills/a/b.py`, `skills.a.b`, `a.b`, or `a/b` into `'a/b'` (max depth 2)."""
    s = str(raw or "").strip().replace("\\", "/")
    if s.startswith("skills/"):
        s = s[len("skills/") :]
    elif s.startswith("skills."):
        s = s[len("skills.") :]
    if s.endswith(".py"):
        s = s[:-3]
    s = s.replace(".", "/").strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) > 2:
        raise PolicyValidationError(
            f"Skill package depth > 2 is not allowed (`{raw}`): use `skills/<domain>/<module>.py`."
        )
    return "/".join(parts)


def skill_key_to_module(key: str) -> str:
    """Converts normalized key `'a/b'` or `'a'` to `'skills.a.b'` or `'skills.a'`."""
    clean = normalize_skill_key(key)
    return f"skills.{clean.replace('/', '.')}" if clean else "skills"


def skill_key_to_relpath(key: str) -> str:
    """Converts normalized key `'a/b'` or `'a'` to `'skills/a/b.py'` or `'skills/a.py'`."""
    clean = normalize_skill_key(key)
    return f"skills/{clean}.py" if clean else "skills/__init__.py"


def resolve_skill_import(mod_str: str, known_keys: set[str]) -> str:
    """Resolves `'skills.domain.sub'` or `'skills.domain'` against normalized `known_keys`."""
    tail = mod_str[len("skills.") :] if mod_str.startswith("skills.") else mod_str
    parts = [p for p in tail.replace("/", ".").split(".") if p]
    if len(parts) >= 2:
        two = f"{parts[0]}/{parts[1]}"
        if two in known_keys or not known_keys:
            return two
    if parts:
        one = parts[0]
        if one in known_keys:
            return one
        return "/".join(parts[:2])
    return ""


def _collect_scope_locals(node: ast.AST) -> set[str]:
    """Collects parameter/local assignment names inside a function or comprehension scope."""
    locals_set: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        for a in (
            list(getattr(args, "posonlyargs", []))
            + list(args.args)
            + list(args.kwonlyargs)
        ):
            locals_set.add(a.arg)
        if args.vararg is not None:
            locals_set.add(args.vararg.arg)
        if args.kwarg is not None:
            locals_set.add(args.kwarg.arg)
    if isinstance(node, ast.Lambda):
        return locals_set

    def _add_target(tgt: ast.AST) -> None:
        if isinstance(tgt, ast.Name):
            locals_set.add(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for elt in tgt.elts:
                _add_target(elt)
        elif isinstance(tgt, ast.Starred):
            _add_target(tgt.value)

    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(cur, ast.Assign):
            for t in cur.targets:
                _add_target(t)
        elif isinstance(cur, ast.AnnAssign):
            _add_target(cur.target)
        elif isinstance(cur, (ast.For, ast.AsyncFor)):
            _add_target(cur.target)
        elif isinstance(cur, (ast.With, ast.AsyncWith)):
            for it in cur.items:
                if it.optional_vars is not None:
                    _add_target(it.optional_vars)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            locals_set.add(cur.name)
        elif isinstance(cur, ast.NamedExpr):
            _add_target(cur.target)
        elif isinstance(cur, ast.comprehension):
            _add_target(cur.target)
        for ch in ast.iter_child_nodes(cur):
            stack.append(ch)
    return locals_set


def _walk_with_shadowed(tree: ast.AST):
    """Yields `(ast_node, shadowed_names)` respecting local bindings in inner scopes."""

    def _visit(node: ast.AST, shadowed: frozenset[str]):
        yield node, shadowed
        next_shadowed = shadowed
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            next_shadowed = shadowed | frozenset(_collect_scope_locals(node))
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            gen_names: set[str] = set()
            for comp in node.generators:
                stack_t: list[ast.AST] = [comp.target]
                while stack_t:
                    t = stack_t.pop()
                    if isinstance(t, ast.Name):
                        gen_names.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        stack_t.extend(t.elts)
                    elif isinstance(t, ast.Starred):
                        stack_t.append(t.value)
            next_shadowed = shadowed | frozenset(gen_names)
        for ch in ast.iter_child_nodes(node):
            yield from _visit(ch, next_shadowed)

    return _visit(tree, frozenset())


def check_undefined_symbols(
    policy_code: str,
    skills: dict[str, str] | None = None,
    allowed_globals: frozenset[str] = frozenset(),
) -> None:
    """
    Fast (<2ms) CPython `symtable` undefined-symbol and cross-skill export gate.
    Raises `PolicyValidationError` if any function, lambda, comprehension, or module
    references a free/global name that is not defined in the module or execution namespace,
    or imports/accesses a missing symbol from `skills.<mod>`.
    """
    import builtins

    known_builtins = frozenset(dir(builtins)) | frozenset(
        {"__name__", "__doc__", "__builtins__", "__package__", "__class__", "skills"}
    )
    global_env = known_builtins | frozenset(allowed_globals)

    skill_map: dict[str, str] = {}
    for raw_k, raw_v in (skills or {}).items():
        nk = normalize_skill_key(raw_k)
        if nk:
            skill_map[nk] = raw_v
    known_skill_keys = set(skill_map.keys())
    skill_pkg_names = {k.split("/", 1)[0] for k in known_skill_keys if "/" in k}
    skill_exports: dict[str, set[str]] = {}
    skill_has_star: dict[str, bool] = {}

    sources: list[tuple[str, str]] = []
    parsed_skill_trees: dict[str, ast.Module] = {}
    for s_name, s_src in sorted(skill_map.items()):
        if not s_src or not s_src.strip():
            continue
        cleaned = smart_dedent(s_src.replace("\t", "    ")).strip()
        rel_fn = skill_key_to_relpath(s_name)
        try:
            tree = ast.parse(cleaned, filename=rel_fn)
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {rel_fn} line {exc.lineno}: {exc.msg}"
            ) from exc
        exp, has_star = _extract_module_exports(tree)
        skill_exports[s_name] = exp
        skill_has_star[s_name] = has_star
        parsed_skill_trees[s_name] = tree
        sources.append((rel_fn, cleaned))

    # Propagate transitive `from skills.<mod> import *` exports across skill modules
    for _ in range(len(parsed_skill_trees) + 1):
        changed_exp = False
        for s_name, s_tree in parsed_skill_trees.items():
            for st in s_tree.body:
                if isinstance(st, ast.ImportFrom) and str(st.module or "").startswith(
                    "skills."
                ):
                    dep = resolve_skill_import(str(st.module), known_skill_keys)
                    if dep in skill_exports and any(al.name == "*" for al in st.names):
                        before_n = len(skill_exports[s_name])
                        skill_exports[s_name] |= skill_exports[dep]
                        if (
                            not skill_has_star.get(dep, False)
                            and len(skill_exports[s_name]) > before_n
                        ):
                            changed_exp = True
        if not changed_exp:
            break

    cleaned_pol = smart_dedent((policy_code or "").replace("\t", "    ")).strip()
    if cleaned_pol:
        sources.append(("policy.py", cleaned_pol))

    for filename, src in sources:
        try:
            tree = ast.parse(src, filename=filename)
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {filename} line {exc.lineno}: {exc.msg}"
            ) from exc

        # 1. Cross-skill import & attribute validation (via bundle.extract_skill_imports SSOT)
        from lox.lang.bundle import extract_skill_imports

        imp_info = extract_skill_imports(
            tree, known_skill_keys, pkg_names=skill_pkg_names
        )
        missing_mods = list(imp_info.missing_refs)
        if skills is not None:
            for rk in sorted(imp_info.referenced_keys - known_skill_keys):
                missing_mods.append(skill_key_to_module(rk))
        if missing_mods and (skill_map or skills is not None):
            raise PolicyValidationError(
                f"Undefined skill module '{missing_mods[0]}' imported in {filename}"
            )
        for s_mod, sym_name in imp_info.direct_symbols.values():
            if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                if sym_name not in skill_exports[s_mod]:
                    raise PolicyValidationError(
                        f"Symbol '{sym_name}' not exported by {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                    )
        skill_aliases: dict[str, str] = dict(imp_info.module_aliases)

        for node, shadowed in _walk_with_shadowed(tree):
            if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
                # Match skills.<pkg>.<mod>.<attr>, skills.<mod>.<attr>, or unshadowed <alias>.<attr>
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Attribute)
                    and isinstance(node.value.value.value, ast.Name)
                    and node.value.value.value.id == "skills"
                    and "skills" not in shadowed
                ):
                    s_mod = f"{node.value.value.attr}/{node.value.attr}"
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol 'skills.{node.value.value.attr}.{node.value.attr}.{node.attr}' "
                                f"not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )
                elif (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "skills"
                    and "skills" not in shadowed
                ):
                    s_mod = node.value.attr
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol 'skills.{s_mod}.{node.attr}' not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )
                elif (
                    isinstance(node.value, ast.Name)
                    and node.value.id in skill_aliases
                    and node.value.id not in shadowed
                ):
                    s_mod = skill_aliases[node.value.id]
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol '{node.value.id}.{node.attr}' not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )

        # 2. CPython symtable undefined-global check
        try:
            top_st = symtable.symtable(src, filename, "exec")
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {filename} line {exc.lineno}: {exc.msg}"
            ) from exc

        mod_defined: set[str] = set()
        for sym in top_st.get_symbols():
            if sym.is_local() or sym.is_imported():
                mod_defined.add(sym.get_name())
        # Also include explicit star-imported symbols from known skills
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                mod = str(node.module or "")
                if mod.startswith("skills."):
                    s_mod = resolve_skill_import(mod, known_skill_keys)
                    if (
                        any(al.name == "*" for al in node.names)
                        and s_mod in skill_exports
                    ):
                        mod_defined.update(skill_exports[s_mod])

        valid_globals = global_env | mod_defined

        def _check_table(tbl: symtable.SymbolTable, is_root: bool) -> None:
            for sym in tbl.get_symbols():
                nm = sym.get_name()
                if is_root:
                    if (
                        sym.is_referenced()
                        and not sym.is_local()
                        and not sym.is_imported()
                        and nm not in valid_globals
                    ):
                        raise PolicyValidationError(
                            f"Undefined global symbol '{nm}' at module level in {filename}"
                        )
                else:
                    if (
                        sym.is_global()
                        and sym.is_referenced()
                        and nm not in valid_globals
                    ):
                        scope_nm = tbl.get_name()
                        raise PolicyValidationError(
                            f"Undefined symbol '{nm}' in {filename}::{scope_nm}"
                        )
            for child in tbl.get_children():
                _check_table(child, is_root=False)

        _check_table(top_st, is_root=True)


# ---------------------------------------------------------------------------
# Cross-Module Contract Checker, Pyright `.pyi` Stubs & #EVOLVE-BLOCK Guards
# ---------------------------------------------------------------------------

_CACHED_STUB_FILES: dict[str, str] | None = None


def _build_class_stub(cls: Any) -> list[str]:
    """Builds compact `.pyi` stub lines for a Python class via runtime inspection."""
    import inspect

    bases = [
        b.__name__
        for b in cls.__bases__
        if b is not object and b.__module__ == cls.__module__
    ]
    base_clause = f"({', '.join(bases)})" if bases else ""
    lines = [f"class {cls.__name__}{base_clause}:"]
    seen_members: set[str] = set()

    def _clean_sig_params(sig_obj: Any, fallback: str) -> str:
        raw = str(sig_obj)
        # Strip inspect.signature return annotation before appending stub return type
        paren_depth = 0
        bracket_depth = 0
        cut_idx = -1
        for i, ch in enumerate(raw):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch in "[{":
                bracket_depth += 1
            elif ch in "]}":
                bracket_depth = max(0, bracket_depth - 1)
            elif paren_depth == 0 and bracket_depth == 0 and raw.startswith("->", i):
                cut_idx = i
                break
        if cut_idx >= 0:
            raw = raw[:cut_idx].rstrip()
        # Sanitize unparseable repr defaults like <factory> or <function ...>
        raw = re.sub(r"=<[^>]+>", "=...", raw)
        return raw if raw.startswith("(") else fallback

    if getattr(cls, "__init__", object.__init__) is not object.__init__:
        seen_members.add("__init__")
        try:
            init_sig = _clean_sig_params(
                inspect.signature(cls.__init__),
                "(self, *args: Any, **kwargs: Any)",
            )
        except Exception:
            init_sig = "(self, *args: Any, **kwargs: Any)"
        lines.append(f"    def __init__{init_sig} -> None: ...")
    annotations = getattr(cls, "__annotations__", {}) or {}
    for attr_name in annotations:
        if attr_name.startswith("__"):
            continue
        seen_members.add(attr_name)
        lines.append(f"    {attr_name}: Any")
    for attr_name, val in cls.__dict__.items():
        if attr_name.startswith("__"):
            continue
        if isinstance(val, property):
            seen_members.add(attr_name)
            lines.append(f"    @property\n    def {attr_name}(self) -> Any: ...")
        elif callable(val) and not isinstance(val, (staticmethod, classmethod)):
            seen_members.add(attr_name)
            try:
                sig = _clean_sig_params(
                    inspect.signature(val), "(self, *args: Any, **kwargs: Any)"
                )
            except Exception:
                sig = "(self, *args: Any, **kwargs: Any)"
            lines.append(f"    def {attr_name}{sig} -> Any: ...")
        elif isinstance(val, (staticmethod, classmethod)):
            seen_members.add(attr_name)
            fn = val.__func__
            dec = "@staticmethod" if isinstance(val, staticmethod) else "@classmethod"
            try:
                sig = _clean_sig_params(
                    inspect.signature(fn), "(*args: Any, **kwargs: Any)"
                )
            except Exception:
                sig = "(*args: Any, **kwargs: Any)"
            lines.append(f"    {dec}\n    def {attr_name}{sig} -> Any: ...")
        elif attr_name not in seen_members:
            seen_members.add(attr_name)
            lines.append(f"    {attr_name}: Any")
    # Dynamic attribute container support only for Blackboard
    if cls.__name__ == "Blackboard":
        lines.append("    def __getattr__(self, name: str) -> Any: ...")
        lines.append("    def __setattr__(self, name: str, value: Any) -> None: ...")
    if len(lines) == 1:
        lines.append("    pass")
    return lines


def _get_lox_stub_files() -> dict[str, str]:
    """Builds and caches minimal `.pyi` stubs for `lox.*` runtime APIs."""
    global _CACHED_STUB_FILES
    if _CACHED_STUB_FILES is not None:
        return _CACHED_STUB_FILES

    import inspect

    stub_map: dict[str, str] = {
        "lox/__init__.pyi": "",
        "lox/env/__init__.pyi": "",
        "lox/lang/__init__.pyi": "",
    }
    try:
        from lox.env import actions as act_mod
        from lox.env import state as st_mod
        from lox.lang import bt as bt_mod

        # lox/env/state.pyi
        st_lines = [
            "from __future__ import annotations",
            "from typing import Any, Iterator",
            "import numpy as np",
            "",
        ]
        for cls_name in (
            "InventoryItem",
            "MonsterInfo",
            "ObjectInfo",
            "VisibleMonster",
            "VisibleObject",
            "FloorMemory",
            "WorldMemory",
            "GameState",
        ):
            cls_obj = getattr(st_mod, cls_name, None)
            if cls_obj is not None:
                st_lines.extend(_build_class_stub(cls_obj))
                st_lines.append("")
        stub_map["lox/env/state.pyi"] = "\n".join(st_lines)

        # lox/env/actions.pyi
        act_lines = [
            "from __future__ import annotations",
            "from enum import IntEnum",
            "from typing import Any",
            "",
        ]
        action_cls = getattr(act_mod, "Action", None)
        if action_cls is not None:
            act_lines.append("class Action(IntEnum):")
            for m in action_cls:
                act_lines.append(f"    {m.name} = {int(m.value)}")
            act_lines.append("")
        for cls_name in ("RawAction", "ActionSequence"):
            cls_obj = getattr(act_mod, cls_name, None)
            if cls_obj is not None:
                act_lines.extend(_build_class_stub(cls_obj))
                act_lines.append("")
        for fn_name in (
            "key",
            "seq",
            "repeat",
            "extcmd",
            "menu_pick",
            "delta_to_dir",
            "dir_to_action",
        ):
            fn_obj = getattr(act_mod, fn_name, None)
            if callable(fn_obj):
                try:
                    sig = str(inspect.signature(fn_obj))
                except Exception:
                    sig = "(*args: Any, **kwargs: Any)"
                act_lines.append(f"def {fn_name}{sig} -> Any: ...")
        for const_name in (
            "DIR_DELTAS",
            "OPPOSITE_DIR",
            "CARDINAL_DELTAS",
            "ACTION_BY_NAME",
            "ACTION_BY_CHAR",
        ):
            if hasattr(act_mod, const_name):
                act_lines.append(f"{const_name}: Any")
        stub_map["lox/env/actions.pyi"] = "\n".join(act_lines)

        # lox/lang/bt.pyi
        bt_lines = [
            "from __future__ import annotations",
            "from typing import Any, Callable",
            "",
            "RUNNING: Any",
            "SUCCESS: Any",
            "FAILURE: Any",
            "",
        ]
        for cls_name in (
            "NodeStatus",
            "Blackboard",
            "BehaviorNode",
            "BehaviorTree",
            "Sequence",
            "MemorySequence",
            "Selector",
            "PrioritySelector",
            "Parallel",
            "ConditionLeaf",
            "ActionLeaf",
            "SubtreeNode",
            "Inverter",
            "RepeatUntilFail",
            "CooldownNode",
            "TimeoutNode",
            "LatchedAction",
        ):
            cls_obj = getattr(bt_mod, cls_name, None)
            if cls_obj is not None:
                bt_lines.extend(_build_class_stub(cls_obj))
                bt_lines.append("")
        bt_lines.append("def bt_leaf(*args: Any, **kwargs: Any) -> Any: ...")
        stub_map["lox/lang/bt.pyi"] = "\n".join(bt_lines)
    except Exception:
        pass

    stub_map["_lox_prelude.pyi"] = "\n".join(
        [
            "from __future__ import annotations",
            "from typing import Any as Any",
            "import math as math, collections as collections, heapq as heapq, itertools as itertools, functools as functools, re as re, enum as enum, dataclasses as dataclasses, typing as typing, random as random, operator as operator, bisect as bisect, copy as copy",
            "from collections import deque as deque, defaultdict as defaultdict, Counter as Counter",
            "from dataclasses import dataclass as dataclass",
            "import numpy as np",
            "import numpy as numpy",
            "from lox.env.actions import Action as Action, RawAction as RawAction, ActionSequence as ActionSequence, DIR_DELTAS as DIR_DELTAS, ACTION_BY_NAME as ACTION_BY_NAME, ACTION_BY_CHAR as ACTION_BY_CHAR, key as key, seq as seq, repeat as repeat, extcmd as extcmd",
            "from lox.env.state import GameState as GameState, WorldMemory as WorldMemory, FloorMemory as FloorMemory, InventoryItem as InventoryItem, MonsterInfo as MonsterInfo, ObjectInfo as ObjectInfo, VisibleMonster as VisibleMonster, VisibleObject as VisibleObject",
            "from lox.lang.bt import BehaviorNode as BehaviorNode, BehaviorTree as BehaviorTree, PrioritySelector as PrioritySelector, Sequence as Sequence, MemorySequence as MemorySequence, ActionLeaf as ActionLeaf, ConditionLeaf as ConditionLeaf, Inverter as Inverter, Blackboard as Blackboard, NodeStatus as NodeStatus, bt_leaf as bt_leaf, RUNNING as RUNNING, SUCCESS as SUCCESS, FAILURE as FAILURE",
            "from lox.lang.contracts import GoalIntent as GoalIntent, SafetyReport as SafetyReport, WorldTracker as WorldTracker, CapabilityScorer as CapabilityScorer, ThreatEvaluator as ThreatEvaluator",
            "from lox.lang.strategy import StrategyStack as StrategyStack, StrategyFrame as StrategyFrame, preempt_if as preempt_if, until as until, with_cleanup as with_cleanup",
            "def bfs_distance_grid(*args: Any, **kwargs: Any) -> Any: ...",
            "def astar_next_step(*args: Any, **kwargs: Any) -> Any: ...",
            "def macro(fn: Any) -> Any: ...",
            "def param(name: str, default: Any, min_val: Any = ..., max_val: Any = ...) -> Any: ...",
        ]
    )

    # Include lox.lang.contracts & lox.lang.strategy stubs when present
    for mod_path, pyi_key in (
        ("lox.lang.contracts", "lox/lang/contracts.pyi"),
        ("lox.lang.strategy", "lox/lang/strategy.pyi"),
    ):
        try:
            import importlib

            mod_obj = importlib.import_module(mod_path)
            m_lines = [
                "from __future__ import annotations",
                "from typing import Any",
                "",
            ]
            for exp_name in dir(mod_obj):
                if exp_name.startswith("_"):
                    continue
                exp_val = getattr(mod_obj, exp_name)
                if isinstance(exp_val, type) and exp_val.__module__ == mod_path:
                    m_lines.extend(_build_class_stub(exp_val))
                    m_lines.append("")
                elif (
                    callable(exp_val) and getattr(exp_val, "__module__", "") == mod_path
                ):
                    try:
                        sig = str(inspect.signature(exp_val))
                    except Exception:
                        sig = "(*args: Any, **kwargs: Any)"
                    m_lines.append(f"def {exp_name}{sig} -> Any: ...")
            stub_map[pyi_key] = "\n".join(m_lines)
        except Exception:
            pass

    _CACHED_STUB_FILES = stub_map
    return stub_map


def validate_cross_module_contracts(
    policy_code: str, skills: dict[str, str] | None = None
) -> None:
    """Checks static cross-module `skills.*` imports, signatures, and `self.*` calls."""
    skills_map: dict[str, str] = {}
    for raw_k, raw_v in (skills or {}).items():
        nk = normalize_skill_key(raw_k)
        if nk:
            skills_map[nk] = raw_v
    known_keys = set(skills_map.keys())
    pkg_names = {k.split("/", 1)[0] for k in known_keys if "/" in k}
    parsed_skills: dict[str, ast.Module] = {}
    skill_fn_sigs: dict[
        str, dict[str, tuple[list[str], int, int, bool, bool, set[str], set[str]]]
    ] = {}

    for s_name, s_src in skills_map.items():
        if not str(s_src or "").strip():
            continue
        rel_fn = skill_key_to_relpath(s_name)
        try:
            s_tree = ast.parse(s_src, filename=rel_fn)
        except SyntaxError:
            continue
        parsed_skills[s_name] = s_tree
        fn_map: dict[
            str, tuple[list[str], int, int, bool, bool, set[str], set[str]]
        ] = {}
        for node in s_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pos_param_names = [a.arg for a in node.args.posonlyargs] + [
                    a.arg for a in node.args.args
                ]
                pos_args = len(pos_param_names)
                req_pos = max(0, pos_args - len(node.args.defaults))
                has_varargs = node.args.vararg is not None
                has_kwargs = node.args.kwarg is not None
                req_kwonly = {
                    a.arg
                    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if d is None
                }
                kw_names = {a.arg for a in (node.args.args + node.args.kwonlyargs)}
                fn_map[node.name] = (
                    pos_param_names,
                    req_pos,
                    pos_args,
                    has_varargs,
                    has_kwargs,
                    kw_names,
                    req_kwonly,
                )
        skill_fn_sigs[s_name] = fn_map

    skill_exports_cache: dict[str, set[str]] = {}

    def _collect_skill_exports(
        mod_name: str, _visited: set[str] | None = None
    ) -> set[str]:
        if mod_name in skill_exports_cache:
            return skill_exports_cache[mod_name]
        visited = set(_visited or set())
        if mod_name in visited:
            return set()
        visited.add(mod_name)
        s_tree = parsed_skills.get(mod_name)
        if s_tree is None:
            return set()
        exports: set[str] = set()
        sig_map = skill_fn_sigs.setdefault(mod_name, {})
        for node in s_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        exports.add(tgt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                exports.add(node.target.id)
            elif isinstance(node, ast.ImportFrom) and node.module:
                src_mod = str(node.module)
                if src_mod.startswith("skills."):
                    dep_mod = resolve_skill_import(src_mod, known_keys)
                    dep_exports = _collect_skill_exports(dep_mod, visited)
                    dep_sigs = skill_fn_sigs.get(dep_mod, {})
                    for al in node.names:
                        if al.name == "*":
                            exports |= dep_exports
                            for k_fn, v_sig in dep_sigs.items():
                                sig_map.setdefault(k_fn, v_sig)
                        else:
                            bound = al.asname or al.name
                            if al.name in dep_exports:
                                exports.add(bound)
                            if al.name in dep_sigs:
                                sig_map.setdefault(bound, dep_sigs[al.name])
                else:
                    for al in node.names:
                        if al.name != "*":
                            exports.add(al.asname or al.name)
            elif isinstance(node, ast.Import):
                for al in node.names:
                    exports.add(al.asname or al.name.split(".")[0])
        skill_exports_cache[mod_name] = exports
        return exports

    for mod_k in list(parsed_skills.keys()):
        _collect_skill_exports(mod_k)

    try:
        p_tree = ast.parse(policy_code or "", filename="policy.py")
    except SyntaxError:
        return

    modules_to_check: list[tuple[str, ast.Module]] = [("policy.py", p_tree)] + [
        (skill_key_to_relpath(k), v) for k, v in parsed_skills.items()
    ]

    from lox.lang.bundle import extract_skill_imports

    for fname, tree in modules_to_check:
        imp_info = extract_skill_imports(tree, known_keys, pkg_names=pkg_names)
        missing_mods = list(imp_info.missing_refs)
        for rk in sorted(imp_info.referenced_keys - known_keys):
            missing_mods.append(skill_key_to_module(rk))
        if missing_mods:
            raise PolicyValidationError(
                f"Cross-module error in {fname}: imports missing skill module '{missing_mods[0]}'."
            )
        for s_mod, sym_name in imp_info.direct_symbols.values():
            if s_mod in parsed_skills:
                mod_exports = _collect_skill_exports(s_mod)
                if sym_name not in mod_exports:
                    raise PolicyValidationError(
                        f"Cross-module error in {fname}: '{skill_key_to_module(s_mod)}' has no symbol '{sym_name}'."
                    )
        module_aliases: dict[str, str] = dict(imp_info.module_aliases)
        direct_imports: dict[str, tuple[str, str]] = dict(imp_info.direct_symbols)

        # Check arity / kwarg mismatches on calls to imported skill functions
        for node, shadowed in _walk_with_shadowed(tree):
            if not isinstance(node, ast.Call):
                continue
            target: tuple[str, str] | None = None
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in direct_imports
                and node.func.id not in shadowed
            ):
                target = direct_imports[node.func.id]
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
                and node.func.value.id not in shadowed
            ):
                target = (module_aliases[node.func.value.id], node.func.attr)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Attribute)
                and isinstance(node.func.value.value.value, ast.Name)
                and node.func.value.value.value.id == "skills"
                and "skills" not in shadowed
            ):
                s_mod = f"{node.func.value.value.attr}/{node.func.value.attr}"
                target = (s_mod, node.func.attr)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "skills"
                and "skills" not in shadowed
            ):
                target = (node.func.value.attr, node.func.attr)
            if target is None:
                continue
            s_mod, fn_name = target
            sig = skill_fn_sigs.get(s_mod, {}).get(fn_name)
            if sig is None:
                continue
            (
                pos_param_names,
                req_pos,
                max_pos,
                has_varargs,
                has_kwargs,
                kw_names,
                req_kwonly,
            ) = sig
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                continue
            n_pos = len(node.args)
            kw_passed = {k.arg for k in node.keywords if k.arg is not None}
            if not has_varargs and n_pos > max_pos:
                raise PolicyValidationError(
                    f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                    f"skills.{s_mod}.{fn_name}() accepts at most {max_pos} positional args, got {n_pos}."
                )
            if not has_kwargs:
                unknown_kws = kw_passed - kw_names
                if unknown_kws:
                    raise PolicyValidationError(
                        f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                        f"skills.{s_mod}.{fn_name}() got unexpected keyword argument(s): {sorted(unknown_kws)}."
                    )
            req_pos_names = pos_param_names[:req_pos]
            missing_pos = [
                p
                for idx_p, p in enumerate(req_pos_names)
                if idx_p >= n_pos and p not in kw_passed
            ]
            missing_kwonly = sorted(req_kwonly - kw_passed)
            if missing_pos or missing_kwonly:
                raise PolicyValidationError(
                    f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                    f"skills.{s_mod}.{fn_name}() missing required argument(s): {missing_pos + missing_kwonly}."
                )


def _insert_prelude_after_future(src: str, prelude_header: str) -> tuple[str, int]:
    """
    Inserts `_lox_prelude` import after leading docstring and `from __future__` imports.
    Returns `(rewritten_src, inserted_at_1idx_line)` so diagnostic line numbers adjust accurately.
    """
    text = str(src or "")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return prelude_header + text, 1
    insert_after_line = 0
    for st in tree.body:
        if (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Constant)
            and isinstance(st.value.value, str)
            and insert_after_line == 0
        ):
            insert_after_line = int(getattr(st, "end_lineno", st.lineno) or st.lineno)
        elif isinstance(st, ast.ImportFrom) and st.module == "__future__":
            insert_after_line = int(getattr(st, "end_lineno", st.lineno) or st.lineno)
        else:
            break
    if insert_after_line <= 0:
        return prelude_header + text, 1
    lines = text.splitlines(keepends=True)
    head = "".join(lines[:insert_after_line])
    if head and not head.endswith("\n"):
        head += "\n"
    tail = "".join(lines[insert_after_line:])
    return head + prelude_header + tail, insert_after_line + 1


def run_pyright_on_package(
    policy_code: str,
    skills: dict[str, str] | None = None,
    timeout_sec: float = 8.0,
) -> list[str]:
    """
    Runs `pyright --outputjson` against `(policy.py, skills/*.py)` plus generated
    `lox/*.pyi` stubs in an isolated temp dir. Returns list of concise error lines.
    Gracefully returns `[]` if `pyright` binary is not installed.
    """
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    pyright_bin = shutil.which("pyright")
    if not pyright_bin:
        return []

    stubs = _get_lox_stub_files()
    skills_map = dict(skills or {})
    prelude_header = "from _lox_prelude import *  # pyright: ignore\n"
    prelude_lines = prelude_header.count("\n")
    prelude_offsets: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="lox_pyright_") as tmp_root:
        root = Path(tmp_root)
        # Write policy & skills with pre-injected LOX globals prelude (after docstring / __future__)
        pol_wrapped, pol_ins_line = _insert_prelude_after_future(
            policy_code or "", prelude_header
        )
        prelude_offsets["policy.py"] = pol_ins_line
        (root / "policy.py").write_text(pol_wrapped, encoding="utf-8")
        skills_dir = root / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "__init__.py").write_text("", encoding="utf-8")
        for s_name, s_src in skills_map.items():
            clean_k = normalize_skill_key(s_name)
            if clean_k and str(s_src or "").strip():
                dst = skills_dir / f"{clean_k}.py"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if "/" in clean_k:
                    pkg_init = dst.parent / "__init__.py"
                    if not pkg_init.exists():
                        pkg_init.write_text("", encoding="utf-8")
                s_wrapped, s_ins_line = _insert_prelude_after_future(
                    str(s_src), prelude_header
                )
                prelude_offsets[f"skills/{clean_k}.py"] = s_ins_line
                dst.write_text(s_wrapped, encoding="utf-8")

        # Write cached lox.*.pyi & _lox_prelude.pyi stubs under typings/
        typings_dir = root / "typings"
        for rel_path, stub_src in stubs.items():
            dst = typings_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(stub_src, encoding="utf-8")

        cfg = {
            "include": ["policy.py", "skills"],
            "stubPath": "typings",
            "pythonVersion": "3.11",
            "typeCheckingMode": "basic",
            "reportMissingImports": "error",
            "reportMissingModuleSource": "none",
            "reportUndefinedVariable": "error",
            "reportAttributeAccessIssue": "error",
            "reportCallIssue": "error",
            "reportArgumentType": "warning",
            "reportGeneralTypeIssues": "warning",
            "reportOptionalMemberAccess": "none",
            "reportOptionalSubscript": "none",
            "reportOptionalCall": "none",
            "reportOptionalOperand": "none",
            "reportOptionalIterable": "none",
            "reportReturnType": "warning",
        }
        (root / "pyrightconfig.json").write_text(json.dumps(cfg), encoding="utf-8")
        try:
            proc = subprocess.run(
                [pyright_bin, "--outputjson"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            data = json.loads(proc.stdout or "{}")
            diags = data.get("generalDiagnostics") or []
            errors: list[str] = []
            for d in diags:
                if str(d.get("severity", "")).lower() != "error":
                    continue
                rule = str(d.get("rule", "") or "")
                msg = str(d.get("message", "") or "").strip()
                if 'Cannot access attribute "param" for class "Policy"' in msg:
                    continue
                fpath = str(d.get("file", "") or "").replace("\\", "/")
                if "/skills/" in fpath:
                    rel_f = f"skills/{fpath.split('/skills/', 1)[1]}"
                else:
                    rel_f = os.path.basename(fpath) if fpath else "policy.py"
                raw_line = int(
                    ((d.get("range") or {}).get("start") or {}).get("line", 0)
                )
                line_1idx = raw_line + 1
                ins_line = prelude_offsets.get(rel_f, 1)
                if line_1idx > ins_line:
                    line_1idx = max(1, line_1idx - prelude_lines)
                if msg:
                    errors.append(f"{rel_f}:{line_1idx} [{rule or 'pyright'}]: {msg}")
            return errors[:12]
        except Exception as exc:
            return [
                f"policy.py:0 [pyright]: typecheck failed ({type(exc).__name__}: {exc})"
            ]


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
