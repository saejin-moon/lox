"""
Near-Full Python AST Validator & Surgical Function Splicer.
Supports full Python 3.11+ syntax (classes, generators, comprehensions, try/except,
type hints, f-strings, match/case, walrus operator, and safe stdlib imports)
while sandboxing OS, filesystem, subprocess, and dynamic code execution escapes.
"""

from __future__ import annotations

import ast
import copy
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
    patch_tree = validate_python_code(patch_code, filename="<patch>")
    if not base_policy_code or not base_policy_code.strip():
        fn_names = [
            n.name
            for n in ast.walk(patch_tree)
            if isinstance(n, ast.FunctionDef) and n.name != "__init__"
        ]
        return ast.unparse(patch_tree), fn_names

    base_tree = validate_python_code(base_policy_code, filename="<base_policy>")

    # Locate target class (`Policy` or `Agent` or first ClassDef) in base_tree
    target_class: ast.ClassDef | None = None
    for node in base_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Policy", "Agent"):
            target_class = node
            break
    if target_class is None:
        for node in base_tree.body:
            if isinstance(node, ast.ClassDef):
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

    for node in patch_tree.body:
        if isinstance(node, ast.ClassDef):
            if target_class is not None and (
                node.name == target_class.name or node.name in ("Policy", "Agent")
            ):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        patch_methods[item.name] = item
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        c_nm = _assign_target_name(item)
                        if c_nm is not None:
                            patch_class_assigns[c_nm] = item
                        else:
                            target_class.body.insert(0, item)
            else:
                patch_top_extras.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_self = bool(node.args.args and node.args.args[0].arg == "self")
            if has_self and target_class is not None:
                patch_methods[node.name] = node
            elif target_class is not None:
                existing_in_class = any(
                    isinstance(m, ast.FunctionDef) and m.name == node.name
                    for m in target_class.body
                )
                if existing_in_class or node.name in self_attrs_referenced:
                    node.args.args.insert(0, ast.arg(arg="self"))
                    patch_methods[node.name] = node
                else:
                    patch_top_funcs[node.name] = node
            else:
                patch_top_funcs[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            patch_top_extras.append(node)

    if target_class is None:
        # Base has no class; just merge top-level functions
        changed: list[str] = []
        existing_map = {
            n.name: i
            for i, n in enumerate(base_tree.body)
            if isinstance(n, ast.FunctionDef)
        }
        for name, fn_node in {**patch_top_funcs, **patch_methods}.items():
            changed.append(name)
            if name in existing_map:
                base_tree.body[existing_map[name]] = fn_node
            else:
                base_tree.body.append(fn_node)
        return ast.unparse(base_tree), changed

    # Merge any class-level assignments in-place onto target_class.body
    if patch_class_assigns:
        existing_cls_assigns = {
            _assign_target_name(st): idx
            for idx, st in enumerate(target_class.body)
            if _assign_target_name(st) is not None
        }
        for c_nm, c_stmt in reversed(list(patch_class_assigns.items())):
            if c_nm in existing_cls_assigns:
                target_class.body[existing_cls_assigns[c_nm]] = c_stmt
            else:
                target_class.body.insert(0, c_stmt)

    # Merge any top-level imports, constants, or helper classes before target_class (replacing constants in-place!)
    existing_top_src = {
        ast.unparse(stmt).strip()
        for stmt in base_tree.body
        if not isinstance(stmt, (ast.ClassDef, ast.FunctionDef))
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
    for extra in reversed(patch_top_extras):
        if isinstance(extra, ast.ClassDef):
            if extra.name in existing_top_classes:
                base_tree.body[existing_top_classes[extra.name]] = extra
            else:
                base_tree.body.insert(0, extra)
        else:
            t_nm = _assign_target_name(extra)
            if t_nm is not None and t_nm in existing_top_assigns:
                base_tree.body[existing_top_assigns[t_nm]] = extra
            elif ast.unparse(extra).strip() not in existing_top_src:
                base_tree.body.insert(0, extra)
                if t_nm is not None:
                    existing_top_assigns = {
                        _assign_target_name(s): i
                        for i, s in enumerate(base_tree.body)
                        if _assign_target_name(s) is not None
                    }

    # Compute which methods in patch_methods are genuinely new or modified compared to target_class
    existing_methods: dict[str, ast.FunctionDef] = {
        m.name: m for m in target_class.body if isinstance(m, ast.FunctionDef)
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

        missing_stmts: list[ast.stmt] = []
        for stmt in exist_init.body:
            stmt_attrs = _get_self_attrs(stmt)
            if stmt_attrs and not (stmt_attrs & patch_assigned):
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
        if isinstance(item, ast.FunctionDef) and item.name in patch_methods:
            new_class_body.append(patch_methods[item.name])
            inserted.add(item.name)
        else:
            new_class_body.append(item)

    for name, fn_node in patch_methods.items():
        if name not in inserted:
            new_class_body.append(fn_node)

    target_class.body = new_class_body

    # Merge any top-level helper functions
    existing_top = {
        n.name: i
        for i, n in enumerate(base_tree.body)
        if isinstance(n, ast.FunctionDef)
    }
    for name, fn_node in patch_top_funcs.items():
        if name not in changed_names:
            changed_names.append(name)
        if name in existing_top:
            base_tree.body[existing_top[name]] = fn_node
        else:
            base_tree.body.insert(0, fn_node)

    # If all methods matched identically, still report the method names present in patch_methods
    if not changed_names and patch_methods:
        changed_names = [k for k in patch_methods.keys() if k != "__init__"] or list(
            patch_methods.keys()
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

    called_bare_names: set[str] = set()
    called_self_attrs: set[str] = set()
    for node in pol_tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in skill_fn_names
        ):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    called_bare_names.add(sub.func.id)
                elif (
                    isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "self"
                ):
                    called_self_attrs.add(sub.func.attr)

    pruned_names: list[str] = []
    new_body: list[ast.stmt] = []
    for node in pol_tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in skill_fn_names
        ):
            if node.name in imported_names or node.name not in called_bare_names:
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
                    if item.name not in called_self_attrs:
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
        elif isinstance(node, ast.FunctionDef):
            top_fn_counts[node.name] = top_fn_counts.get(node.name, 0) + 1
            top_fn_arity[node.name] = len(node.args.args)
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
        cls_methods: dict[str, ast.FunctionDef] = {}
        cls_method_counts: dict[str, int] = {}
        for item in policy_cls.body:
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
            ):
                nm = item.targets[0].id
                cls_assign_counts[nm] = cls_assign_counts.get(nm, 0) + 1
            elif isinstance(item, ast.FunctionDef):
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

    macro_specs: dict[str, dict[str, Any]] = {}
    for node in macro_tree.body:
        if isinstance(node, ast.FunctionDef):
            is_gen = any(
                isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node)
            )
            args = [a.arg for a in node.args.args if a.arg != "self"]
            defaults_count = len(node.args.defaults)
            min_args = len(args) - defaults_count
            max_args = len(args)
            macro_specs[node.name] = {
                "min_args": min_args,
                "max_args": max_args,
                "is_generator": is_gen,
                "arg_names": set(args),
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
                passed_args = len(node.args)
                passed_kwargs = len(node.keywords)
                total_passed = passed_args + passed_kwargs
                if total_passed < spec["min_args"]:
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' requires at least {spec['min_args']} argument(s), "
                        f"but only {total_passed} were provided."
                    )
                elif total_passed > spec["max_args"]:
                    errors.append(
                        f"MacroContractError: Macro '{fn_name}' takes at most {spec['max_args']} argument(s), "
                        f"but {total_passed} were provided."
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

    lines = [f"# Module: {module_name}"] if module_name else []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        first_line = mod_doc.strip().split("\n")[0]
        lines.append(f'"""{first_line}"""')

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
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
