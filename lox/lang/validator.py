"""
Near-Full Python AST Validator & Surgical Function Splicer.
Supports full Python 3.11+ syntax (classes, generators, comprehensions, try/except,
type hints, f-strings, match/case, walrus operator, and safe stdlib imports)
while sandboxing OS, filesystem, subprocess, and dynamic code execution escapes.
"""

from __future__ import annotations

import ast
import copy
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
    lines = code.splitlines()
    code_lines = [
        line_str
        for line_str in lines
        if line_str.strip() and not line_str.strip().startswith("#")
    ]
    if not code_lines:
        return code
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
    return code


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

        def _insert_pos_for(stmt: ast.stmt) -> int:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                idx = 0
                if (
                    base_tree.body
                    and isinstance(base_tree.body[0], ast.Expr)
                    and isinstance(base_tree.body[0].value, ast.Constant)
                    and isinstance(base_tree.body[0].value.value, str)
                ):
                    idx = 1
                while idx < len(base_tree.body) and isinstance(
                    base_tree.body[idx], (ast.Import, ast.ImportFrom)
                ):
                    idx += 1
                return idx
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
                    have = {(a.name, a.asname) for a in ex_if.names}
                    for a in extra.names:
                        if (a.name, a.asname) not in have:
                            ex_if.names.append(copy.deepcopy(a))
                            have.add((a.name, a.asname))
                            changed_top.append(f"import:{extra.module or ''}.{a.name}")
                else:
                    base_tree.body.insert(_insert_pos_for(extra), extra)
                    changed_top.append(f"import:{extra.module or ''}")
            elif isinstance(extra, ast.Import):
                raw = ast.unparse(extra).strip()
                if raw not in existing_top_src:
                    base_tree.body.insert(_insert_pos_for(extra), extra)
                    existing_top_src.add(raw)
                    changed_top.append(raw)
            else:
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
                        existing_top_assigns = {
                            _assign_target_name(s): i
                            for i, s in enumerate(base_tree.body)
                            if _assign_target_name(s) is not None
                        }
                        existing_top_classes = {
                            s.name: i
                            for i, s in enumerate(base_tree.body)
                            if isinstance(s, ast.ClassDef)
                        }
        return changed_top

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

    # Merge top-level imports, constants, and helper classes BEFORE checking target_class is None
    # so modular skill patches (`skills/*.py`) keep imports & module constants too.
    top_changed = _merge_top_extras(patch_top_extras)

    if target_class is None:
        # Base has no class; merge top-level functions (deduplicating existing names in-place)
        changed: list[str] = list(top_changed)
        existing_map = {
            n.name: i
            for i, n in enumerate(base_tree.body)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, fn_node in {**patch_top_funcs, **patch_methods}.items():
            if name in existing_map:
                if ast.dump(base_tree.body[existing_map[name]]) != ast.dump(fn_node):
                    changed.append(name)
                base_tree.body[existing_map[name]] = fn_node
            else:
                changed.append(name)
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
                if (
                    fid in ("PrioritySelector", "BehaviorTree")
                    and n.args
                    and isinstance(n.args[0], ast.List)
                ):
                    found.append(n.args[0])
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
    for lb, la, lc, lm in zip(ps_base, ps_a, ps_b, ps_merged):
        base_dumps = [_node_dump(e) for e in lb.elts]
        a_dumps = [_node_dump(e) for e in la.elts]
        b_dumps = [_node_dump(e) for e in lc.elts]
        # Simple 3-way list merge: keep surviving/updated base elements + append additions from A and B
        a_set = set(a_dumps)
        b_set = set(b_dumps)
        new_elts: list[ast.expr] = []
        seen_dumps: set[str] = set()
        # Keep items from base present in both A and B (unless removed only by one -> respect removal)
        for elt, d in zip(lb.elts, base_dumps):
            if d in a_set and d in b_set and d not in seen_dumps:
                new_elts.append(copy.deepcopy(elt))
                seen_dumps.add(d)
        # Additions from A (or kept ordering relative to additions)
        base_set = set(base_dumps)
        for elt, d in zip(la.elts, a_dumps):
            if d not in base_set and d not in seen_dumps:
                new_elts.append(copy.deepcopy(elt))
                seen_dumps.add(d)
        for elt, d in zip(lc.elts, b_dumps):
            if d not in base_set and d not in seen_dumps:
                new_elts.append(copy.deepcopy(elt))
                seen_dumps.add(d)
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
        # Union-merge Import / ImportFrom statements first
        imports_merged: list[ast.stmt] = []
        import_from_map: dict[tuple[str | None, int], ast.ImportFrom] = {}
        seen_imports: set[str] = set()
        other_base: list[ast.stmt] = []
        other_a: list[ast.stmt] = []
        other_b: list[ast.stmt] = []

        for body, bucket in (
            (b_body, other_base),
            (a_body, other_a),
            (c_body, other_b),
        ):
            for st in body:
                if isinstance(st, ast.ImportFrom):
                    key = (st.module, int(st.level or 0))
                    if key not in import_from_map:
                        cp = copy.deepcopy(st)
                        import_from_map[key] = cp
                        imports_merged.append(cp)
                    else:
                        ex = import_from_map[key]
                        have = {(al.name, al.asname) for al in ex.names}
                        for al in st.names:
                            if (al.name, al.asname) not in have:
                                ex.names.append(copy.deepcopy(al))
                                have.add((al.name, al.asname))
                elif isinstance(st, ast.Import):
                    d = _node_dump(st)
                    if d not in seen_imports:
                        seen_imports.add(d)
                        imports_merged.append(copy.deepcopy(st))
                else:
                    bucket.append(st)

        # Keep module docstring if any
        doc_stmt: ast.stmt | None = None
        for bucket in (other_a, other_b, other_base):
            if (
                bucket
                and isinstance(bucket[0], ast.Expr)
                and isinstance(bucket[0].value, ast.Constant)
                and isinstance(bucket[0].value.value, str)
            ):
                if doc_stmt is None:
                    doc_stmt = copy.deepcopy(bucket[0])
                bucket.pop(0)

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
                merged_cls = copy.deepcopy(nb)
                merged_cls.body = cls_body or [ast.Pass()]
                merged_named.append(merged_cls)
            elif (
                allow_selector_merge
                and isinstance(nb, ast.FunctionDef)
                and isinstance(na, ast.FunctionDef)
                and isinstance(nc, ast.FunctionDef)
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

    skill_map = dict(skills or {})
    skill_exports: dict[str, set[str]] = {}
    skill_has_star: dict[str, bool] = {}

    sources: list[tuple[str, str]] = []
    for s_name, s_src in sorted(skill_map.items()):
        if not s_src or not s_src.strip():
            continue
        cleaned = smart_dedent(s_src.replace("\t", "    ")).strip()
        try:
            tree = ast.parse(cleaned, filename=f"skills/{s_name}.py")
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in skills/{s_name}.py line {exc.lineno}: {exc.msg}"
            ) from exc
        exp, has_star = _extract_module_exports(tree)
        skill_exports[s_name] = exp
        skill_has_star[s_name] = has_star
        sources.append((f"skills/{s_name}.py", cleaned))

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

        # 1. Cross-skill import & attribute validation
        skill_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = str(node.module or "")
                if mod == "skills":
                    for al in node.names:
                        s_mod = al.name
                        if skill_map and s_mod not in skill_exports:
                            raise PolicyValidationError(
                                f"Undefined skill module 'skills.{s_mod}' imported in {filename}"
                            )
                        skill_aliases[al.asname or s_mod] = s_mod
                elif mod.startswith("skills."):
                    s_mod = mod.split(".", 1)[1].split(".", 1)[0]
                    if skill_map and s_mod not in skill_exports:
                        raise PolicyValidationError(
                            f"Undefined skill module 'skills.{s_mod}' imported in {filename}"
                        )
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        for al in node.names:
                            if al.name != "*" and al.name not in skill_exports[s_mod]:
                                raise PolicyValidationError(
                                    f"Symbol '{al.name}' not exported by skills/{s_mod}.py (referenced in {filename})"
                                )
            elif isinstance(node, ast.Import):
                for al in node.names:
                    if al.name.startswith("skills."):
                        s_mod = al.name.split(".", 1)[1].split(".", 1)[0]
                        if skill_map and s_mod not in skill_exports:
                            raise PolicyValidationError(
                                f"Undefined skill module 'skills.{s_mod}' imported in {filename}"
                            )
                        if al.asname:
                            skill_aliases[al.asname] = s_mod

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
                # Match skills.<mod>.<attr> or <alias>.<attr>
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "skills"
                ):
                    s_mod = node.value.attr
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol 'skills.{s_mod}.{node.attr}' not found in skills/{s_mod}.py (referenced in {filename})"
                            )
                elif (
                    isinstance(node.value, ast.Name) and node.value.id in skill_aliases
                ):
                    s_mod = skill_aliases[node.value.id]
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol '{node.value.id}.{node.attr}' not found in skills/{s_mod}.py (referenced in {filename})"
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
                    s_mod = mod.split(".", 1)[1].split(".", 1)[0]
                    if (
                        any(al.name == "*" for al in node.names)
                        and s_mod in skill_exports
                    ):
                        mod_defined.update(skill_exports[s_mod])

        valid_globals = global_env | mod_defined | set(skill_exports.keys())

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
