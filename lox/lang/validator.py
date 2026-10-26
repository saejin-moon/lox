"""
Near-Full Python AST Validator & Surgical Function Splicer.
Supports full Python 3.11+ syntax (classes, generators, comprehensions, try/except,
type hints, f-strings, match/case, walrus operator, and safe stdlib imports)
while sandboxing OS, filesystem, subprocess, and dynamic code execution escapes.
"""

from __future__ import annotations

import ast
import textwrap


class PolicyValidationError(Exception):
    """Raised when policy or macro code contains syntax errors or forbidden sandbox escapes."""


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


def validate_python_code(code: str, filename: str = "<policy>") -> ast.Module:
    """Parses and validates Python source code, returning the parsed AST Module."""
    if not code or not code.strip():
        raise PolicyValidationError(f"Empty code submitted for {filename}.")
    cleaned = code.replace("\t", "    ").strip()
    try:
        tree = ast.parse(cleaned, filename=filename)
    except SyntaxError as exc:
        raise PolicyValidationError(
            f"SyntaxError in {filename} at line {exc.lineno}: {exc.msg}"
        ) from exc

    visitor = SandboxASTVisitor()
    visitor.visit(tree)
    return tree


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
    patch_top_funcs: dict[str, ast.FunctionDef] = {}
    patch_top_extras: list[ast.stmt] = []

    for node in patch_tree.body:
        if isinstance(node, ast.ClassDef):
            if target_class is not None and (
                node.name == target_class.name or node.name in ("Policy", "Agent")
            ):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        patch_methods[item.name] = item
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        target_class.body.insert(0, item)
            else:
                patch_top_extras.append(node)
        elif isinstance(node, ast.FunctionDef):
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

    # Merge any top-level imports, constants, or helper classes before target_class
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
    for extra in reversed(patch_top_extras):
        if isinstance(extra, ast.ClassDef):
            if extra.name in existing_top_classes:
                base_tree.body[existing_top_classes[extra.name]] = extra
            else:
                base_tree.body.insert(0, extra)
        elif ast.unparse(extra).strip() not in existing_top_src:
            base_tree.body.insert(0, extra)

    # Compute which methods in patch_methods are genuinely new or modified compared to target_class
    existing_methods: dict[str, ast.FunctionDef] = {
        m.name: m for m in target_class.body if isinstance(m, ast.FunctionDef)
    }
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

    # When a caller entrypoint (`run`/`step`/`act`) was modified alongside specific subroutines,
    # exclude root entrypoints from `changed_names` so conditional trigger filtering tracks the subroutines!
    root_entrypoints = {"run", "step", "act", "__init__"}
    non_root_changed = [fn for fn in changed_names if fn not in root_entrypoints]
    if non_root_changed:
        changed_names = non_root_changed

    return ast.unparse(base_tree), changed_names
