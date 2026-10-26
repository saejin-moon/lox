"""
Macro Library Manager (`data/workspace/macros.py`).
Allows the Author Agent to build, update, and reuse a growing library of helper
functions and generator subroutines across generations without hardcoding them in the harness.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

from lox.lang.validator import validate_python_code

DEFAULT_MACROS_PATH = "data/workspace/macros.py"


@dataclass(frozen=True, slots=True)
class MacroSpec:
    """Extracted signature and docstring for a macro in `macros.py`."""

    name: str
    signature: str
    docstring: str
    is_generator: bool
    source: str


class MacroLibrary:
    """Manages persistent storage, AST merging, and documentation of reusable macros."""

    def __init__(self, path: str = DEFAULT_MACROS_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

    def load_code(self) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def save_code(self, code: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n" if code.strip() else "")

    def merge_macros(self, new_macro_code: str) -> tuple[str, list[str]]:
        """
        Validates `new_macro_code` and merges all functions/classes into `self.path`.
        Existing macros with the same name are updated in place; new macros are appended.
        Returns `(merged_code, updated_or_added_macro_names)`.
        """
        new_tree = validate_python_code(new_macro_code, filename="<macros_patch>")
        current_code = self.load_code()
        if not current_code:
            names = [
                n.name
                for n in new_tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            ]
            merged = ast.unparse(new_tree)
            self.save_code(merged)
            return merged, names

        base_tree = validate_python_code(current_code, filename=self.path)
        existing_idx: dict[str, int] = {}
        for idx, node in enumerate(base_tree.body):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                existing_idx[node.name] = idx

        updated_names: list[str] = []
        for node in new_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                updated_names.append(node.name)
                if node.name in existing_idx:
                    base_tree.body[existing_idx[node.name]] = node
                else:
                    existing_idx[node.name] = len(base_tree.body)
                    base_tree.body.append(node)
            elif isinstance(node, ast.Assign):
                base_tree.body.append(node)

        merged = ast.unparse(base_tree)
        self.save_code(merged)
        return merged, updated_names

    def upsert_macros(self, new_macro_code: str) -> list[str]:
        """Merges `new_macro_code` into `self.path` and returns the list of saved macro names."""
        _, names = self.merge_macros(new_macro_code)
        return names

    def describe_macros(self) -> str:
        """Alias for `format_docs()`."""
        return self.format_docs()

    def list_specs(self) -> list[MacroSpec]:
        code = self.load_code()
        if not code:
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        specs: list[MacroSpec] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args_str = ast.unparse(node.args)
                ret_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                sig = f"def {node.name}({args_str}){ret_str}"
                doc = ast.get_docstring(node) or "No docstring provided."
                is_gen = any(
                    isinstance(sub, (ast.Yield, ast.YieldFrom))
                    for sub in ast.walk(node)
                )
                specs.append(
                    MacroSpec(
                        name=node.name,
                        signature=sig,
                        docstring=doc.strip(),
                        is_generator=is_gen,
                        source=ast.unparse(node),
                    )
                )
        return specs

    def format_docs(self) -> str:
        specs = self.list_specs()
        if not specs:
            return "No custom macros have been defined in `macros.py` yet."
        lines = ["### Registered Macros (`macros.py`):"]
        for s in specs:
            kind = (
                "[Generator Macro: call via `state = yield from "
                + s.name
                + "(state, ...)`]"
                if s.is_generator
                else "[Helper Function: call via `" + s.name + "(...)`]"
            )
            lines.append(f"- `{s.signature}` {kind}\n  {s.docstring}")
        return "\n".join(lines)
