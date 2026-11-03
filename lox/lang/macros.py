"""
Macro Library Manager (`data/workspace/macros.py`).
Allows the Author Agent to build, update, and reuse a growing library of helper
functions and generator subroutines across generations without hardcoding them in the harness.
"""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass
from typing import Any

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
        existing_assign_idx: dict[str, int] = {}
        existing_imports: set[str] = set()
        for idx, node in enumerate(base_tree.body):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                existing_idx[node.name] = idx
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                existing_assign_idx[node.targets[0].id] = idx
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                existing_assign_idx[node.target.id] = idx
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                existing_imports.add(ast.unparse(node).strip())

        updated_names: list[str] = []
        for node in new_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                updated_names.append(node.name)
                if node.name in existing_idx:
                    base_tree.body[existing_idx[node.name]] = node
                else:
                    existing_idx[node.name] = len(base_tree.body)
                    base_tree.body.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imp_src = ast.unparse(node).strip()
                if imp_src not in existing_imports:
                    existing_imports.add(imp_src)
                    base_tree.body.insert(0, node)
                    existing_idx = {k: v + 1 for k, v in existing_idx.items()}
                    existing_assign_idx = {
                        k: v + 1 for k, v in existing_assign_idx.items()
                    }
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                var_name = None
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    var_name = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    var_name = node.target.id
                if var_name and var_name in existing_assign_idx:
                    base_tree.body[existing_assign_idx[var_name]] = node
                else:
                    if var_name:
                        existing_assign_idx[var_name] = len(base_tree.body)
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


class MacroHarvester:
    """
    Bottom-up library learning utility (DreamCoder/Voyager style).
    Inspects promoted champion policies and harvests verified, stateless helper methods
    into `data/workspace/macros.py`.
    """

    @classmethod
    def harvest_from_policy(
        cls,
        policy_code: str,
        macro_lib: MacroLibrary,
        fn_stats: list[dict[str, Any]] | None = None,
        min_ochiai: float = 0.2,
    ) -> list[str]:
        """
        Extracts candidate helper methods from `policy_code` that:
        1. Are not root entrypoints (`run`, `step`, `act`, `__init__`).
        2. Do not access `self.<attr>` properties or mutate `self`.
        3. Had positive or neutral good-run correlation in the winning batch.
        Merges qualified methods into `macro_lib` and returns list of harvested macro names.
        """
        if not policy_code or not policy_code.strip():
            return []

        try:
            tree = ast.parse(policy_code)
        except Exception:
            return []

        # Find target class
        target_cls: ast.ClassDef | None = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and (
                node.name in ("Policy", "Agent")
                or any(
                    isinstance(m, ast.FunctionDef) and m.name in ("run", "step", "act")
                    for m in node.body
                )
            ):
                target_cls = node
                break

        if target_cls is None:
            return []

        # Build lookup for Ochiai scores if available
        fn_ochiai: dict[str, float] = {}
        if fn_stats:
            for fs in fn_stats:
                fn_ochiai[fs.get("function_name", "")] = float(
                    fs.get("ochiai_good_run_score", 0.0)
                )

        existing_macro_names = {s.name for s in macro_lib.list_specs()}
        root_names = {"run", "step", "act", "__init__", "explore"}
        harvested_code_snippets: list[str] = []
        harvested_names: list[str] = []

        for item in target_cls.body:
            if (
                not isinstance(item, ast.FunctionDef)
                or item.name in root_names
                or item.name.startswith("handle_")
            ):
                continue

            # Check if function name had poor statistical correlation
            if item.name in fn_ochiai and fn_ochiai[item.name] < min_ochiai:
                continue

            # Clone method and rewrite self accesses to state.mem or helper names
            cloned = copy.deepcopy(item)
            has_state_arg = any(a.arg in ("state", "st") for a in cloned.args.args)
            state_arg_name = "state"
            for a in cloned.args.args:
                if a.arg in ("state", "st"):
                    state_arg_name = a.arg
                    break

            class SelfRewriter(ast.NodeTransformer):
                def __init__(self) -> None:
                    self.invalid = False

                def visit_Attribute(self, node: ast.Attribute) -> Any:
                    self.generic_visit(node)
                    if isinstance(node.value, ast.Name) and node.value.id == "self":
                        if has_state_arg and node.attr in (
                            "visited",
                            "visit_count",
                            "pos_history",
                            "locked_doors",
                            "remembered_chars",
                            "is_oscillating",
                            "passable_mask",
                            "door_mask",
                            "searched_counts",
                            "stair_graph",
                            "kv",
                            "recent_messages",
                        ):
                            return ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=state_arg_name, ctx=ast.Load()),
                                    attr="mem",
                                    ctx=ast.Load(),
                                ),
                                attr=node.attr,
                                ctx=node.ctx,
                            )
                        elif (
                            node.attr in existing_macro_names
                            or node.attr in harvested_names
                        ):
                            return ast.Name(id=node.attr, ctx=node.ctx)
                        else:
                            self.invalid = True
                    return node

            rewriter = SelfRewriter()
            cloned = rewriter.visit(cloned)
            ast.fix_missing_locations(cloned)

            if rewriter.invalid:
                continue

            # Remove self from arguments
            if cloned.args.args and cloned.args.args[0].arg == "self":
                cloned.args.args.pop(0)

            # Ensure docstring
            doc = ast.get_docstring(cloned)
            if not doc:
                doc_node = ast.Expr(
                    value=ast.Constant(
                        value=f"Harvested helper macro `{cloned.name}` from champion policy."
                    )
                )
                cloned.body.insert(0, doc_node)

            harvested_code_snippets.append(ast.unparse(cloned))
            harvested_names.append(cloned.name)

        if not harvested_code_snippets:
            return []

        combined_code = "\n\n".join(harvested_code_snippets)
        try:
            macro_lib.merge_macros(combined_code)
            return harvested_names
        except Exception:
            return []
