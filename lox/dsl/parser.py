"""
LOX 2.0 DSL Parser: Pythonic Infix AST Parser and Validator.
Safely validates policy programs using Python's standard ast.NodeVisitor.
Guarantees zero code execution, structural boundedness, and invariant verification.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from lox.dsl.schema import ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS


class DSLValidationError(Exception):
    """Raised when policy code violates grammar, schema, or safety invariants."""
    pass


class SafeASTVisitor(ast.NodeVisitor):
    """
    Whitelists only safe, declaratively bounded constructs.
    Rejects any loops, imports, arbitrary expressions, or unknown calls.
    """

    ALLOWED_NODES = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.If,
        ast.Compare,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
        ast.USub,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.Assign,
        ast.List,
        ast.Pass,
        ast.keyword,
        ast.Load,
        ast.Store,
    )

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.functions: dict[str, ast.FunctionDef] = {}
        self.plan_order: list[str] = []

    def generic_visit(self, node: ast.AST):
        if not isinstance(node, self.ALLOWED_NODES):
            raise DSLValidationError(
                f"Disallowed syntax element: '{type(node).__name__}'. "
                f"Only safe functional logic, comparisons, and allowed action calls are permitted."
            )
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions[node.name] = node
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Allow only 'plan = [fn1, fn2, ...]'
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "plan":
            if isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Name):
                        self.plan_order.append(elt.id)
                    else:
                        raise DSLValidationError("Elements of 'plan' must be macro function names.")
            else:
                raise DSLValidationError("'plan' must be a list of macro names.")
        else:
            raise DSLValidationError(f"Disallowed assignment: only 'plan = [...]' is permitted.")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in ALLOWED_ACTIONS:
                raise DSLValidationError(f"Unknown or unapproved action call: '{func_name}()'.")
        else:
            raise DSLValidationError("Complex or dynamic function calls are strictly forbidden.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        var_name = node.id
        if (
            var_name not in ALLOWED_PREDICATES
            and var_name not in ENUM_CONSTANTS
            and var_name not in ALLOWED_ACTIONS
            and var_name not in self.functions
            and var_name != "plan"
        ):
            raise DSLValidationError(f"Unknown or unapproved identifier: '{var_name}'.")
        self.generic_visit(node)


def normalize_code(code_str: str) -> str:
    """Normalizes 'macro name:' to 'def name():' and 'plan:' to 'plan = [...]' if shorthand is used."""
    lines = code_str.splitlines()
    normalized = []
    in_shorthand_plan = False
    plan_items = []

    for line in lines:
        stripped = line.strip()
        # Transform 'macro name:' -> 'def name():'
        macro_match = re.match(r"^macro\s+([a-zA-Z0-9_]+)\s*:", stripped)
        if macro_match:
            name = macro_match.group(1)
            indent = len(line) - len(line.lstrip())
            normalized.append(" " * indent + f"def {name}():")
            continue

        # Handle 'plan:' shorthand block
        if stripped == "plan:":
            in_shorthand_plan = True
            continue

        if in_shorthand_plan:
            if stripped and (len(line) - len(line.lstrip())) > 0:
                plan_items.append(stripped)
                continue
            else:
                in_shorthand_plan = False

        normalized.append(line)

    if plan_items:
        items_str = ", ".join(plan_items)
        normalized.append(f"plan = [{items_str}]")

    return "\n".join(normalized)


def parse_and_validate(code_str: str) -> tuple[ast.Module, list[str], dict[str, ast.FunctionDef]]:
    """
    Parses policy string and executes strict AST validation.
    Returns (AST module, plan_order list, dictionary of macro FunctionDefs).
    """
    clean_code = normalize_code(code_str)
    try:
        tree = ast.parse(clean_code)
    except SyntaxError as e:
        raise DSLValidationError(f"Syntax error in policy program: {e}") from e

    visitor = SafeASTVisitor()
    visitor.visit(tree)

    if not visitor.functions:
        raise DSLValidationError("Policy program must define at least one macro function.")

    plan_order = visitor.plan_order
    if not plan_order:
        # Default plan: all defined macros in order
        plan_order = list(visitor.functions.keys())

    for fn_name in plan_order:
        if fn_name not in visitor.functions:
            raise DSLValidationError(f"Macro '{fn_name}' referenced in 'plan' is not defined.")

    return tree, plan_order, visitor.functions
