"""
LOX-ψ Policy Layer: Pythonic Infix AST Parser and Validator.

Provides safe, sandboxed parsing of Pythonic infix boolean conditions into LOX-ψ's
canonical condition tree. Operates via Python's standard library `ast` module with
a strict AST NodeVisitor whitelist — guaranteeing zero code execution (no eval/exec),
deterministic operator precedence, depth bounding (<= 3), and near-100% LLM syntactic
accuracy compared to Lisp S-expressions.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any


class InfixSyntaxError(ValueError):
    """Raised when an infix expression violates grammar, safety, or depth rules."""


# Allowed comparison operators mapped to canonical string representation
_CMP_OPS = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


class SafeASTVisitor(ast.NodeVisitor):
    """
    Validates that an AST consists strictly of safe boolean logic and comparisons.
    Enforces maximum tree depth and rejects any executable constructs.
    """

    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
    )

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.current_depth = 0

    def generic_visit(self, node: ast.AST):
        if not isinstance(node, self.ALLOWED_NODES):
            raise InfixSyntaxError(
                f"Disallowed syntax element: {type(node).__name__}. "
                f"Only boolean logic ('and', 'or', 'not'), comparisons, identifiers, and numbers are permitted."
            )
        super().generic_visit(node)


def _measure_depth(node: ast.AST) -> int:
    """Measures logical condition nesting depth."""
    if isinstance(node, ast.Expression):
        return _measure_depth(node.body)
    elif isinstance(node, ast.BoolOp):
        return 1 + max((_measure_depth(val) for val in node.values), default=0)
    elif isinstance(node, ast.UnaryOp):
        return 1 + _measure_depth(node.operand)
    elif isinstance(node, ast.Compare):
        return 1
    elif isinstance(node, (ast.Name, ast.Constant)):
        return 0
    return 1


def _ast_to_condition_tuple(node: ast.AST) -> tuple | str | int | float | bool:
    """
    Transforms a validated Python AST node into the canonical LOX-ψ condition tuple.
    """
    if isinstance(node, ast.Expression):
        return _ast_to_condition_tuple(node.body)

    if isinstance(node, ast.BoolOp):
        op_name = "and" if isinstance(node.op, ast.And) else "or"
        children = [_ast_to_condition_tuple(val) for val in node.values]
        return (op_name, *children)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ("not", _ast_to_condition_tuple(node.operand))
        raise InfixSyntaxError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise InfixSyntaxError("Chained comparisons (e.g. 1 < x < 5) are not permitted; use 'and'.")
        op_type = type(node.ops[0])
        if op_type not in _CMP_OPS:
            raise InfixSyntaxError(f"Unsupported comparison operator: {op_type.__name__}")
        op_str = _CMP_OPS[op_type]

        # Extract left side (identifier)
        if isinstance(node.left, ast.Name):
            left_id = node.left.id
        else:
            raise InfixSyntaxError("Left side of comparison must be a valid attribute or statistic identifier.")

        # Extract right side (constant literal)
        right_node = node.comparators[0]
        if isinstance(right_node, ast.Constant):
            right_val = right_node.value
        elif isinstance(right_node, ast.UnaryOp) and isinstance(right_node.op, ast.USub) and isinstance(right_node.operand, ast.Constant):
            right_val = -right_node.operand.value
        else:
            raise InfixSyntaxError("Right side of comparison must be a literal constant.")

        return ("compare", left_id, op_str, right_val)

    if isinstance(node, ast.Name):
        return (node.id,)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ("true",) if node.value else ("false",)
        raise InfixSyntaxError(f"Bare constant '{node.value}' not allowed as standalone condition.")

    raise InfixSyntaxError(f"Cannot convert AST node {type(node).__name__} to condition tuple.")


def parse_infix_condition(expr_str: str, max_depth: int = 3) -> tuple:
    """
    Parses a Pythonic infix condition string into a canonical LOX-ψ condition tuple.

    Examples:
        'hp_frac <= 0.5' -> ('compare', 'hp_frac', '<=', 0.5)
        'stairs_known and not blind' -> ('and', ('stairs_known',), ('not', ('blind',)))
        'low_hp and (adjacent_hostiles >= 2 or stunned)' -> ...

    Raises InfixSyntaxError on invalid syntax, disallowed AST elements, or depth budget exceeded.
    """
    cleaned = expr_str.strip()
    if not cleaned:
        raise InfixSyntaxError("Empty condition expression.")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise InfixSyntaxError(f"Syntax error in expression '{cleaned}': {e.msg}") from e

    # Security check: verify no dangerous or executable nodes
    validator = SafeASTVisitor(max_depth=max_depth)
    validator.visit(tree)

    # Budget check: verify nesting depth <= max_depth
    depth = _measure_depth(tree)
    if depth > max_depth:
        raise InfixSyntaxError(
            f"Condition nesting depth {depth} exceeds allowed budget of {max_depth} in '{cleaned}'"
        )

    return _ast_to_condition_tuple(tree)


@dataclass
class InfixMacro:
    name: str
    expr_str: str
    body: tuple


def parse_infix_macro(line: str, max_depth: int = 3) -> InfixMacro:
    """
    Parses a macro definition line in format:
        defmacro <name> = <infix-expression>
        OR macro <name> = <infix-expression>
    """
    cleaned = line.strip()
    match = re.match(r"^(?:defmacro|macro)\s+([a-z_][a-z0-9_]*)\s*=\s*(.+)$", cleaned, re.IGNORECASE)
    if not match:
        raise InfixSyntaxError(
            f"Invalid macro definition: '{cleaned}'. Expected: 'defmacro <name> = <expression>'"
        )

    name = match.group(1).lower()
    expr_str = match.group(2).strip()
    body = parse_infix_condition(expr_str, max_depth=max_depth)
    return InfixMacro(name=name, expr_str=expr_str, body=body)


@dataclass
class InfixRule:
    when: tuple
    do: str
    unless: tuple | None = None
    note: str | None = None


def parse_infix_rule(text: str, max_depth: int = 3) -> InfixRule:
    """
    Parses a declarative rule line or block.
    Format:
        rule: when <expr> do <verb> [unless <expr>] [note "<text>"]
    """
    cleaned = text.strip()
    if cleaned.lower().startswith("rule:"):
        cleaned = cleaned[5:].strip()

    # Extract note if present
    note = None
    note_match = re.search(r'note\s+"([^"]*)"', cleaned, re.IGNORECASE)
    if note_match:
        note = note_match.group(1)
        cleaned = cleaned[:note_match.start()] + cleaned[note_match.end():]

    # Pattern: when ... do ... [unless ...]
    pattern = r"when\s+(.+?)\s+do\s+([a-z_][a-z0-9_]*)(?:\s+unless\s+(.+))?$"
    match = re.search(pattern, cleaned, re.IGNORECASE)
    if not match:
        raise InfixSyntaxError(
            f"Invalid rule format: '{text}'. Expected: 'rule: when <expr> do <verb> [unless <expr>]'"
        )

    when_str = match.group(1).strip()
    do_verb = match.group(2).strip().lower()
    unless_str = match.group(3).strip() if match.group(3) else None

    when_tuple = parse_infix_condition(when_str, max_depth=max_depth)
    unless_tuple = parse_infix_condition(unless_str, max_depth=max_depth) if unless_str else None

    return InfixRule(when=when_tuple, do=do_verb, unless=unless_tuple, note=note)
