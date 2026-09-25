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
                f"Only boolean logic ('and', 'or', 'not'), comparisons, identifiers, calls, and numbers are permitted."
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

    if isinstance(node, ast.Call):
        args = []
        for a in node.args:
            if isinstance(a, ast.Constant):
                args.append(a.value)
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
                args.append(-a.operand.value)
            else:
                raise InfixSyntaxError("Arguments to predicate calls must be literal constants.")
        return ("call", node.func.id, args)

    if isinstance(node, ast.Name):
        return (node.id,)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ("true",) if node.value else ("false",)
        raise InfixSyntaxError(f"Bare constant '{node.value}' not allowed as standalone condition.")

    raise InfixSyntaxError(f"Cannot convert AST node {type(node).__name__} to condition tuple.")


def ast_to_canonical_expr(node: ast.AST) -> tuple:
    """Transforms a validated AST node into LOX-ψ's canonical ('call', name, args) tuple."""
    if isinstance(node, ast.Expression):
        return ast_to_canonical_expr(node.body)

    if isinstance(node, ast.BoolOp):
        op_name = "and" if isinstance(node.op, ast.And) else "or"
        children = [ast_to_canonical_expr(val) for val in node.values]
        return ("call", op_name, children)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ("call", "not", [ast_to_canonical_expr(node.operand)])
        raise InfixSyntaxError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise InfixSyntaxError("Chained comparisons (e.g. 1 < x < 5) are not permitted; use 'and'.")
        op_type = type(node.ops[0])
        if op_type not in _CMP_OPS:
            raise InfixSyntaxError(f"Unsupported comparison operator: {op_type.__name__}")
        op_str = _CMP_OPS[op_type]

        if not isinstance(node.left, ast.Name):
            raise InfixSyntaxError("Left side of comparison must be a valid identifier.")
        left_id = node.left.id

        right_node = node.comparators[0]
        if isinstance(right_node, ast.Constant):
            right_val = right_node.value
        elif isinstance(right_node, ast.UnaryOp) and isinstance(right_node.op, ast.USub) and isinstance(right_node.operand, ast.Constant):
            right_val = -right_node.operand.value
        else:
            raise InfixSyntaxError("Right side of comparison must be a literal constant.")

        if left_id in ("monster", "item") and op_str == "==":
            return ("call", left_id, [right_val])
        return ("call", left_id, [op_str, right_val])

    if isinstance(node, ast.Call):
        args = []
        for a in node.args:
            if isinstance(a, ast.Constant):
                args.append(a.value)
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
                args.append(-a.operand.value)
            else:
                raise InfixSyntaxError("Arguments to predicate calls must be literal constants.")
        return ("call", node.func.id, args)

    if isinstance(node, ast.Name):
        return ("call", node.id, [])

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ("call", "true" if node.value else "false", [])
        raise InfixSyntaxError(f"Bare constant '{node.value}' not allowed as standalone condition.")

    raise InfixSyntaxError(f"Cannot convert AST node {type(node).__name__} to canonical expression.")


def parse_infix_to_canonical(expr_str: str, max_depth: int = 3) -> tuple:
    """Parses a Pythonic infix condition string into LOX-ψ's canonical ('call', name, args) tuple."""
    cleaned = expr_str.strip()
    if not cleaned:
        raise InfixSyntaxError("Empty condition expression.")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise InfixSyntaxError(f"Syntax error in expression '{cleaned}': {e.msg}") from e

    validator = SafeASTVisitor(max_depth=max_depth)
    validator.visit(tree)

    depth = _measure_depth(tree)
    if depth > max_depth:
        raise InfixSyntaxError(
            f"Condition nesting depth {depth} exceeds allowed budget of {max_depth} in '{cleaned}'"
        )

    return ast_to_canonical_expr(tree)


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


def parse_pythonic_diff(text: str, max_depth: int = 3):
    """
    Parses a pure Pythonic Infix AST policy diff document into a PolicyDiff.
    Zero Lisp S-expression parentheses required.
    """
    from lox.policy.dsl import (  # noqa: PLC0415
        PolicyDiff,
        DiffHeader,
        SetOp,
        RuleOp,
        GoalOp,
        NogoodOp,
        DefmacroOp,
        NoteOp,
        DiffParseError,
    )

    cleaned = text.strip()
    if not cleaned:
        raise DiffParseError("ERR_PARSE", "empty diff document")

    # Header extraction
    m_rev = re.search(r"revision[:\s=]+(\d+)", cleaned, re.I)
    m_par = re.search(r"parent[:\s=]+(\d+)", cleaned, re.I)
    if not m_rev:
        raise DiffParseError("ERR_PARSE", "first statement must specify revision (e.g. revision: 5)")
    if not m_par:
        raise DiffParseError("ERR_HEADER", "revision header requires parent (e.g. parent: 4)")

    m_auth = re.search(r'author[:\s=]+["\']?([^,"\']+)["\']?', cleaned, re.I)
    m_dom = re.search(r'domain[:\s=]+["\']?([^,"\']+)["\']?', cleaned, re.I)
    m_reas = re.search(r'reason[:\s=]+["\']?([^"\'\n]+)["\']?', cleaned, re.I)

    header = DiffHeader(
        revision=int(m_rev.group(1)),
        parent=int(m_par.group(1)),
        author=m_auth.group(1).strip() if m_auth else "unknown",
        domain=m_dom.group(1).strip() if m_dom else "nethack",
        reason=m_reas.group(1).strip() if m_reas else "",
    )

    diff = PolicyDiff(header=header)

    lines = [line.strip() for line in cleaned.splitlines()]
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//") or line.startswith(";") or line.startswith("```"):
            continue

        # Skip standalone header line
        if re.search(r"^\s*revision\b", line, re.I) and not re.search(r"\b(set|rule|goal|macro|defmacro|nogood)\b", line, re.I):
            continue

        # Note / Reason statement
        m_note = re.match(r'^(?:note|reason)[:\s]+["\']?([^"\']+)["\']?$', line, re.I)
        if m_note:
            diff.notes.append(NoteOp(text=m_note.group(1).strip()))
            continue

        # Set tunable: `set policy_params.x = y` or `set x.y = z` or `x = y`
        m_set = re.match(r"^(?:set\s+)?([a-zA-Z0-9_.]+)\s*[:= ]\s*(.+)$", line, re.I)
        if m_set and "." in m_set.group(1):
            raw_path = m_set.group(1).strip()
            path = raw_path if raw_path.startswith("policy_params.") else f"policy_params.{raw_path}"
            val_str = m_set.group(2).strip()
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str
            if not isinstance(val, (int, float, bool, str)):
                raise DiffParseError("ERR_PARSE", f"invalid set value: {val_str}")
            diff.sets.append(SetOp(path=path, value=val))
            continue

        # Rule remove: `rule remove tactic_rules: 2` or `rule remove: 2`
        m_rrem = re.match(r"^rule\s+remove(?:\s+tactic_rules)?\s*[:= ]\s*(\d+)$", line, re.I)
        if m_rrem:
            diff.rules.append(RuleOp(action="remove", index=int(m_rrem.group(1)), rule=None))
            continue

        # Rule add: `rule add tactic_rules: when ...` or `rule: when ...` or `when ... do ...`
        m_radd = re.match(r"^(?:rule(?:\s+add)?(?:\s+tactic_rules)?\s*[:]\s*)?(when\s+.+)$", line, re.I)
        if m_radd:
            rule_content = m_radd.group(1).strip()
            note = None
            note_match = re.search(r'note\s+"([^"]*)"', rule_content, re.I)
            if note_match:
                note = note_match.group(1)
                rule_content = rule_content[:note_match.start()] + rule_content[note_match.end():]

            pat = r"when\s+(.+?)\s+do\s+([a-zA-Z0-9_]+)(?:\s+unless\s+(.+))?$"
            rmatch = re.search(pat, rule_content, re.I)
            if not rmatch:
                raise DiffParseError("ERR_PARSE", f"Invalid rule format in '{line}'. Expected: 'when <expr> do <verb> [unless <expr>]'")
            when_str = rmatch.group(1).strip()
            do_verb = rmatch.group(2).strip().lower()
            unless_str = rmatch.group(3).strip() if rmatch.group(3) else None

            when_ast = parse_infix_to_canonical(when_str, max_depth=max_depth)
            unless_ast = parse_infix_to_canonical(unless_str, max_depth=max_depth) if unless_str else None
            diff.rules.append(RuleOp(action="add", rule={"when": when_ast, "do": do_verb, "unless": unless_ast, "note": note}))
            continue

        # Goal prioritize: `goal prioritize reach_stairs: when ...`
        m_g_prio = re.match(r"^goal\s+prioritize\s+([a-zA-Z0-9_]+)\s*[:]?\s*(?:when\s+(.+?))?(?:\s+until\s+(.+))?$", line, re.I)
        if m_g_prio:
            goal_name = m_g_prio.group(1).strip()
            when_str = m_g_prio.group(2).strip() if m_g_prio.group(2) else None
            until_str = m_g_prio.group(3).strip() if m_g_prio.group(3) else None
            when_ast = parse_infix_to_canonical(when_str, max_depth=max_depth) if when_str else None
            until_ast = parse_infix_to_canonical(until_str, max_depth=max_depth) if until_str else None
            diff.goals.append(GoalOp(kind="prioritize", goal=goal_name, when=when_ast, until=until_ast))
            continue

        # Goal deprioritize: `goal deprioritize explore: after ...`
        m_g_deprio = re.match(r"^goal\s+deprioritize\s+([a-zA-Z0-9_]+)\s*[:]?\s*(?:after\s+(.+))?$", line, re.I)
        if m_g_deprio:
            goal_name = m_g_deprio.group(1).strip()
            after_str = m_g_deprio.group(2).strip() if m_g_deprio.group(2) else None
            after_ast = parse_infix_to_canonical(after_str, max_depth=max_depth) if after_str else None
            diff.goals.append(GoalOp(kind="deprioritize", goal=goal_name, after=after_ast))
            continue

        # Goal set threshold: `goal set_threshold reach_stairs: timeout = 500`
        m_g_thresh = re.match(r"^goal\s+set_threshold\s+([a-zA-Z0-9_]+)\s*[:]?\s*([a-zA-Z0-9_.]+)\s*[:= ]\s*(.+)$", line, re.I)
        if m_g_thresh:
            goal_name = m_g_thresh.group(1).strip()
            path = m_g_thresh.group(2).strip()
            val_str = m_g_thresh.group(3).strip()
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str
            diff.goals.append(GoalOp(kind="set_threshold", goal=goal_name, path=path, value=val))
            continue

        # Macro: `macro low_hp = hp_frac <= 0.50`
        m_macro = re.match(r"^(?:defmacro|macro)\s+([a-zA-Z0-9_]+)\s*=\s*(.+)$", line, re.I)
        if m_macro:
            m_name = m_macro.group(1).strip()
            body_str = m_macro.group(2).strip()
            body_ast = parse_infix_to_canonical(body_str, max_depth=max_depth)
            diff.defmacros.append(DefmacroOp(name=m_name, body=body_ast))
            continue

        # Nogood: `nogood: when ... cause "..." [forbid ...]`
        m_ng = re.match(r"^nogood\s*[:]?\s*when\s+(.+?)\s+cause\s+\"([^\"]+)\"(?:\s+forbid\s+([a-zA-Z0-9_]+))?$", line, re.I)
        if m_ng:
            when_ast = parse_infix_to_canonical(m_ng.group(1).strip(), max_depth=max_depth)
            cause = m_ng.group(2).strip()
            forbid = m_ng.group(3).strip() if m_ng.group(3) else None
            diff.nogoods.append(NogoodOp(when=when_ast, cause=cause, forbid=forbid))
            continue

        raise DiffParseError("ERR_PARSE", f"unrecognized statement: '{line}'")

    return diff
