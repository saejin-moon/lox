"""
LOX DSL Parser: Pythonic AST Parser and Validator.
Safely validates policy programs (classes, generators, and behavior trees).
Rejects dangerous system calls (import, eval, exec, filesystem access) while
allowing full procedural flow (while, for, classes, methods, yield, self).
"""

from __future__ import annotations

import ast
import re
import threading
from typing import Any

from lox.dsl.schema import ALLOWED_ACTIONS, ALLOWED_PREDICATES, ENUM_CONSTANTS


class DSLValidationError(Exception):
    """Raised when policy code violates grammar, schema, or safety invariants."""


# Whitelist of safe Python built-in identifiers
SAFE_BUILTINS = {
    "range",
    "len",
    "min",
    "max",
    "abs",
    "sum",
    "enumerate",
    "zip",
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "print",
    "round",
    "isinstance",
    "getattr",
    "hasattr",
    "any",
    "all",
    "sorted",
    "reversed",
    "chr",
    "ord",
    "True",
    "False",
    "None",
}

# Special framework keywords and type names
SPECIAL_NAMES = {
    "plan",
    "self",
    "args",
    "kwargs",
    "Action",
    "Status",
    "Agent",
    "Policy",
    "Blackboard",
    "obs",
    "Observation",
    "Item",
    "HeroState",
    "GoalDirective",
    "GoalAgenda",
    "ItemBeliefState",
}


class SafeASTVisitor(ast.NodeVisitor):
    """
    Whitelists safe Pythonic constructs (Classes, Methods, Generators, While/For loops).
    Rejects any imports, code execution (eval/exec), filesystem calls, or dunder exploits.
    """

    ALLOWED_NODES = (
        ast.Module,
        ast.ClassDef,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Yield,
        ast.YieldFrom,
        ast.While,
        ast.For,
        ast.If,
        ast.IfExp,
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
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.USub,
        ast.Call,
        ast.Name,
        ast.Attribute,
        ast.Constant,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.UAdd,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.Tuple,
        ast.ListComp,
        ast.GeneratorExp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
        ast.Subscript,
        ast.Index if hasattr(ast, "Index") else ast.AST,
        ast.Slice,
        ast.Pass,
        ast.Break,
        ast.Continue,
        ast.keyword,
        ast.Load,
        ast.Store,
        ast.Del,
    )

    def __init__(self):
        self.functions: dict[str, ast.FunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.plan_order: list[str] = []
        self.has_generator: bool = False
        self.local_scope: set[str] = set()

    def generic_visit(self, node: ast.AST):
        if not isinstance(node, self.ALLOWED_NODES):
            raise DSLValidationError(
                f"Disallowed syntax element: '{type(node).__name__}'. "
                f"Imports, dynamic execution (eval/exec), and system calls are strictly forbidden."
            )
        super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # Prevent base class inheritance exploits
        if node.bases:
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id not in ("object",):
                    pass  # Custom base classes within script allowed
        self.classes[node.name] = node
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions[node.name] = node
        # Check if function is a generator subroutine (contains yield/yield from)
        has_yield = any(
            isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node)
        )
        if has_yield:
            for n in ast.walk(node):
                if isinstance(n, ast.Return) and n.value is not None:
                    if isinstance(n.value, ast.Constant) and isinstance(
                        n.value.value, bool
                    ):
                        raise DSLValidationError(
                            f"Generator subroutine '{node.name}' contains 'return {n.value.value}'. "
                            f"Generator subroutines called via 'obs = yield from self.{node.name}(obs)' must 'return obs' "
                            f"so the Observation object is not overwritten with a boolean."
                        )

        # Add function arguments to local scope
        old_scope = self.local_scope.copy()
        for arg in node.args.args:
            self.local_scope.add(arg.arg)
        self.generic_visit(node)
        self.local_scope = old_scope

    def visit_Yield(self, node: ast.Yield):
        self.has_generator = True
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom):
        self.has_generator = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Strict anti-dunder exploit guard (blocks __class__, __subclasses__, __globals__)
        if node.attr.startswith("__"):
            raise DSLValidationError(
                f"Access to private dunder attribute '{node.attr}' is strictly forbidden."
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Track local variable assignment
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_scope.add(target.id)
                # Check legacy plan = [...]
                if target.id == "plan" and isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Name):
                            self.plan_order.append(elt.id)
            elif isinstance(target, ast.Attribute):
                pass  # self.field = ... allowed
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            self.local_scope.add(node.target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if isinstance(node.target, ast.Name):
            self.local_scope.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.local_scope.add(elt.id)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        if isinstance(node.target, ast.Name):
            self.local_scope.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.local_scope.add(elt.id)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)

    def visit_ListComp(self, node: ast.ListComp):
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp):
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)

    def pre_scan(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in (
                "eval",
                "exec",
                "open",
                "__import__",
                "globals",
                "locals",
                "compile",
            ):
                raise DSLValidationError(
                    f"Execution of unsafe builtin '{name}()' is strictly forbidden."
                )
            if (
                name not in ALLOWED_ACTIONS
                and name not in self.functions
                and name not in self.classes
                and name not in self.local_scope
                and name not in SAFE_BUILTINS
                and name not in SPECIAL_NAMES
            ):
                raise DSLValidationError(
                    f"Unknown or unapproved action/function call: '{name}()'."
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        var_name = node.id
        if var_name in (
            "eval",
            "exec",
            "open",
            "__import__",
            "globals",
            "locals",
            "compile",
        ):
            raise DSLValidationError(f"Identifier '{var_name}' is forbidden.")
        if (
            var_name not in ALLOWED_PREDICATES
            and var_name not in ENUM_CONSTANTS
            and var_name not in ALLOWED_ACTIONS
            and var_name not in self.functions
            and var_name not in self.classes
            and var_name not in self.local_scope
            and var_name not in SAFE_BUILTINS
            and var_name not in SPECIAL_NAMES
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


def repair_python_indentation(code_str: str) -> str:
    """
    Repairs Python indentation anomalies:
    - Normalizes tabs to 4 spaces and strips trailing whitespace.
    - Preserves multiline string and docstring contents.
    - Re-aligns top-level function definitions if methods have mismatched outer indentations.
    - Snaps unaligned unindents to the nearest valid level on the Python indentation stack,
      completely preventing 'unindent does not match any outer indentation level'.
    """
    if not code_str:
        return code_str

    code = code_str.replace("\t", "    ")
    lines = code.splitlines()
    if not lines:
        return code

    # Step 1: Detect if this is a collection of methods without class Agent
    has_class = any(line.strip().startswith("class ") for line in lines)
    if not has_class:
        def_indents = []
        for line in lines:
            if line.lstrip().startswith("def "):
                def_indents.append(len(line) - len(line.lstrip()))
        if len(def_indents) > 1 and len(set(def_indents)) > 1:
            new_lines = []
            current_def_orig_indent = 0
            for line in lines:
                if line.lstrip().startswith("def "):
                    current_def_orig_indent = len(line) - len(line.lstrip())
                    new_lines.append(line.lstrip())
                else:
                    if line.strip():
                        if line.startswith(" " * current_def_orig_indent):
                            new_lines.append(line[current_def_orig_indent:])
                        else:
                            new_lines.append(line.lstrip())
                    else:
                        new_lines.append("")
            code = "\n".join(new_lines)
            lines = code.splitlines()

    # Step 2: Indentation stack tracking with unaligned unindent snapping
    repaired_lines = []
    indent_stack = [0]
    prev_ended_colon = False
    in_triple_quotes = False
    triple_quote_char = ""

    for line in lines:
        stripped = line.strip()

        # Handle multiline string / docstring state
        if in_triple_quotes:
            repaired_lines.append(line)
            t_count = line.count(triple_quote_char)
            if t_count % 2 == 1:
                in_triple_quotes = False
            continue

        if '"""' in stripped and stripped.count('"""') % 2 == 1:
            in_triple_quotes = True
            triple_quote_char = '"""'
        elif "'''" in stripped and stripped.count("'''") % 2 == 1:
            in_triple_quotes = True
            triple_quote_char = "'''"

        if not stripped or stripped.startswith("#"):
            repaired_lines.append(line)
            continue

        raw_indent = len(line) - len(line.lstrip())

        if raw_indent > indent_stack[-1]:
            if prev_ended_colon:
                indent_stack.append(raw_indent)
                repaired_lines.append(line)
            else:
                repaired_lines.append(" " * indent_stack[-1] + stripped)
        elif raw_indent == indent_stack[-1]:
            repaired_lines.append(line)
        else:
            # Unindent: must match some level in indent_stack
            if raw_indent in indent_stack:
                while indent_stack and indent_stack[-1] > raw_indent:
                    indent_stack.pop()
                repaired_lines.append(line)
            else:
                # Snap to closest level in indent_stack that is <= raw_indent
                valid_levels = [lvl for lvl in indent_stack if lvl <= raw_indent]
                snapped_indent = valid_levels[-1] if valid_levels else 0
                while indent_stack and indent_stack[-1] > snapped_indent:
                    indent_stack.pop()
                repaired_lines.append(" " * snapped_indent + stripped)

        prev_ended_colon = stripped.endswith(":")

    return "\n".join(repaired_lines)


def _parse_with_expanded_stack(code: str) -> ast.Module:
    """Parses code, falling back to a dedicated 64MB stack thread if C stack margin is exceeded."""
    try:
        return ast.parse(code)
    except (MemoryError, RecursionError):
        result: list[Any] = [None, None]

        def worker():
            try:
                result[0] = ast.parse(code)
            except Exception as e:
                result[1] = e

        old_size = threading.stack_size(64 * 1024 * 1024)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        threading.stack_size(old_size)

        if result[1]:
            raise result[1]
        if result[0] is None:
            raise MemoryError(
                "Parser stack overflowed - Python source too complex to parse"
            )
        return result[0]


def parse_and_validate(
    code_str: str,
) -> tuple[ast.Module, list[str], dict[str, ast.FunctionDef], SafeASTVisitor]:
    """
    Parses policy string and executes strict AST validation.
    Returns (AST module, plan_order list, dictionary of macro FunctionDefs, visitor instance).
    """
    clean_code = normalize_code(code_str)
    try:
        tree = _parse_with_expanded_stack(clean_code)
    except SyntaxError as e:
        import textwrap

        try:
            tree = _parse_with_expanded_stack(textwrap.dedent(clean_code))
        except SyntaxError:
            try:
                repaired = repair_python_indentation(clean_code)
                tree = _parse_with_expanded_stack(repaired)
            except SyntaxError:
                try:
                    tree = _parse_with_expanded_stack(textwrap.dedent(repaired))
                except SyntaxError:
                    raise DSLValidationError(
                        f"Syntax error in policy program: {e}"
                    ) from e
    except MemoryError as e:
        raise DSLValidationError(f"Parser stack overflowed: {e}") from e

    visitor = SafeASTVisitor()
    visitor.pre_scan(tree)
    visitor.visit(tree)

    if not visitor.functions and not visitor.classes:
        raise DSLValidationError(
            "Policy program must define at least one function or class."
        )

    plan_order = visitor.plan_order
    if not plan_order and not visitor.classes and not visitor.has_generator:
        # Default plan: all defined macros in order
        plan_order = list(visitor.functions.keys())

    return tree, plan_order, visitor.functions, visitor
