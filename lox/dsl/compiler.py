"""
LOX 2.0 DSL Compiler: Compiles AST into an Executable Behavior Tree.
Transforms parsed Pythonic Infix AST into a high-performance DAG of BehaviorNodes.
Inner evaluation executes in <20 µs per turn.
"""
from __future__ import annotations

import ast
from typing import Any, Callable

from lox.core.types import Status, Action, Observation, HeroState, HungerState
from lox.core.tree import (
    BehaviorTree,
    BehaviorNode,
    Selector,
    Sequence,
    Condition,
    ActionNode,
    Blackboard,
)
from lox.dsl.parser import parse_and_validate
from lox.dsl.schema import ENUM_CONSTANTS


# Map AST comparison operators to python functions
_CMP_OPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _extract_val(node: ast.AST, bb: Blackboard) -> Any:
    """Extracts value of a constant, hero attribute, or blackboard predicate."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        name = node.id
        if name in ENUM_CONSTANTS:
            return ENUM_CONSTANTS[name]
        # Check hero attributes
        hero = bb.obs.hero
        if hasattr(hero, name):
            val = getattr(hero, name)
            if isinstance(val, HungerState):
                return val.value
            return val
        # Check blackboard memory / flags
        return bb.memory.get(name, False)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_extract_val(node.operand, bb)
    return False


def _compile_condition_node(node: ast.AST) -> Callable[[Blackboard], bool]:
    """Recursively compiles an AST expression into a fast boolean evaluator."""
    if isinstance(node, ast.Name):
        return lambda bb: bool(_extract_val(node, bb))

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand_fn = _compile_condition_node(node.operand)
        return lambda bb: not operand_fn(bb)

    elif isinstance(node, ast.BoolOp):
        fns = [_compile_condition_node(val) for val in node.values]
        if isinstance(node.op, ast.And):
            return lambda bb: all(fn(bb) for fn in fns)
        elif isinstance(node.op, ast.Or):
            return lambda bb: any(fn(bb) for fn in fns)

    elif isinstance(node, ast.Compare):
        left_node = node.left
        # Only single comparison supported: left op comparator
        op_type = type(node.ops[0])
        op_fn = _CMP_OPS.get(op_type, lambda a, b: False)
        right_node = node.comparators[0]
        return lambda bb: op_fn(_extract_val(left_node, bb), _extract_val(right_node, bb))

    return lambda bb: False


def _compile_action_call(
    call_node: ast.Call,
    action_handlers: dict[str, Callable[[Blackboard, dict[str, Any]], Status | Action | None]],
) -> ActionNode:
    """Compiles a function call into an ActionNode."""
    action_name = call_node.func.id  # type: ignore
    kwargs = {}
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Constant):
            kwargs[kw.arg] = kw.value.value

    handler = action_handlers.get(
        action_name,
        lambda bb, args: Action(name=action_name, extra=args),
    )

    def exec_action(bb: Blackboard) -> Status | Action | None:
        return handler(bb, kwargs)

    return ActionNode(exec_action, name=action_name)


def _compile_statements(
    stmts: list[ast.stmt],
    action_handlers: dict[str, Callable],
) -> list[BehaviorNode]:
    """Compiles a list of statements into BehaviorTree nodes."""
    nodes: list[BehaviorNode] = []

    for stmt in stmts:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            nodes.append(_compile_action_call(stmt.value, action_handlers))

        elif isinstance(stmt, ast.If):
            cond_fn = _compile_condition_node(stmt.test)
            body_nodes = _compile_statements(stmt.body, action_handlers)
            if len(body_nodes) == 1:
                seq = Sequence([Condition(cond_fn), body_nodes[0]])
            else:
                seq = Sequence([Condition(cond_fn)] + body_nodes)

            if stmt.orelse:
                else_nodes = _compile_statements(stmt.orelse, action_handlers)
                # Selector between if-branch and else-branch
                nodes.append(Selector([seq] + else_nodes))
            else:
                nodes.append(seq)

        elif isinstance(stmt, ast.Pass):
            nodes.append(ActionNode(lambda bb: Status.SUCCESS, name="pass"))

    return nodes


def compile_policy(
    policy_code: str,
    action_handlers: dict[str, Callable] | None = None,
    tree_name: str = "LOXPolicyTree",
) -> BehaviorTree:
    """
    Parses, validates, and compiles policy code into an executable BehaviorTree.
    """
    _, plan_order, functions = parse_and_validate(policy_code)
    handlers = action_handlers or {}

    macro_nodes: list[BehaviorNode] = []
    for macro_name in plan_order:
        fn_def = functions[macro_name]
        body_nodes = _compile_statements(fn_def.body, handlers)
        if len(body_nodes) == 1:
            macro_nodes.append(body_nodes[0])
        else:
            macro_nodes.append(Sequence(body_nodes, name=macro_name))

    # Root priority selector
    root = Selector(macro_nodes, name=f"{tree_name}_Root")
    return BehaviorTree(root=root, name=tree_name)
