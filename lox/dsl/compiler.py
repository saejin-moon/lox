"""
LOX DSL Compiler: Compiles AST into an Executable Policy.
Supports:
1. Object-Oriented Classes (`class Agent:` with generator methods)
2. Functional Generators (`def episode_policy(obs): ... yield action`)
3. High-Performance microsecond Behavior Trees (`plan = [...]`)
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import Any

from lox.core.agenda import GoalAgenda, GoalDirective
from lox.core.tree import (
    ActionNode,
    BehaviorNode,
    BehaviorTree,
    Blackboard,
    Condition,
    Selector,
    Sequence,
)
from lox.core.types import Action, HeroState, HungerState, Observation, Status
from lox.dsl.parser import normalize_code, parse_and_validate
from lox.dsl.schema import ALLOWED_ACTIONS, ENUM_CONSTANTS

# Map AST comparison operators to python functions
_CMP_OPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _create_action_builder(name: str) -> Callable[..., Action]:
    """Creates an Action factory for policy scripts."""

    def action_fn(*args, **kwargs) -> Action:
        direction = kwargs.get("direction")
        slot = kwargs.get("slot")
        target_pos = kwargs.get("target_pos")
        subroutine = kwargs.get("subroutine", "")
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            target_pos = (args[0], args[1])
        elif args and isinstance(args[0], tuple):
            if len(args[0]) == 2:
                target_pos = args[0]
                direction = args[0]
        elif args and isinstance(args[0], str):
            slot = args[0]
        return Action(
            name=name,
            direction=direction,
            slot=slot,
            target_pos=target_pos,
            subroutine=subroutine,
            extra=kwargs,
        )

    return action_fn


# Default action constructors available inside policy scripts
DEFAULT_ACTION_BUILDERS = {
    act_name: _create_action_builder(act_name) for act_name in ALLOWED_ACTIONS
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
        # Check epistemic view
        epistemic = getattr(bb.obs, "epistemic", None)
        if epistemic and hasattr(epistemic, name):
            return getattr(epistemic, name)
        # Check agenda view
        agenda = getattr(bb.obs, "agenda", None)
        if agenda and hasattr(agenda, name):
            return getattr(agenda, name)
        # Check combat, spatial, dungeon, status, inventory
        for sub_ns in ("combat", "spatial", "dungeon", "status", "inventory"):
            sub_obj = getattr(bb.obs, sub_ns, None)
            if sub_obj and hasattr(sub_obj, name):
                return getattr(sub_obj, name)
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
        child_fns = [_compile_condition_node(val) for val in node.values]
        if isinstance(node.op, ast.And):
            return lambda bb: all(fn(bb) for fn in child_fns)
        elif isinstance(node.op, ast.Or):
            return lambda bb: any(fn(bb) for fn in child_fns)

    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        op_type = type(node.ops[0])
        op_fn = _CMP_OPS.get(op_type)
        if op_fn is None:
            return lambda bb: False
        left_node = node.left
        right_node = node.comparators[0]
        return lambda bb: op_fn(
            _extract_val(left_node, bb), _extract_val(right_node, bb)
        )

    return lambda bb: False


def _compile_action_call(
    call_node: ast.Call,
    action_handlers: dict[
        str, Callable[[Blackboard, dict[str, Any]], Status | Action | None]
    ],
) -> ActionNode:
    """Compiles a function call into an ActionNode."""
    action_name = call_node.func.id if isinstance(call_node.func, ast.Name) else "wait"
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
                nodes.append(Selector([seq] + else_nodes))
            else:
                nodes.append(seq)

        elif isinstance(stmt, ast.Pass):
            nodes.append(ActionNode(lambda bb: Status.SUCCESS, name="pass"))

    return nodes


class ExecutionGuard:
    """Watchdog that bounds consecutive generator loop iterations without yielding."""

    def __init__(self, limit: int = 2000):
        self.count = 0
        self.limit = limit

    def tick(self) -> None:
        self.count += 1
        if self.count > self.limit:
            raise RuntimeError(
                f"Infinite loop detected: Policy looped {self.limit} times without yielding an action!"
            )

    def reset(self) -> None:
        self.count = 0


class LoopGuardTransformer(ast.NodeTransformer):
    """AST Transformer that instruments while and for loops with _guard.tick()."""

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        tick_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_guard", ctx=ast.Load()),
                    attr="tick",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )
        node.body.insert(0, tick_stmt)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        tick_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_guard", ctx=ast.Load()),
                    attr="tick",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )
        node.body.insert(0, tick_stmt)
        return node


class PolicyRunner:
    """Wrapper that smoothly drives Python generators, classes, or trees with .send(obs) or next()."""

    def __init__(self, gen_or_callable: Any, guard: ExecutionGuard | None = None):
        self.gen = gen_or_callable
        self.is_generator = inspect.isgenerator(gen_or_callable)
        self.started = False
        self.guard = guard

    def send(self, obs: Observation | None = None) -> Action:
        if self.guard is not None:
            self.guard.reset()

        if not self.is_generator:
            if callable(self.gen):
                res = self.gen(obs)
                return res if isinstance(res, Action) else Action(name="wait")
            return Action(name="wait")

        try:
            if not self.started:
                self.started = True
                act = next(self.gen)
            else:
                act = self.gen.send(obs)
            return act if isinstance(act, Action) else Action(name="wait")
        except RuntimeError as exc:
            if "Infinite loop detected" in str(exc):
                # Fallback to wait rather than hanging or crashing the game process
                return Action(name="wait")
            raise
        except StopIteration:
            return Action(name="wait")

    def __iter__(self):
        return self

    def __next__(self) -> Action:
        return self.send(None)


class PolicyExecutor:
    """Unified wrapper around classes, generators, and behavior trees."""

    def __init__(
        self,
        target_callable: Any,
        is_class: bool = False,
        is_tree: bool = False,
        guard: ExecutionGuard | None = None,
    ):
        self.target_callable = target_callable
        self.is_class = is_class
        self.is_tree = is_tree
        self.guard = guard
        self._default_runner: PolicyRunner | None = None

    def create_runner(self, initial_obs: Observation | None = None) -> PolicyRunner:
        """Instantiates a fresh agent / generator for an episode."""
        if self.is_tree:
            tree = self.target_callable
            bb = Blackboard(
                initial_obs or Observation(chars=None, glyphs=None, hero=HeroState())
            )

            def tree_generator(obs):
                while True:
                    bb.obs = obs
                    tree.tick(bb)
                    act = bb.last_action or Action(name="wait")
                    obs = yield act

            gen = tree_generator(initial_obs)
            return PolicyRunner(gen, guard=self.guard)

        elif self.is_class:
            agent = self.target_callable()
            try:
                if hasattr(agent, "run"):
                    gen = agent.run(initial_obs)
                elif hasattr(agent, "episode_policy"):
                    gen = agent.episode_policy(initial_obs)
                else:
                    gen = agent(initial_obs)
            except RuntimeError as exc:
                if "Infinite loop detected" in str(exc):

                    def fallback_gen(obs):
                        while True:
                            yield Action(name="wait")

                    return PolicyRunner(fallback_gen(initial_obs), guard=self.guard)
                raise
            return PolicyRunner(gen, guard=self.guard)

        else:
            try:
                gen = self.target_callable(initial_obs)
            except RuntimeError as exc:
                if "Infinite loop detected" in str(exc):

                    def fallback_gen(obs):
                        while True:
                            yield Action(name="wait")

                    return PolicyRunner(fallback_gen(initial_obs), guard=self.guard)
                raise
            return PolicyRunner(gen, guard=self.guard)

    def execute(self, obs: Observation, memory: dict[str, Any] | None = None) -> Action:
        """Ticking interface for single-step execution."""
        if self._default_runner is None:
            self._default_runner = self.create_runner(obs)
        return self._default_runner.send(obs)


def compile_policy(
    policy_code: str,
    action_handlers: dict[str, Callable] | None = None,
    tree_name: str = "LOXPolicyTree",
) -> Any:
    """
    Parses, validates, and compiles policy code.
    Seamlessly handles Class-based agents, Generator functions, and legacy Behavior Trees.
    """
    tree_ast, plan_order, functions, visitor = parse_and_validate(policy_code)
    normalize_code(policy_code)

    # 1. If policy defines classes or generators, compile into sandbox namespace
    if visitor.classes or visitor.has_generator:
        guarded_ast = LoopGuardTransformer().visit(tree_ast)
        ast.fix_missing_locations(guarded_ast)

        guard = ExecutionGuard(limit=2000)
        sandbox: dict[str, Any] = {
            "Action": Action,
            "GoalDirective": GoalDirective,
            "GoalAgenda": GoalAgenda,
            "_guard": guard,
            **ENUM_CONSTANTS,
            **DEFAULT_ACTION_BUILDERS,
        }

        compiled_code = compile(guarded_ast, "<policy>", "exec")
        exec(compiled_code, sandbox)

        # Look for primary agent class (e.g. Agent, Policy, or first defined class)
        target_cls = None
        for preferred in ("Agent", "Policy"):
            if preferred in sandbox and isinstance(sandbox[preferred], type):
                target_cls = sandbox[preferred]
                break
        if target_cls is None:
            for val in sandbox.values():
                if isinstance(val, type) and val.__module__ == "<policy>":
                    target_cls = val
                    break

        if target_cls is not None:
            return PolicyExecutor(target_cls, is_class=True, guard=guard)

        # Look for generator function
        target_fn = None
        for preferred in ("episode_policy", "run"):
            if preferred in sandbox and inspect.isgeneratorfunction(sandbox[preferred]):
                target_fn = sandbox[preferred]
                break
        if target_fn is None:
            for val in sandbox.values():
                if inspect.isgeneratorfunction(val):
                    target_fn = val
                    break

        if target_fn is not None:
            return PolicyExecutor(target_fn, is_class=False, guard=guard)

    # 2. Legacy Behavior Tree compilation
    handlers = action_handlers or {}
    macro_nodes: list[BehaviorNode] = []
    for macro_name in plan_order:
        fn_def = functions.get(macro_name)
        if fn_def is not None:
            body_nodes = _compile_statements(fn_def.body, handlers)
            if len(body_nodes) == 1:
                macro_nodes.append(body_nodes[0])
            else:
                macro_nodes.append(Sequence(body_nodes, name=macro_name))

    root = Selector(macro_nodes, name=f"{tree_name}_Root")
    btree = BehaviorTree(root=root, name=tree_name)
    return btree
