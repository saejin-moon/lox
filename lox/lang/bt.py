"""
Reactive Behavior Tree Engine for NetHack Policies.
Provides modular, decoupled decision execution:
- NodeStatus: SUCCESS, FAILURE, RUNNING
- Blackboard: shared state dictionary across nodes
- PrioritySelector (Fallback): ticks children until one succeeds/runs
- Sequence: ticks children until one fails
- ActionLeaf: executes leaf action routines
- ConditionLeaf: evaluates tactical preconditions
- BehaviorTree: root coordinator that executes tree ticks
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class NodeStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Blackboard(dict[str, Any]):
    """Shared blackboard data store for inter-node communication and state persistence."""

    def __init__(self, *args: Any, strict_errors: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strict_errors = strict_errors
        self.leaf_errors: list[tuple[str, str]] = []

    def record_leaf_error(self, node_name: str, exc: Exception) -> None:
        if len(self.leaf_errors) < 50:
            self.leaf_errors.append((str(node_name), f"{type(exc).__name__}: {exc}"))

    def get_var(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)

    def set_var(self, key: str, value: Any) -> None:
        self[key] = value

    def has_var(self, key: str) -> bool:
        return key in self

    def clear_vars(self) -> None:
        self.clear()


class BehaviorNode:
    """Base class for all Behavior Tree nodes."""

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        """
        Ticks the node with current game state and blackboard.
        Returns `(status, action_or_none)`.
        """
        raise NotImplementedError


class ActionLeaf(BehaviorNode):
    """
    Leaf node that executes an action function.
    Function signature: `fn(state, blackboard)` or `fn(state)` -> Action | None.
    If action is produced, status is SUCCESS (or RUNNING if explicitly returned);
    if None is returned, status is FAILURE.
    """

    def __init__(
        self,
        action_fn: Callable[..., Any],
        name: str = "",
        is_running: bool = False,
    ) -> None:
        super().__init__(name=name or getattr(action_fn, "__name__", "ActionLeaf"))
        self.action_fn = action_fn
        self.is_running = is_running

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        try:
            # Inspect parameter count
            code = getattr(self.action_fn, "__code__", None)
            arg_count = code.co_argcount if code else 1
            if arg_count >= 2:
                act = self.action_fn(state, blackboard)
            else:
                act = self.action_fn(state)
        except Exception as exc:
            if hasattr(blackboard, "record_leaf_error"):
                blackboard.record_leaf_error(self.name, exc)
            if getattr(blackboard, "strict_errors", False):
                raise
            return NodeStatus.FAILURE, None

        if act is not None:
            status = NodeStatus.RUNNING if self.is_running else NodeStatus.SUCCESS
            return status, act
        return NodeStatus.FAILURE, None


class ConditionLeaf(BehaviorNode):
    """
    Leaf node that evaluates a boolean precondition.
    Function signature: `cond_fn(state, blackboard)` or `cond_fn(state)` -> bool.
    Returns SUCCESS if True, FAILURE if False, action is always None.
    """

    def __init__(
        self,
        cond_fn: Callable[..., bool],
        name: str = "",
    ) -> None:
        super().__init__(name=name or getattr(cond_fn, "__name__", "ConditionLeaf"))
        self.cond_fn = cond_fn

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        try:
            code = getattr(self.cond_fn, "__code__", None)
            arg_count = code.co_argcount if code else 1
            if arg_count >= 2:
                ok = bool(self.cond_fn(state, blackboard))
            else:
                ok = bool(self.cond_fn(state))
        except Exception as exc:
            if hasattr(blackboard, "record_leaf_error"):
                blackboard.record_leaf_error(self.name, exc)
            if getattr(blackboard, "strict_errors", False):
                raise
            ok = False

        status = NodeStatus.SUCCESS if ok else NodeStatus.FAILURE
        return status, None


class PrioritySelector(BehaviorNode):
    """
    Composite Selector / Fallback node.
    Ticks children sequentially from left to right.
    Returns SUCCESS or RUNNING on the first child that does not return FAILURE.
    Returns FAILURE if all children return FAILURE.
    """

    def __init__(
        self, children: list[BehaviorNode] | None = None, name: str = ""
    ) -> None:
        super().__init__(name=name)
        self.children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> PrioritySelector:
        self.children.append(child)
        return self

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        for child in self.children:
            status, act = child.tick(state, blackboard)
            if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
                return status, act
        return NodeStatus.FAILURE, None


class Sequence(BehaviorNode):
    """
    Composite Sequence node.
    Ticks children sequentially from left to right.
    If any child returns FAILURE, immediately returns FAILURE (short-circuit).
    If a child returns RUNNING, returns RUNNING.
    Returns SUCCESS if all children return SUCCESS.
    """

    def __init__(
        self, children: list[BehaviorNode] | None = None, name: str = ""
    ) -> None:
        super().__init__(name=name)
        self.children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> Sequence:
        self.children.append(child)
        return self

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        last_action = None
        for child in self.children:
            status, act = child.tick(state, blackboard)
            if act is not None:
                last_action = act
            if status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE, None
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING, act
        return NodeStatus.SUCCESS, last_action


class Inverter(BehaviorNode):
    """Decorator node that inverts SUCCESS and FAILURE."""

    def __init__(self, child: BehaviorNode, name: str = "") -> None:
        super().__init__(name=name or f"Invert_{child.name}")
        self.child = child

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        status, act = self.child.tick(state, blackboard)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE, None
        if status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS, act
        return status, act


class BehaviorTree:
    """
    Root coordinator for executing a Reactive Behavior Tree.
    Manages the Blackboard and ticks the root node on each game turn.
    """

    def __init__(
        self,
        root: BehaviorNode,
        name: str = "RootTree",
        blackboard: Blackboard | None = None,
    ) -> None:
        self.root = root
        self.name = name
        self.blackboard = blackboard if blackboard is not None else Blackboard()

    def tick(self, state: Any) -> Any:
        """Ticks the tree and returns the resulting action (or None if all failed)."""
        status, act = self.root.tick(state, self.blackboard)
        return act
