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
        self.last_active_leaf: str = ""
        self._selector_commits: dict[int, tuple[int, int]] = {}

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
        self.last_active_leaf = ""
        self._selector_commits.clear()


class BehaviorNode:
    """Base class for all Behavior Tree nodes."""

    def __init__(self, name: str = "", min_commit: int = 0) -> None:
        self.name = name or self.__class__.__name__
        self.min_commit = max(0, int(min_commit))

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
        min_commit: int = 0,
    ) -> None:
        super().__init__(
            name=name or getattr(action_fn, "__name__", "ActionLeaf"),
            min_commit=min_commit,
        )
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
            if hasattr(blackboard, "last_active_leaf"):
                blackboard.last_active_leaf = self.name
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
    Supports native hysteresis commitment (`hysteresis` or child `min_commit`)
    so committed subtrees avoid per-step thrashing until FAILURE or expiry.
    Returns SUCCESS or RUNNING on the first child that does not return FAILURE.
    Returns FAILURE if all children return FAILURE.
    """

    def __init__(
        self,
        children: list[BehaviorNode] | None = None,
        name: str = "",
        hysteresis: int = 0,
        min_commit: int = 0,
    ) -> None:
        super().__init__(name=name, min_commit=min_commit)
        self.children: list[BehaviorNode] = children or []
        self.hysteresis = max(0, int(hysteresis))

    def add_child(self, child: BehaviorNode) -> PrioritySelector:
        self.children.append(child)
        return self

    def tick(self, state: Any, blackboard: Blackboard) -> tuple[NodeStatus, Any]:
        commits = getattr(blackboard, "_selector_commits", None)
        sel_id = id(self)
        if isinstance(commits, dict) and sel_id in commits:
            committed_idx, rem = commits[sel_id]
            if rem > 0 and 0 <= committed_idx < len(self.children):
                status, act = self.children[committed_idx].tick(state, blackboard)
                if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
                    if rem - 1 > 0:
                        commits[sel_id] = (committed_idx, rem - 1)
                    else:
                        commits.pop(sel_id, None)
                    return status, act
            commits.pop(sel_id, None)

        for idx, child in enumerate(self.children):
            status, act = child.tick(state, blackboard)
            if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
                commit_len = max(
                    int(getattr(child, "min_commit", 0) or 0),
                    self.hysteresis,
                )
                if commit_len > 1 and isinstance(commits, dict):
                    commits[sel_id] = (idx, commit_len - 1)
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
        self,
        children: list[BehaviorNode] | None = None,
        name: str = "",
        min_commit: int = 0,
    ) -> None:
        super().__init__(name=name, min_commit=min_commit)
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
        if hasattr(self.blackboard, "last_active_leaf"):
            self.blackboard.last_active_leaf = ""
        status, act = self.root.tick(state, self.blackboard)
        return act


def extract_runner_leaf_errors(runner_or_instance: Any) -> list[tuple[str, str]]:
    """Extracts any swallowed `(node_name, error_str)` entries from a PolicyRunner or Policy instance's Blackboard."""
    if runner_or_instance is None:
        return []
    inst = getattr(runner_or_instance, "instance", runner_or_instance)
    candidates = [
        getattr(inst, "blackboard", None),
        getattr(inst, "bb", None),
        getattr(getattr(inst, "tree", None), "blackboard", None),
        getattr(runner_or_instance, "blackboard", None),
    ]
    seen: list[tuple[str, str]] = []
    for bb in candidates:
        errs = getattr(bb, "leaf_errors", None)
        if isinstance(errs, list):
            for item in errs:
                if isinstance(item, tuple) and len(item) == 2:
                    pair = (str(item[0]), str(item[1]))
                    if pair not in seen:
                        seen.append(pair)
    return seen
