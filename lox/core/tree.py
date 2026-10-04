"""
LOX Behavior Tree Core: Ultra-Fast Hierarchical Policy Engine.
Composable, deterministic, microsecond execution tree.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from lox.core.types import Status, Action, Observation


@dataclass
class Blackboard:
    """Shared state between nodes during a single turn's execution."""
    obs: Observation
    selected_action: Action | None = None
    target_pos: tuple[int, int] | None = None
    path_cache: list[tuple[int, int]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)


class BehaviorNode(ABC):
    """Abstract base class for all nodes in the Behavior Tree."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def tick(self, bb: Blackboard) -> Status:
        pass


class Selector(BehaviorNode):
    """
    Fallback Priority Node (equivalent to HTN method prioritization).
    Executes children in order until one returns SUCCESS or RUNNING.
    Fails if and only if all children fail.
    """

    def __init__(self, children: list[BehaviorNode], name: str = "Selector"):
        super().__init__(name)
        self.children = children

    def tick(self, bb: Blackboard) -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


class Sequence(BehaviorNode):
    """
    Procedural Execution Node (equivalent to HTN compound task decomposition).
    Executes children in order until all succeed or one returns FAILURE or RUNNING.
    """

    def __init__(self, children: list[BehaviorNode], name: str = "Sequence"):
        super().__init__(name)
        self.children = children

    def tick(self, bb: Blackboard) -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


class Condition(BehaviorNode):
    """
    Leaf Node: Evaluates a boolean predicate against the blackboard/observation.
    Returns SUCCESS if true, FAILURE if false.
    """

    def __init__(self, predicate_fn: Callable[[Blackboard], bool], name: str = "Condition"):
        super().__init__(name)
        self.predicate_fn = predicate_fn

    def tick(self, bb: Blackboard) -> Status:
        return Status.SUCCESS if self.predicate_fn(bb) else Status.FAILURE


class ActionNode(BehaviorNode):
    """
    Leaf Node: Executes an action function that produces a concrete Action.
    Returns SUCCESS (action selected), FAILURE (cannot perform action), or RUNNING.
    """

    def __init__(self, action_fn: Callable[[Blackboard], Status | Action | None], name: str = "Action"):
        super().__init__(name)
        self.action_fn = action_fn

    def tick(self, bb: Blackboard) -> Status:
        res = self.action_fn(bb)
        if isinstance(res, Action):
            bb.selected_action = res
            return Status.SUCCESS
        elif isinstance(res, Status):
            return res
        elif res is None:
            return Status.FAILURE
        return Status.SUCCESS


class Inverter(BehaviorNode):
    """Decorator: Inverts SUCCESS <-> FAILURE."""

    def __init__(self, child: BehaviorNode, name: str = "Inverter"):
        super().__init__(name)
        self.child = child

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        if status == Status.SUCCESS:
            return Status.FAILURE
        elif status == Status.FAILURE:
            return Status.SUCCESS
        return status


class BehaviorTree:
    """The root container holding the behavior hierarchy."""

    def __init__(self, root: BehaviorNode, name: str = "Root"):
        self.root = root
        self.name = name

    def execute(self, obs: Observation, memory: dict[str, Any] | None = None) -> Action | None:
        """Evaluates the behavior tree for the current observation and returns the chosen Action."""
        bb = Blackboard(obs=obs, memory=memory or {})
        self.root.tick(bb)
        return bb.selected_action

    def create_runner(self, initial_obs: Observation | None = None) -> Any:
        """Provides a runner interface matching PolicyRunner for generator compatibility."""
        class BTRunner:
            def __init__(self, tree):
                self.tree = tree
            def send(self, obs: Observation | None = None) -> Action:
                act = self.tree.execute(obs)
                return act if act is not None else Action(name="wait")
        return BTRunner(self)
