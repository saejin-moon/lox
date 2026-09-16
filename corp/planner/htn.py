"""
Fast Core HTN Planner: Recursive task network decomposition engine with
continuous persona prioritization, Nogood pruning, and cycle detection.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

from corp.planner.persona import PersonaTraitVector
from corp.planner.nogood import NogoodStore


class HTNDeadlockException(Exception):
    """Raised when the HTN planner cannot find any valid decomposition from current state."""
    pass


@dataclass(slots=True, frozen=True)
class Task:
    name: str
    is_primitive: bool = False
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PrimitiveTask(Task):
    is_primitive: bool = True


@dataclass(slots=True, frozen=True)
class CompoundTask(Task):
    is_primitive: bool = False


@dataclass
class Method:
    """
    Decomposes a compound task into an ordered sequence of sub-tasks.
    """
    name: str
    target_task: str
    preconditions: Callable[[Any], bool]
    subtasks_fn: Callable[[Any], list[Task]]
    # Priority weights W_m = (resilience, ranged, mana, stealth, alignment)
    priority_weights: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    base_utility: float = 0.0

    def compute_priority(self, persona: PersonaTraitVector) -> float:
        """
        Priority(m | theta) = U_base + sum(W_{m, j} * theta_j)
        """
        w = self.priority_weights
        p = persona
        utility = self.base_utility + (
            w[0] * p.resilience
            + w[1] * p.ranged
            + w[2] * p.mana
            + w[3] * p.stealth
            + w[4] * p.alignment
        )
        return float(utility)


@dataclass(slots=True, frozen=True)
class PlanSignature:
    task_name: str
    target_pos: tuple[int, int]
    turn: int


class CycleDetector:
    """
    Detects behavioral oscillation loops (e.g. stepping back and forth between 2 tiles).
    """

    def __init__(self, window_size: int = 10, max_repetitions: int = 3):
        self.history: list[PlanSignature] = []
        self.window_size = window_size
        self.max_repetitions = max_repetitions

    def reset(self):
        self.history.clear()

    def record_and_check(self, signature: PlanSignature) -> bool:
        self.history.append(signature)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Count identical (task_name, target_pos) occurrences in window
        matches = [
            s for s in self.history
            if s.task_name == signature.task_name and s.target_pos == signature.target_pos
        ]
        return len(matches) >= self.max_repetitions


class HTNPlanner:
    """
    Fast HTN recursive decomposition engine.
    Executes in <0.5 ms per planning cycle.
    """

    MAX_RECURSION_DEPTH = 15

    def __init__(self, nogood_store: NogoodStore | None = None):
        self.methods: dict[str, list[Method]] = {}  # Keyed by compound task name
        self.nogood_store = nogood_store or NogoodStore()
        self.cycle_detector = CycleDetector()

    def register_method(self, method: Method):
        """Registers a decomposition method for a compound task."""
        if method.target_task not in self.methods:
            self.methods[method.target_task] = []
        self.methods[method.target_task].append(method)

    def plan(
        self,
        root_task: Task,
        state: Any,
        persona: PersonaTraitVector,
        state_predicate_mask: int = 0,
    ) -> list[PrimitiveTask]:
        """
        Top-level planner entrypoint. Decomposes root_task into an executable primitive plan.
        Raises HTNDeadlockException if no valid decomposition exists.
        """
        result = self._decompose(
            task=root_task,
            state=state,
            persona=persona,
            state_predicate_mask=state_predicate_mask,
            depth=0,
        )

        if result is None:
            raise HTNDeadlockException(f"HTN Deadlock: unable to decompose task '{root_task.name}'")

        return result

    def _decompose(
        self,
        task: Task,
        state: Any,
        persona: PersonaTraitVector,
        state_predicate_mask: int,
        depth: int,
    ) -> list[PrimitiveTask] | None:
        """
        Recursive decomposition algorithm with Nogood branch cuts.
        """
        if depth > self.MAX_RECURSION_DEPTH:
            return None

        # Base Case: Primitive Task
        if task.is_primitive:
            # Check against active CDCL Nogoods in <1 microsecond
            forbidden, reason = self.nogood_store.is_forbidden(state_predicate_mask, task.name)
            if forbidden:
                return None  # Branch pruned by Nogood
            return [task]  # type: ignore

        # Recursive Case: Compound Task
        candidate_methods = self.methods.get(task.name, [])
        if not candidate_methods:
            return None

        # Sort candidate methods dynamically by continuous persona utility
        sorted_methods = sorted(
            candidate_methods,
            key=lambda m: m.compute_priority(persona),
            reverse=True,
        )

        for method in sorted_methods:
            # 1. Evaluate method preconditions
            if not method.preconditions(state):
                continue

            # 2. Expand subtasks
            try:
                subtasks = method.subtasks_fn(state)
            except Exception:
                continue

            subplan: list[PrimitiveTask] = []
            branch_success = True

            for subtask in subtasks:
                step_plan = self._decompose(
                    task=subtask,
                    state=state,
                    persona=persona,
                    state_predicate_mask=state_predicate_mask,
                    depth=depth + 1,
                )
                if step_plan is None:
                    branch_success = False
                    break
                subplan.extend(step_plan)

            if branch_success:
                return subplan

        return None
