"""
Fast Core HTN Planner: Recursive task network decomposition engine with
continuous persona prioritization, Nogood pruning, and cycle detection.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

from lox.planner.persona import PersonaTraitVector
from lox.planner.nogood import NogoodStore


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
    Detects behavioral oscillation loops (e.g. stepping back and forth between 2 tiles
    or stalling on the same tile).
    """

    def __init__(self, window_size: int = 8, max_repetitions: int = 3):
        self.pos_history: list[tuple[int, int]] = []
        self.window_size = window_size
        self.max_repetitions = max_repetitions

    def reset(self):
        self.pos_history.clear()

    def record_and_check(self, pos_or_sig: Any, task_name: str = "") -> bool:
        """
        Records position and detects 2-cycle or 3-cycle oscillation or stall.
        Returns True once if cycle detected, then resets to prevent lock-in.
        """
        if isinstance(pos_or_sig, PlanSignature):
            pos = pos_or_sig.target_pos
            task_name = pos_or_sig.task_name
        else:
            pos = pos_or_sig

        if task_name in ("SEARCH", "WAIT", "PRAY", "EAT"):
            return False

        self.pos_history.append(pos)
        if len(self.pos_history) > self.window_size:
            self.pos_history.pop(0)

        h = self.pos_history
        if len(h) >= 5:
            # Check 2-cycle: A, B, A, B, A
            if h[-1] == h[-3] == h[-5] and h[-2] == h[-4] and h[-1] != h[-2]:
                self.reset()
                return True

        if len(h) >= 6:
            # Check 3-cycle: A, B, C, A, B, C
            if h[-1] == h[-4] and h[-2] == h[-5] and h[-3] == h[-6] and len(set(h[-3:])) == 3:
                self.reset()
                return True

        # Check stall: same tile 4 times attempting movement
        if len(h) >= 4 and all(p == pos for p in h[-4:]):
            self.reset()
            return True

        return False


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
        self._sorted_methods_cache: dict[tuple[int, str], list[Method]] = {}

    def clear_cache(self):
        """Clears the persona method sort cache."""
        self._sorted_methods_cache.clear()

    def register_method(self, method: Method):
        """Registers a decomposition method for a compound task."""
        if method.target_task not in self.methods:
            self.methods[method.target_task] = []
        self.methods[method.target_task].append(method)
        self.clear_cache()

    def _get_sorted_methods(self, task_name: str, persona: PersonaTraitVector) -> list[Method]:
        """Returns candidate methods sorted by persona utility, with memoized caching."""
        candidates = self.methods.get(task_name, [])
        if len(candidates) <= 1:
            return candidates
        cache_key = (id(persona), task_name)
        cached = self._sorted_methods_cache.get(cache_key)
        if cached is not None:
            return cached
        sorted_methods = sorted(
            candidates,
            key=lambda m: m.compute_priority(persona),
            reverse=True,
        )
        self._sorted_methods_cache[cache_key] = sorted_methods
        return sorted_methods

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
        sorted_methods = self._get_sorted_methods(task.name, persona)
        if not sorted_methods:
            return None

        for method in sorted_methods:
            # 1. Evaluate method preconditions
            if not method.preconditions(state):
                continue

            # 2. Expand subtasks
            try:
                subtasks = method.subtasks_fn(state)
            except Exception:
                continue

            if not subtasks:
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
