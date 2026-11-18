"""
Frozen Domain-Agnostic Runtime Interface Contracts (`lox.lang.contracts`).
Defines typed structural contracts for evolved world-trackers, capability scorers,
threat evaluators, and explicit preemption safety reports.
("Hard Code Nothing": zero NetHack item/monster heuristics or strategy constants.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GoalIntent:
    """Structured intent emitted by an evolved module before action emission."""

    goal_id: str
    priority: float
    target_pos: tuple[int, int] | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyReport:
    """Explicit pre-action safety assessment returned by threat/safety evaluators."""

    safe_to_continue: bool
    risk_score: float = 0.0
    abort_reason: str = ""
    recommended_action: Any = None


@runtime_checkable
class WorldTracker(Protocol):
    """Contract for modules that maintain cross-turn/cross-floor spatial state."""

    def update(self, state: Any, blackboard: Any = None) -> None: ...

    def query_goal(self, state: Any, blackboard: Any = None) -> GoalIntent | None: ...


@runtime_checkable
class CapabilityScorer(Protocol):
    """Contract for modules that summarize readiness/capability on [0, +inf)."""

    def score_capability(self, state: Any, blackboard: Any = None) -> float: ...


@runtime_checkable
class ThreatEvaluator(Protocol):
    """Contract for modules that evaluate immediate risk and preemption triggers."""

    def evaluate_safety(self, state: Any, blackboard: Any = None) -> SafetyReport: ...


def validate_contract_instance(obj: Any) -> list[str]:
    """Verifies runtime protocol conformance if `obj` advertises contract methods."""
    issues: list[str] = []
    if obj is None:
        return issues
    if hasattr(obj, "evaluate_safety") and not callable(obj.evaluate_safety):
        issues.append("ThreatEvaluator.evaluate_safety must be callable")
    if hasattr(obj, "score_capability") and not callable(obj.score_capability):
        issues.append("CapabilityScorer.score_capability must be callable")
    if hasattr(obj, "query_goal") and not callable(obj.query_goal):
        issues.append("WorldTracker.query_goal must be callable")
    if hasattr(obj, "update") and not callable(obj.update):
        issues.append("WorldTracker.update must be callable")
    return issues


__all__ = [
    "CapabilityScorer",
    "GoalIntent",
    "SafetyReport",
    "ThreatEvaluator",
    "WorldTracker",
    "validate_contract_instance",
]
