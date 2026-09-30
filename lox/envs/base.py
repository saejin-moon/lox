"""
LOX 2.0 Environment Layer: Standardized Environment Adapter Interface.
Decouples agent cognition and spatial reasoning from underlying Gym/NLE/JAX engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lox.core.types import Observation, Action


class EnvironmentAdapter(ABC):
    """Abstract interface providing uniform observations and action dispatch."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Resets the environment and returns the initial Observation."""
        pass

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Executes a high-level or primitive Action.
        Returns (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleans up environment handles and resources."""
        pass
