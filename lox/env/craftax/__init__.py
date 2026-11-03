"""
Craftax Benchmark Plugin for LOX ("Hard Code Nothing").
"""

from __future__ import annotations

from lox.env.craftax.actions import ACTION_DELTAS, CraftaxAction
from lox.env.craftax.adapter import CraftaxAdapter
from lox.env.craftax.harness import CraftaxHarness
from lox.env.craftax.state import CraftaxGameState

__all__ = [
    "ACTION_DELTAS",
    "CraftaxAction",
    "CraftaxAdapter",
    "CraftaxGameState",
    "CraftaxHarness",
]
