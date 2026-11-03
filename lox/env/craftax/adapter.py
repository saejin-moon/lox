"""
Craftax Benchmark Adapter ("Hard Code Nothing").
Implements the BenchmarkAdapter protocol for Craftax.
"""

from __future__ import annotations

from typing import Any

from lox.env.base import BenchmarkAdapter, EnvHarness
from lox.env.craftax.harness import CraftaxHarness


class CraftaxAdapter(BenchmarkAdapter):
    """Benchmark adapter for Craftax."""

    @property
    def name(self) -> str:
        return "craftax"

    def create_harness(self, max_steps: int = 1000, **kwargs: Any) -> EnvHarness:
        return CraftaxHarness(max_steps=max_steps)

    def progression_metric(self, run: dict[str, Any]) -> float:
        """Composite score based on achievements unlocked, inventory tiers, and survival."""
        achievements = float(run.get("achievements", 0))
        steps = float(run.get("turns", run.get("steps", 0)))
        return achievements * 100.0 + steps * 0.1

    def extract_milestones_and_stalls(
        self, ticks: list[dict[str, Any]]
    ) -> tuple[str, str, dict[str, Any]]:
        ach_count = 0
        if ticks:
            last = ticks[-1]
            ach_count = last.get("achievements", 0)
        summary = f"Craftax run: {len(ticks)} steps, {ach_count} achievements"
        return summary, summary, {"achievements": ach_count, "steps": len(ticks)}

    def get_api_docs(self) -> str:
        return """
# Craftax API Specification
- `Action`: Discrete IntEnum:
  - Movement: `LEFT` (1), `RIGHT` (2), `UP` (3), `DOWN` (4)
  - Interaction: `DO` (5) to mine/attack facing block
  - Crafting: `MAKE_WOOD_PICKAXE` (11), `MAKE_STONE_PICKAXE` (12), etc.
- `state`:
  - `state.y`, `state.x`: coordinates
  - `state.health`, `state.food`, `state.drink`, `state.energy`
  - `state.inventory`: dict of resource counts (e.g. `state.inventory['wood']`)
  - `state.mem`: WorldMemory tracking visited coordinates and oscillation
"""

    def search_knowledge_base(self, query: str, top_k: int = 3) -> str:
        tech_tree = {
            "wood": "Collect wood by facing a tree (1) and using `Action.DO`.",
            "stone": "Collect stone by facing stone (2) with a pickaxe and using `Action.DO`.",
            "pickaxe": "Craft a wood pickaxe using 1 wood with `Action.MAKE_WOOD_PICKAXE`.",
            "food": "Eat food or maintain energy before hunger drains health.",
        }
        for k, v in tech_tree.items():
            if k in query.lower():
                return f"### [[{k.title()}]]\n{v}"
        return "Craftax tech tree: collect wood -> craft pickaxe -> mine stone."
