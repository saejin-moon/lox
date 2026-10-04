"""
LOX Poison Resistance Harvest Solver:
Prioritizes and navigates to safe corpses that confer permanent poison resistance (killer bees, soldier ants, snakes)
before descending to deeper dungeon levels (depth >= 4).
"""

from __future__ import annotations

from typing import Any

from lox.core.types import Action, Observation

# Monsters granting poison resistance upon eating corpse (with high probability in NetHack 3.6.6)
POISON_RES_PROVIDERS: set[str] = {
    "killer bee",
    "soldier ant",
    "giant ant",
    "fire ant",
    "black nagaspur",
    "golden naga",
    "guardian naga",
    "red naga",
    "python",
    "cobra",
    "pit viper",
    "spider",
    "centipede",
}


class PoisonResHarvestSolver:
    """Selects and routes the hero to safe poison-resistance conferring corpses."""

    @classmethod
    def can_harvest_from_corpses(
        cls,
        corpses: list[Any],
        has_poison_res: bool = False,
        hostile_count: int = 0,
        adjacent_hostile: bool = False,
    ) -> bool:
        """Fast predicate checking if an eligible poison provider corpse is present and safe to harvest."""
        if has_poison_res or hostile_count > 0 or adjacent_hostile:
            return False
        for corpse in corpses:
            if getattr(corpse, "is_deadly", False) or not getattr(
                corpse, "is_fresh", True
            ):
                continue
            name_lower = getattr(corpse, "name", "").lower()
            if any(prov in name_lower for prov in POISON_RES_PROVIDERS):
                return True
        return False

    @staticmethod
    def plan_step(obs: Observation) -> Action | None:
        """
        If hero lacks poison resistance and enemies are not threatening immediately,
        steps to or consumes candidate corpses.
        """
        has_res = getattr(obs.hero, "has_poison_res", False) or getattr(
            obs.status, "has_poison_res", False
        )
        if has_res:
            return None

        # Do not eat corpses while hostiles are in FOV
        if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            return None

        # Find closest eligible corpse
        hy, hx = obs.hero.y, obs.hero.x
        best_corpse = None
        best_dist = 999.0

        for corpse in obs.corpses:
            if corpse.is_deadly or not corpse.is_fresh:
                continue

            name_lower = corpse.name.lower()
            is_provider = any(prov in name_lower for prov in POISON_RES_PROVIDERS)
            if is_provider:
                dist = abs(corpse.y - hy) + abs(corpse.x - hx)
                if dist < best_dist:
                    best_dist = dist
                    best_corpse = corpse

        if best_corpse is None:
            return None

        # If already standing on corpse, eat it
        if (hy, hx) == (best_corpse.y, best_corpse.x):
            return Action(name="eat_floor_corpse")

        # Step toward corpse
        return Action(name="step_to", target_pos=(best_corpse.y, best_corpse.x))
