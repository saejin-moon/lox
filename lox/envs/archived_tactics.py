"""
LOX Archived Tactical Heuristics Reference:
Preserves historical hand-tuned tactics previously embedded inside NetHackAdapter.

Archived for research reproducibility, algorithmic provenance, and ablation benchmarks.
In the pure synthesis architecture (v3.0+), these heuristics are deprecated in the
environment adapter and migrated directly into the policy generator skills
(e.g. data/modular_starter_policy.py) to prove genuine autonomous policy synthesis.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

# Direction character mapping for NetHack
DIR_CHARS: dict[tuple[int, int], str] = {
    (-1, 0): "k",  # North
    (0, 1): "l",   # East
    (1, 0): "j",   # South
    (0, -1): "h",  # West
    (-1, 1): "u",  # Northeast
    (1, 1): "n",   # Southeast
    (1, -1): "b",  # Southwest
    (-1, -1): "y", # Northwest
}

# Canonical armor tier ranking historically used for autonomous replacement
HISTORICAL_ARMOR_TIERS: dict[str, int] = {
    "dragon scale mail": 10,
    "dragon scale": 9,
    "plate mail": 8,
    "crystal plate mail": 8,
    "bronze plate mail": 7,
    "splint mail": 6,
    "banded mail": 6,
    "dwarvish mithril": 7,
    "elven mithril": 7,
    "chain mail": 5,
    "orcish chain mail": 4,
    "scale mail": 4,
    "ring mail": 3,
    "studded leather armor": 3,
    "leather armor": 2,
    "leather jacket": 1,
}


class HistoricalTactics:
    """Namespace of historical tactics decoupled from the environment adapter."""

    @staticmethod
    def evaluate_armor_tier(item_name: str) -> int:
        """Computes tier score of body armor suit based on material and type."""
        name_low = item_name.lower()
        for key, tier in HISTORICAL_ARMOR_TIERS.items():
            if key in name_low:
                return tier
        return 0

    @staticmethod
    def build_wear_armor_sequence(
        target_slot: str,
        worn_cloak_slot: str | None = None,
    ) -> list[str]:
        """Historical 6-key macro: took off cloak before wearing body armor, then re-wore cloak."""
        if worn_cloak_slot:
            return ["T", worn_cloak_slot, "W", target_slot, "W", worn_cloak_slot]
        return ["W", target_slot]

    @staticmethod
    def build_buffered_walkable_nav(
        walkable: np.ndarray,
        chars: np.ndarray,
        glyphs: np.ndarray | None,
        hostile_positions: list[tuple[int, int]],
        has_ranged_weapons: bool,
    ) -> np.ndarray:
        """
        Historical passive hazard pathfinding buffering:
        Masked out adjacent tiles of floating eyes and gas spores when the hero
        lacked ranged weapons, forcing A* navigation to detour around them at distance >= 2.
        """
        walkable_nav = walkable.copy()
        if not has_ranged_weapons and glyphs is not None:
            GLYPH_CMAP_OFF = 2359
            h, w = walkable.shape
            for gy, gx in hostile_positions:
                # Mask out immediate neighbors if floating eye or gas spore
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            walkable_nav[ny, nx] = False
        return walkable_nav

    @staticmethod
    def solve_deadlock_emergency_strike(
        consecutive_waits: int,
        adjacent_hostiles: list[tuple[int, int]],
        has_daggers: bool,
        has_wands: bool,
    ) -> str:
        """
        Historical deadlock circuit breaker:
        When cornered in a dead end next to a passive hazard with no retreat route,
        after 2 consecutive waits, triggered ranged harassment or emergency strike
        rather than dying of 20,000-turn starvation.
        """
        if consecutive_waits >= 2:
            if has_wands:
                return "zap_offensive_wand"
            if has_daggers:
                return "throw_dagger"
            return "melee_attack_hostile"
        return "wait"
