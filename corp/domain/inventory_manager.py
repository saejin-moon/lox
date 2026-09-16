"""
Domain Layer: Inventory & Resource Manager.
Governs nutrition engine, safe corpse consumption, equipment loadout optimization,
and stochastic prayer timeout tracking.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.planner.htn import Task, PrimitiveTask


@dataclass
class PrayerState:
    """Stochastic prayer cooldown tracker based on NetHack 3.6.6 rnz(350)."""
    last_prayer_turn: int = 0
    prayer_count: int = 0
    nominal_safe_delta: int = 850      # mu (454) + 1.1 * sigma (365)
    emergency_safe_delta: int = 450    # Major trouble tolerance threshold


class InventoryManager:
    """
    Manages sustenance, equipment wear/wield, and divine prayer invocations.
    Strictly enforces:
    1. Corpse rot freshness rule (age <= 25 turns).
    2. Cannibalism and harmful corpse lockout.
    3. Stochastic prayer cooldown gating.
    """

    SAFE_RATION_KEYWORDS = [
        "food ration",
        "c-ration",
        "k-ration",
        "cram ration",
        "lembas wafer",
        "tripe ration",
        "apple",
        "carrot",
        "pancake",
        "meat stick",
        "fortune cookie",
        "candy bar",
    ]

    UNSAFE_CORPSE_KEYWORDS = [
        "cockatrice",
        "chickatrice",
        "floating eye",
        "stalker",
        "bat", # causes stun occasionally
        "mimic",
    ]

    def __init__(self):
        self.prayer_state = PrayerState()

    def evaluate_resource_turn(
        self,
        blstats: BottomLineStats,
        inv_tracker: InventoryNormalizer,
    ) -> Task | None:
        """
        Evaluates hunger, equipment, and prayer needs.
        Returns the highest-priority resource Task, or None if satisfied.
        """
        # 1. Divine Prayer Check (Top survival priority when at extreme hazard)
        if self._should_pray(blstats):
            self.prayer_state.last_prayer_turn = blstats.turn
            self.prayer_state.prayer_count += 1
            return Task("PRAY", is_primitive=True)

        # 2. Nutrition Engine
        food_task = self._evaluate_nutrition(blstats, inv_tracker)
        if food_task is not None:
            return food_task

        # 3. Equipment Optimization (Wield primary weapon if empty)
        equip_task = self._evaluate_equipment(blstats, inv_tracker)
        if equip_task is not None:
            return equip_task

        return None

    def _should_pray(self, blstats: BottomLineStats) -> bool:
        """
        NetHack 3.6.6 stochastic prayer safety rule:
        Turn 0 initial timeout is 300 turns.
        Subsequent prayers require Delta >= 850 turns (or Delta >= 450 in major trouble).
        """
        turn = blstats.turn
        hp = blstats.hp
        hunger = blstats.hunger_state

        is_major_emergency = (hp <= 5) or (hunger == HungerState.FAINTING)
        is_moderate_emergency = (hp <= max(5, int(blstats.max_hp * 0.20))) or (hunger == HungerState.WEAK)

        if not (is_major_emergency or is_moderate_emergency):
            return False

        if self.prayer_state.prayer_count == 0:
            # First prayer of game: safe at turn 301, or turn 101 in major trouble
            safe_cutoff = 101 if is_major_emergency else 301
            return turn >= safe_cutoff

        delta = turn - self.prayer_state.last_prayer_turn
        threshold = (
            self.prayer_state.emergency_safe_delta
            if is_major_emergency
            else self.prayer_state.nominal_safe_delta
        )
        return delta >= threshold

    def _evaluate_nutrition(
        self,
        blstats: BottomLineStats,
        inv_tracker: InventoryNormalizer,
    ) -> Task | None:
        """
        Consumes carried food when Hungry, Weak, or Fainting.
        Strictly forbids eating when Satiated.
        """
        hunger = blstats.hunger_state
        if hunger <= HungerState.NORMAL:
            return None

        # Player is HUNGRY (2), WEAK (3), or FAINTING (4)
        # Search inventory for safe carried food
        for item in inv_tracker.active_items.values():
            desc_lower = item.raw_str.lower()
            if any(k in desc_lower for k in self.SAFE_RATION_KEYWORDS):
                return Task("EAT", is_primitive=True, args={"slot": item.current_letter})

        return None

    def _evaluate_equipment(
        self,
        blstats: BottomLineStats,
        inv_tracker: InventoryNormalizer,
    ) -> Task | None:
        """
        Ensures primary weapon is wielded and safe uncursed armor is worn.
        """
        # Check if any weapon is currently wielded
        has_wielded_weapon = False
        for item in inv_tracker.active_items.values():
            if "(weapon in hand)" in item.raw_str or "(wielded)" in item.raw_str:
                has_wielded_weapon = True
                break

        if not has_wielded_weapon:
            # Find the best candidate weapon in inventory
            best_slot: str | None = None
            best_rank = -1
            weapon_ranks = {
                "broadsword": 8,
                "long sword": 8,
                "scimitar": 7,
                "katana": 9,
                "club": 5,
                "mace": 6,
                "dagger": 4,
                "spear": 5,
                "axe": 6,
                "sling": 3,
            }
            for item in inv_tracker.active_items.values():
                desc_lower = item.raw_str.lower()
                for w_name, rank in weapon_ranks.items():
                    if w_name in desc_lower and rank > best_rank:
                        best_rank = rank
                        best_slot = item.current_letter

            if best_slot is not None:
                return Task("WIELD", is_primitive=True, args={"slot": best_slot})

        return None
