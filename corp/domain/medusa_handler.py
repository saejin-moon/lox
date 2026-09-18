"""
Domain Layer: Medusa Encounter Handler (DL 20-25 preparation and traversal).

Medusa's gaze petrifies instantly (fatal in ~3 turns once STONE starts).
NetHack 3.6.6 ground-truth protections, in order of reliability:
1. Extrinsic reflection (shield of reflection / amulet of reflection):
   the gaze reflects back at Medusa, killing her.
2. Blindness (blindfold or towel applied): a blind hero cannot be petrified
   by the gaze, but Medusa gains melee advantage.
3. Perseus statue on Medusa's island guarantees a shield of reflection:
   loot containers on her level before engaging.

The handler equips reflection BEFORE the hero approaches Medusa's level,
applies a blindfold as a fallback, and coordinates the Perseus statue loot.
"""

from dataclasses import dataclass
from typing import Any

from corp.env.blstats import BottomLineStats
from corp.planner.htn import Task


# Medusa's island sits at DL 20-21 in the Dungeons of Doom; prep starts one level early
MEDUSA_PREP_DEPTH_RANGE = (19, 25)


@dataclass
class MedusaPrepState:
    reflection_equipped: bool = False
    blindfold_applied: bool = False
    statue_looted: bool = False


class MedusaHandler:
    """
    Specialized handler for the DL 20-25 Medusa encounter:
    1. Auto-equip shield of reflection or amulet of reflection before her level.
    2. If no reflection: auto-apply blindfold/towel before Medusa's level.
    3. Handle Perseus statue (guaranteed shield of reflection) looting.
    """

    REFLECTION_KEYS = ("shield of reflection", "amulet of reflection")
    BLINDFOLD_KEYS = ("blindfold", "towel")

    def __init__(self):
        self.state = MedusaPrepState()

    def reset(self):
        self.state = MedusaPrepState()

    @staticmethod
    def _iter_items(inv_tracker: Any):
        if inv_tracker is None:
            return []
        return (
            inv_tracker.get_active_items()
            if hasattr(inv_tracker, "get_active_items")
            else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
        )

    def is_medusa_approach(self, blstats: BottomLineStats) -> bool:
        """True when the hero is on the approach corridor to Medusa's island."""
        return blstats.dungeon_number == 0 and (
            MEDUSA_PREP_DEPTH_RANGE[0] <= blstats.depth <= MEDUSA_PREP_DEPTH_RANGE[1]
        )

    def evaluate_medusa_prep(
        self,
        blstats: BottomLineStats,
        inv_tracker: Any = None,
        has_reflection: bool = False,
    ) -> Task | None:
        """
        Returns an equipping Task when approaching Medusa without adequate protection.
        Returns None when protection is adequate or Medusa is not imminent.
        """
        if not self.is_medusa_approach(blstats):
            return None

        if has_reflection or self.state.reflection_equipped:
            return None

        items = self._iter_items(inv_tracker)

        # 1. Equip carried reflection (shield -> WEAR, amulet -> PUTON)
        for item in items:
            desc = item.raw_str.lower()
            is_cursed = desc.startswith("cursed") or " cursed" in desc
            if is_cursed:
                continue
            if "(being worn)" in desc or "(worn)" in desc:
                continue
            if "amulet of reflection" in desc:
                self.state.reflection_equipped = True
                return Task("PUTON", is_primitive=True, args={"slot": item.current_letter})
            if "shield of reflection" in desc:
                self.state.reflection_equipped = True
                return Task("WEAR", is_primitive=True, args={"slot": item.current_letter})

        # 2. No reflection: apply blindfold/towel so the gaze cannot petrify
        if not self.state.blindfold_applied:
            for item in items:
                desc = item.raw_str.lower()
                is_cursed = desc.startswith("cursed") or " cursed" in desc
                if is_cursed:
                    continue
                if "(being worn)" in desc:
                    continue
                if any(k in desc for k in self.BLINDFOLD_KEYS):
                    self.state.blindfold_applied = True
                    return Task("APPLY", is_primitive=True, args={"slot": item.current_letter})

        return None

    def should_loot_perseus_statue(self, blstats: BottomLineStats, has_reflection: bool) -> bool:
        """
        The statue of Perseus on Medusa's island guarantees a shield of reflection.
        When the hero lacks reflection, containers on Medusa's level are priority loot.
        """
        return (
            blstats.dungeon_number == 0
            and 20 <= blstats.depth <= 21
            and not has_reflection
            and not self.state.reflection_equipped
        )
