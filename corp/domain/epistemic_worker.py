"""
Domain Layer: Epistemic Worker.
Conducts active epistemic experimentation in NetHack 3.6.6:
1. Altar B.U.C. Testing (deterministic beatitude collapse via divine flash observation).
2. Altar Sacrificing (corpse offering on co-aligned altars for Luck, favor, and artifact gifts).
3. Wand Engrave-Testing (floor dust scratch diagnostics to prune wand candidate identities).
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.epistemic.epistemic_manager import EpistemicManager
from corp.planner.htn import Task


@dataclass
class EpistemicWorker:
    """
    Directs active epistemic experimentation to eliminate uncertainty over critical items.
    """
    epistemic_mgr: EpistemicManager
    tested_uids: set[str] = field(default_factory=set)
    tested_descriptions: set[str] = field(default_factory=set)
    engraved_wand_uids: set[str] = field(default_factory=set)
    altar_test_count: int = 0

    UNSAFE_OFFER_KEYWORDS = (
        "cockatrice",
        "chickatrice",
        "human",
        "cat",
        "dog",
        "kitten",
        "puppy",
        "little dog",
        "large dog",
        "horse",
        "pony",
        "warhorse",
    )

    TESTABLE_KEYWORDS = (
        # Armor pieces (highest AC impact)
        "mail", "armor", "cloak", "helm", "helmet", "cap", "hat", "plate", "coat", "robe", "shirt", "suit", "boots", "shoes", "gloves", "gauntlets", "shield",
        # Weapons
        "sword", "dagger", "axe", "mace", "spear", "bow", "crossbow", "sling", "dart", "flail", "hammer",
        # Magic items
        "wand", "ring", "amulet", "scroll", "potion",
    )

    def reset(self):
        """Clears episode experimentation state."""
        self.tested_uids.clear()
        self.tested_descriptions.clear()
        self.engraved_wand_uids.clear()
        self.altar_test_count = 0

    def has_untested_items(self, inv_tracker: InventoryNormalizer) -> bool:
        """Returns True if player holds any unequipped item with UNKNOWN BUC that can be tested on an altar."""
        if not inv_tracker or self.altar_test_count >= 10:
            return False
        for item in inv_tracker.get_active_items():
            if item.equipped or "(being worn)" in item.raw_str.lower():
                continue
            if item.uid in self.tested_uids or item.raw_str.lower() in self.tested_descriptions:
                continue
            if item.buc_state == "UNKNOWN":
                desc = item.raw_str.lower()
                if any(k in desc for k in self.TESTABLE_KEYWORDS):
                    return True
        return False

    def get_sacrificable_corpse(
        self,
        inv_tracker: InventoryNormalizer,
        current_turn: int,
    ) -> NormalizedItem | None:
        """
        Finds a fresh (< 50 turns old), non-hazardous corpse in inventory suitable for altar sacrifice.
        """
        if not inv_tracker:
            return None
        for item in inv_tracker.get_active_items():
            desc = item.raw_str.lower()
            if "corpse" in desc:
                # Freshness check: NetHack corpses rot in ~50 turns
                age = current_turn - item.first_seen_turn
                if age <= 50:
                    if not any(k in desc for k in self.UNSAFE_OFFER_KEYWORDS):
                        return item
        return None

    def _get_best_untested_item(
        self,
        inv_tracker: InventoryNormalizer,
    ) -> NormalizedItem | None:
        """
        Picks the most tactically valuable untested item (Armor > Weapon > Wand > Ring > Potion/Scroll).
        """
        candidates = []
        for item in inv_tracker.get_active_items():
            if item.equipped or "(being worn)" in item.raw_str.lower():
                continue
            if item.uid in self.tested_uids or item.raw_str.lower() in self.tested_descriptions:
                continue
            if item.buc_state == "UNKNOWN":
                desc = item.raw_str.lower()
                # Determine priority tier
                tier = 5
                if any(k in desc for k in ("mail", "armor", "cloak", "helm", "helmet", "cap", "hat", "plate", "coat", "robe", "shirt", "suit", "boots", "shoes", "gloves", "gauntlets", "shield")):
                    tier = 1  # Highest: Armor
                elif any(k in desc for k in ("sword", "dagger", "axe", "mace", "spear", "bow", "crossbow", "sling", "dart", "flail", "hammer")):
                    tier = 2  # Weapons
                elif "wand" in desc:
                    tier = 3  # Wands
                elif any(k in desc for k in ("ring", "amulet")):
                    tier = 4  # Jewelry
                elif any(k in desc for k in ("potion", "scroll")):
                    tier = 5  # Consumables
                else:
                    continue
                candidates.append((tier, item))

        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    def _get_untested_wand(
        self,
        inv_tracker: InventoryNormalizer,
    ) -> NormalizedItem | None:
        """
        Finds an unidentified, non-cursed wand suitable for dust engrave-testing.
        """
        if not inv_tracker:
            return None
        for item in inv_tracker.get_active_items():
            desc = item.raw_str.lower()
            if "wand" in desc and item.uid not in self.engraved_wand_uids:
                # Safety check: Cursed wands have a 1% explosion risk when engraved!
                # Wand MUST be verified uncursed or blessed before engrave-testing.
                belief = self.epistemic_mgr.get_or_create_belief(item)
                if belief.p_cursed == 0.0 or item.buc_state in ("UNCURSED", "BLESSED"):
                    # Check if wand is already identified
                    if not belief.is_formally_identified():
                        return item
        return None

    def evaluate_epistemic_turn(
        self,
        blstats: BottomLineStats,
        chars: np.ndarray,
        inv_tracker: InventoryNormalizer,
        lvl_altars: set[tuple[int, int]],
        has_adjacent_hostiles: bool = False,
        visible_monster_count: int = 0,
    ) -> Task | None:
        """
        Evaluates potential active epistemic actions:
        1. Altar Sacrifice (when on altar with fresh corpse).
        2. Altar B.U.C. Testing (when on altar with untested inventory items).
        3. Wand Engrave-Testing (when in safe room floor tile with un-IDed safe wand).
        """
        # Tactical interlock: never conduct epistemic experiments during active melee combat
        if has_adjacent_hostiles:
            return None

        py, px = blstats.y, blstats.x
        is_on_altar = (py, px) in lvl_altars

        # --- Subsystem A: Altar Actions ---
        if is_on_altar:
            # 1. Altar Sacrifice (Corpse offering for Luck & artifacts)
            corpse_item = self.get_sacrificable_corpse(inv_tracker, blstats.turn)
            if corpse_item is not None:
                return Task(
                    "OFFER",
                    is_primitive=True,
                    args={"slot": corpse_item.current_letter, "uid": corpse_item.uid},
                )

            # 2. Altar B.U.C. Testing (Drop -> Flash -> Pickup)
            # Only test if unencumbered and test count < 10 to avoid altar camping
            if blstats.encumbrance == 0 and self.altar_test_count < 10:
                untested_item = self._get_best_untested_item(inv_tracker)
                if untested_item is not None:
                    return Task(
                        "ALTAR_TEST",
                        is_primitive=True,
                        args={"slot": untested_item.current_letter, "uid": untested_item.uid},
                    )

        # --- Subsystem B: Wand Engrave-Testing ---
        # Conduct only in calm conditions: no visible monsters, safe room floor tile, healthy HP
        if not is_on_altar and visible_monster_count == 0:
            if chars[py, px] == ord(".") and blstats.hunger_state <= HungerState.NORMAL:
                if blstats.hp >= int(blstats.max_hp * 0.70):
                    wand = self._get_untested_wand(inv_tracker)
                    if wand is not None:
                        return Task(
                            "ENGRAVE_WAND",
                            is_primitive=True,
                            args={"slot": wand.current_letter, "uid": wand.uid},
                        )

        return None

    def on_altar_test_result(
        self,
        uid: str,
        flash_msg: str,
        blstats: BottomLineStats,
        inv_tracker: InventoryNormalizer,
    ):
        """
        Processes observation from altar drop, collapses belief, and synchronizes with NormalizedItem.
        """
        self.altar_test_count += 1
        self.tested_uids.add(uid)
        item = inv_tracker.active_items.get(uid) if inv_tracker else None
        if not item:
            return

        self.tested_descriptions.add(item.raw_str.lower())
        belief = self.epistemic_mgr.get_or_create_belief(item)
        self.epistemic_mgr.on_altar_drop(uid, blstats, flash_msg)

        # Synchronize collapsed BUC state back to the concrete item
        item.buc_state = belief.buc_state

    def on_wand_engrave_result(
        self,
        uid: str,
        result_msg: str,
        inv_tracker: InventoryNormalizer,
    ):
        """
        Processes diagnostic message from wand scratch-testing, pruning candidate identities.
        """
        self.engraved_wand_uids.add(uid)
        item = inv_tracker.active_items.get(uid) if inv_tracker else None
        if not item:
            return

        belief = self.epistemic_mgr.get_or_create_belief(item)
        surviving = self.epistemic_mgr.on_wand_engrave(uid, result_msg)

        # If collapsed to single identity, update appearance / raw string
        if belief.is_formally_identified() and surviving:
            identified_name = surviving[0]
            if identified_name not in item.raw_str.lower():
                item.raw_str = f"{item.raw_str} ({identified_name})"
