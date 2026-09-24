"""
EpistemicManager: Central coordinator managing item beliefs, sensory dispatch,
and entropy safe-gates across the session.
"""

from typing import Any
import numpy as np

from lox.epistemic.belief_state import ItemBeliefState
from lox.epistemic.entropy_gates import ShannonSafeGate
from lox.epistemic.listeners.altar_listener import AltarListener
from lox.epistemic.listeners.pet_listener import PetListener
from lox.epistemic.listeners.price_listener import PriceIDListener
from lox.epistemic.listeners.engrave_listener import EngraveTestListener
from lox.env.inventory_tracker import NormalizedItem
from lox.env.blstats import BottomLineStats


class EpistemicManager:
    """
    Maintains Bayesian belief states over all tracked items in the POMDP.
    """

    def __init__(self):
        self.item_beliefs: dict[str, ItemBeliefState] = {}  # Keyed by UID
        self.static_price_map: dict[str, int] = {}          # Item name -> base price

    def register_price_map(self, price_map: dict[str, int]):
        """Injects static base price dictionary for Price ID candidate elimination."""
        self.static_price_map.update(price_map)

    def get_or_create_belief(self, item: NormalizedItem) -> ItemBeliefState:
        """Retrieves existing belief state or initializes a new one from a NormalizedItem."""
        if item.uid in self.item_beliefs:
            belief = self.item_beliefs[item.uid]
            # Synchronize explicit BUC if revealed in text
            if item.buc_state != "UNKNOWN" and not belief.buc_determined:
                belief.collapse_buc(item.buc_state)
            return belief

        # Initialize new belief state
        belief = ItemBeliefState(
            uid=item.uid,
            appearance=item.raw_str,
            item_class=self._infer_item_class(item),
        )

        if item.buc_state != "UNKNOWN":
            belief.collapse_buc(item.buc_state)

        self.item_beliefs[item.uid] = belief
        return belief

    def synchronize_inventory(self, active_items: list[NormalizedItem]):
        """Ensures all active inventory items have corresponding belief states."""
        for item in active_items:
            self.get_or_create_belief(item)

    def on_altar_drop(self, uid: str, blstats: BottomLineStats, msg: str) -> bool:
        """Processes an altar drop observation for the specified item UID."""
        belief = self.item_beliefs.get(uid)
        if not belief:
            return False
        return AltarListener.process_drop(belief, blstats, msg)

    def on_pet_interaction(self, uid: str, stepped_willingly: bool) -> bool:
        """Processes a pet movement observation over the specified item UID."""
        belief = self.item_beliefs.get(uid)
        if not belief:
            return False
        return PetListener.process_pet_interaction(belief, stepped_willingly)

    def on_price_observed(
        self,
        uid: str,
        observed_price: int,
        is_selling: bool,
        cha: int = 10,
        is_sucker: bool = False,
    ) -> list[str]:
        """Filters candidate identities for an item using observed merchant price."""
        belief = self.item_beliefs.get(uid)
        if not belief:
            return []
        return PriceIDListener.filter_candidates_by_price(
            item_belief=belief,
            candidate_price_map=self.static_price_map,
            observed_price=observed_price,
            is_selling=is_selling,
            cha=cha,
            is_sucker=is_sucker,
        )

    def on_wand_engrave(self, uid: str, terminal_message: str) -> list[str]:
        """Prunes wand candidate identities based on floor engrave diagnostic message."""
        belief = self.item_beliefs.get(uid)
        if not belief:
            return []
        return EngraveTestListener.process_engrave_result(belief, terminal_message)

    def verify_action_safety(self, action_type: str, uid: str) -> tuple[bool, str]:
        """
        Enforces Shannon entropy safe-gates before executing primitive actions on an item.
        """
        belief = self.item_beliefs.get(uid)
        if not belief:
            return True, "SAFE"

        act = action_type.upper()
        if act == "EQUIP":
            return ShannonSafeGate.can_safely_equip(belief)
        elif act == "QUAFF":
            return ShannonSafeGate.can_safely_quaff(belief)
        elif act == "READ":
            return ShannonSafeGate.can_safely_read(belief)

        return True, "SAFE"

    def _infer_item_class(self, item: NormalizedItem) -> str:
        s = item.raw_str.lower()
        if "potion" in s:
            return "potion"
        elif "scroll" in s:
            return "scroll"
        elif "wand" in s:
            return "wand"
        elif "ring" in s:
            return "ring"
        elif "amulet" in s:
            return "amulet"
        elif "sword" in s or "dagger" in s or "bow" in s or "axe" in s or "mace" in s or "spear" in s:
            return "weapon"
        elif "armor" in s or "shield" in s or "cloak" in s or "helm" in s or "boots" in s or "gloves" in s:
            return "armor"
        elif "ration" in s or "corpse" in s or "food" in s or "lichen" in s or "apple" in s or "meat" in s:
            return "food"
        return "tool"
