"""
LOX Epistemic Belief-State Engine: POMDP latent property tracking.
Tracks BUC beatitude distributions (P(blessed), P(uncursed), P(cursed)),
candidate item identities, charges/enchantment bounds, and sensory message listeners.
Exposes Shannon entropy safe-gates to inform policy predicates without engine-level hard stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import numpy.typing as npt


@dataclass
class ItemBeliefState:
    """
    Tracks epistemic uncertainty over an individual item's latent properties.
    """
    uid: str
    name: str = ""
    item_class: str = ""  # "potion", "scroll", "wand", "ring", "amulet", "armor", "weapon", "food", "tool"
    appearance: str = ""
    slot_letter: str = ""
    candidate_identities: list[str] = field(default_factory=list)
    identity_probs: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # BUC probabilities: [P(Blessed), P(Uncursed), P(Cursed)]
    # Default prior for random dungeon items: 10% blessed, 80% uncursed, 10% cursed
    p_buc: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.10, 0.80, 0.10], dtype=np.float64)
    )

    charges_min: int = 0
    charges_max: int = 0
    charges_known: bool = False

    enchantment_min: int = -5
    enchantment_max: int = 7
    enchantment_known: bool = False

    erosion_level: int = 0

    def __post_init__(self):
        if len(self.candidate_identities) > 0 and len(self.identity_probs) == 0:
            k = len(self.candidate_identities)
            self.identity_probs = np.full(k, 1.0 / k, dtype=np.float64)

    @property
    def p_blessed(self) -> float:
        return float(self.p_buc[0])

    @property
    def p_uncursed(self) -> float:
        return float(self.p_buc[1])

    @property
    def p_cursed(self) -> float:
        return float(self.p_buc[2])

    def entropy_identity(self) -> float:
        """Computes Shannon entropy H(Identity) in bits."""
        if len(self.identity_probs) <= 1:
            return 0.0
        p = self.identity_probs[self.identity_probs > 1e-12]
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def entropy_buc(self) -> float:
        """Computes Shannon entropy H(BUC) in bits."""
        p = self.p_buc[self.p_buc > 1e-12]
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def collapse_buc(self, state: str) -> None:
        """Deterministically collapses BUC distribution into a one-hot vector."""
        state_upper = state.upper()
        if state_upper == "BLESSED":
            self.p_buc = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif state_upper == "UNCURSED":
            self.p_buc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif state_upper == "CURSED":
            self.p_buc = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError(f"Unknown BUC state: {state}")

    def update_buc_pet_step(self) -> None:
        """Applies likelihood update when a pet willingly steps onto item tile (Cursed = 0)."""
        p_b = self.p_buc[0]
        p_u = self.p_buc[1]
        denom = p_b + p_u
        if denom > 1e-12:
            self.p_buc = np.array([p_b / denom, p_u / denom, 0.0], dtype=np.float64)
        else:
            self.p_buc = np.array([0.10, 0.90, 0.0], dtype=np.float64)

    def update_buc_pet_refusal(self) -> None:
        """Applies likelihood update when a pet stops/whines and refuses to step onto tile."""
        self.p_buc = np.array([0.005, 0.005, 0.99], dtype=np.float64)

    def prune_candidates(self, surviving_candidates: set[str]) -> None:
        """
        Prunes candidate identities and re-normalizes the remaining probability mass.
        """
        if not self.candidate_identities:
            return

        new_names = []
        new_mass = []
        for c_name, prob in zip(self.candidate_identities, self.identity_probs):
            if c_name in surviving_candidates:
                new_names.append(c_name)
                new_mass.append(prob)

        if not new_names:
            self.candidate_identities = list(surviving_candidates)
            k = len(self.candidate_identities)
            self.identity_probs = np.full(k, 1.0 / k, dtype=np.float64)
            return

        total = sum(new_mass)
        self.candidate_identities = new_names
        if total > 1e-12:
            self.identity_probs = np.array([m / total for m in new_mass], dtype=np.float64)
        else:
            k = len(new_names)
            self.identity_probs = np.full(k, 1.0 / k, dtype=np.float64)

    def is_formally_identified(self) -> bool:
        """True if identity is 100% collapsed to a single candidate."""
        return len(self.candidate_identities) == 1 and (
            len(self.identity_probs) == 0 or self.identity_probs[0] >= 0.999
        )


class AltarListener:
    """Listens to game messages produced by dropping items on altars."""

    @staticmethod
    def process_drop(
        belief: ItemBeliefState,
        is_blind: bool,
        message: str,
    ) -> bool:
        """
        Parses NetHack 3.6.6 altar drop messages.
        Returns True if a valid BUC update was processed.
        """
        if is_blind:
            # Blind hero cannot observe flash colors
            return False

        msg = message.lower()
        if "flash of black light" in msg:
            belief.collapse_buc("CURSED")
            return True
        elif "amber flash" in msg:
            belief.collapse_buc("BLESSED")
            return True
        else:
            # Dropped on an altar without a flash indicates UNCURSED
            belief.collapse_buc("UNCURSED")
            return True

        return False


class PetListener:
    """Listens to pet movement over item piles to infer BUC beatitude."""

    @staticmethod
    def process_pet_interaction(
        belief: ItemBeliefState,
        pet_stepped_willingly: bool,
    ) -> bool:
        """
        Pets refuse to step on cursed items unless starved/confused.
        Exception: Comestibles (pets greedily eat cursed food).
        """
        if belief.item_class == "food":
            return False

        if pet_stepped_willingly:
            belief.update_buc_pet_step()
            return True
        else:
            belief.update_buc_pet_refusal()
            return True


class PriceIDListener:
    """Filters candidate item identities by shopkeeper buying and selling prices."""

    @staticmethod
    def filter_candidates_by_price(
        item_belief: ItemBeliefState,
        candidate_price_map: dict[str, int],
        observed_price: int,
        is_selling: bool = False,
        charisma: int = 10,
        surcharge: float = 1.0,
    ) -> list[str]:
        """
        In NetHack 3.6.6:
        Selling to shopkeeper: offer = base_price // 2 (modified by charisma/surcharge).
        Buying from shopkeeper: offer = base_price * markup.
        """
        surviving = []
        for name in item_belief.candidate_identities:
            base_price = candidate_price_map.get(name)
            if base_price is None:
                continue

            if is_selling:
                expected_offer = base_price // 2
                if abs(expected_offer - observed_price) <= 1:
                    surviving.append(name)
            else:
                expected_price = int(base_price * surcharge)
                if abs(expected_price - observed_price) <= 2:
                    surviving.append(name)

        if surviving:
            item_belief.prune_candidates(set(surviving))
        return surviving


class ShannonSafeGate:
    """
    Shannon Entropy & Probability Safe-Gates.
    Evaluates whether an action (quaff, equip, read) is safe given current epistemic uncertainty.
    """

    LETHAL_POTIONS: set[str] = {
        "potion of death",
        "potion of paralysis",
        "potion of sleeping",
        "potion of poison",
        "potion of blindness",
        "potion of hallucination",
        "potion of amnesia",
    }

    LETHAL_SCROLLS: set[str] = {
        "scroll of destroy armor",
        "scroll of punish",
        "scroll of amnesia",
        "scroll of scare monster",  # Dangerous when read incorrectly
    }

    @classmethod
    def can_safely_equip(cls, belief: ItemBeliefState, max_cursed_prob: float = 0.05) -> tuple[bool, str]:
        """
        Equipping cursed armor or weapons welds/locks them to the hero, preventing removal.
        Random dungeon armor has an 80% uncursed, 10% blessed, 10% cursed prior (90% safe).
        """
        if belief.p_cursed > max_cursed_prob:
            return False, f"VETO: P(Cursed)={belief.p_cursed:.2f} > {max_cursed_prob:.2f}"
        return True, "SAFE"

    @classmethod
    def can_safely_quaff(cls, belief: ItemBeliefState, max_cursed_prob: float = 0.05) -> tuple[bool, str]:
        """
        Quaffing unknown potions risks paralysis, poison, or blindness.
        """
        if belief.p_cursed > max_cursed_prob:
            return False, f"VETO: P(Cursed)={belief.p_cursed:.2f} > {max_cursed_prob:.2f}"

        # If identity is uncollapsed, verify no lethal candidates exist in candidate pool
        if not belief.is_formally_identified():
            lethal_in_pool = [c for c in belief.candidate_identities if c.lower() in cls.LETHAL_POTIONS]
            if lethal_in_pool:
                return False, f"VETO: Lethal candidate identities in pool: {lethal_in_pool}"

        return True, "SAFE"

    @classmethod
    def can_safely_read(cls, belief: ItemBeliefState, max_cursed_prob: float = 0.05) -> tuple[bool, str]:
        """
        Reading scrolls: cursed scrolls often backfire or curse other inventory items.
        """
        if belief.p_cursed > max_cursed_prob:
            return False, f"VETO: P(Cursed)={belief.p_cursed:.2f} > {max_cursed_prob:.2f}"

        if not belief.is_formally_identified():
            lethal_in_pool = [c for c in belief.candidate_identities if c.lower() in cls.LETHAL_SCROLLS]
            if lethal_in_pool:
                return False, f"VETO: Lethal candidate identities in pool: {lethal_in_pool}"

        return True, "SAFE"


class EpistemicEngine:
    """
    Session-level Epistemic Manager maintaining belief-states across inventory and floor items.
    """

    def __init__(self):
        self.beliefs: dict[str, ItemBeliefState] = {}
        self.tested_altar_uids: set[str] = set()

    def get_or_create(
        self,
        uid: str,
        name: str = "",
        item_class: str = "",
        slot_letter: str = "",
    ) -> ItemBeliefState:
        """Retrieves or registers an item belief state."""
        if uid not in self.beliefs:
            # Check if name contains explicit BUC tags from game (e.g. "blessed", "uncursed", "cursed")
            name_lower = name.lower()
            initial_buc = np.array([0.10, 0.80, 0.10], dtype=np.float64)
            if "blessed" in name_lower:
                initial_buc = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif "uncursed" in name_lower:
                initial_buc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            elif "cursed" in name_lower:
                initial_buc = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            self.beliefs[uid] = ItemBeliefState(
                uid=uid,
                name=name,
                item_class=item_class,
                slot_letter=slot_letter,
                p_buc=initial_buc,
            )
        else:
            belief = self.beliefs[uid]
            if name and not belief.name:
                belief.name = name
            if slot_letter:
                belief.slot_letter = slot_letter
            if item_class and not belief.item_class:
                belief.item_class = item_class

        return self.beliefs[uid]

    def update_from_altar_drop(
        self,
        uid: str,
        is_blind: bool,
        message: str,
    ) -> bool:
        """Applies altar drop listener update."""
        if uid in self.beliefs:
            updated = AltarListener.process_drop(self.beliefs[uid], is_blind, message)
            if updated:
                self.tested_altar_uids.add(uid)
            return updated
        return False

    def count_untested_buc(self) -> int:
        """Returns count of inventory items whose BUC entropy is non-zero."""
        count = 0
        for b in self.beliefs.values():
            if b.entropy_buc() > 0.01:
                count += 1
        return count

    def can_safely_wear(self, uid: str, max_cursed_prob: float = 0.05) -> bool:
        if uid in self.beliefs:
            safe, _ = ShannonSafeGate.can_safely_equip(self.beliefs[uid], max_cursed_prob=max_cursed_prob)
            return safe
        return False

    def can_safely_quaff(self, uid: str) -> bool:
        if uid in self.beliefs:
            safe, _ = ShannonSafeGate.can_safely_quaff(self.beliefs[uid])
            return safe
        return False
