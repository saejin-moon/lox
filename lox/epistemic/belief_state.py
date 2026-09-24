"""
ItemBeliefState: Probabilistic representation of an item's hidden properties
(Identity distribution, BUC beatitude distribution, charge/enchantment bounds).
"""

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


@dataclass
class ItemBeliefState:
    """
    Tracks epistemic uncertainty over an individual physical game item.
    """
    uid: str
    appearance: str = ""
    item_class: str = ""  # "potion", "scroll", "wand", "ring", "amulet", "armor", "weapon", "food", "tool"
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
    def buc_state(self) -> str:
        if self.p_blessed >= 0.99:
            return "BLESSED"
        elif self.p_uncursed >= 0.99:
            return "UNCURSED"
        elif self.p_cursed >= 0.99:
            return "CURSED"
        return "UNKNOWN"

    @property
    def p_blessed(self) -> float:
        return float(self.p_buc[0])

    @property
    def buc_determined(self) -> bool:
        """True once the BUC distribution is effectively one-hot (max prob >= 0.999).
        Equivalent to `self.p_buc.max() >= 0.999` but evaluated with scalar indexing —
        the numpy reduce was a measured hot spot (called ~75×/turn in equipment eval)."""
        p = self.p_buc
        if len(p) != 3:  # defensive: unexpected shape falls back to numpy
            return bool(p.max() >= 0.999) if len(p) else False
        return bool(p[0] >= 0.999 or p[1] >= 0.999 or p[2] >= 0.999)

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
        # Mask out zeros to avoid log2(0)
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

    def collapse_buc(self, state: str):
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

    def update_buc_pet_step(self):
        """Applies likelihood update when a pet willingly steps onto item tile (Cursed = 0)."""
        p_b = self.p_buc[0]
        p_u = self.p_buc[1]
        denom = p_b + p_u
        if denom > 1e-12:
            self.p_buc = np.array([p_b / denom, p_u / denom, 0.0], dtype=np.float64)
        else:
            self.p_buc = np.array([0.10, 0.90, 0.0], dtype=np.float64)

    def update_buc_pet_refusal(self):
        """Applies likelihood update when a pet stops/whines and refuses to step onto tile."""
        # P(Cursed|Refusal) is very high (~0.99)
        self.p_buc = np.array([0.005, 0.005, 0.99], dtype=np.float64)

    def prune_candidates(self, surviving_candidates: set[str]):
        """
        Prunes candidate identities and re-normalizes the remaining probability mass.
        """
        if not self.candidate_identities:
            return

        new_names = []
        new_mass = []
        for name, prob in zip(self.candidate_identities, self.identity_probs):
            if name in surviving_candidates:
                new_names.append(name)
                new_mass.append(prob)

        if not new_names:
            # Fallback: if all were eliminated, keep surviving_candidates with uniform prior
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
        """True if identity is 100% collapsed to a single item."""
        return len(self.candidate_identities) == 1 and self.identity_probs[0] >= 0.999
