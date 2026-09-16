"""
ShannonSafeGate: Enforces mathematical Shannon entropy thresholds and BUC risk-gates
to prevent fatal blind actions (quaffing death, equipping strangulation, reading amnesia).
"""

from typing import Tuple
from corp.epistemic.belief_state import ItemBeliefState


class ShannonSafeGate:
    """
    Evaluates action risk on items before HTN decomposition executes primitives.
    Vetoes actions when P(Cursed) > tau_cursed or Identity Entropy > tau_entropy.
    """

    DEFAULT_CURSED_VETO_THRESHOLD = 0.05  # Max 5% probability of curse tolerated for equip/quaff/read
    DEFAULT_IDENTITY_ENTROPY_THRESHOLD = 0.80  # Max Shannon entropy in bits for risky items

    # Known lethal items and their base fatality weights [0.0 - 100.0]
    LETHAL_CANDIDATE_SCORES: dict[str, float] = {
        # Potions
        "potion of death": 100.0,
        "potion of polymorph": 60.0,
        "potion of acid": 40.0,
        "potion of sickness": 50.0,
        # Amulets
        "amulet of strangulation": 100.0,
        "amulet of restful sleep": 50.0,
        # Scrolls
        "scroll of amnesia": 80.0,
        "scroll of fire": 70.0,
        "scroll of create monster": 30.0,
        # Rings
        "ring of polymorph": 60.0,
        "ring of teleportation": 30.0,
    }

    @classmethod
    def calculate_risk_score(cls, action_type: str, item_belief: ItemBeliefState) -> float:
        """
        Computes Risk(a, i) = P(Cursed) * L_CursedLock + sum(p_k * FatalityScore(k))
        """
        action_upper = action_type.upper()
        p_cursed = item_belief.p_cursed

        # Base lock-in penalty for being welded or stuck
        cursed_loss_weight = 40.0 if action_upper == "EQUIP" else 20.0
        risk = p_cursed * cursed_loss_weight

        # Add expected candidate fatality
        if len(item_belief.candidate_identities) > 0 and len(item_belief.identity_probs) > 0:
            for name, prob in zip(item_belief.candidate_identities, item_belief.identity_probs):
                fatality = cls.LETHAL_CANDIDATE_SCORES.get(name.lower(), 0.0)
                risk += prob * fatality

        return risk

    @classmethod
    def can_safely_equip(
        cls,
        item_belief: ItemBeliefState,
        tau_cursed: float = DEFAULT_CURSED_VETO_THRESHOLD,
    ) -> Tuple[bool, str]:
        """
        Vetoes equipping items with non-negligible curse probability or high fatality risk.
        """
        if item_belief.p_cursed > tau_cursed:
            return False, f"VETO: P(Cursed)={item_belief.p_cursed:.2f} > {tau_cursed} (Equip cursed lock risk)"

        # Check for lethal accessories (e.g. amulet of strangulation)
        if item_belief.item_class in ("amulet", "ring"):
            for name, prob in zip(item_belief.candidate_identities, item_belief.identity_probs):
                if name.lower() == "amulet of strangulation" and prob > 0.05:
                    return False, f"VETO: Potential strangulation amulet (P={prob:.2f})"

        return True, "SAFE"

    @classmethod
    def can_safely_quaff(
        cls,
        item_belief: ItemBeliefState,
        tau_cursed: float = DEFAULT_CURSED_VETO_THRESHOLD,
        tau_entropy: float = DEFAULT_IDENTITY_ENTROPY_THRESHOLD,
    ) -> Tuple[bool, str]:
        """
        Vetoes drinking potions if curse probability or identity entropy is unsafe.
        """
        if item_belief.p_cursed > tau_cursed:
            return False, f"VETO: P(Cursed)={item_belief.p_cursed:.2f} > {tau_cursed} (Cursed potion quaff risk)"

        entropy = item_belief.entropy_identity()
        if entropy > tau_entropy and not item_belief.is_formally_identified():
            return False, f"VETO: Identity entropy too high ({entropy:.2f} bits > {tau_entropy})"

        # Check for lethal potions (death, acid, sickness)
        for name, prob in zip(item_belief.candidate_identities, item_belief.identity_probs):
            if name.lower() == "potion of death" and prob > 0.01:
                return False, f"VETO: Potion of death risk (P={prob:.2f})"

        return True, "SAFE"

    @classmethod
    def can_safely_read(
        cls,
        item_belief: ItemBeliefState,
        tau_cursed: float = DEFAULT_CURSED_VETO_THRESHOLD,
        tau_entropy: float = DEFAULT_IDENTITY_ENTROPY_THRESHOLD,
    ) -> Tuple[bool, str]:
        """
        Vetoes reading scrolls if curse probability or identity entropy is unsafe.
        """
        if item_belief.p_cursed > tau_cursed:
            return False, f"VETO: P(Cursed)={item_belief.p_cursed:.2f} > {tau_cursed} (Cursed scroll risk)"

        entropy = item_belief.entropy_identity()
        if entropy > tau_entropy and not item_belief.is_formally_identified():
            return False, f"VETO: Identity entropy too high ({entropy:.2f} bits > {tau_entropy})"

        for name, prob in zip(item_belief.candidate_identities, item_belief.identity_probs):
            if name.lower() == "scroll of amnesia" and prob > 0.05:
                return False, f"VETO: Potential amnesia scroll (P={prob:.2f})"

        return True, "SAFE"
