"""
EngraveTestListener: NetHack 3.6.6 wand scratch-testing for candidate pruning.
Enforces the 1% cursed wand explosion interlock.
"""

from corp.epistemic.belief_state import ItemBeliefState


class CursedWandExplosionRiskError(Exception):
    """Raised when an attempt is made to engrave with a potentially cursed wand."""
    pass


class EngraveTestListener:
    """
    Evaluates messages from engraving with a wand to eliminate candidate identities.
    """

    MESSAGE_MAP: list[tuple[str, set[str]]] = [
        ("bugs on the", {"wand of sleep", "wand of death"}),  # "The bugs on the floor stop moving!"
        ("vanishes", {"wand of cancellation", "wand of teleportation", "wand of make invisible", "wand of cold"}),
        ("ice cubes", {"wand of cold"}),
        ("digging", {"wand of digging"}),
        ("gravel flies up", {"wand of digging"}),
        ("wand of fire", {"wand of fire"}),
        ("lightning arcs", {"wand of lightning"}),
        ("bullet holes", {"wand of magic missile"}),
        ("engraving now reads", {"wand of polymorph"}),
        ("slow down", {"wand of slow monster"}),
        ("speed up", {"wand of speed monster"}),
        ("fights your attempt", {"wand of striking"}),
    ]

    INERT_WANDS: set[str] = {
        "wand of locking",
        "wand of opening",
        "wand of probing",
        "wand of undead turning",
        "wand of secret door detection",
        "wand of nothing",
    }

    @classmethod
    def can_safely_engrave_test(cls, item_belief: ItemBeliefState) -> bool:
        """
        NetHack 3.6.0+ Safety Guard:
        Cursed wands have a 1% chance of exploding when used to engrave!
        Wand MUST be verified uncursed or blessed before engrave-testing.
        """
        return item_belief.p_cursed == 0.0

    @classmethod
    def process_engrave_result(
        cls,
        item_belief: ItemBeliefState,
        terminal_message: str,
        enforce_safety: bool = True,
    ) -> list[str]:
        """
        Prunes wand candidate identities based on observed engrave diagnostic message.
        """
        if enforce_safety and not cls.can_safely_engrave_test(item_belief):
            raise CursedWandExplosionRiskError(
                f"Cannot engrave with wand {item_belief.uid}: P(Cursed)={item_belief.p_cursed:.2f} > 0.0"
            )

        msg_lower = terminal_message.lower()

        # Match against specific diagnostic messages
        matched_set = None
        for pattern, wands in cls.MESSAGE_MAP:
            if pattern in msg_lower:
                matched_set = wands
                break

        if matched_set is None:
            # If no special message was produced, wand is inert (or empty)
            matched_set = cls.INERT_WANDS

        # Prune candidate identities
        item_belief.prune_candidates(matched_set)
        return item_belief.candidate_identities
