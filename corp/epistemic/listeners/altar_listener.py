"""
AltarListener: Observes divine flash text on altar drops to collapse BUC uncertainty.
"""

from corp.epistemic.belief_state import ItemBeliefState
from corp.env.blstats import BottomLineStats


class AltarListener:
    """
    Applies exact Bayesian likelihood updates to ItemBeliefState based on altar flashes.
    Enforces sensory gates (blindness and hallucination).
    """

    @staticmethod
    def process_drop(
        item_belief: ItemBeliefState,
        blstats: BottomLineStats,
        terminal_message: str,
    ) -> bool:
        """
        Updates item_belief.p_buc if sensory conditions are met.
        Returns True if observation was validly processed, False if sensory gate blocked.
        """
        # 1. Sensory Gate: Cannot observe flashes if blind or hallucinating
        if blstats.is_blind:
            return False
        if blstats.is_hallucinating:
            # Hallucinatory flashes have randomized colors: uncursed items produce no flash,
            # but blessed/cursed cannot be distinguished reliably.
            if "flash" not in terminal_message.lower():
                item_belief.collapse_buc("UNCURSED")
                return True
            return False

        msg_lower = terminal_message.lower()

        # 2. Black light flash -> Cursed
        if "black light" in msg_lower or "flash of black" in msg_lower:
            item_belief.collapse_buc("CURSED")
            return True

        # 3. Amber / Purple flash -> Blessed
        if "amber flash" in msg_lower or "purple flash" in msg_lower or "amber light" in msg_lower or "purple light" in msg_lower:
            item_belief.collapse_buc("BLESSED")
            return True

        # 4. No flash -> Uncursed (provided an item was indeed dropped on an altar)
        # Note: In NetHack, uncursed items on an altar emit no special flash text.
        item_belief.collapse_buc("UNCURSED")
        return True
