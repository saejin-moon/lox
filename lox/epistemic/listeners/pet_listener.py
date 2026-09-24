"""
PetListener: Observes domestic pet hesitation / stepping over floor items to infer beatitude.
"""

from lox.epistemic.belief_state import ItemBeliefState


class PetListener:
    """
    Applies Bayesian updates to item BUC based on pet stepping or hesitation.
    Enforces the comestible appetite exception.
    """

    @staticmethod
    def process_pet_interaction(
        item_belief: ItemBeliefState,
        pet_stepped_willingly: bool,
    ) -> bool:
        """
        Updates item_belief based on pet movement.
        Returns True if inference was valid, False if excluded (e.g. food).
        """
        # Comestible exclusion gate: pets greedily consume food regardless of curse
        if item_belief.item_class.lower() in ("food", "comestible", "corpse", "ration"):
            return False

        if pet_stepped_willingly:
            item_belief.update_buc_pet_step()
            return True
        else:
            item_belief.update_buc_pet_refusal()
            return True
