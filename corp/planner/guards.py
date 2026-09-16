"""
HTNGuards: Domain precondition evaluators and invariant safety checks for method decomposition.
"""

from corp.env.blstats import BottomLineStats, HungerState


class HTNGuards:
    """
    Evaluates preconditions before candidate HTN methods can decompose.
    """

    @staticmethod
    def can_descend(blstats: BottomLineStats, has_poison_res: bool = False) -> bool:
        """Forbids descending staircase into deeper dungeon if unstable."""
        if blstats.hp < int(blstats.max_hp * 0.50):
            return False
        if blstats.hunger_state >= HungerState.WEAK:
            return False
        if blstats.depth >= 8 and not has_poison_res:
            # Fatal poison stings (soldier ants, giant bees) spike at dlvl 8+
            return False
        return True

    @staticmethod
    def can_pray(
        blstats: BottomLineStats,
        turns_since_prayer: int,
        is_first_prayer: bool = False,
        estimated_luck: int = 0,
    ) -> bool:
        """
        NetHack 3.6.6 stochastic prayer cooldown model:
        Initial cooldown = 300; subsequent = rnz(350) (mean 454, std 365).
        """
        if estimated_luck < 0:
            return False

        # Major emergency (HP <= 5 or Fainting): timeout safe up to 200
        is_major_emergency = (blstats.hp <= 5 or blstats.hunger_state >= HungerState.FAINTING)
        if is_major_emergency:
            required_turns = 101 if is_first_prayer else 450
            return turns_since_prayer >= required_turns

        # Routine favors threshold: mu + 1.1*sigma
        required_turns = 301 if is_first_prayer else 850
        return turns_since_prayer >= required_turns

    @staticmethod
    def safe_to_eat(
        monster_type: str,
        corpse_age: int,
        is_poisonous: bool = False,
        has_poison_res: bool = False,
        player_race: str = "",
        corpse_race: str = "",
    ) -> bool:
        """
        NetHack 3.6.6 corpse consumption rules:
        - Lichen & lizard never rot and are safe indefinitely.
        - Tainted cutoff strictly <= 25 turns for general mortal corpses.
        """
        m_lower = monster_type.lower()
        if m_lower in ("lichen", "lizard"):
            return True
        if m_lower in ("cockatrice", "chickatrice", "medusa"):
            return False  # Instant petrification
        if m_lower == "green slime":
            return False  # Fatal sliming
        if is_poisonous and not has_poison_res:
            return False  # Poison roll
        if player_race and corpse_race and player_race.lower() == corpse_race.lower():
            return False  # Cannibalism (luck penalty + telepathy loss)

        # Conservative food safety threshold based on rottenness math
        if corpse_age > 25:
            return False
        return True
