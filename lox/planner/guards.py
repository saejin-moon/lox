"""
HTNGuards: Domain precondition evaluators and invariant safety checks for method decomposition.
"""

from lox.env.blstats import BottomLineStats, HungerState
from lox.policy.config import PolicyConfig, default_config


class HTNGuards:
    """
    Evaluates preconditions before candidate HTN methods can decompose.
    """

    @staticmethod
    def can_descend(
        blstats: BottomLineStats,
        has_poison_res: bool = False,
        config: PolicyConfig | None = None,
    ) -> bool:
        """Forbids descending staircase into deeper dungeon if unstable."""
        cfg = config or default_config()
        d = cfg.descent
        if blstats.hp < int(blstats.max_hp * d.min_hp_frac):
            return False
        # Do not forbid descent when hungry/weak: descending is essential to find food and prevent starvation stalls
        if blstats.depth >= d.deep_min_depth and not has_poison_res and blstats.hp < int(blstats.max_hp * d.deep_min_hp_frac):
            # Fatal poison stings spike at dlvl 8+; allow descent if healthy (HP >= 70%)
            return False
        if blstats.dungeon_number == 2 and blstats.experience_level < d.mines_min_xl and not has_poison_res:
            # Gnomish Mines: dark levels with lethal gnome wands; forbid descending deeper if XL < 5
            return False
        return True

    @staticmethod
    def should_descend(
        blstats: BottomLineStats,
        stairs_down_known: bool,
        unvisited_count: int = 0,
        turns_spent: int = 0,
        has_adjacent_hostiles: bool = False,
        has_poison_res: bool = False,
        config: PolicyConfig | None = None,
    ) -> bool:
        """
        Determines whether the agent should actively prioritize descending to deeper dungeon levels.
        Aggressive descent triggers:
        1. On Level 1, dive to level 2 once stairs found and healthy (HP >= 70%).
        2. Character level XL >= 2 and healthy (HP >= 65%) with no adjacent hostiles.
        3. Explored >= 75% of level (unvisited_count <= 25) and healthy (HP >= 60%).
        4. Turns spent on this level >= 80 turns.
        5. Hunger >= Hungry (dive to find food).
        """
        cfg = config or default_config()
        d = cfg.descent
        if not stairs_down_known or has_adjacent_hostiles:
            return False
        if not HTNGuards.can_descend(blstats, has_poison_res=has_poison_res, config=config):
            return False

        # 1. Starving or hungry: descend to find food
        if blstats.hunger_state >= HungerState.HUNGRY:
            return True

        # 2. Substantially explored level (unvisited <= 25) and healthy
        if unvisited_count <= d.should_descend_unvisited and blstats.hp >= int(blstats.max_hp * d.should_descend_any_hp_frac):
            return True

        # 3. High turns spent exploring current level (>= 100 turns)
        if turns_spent >= d.should_descend_turns and blstats.hp >= int(blstats.max_hp * d.should_descend_any_hp_frac):
            return True

        # 4. Healthy character with experience level XL >= 2 and moderate turns/exploration
        if blstats.experience_level >= d.should_descend_xl and blstats.hp >= int(blstats.max_hp * d.should_descend_xl_hp_frac) and (turns_spent >= d.should_descend_xl_turns or unvisited_count <= d.should_descend_unvisited):
            return True

        return False

    @staticmethod
    def should_rest_on_stairs(
        blstats: BottomLineStats,
        on_stairs_down: bool,
        config: PolicyConfig | None = None,
    ) -> bool:
        """
        If standing on stairs down but can't descend due to low HP, rest here instead of
        wandering off the stairs (which previously caused aimless 1000+ turn floor stalls).
        Resting is safe when HP is below the descent threshold (45%) but above critical
        triage range (20%), and combat/triage priorities already handled adjacent threats.
        """
        if not on_stairs_down:
            return False
        s = (config or default_config()).survival
        return int(blstats.max_hp * s.rest_stairs_lower_frac) < blstats.hp < int(blstats.max_hp * s.rest_stairs_upper_frac)

    @staticmethod
    def can_pray(
        blstats: BottomLineStats,
        turns_since_prayer: int,
        is_first_prayer: bool = False,
        estimated_luck: int = 0,
        config: PolicyConfig | None = None,
    ) -> bool:
        """
        NetHack 3.6.6 stochastic prayer cooldown model:
        Initial cooldown = 300; subsequent = rnz(350) (mean 454, std 365).
        """
        if estimated_luck < 0:
            return False
        cfg = config or default_config()
        n = cfg.nutrition

        # Major emergency (HP <= 5 or Fainting): timeout safe up to 200
        is_major_emergency = (blstats.hp <= 5 or blstats.hunger_state >= HungerState.FAINTING)
        if is_major_emergency:
            required_turns = n.pray_major_first if is_first_prayer else n.pray_major
            return turns_since_prayer >= required_turns

        # Routine favors threshold: mu + 1.1*sigma
        required_turns = n.pray_routine_first if is_first_prayer else n.pray_routine
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
        if corpse_age > default_config().nutrition.corpse_fresh_turns:
            return False
        return True
