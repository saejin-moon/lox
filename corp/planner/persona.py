"""
Dynamic Persona Profiler: Turn-0 continuous trait vector derivation.
Eliminates brittle role hardcoding by mapping observed base stats into continuous traits.
"""

from dataclasses import dataclass
import math
import numpy as np
from corp.env.blstats import BottomLineStats
from corp.env.inventory_tracker import InventoryNormalizer


def sigmoid(z: float) -> float:
    """Standard numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


@dataclass(slots=True, frozen=True)
class PersonaTraitVector:
    """
    Continuous 5-dimensional persona embedding theta in [0, 1]^5.
    """
    resilience: float   # T_resilience: Melee aggression vs. fragile kiting preference
    ranged: float       # T_ranged: Projectile / missile preference
    mana: float         # T_mana: Spellcasting frequency and energy rest thresholds
    stealth: float      # T_stealth: Monster avoidance vs. aggressive engagement
    alignment: float    # T_alignment: 1.0=Lawful, 0.5=Neutral, 0.0=Chaotic

    def to_array(self) -> np.ndarray:
        return np.array([
            self.resilience,
            self.ranged,
            self.mana,
            self.stealth,
            self.alignment,
        ], dtype=np.float64)


class PersonaProfiler:
    """
    Derives continuous persona vector from initial Turn 0 observations.
    """

    @classmethod
    def derive_persona(
        cls,
        blstats: BottomLineStats,
        inventory: InventoryNormalizer | None = None,
    ) -> PersonaTraitVector:
        """
        Derives theta in [0, 1]^5 mathematically from stats and initial carried inventory.
        """
        # 1. Resilience
        # T_resilience = sigmoid((BaseHP - 12)/4 + (Con - 12)/3 + (10 - AC)/2)
        base_hp = float(blstats.max_hp)
        con = float(blstats.constitution)
        ac = float(blstats.ac)
        z_resilience = ((base_hp - 12.0) / 4.0) + ((con - 12.0) / 3.0) + ((10.0 - ac) / 2.0)
        t_resilience = sigmoid(z_resilience)

        # 2. Ranged Affinity
        # T_ranged = sigmoid((Dex - 12)/3 + 2.0*LauncherEquipped + 0.5*Count(Daggers))
        dex = float(blstats.dexterity)
        launcher_equipped = 0.0
        dagger_count = 0.0

        if inventory:
            for item in inventory.get_active_items():
                s = item.raw_str.lower()
                if item.equipped and ("bow" in s or "crossbow" in s or "sling" in s):
                    launcher_equipped = 1.0
                if "dagger" in s or "dart" in s or "shuriken" in s:
                    dagger_count += float(item.quantity)

        z_ranged = ((dex - 12.0) / 3.0) + (2.0 * launcher_equipped) + (0.5 * min(10.0, dagger_count))
        t_ranged = sigmoid(z_ranged)

        # 3. Mana Pool
        # T_mana = sigmoid((MaxEnergy - 5)/5 + (Int + Wis - 24)/6)
        max_ene = float(blstats.max_energy)
        intel = float(blstats.intelligence)
        wis = float(blstats.wisdom)
        z_mana = ((max_ene - 5.0) / 5.0) + ((intel + wis - 24.0) / 6.0)
        t_mana = sigmoid(z_mana)

        # 4. Stealth & Evasion
        # T_stealth = sigmoid((Dex - 12)/4 - ArmorWeight/100 + 1.5*StealthIntrinsic)
        # Approximate armor encumbrance
        z_stealth = (dex - 12.0) / 4.0
        t_stealth = sigmoid(z_stealth)

        # 5. Alignment Strictness
        # In NLE blstats, alignment is typically -1=Chaotic, 0=Neutral, 1=Lawful
        # We normalize to [0.0, 1.0]
        align_val = float(blstats.alignment)
        if align_val > 0:
            t_align = 1.0   # Lawful
        elif align_val < 0:
            t_align = 0.0   # Chaotic
        else:
            t_align = 0.5   # Neutral

        return PersonaTraitVector(
            resilience=float(t_resilience),
            ranged=float(t_ranged),
            mana=float(t_mana),
            stealth=float(t_stealth),
            alignment=float(t_align),
        )
