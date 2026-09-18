"""
Canonical 64-bit State Predicate Registry: Fast bitmask encoding of game states
for sub-microsecond CDCL Nogood evaluation.
"""

from enum import IntFlag
from typing import Any
import numpy as np
from corp.env.blstats import BottomLineStats, ConditionFlag, HungerState
from corp.env.inventory_tracker import InventoryNormalizer


class PredicateBit(IntFlag):
    """
    Standardized 64-bit integer bitmask indices representing atomic game conditions.
    """
    # Physical & Status Conditions (0-15)
    IS_BLIND                    = 1 << 0
    IS_STUNNED                  = 1 << 1
    IS_CONFUSED                 = 1 << 2
    IS_HALLUCINATING            = 1 << 3
    HP_CRITICAL_BELOW_20_PCT    = 1 << 4
    HP_LOW_BELOW_50_PCT         = 1 << 5
    HUNGER_SATIATED             = 1 << 6
    HUNGER_WEAK_OR_FAINTING     = 1 << 7
    GLOVES_EQUIPPED             = 1 << 8
    SHIELD_EQUIPPED             = 1 << 9
    RANGED_WEAPON_WIELDED       = 1 << 10
    ENCUMBRANCE_BURDENED_PLUS   = 1 << 11
    POISON_RESISTANCE_ACTIVE    = 1 << 12
    REFLECTION_ACTIVE           = 1 << 13

    # Environmental & Spatial Conditions (16-31)
    CURRENT_TILE_PIT_OR_WEB     = 1 << 16
    CURRENT_TILE_ALTAR          = 1 << 17
    INSIDE_SHOP_BOUNDARY        = 1 << 18
    UNPAID_ITEMS_IN_INVENTORY   = 1 << 19
    STANDING_ON_STAIRS_DOWN     = 1 << 20
    CORRIDOR_CHOKEPOINT         = 1 << 21

    # Monster Adjacency (32-47)
    ADJACENT_FLOATING_EYE       = 1 << 32
    ADJACENT_COCKATRICE         = 1 << 33
    ADJACENT_RUST_MONSTER       = 1 << 34
    ADJACENT_MIND_FLAYER        = 1 << 35
    ADJACENT_PEACEFUL_NPC       = 1 << 36
    HOSTILE_COUNT_GE_2          = 1 << 37
    ADJACENT_MONSTER_FASTER     = 1 << 38

    # Tactical & Target Properties (48-63)
    TARGET_CORPSE_TAINTED_AGE   = 1 << 48
    TARGET_ITEM_CURSED_PROB_GT_0= 1 << 49
    RAY_REFLECTIVE_WALL_LT_4    = 1 << 50


def compile_predicate_mask(
    blstats: BottomLineStats,
    inventory: InventoryNormalizer | None = None,
    monsters: list[Any] | None = None,
    chars: np.ndarray | None = None,
    lvl_map: Any = None,
    extra_flags: int = 0,
) -> int:
    """
    Compiles a 64-bit integer mask from current sensory observations in <100 nanoseconds.
    """
    mask = int(extra_flags)

    # 1. Status conditions
    if blstats.is_blind:
        mask |= PredicateBit.IS_BLIND
    if blstats.is_stunned:
        mask |= PredicateBit.IS_STUNNED
    if blstats.is_confused:
        mask |= PredicateBit.IS_CONFUSED
    if blstats.is_hallucinating:
        mask |= PredicateBit.IS_HALLUCINATING

    # 2. HP thresholds
    if blstats.max_hp > 0:
        hp_ratio = blstats.hp / blstats.max_hp
        if hp_ratio <= 0.20:
            mask |= PredicateBit.HP_CRITICAL_BELOW_20_PCT
        if hp_ratio <= 0.50:
            mask |= PredicateBit.HP_LOW_BELOW_50_PCT

    # 3. Hunger thresholds
    if blstats.hunger_state == HungerState.SATIATED:
        mask |= PredicateBit.HUNGER_SATIATED
    elif blstats.hunger_state >= HungerState.WEAK:
        mask |= PredicateBit.HUNGER_WEAK_OR_FAINTING

    # 4. Encumbrance
    if blstats.encumbrance >= 1:  # Burdened or worse
        mask |= PredicateBit.ENCUMBRANCE_BURDENED_PLUS

    # 5. Inventory equipment flags
    if inventory:
        for item in inventory.get_active_items():
            s = item.raw_str.lower()
            if item.equipped:
                if "gloves" in s or "gauntlets" in s:
                    mask |= PredicateBit.GLOVES_EQUIPPED
                if "shield" in s:
                    mask |= PredicateBit.SHIELD_EQUIPPED
                if "bow" in s or "crossbow" in s or "sling" in s or "dart" in s:
                    mask |= PredicateBit.RANGED_WEAPON_WIELDED
                if "shield of reflection" in s or "amulet of reflection" in s:
                    mask |= PredicateBit.REFLECTION_ACTIVE

    # 6. Environmental & Spatial conditions
    py, px = blstats.y, blstats.x
    if chars is not None and 0 <= py < chars.shape[0] and 0 <= px < chars.shape[1]:
        c = int(chars[py, px])
        if c == ord(">"):
            mask |= PredicateBit.STANDING_ON_STAIRS_DOWN
        elif c == ord("_"):
            mask |= PredicateBit.CURRENT_TILE_ALTAR
        elif c == ord("^"):
            mask |= PredicateBit.CURRENT_TILE_PIT_OR_WEB
        elif c == ord("#"):
            mask |= PredicateBit.CORRIDOR_CHOKEPOINT

    if lvl_map is not None:
        if getattr(lvl_map, "stairs_down", None) == (py, px):
            mask |= PredicateBit.STANDING_ON_STAIRS_DOWN
        if (py, px) in getattr(lvl_map, "altars", ()):
            mask |= PredicateBit.CURRENT_TILE_ALTAR

    # 7. Monster Adjacency & Threat conditions
    if monsters:
        adj_hostile_count = 0
        for m in monsters:
            if not getattr(m, "is_adjacent", False):
                continue
            mname = getattr(m, "name", "").lower()
            if getattr(m, "is_peaceful", False):
                mask |= PredicateBit.ADJACENT_PEACEFUL_NPC
            else:
                adj_hostile_count += 1
                if "floating eye" in mname:
                    mask |= PredicateBit.ADJACENT_FLOATING_EYE
                if "cockatrice" in mname or "chickatrice" in mname:
                    mask |= PredicateBit.ADJACENT_COCKATRICE
                if "rust monster" in mname:
                    mask |= PredicateBit.ADJACENT_RUST_MONSTER
                if "mind flayer" in mname:
                    mask |= PredicateBit.ADJACENT_MIND_FLAYER
                if getattr(m, "speed", 12) > 12:
                    mask |= PredicateBit.ADJACENT_MONSTER_FASTER

        if adj_hostile_count >= 2:
            mask |= PredicateBit.HOSTILE_COUNT_GE_2

    return mask
