"""
LOX 2.0 DSL Schema: Vocabulary Whitelist and Safety Invariants.
Defines all allowed predicates, actions, and non-negotiable guardrails for full ascension runs.
"""
from __future__ import annotations

from typing import Any, Callable

# Whitelist of allowed predicate identifiers and their valid types
ALLOWED_PREDICATES: dict[str, str] = {
    # Hero Vitals & Stats
    "hp_frac": "float",
    "energy_frac": "float",
    "depth": "int",
    "turn": "int",
    "turns_on_level": "int",
    "experience_level": "int",
    "gold": "int",
    "hunger_state": "enum",
    # Status Effects
    "is_blind": "bool",
    "is_poisoned": "bool",
    # Tactical Combat Environment
    "adjacent_hostile": "bool",
    "hostile_count_fov": "int",
    "is_surrounded": "bool",
    "in_corridor": "bool",
    "can_retreat": "bool",
    "standing_on_elbereth": "bool",
    # Divine & Recovery
    "can_safely_pray": "bool",
    "has_healing": "bool",
    # Physical / Environmental
    "adjacent_closed_door": "bool",
    "adjacent_fountain": "bool",
    "stairs_down_known": "bool",
    "stairs_up_known": "bool",
    "standing_on_stairs_down": "bool",
    "standing_on_stairs_up": "bool",
    "floor_explored": "bool",
    "has_unvisited_frontier": "bool",
    "has_unsearched_dead_end": "bool",
    # Nutrition & Corpse Intrinsics
    "has_carried_food": "bool",
    "floor_corpse_adjacent": "bool",
    "corpse_is_fresh": "bool",
    "corpse_is_safe": "bool",
    "has_poison_res": "bool",
    # Artifacts & Special
    "can_forge_excalibur": "bool",
}

# Whitelist of allowed action primitives
ALLOWED_ACTIONS: set[str] = {
    # Movement & Spatial Navigation
    "step_to_frontier",
    "step_to_stairs_down",
    "step_to_stairs_up",
    "step_to_dead_end",
    "step_away_from_hostile",
    "step_to_chokepoint",
    "step_to_fountain",
    "descend",
    "ascend",
    # Tactical Combat & Defense
    "melee_attack_hostile",
    "engrave_elbereth",
    # Physical Environmental Manipulation
    "open_door",
    "kick_closed_door",
    "search",
    "wait",
    # Nutrition & Safe Corpse Eating
    "eat_carried_food",
    "eat_floor_corpse",
    # Divine & Rituals
    "pray",
    "dip_excalibur",
    # Equipment & Consumables
    "quaff_healing",
    "wield_weapon",
    "wear_armor",
}

# Enum constants exposed to conditions
ENUM_CONSTANTS: dict[str, int] = {
    "SATIATED": 0,
    "NORMAL": 1,
    "HUNGRY": 2,
    "WEAK": 3,
    "FAINTING": 4,
}
