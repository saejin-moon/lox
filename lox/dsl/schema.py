"""
LOX 2.0 DSL Schema: Vocabulary Whitelist and Safety Invariants.
Defines all allowed predicates, actions, and non-negotiable guardrails.
"""
from __future__ import annotations

from typing import Any, Callable

# Whitelist of allowed predicate identifiers and their valid types
ALLOWED_PREDICATES: dict[str, str] = {
    "hp_frac": "float",
    "energy_frac": "float",
    "depth": "int",
    "turn": "int",
    "turns_on_level": "int",
    "hunger_state": "enum",
    "adjacent_hostile": "bool",
    "can_retreat": "bool",
    "can_safely_pray": "bool",
    "has_healing": "bool",
    "has_carried_food": "bool",
    "floor_corpse_adjacent": "bool",
    "corpse_is_fresh": "bool",
    "stairs_down_known": "bool",
    "floor_explored": "bool",
    "has_unvisited_frontier": "bool",
    "has_unsearched_dead_end": "bool",
    "is_blind": "bool",
}

# Whitelist of allowed action primitives
ALLOWED_ACTIONS: set[str] = {
    "step_to_frontier",
    "step_to_stairs_down",
    "step_to_stairs_up",
    "step_to_dead_end",
    "step_away_from_hostile",
    "melee_attack_hostile",
    "quaff_healing",
    "eat_carried_food",
    "eat_floor_corpse",
    "pray",
    "descend",
    "ascend",
    "search",
    "wait",
}

# Enum constants exposed to conditions
ENUM_CONSTANTS: dict[str, int] = {
    "SATIATED": 0,
    "NORMAL": 1,
    "HUNGRY": 2,
    "WEAK": 3,
    "FAINTING": 4,
}
