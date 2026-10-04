"""
LOX 2.0 DSL Schema: Comprehensive NetHack Vocabulary Whitelist.
Exposes full NetHack action primitives, rich tactical predicates, and constants.
"""
from __future__ import annotations

# Whitelist of allowed predicate identifiers and their valid types
ALLOWED_PREDICATES: dict[str, str] = {
    # Hero Vitals & Stats
    "hp": "int",
    "max_hp": "int",
    "hp_frac": "float",
    "energy": "int",
    "max_energy": "int",
    "energy_frac": "float",
    "ac": "int",
    "level": "int",
    "depth": "int",
    "turn": "int",
    "turns_on_level": "int",
    "gold": "int",
    "hunger_state": "enum",
    "dungeon_branch": "str",
    "is_dead": "bool",

    # Status Effects
    "is_blind": "bool",
    "is_poisoned": "bool",
    "is_confused": "bool",
    "is_stunned": "bool",
    "is_hallucinating": "bool",
    "is_sick": "bool",
    "is_held": "bool",
    "is_slowed": "bool",
    "is_fast": "bool",
    "is_levitating": "bool",
    "is_encumbered": "bool",
    "encumbrance_level": "enum",

    # Tactical Combat Environment
    "adjacent_hostile": "bool",
    "hostile_count_fov": "int",
    "closest_hostile_name": "str",
    "closest_hostile_dist": "float",
    "closest_hostile_pos": "tuple",
    "is_surrounded": "bool",
    "in_corridor": "bool",
    "can_retreat": "bool",
    "standing_on_elbereth": "bool",
    "floating_eye_in_fov": "bool",
    "adjacent_pet": "bool",
    "adjacent_peaceful": "bool",
    "can_safely_pray": "bool",
    "is_fast_dangerous": "bool",
    "gas_spore_in_fov": "bool",
    "adjacent_gas_spore": "bool",
    "adjacent_floating_eye": "bool",
    "hostile_ignores_elbereth": "bool",
    "has_panic_escape": "bool",
    "has_safe_melee_target": "bool",
    "has_active_hostile": "bool",
    "active_hostile_count": "int",

    # Spatial Topology & Navigation
    "stairs_down_known": "bool",
    "stairs_up_known": "bool",
    "stairs_down_pos": "tuple",
    "stairs_up_pos": "tuple",
    "standing_on_stairs_down": "bool",
    "standing_on_stairs_up": "bool",
    "standing_on_dead_end": "bool",
    "has_unvisited_frontier": "bool",
    "has_unsearched_dead_end": "bool",
    "unvisited_frontier_count": "int",
    "floor_explored": "bool",
    "has_nearby_loot": "bool",

    # Dungeon Features & Environment
    "tile_type": "str",
    "in_shop": "bool",
    "in_temple": "bool",
    "is_dark_level": "bool",
    "adjacent_closed_door": "bool",
    "adjacent_open_door": "bool",
    "has_closed_door": "bool",
    "closed_door_in_fov": "bool",
    "door_is_locked": "bool",
    "adjacent_fountain": "bool",
    "standing_on_fountain": "bool",
    "adjacent_altar": "bool",
    "standing_on_altar": "bool",
    "altar_is_aligned": "bool",
    "adjacent_trap": "bool",
    "standing_on_trap": "bool",

    # Nutrition & Corpse Intrinsics
    "has_food": "bool",
    "has_healing": "bool",
    "has_carried_food": "bool",
    "floor_corpse_adjacent": "bool",
    "corpse_is_fresh": "bool",
    "corpse_is_safe": "bool",
    "corpse_is_deadly": "bool",
    "has_poison_res": "bool",
    "can_harvest_poison": "bool",

    # Equipment & Item Intrinsics
    "has_wand_of_teleport": "bool",
    "has_wand_of_digging": "bool",
    "has_offensive_wand": "bool",
    "has_scroll_of_teleport": "bool",
    "has_pick_axe": "bool",
    "has_lamp": "bool",
    "has_unworn_armor": "bool",
    "has_daggers": "bool",
    "weapon_is_cursed": "bool",
    "can_forge_excalibur": "bool",
    "fountain_in_fov": "bool",

    # Epistemic POMDP Belief & Safety Gates
    "untested_buc_count": "int",
    "has_untested_items": "bool",
    "can_safely_wear_armor": "bool",
    "can_safely_quaff_healing": "bool",

    # Strategic Goal Agenda
    "active_goal": "str",
}

# Whitelist of allowed action primitives
ALLOWED_ACTIONS: set[str] = {
    # Movement & Spatial Navigation
    "step_to_frontier",
    "step_to_stairs_down",
    "step_to_stairs_up",
    "step_to_dead_end",
    "step_to_closed_door",
    "step_away_from_hostile",
    "step_to_chokepoint",
    "step_to_loot",
    "step_to_fountain",
    "step_to_altar",
    "step_to",
    "step_direction",
    "descend",
    "ascend",

    # Tactical Combat & Defense
    "melee_attack_hostile",
    "melee_attack",
    "fire_missile",
    "throw_item",
    "throw_dagger",
    "engrave_dust_elbereth",
    "engrave_elbereth",
    "engrave",

    # Physical Environmental Manipulation
    "open_door",
    "kick_closed_door",
    "search",
    "wait",
    "pickup",
    "drop",
    "pay",
    "chat",

    # Nutrition & Safe Corpse Eating
    "eat_carried_food",
    "eat_food",
    "eat_floor_corpse",
    "harvest_poison_res",

    # Divine, Rituals & Epistemic Testing
    "pray",
    "dip_excalibur",
    "dip_in_fountain",
    "test_altar_buc",

    # Equipment & Consumables
    "quaff_healing",
    "quaff",
    "read_scroll",
    "read_scroll_teleport",
    "zap_wand",
    "zap_offensive_wand",
    "zap_wand_teleport",
    "apply_item",
    "wield_weapon",
    "wear_armor",
    "retreat",
    "rest",
    "idle",
}

# Enum constants exposed to conditions
ENUM_CONSTANTS: dict[str, int | str] = {
    # Hunger States
    "SATIATED": 0,
    "NORMAL": 1,
    "HUNGRY": 2,
    "WEAK": 3,
    "FAINTING": 4,

    # Encumbrance States
    "UNENCUMBERED": 0,
    "BURDENED": 1,
    "STRESSED": 2,
    "STRAINED": 3,
    "OVERTAXED": 4,
    "OVERLOADED": 5,

    # Branches
    "BRANCH_DUNGEON": "dungeon",
    "BRANCH_MINES": "mines",
    "BRANCH_SOKOBAN": "sokoban",
    "BRANCH_QUEST": "quest",

    # Strategic Goal Directives
    "GOAL_EXPLORE_FLOOR": "explore_floor",
    "GOAL_COLLECT_POISON_RES": "collect_poison_res",
    "GOAL_FORGE_EXCALIBUR": "forge_excalibur",
    "GOAL_TEST_BUC_ALTAR": "test_buc_altar",
    "GOAL_CLEAR_MINES": "clear_mines",
    "GOAL_SOLVE_SOKOBAN": "solve_sokoban",
    "GOAL_DESCEND_STAIRS": "descend_stairs",
}
