"""
LOX Core: Standardized Types and Interfaces.
Pure, lightweight dataclasses without circular dependencies or runtime overhead.
Equipped with rich observation namespaces for Generator and Class-based policy execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any

import numpy as np


class Status(Enum):
    """Execution status returned by Behavior Tree nodes on each tick."""

    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class HungerState(IntEnum):
    """Standardized hunger levels."""

    SATIATED = 0
    NORMAL = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


class EncumbranceState(IntEnum):
    """Standardized encumbrance levels."""

    UNENCUMBERED = 0
    BURDENED = 1
    STRESSED = 2
    STRAINED = 3
    OVERTAXED = 4
    OVERLOADED = 5


class DungeonPhase(str, Enum):
    """High-level progression phases of a complete NetHack ascension campaign."""

    EARLY_RUSH = "early_rush"            # DL 1-4: scavenge armor, daggers, rations, dive
    EXCALIBUR_FORGE = "excalibur_forge"  # XL >= 5: seek fountain and forge Excalibur
    SOKOBAN = "sokoban"                  # DL 6-10: enter Sokoban, clear for reflection/holding
    MID_GAME_DIVE = "mid_game_dive"      # DL 10-20: Medusa lair, Castle approach
    CASTLE_BREACH = "castle_breach"      # DL 25-29: breach drawbridge, acquire Wand of Wishing
    QUEST = "quest"                      # XL 14: enter Quest, chat with leader, defeat nemesis
    GEHENNOM_CRAWL = "gehennom_crawl"    # DL 30-45: Vlad's Tower, Wizard's Tower, Vibrating Square
    INVOCATION = "invocation"            # Vibrating Square: Bell, Book, Candelabrum
    ASCENSION_RUN = "ascension_run"      # Has Amulet: reverse climb to Astral High Altar


@dataclass(slots=True)
class HeroState:
    """Hero attributes extracted from observation for fast predicate evaluation."""

    y: int = 0
    x: int = 0
    hp: int = 1
    max_hp: int = 1
    energy: int = 0
    max_energy: int = 0
    ac: int = 10
    level: int = 1  # XL (Experience level)
    depth: int = 1  # Dungeon depth
    dungeon_num: int = 0  # 0 = Doom, 1 = Mines, 2 = Sokoban, 3 = Quest
    gold: int = 0
    score: int = 0
    turn: int = 0
    turns_on_level: int = 0
    hunger_state: HungerState = HungerState.NORMAL
    dungeon_branch: str = "dungeon"
    is_dead: bool = False

    # Status flags
    is_blind: bool = False
    is_confused: bool = False
    is_stunned: bool = False
    is_hallucinating: bool = False
    is_poisoned: bool = False
    is_sick: bool = False
    has_poison_res: bool = False
    has_magic_res: bool = False
    has_reflection: bool = False
    can_enhance_skills: bool = False
    can_pray: bool = True

    @property
    def can_safely_pray(self) -> bool:
        return self.can_pray

    @property
    def hp_frac(self) -> float:
        return self.hp / max(1, self.max_hp)

    @property
    def energy_frac(self) -> float:
        return self.energy / max(1, self.max_energy)


@dataclass(slots=True)
class HeroStatus:
    """Detailed status effect flags."""

    is_blind: bool = False
    is_poisoned: bool = False
    is_confused: bool = False
    is_stunned: bool = False
    is_hallucinating: bool = False
    is_sick: bool = False
    is_held: bool = False
    is_slowed: bool = False
    is_fast: bool = False
    is_levitating: bool = False
    is_encumbered: bool = False
    encumbrance_level: EncumbranceState = EncumbranceState.UNENCUMBERED


@dataclass(slots=True)
class Item:
    """Item representation."""

    slot: str
    name: str
    glyph: int = 0
    quantity: int = 1
    buc: str = "uncursed"  # "blessed", "uncursed", "cursed", "unknown"
    is_equipped: bool = False
    category: str = (
        "unknown"  # "weapon", "armor", "food", "potion", "scroll", "wand", "tool"
    )


class InventoryView(list):
    """List-compatible inventory with fast query helpers for policies."""

    def __init__(
        self,
        items: list[Item] | None = None,
        failed_armor_slots: set[str] | None = None,
    ):
        super().__init__(items or [])
        self.failed_armor_slots: set[str] = set(failed_armor_slots or [])

    @property
    def items(self) -> list[Item]:
        return list(self)

    @staticmethod
    def _is_safe_food_item(it: Item) -> bool:
        if it.category != "food":
            return False
        n = it.name.lower()
        if "lichen corpse" in n or "lizard corpse" in n:
            return True
        if "corpse" in n or "egg" in n or "tripe" in n:
            return False
        return True

    @property
    def has_food(self) -> bool:
        return any(self._is_safe_food_item(it) for it in self)

    @property
    def has_carried_food(self) -> bool:
        return self.has_food

    @property
    def has_healing(self) -> bool:
        return any(
            it.category == "potion"
            and any(k in it.name.lower() for k in ["heal", "extra heal"])
            for it in self
        )

    @property
    def has_wand_of_teleport(self) -> bool:
        return any(
            it.category == "wand" and "teleport" in it.name.lower() for it in self
        )

    def get_wand_of_teleport_slot(self) -> str | None:
        for it in self:
            if it.category == "wand" and "teleport" in it.name.lower():
                return it.slot
        return None

    @property
    def has_offensive_wand(self) -> bool:
        return any(
            it.category == "wand"
            and any(
                k in it.name.lower()
                for k in (
                    "striking",
                    "sleep",
                    "fire",
                    "cold",
                    "lightning",
                    "magic missile",
                    "death",
                    "slow monster",
                )
            )
            for it in self
        )

    def get_offensive_wand_slot(self) -> str | None:
        for it in self:
            if it.category == "wand" and any(
                k in it.name.lower()
                for k in (
                    "striking",
                    "sleep",
                    "fire",
                    "cold",
                    "lightning",
                    "magic missile",
                    "death",
                    "slow monster",
                )
            ):
                return it.slot
        return None

    @property
    def has_wand_of_digging(self) -> bool:
        return any(
            it.category == "wand" and "digging" in it.name.lower() for it in self
        )

    @property
    def has_scroll_of_teleport(self) -> bool:
        return any(
            it.category == "scroll" and "teleport" in it.name.lower() for it in self
        )

    def get_scroll_of_teleport_slot(self) -> str | None:
        for it in self:
            if it.category == "scroll" and "teleport" in it.name.lower():
                return it.slot
        return None

    @property
    def has_pick_axe(self) -> bool:
        return any(
            "pick-axe" in it.name.lower() or "pickaxe" in it.name.lower() for it in self
        )

    @property
    def has_lamp(self) -> bool:
        return any(
            "lamp" in it.name.lower() or "lantern" in it.name.lower() for it in self
        )

    @property
    def weapon_is_cursed(self) -> bool:
        for it in self:
            if it.category == "weapon" and it.is_equipped and it.buc == "cursed":
                return True
        return False

    def get_food_slot(self) -> str | None:
        for it in self:
            if self._is_safe_food_item(it):
                return it.slot
        return None

    def get_healing_slot(self) -> str | None:
        for it in self:
            if it.category == "potion" and any(
                k in it.name.lower() for k in ["heal", "extra heal"]
            ):
                return it.slot
        return None

    def get_weapon_slot(self) -> str | None:
        for it in self:
            if it.category == "weapon":
                return it.slot
        return None

    @property
    def has_daggers(self) -> bool:
        return any(
            (
                it.category == "weapon"
                and any(
                    k in it.name.lower()
                    for k in ("dagger", "dart", "arrow", "shuriken", "spear", "javelin")
                )
            )
            or (
                "rock" in it.name.lower()
                and it.category in ("gem", "weapon", "unknown")
            )
            for it in self
        )

    def get_dagger_slot(self) -> str | None:
        for it in self:
            if it.category == "weapon" and "dagger" in it.name.lower():
                return it.slot
        for it in self:
            if any(
                k in it.name.lower()
                for k in ("dart", "arrow", "shuriken", "rock", "spear", "javelin")
            ):
                return it.slot
        return None

    def find_items_by_category(self, cat: str) -> list[Item]:
        return [it for it in self if it.category == cat]

    @property
    def has_worn_cloak(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(k in n for k in ("cloak", "apron", "cape", "robe")):
                    return True
        return False

    @property
    def has_worn_helmet(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(k in n for k in ("helmet", "helm", "hat", "cap", "coif")):
                    return True
        return False

    @property
    def has_worn_gloves(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(k in n for k in ("gloves", "gauntlets")):
                    return True
        return False

    @property
    def has_worn_boots(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(k in n for k in ("boots", "shoes")):
                    return True
        return False

    @property
    def has_worn_shield(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                if "shield" in it.name.lower():
                    return True
        return False

    @property
    def has_unworn_armor(self) -> bool:
        return self.get_unworn_armor_slot() is not None

    def get_unworn_armor_slot(self) -> str | None:
        has_body = self.has_worn_body_armor
        has_helm = self.has_worn_helmet
        has_boots = self.has_worn_boots
        has_cloak = self.has_worn_cloak
        has_gloves = self.has_worn_gloves
        has_shield = self.has_worn_shield

        # Prioritize empty slots in order: Body > Helm > Boots > Cloak > Gloves > Shield
        slot_priorities = [
            (not has_body, ("mail", "suit", "coat", "cuirass", "jacket", "plate", "leather armor", "dragon scale")),
            (not has_helm, ("helmet", "helm", "hat", "cap", "coif")),
            (not has_boots, ("boots", "shoes")),
            (not has_cloak, ("cloak", "apron", "cape", "robe")),
            (not has_gloves, ("gloves", "gauntlets")),
            (not has_shield, ("shield",)),
        ]

        for needed, keywords in slot_priorities:
            if not needed:
                continue
            for it in self:
                if (
                    it.category == "armor"
                    and not it.is_equipped
                    and it.slot not in self.failed_armor_slots
                ):
                    n = it.name.lower()
                    if any(k in n for k in keywords):
                        return it.slot

        # Fallback to any unworn armor that doesn't duplicate an already equipped slot
        for it in self:
            if (
                it.category == "armor"
                and not it.is_equipped
                and it.slot not in self.failed_armor_slots
            ):
                n = it.name.lower()
                if has_cloak and any(k in n for k in ("cloak", "apron", "cape", "robe")):
                    continue
                if has_body and any(k in n for k in ("mail", "suit", "coat", "cuirass", "jacket", "plate", "leather armor", "dragon scale")):
                    continue
                if has_helm and any(k in n for k in ("helmet", "helm", "hat", "cap", "coif")):
                    continue
                if has_boots and any(k in n for k in ("boots", "shoes")):
                    continue
                if has_gloves and any(k in n for k in ("gloves", "gauntlets")):
                    continue
                if has_shield and "shield" in n:
                    continue
                return it.slot
        return None

    @property
    def has_worn_body_armor(self) -> bool:
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(
                    k in n
                    for k in (
                        "mail",
                        "suit",
                        "coat",
                        "cuirass",
                        "jacket",
                        "plate",
                        "leather armor",
                    )
                ):
                    return True
        return False

    @property
    def has_unworn_body_armor(self) -> bool:
        for it in self:
            if (
                it.category == "armor"
                and not it.is_equipped
                and it.slot not in self.failed_armor_slots
            ):
                n = it.name.lower()
                if any(
                    k in n
                    for k in (
                        "mail",
                        "suit",
                        "coat",
                        "cuirass",
                        "jacket",
                        "plate",
                        "leather armor",
                    )
                ):
                    return True
        return False

    def get_unworn_body_armor_slot(self) -> str | None:
        for it in self:
            if (
                it.category == "armor"
                and not it.is_equipped
                and it.slot not in self.failed_armor_slots
            ):
                n = it.name.lower()
                if any(
                    k in n
                    for k in (
                        "mail",
                        "suit",
                        "coat",
                        "cuirass",
                        "jacket",
                        "plate",
                        "leather armor",
                    )
                ):
                    return it.slot
        return None

    def get_superior_body_armor_slot(self) -> tuple[str, str] | None:
        """
        Returns (worn_slot, unworn_superior_slot) if there is an unworn body armor
        in inventory with higher base AC tier than the currently worn body armor.
        """
        tiers = {
            "crystal plate": 8,
            "dragon scale": 9,
            "plate mail": 7,
            "splint mail": 6,
            "banded mail": 6,
            "dwarvish mithril": 5,
            "elven mithril": 5,
            "mithril": 5,
            "chain mail": 4,
            "scale mail": 4,
            "cuirass": 4,
            "ring mail": 3,
            "studded leather": 3,
            "leather armor": 2,
            "leather jacket": 1,
        }
        worn_body_item = None
        for it in self:
            if it.category == "armor" and it.is_equipped:
                n = it.name.lower()
                if any(
                    k in n
                    for k in ("mail", "suit", "coat", "cuirass", "jacket", "plate")
                ):
                    worn_body_item = it
                    break

        worn_tier = 0
        worn_slot = None
        if worn_body_item:
            worn_slot = worn_body_item.slot
            n = worn_body_item.name.lower()
            for k, val in tiers.items():
                if k in n:
                    worn_tier = max(worn_tier, val)

        best_unworn_slot = None
        best_unworn_tier = worn_tier

        for it in self:
            if (
                it.category == "armor"
                and not it.is_equipped
                and it.slot not in self.failed_armor_slots
            ):
                n = it.name.lower()
                for k, val in tiers.items():
                    if k in n and val > best_unworn_tier:
                        best_unworn_tier = val
                        best_unworn_slot = it.slot

        if best_unworn_slot and worn_slot and best_unworn_tier > worn_tier:
            return (worn_slot, best_unworn_slot)
        return None

    @property
    def has_bag_of_holding(self) -> bool:
        return any(
            it.category in ("tool", "container") and "bag of holding" in it.name.lower()
            for it in self
        )

    @property
    def has_speed_boots(self) -> bool:
        return any(
            it.category == "armor" and "speed boots" in it.name.lower() for it in self
        )

    @property
    def has_gray_dragon_scale_mail(self) -> bool:
        return any(
            it.category == "armor" and "gray dragon scale mail" in it.name.lower()
            for it in self
        )

    @property
    def has_gdsm(self) -> bool:
        return self.has_gray_dragon_scale_mail

    @property
    def has_silver_dragon_scale_mail(self) -> bool:
        return any(
            it.category == "armor" and "silver dragon scale mail" in it.name.lower()
            for it in self
        )

    @property
    def has_sdsm(self) -> bool:
        return self.has_silver_dragon_scale_mail

    @property
    def has_wand_of_striking(self) -> bool:
        return any(
            it.category == "wand" and "striking" in it.name.lower() for it in self
        )

    @property
    def has_wand_of_wishing(self) -> bool:
        return any(
            it.category == "wand" and "wishing" in it.name.lower() for it in self
        )

    @property
    def has_magic_marker(self) -> bool:
        return any(
            it.category in ("tool", "unknown") and "magic marker" in it.name.lower()
            for it in self
        )

    @property
    def has_unicorn_horn(self) -> bool:
        return any(
            it.category in ("tool", "weapon") and "unicorn horn" in it.name.lower()
            for it in self
        )

    def get_striking_slot(self) -> str | None:
        for it in self:
            if it.category == "wand" and "striking" in it.name.lower():
                return it.slot
        return None

    def get_unicorn_horn_slot(self) -> str | None:
        for it in self:
            if it.category in ("tool", "weapon") and "unicorn horn" in it.name.lower():
                return it.slot
        return None

    def get_bag_of_holding_slot(self) -> str | None:
        for it in self:
            if it.category in ("tool", "container") and "bag of holding" in it.name.lower():
                return it.slot
        return None

    def get_wand_of_wishing_slot(self) -> str | None:
        for it in self:
            if it.category == "wand" and "wishing" in it.name.lower():
                return it.slot
        return None

    @property
    def has_amulet_of_yendor(self) -> bool:
        return any("amulet of yendor" in it.name.lower() for it in self)

    def get_amulet_of_yendor_slot(self) -> str | None:
        for it in self:
            if "amulet of yendor" in it.name.lower():
                return it.slot
        return None

    @property
    def has_blindfold(self) -> bool:
        return any(
            it.category in ("tool", "unknown")
            and any(k in it.name.lower() for k in ("blindfold", "towel"))
            for it in self
        )

    def get_blindfold_slot(self) -> str | None:
        for it in self:
            if it.category in ("tool", "unknown") and any(
                k in it.name.lower() for k in ("blindfold", "towel")
            ):
                return it.slot
        return None

    @property
    def has_corpse(self) -> bool:
        return any("corpse" in it.name.lower() for it in self)

    def get_corpse_slot(self) -> str | None:
        for it in self:
            if "corpse" in it.name.lower():
                return it.slot
        return None

    @property
    def has_athame(self) -> bool:
        return any(
            it.category == "weapon"
            and any(k in it.name.lower() for k in ("athame", "dagger"))
            for it in self
        )

    def get_athame_slot(self) -> str | None:
        for it in self:
            if it.category == "weapon" and any(
                k in it.name.lower() for k in ("athame", "dagger")
            ):
                return it.slot
        return None

    @property
    def has_burn_wand(self) -> bool:
        return any(
            it.category == "wand"
            and any(k in it.name.lower() for k in ("fire", "lightning", "digging"))
            for it in self
        )

    def get_burn_wand_slot(self) -> str | None:
        for it in self:
            if it.category == "wand" and any(
                k in it.name.lower() for k in ("fire", "lightning", "digging")
            ):
                return it.slot
        return None

    @property
    def has_bell_of_opening(self) -> bool:
        return any(
            "bell of opening" in it.name.lower()
            or ("bell" in it.name.lower() and it.category in ("tool", "unknown"))
            for it in self
        )

    def get_bell_slot(self) -> str | None:
        for it in self:
            if "bell of opening" in it.name.lower() or (
                "bell" in it.name.lower() and it.category in ("tool", "unknown")
            ):
                return it.slot
        return None

    @property
    def has_book_of_the_dead(self) -> bool:
        return any(
            "book of the dead" in it.name.lower()
            or ("book" in it.name.lower() and "dead" in it.name.lower())
            for it in self
        )

    def get_book_slot(self) -> str | None:
        for it in self:
            if "book of the dead" in it.name.lower() or (
                "book" in it.name.lower() and "dead" in it.name.lower()
            ):
                return it.slot
        return None

    @property
    def has_candelabrum(self) -> bool:
        return any("candelabrum" in it.name.lower() for it in self)

    def get_candelabrum_slot(self) -> str | None:
        for it in self:
            if "candelabrum" in it.name.lower():
                return it.slot
        return None

    @property
    def candle_count(self) -> int:
        return sum(it.quantity for it in self if "candle" in it.name.lower())

    @property
    def dagger_count(self) -> int:
        return sum(
            it.quantity
            for it in self
            if (
                it.category == "weapon"
                and any(
                    k in it.name.lower()
                    for k in ("dagger", "dart", "arrow", "shuriken", "spear", "javelin")
                )
            )
            or (
                "rock" in it.name.lower()
                and it.category in ("gem", "weapon", "unknown")
            )
        )

    @property
    def food_count(self) -> int:
        return sum(it.quantity for it in self if self._is_safe_food_item(it))

    @property
    def potion_count(self) -> int:
        return sum(it.quantity for it in self if it.category == "potion")

    @property
    def scroll_count(self) -> int:
        return sum(it.quantity for it in self if it.category == "scroll")

    @property
    def has_unidentified_potion(self) -> bool:
        return self.get_unidentified_potion_slot() is not None

    def get_unidentified_potion_slot(self) -> str | None:
        for it in self:
            if it.category == "potion":
                n = it.name.lower()
                if "potion of " not in n:
                    return it.slot
        return None

    @property
    def has_unidentified_scroll(self) -> bool:
        return self.get_unidentified_scroll_slot() is not None

    def get_unidentified_scroll_slot(self) -> str | None:
        for it in self:
            if it.category == "scroll":
                n = it.name.lower()
                if "scroll of " not in n:
                    return it.slot
        return None

    @property
    def equipped_weapon_name(self) -> str:
        for it in self:
            if it.category == "weapon" and it.is_equipped:
                return it.name
        return "bare hands"


@dataclass(slots=True)
class CombatView:
    """Tactical combat situational awareness."""

    adjacent_hostile: bool = False
    hostile_count_fov: int = 0
    closest_hostile_name: str = ""
    closest_hostile_dist: float = 999.0
    closest_hostile_pos: tuple[int, int] | None = None
    is_surrounded: bool = False
    in_corridor: bool = False
    can_retreat: bool = True
    standing_on_elbereth: bool = False
    floating_eye_in_fov: bool = False
    adjacent_pet: bool = False
    adjacent_peaceful: bool = False
    is_fast_dangerous: bool = False
    is_pack_threat: bool = False
    is_corrosive_target: bool = False
    is_heavy_weapon_threat: bool = False
    gas_spore_in_fov: bool = False
    adjacent_gas_spore: bool = False
    adjacent_floating_eye: bool = False
    hostile_ignores_elbereth: bool = False
    has_panic_escape: bool = False
    has_safe_melee_target: bool = False
    has_active_hostile: bool = False
    active_hostile_count: int = 0
    adjacent_monsters: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SpatialView:
    """Spatial dungeon topology and navigation state."""

    stairs_down_known: bool = False
    stairs_up_known: bool = False
    stairs_down_pos: tuple[int, int] | None = None
    stairs_up_pos: tuple[int, int] | None = None
    standing_on_stairs_down: bool = False
    standing_on_stairs_up: bool = False
    standing_on_elbereth: bool = False
    standing_on_dead_end: bool = False
    has_unvisited_frontier: bool = False
    has_unsearched_dead_end: bool = False
    unvisited_frontier_count: int = 0
    dead_ends_count: int = 0
    target_pos: tuple[int, int] | None = None
    floor_explored: bool = False
    has_nearby_loot: bool = False
    nearby_loot_pos: tuple[int, int] | None = None
    should_descend_urgently: bool = False


@dataclass(slots=True)
class DungeonView:
    """Dungeon features, level properties, and branch context."""

    tile_type: str = "room"  # "room", "corridor", "doorway", "fountain", "altar", "trap", "stairs_down", "stairs_up"
    in_shop: bool = False
    in_temple: bool = False
    is_dark_level: bool = False
    dungeon_branch: str = "dungeon"  # "dungeon", "mines", "sokoban", "quest"
    adjacent_closed_door: bool = False
    adjacent_open_door: bool = False
    has_closed_door: bool = False
    closed_door_in_fov: bool = False
    closest_door_pos: tuple[int, int] | None = None
    door_is_locked: bool = False
    adjacent_fountain: bool = False
    standing_on_fountain: bool = False
    fountain_in_fov: bool = False
    closest_fountain_pos: tuple[int, int] | None = None
    adjacent_altar: bool = False
    standing_on_altar: bool = False
    altar_is_aligned: bool = False
    adjacent_trap: bool = False
    standing_on_trap: bool = False
    can_forge_excalibur: bool = False
    has_known_fountain: bool = False
    closest_fountain_depth: int | None = None
    known_fountain_depths: list[int] = field(default_factory=list)
    can_harvest_poison: bool = False
    is_sokoban: bool = False
    has_boulders: bool = False
    drawbridge_in_fov: bool = False
    closest_drawbridge_pos: tuple[int, int] | None = None
    has_priest: bool = False
    adjacent_priest: bool = False
    priest_pos: tuple[int, int] | None = None
    can_donate_to_priest: bool = False
    can_sacrifice: bool = False
    can_solve_sokoban: bool = False
    has_sokoban_entrance: bool = False
    sokoban_entrance_in_fov: bool = False
    sokoban_entrance_pos: tuple[int, int] | None = None
    can_breach_drawbridge: bool = False
    can_tunnel_gehennom: bool = False
    standing_on_vibrating_square: bool = False
    can_perform_invocation: bool = False
    # Endgame & Ascension Milestones
    has_quest_portal: bool = False
    has_plane_portal: bool = False
    standing_on_quest_portal: bool = False
    standing_on_plane_portal: bool = False
    standing_on_high_altar: bool = False
    quest_portal_pos: tuple[int, int] | None = None
    plane_portal_pos: tuple[int, int] | None = None
    can_wish: bool = False
    can_chat_with_leader: bool = False
    phase: str = DungeonPhase.EARLY_RUSH.value
    is_castle_level: bool = False
    is_sokoban_level: bool = False


@dataclass(slots=True)
class FloorCorpse:
    """Tracked floor corpse."""

    name: str = ""
    y: int = 0
    x: int = 0
    drop_turn: int = 0
    age_turns: int = 0
    is_poisonous: bool = False
    is_deadly: bool = False
    is_fresh: bool = True

    @property
    def is_safe(self) -> bool:
        return self.is_fresh and not self.is_poisonous and not self.is_deadly


@dataclass
class EpistemicView:
    """Belief state over latent properties (BUC, safe-gates, identity)."""

    untested_buc_count: int = 0
    has_untested_items: bool = False
    can_safely_wear_armor: bool = True
    can_safely_quaff_healing: bool = True
    items_belief: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgendaView:
    """Read-only view of the active strategic agenda exposed to policies."""

    active_goal: str = "explore_floor"
    goal_stack: list[str] = field(default_factory=lambda: ["explore_floor"])
    dungeon_phase: str = "early_rush"

    @property
    def phase(self) -> str:
        return self.dungeon_phase

    @property
    def current_goal(self) -> str:
        return self.active_goal

    def is_active(self, goal_name: Any) -> bool:
        target = goal_name.value if hasattr(goal_name, "value") else str(goal_name)
        return self.active_goal == target


@dataclass
class AltarRecord:
    y: int
    x: int
    alignment: str = "unaligned"  # "lawful", "neutral", "chaotic", "unaligned"


@dataclass
class FountainRecord:
    y: int
    x: int
    is_active: bool = True


@dataclass
class RunMemory:
    """Persistent, cross-floor episodic memory maintained across the entire episode."""

    altars: dict[int, list[AltarRecord]] = field(default_factory=dict)
    fountains: dict[int, list[FountainRecord]] = field(default_factory=dict)
    shops: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    stairs_down: dict[int, tuple[int, int]] = field(default_factory=dict)
    stairs_up: dict[int, tuple[int, int]] = field(default_factory=dict)
    branch_stairs: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # "mines", "sokoban" -> (dnum, dlevel, dl)
    portals: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # "quest", "plane"
    stashes: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    visited_depths: set[int] = field(default_factory=set)
    milestones_achieved: set[str] = field(default_factory=set)


@dataclass
class Observation:
    """Standardized environment observation across all domains."""

    chars: np.ndarray  # 2D character grid (uint8)
    glyphs: np.ndarray | None  # 2D glyph grid (int16/int32)
    hero: HeroState  # Hero status
    inventory: InventoryView = field(default_factory=InventoryView)
    status: HeroStatus = field(default_factory=HeroStatus)
    combat: CombatView = field(default_factory=CombatView)
    spatial: SpatialView = field(default_factory=SpatialView)
    dungeon: DungeonView = field(default_factory=DungeonView)
    epistemic: EpistemicView = field(default_factory=EpistemicView)
    agenda: AgendaView = field(default_factory=AgendaView)
    memory: RunMemory = field(default_factory=RunMemory)
    corpses: list[FloorCorpse] = field(default_factory=list)
    message: str = ""  # Last in-game message text
    raw_obs: Any = None  # Original environment observation dict

    @property
    def floor_corpse_adjacent(self) -> bool:
        return any(
            max(abs(c.y - self.hero.y), abs(c.x - self.hero.x)) <= 1
            for c in self.corpses
        )

    @property
    def corpse_is_safe(self) -> bool:
        return any(c.is_safe for c in self.corpses)

    @property
    def corpse_is_fresh(self) -> bool:
        return any(c.is_fresh for c in self.corpses)

    @property
    def corpse_is_deadly(self) -> bool:
        return any(c.is_deadly for c in self.corpses)


@dataclass(slots=True, frozen=True)
class Action:
    """Atomic or composite action emitted by policy."""

    name: str
    direction: tuple[int, int] | None = None  # (dy, dx)
    char: str | None = None  # Direct command character (e.g. ">", "<", "e", "q")
    slot: str | None = None  # Inventory slot letter (e.g. "a", "b")
    target_pos: tuple[int, int] | None = None
    count: int = 1  # Repeat count
    subroutine: str = ""  # Policy goal/subroutine name
    target_depth: int | None = None  # Target depth for backtrack_to_depth
    goal_name: str | None = None  # Active goal name for set_strategic_goal
    extra: dict[str, Any] = field(default_factory=dict)
