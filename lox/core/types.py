"""
LOX 2.0 Core: Standardized Types and Interfaces.
Pure, lightweight dataclasses without circular dependencies or runtime overhead.
Equipped with rich observation namespaces for Generator and Class-based policy execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable
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
    level: int = 1         # XL (Experience level)
    depth: int = 1         # Dungeon depth
    dungeon_num: int = 0   # 0 = Doom, 1 = Mines, 2 = Sokoban, 3 = Quest
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
    category: str = "unknown"  # "weapon", "armor", "food", "potion", "scroll", "wand", "tool"


class InventoryView(list):
    """List-compatible inventory with fast query helpers for policies."""

    def __init__(self, items: list[Item] | None = None, failed_armor_slots: set[str] | None = None):
        super().__init__(items or [])
        self.failed_armor_slots: set[str] = set(failed_armor_slots or [])

    @property
    def items(self) -> list[Item]:
        return list(self)

    @property
    def has_food(self) -> bool:
        return any(it.category == "food" for it in self)

    @property
    def has_healing(self) -> bool:
        return any(
            it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"])
            for it in self
        )

    @property
    def has_wand_of_teleport(self) -> bool:
        return any(it.category == "wand" and "teleport" in it.name.lower() for it in self)

    @property
    def has_wand_of_digging(self) -> bool:
        return any(it.category == "wand" and "digging" in it.name.lower() for it in self)

    @property
    def has_scroll_of_teleport(self) -> bool:
        return any(it.category == "scroll" and "teleport" in it.name.lower() for it in self)

    @property
    def has_pick_axe(self) -> bool:
        return any("pick-axe" in it.name.lower() or "pickaxe" in it.name.lower() for it in self)

    @property
    def has_lamp(self) -> bool:
        return any("lamp" in it.name.lower() or "lantern" in it.name.lower() for it in self)

    @property
    def weapon_is_cursed(self) -> bool:
        for it in self:
            if it.category == "weapon" and it.is_equipped and it.buc == "cursed":
                return True
        return False

    def get_food_slot(self) -> str | None:
        for it in self:
            if it.category == "food":
                return it.slot
        return None

    def get_healing_slot(self) -> str | None:
        for it in self:
            if it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"]):
                return it.slot
        return None

    def get_weapon_slot(self) -> str | None:
        for it in self:
            if it.category == "weapon":
                return it.slot
        return None

    @property
    def has_daggers(self) -> bool:
        return any(it.category == "weapon" and "dagger" in it.name.lower() for it in self)

    def get_dagger_slot(self) -> str | None:
        for it in self:
            if it.category == "weapon" and "dagger" in it.name.lower():
                return it.slot
        return None

    def find_items_by_category(self, cat: str) -> list[Item]:
        return [it for it in self if it.category == cat]

    @property
    def has_unworn_armor(self) -> bool:
        return any(
            it.category == "armor" and not it.is_equipped and it.slot not in self.failed_armor_slots
            for it in self
        )

    def get_unworn_armor_slot(self) -> str | None:
        for it in self:
            if it.category == "armor" and not it.is_equipped and it.slot not in self.failed_armor_slots:
                return it.slot
        return None


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


@dataclass(slots=True)
class SpatialView:
    """Spatial dungeon topology and navigation state."""
    stairs_down_known: bool = False
    stairs_up_known: bool = False
    stairs_down_pos: tuple[int, int] | None = None
    stairs_up_pos: tuple[int, int] | None = None
    standing_on_stairs_down: bool = False
    standing_on_stairs_up: bool = False
    has_unvisited_frontier: bool = False
    has_unsearched_dead_end: bool = False
    unvisited_frontier_count: int = 0
    floor_explored: bool = False


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
    door_is_locked: bool = False
    adjacent_fountain: bool = False
    fountain_in_fov: bool = False
    closest_fountain_pos: tuple[int, int] | None = None
    adjacent_altar: bool = False
    standing_on_altar: bool = False
    altar_is_aligned: bool = False
    adjacent_trap: bool = False
    standing_on_trap: bool = False
    can_forge_excalibur: bool = False


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
class Observation:
    """Standardized environment observation across all domains."""
    chars: np.ndarray             # 2D character grid (uint8)
    glyphs: np.ndarray | None     # 2D glyph grid (int16/int32)
    hero: HeroState               # Hero status
    inventory: InventoryView = field(default_factory=InventoryView)
    status: HeroStatus = field(default_factory=HeroStatus)
    combat: CombatView = field(default_factory=CombatView)
    spatial: SpatialView = field(default_factory=SpatialView)
    dungeon: DungeonView = field(default_factory=DungeonView)
    corpses: list[FloorCorpse] = field(default_factory=list)
    message: str = ""             # Last in-game message text
    raw_obs: Any = None           # Original environment observation dict


@dataclass(slots=True, frozen=True)
class Action:
    """Atomic or composite action emitted by policy."""
    name: str
    direction: tuple[int, int] | None = None  # (dy, dx)
    char: str | None = None                  # Direct command character (e.g. ">", "<", "e", "q")
    slot: str | None = None                  # Inventory slot letter (e.g. "a", "b")
    target_pos: tuple[int, int] | None = None
    count: int = 1                           # Repeat count
    extra: dict[str, Any] = field(default_factory=dict)
