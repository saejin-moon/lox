"""
LOX 2.0 Core: Standardized Types and Interfaces.
Pure, lightweight dataclasses without circular dependencies or runtime overhead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import numpy as np


class Status(Enum):
    """Execution status returned by Behavior Tree nodes on each tick."""
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class HungerState(Enum):
    """Standardized hunger levels."""
    SATIATED = 0
    NORMAL = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


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
    dungeon_num: int = 0   # 0 = Doom, 2 = Mines, 3 = Sokoban
    gold: int = 0
    score: int = 0
    turn: int = 0
    turns_on_level: int = 0
    hunger_state: HungerState = HungerState.NORMAL
    is_blind: bool = False
    is_confused: bool = False
    is_stunned: bool = False
    is_hallucinating: bool = False

    @property
    def hp_frac(self) -> float:
        return self.hp / max(1, self.max_hp)

    @property
    def energy_frac(self) -> float:
        return self.energy / max(1, self.max_energy)


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


@dataclass
class Observation:
    """Standardized environment observation across all domains."""
    chars: np.ndarray             # 2D character grid (uint8)
    glyphs: np.ndarray | None     # 2D glyph grid (int16/int32)
    hero: HeroState               # Hero status
    inventory: list[Item] = field(default_factory=list)
    message: str = ""             # Last in-game message text
    raw_obs: Any = None           # Original environment observation dict


@dataclass(slots=True, frozen=True)
class Action:
    """Atomic action emitted by a behavior leaf node."""
    name: str
    direction: tuple[int, int] | None = None  # (dy, dx)
    char: str | None = None                  # Direct command character (e.g. ">", "<", "e", "q")
    slot: str | None = None                  # Inventory slot letter (e.g. "a", "b")
    count: int = 1                           # Repeat count
    extra: dict[str, Any] = field(default_factory=dict)
