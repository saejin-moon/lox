"""
Craftax GameState & Observation Representations ("Hard Code Nothing").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lox.env.state import WorldMemory


@dataclass
class CraftaxMonsterInfo:
    name: str = "monster"


@dataclass
class CraftaxMonster:
    char: str
    y: int
    x: int
    info: CraftaxMonsterInfo = field(default_factory=CraftaxMonsterInfo)


@dataclass
class CraftaxGameState:
    """Zero-copy observation snapshot for Craftax / Crafter benchmark."""

    y: int = 24
    x: int = 24
    facing_dy: int = 0
    facing_dx: int = 1
    health: int = 9
    max_health: int = 9
    food: int = 9
    drink: int = 9
    energy: int = 9
    step_count: int = 0
    inventory: dict[str, int] = field(default_factory=dict)
    achievements: set[str] = field(default_factory=set)
    map: np.ndarray = field(default_factory=lambda: np.zeros((48, 48), dtype=np.int32))
    entities: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    mem: WorldMemory = field(default_factory=WorldMemory)

    @property
    def hp(self) -> int:
        return self.health

    @property
    def max_hp(self) -> int:
        return self.max_health

    @property
    def depth(self) -> int:
        return len(self.achievements) + 1

    @property
    def score(self) -> int:
        inv_score = (
            self.inventory.get("wood", 0) * 10
            + self.inventory.get("stone", 0) * 20
            + self.inventory.get("coal", 0) * 30
            + self.inventory.get("iron", 0) * 50
            + self.inventory.get("diamond", 0) * 200
        )
        return len(self.achievements) * 100 + inv_score

    @property
    def turn(self) -> int:
        return self.step_count

    @property
    def ac(self) -> int:
        base_ac = 10
        if self.inventory.get("iron_sword", 0) > 0:
            base_ac -= 4
        elif self.inventory.get("stone_sword", 0) > 0:
            base_ac -= 2
        elif self.inventory.get("wood_sword", 0) > 0:
            base_ac -= 1
        return base_ac

    @property
    def xp_level(self) -> int:
        return len(self.achievements) + 1

    @property
    def gold(self) -> int:
        return (
            self.inventory.get("diamond", 0) * 100 + self.inventory.get("iron", 0) * 20
        )

    @property
    def dungeon_num(self) -> int:
        return 0

    @property
    def level_num(self) -> int:
        return 1

    @property
    def hunger_state(self) -> int:
        if self.food > 4:
            return 0  # Satiated / Normal
        elif self.food > 2:
            return 1  # Hungry
        elif self.food > 0:
            return 2  # Weak
        return 3  # Fainting / Starving

    @property
    def condition_mask(self) -> int:
        return 0

    @property
    def stairs_down_pos(self) -> tuple[int, int] | None:
        return None

    def is_stairs_down(self, py: int, px: int) -> bool:
        return False

    def tile_at(self, py: int, px: int) -> int:
        if 0 <= py < self.map.shape[0] and 0 <= px < self.map.shape[1]:
            return int(self.map[py, px])
        return -1

    def visible_monsters(self) -> list[CraftaxMonster]:
        vis: list[CraftaxMonster] = []
        for ent in self.entities:
            ey, ex = ent.get("y", -1), ent.get("x", -1)
            if (
                abs(ey - self.y) <= 8
                and abs(ex - self.x) <= 8
                and ent.get("health", 0) > 0
            ):
                name = ent.get("type", "monster")
                char = name[0].upper() if name else "M"
                vis.append(
                    CraftaxMonster(
                        char=char, y=ey, x=ex, info=CraftaxMonsterInfo(name=name)
                    )
                )
        return vis
