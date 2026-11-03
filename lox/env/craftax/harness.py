"""
Deterministic Simulation & Prefix-Replay Harness for Craftax / Crafter ("Hard Code Nothing").
Full 22-achievement survival tech tree with tool-gated mining and crafting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lox.env.base import EnvHarness
from lox.env.craftax.actions import ACTION_DELTAS, CraftaxAction
from lox.env.craftax.state import CraftaxGameState
from lox.env.state import WorldMemory

# Tile ID constants
EMPTY = 0
GRASS = 0
TREE = 1
STONE = 2
WATER = 3
IRON = 4
TABLE = 5
FURNACE = 6
COAL = 7
DIAMOND = 8
PLANT = 9
SAND = 10
LAVA = 11

PASSABLE_TILES = {GRASS, SAND}


class CraftaxHarness(EnvHarness):
    """Deterministic simulation harness for Craftax / Crafter benchmark."""

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        self.step_count = 0
        self.done = False
        self.action_bytes = bytearray()
        self.step_byte_offsets: list[int] = [0]
        self.current_seed = 42
        self.mem = WorldMemory()
        self._state = CraftaxGameState()

    def reset(self, seed: int = 42) -> CraftaxGameState:
        self.current_seed = int(seed)
        self.step_count = 0
        self.done = False
        self.action_bytes.clear()
        self.step_byte_offsets = [0]
        self.mem.reset()

        rng = np.random.default_rng(seed)
        # 48x48 procedural world
        h_map = np.zeros((48, 48), dtype=np.int32)
        # Background generation
        choices = [GRASS, TREE, STONE, WATER, COAL, IRON, DIAMOND, PLANT, SAND]
        probs = [0.62, 0.14, 0.10, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01]
        h_map = rng.choice(choices, size=(48, 48), p=probs).astype(np.int32)

        # Guarantee local cluster around spawn (24, 24) for determinism & exploration
        h_map[24, 24] = GRASS  # Spawn tile
        h_map[24, 25] = GRASS  # East: Grass
        h_map[24, 26] = TREE  # East+1: Tree
        h_map[24, 23] = GRASS  # West: Grass
        h_map[24, 22] = WATER  # West+1: Water
        h_map[23, 24] = GRASS  # North: Grass
        h_map[22, 24] = STONE  # North+1: Stone
        h_map[25, 24] = GRASS  # South: Grass
        h_map[25, 25] = COAL  # SE: Coal
        h_map[23, 25] = IRON  # NE: Iron
        h_map[23, 23] = DIAMOND  # NW: Diamond

        # Procedural entities
        entities: list[dict[str, Any]] = [
            {"type": "cow", "y": 26, "x": 26, "health": 3},
            {"type": "zombie", "y": 30, "x": 30, "health": 5},
            {"type": "skeleton", "y": 20, "x": 20, "health": 3},
        ]

        self._state = CraftaxGameState(
            y=24,
            x=24,
            facing_dy=0,
            facing_dx=1,
            health=9,
            max_health=9,
            food=9,
            drink=9,
            energy=9,
            step_count=0,
            inventory={
                "wood": 0,
                "stone": 0,
                "coal": 0,
                "iron": 0,
                "diamond": 0,
                "sapling": 0,
                "wood_pickaxe": 0,
                "stone_pickaxe": 0,
                "iron_pickaxe": 0,
                "wood_sword": 0,
                "stone_sword": 0,
                "iron_sword": 0,
            },
            achievements=set(),
            map=h_map,
            entities=entities,
            message="Spawned in Craftax world.",
            mem=self.mem,
        )
        self.mem.update(self._state)
        return self._state

    def _adjacent_has(self, tile_id: int) -> bool:
        y, x = self._state.y, self._state.x
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 48 and 0 <= nx < 48:
                    if self._state.map[ny, nx] == tile_id:
                        return True
        return False

    def step(self, action: Any) -> tuple[CraftaxGameState, bool, dict[str, Any]]:
        if self.done:
            return self._state, True, {"action_name": "DONE"}

        act_val = int(action) if hasattr(action, "value") else int(action)
        self.action_bytes.append(act_val & 0xFF)
        self.step_byte_offsets.append(len(self.action_bytes))
        self.step_count += 1

        action_name = "NOOP"
        try:
            c_act = CraftaxAction(act_val)
            action_name = c_act.name
        except Exception:
            c_act = CraftaxAction.NOOP

        # 1. Movement & Facing Direction
        if c_act in ACTION_DELTAS:
            dy, dx = ACTION_DELTAS[c_act]
            self._state.facing_dy = dy
            self._state.facing_dx = dx
            ny = max(0, min(47, self._state.y + dy))
            nx = max(0, min(47, self._state.x + dx))

            # Tile passability & entity blocking
            is_passable = self._state.map[ny, nx] in PASSABLE_TILES
            is_blocked_by_ent = any(
                e.get("health", 0) > 0 and e.get("y") == ny and e.get("x") == nx
                for e in self._state.entities
            )
            if is_passable and not is_blocked_by_ent:
                self._state.y = ny
                self._state.x = nx

        # 2. DO (Mine, Harvest, Drink, Attack)
        elif c_act == CraftaxAction.DO:
            ty = max(0, min(47, self._state.y + self._state.facing_dy))
            tx = max(0, min(47, self._state.x + self._state.facing_dx))

            # Attack entity if present
            attacked_ent = None
            for e in self._state.entities:
                if e.get("health", 0) > 0 and e.get("y") == ty and e.get("x") == tx:
                    attacked_ent = e
                    break

            if attacked_ent is not None:
                # Calculate weapon damage
                dmg = 1.0
                if self._state.inventory.get("iron_sword", 0) > 0:
                    dmg = 5.0
                elif self._state.inventory.get("stone_sword", 0) > 0:
                    dmg = 3.0
                elif self._state.inventory.get("wood_sword", 0) > 0:
                    dmg = 2.0
                elif self._state.inventory.get("iron_pickaxe", 0) > 0:
                    dmg = 3.0
                elif self._state.inventory.get("stone_pickaxe", 0) > 0:
                    dmg = 2.0
                elif self._state.inventory.get("wood_pickaxe", 0) > 0:
                    dmg = 1.5

                attacked_ent["health"] -= dmg
                if attacked_ent["health"] <= 0:
                    etype = attacked_ent.get("type", "")
                    if etype == "cow":
                        self._state.food = min(9, self._state.food + 4)
                        self._state.achievements.add("eat_cow")
                        self._state.message = "Defeated cow and ate beef."
                    elif etype == "zombie":
                        self._state.achievements.add("defeat_zombie")
                        self._state.message = "Defeated zombie."
                    elif etype == "skeleton":
                        self._state.achievements.add("defeat_skeleton")
                        self._state.message = "Defeated skeleton."
                else:
                    self._state.message = (
                        f"Hit {attacked_ent.get('type')} for {dmg} dmg."
                    )
            else:
                # Interact with block at (ty, tx)
                target_tile = int(self._state.map[ty, tx])
                if target_tile == TREE:
                    self._state.map[ty, tx] = GRASS
                    self._state.inventory["wood"] = (
                        self._state.inventory.get("wood", 0) + 1
                    )
                    self._state.inventory["sapling"] = (
                        self._state.inventory.get("sapling", 0) + 1
                    )
                    self._state.achievements.add("collect_wood")
                    self._state.achievements.add("collect_sapling")
                    self._state.message = "Chopped wood from tree."
                elif target_tile == WATER:
                    self._state.drink = 9
                    self._state.achievements.add("collect_drink")
                    self._state.message = "Drank clean water."
                elif target_tile == PLANT:
                    self._state.map[ty, tx] = GRASS
                    self._state.food = min(9, self._state.food + 3)
                    self._state.achievements.add("eat_plant")
                    self._state.message = "Ate ripe plant."
                elif target_tile == STONE:
                    if (
                        self._state.inventory.get("wood_pickaxe", 0) > 0
                        or self._state.inventory.get("stone_pickaxe", 0) > 0
                        or self._state.inventory.get("iron_pickaxe", 0) > 0
                    ):
                        self._state.map[ty, tx] = GRASS
                        self._state.inventory["stone"] = (
                            self._state.inventory.get("stone", 0) + 1
                        )
                        self._state.achievements.add("collect_stone")
                        self._state.message = "Mined stone with pickaxe."
                    else:
                        self._state.message = "Need a pickaxe to mine stone."
                elif target_tile == COAL:
                    if (
                        self._state.inventory.get("wood_pickaxe", 0) > 0
                        or self._state.inventory.get("stone_pickaxe", 0) > 0
                        or self._state.inventory.get("iron_pickaxe", 0) > 0
                    ):
                        self._state.map[ty, tx] = GRASS
                        self._state.inventory["coal"] = (
                            self._state.inventory.get("coal", 0) + 1
                        )
                        self._state.achievements.add("collect_coal")
                        self._state.message = "Mined coal with pickaxe."
                    else:
                        self._state.message = "Need a pickaxe to mine coal."
                elif target_tile == IRON:
                    if (
                        self._state.inventory.get("stone_pickaxe", 0) > 0
                        or self._state.inventory.get("iron_pickaxe", 0) > 0
                    ):
                        self._state.map[ty, tx] = GRASS
                        self._state.inventory["iron"] = (
                            self._state.inventory.get("iron", 0) + 1
                        )
                        self._state.achievements.add("collect_iron")
                        self._state.message = "Mined iron with stone or iron pickaxe."
                    else:
                        self._state.message = "Need stone or iron pickaxe to mine iron."
                elif target_tile == DIAMOND:
                    if self._state.inventory.get("iron_pickaxe", 0) > 0:
                        self._state.map[ty, tx] = GRASS
                        self._state.inventory["diamond"] = (
                            self._state.inventory.get("diamond", 0) + 1
                        )
                        self._state.achievements.add("collect_diamond")
                        self._state.message = "Mined diamond with iron pickaxe."
                    else:
                        self._state.message = "Need iron pickaxe to mine diamond."

        # 3. SLEEP
        elif c_act == CraftaxAction.SLEEP:
            self._state.energy = 9
            self._state.health = min(9, self._state.health + 1)
            self._state.achievements.add("wake_up")
            self._state.message = "Slept and recovered energy."

        # 4. Placement actions
        elif c_act == CraftaxAction.PLACE_STONE:
            ty = max(0, min(47, self._state.y + self._state.facing_dy))
            tx = max(0, min(47, self._state.x + self._state.facing_dx))
            if (
                self._state.inventory.get("stone", 0) >= 1
                and self._state.map[ty, tx] in PASSABLE_TILES
            ):
                self._state.inventory["stone"] -= 1
                self._state.map[ty, tx] = STONE
                self._state.achievements.add("place_stone")
                self._state.message = "Placed stone barrier."

        elif c_act == CraftaxAction.PLACE_PLANT:
            ty = max(0, min(47, self._state.y + self._state.facing_dy))
            tx = max(0, min(47, self._state.x + self._state.facing_dx))
            if (
                self._state.inventory.get("sapling", 0) >= 1
                and self._state.map[ty, tx] == GRASS
            ):
                self._state.inventory["sapling"] -= 1
                self._state.map[ty, tx] = PLANT
                self._state.achievements.add("place_plant")
                self._state.message = "Planted sapling."

        elif c_act == CraftaxAction.PLACE_TABLE:
            ty = max(0, min(47, self._state.y + self._state.facing_dy))
            tx = max(0, min(47, self._state.x + self._state.facing_dx))
            if (
                self._state.inventory.get("wood", 0) >= 2
                and self._state.map[ty, tx] in PASSABLE_TILES
            ):
                self._state.inventory["wood"] -= 2
                self._state.map[ty, tx] = TABLE
                self._state.achievements.add("place_table")
                self._state.message = "Constructed crafting table."

        elif c_act == CraftaxAction.PLACE_FURNACE:
            ty = max(0, min(47, self._state.y + self._state.facing_dy))
            tx = max(0, min(47, self._state.x + self._state.facing_dx))
            if (
                self._adjacent_has(TABLE)
                and self._state.inventory.get("stone", 0) >= 4
                and self._state.map[ty, tx] in PASSABLE_TILES
            ):
                self._state.inventory["stone"] -= 4
                self._state.map[ty, tx] = FURNACE
                self._state.achievements.add("place_furnace")
                self._state.message = "Constructed stone furnace near table."

        # 5. Crafting actions
        elif c_act == CraftaxAction.MAKE_WOOD_PICKAXE:
            if self._adjacent_has(TABLE) and self._state.inventory.get("wood", 0) >= 1:
                self._state.inventory["wood"] -= 1
                self._state.inventory["wood_pickaxe"] = (
                    self._state.inventory.get("wood_pickaxe", 0) + 1
                )
                self._state.achievements.add("make_wood_pickaxe")
                self._state.message = "Crafted wood pickaxe."

        elif c_act == CraftaxAction.MAKE_WOOD_SWORD:
            if self._adjacent_has(TABLE) and self._state.inventory.get("wood", 0) >= 1:
                self._state.inventory["wood"] -= 1
                self._state.inventory["wood_sword"] = (
                    self._state.inventory.get("wood_sword", 0) + 1
                )
                self._state.achievements.add("make_wood_sword")
                self._state.message = "Crafted wood sword."

        elif c_act == CraftaxAction.MAKE_STONE_PICKAXE:
            if (
                self._adjacent_has(TABLE)
                and self._state.inventory.get("wood", 0) >= 1
                and self._state.inventory.get("stone", 0) >= 1
            ):
                self._state.inventory["wood"] -= 1
                self._state.inventory["stone"] -= 1
                self._state.inventory["stone_pickaxe"] = (
                    self._state.inventory.get("stone_pickaxe", 0) + 1
                )
                self._state.achievements.add("make_stone_pickaxe")
                self._state.message = "Crafted stone pickaxe."

        elif c_act == CraftaxAction.MAKE_STONE_SWORD:
            if (
                self._adjacent_has(TABLE)
                and self._state.inventory.get("wood", 0) >= 1
                and self._state.inventory.get("stone", 0) >= 1
            ):
                self._state.inventory["wood"] -= 1
                self._state.inventory["stone"] -= 1
                self._state.inventory["stone_sword"] = (
                    self._state.inventory.get("stone_sword", 0) + 1
                )
                self._state.achievements.add("make_stone_sword")
                self._state.message = "Crafted stone sword."

        elif c_act == CraftaxAction.MAKE_IRON_PICKAXE:
            if (
                self._adjacent_has(TABLE)
                and self._adjacent_has(FURNACE)
                and self._state.inventory.get("wood", 0) >= 1
                and self._state.inventory.get("coal", 0) >= 1
                and self._state.inventory.get("iron", 0) >= 1
            ):
                self._state.inventory["wood"] -= 1
                self._state.inventory["coal"] -= 1
                self._state.inventory["iron"] -= 1
                self._state.inventory["iron_pickaxe"] = (
                    self._state.inventory.get("iron_pickaxe", 0) + 1
                )
                self._state.achievements.add("make_iron_pickaxe")
                self._state.message = "Forged iron pickaxe at furnace."

        elif c_act == CraftaxAction.MAKE_IRON_SWORD:
            if (
                self._adjacent_has(TABLE)
                and self._adjacent_has(FURNACE)
                and self._state.inventory.get("wood", 0) >= 1
                and self._state.inventory.get("coal", 0) >= 1
                and self._state.inventory.get("iron", 0) >= 1
            ):
                self._state.inventory["wood"] -= 1
                self._state.inventory["coal"] -= 1
                self._state.inventory["iron"] -= 1
                self._state.inventory["iron_sword"] = (
                    self._state.inventory.get("iron_sword", 0) + 1
                )
                self._state.achievements.add("make_iron_sword")
                self._state.message = "Forged iron sword at furnace."

        # Vitality decay
        if self.step_count % 30 == 0:
            self._state.energy = max(0, self._state.energy - 1)
        if self.step_count % 60 == 0:
            self._state.food = max(0, self._state.food - 1)
            self._state.drink = max(0, self._state.drink - 1)
            if self._state.food == 0 or self._state.drink == 0:
                self._state.health = max(0, self._state.health - 1)

        if self._state.health <= 0:
            self.done = True
            self._state.message = "Died of injuries or starvation."
        elif self.step_count >= self.max_steps:
            self.done = True

        self._state.step_count = self.step_count
        self.mem.update(self._state, action_name=action_name)
        return (
            self._state,
            self.done,
            {
                "action_name": action_name,
                "achievements": len(self._state.achievements),
                "health": self._state.health,
            },
        )

    def replay_bytes(
        self,
        seed: int,
        recorded_bytes: bytes | bytearray | None = None,
        stop_at_byte_idx: int | None = None,
    ) -> CraftaxGameState:
        buf = bytes(recorded_bytes or b"")
        self.reset(seed)
        limit = (
            len(buf) if stop_at_byte_idx is None else min(len(buf), stop_at_byte_idx)
        )
        for i in range(limit):
            self.step(buf[i])
            if self.done:
                break
        return self._state

    def close(self) -> None:
        pass
