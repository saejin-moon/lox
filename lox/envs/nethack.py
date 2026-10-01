"""
LOX 2.0 NetHack Adapter: High-performance, clean NLE wrapper.
Directly consumes structured NLE observation channels (blstats, inv_strs, inv_letters)
with zero regex scraping and automatic --More-- prompt dismissal.
Pure agentic autonomy: zero hardcoded safety interlocks or engine-level overrides.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import gymnasium as gym
import nle
from nle import nethack

from lox.core.types import (
    Observation,
    HeroState,
    HeroStatus,
    HungerState,
    EncumbranceState,
    Item,
    Action,
    InventoryView,
    CombatView,
    SpatialView,
    DungeonView,
    FloorCorpse,
)
from lox.core.spatial import SpatialEngine, build_walkable_mask
from lox.envs.base import EnvironmentAdapter


# Direction character mapping for NetHack
DIR_CHARS: dict[tuple[int, int], str] = {
    (-1, 0): "k",  # North
    (0, 1): "l",   # East
    (1, 0): "j",   # South
    (0, -1): "h",  # West
    (-1, 1): "u",  # North-East
    (1, 1): "n",   # South-East
    (1, -1): "b",  # South-West
    (-1, -1): "y", # North-West
}

OCLASS_MAP: dict[int, str] = {
    nethack.WEAPON_CLASS: "weapon",
    nethack.ARMOR_CLASS: "armor",
    nethack.RING_CLASS: "ring",
    nethack.AMULET_CLASS: "amulet",
    nethack.TOOL_CLASS: "tool",
    nethack.FOOD_CLASS: "food",
    nethack.POTION_CLASS: "potion",
    nethack.SCROLL_CLASS: "scroll",
    nethack.WAND_CLASS: "wand",
    nethack.COIN_CLASS: "gold",
    nethack.GEM_CLASS: "gem",
}

BRANCH_NAMES: dict[int, str] = {
    0: "dungeon",
    1: "mines",
    2: "quest",
    3: "sokoban",
    4: "fort_ludios",
    5: "vlad_tower",
}


class NetHackAdapter(EnvironmentAdapter):
    """Clean NetHack adapter with unconstrained agentic execution."""

    def __init__(self, env_id: str = "NetHackChallenge-v0", role: str = "valkyrie"):
        self.env_id = env_id
        self.role = role
        self.env = gym.make(env_id, character=role, options=("autopickup", "pickup_types:?!/%=[$"))
        self.visited = np.zeros((21, 79), dtype=bool)
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0

        # Memory tracking
        self.last_prayer_turn = -1000
        self.floor_corpses: dict[tuple[int, int], tuple[str, int, bool]] = {}  # (y, x) -> (name, drop_turn, is_poisonous)
        self.known_stairs_down: tuple[int, int] | None = None
        self.known_stairs_up: tuple[int, int] | None = None
        self.searched_count = np.zeros((21, 79), dtype=np.int32)
        self.peaceful_positions: set[tuple[int, int]] = set()
        self.blocked_tiles: set[tuple[int, int]] = set()
        self.non_door_tiles: set[tuple[int, int]] = set()

        # Build ASCII char -> action index map
        unwrapped = getattr(self.env, "unwrapped", self.env)
        self.actions = list(getattr(unwrapped, "actions", []))
        self.char_to_act: dict[str, int] = {}
        for idx, act in enumerate(self.actions):
            if hasattr(act, "value"):
                val = act.value
            else:
                val = int(act)
            if 0 <= val <= 255:
                self.char_to_act[chr(val)] = idx

        # Command.PRAY index in standard NLE actions is 62
        self.pray_action_idx = 62 if len(self.actions) > 62 else self.char_to_act.get("#", 0)
        self.last_prayer_turn: int = -1000

    def _get_doors_mask(self, glyphs: np.ndarray) -> np.ndarray:
        """Returns boolean mask of real closed doors using NLE CMAP glyphs."""
        doors_mask = (glyphs == (nethack.GLYPH_CMAP_OFF + 15)) | (glyphs == (nethack.GLYPH_CMAP_OFF + 16))
        for dy, dx in self.non_door_tiles:
            if 0 <= dy < 21 and 0 <= dx < 79:
                doors_mask[dy, dx] = False
        return doors_mask

    def is_target_floating_eye(self, glyphs: np.ndarray, y: int, x: int) -> bool:
        """Informational check: returns True if monster at (y, x) is a floating eye."""
        if not (0 <= y < 21 and 0 <= x < 79):
            return False
        g = int(glyphs[y, x])
        if nethack.glyph_is_monster(g):
            mon_id = nethack.glyph_to_mon(g)
            return mon_id == 28
        return False

    def can_safely_pray(self, turn: int) -> bool:
        """Informational check: returns True if prayer cooldown (350 turns) has elapsed."""
        return (turn - self.last_prayer_turn) >= 350

    def _decode_message(self, msg_raw: Any) -> str:
        if isinstance(msg_raw, np.ndarray):
            return "".join(chr(c) for c in msg_raw if c > 0).strip()
        elif isinstance(msg_raw, bytes):
            return msg_raw.decode("ascii", errors="ignore").strip()
        return str(msg_raw).strip()

    def _extract_obs(self, raw_obs: dict[str, Any]) -> Observation:
        bl = raw_obs["blstats"]
        x, y = int(bl[0]), int(bl[1])
        hp, max_hp = int(bl[10]), int(bl[11])
        depth = int(bl[12])
        gold = int(bl[13])
        energy, max_energy = int(bl[14]), int(bl[15])
        ac = int(bl[16])
        level = int(bl[18])
        turn = int(bl[20])
        hunger_raw = int(bl[21])
        hunger = HungerState(min(4, max(0, hunger_raw)))
        cap_raw = int(bl[22]) if len(bl) > 22 else 0
        encumbrance = EncumbranceState(min(5, max(0, cap_raw)))
        dnum = int(bl[23]) if len(bl) > 23 else 0
        branch_name = BRANCH_NAMES.get(dnum, "dungeon")

        # C-level condition bitmask from NLE blstats[25]
        cond = int(bl[25]) if len(bl) > 25 else 0
        is_blind = bool(cond & nethack.BL_MASK_BLIND) if hasattr(nethack, "BL_MASK_BLIND") else False
        is_confused = bool(cond & nethack.BL_MASK_CONF) if hasattr(nethack, "BL_MASK_CONF") else False
        is_stunned = bool(cond & nethack.BL_MASK_STUN) if hasattr(nethack, "BL_MASK_STUN") else False
        is_hallucinating = bool(cond & nethack.BL_MASK_HALLU) if hasattr(nethack, "BL_MASK_HALLU") else False
        is_levitating = bool(cond & nethack.BL_MASK_LEV) if hasattr(nethack, "BL_MASK_LEV") else False
        is_sick = bool(cond & (getattr(nethack, "BL_MASK_FOODPOIS", 8) | getattr(nethack, "BL_MASK_TERMILL", 16)))

        if depth != self.last_depth or dnum != self.last_dnum:
            self.turns_on_level = 0
            self.visited.fill(False)
            self.searched_count.fill(0)
            self.peaceful_positions.clear()
            self.blocked_tiles.clear()
            self.non_door_tiles.clear()
            self.floor_corpses.clear()
            self.known_stairs_down = None
            self.known_stairs_up = None
            self.last_depth = depth
            self.last_dnum = dnum
        else:
            self.turns_on_level += 1

        self.visited[y, x] = True

        hero = HeroState(
            y=y,
            x=x,
            hp=hp,
            max_hp=max_hp,
            energy=energy,
            max_energy=max_energy,
            ac=ac,
            level=level,
            depth=depth,
            dungeon_num=dnum,
            gold=gold,
            turn=turn,
            turns_on_level=self.turns_on_level,
            hunger_state=hunger,
            dungeon_branch=branch_name,
            is_dead=(hp <= 0),
            is_blind=is_blind,
            is_confused=is_confused,
            is_stunned=is_stunned,
            is_hallucinating=is_hallucinating,
            is_sick=is_sick,
        )

        status = HeroStatus(
            is_blind=is_blind,
            is_confused=is_confused,
            is_stunned=is_stunned,
            is_hallucinating=is_hallucinating,
            is_sick=is_sick,
            is_levitating=is_levitating,
            is_encumbered=(encumbrance != EncumbranceState.UNENCUMBERED),
            encumbrance_level=encumbrance,
        )

        inventory_items: list[Item] = []
        inv_letters = raw_obs.get("inv_letters", [])
        inv_strs = raw_obs.get("inv_strs", [])
        inv_oclasses = raw_obs.get("inv_oclasses", [])

        for letter, desc_bytes, oclass in zip(inv_letters, inv_strs, inv_oclasses):
            if letter > 0:
                slot = chr(int(letter))
                desc = self._decode_message(desc_bytes)
                cat = OCLASS_MAP.get(int(oclass), "unknown")
                is_equipped = "(weapon in hands)" in desc or "(being worn)" in desc
                buc = "uncursed"
                if "cursed" in desc:
                    buc = "cursed"
                elif "blessed" in desc:
                    buc = "blessed"
                inventory_items.append(
                    Item(slot=slot, name=desc, category=cat, is_equipped=is_equipped, buc=buc)
                )

        inv_view = InventoryView(inventory_items)
        message = self._decode_message(raw_obs.get("message", ""))
        chars = raw_obs["chars"]
        glyphs = raw_obs["glyphs"]

        # Track fresh corpses from message
        if "you kill the" in message.lower() or "you destroy the" in message.lower():
            m_killed = message.lower().replace("you kill the ", "").replace("you destroy the ", "").strip().rstrip("!.")
            is_pois = any(k in m_killed for k in ("poison", "kobold", "snake", "spider", "viper", "beetle"))
            is_deadly = any(k in m_killed for k in ("cockatrice", "chickatrice", "medusa"))
            self.floor_corpses[(y, x)] = (m_killed, turn, is_pois, is_deadly)

        # Tactical combat analysis from glyphs
        adjacent_hostile = False
        hostile_count = 0
        closest_name = ""
        closest_dist = 999.0
        closest_pos = None
        floating_eye_fov = False
        adjacent_hostiles_count = 0

        if glyphs is not None:
            for gy in range(max(0, y - 8), min(21, y + 9)):
                for gx in range(max(0, x - 8), min(79, x + 9)):
                    if gy == y and gx == x:
                        continue
                    if (gy, gx) in self.peaceful_positions:
                        continue
                    g = int(glyphs[gy, gx])
                    if nethack.glyph_is_monster(g) and not nethack.glyph_is_pet(g):
                        hostile_count += 1
                        mon_id = nethack.glyph_to_mon(g)
                        try:
                            mname = nethack.permonst(mon_id).mname
                        except Exception:
                            mname = "monster"

                        if mname == "floating eye":
                            floating_eye_fov = True

                        dist = math.hypot(gy - y, gx - x)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_name = mname
                            closest_pos = (gy, gx)

                        if abs(gy - y) <= 1 and abs(gx - x) <= 1:
                            adjacent_hostile = True
                            adjacent_hostiles_count += 1

        adjacent_peaceful = any(
            abs(py - y) <= 1 and abs(px - x) <= 1
            for py, px in self.peaceful_positions
        )

        # Track stairs coordinates anywhere on the revealed map
        stairs_down = np.argwhere(chars == ord(">"))
        if len(stairs_down) > 0:
            self.known_stairs_down = (int(stairs_down[0][0]), int(stairs_down[0][1]))
        stairs_up = np.argwhere(chars == ord("<"))
        if len(stairs_up) > 0:
            self.known_stairs_up = (int(stairs_up[0][0]), int(stairs_up[0][1]))

        # Spatial topology & navigation analysis
        walkable = build_walkable_mask(raw_obs)
        walkable_ex = walkable.copy()
        doors_mask = self._get_doors_mask(glyphs)
        walkable_ex[doors_mask] = True
        if self.known_stairs_down:
            walkable_ex[self.known_stairs_down[0], self.known_stairs_down[1]] = True
        if self.known_stairs_up:
            walkable_ex[self.known_stairs_up[0], self.known_stairs_up[1]] = True

        target_frontier_mask = (walkable & (~self.visited)) | doors_mask
        frontier_tile = SpatialEngine.find_nearest_frontier((y, x), walkable_ex, self.visited, target_mask=target_frontier_mask)
        has_frontier = frontier_tile is not None

        # Corridor detection
        walkable_adj = sum(
            1 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= y + dy < 21 and 0 <= x + dx < 79 and walkable[y + dy, x + dx]
        )
        in_corridor = (walkable_adj <= 2 and chr(chars[y, x]) == "#")

        is_fast_dangerous = closest_name in ("soldier ant", "killer bee", "giant spider", "centipede")
        combat = CombatView(
            adjacent_hostile=adjacent_hostile,
            hostile_count_fov=hostile_count,
            closest_hostile_name=closest_name,
            closest_hostile_dist=closest_dist,
            closest_hostile_pos=closest_pos,
            is_surrounded=(adjacent_hostiles_count >= 2),
            in_corridor=in_corridor,
            can_retreat=(walkable_adj > adjacent_hostiles_count),
            standing_on_elbereth=("Elbereth" in message),
            floating_eye_in_fov=floating_eye_fov,
            adjacent_peaceful=adjacent_peaceful,
            is_fast_dangerous=is_fast_dangerous,
        )

        # Check for unsearched corridor dead ends or room perimeter tiles across the entire floor
        has_dead_ends = False
        for cy in range(21):
            for cx in range(79):
                if walkable[cy, cx] and chr(chars[cy, cx]) == "#" and self.searched_count[cy, cx] < 15:
                    adj = sum(1 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                              if 0 <= cy + dy < 21 and 0 <= cx + dx < 79 and walkable[cy + dy, cx + dx])
                    if adj <= 1:
                        has_dead_ends = True
                        break
            if has_dead_ends:
                break

        # If all corridor dead ends searched, check unsearched room perimeter tiles adjacent to walls
        if not has_dead_ends:
            for cy in range(21):
                for cx in range(79):
                    if walkable[cy, cx] and chr(chars[cy, cx]) == "." and self.searched_count[cy, cx] < 10:
                        if any(0 <= cy + dy < 21 and 0 <= cx + dx < 79 and chr(chars[cy + dy, cx + dx]) in ("-", "|", " ")
                               for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                            has_dead_ends = True
                            break
                if has_dead_ends:
                    break

        spatial = SpatialView(
            stairs_down_known=(self.known_stairs_down is not None),
            stairs_up_known=(self.known_stairs_up is not None),
            stairs_down_pos=self.known_stairs_down,
            stairs_up_pos=self.known_stairs_up,
            standing_on_stairs_down=(self.known_stairs_down == (y, x) or chr(chars[y, x]) == ">"),
            standing_on_stairs_up=(self.known_stairs_up == (y, x) or chr(chars[y, x]) == "<"),
            has_unvisited_frontier=has_frontier,
            has_unsearched_dead_end=has_dead_ends,
            unvisited_frontier_count=int(np.sum(walkable_ex & (~self.visited))),
            floor_explored=(not has_frontier and self.known_stairs_down is not None),
        )

        # Dungeon tile type
        curr_char = chr(chars[y, x])
        curr_glyph = int(glyphs[y, x])
        tile_type = "room"
        if curr_char == "#":
            tile_type = "corridor"
        elif nethack.glyph_is_cmap(curr_glyph) and nethack.glyph_to_cmap(curr_glyph) in (12, 13, 14, 15, 16):
            tile_type = "doorway"
        elif nethack.glyph_is_cmap(curr_glyph) and nethack.glyph_to_cmap(curr_glyph) == 31:
            tile_type = "fountain"
        elif nethack.glyph_is_cmap(curr_glyph) and nethack.glyph_to_cmap(curr_glyph) == 27:
            tile_type = "altar"
        elif nethack.glyph_is_trap(curr_glyph):
            tile_type = "trap"
        elif curr_char == ">":
            tile_type = "stairs_down"
        elif curr_char == "<":
            tile_type = "stairs_up"

        # Check adjacent features (doors can only be interacted with cardinally)
        adj_door = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 21 and 0 <= nx < 79:
                if doors_mask[ny, nx]:
                    adj_door = True
                    break

        adj_fountain = False
        adj_altar = False
        adj_trap = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 21 and 0 <= nx < 79:
                    ac_char = chr(chars[ny, nx])
                    if ac_char == "{":
                        adj_fountain = True
                    elif ac_char == "_":
                        adj_altar = True
                    elif ac_char == "^":
                        adj_trap = True

        can_forge = (
            self.role == "valkyrie"
            and level >= 5
            and any("long sword" in it.name.lower() for it in inventory_items)
            and not any("excalibur" in it.name.lower() for it in inventory_items)
        )

        door_is_locked = bool(adj_door and ("locked" in message.lower() or "won't open" in message.lower()))

        dungeon = DungeonView(
            tile_type=tile_type,
            in_shop=("shop" in message.lower()),
            in_temple=("temple" in message.lower()),
            is_dark_level=(branch_name == "mines"),
            dungeon_branch=branch_name,
            adjacent_closed_door=adj_door,
            adjacent_open_door=False,
            door_is_locked=door_is_locked,
            adjacent_fountain=adj_fountain,
            adjacent_altar=adj_altar,
            standing_on_altar=(curr_char == "_"),
            adjacent_trap=adj_trap,
            standing_on_trap=(curr_char == "^"),
            can_forge_excalibur=can_forge,
        )

        # Corpses
        corpse_list = []
        for (cy, cx), corpse_data in self.floor_corpses.items():
            cname = corpse_data[0]
            dturn = corpse_data[1]
            is_pois = corpse_data[2] if len(corpse_data) > 2 else False
            is_deadly = corpse_data[3] if len(corpse_data) > 3 else False
            age = turn - dturn
            corpse_list.append(
                FloorCorpse(
                    name=cname,
                    y=cy,
                    x=cx,
                    drop_turn=dturn,
                    age_turns=age,
                    is_poisonous=is_pois,
                    is_deadly=is_deadly,
                    is_fresh=(age < 50),
                )
            )

        return Observation(
            chars=chars,
            glyphs=glyphs,
            hero=hero,
            inventory=inv_view,
            status=status,
            combat=combat,
            spatial=spatial,
            dungeon=dungeon,
            corpses=corpse_list,
            message=message,
            raw_obs=raw_obs,
        )

    def _dismiss_more(self, raw_obs: dict[str, Any], term: bool, trunc: bool) -> tuple[dict[str, Any], bool, bool]:
        space_idx = self.char_to_act.get(" ", 18)
        for _ in range(25):
            if term or trunc:
                break
            msg = self._decode_message(raw_obs.get("message", ""))
            misc = raw_obs.get("misc")
            in_more = misc is not None and len(misc) > 2 and misc[2] == 1
            if "--More--" in msg or in_more:
                raw_obs, _, term, trunc, _ = self.env.step(space_idx)
            elif "Are you sure you want to pray?" in msg:
                raw_obs, _, term, trunc, _ = self.env.step(self.char_to_act.get("y", space_idx))
            elif "(y/n)" in msg or "[yn]" in msg or "really attack" in msg.lower() or "Really quit?" in msg:
                raw_obs, _, term, trunc, _ = self.env.step(self.char_to_act.get("n", space_idx))
            elif "who are you" in msg.lower() or "what is your name" in msg.lower() or "call this" in msg.lower() or "hello stranger" in msg.lower():
                esc_idx = self.char_to_act.get("\x1b", 38)
                raw_obs, _, term, trunc, _ = self.env.step(esc_idx)
            else:
                break
            if term or trunc:
                break
        return raw_obs, term, trunc

    def _step_sequence(self, action_indices: list[int]) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        term = False
        trunc = False
        info: dict[str, Any] = {}
        raw_obs = getattr(self, "_last_raw_obs", {})
        for act in action_indices:
            if term or trunc:
                break
            raw_obs, r, term, trunc, info = self.env.step(act)
            total_reward += float(r)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            if term or trunc:
                break
        self._last_raw_obs = raw_obs
        obs = self._extract_obs(raw_obs)
        self._last_obs = obs

        msg = obs.message.lower()
        if "really attack" in msg or "who are you" in msg or "hello stranger" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.peaceful_positions.add((py + dy, px + dx))
        if "cannot pass through the bars" in msg or "it's a wall" in msg or "it's solid stone" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.blocked_tiles.add((py + dy, px + dx))
        if "you see no door there" in msg or "cannot open that" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.non_door_tiles.add((py + dy, px + dx))

        return obs, total_reward, bool(term), bool(trunc), info

    def reset(self, seed: int | None = None) -> Observation:
        self.visited.fill(False)
        self.searched_count.fill(0)
        self.peaceful_positions.clear()
        self.blocked_tiles.clear()
        self.non_door_tiles.clear()
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0
        self.last_prayer_turn = -1000
        self.floor_corpses.clear()
        self.known_stairs_down = None
        self.known_stairs_up = None
        raw_obs, _ = self.env.reset(seed=seed)
        raw_obs, _, _ = self._dismiss_more(raw_obs, False, False)
        obs = self._extract_obs(raw_obs)
        self._last_obs = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Executes action without safety interception.
        Supports high-level navigation, combat, items, and atomic keyboard commands.
        """
        obs_prev = getattr(self, "_last_obs", None)
        self._prev_hero_pos = (obs_prev.hero.y, obs_prev.hero.x) if obs_prev else (0, 0)
        self._last_attempted_dir = action.direction
        target_char = "."

        # Handle composite navigation actions
        if action.name == "step_to_frontier" and obs_prev is not None:
            hero = obs_prev.hero
            walkable = build_walkable_mask(obs_prev.raw_obs)
            walkable_nav = walkable.copy()
            for by, bx in self.blocked_tiles:
                if 0 <= by < 21 and 0 <= bx < 79:
                    walkable_nav[by, bx] = False
            doors_mask = self._get_doors_mask(obs_prev.glyphs)
            walkable_nav[doors_mask] = True
            if self.known_stairs_down:
                walkable_nav[self.known_stairs_down[0], self.known_stairs_down[1]] = True
            target_mask = (walkable & (~self.visited)) | doors_mask
            frontier = SpatialEngine.find_nearest_frontier((hero.y, hero.x), walkable_nav, self.visited, target_mask=target_mask)
            if frontier:
                path = SpatialEngine.find_path((hero.y, hero.x), frontier, walkable_nav)
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    action = Action(name="step_direction", direction=(dy, dx))

        elif action.name == "step_to_stairs_down" and obs_prev is not None and self.known_stairs_down:
            hero = obs_prev.hero
            walkable = build_walkable_mask(obs_prev.raw_obs)
            walkable_nav = walkable.copy()
            for by, bx in self.blocked_tiles:
                if 0 <= by < 21 and 0 <= bx < 79:
                    walkable_nav[by, bx] = False
            doors_mask = self._get_doors_mask(obs_prev.glyphs)
            walkable_nav[doors_mask] = True
            walkable_nav[self.known_stairs_down[0], self.known_stairs_down[1]] = True
            path = SpatialEngine.find_path((hero.y, hero.x), self.known_stairs_down, walkable_nav)
            if path:
                dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                action = Action(name="step_direction", direction=(dy, dx))

        elif action.name == "step_to_stairs_up" and obs_prev is not None and self.known_stairs_up:
            hero = obs_prev.hero
            walkable = build_walkable_mask(obs_prev.raw_obs)
            walkable_nav = walkable.copy()
            for by, bx in self.blocked_tiles:
                if 0 <= by < 21 and 0 <= bx < 79:
                    walkable_nav[by, bx] = False
            doors_mask = self._get_doors_mask(obs_prev.glyphs)
            walkable_nav[doors_mask] = True
            walkable_nav[self.known_stairs_up[0], self.known_stairs_up[1]] = True
            path = SpatialEngine.find_path((hero.y, hero.x), self.known_stairs_up, walkable_nav)
            if path:
                dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                action = Action(name="step_direction", direction=(dy, dx))

        elif action.name == "step_to_dead_end" and obs_prev is not None:
            hero = obs_prev.hero
            chars = obs_prev.chars
            walkable = build_walkable_mask(obs_prev.raw_obs)
            for by, bx in self.blocked_tiles:
                if 0 <= by < 21 and 0 <= bx < 79:
                    walkable[by, bx] = False
            dead_ends_mask = np.zeros((21, 79), dtype=bool)
            for cy in range(21):
                for cx in range(79):
                    if walkable[cy, cx] and chr(chars[cy, cx]) == "#" and self.searched_count[cy, cx] < 15:
                        adj = sum(1 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                                  if 0 <= cy + dy < 21 and 0 <= cx + dx < 79 and walkable[cy + dy, cx + dx])
                        if adj <= 1:
                            dead_ends_mask[cy, cx] = True

            # If no unsearched corridor dead ends remain, search room perimeter tiles adjacent to walls
            if not np.any(dead_ends_mask):
                for cy in range(21):
                    for cx in range(79):
                        if walkable[cy, cx] and chr(chars[cy, cx]) == "." and self.searched_count[cy, cx] < 10:
                            if any(0 <= cy + dy < 21 and 0 <= cx + dx < 79 and chr(chars[cy + dy, cx + dx]) in ("-", "|", " ")
                                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                                dead_ends_mask[cy, cx] = True

            if np.any(dead_ends_mask):
                if dead_ends_mask[hero.y, hero.x]:
                    action = Action(name="search")
                else:
                    target = SpatialEngine.find_nearest_frontier((hero.y, hero.x), walkable, self.visited, target_mask=dead_ends_mask)
                    if target:
                        path = SpatialEngine.find_path((hero.y, hero.x), target, walkable)
                        if path:
                            dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                            action = Action(name="step_direction", direction=(dy, dx))
            else:
                action = Action(name="search")

        elif action.name == "step_to" and obs_prev is not None:
            hero = obs_prev.hero
            target = action.target_pos or action.extra.get("target_pos") or action.direction
            if target and isinstance(target, tuple) and len(target) == 2:
                ty, tx = int(target[0]), int(target[1])
                walkable = build_walkable_mask(obs_prev.raw_obs)
                walkable_nav = walkable.copy()
                for by, bx in self.blocked_tiles:
                    if 0 <= by < 21 and 0 <= bx < 79:
                        walkable_nav[by, bx] = False
                doors_mask = self._get_doors_mask(obs_prev.glyphs)
                walkable_nav[doors_mask] = True
                walkable_nav[ty, tx] = True
                path = SpatialEngine.find_path((hero.y, hero.x), (ty, tx), walkable_nav)
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    action = Action(name="step_direction", direction=(dy, dx))

        elif action.name in ("step_away_from_hostile", "step_to_chokepoint", "retreat") and obs_prev is not None:
            hero = obs_prev.hero
            glyphs = obs_prev.glyphs
            # Find open tile away from hostile
            walkable = build_walkable_mask(obs_prev.raw_obs)
            best_tile = None
            max_dist = -1.0
            closest_pos = obs_prev.combat.closest_hostile_pos
            if closest_pos:
                hy, hx = closest_pos
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ny, nx = hero.y + dy, hero.x + dx
                    if 0 <= ny < 21 and 0 <= nx < 79 and walkable[ny, nx]:
                        d = math.hypot(ny - hy, nx - hx)
                        if d > max_dist:
                            max_dist = d
                            best_tile = (dy, dx)
            if best_tile:
                action = Action(name="step_direction", direction=best_tile)

        elif action.name == "melee_attack_hostile" and obs_prev is not None:
            hero = obs_prev.hero
            glyphs = obs_prev.glyphs
            if glyphs is not None:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ty, tx = hero.y + dy, hero.x + dx
                        if 0 <= ty < 21 and 0 <= tx < 79:
                            if (ty, tx) in self.peaceful_positions:
                                continue
                            g = int(glyphs[ty, tx])
                            if nethack.glyph_is_monster(g) and not nethack.glyph_is_pet(g):
                                action = Action(name="melee_attack", direction=(dy, dx))
                                break
            # If no adjacent monster, approach closest hostile if known
            if action.direction is None and obs_prev.combat.closest_hostile_pos:
                hy, hx = obs_prev.combat.closest_hostile_pos
                if (hy, hx) not in self.peaceful_positions:
                    walkable = build_walkable_mask(obs_prev.raw_obs)
                    for by, bx in self.blocked_tiles:
                        if 0 <= by < 21 and 0 <= bx < 79:
                            walkable[by, bx] = False
                    walkable[hy, hx] = True
                    path = SpatialEngine.find_path((hero.y, hero.x), (hy, hx), walkable)
                    if path:
                        dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                        action = Action(name="step_direction", direction=(dy, dx))

        # Translate direction or atomic action
        if action.direction is not None and action.direction in DIR_CHARS:
            self._last_attempted_dir = action.direction
            target_char = DIR_CHARS[action.direction]
        elif action.char is not None:
            target_char = action.char
        elif action.name == "descend":
            target_char = ">"
        elif action.name == "ascend":
            target_char = "<"
        elif action.name == "search":
            hero = obs_prev.hero if obs_prev else None
            if hero is not None:
                self.searched_count[hero.y, hero.x] += 1
            target_char = "s"
        elif action.name in ("wait", "rest"):
            target_char = "."
        elif action.name == "pray":
            obs, r, term, trunc, info = self._step_sequence([self.pray_action_idx])
            self.last_prayer_turn = obs.hero.turn
            return obs, r, term, trunc, info
        elif action.name in ("eat_carried_food", "eat_food"):
            slot = action.slot or (obs_prev.inventory.get_food_slot() if obs_prev else "a")
            return self._step_sequence([self.char_to_act.get("e", 0), self.char_to_act.get(slot or "a", 0)])
        elif action.name == "eat_floor_corpse":
            if obs_prev and (obs_prev.hero.y, obs_prev.hero.x) not in self.floor_corpses and self.floor_corpses:
                hero = obs_prev.hero
                nearest_pos = min(self.floor_corpses.keys(), key=lambda p: math.hypot(p[0] - hero.y, p[1] - hero.x))
                walkable = build_walkable_mask(obs_prev.raw_obs)
                walkable[nearest_pos[0], nearest_pos[1]] = True
                path = SpatialEngine.find_path((hero.y, hero.x), nearest_pos, walkable)
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    target_char = DIR_CHARS.get((dy, dx), ".")
                    self._last_attempted_dir = (dy, dx)
                    return self._step_sequence([self.char_to_act.get(target_char, 0)])
            return self._step_sequence([35, self.char_to_act.get("y", 0)])
        elif action.name in ("quaff_healing", "quaff"):
            slot = action.slot or (obs_prev.inventory.get_healing_slot() if obs_prev else "a")
            return self._step_sequence([self.char_to_act.get("q", 0), self.char_to_act.get(slot or "a", 0)])
        elif action.name == "read_scroll":
            slot = action.slot or "a"
            return self._step_sequence([self.char_to_act.get("r", 0), self.char_to_act.get(slot, 0)])
        elif action.name == "zap_wand":
            slot = action.slot or "a"
            dir_char = DIR_CHARS.get(action.direction, ".") if action.direction else "."
            return self._step_sequence([self.char_to_act.get("z", 0), self.char_to_act.get(slot, 0), self.char_to_act.get(dir_char, 0)])
        elif action.name == "open_door":
            dir_char = "l"
            found = False
            if action.direction:
                dir_char = DIR_CHARS.get(action.direction, "l")
                found = True
            elif obs_prev:
                hero = obs_prev.hero
                glyphs = obs_prev.glyphs
                doors_mask = self._get_doors_mask(glyphs)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= hero.y + dy < 21 and 0 <= hero.x + dx < 79:
                        if doors_mask[hero.y + dy, hero.x + dx]:
                            dir_char = DIR_CHARS[(dy, dx)]
                            found = True
                            break
            if not found:
                return self.step(Action(name="wait"))
            return self._step_sequence([57, self.char_to_act.get(dir_char, 0)])
        elif action.name == "kick_closed_door":
            dir_char = "l"
            found = False
            if action.direction:
                dir_char = DIR_CHARS.get(action.direction, "l")
                found = True
            elif obs_prev:
                hero = obs_prev.hero
                glyphs = obs_prev.glyphs
                doors_mask = self._get_doors_mask(glyphs)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= hero.y + dy < 21 and 0 <= hero.x + dx < 79:
                        if doors_mask[hero.y + dy, hero.x + dx]:
                            dir_char = DIR_CHARS[(dy, dx)]
                            found = True
                            break
            if not found:
                return self.step(Action(name="wait"))
            return self._step_sequence([48, self.char_to_act.get(dir_char, 0)])
        elif action.name == "wear_armor":
            slot = action.slot or (obs_prev.inventory.get_unworn_armor_slot() if obs_prev else None)
            if slot:
                return self._step_sequence([self.char_to_act.get("W", 0), self.char_to_act.get(slot, 0)])
            return self.step(Action(name="wait"))
        elif action.name == "pickup":
            target_char = ","
        elif action.name == "pay":
            target_char = "p"

        act_idx = self.char_to_act.get(target_char, self.char_to_act.get(".", 0))
        return self._step_sequence([act_idx])

    def close(self) -> None:
        self.env.close()
