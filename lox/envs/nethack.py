"""
LOX NetHack Adapter: High-performance, clean NLE wrapper.
Directly consumes structured NLE observation channels (blstats, inv_strs, inv_letters)
with zero regex scraping and automatic --More-- prompt dismissal.
Pure agentic autonomy: zero hardcoded safety interlocks or engine-level overrides.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from nle import nethack
from nle.nethack import permonst

from lox.core.agenda import GoalAgenda
from lox.core.epistemic import EpistemicEngine
from lox.core.spatial import SpatialEngine, build_walkable_mask
from lox.core.types import (
    Action,
    CombatView,
    DungeonView,
    EncumbranceState,
    EpistemicView,
    FloorCorpse,
    HeroState,
    HeroStatus,
    HungerState,
    InventoryView,
    Item,
    Observation,
    SpatialView,
)
from lox.core.digging import DiggingRouter
from lox.core.sokoban import SokobanSolver
from lox.envs.base import EnvironmentAdapter
from lox.envs.solvers.altar_solver import AltarBUCSolver
from lox.envs.solvers.castle_solver import CastleDrawbridgeSolver
from lox.envs.solvers.poison_solver import PoisonResHarvestSolver

# Direction character mapping for NetHack
DIR_CHARS: dict[tuple[int, int], str] = {
    (-1, 0): "k",  # North
    (0, 1): "l",  # East
    (1, 0): "j",  # South
    (0, -1): "h",  # West
    (-1, 1): "u",  # North-East
    (1, 1): "n",  # South-East
    (1, -1): "b",  # South-West
    (-1, -1): "y",  # North-West
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

IGNORES_ELBERETH_SPECIES: tuple[str, ...] = (
    "orc",
    "uruk",
    "elf",
    "human",
    "soldier",
    "guard",
    "captain",
    "watchman",
    "priest",
    "shopkeeper",
    "minotaur",
    "skeleton",
    "demon",
    "devil",
    "ghost",
    "goblin",
    "hobgoblin",
    "gnome",
    "dwarf",
    "giant",
    "ogre",
    "troll",
    "zombie",
    "mummy",
    "vampire",
    "shade",
    "wraith",
    "lich",
)

BRANCH_NAMES: dict[int, str] = {
    0: "dungeon",
    1: "gehennom",
    2: "mines",
    3: "quest",
    4: "sokoban",
    5: "fort_ludios",
    6: "vlad_tower",
    7: "planes",
}

_MAX_GLYPH: int = nethack.MAX_GLYPH

# Precomputed NumPy Boolean Lookup Tables (LUTs) for O(1) vectorized indexing
GLYPH_IS_MONSTER_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_PET_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_BODY_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_MON_HOSTILE_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_PASSIVE_HAZARD_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_PEACEFUL_SPECIES_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IGNORES_ELBERETH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_FAST_LUT = np.zeros(_MAX_GLYPH, dtype=bool)

# Metadata arrays/lists for fast lookup
GLYPH_MON_NAME: list[str] = [""] * _MAX_GLYPH
GLYPH_IS_FLOATING_EYE = np.zeros(_MAX_GLYPH, dtype=bool)
GLYPH_IS_GAS_SPORE = np.zeros(_MAX_GLYPH, dtype=bool)

# Precomputed body metadata
BODY_SPECIES_NAME: list[str] = [""] * _MAX_GLYPH
BODY_IS_POISONOUS = np.zeros(_MAX_GLYPH, dtype=bool)
BODY_IS_DEADLY = np.zeros(_MAX_GLYPH, dtype=bool)

for _g in range(_MAX_GLYPH):
    if nethack.glyph_is_monster(_g):
        GLYPH_IS_MONSTER_LUT[_g] = True
        if nethack.glyph_is_pet(_g):
            GLYPH_IS_PET_LUT[_g] = True
        else:
            GLYPH_IS_MON_HOSTILE_LUT[_g] = True
            _mid = nethack.glyph_to_mon(_g)
            try:
                _pm = permonst(_mid)
                _mname = _pm.mname
                GLYPH_MON_NAME[_g] = _mname
                _ml = _mname.lower()
                if (
                    _ml in ("floating eye", "gas spore", "yellow light", "black light")
                    or "mold" in _ml
                    or "jelly" in _ml
                    or "sphere" in _ml
                    or "light" in _ml
                ):
                    GLYPH_IS_PASSIVE_HAZARD_LUT[_g] = True
                if _ml == "floating eye":
                    GLYPH_IS_FLOATING_EYE[_g] = True
                elif _ml in ("gas spore", "yellow light", "black light") or "sphere" in _ml:
                    GLYPH_IS_GAS_SPORE[_g] = True

                if _ml in (
                    "watchman",
                    "watch captain",
                    "shopkeeper",
                    "guard",
                    "priest",
                    "priestess",
                    "aligned priest",
                    "high priest",
                    "oracle",
                ):
                    GLYPH_IS_PEACEFUL_SPECIES_LUT[_g] = True

                if any(_ign in _ml for _ign in IGNORES_ELBERETH_SPECIES):
                    GLYPH_IGNORES_ELBERETH_LUT[_g] = True

                if _pm.mmove > 12:
                    GLYPH_IS_FAST_LUT[_g] = True
            except Exception:
                GLYPH_MON_NAME[_g] = "monster"

    elif nethack.glyph_is_body(_g):
        GLYPH_IS_BODY_LUT[_g] = True
        _mid = _g - nethack.GLYPH_BODY_OFF
        try:
            _mname = (
                permonst(_mid).mname.lower()
                if 0 <= _mid < nethack.NUMMONS
                else "corpse"
            )
        except Exception:
            _mname = "corpse"
        BODY_SPECIES_NAME[_g] = _mname
        if any(
            _k in _mname
            for _k in ("poison", "kobold", "snake", "spider", "viper", "beetle")
        ):
            BODY_IS_POISONOUS[_g] = True
        if any(_k in _mname for _k in ("cockatrice", "chickatrice", "medusa")):
            BODY_IS_DEADLY[_g] = True


class NetHackAdapter(EnvironmentAdapter):
    """Clean NetHack adapter with unconstrained agentic execution."""

    def __init__(self, env_id: str = "NetHackChallenge-v0", role: str = "valkyrie"):
        self.env_id = env_id
        self.role = role
        self.options = ("autopickup", "pickup_thrown", "pickup_types:?!/%=[$*")
        self.env = gym.make(
            env_id,
            character=role,
            options=self.options,
        )
        self.visited = np.zeros((21, 79), dtype=bool)
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0

        # Memory tracking
        self.last_prayer_turn = -1000
        self.floor_corpses: dict[
            tuple[int, int], tuple[str, int, bool]
        ] = {}  # (y, x) -> (name, drop_turn, is_poisonous)
        self.known_stairs_down: tuple[int, int] | None = None
        self.known_stairs_up: tuple[int, int] | None = None
        self.stairs_down_discovery_turn: int = -1
        self.last_target_pos: tuple[int, int] | None = None
        self.searched_count = np.zeros((21, 79), dtype=np.int32)
        self.peaceful_positions: set[tuple[int, int]] = set()
        self.blocked_tiles: set[tuple[int, int]] = set()
        self.non_door_tiles: set[tuple[int, int]] = set()
        self.failed_wear_slots: set[str] = set()
        self.elbereth_positions: set[tuple[int, int]] = set()
        self.locked_doors: set[tuple[int, int]] = set()
        self.door_kick_count: dict[tuple[int, int], int] = {}
        self.hostile_npc_positions: set[tuple[int, int]] = set()
        self.looted_tiles: set[tuple[int, int]] = set()
        self.mines_stairs_positions: set[tuple[int, int, int]] = (
            set()
        )  # (y, x, depth) in Dungeons of Doom
        self._last_descended_stair: tuple[int, int, int] | None = None
        self.consecutive_zero_turns: int = 0
        self._prev_turn: int = 0
        self.known_chars = np.zeros((21, 79), dtype=np.uint8)

        # Epistemic POMDP Engine & Hybrid HTN-BT Goal Agenda
        self.epistemic = EpistemicEngine()
        self.agenda = GoalAgenda()
        self.altar_solver = AltarBUCSolver()
        self.poison_solver = PoisonResHarvestSolver()
        self.castle_solver = CastleDrawbridgeSolver()
        self.has_magic_res = False
        self.has_reflection = False
        self.consecutive_passive_waits = 0
        self.known_altar_pos: tuple[int, int] | None = None
        self.known_fountain_pos: tuple[int, int] | None = None

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
        self.pray_action_idx = (
            62 if len(self.actions) > 62 else self.char_to_act.get("#", 0)
        )
        self.dip_action_idx = (
            32
            if len(self.actions) > 32
            and getattr(self.actions[32], "value", int(self.actions[32])) == 228
            else self.char_to_act.get("#", 0)
        )
        self.last_prayer_turn: int = -1000

    def _get_doors_mask(self, glyphs: np.ndarray) -> np.ndarray:
        """Returns boolean mask of real closed doors using NLE CMAP glyphs and persistent door memory."""
        doors_mask = np.zeros((21, 79), dtype=bool)
        if glyphs is not None:
            doors_mask |= (glyphs == (nethack.GLYPH_CMAP_OFF + 15)) | (
                glyphs == (nethack.GLYPH_CMAP_OFF + 16)
            )
        if hasattr(self, "known_chars"):
            doors_mask |= self.known_chars == ord("+")
        for dy, dx in self.non_door_tiles:
            if 0 <= dy < 21 and 0 <= dx < 79:
                doors_mask[dy, dx] = False
        for by, bx in self.blocked_tiles:
            if 0 <= by < 21 and 0 <= bx < 79:
                doors_mask[by, bx] = False
        return doors_mask

    def _get_all_doors_mask(self, obs_or_raw: Any) -> np.ndarray:
        """Returns boolean mask of all doorways (open, closed, broken) to enforce cardinal movement."""
        glyphs = getattr(obs_or_raw, "glyphs", None)
        chars = getattr(obs_or_raw, "chars", None)
        if glyphs is None and isinstance(obs_or_raw, dict):
            glyphs = obs_or_raw.get("glyphs")
            chars = obs_or_raw.get("chars")
        door_mask = np.zeros((21, 79), dtype=bool)
        if glyphs is not None:
            door_mask |= (glyphs >= (nethack.GLYPH_CMAP_OFF + 13)) & (
                glyphs <= (nethack.GLYPH_CMAP_OFF + 16)
            )
        if chars is not None:
            door_mask |= chars == ord("+")
        if hasattr(self, "known_chars"):
            door_mask |= self.known_chars == ord("+")
        return door_mask

    def _build_walkable_nav(self, obs_or_raw: Any) -> tuple[np.ndarray, np.ndarray]:
        """Builds navigation mask with passable closed doors, excluding blocked tiles and shop/iron locked doors."""
        if hasattr(obs_or_raw, "raw_obs"):
            raw_obs = obs_or_raw.raw_obs
            glyphs = obs_or_raw.glyphs
            in_shop = getattr(obs_or_raw.dungeon, "in_shop", False)
        elif isinstance(obs_or_raw, dict):
            raw_obs = obs_or_raw
            glyphs = obs_or_raw.get("glyphs")
            in_shop = False
        else:
            return np.zeros((21, 79), dtype=bool), np.zeros((21, 79), dtype=bool)

        walkable = build_walkable_mask(raw_obs)
        if glyphs is not None:
            open_doors = (glyphs >= (nethack.GLYPH_CMAP_OFF + 12)) & (
                glyphs <= (nethack.GLYPH_CMAP_OFF + 14)
            )
            walkable[open_doors] = True
        doors_mask = (
            self._get_doors_mask(glyphs)
            if glyphs is not None
            else np.zeros((21, 79), dtype=bool)
        )
        walkable_nav = walkable.copy()
        walkable_nav[doors_mask] = True

        # Augment with persistent spatial memory of visited tiles and known walkable characters
        walkable[self.visited] = True
        walkable_nav[self.visited] = True

        if hasattr(self, "known_chars"):
            known_walkable = (
                (self.known_chars == ord("."))
                | (self.known_chars == ord("#"))
                | (self.known_chars == ord("<"))
                | (self.known_chars == ord(">"))
                | (self.known_chars == ord("_"))
                | (self.known_chars == ord("{"))
            )
            walkable[known_walkable] = True
            walkable_nav[known_walkable] = True

        # Extract hero position if available
        hy, hx = None, None
        if hasattr(obs_or_raw, "hero"):
            hy, hx = obs_or_raw.hero.y, obs_or_raw.hero.x
        elif isinstance(obs_or_raw, dict) and "blstats" in obs_or_raw:
            blstats = obs_or_raw["blstats"]
            hx, hy = int(blstats[0]), int(blstats[1])

        if hy is not None and 0 <= hy < 21 and 0 <= hx < 79:
            self.blocked_tiles.discard((hy, hx))
            self.non_door_tiles.discard((hy, hx))

        # Exclude permanently/dynamically blocked tiles (walls, iron bars, boulders, etc.)
        for by, bx in self.blocked_tiles:
            if 0 <= by < 21 and 0 <= bx < 79:
                walkable_nav[by, bx] = False

        # Exclude locked doors in shops to prevent angering shopkeeper
        if in_shop:
            for ly, lx in self.locked_doors:
                if 0 <= ly < 21 and 0 <= lx < 79:
                    walkable_nav[ly, lx] = False

        # Exclude non-door tiles
        for ndy, ndx in self.non_door_tiles:
            if 0 <= ndy < 21 and 0 <= ndx < 79:
                walkable_nav[ndy, ndx] = False

        # Exclude passive and exploding hazards from pathfinding navigation
        if glyphs is not None:
            valid_glyphs = np.clip(glyphs, 0, _MAX_GLYPH - 1)
            walkable_nav[GLYPH_IS_PASSIVE_HAZARD_LUT[valid_glyphs]] = False

        # Hero's current position is always walkable and can depart
        if hy is not None and 0 <= hy < 21 and 0 <= hx < 79:
            walkable[hy, hx] = True
            walkable_nav[hy, hx] = True

        return walkable, walkable_nav

    def _step_or_breach(
        self, obs: Observation, dy: int, dx: int
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Steps towards (dy, dx), or opens/kicks if destination is a closed door."""
        ny, nx = obs.hero.y + dy, obs.hero.x + dx
        doors_mask = self._get_doors_mask(obs.glyphs)
        if 0 <= ny < 21 and 0 <= nx < 79 and doors_mask[ny, nx]:
            if (ny, nx) in self.locked_doors:
                if (
                    not obs.dungeon.in_shop
                    and self.door_kick_count.get((ny, nx), 0) < 6
                ):
                    return self.step(
                        Action(name="kick_closed_door", direction=(dy, dx))
                    )
                else:
                    self.blocked_tiles.add((ny, nx))
                    if obs.spatial.has_unvisited_frontier:
                        return self.step(Action(name="step_to_frontier"))
                    return self.step(Action(name="step_to_dead_end"))
            return self.step(Action(name="open_door", direction=(dy, dx)))
        return self.step(Action(name="step_direction", direction=(dy, dx)))

    def _compute_dead_ends_mask(
        self,
        chars: np.ndarray,
        walkable: np.ndarray,
        max_corridor: int = 15,
        max_perimeter: int = 10,
    ) -> np.ndarray:
        """Unified dead end and perimeter secret door candidate mask."""
        effective_chars = chars.copy()
        if hasattr(self, "known_chars"):
            known_mask = self.known_chars > 0
            blank_mask = (chars == 0) | (chars == ord(" ")) | (chars == ord("@"))
            effective_chars[known_mask & blank_mask] = self.known_chars[
                known_mask & blank_mask
            ]
        return SpatialEngine.compute_dead_ends_mask(
            effective_chars, walkable, self.searched_count, max_corridor, max_perimeter
        )

    def is_target_floating_eye(self, glyphs: np.ndarray, y: int, x: int) -> bool:
        """Informational check: returns True if monster at (y, x) is a floating eye."""
        if not (0 <= y < 21 and 0 <= x < 79):
            return False
        g = int(glyphs[y, x])
        return 0 <= g < _MAX_GLYPH and bool(GLYPH_IS_FLOATING_EYE[g])

    def can_safely_pray(self, turn: int) -> bool:
        """Informational check: returns True if safe prayer cooldown (850 turns) has elapsed."""
        return (turn - self.last_prayer_turn) >= 850

    def _decode_message(self, msg_raw: Any) -> str:
        if isinstance(msg_raw, np.ndarray):
            return (
                bytes(msg_raw)
                .split(b"\x00", 1)[0]
                .decode("ascii", errors="ignore")
                .strip()
            )
        elif isinstance(msg_raw, (bytes, bytearray)):
            return msg_raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        elif isinstance(msg_raw, list):
            try:
                return (
                    bytes(msg_raw)
                    .split(b"\x00", 1)[0]
                    .decode("ascii", errors="ignore")
                    .strip()
                )
            except Exception:
                return "".join(str(c) for c in msg_raw).strip()
        return str(msg_raw).strip()

    def _extract_obs(self, raw_obs: dict[str, Any]) -> Observation:
        bl = raw_obs["blstats"]
        x, y = int(bl[0]), int(bl[1])
        score = int(bl[9]) if len(bl) > 9 else 0
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
        is_blind = (
            bool(cond & nethack.BL_MASK_BLIND)
            if hasattr(nethack, "BL_MASK_BLIND")
            else False
        )
        is_confused = (
            bool(cond & nethack.BL_MASK_CONF)
            if hasattr(nethack, "BL_MASK_CONF")
            else False
        )
        is_stunned = (
            bool(cond & nethack.BL_MASK_STUN)
            if hasattr(nethack, "BL_MASK_STUN")
            else False
        )
        is_hallucinating = (
            bool(cond & nethack.BL_MASK_HALLU)
            if hasattr(nethack, "BL_MASK_HALLU")
            else False
        )
        is_levitating = (
            bool(cond & nethack.BL_MASK_LEV)
            if hasattr(nethack, "BL_MASK_LEV")
            else False
        )
        is_sick = bool(
            cond
            & (
                getattr(nethack, "BL_MASK_FOODPOIS", 8)
                | getattr(nethack, "BL_MASK_TERMILL", 16)
            )
        )

        if depth != self.last_depth or dnum != self.last_dnum:
            if (
                branch_name == "mines"
                and getattr(self, "_last_descended_stair", None) is not None
            ):
                self.mines_stairs_positions.add(self._last_descended_stair)
            had_explicit_descent = (
                getattr(self, "_last_descended_stair", None) is not None
            )
            self._last_descended_stair = None
            self.turns_on_level = 0
            self.visited.fill(False)
            self.searched_count.fill(0)
            self.peaceful_positions.clear()
            self.blocked_tiles.clear()
            self.non_door_tiles.clear()
            self.locked_doors.clear()
            self.door_kick_count.clear()
            self.floor_corpses.clear()
            self.elbereth_positions.clear()
            self.looted_tiles.clear()
            self.known_fountain_pos = None
            self.known_altar_pos = None
            self.known_stairs_down = None
            self.known_stairs_up = (y, x) if had_explicit_descent else None
            self.stairs_down_discovery_turn = -1
            self.last_target_pos = None
            self.known_chars.fill(0)
            self.last_depth = depth
            self.last_dnum = dnum
        else:
            self.turns_on_level += 1

        self.visited[y, x] = True

        inventory_items: list[Item] = []
        inv_letters = raw_obs.get("inv_letters", [])
        inv_strs = raw_obs.get("inv_strs", [])
        inv_oclasses = raw_obs.get("inv_oclasses", [])

        for letter, desc_bytes, oclass in zip(inv_letters, inv_strs, inv_oclasses):
            if letter > 0:
                slot = chr(int(letter))
                desc = self._decode_message(desc_bytes)
                cat = OCLASS_MAP.get(int(oclass), "unknown")
                is_equipped = (
                    "weapon in hand" in desc
                    or "(being worn)" in desc
                    or "wielded" in desc
                )
                buc = "uncursed"
                if "cursed" in desc:
                    buc = "cursed"
                elif "blessed" in desc:
                    buc = "blessed"
                qty = 1
                words = desc.split()
                if words and words[0].isdigit():
                    qty = int(words[0])
                inventory_items.append(
                    Item(
                        slot=slot,
                        name=desc,
                        quantity=qty,
                        category=cat,
                        is_equipped=is_equipped,
                        buc=buc,
                    )
                )

        self.failed_wear_slots = {
            s
            for s in self.failed_wear_slots
            if any(it.slot == s for it in inventory_items)
        }
        inv_view = InventoryView(
            inventory_items, failed_armor_slots=self.failed_wear_slots
        )

        has_magic_res = getattr(self, "has_magic_res", False) or any(
            ("gray dragon scale" in it.name.lower() or "cloak of magic resistance" in it.name.lower())
            and it.is_equipped
            for it in inventory_items
        )
        has_reflection = getattr(self, "has_reflection", False) or any(
            ("silver dragon scale" in it.name.lower() or "shield of reflection" in it.name.lower() or "amulet of reflection" in it.name.lower())
            and it.is_equipped
            for it in inventory_items
        )

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
            score=score,
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
            has_poison_res=getattr(self, "has_poison_res", False),
            has_magic_res=has_magic_res,
            has_reflection=has_reflection,
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
        message = self._decode_message(raw_obs.get("message", ""))
        chars = raw_obs["chars"]
        glyphs = raw_obs["glyphs"]

        # Track clean underlying terrain characters
        if glyphs is not None and chars is not None:
            valid_glyphs = np.clip(glyphs, 0, _MAX_GLYPH - 1)
            valid_c = (
                (~GLYPH_IS_MONSTER_LUT[valid_glyphs])
                & (chars != ord("@"))
                & (chars != 0)
                & (chars != ord(" "))
            )
            self.known_chars[valid_c] = chars[valid_c]
            if self.known_chars[y, x] == 0:
                self.known_chars[y, x] = ord(".")

        # Track corpses on the floor from visible glyphs
        visible_corpse_positions = set()
        if glyphs is not None:
            valid_glyphs = np.clip(glyphs, 0, _MAX_GLYPH - 1)
            body_mask = GLYPH_IS_BODY_LUT[valid_glyphs]
            if np.any(body_mask):
                for gy, gx in np.argwhere(body_mask):
                    gy, gx = int(gy), int(gx)
                    visible_corpse_positions.add((gy, gx))
                    if (gy, gx) not in self.floor_corpses:
                        g = int(glyphs[gy, gx])
                        mname = BODY_SPECIES_NAME[g] if g < _MAX_GLYPH else "corpse"
                        is_pois = (
                            bool(BODY_IS_POISONOUS[g]) if g < _MAX_GLYPH else False
                        )
                        is_deadly = bool(BODY_IS_DEADLY[g]) if g < _MAX_GLYPH else False
                        self.floor_corpses[(gy, gx)] = (
                            mname,
                            turn,
                            is_pois,
                            is_deadly,
                        )
        # Prune corpses that have disappeared in hero's line of sight
        for cy, cx in list(self.floor_corpses.keys()):
            if (cy, cx) not in visible_corpse_positions and (
                abs(cy - y) <= 8
                and abs(cx - x) <= 8
                and glyphs is not None
                and not GLYPH_IS_BODY_LUT[
                    min(max(0, int(glyphs[cy, cx])), _MAX_GLYPH - 1)
                ]
            ):
                del self.floor_corpses[(cy, cx)]

        # Tactical combat analysis from glyphs
        adjacent_hostile = False
        hostile_count = 0
        closest_name = ""
        closest_dist = 999.0
        closest_pos = None
        floating_eye_fov = False
        gas_spore_fov = False
        adjacent_gas_spore = False
        adjacent_floating_eye = False
        adjacent_hostiles_count = 0
        active_hostile_count = 0

        if glyphs is not None:
            y_min, y_max = max(0, y - 8), min(21, y + 9)
            x_min, x_max = max(0, x - 8), min(79, x + 9)
            sub_g = glyphs[y_min:y_max, x_min:x_max]
            valid_sub = np.clip(sub_g, 0, _MAX_GLYPH - 1)
            hostile_mask = GLYPH_IS_MON_HOSTILE_LUT[valid_sub]
            if np.any(hostile_mask):
                for ry, rx in np.argwhere(hostile_mask):
                    gy, gx = y_min + int(ry), x_min + int(rx)
                    if gy == y and gx == x:
                        continue
                    if (gy, gx) in self.peaceful_positions:
                        continue
                    g = int(glyphs[gy, gx])
                    if g >= _MAX_GLYPH:
                        continue
                    if GLYPH_IS_PEACEFUL_SPECIES_LUT[g] and (gy, gx) not in getattr(
                        self, "hostile_npc_positions", set()
                    ):
                        self.peaceful_positions.add((gy, gx))
                        continue

                    mname = GLYPH_MON_NAME[g]
                    is_passive_hazard = bool(GLYPH_IS_PASSIVE_HAZARD_LUT[g])
                    is_adjacent = abs(gy - y) <= 1 and abs(gx - x) <= 1

                    if not is_passive_hazard or is_adjacent:
                        hostile_count += 1
                    if not is_passive_hazard:
                        active_hostile_count += 1
                    if GLYPH_IS_FLOATING_EYE[g]:
                        floating_eye_fov = True
                    if GLYPH_IS_GAS_SPORE[g]:
                        gas_spore_fov = True

                    if not is_passive_hazard or is_adjacent:
                        dist = math.hypot(gy - y, gx - x)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_name = mname
                            closest_pos = (gy, gx)

                    if is_adjacent:
                        adjacent_hostile = True
                        adjacent_hostiles_count += 1
                        if GLYPH_IS_GAS_SPORE[g]:
                            adjacent_gas_spore = True
                        if GLYPH_IS_FLOATING_EYE[g]:
                            adjacent_floating_eye = True

        adjacent_monsters: list[str] = []
        hostile_ignores_elbereth = False
        has_safe_melee_target = False
        if glyphs is not None:
            for dy, dx in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < 21
                    and 0 <= nx < 79
                    and (ny, nx) not in self.peaceful_positions
                ):
                    g = int(glyphs[ny, nx])
                    if 0 <= g < _MAX_GLYPH and GLYPH_IS_MON_HOSTILE_LUT[g]:
                        if GLYPH_IS_PEACEFUL_SPECIES_LUT[g] and (ny, nx) not in getattr(
                            self, "hostile_npc_positions", set()
                        ):
                            self.peaceful_positions.add((ny, nx))
                            continue
                        mname = GLYPH_MON_NAME[g]
                        adjacent_monsters.append(mname)
                        if GLYPH_IGNORES_ELBERETH_LUT[g]:
                            hostile_ignores_elbereth = True
                        if not GLYPH_IS_PASSIVE_HAZARD_LUT[g]:
                            has_safe_melee_target = True

        adjacent_peaceful = any(
            abs(py - y) <= 1 and abs(px - x) <= 1 for py, px in self.peaceful_positions
        )

        # Track stairs coordinates anywhere on the revealed map using CMAP glyphs and chars
        stair_down_glyph = nethack.GLYPH_CMAP_OFF + 24
        stair_up_glyph = nethack.GLYPH_CMAP_OFF + 23
        glyph_stairs_down = np.argwhere(glyphs == stair_down_glyph)
        stairs_candidates = [(int(pt[0]), int(pt[1])) for pt in glyph_stairs_down]
        if not stairs_candidates:
            chars_stairs_down = np.argwhere(chars == ord(">"))
            stairs_candidates = [(int(pt[0]), int(pt[1])) for pt in chars_stairs_down]

        if dnum == 0 and hasattr(self, "mines_stairs_positions"):
            valid_stairs = [
                pos
                for pos in stairs_candidates
                if (pos[0], pos[1], depth) not in self.mines_stairs_positions
            ]
        else:
            valid_stairs = stairs_candidates

        if valid_stairs:
            self.known_stairs_down = valid_stairs[0]
        elif (
            dnum == 0
            and hasattr(self, "mines_stairs_positions")
            and self.known_stairs_down is not None
        ) and (
            self.known_stairs_down[0],
            self.known_stairs_down[1],
            depth,
        ) in self.mines_stairs_positions:
            self.known_stairs_down = None

        if self.known_stairs_down is not None and self.stairs_down_discovery_turn == -1:
            self.stairs_down_discovery_turn = turn

        glyph_stairs_up = np.argwhere(glyphs == stair_up_glyph)
        if len(glyph_stairs_up) > 0:
            self.known_stairs_up = (
                int(glyph_stairs_up[0][0]),
                int(glyph_stairs_up[0][1]),
            )
        else:
            stairs_up = np.argwhere(chars == ord("<"))
            if len(stairs_up) > 0:
                self.known_stairs_up = (int(stairs_up[0][0]), int(stairs_up[0][1]))

        if any(
            msg in message.lower()
            for msg in (
                "can't go up here",
                "cannot go up here",
                "trap door",
                "fall through a trap",
            )
        ):
            self.known_stairs_up = None

        # Spatial topology & navigation analysis
        walkable, walkable_nav = self._build_walkable_nav(raw_obs)
        all_doors = self._get_all_doors_mask(raw_obs)
        if self.known_stairs_down:
            walkable_nav[self.known_stairs_down[0], self.known_stairs_down[1]] = True
        if self.known_stairs_up:
            walkable_nav[self.known_stairs_up[0], self.known_stairs_up[1]] = True

        doors_mask = self._get_doors_mask(glyphs)
        target_frontier_mask = (walkable & (~self.visited)) | doors_mask
        target_frontier_mask[y, x] = False
        frontier_tile = SpatialEngine.find_nearest_frontier(
            (y, x),
            walkable_nav,
            self.visited,
            target_mask=target_frontier_mask,
            is_door=all_doors,
        )
        has_frontier = frontier_tile is not None

        # Corridor detection
        walkable_adj = sum(
            1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= y + dy < 21 and 0 <= x + dx < 79 and walkable[y + dy, x + dx]
        )
        in_corridor = (walkable_adj <= 2 and chr(chars[y, x]) == "#") or bool(
            all_doors[y, x]
        )

        is_fast_dangerous = closest_name.lower() in (
            "soldier ant",
            "killer bee",
            "giant spider",
            "centipede",
            "giant bat",
            "bat",
            "fox",
            "coyote",
            "jaguar",
            "leocrotta",
        )
        if closest_pos and glyphs is not None:
            cg = int(glyphs[closest_pos[0], closest_pos[1]])
            if 0 <= cg < _MAX_GLYPH and GLYPH_IS_MON_HOSTILE_LUT[cg]:
                if GLYPH_IS_FAST_LUT[cg]:
                    is_fast_dangerous = True
        if closest_name and any(
            ign in closest_name.lower() for ign in IGNORES_ELBERETH_SPECIES
        ):
            hostile_ignores_elbereth = True
        combat = CombatView(
            adjacent_hostile=adjacent_hostile,
            hostile_count_fov=hostile_count,
            closest_hostile_name=closest_name,
            closest_hostile_dist=closest_dist,
            closest_hostile_pos=closest_pos,
            is_surrounded=(adjacent_hostiles_count >= 2),
            in_corridor=in_corridor,
            can_retreat=(walkable_adj > adjacent_hostiles_count),
            standing_on_elbereth=(
                "Elbereth" in message or (y, x) in self.elbereth_positions
            ),
            floating_eye_in_fov=floating_eye_fov,
            adjacent_peaceful=adjacent_peaceful,
            is_fast_dangerous=is_fast_dangerous,
            gas_spore_in_fov=gas_spore_fov,
            adjacent_gas_spore=adjacent_gas_spore,
            adjacent_floating_eye=adjacent_floating_eye,
            hostile_ignores_elbereth=hostile_ignores_elbereth,
            has_panic_escape=bool(
                inv_view.has_scroll_of_teleport or inv_view.has_wand_of_teleport
            ),
            has_safe_melee_target=has_safe_melee_target,
            has_active_hostile=(active_hostile_count > 0),
            active_hostile_count=active_hostile_count,
            adjacent_monsters=adjacent_monsters,
        )

        # Check for unsearched corridor dead ends or room perimeter tiles reachable from hero
        dead_ends_mask = self._compute_dead_ends_mask(chars, walkable)
        dead_end_target = SpatialEngine.find_nearest_target(
            (y, x), walkable_nav, target_mask=dead_ends_mask, is_door=all_doors
        )
        has_dead_ends = dead_end_target is not None and dead_end_target != (-1, -1)

        # Stagnation Auto-Recovery: If no visible frontiers or reachable dead ends remain and stairs are unknown,
        # decay search counts so the hero performs a fresh search sweep instead of freezing in place.
        if not has_frontier and not has_dead_ends and self.known_stairs_down is None:
            if turn - getattr(self, "last_search_decay_turn", -1000) >= 50:
                self.last_search_decay_turn = turn
                self.searched_count = np.maximum(0, self.searched_count - 10)
                dead_ends_mask = self._compute_dead_ends_mask(chars, walkable)
                dead_end_target = SpatialEngine.find_nearest_target(
                    (y, x), walkable_nav, target_mask=dead_ends_mask, is_door=all_doors
                )
                has_dead_ends = dead_end_target is not None and dead_end_target != (
                    -1,
                    -1,
                )

        # Nearby dropped loot discovery (armor, wands, potions, scrolls, rings, gold, food)
        has_nearby_loot = False
        nearby_loot_pos = None
        loot_chars = (
            ord("["),
            ord("!"),
            ord("?"),
            ord("/"),
            ord("="),
            ord("$"),
            ord("%"),
            ord("*"),
        )
        if int(chars[y, x]) in loot_chars:
            self.looted_tiles.add((y, x))
        if "shop" not in message.lower():
            loot_candidates = []
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if dy == 0 and dx == 0:
                        continue
                    ly, lx = y + dy, x + dx
                    if 0 <= ly < 21 and 0 <= lx < 79 and walkable_nav[ly, lx]:
                        if (
                            (ly, lx) not in self.blocked_tiles
                            and (ly, lx) not in self.looted_tiles
                            and int(chars[ly, lx]) in loot_chars
                        ):
                            if glyphs is not None:
                                lg = int(glyphs[ly, lx])
                                if 0 <= lg < _MAX_GLYPH and GLYPH_IS_MONSTER_LUT[lg]:
                                    continue
                            loot_candidates.append((ly, lx))
            if loot_candidates:
                has_nearby_loot = True
                closest_idx = int(
                    np.argmin(
                        [math.hypot(ly - y, lx - x) for ly, lx in loot_candidates]
                    )
                )
                nearby_loot_pos = loot_candidates[closest_idx]

        spatial = SpatialView(
            stairs_down_known=(self.known_stairs_down is not None),
            stairs_up_known=(self.known_stairs_up is not None),
            stairs_down_pos=self.known_stairs_down,
            stairs_up_pos=self.known_stairs_up,
            standing_on_stairs_down=(
                self.known_stairs_down == (y, x) or chr(chars[y, x]) == ">"
            ),
            standing_on_stairs_up=(
                self.known_stairs_up == (y, x) or chr(chars[y, x]) == "<"
            ),
            standing_on_elbereth=(
                "Elbereth" in message or (y, x) in self.elbereth_positions
            ),
            standing_on_dead_end=bool(dead_ends_mask[y, x]),
            has_unvisited_frontier=has_frontier,
            has_unsearched_dead_end=has_dead_ends,
            unvisited_frontier_count=int(np.sum(walkable_nav & (~self.visited))),
            dead_ends_count=int(np.sum(dead_ends_mask)),
            target_pos=self.last_target_pos,
            floor_explored=(not has_frontier and self.known_stairs_down is not None),
            has_nearby_loot=has_nearby_loot,
            nearby_loot_pos=nearby_loot_pos,
        )

        # Dungeon tile type
        curr_char = chr(chars[y, x])
        curr_glyph = int(glyphs[y, x])
        tile_type = "room"
        if curr_char == "#":
            tile_type = "corridor"
        elif nethack.glyph_is_cmap(curr_glyph) and nethack.glyph_to_cmap(
            curr_glyph
        ) in (12, 13, 14, 15, 16):
            tile_type = "doorway"
        elif (
            nethack.glyph_is_cmap(curr_glyph)
            and nethack.glyph_to_cmap(curr_glyph) == 31
        ):
            tile_type = "fountain"
        elif (
            nethack.glyph_is_cmap(curr_glyph)
            and nethack.glyph_to_cmap(curr_glyph) == 27
        ):
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
            if 0 <= ny < 21 and 0 <= nx < 79 and doors_mask[ny, nx]:
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

        door_is_locked = bool(
            adj_door
            and (
                any(
                    (y + dy, x + dx) in self.locked_doors
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                or "locked" in message.lower()
                or "won't open" in message.lower()
            )
        )

        # Fountains in FOV
        msg_lower = message.lower()
        fountain_vanished = any(
            k in msg_lower
            for k in ("disappears", "dries up", "boils away", "silly thing to dip")
        )
        if fountain_vanished:
            self.known_fountain_pos = None

        fountain_in_fov = False
        closest_fountain_pos = None
        fountain_coords = np.argwhere(chars == ord("{"))
        if len(fountain_coords) > 0:
            fountain_in_fov = True
            closest_idx = int(
                np.argmin([math.hypot(fy - y, fx - x) for fy, fx in fountain_coords])
            )
            closest_fountain_pos = (
                int(fountain_coords[closest_idx][0]),
                int(fountain_coords[closest_idx][1]),
            )
            self.known_fountain_pos = closest_fountain_pos
        elif getattr(self, "known_fountain_pos", None) is not None:
            kfy, kfx = self.known_fountain_pos
            if fountain_vanished:
                self.known_fountain_pos = None
            elif (kfy, kfx) == (y, x):
                # Hero is standing directly on the fountain tile; hero '@' occludes '{'
                fountain_in_fov = True
                closest_fountain_pos = self.known_fountain_pos
            elif (
                abs(kfy - y) <= 1 and abs(kfx - x) <= 1 and chr(chars[kfy, kfx]) != "{"
            ):
                # Adjacent and fountain is genuinely gone
                self.known_fountain_pos = None
            else:
                fountain_in_fov = True
                closest_fountain_pos = self.known_fountain_pos

        doors_coords = np.argwhere(doors_mask)
        has_closed_door = len(doors_coords) > 0
        closed_door_in_fov = False
        closest_door_pos = None
        if has_closed_door:
            closest_door_idx = int(
                np.argmin([math.hypot(dy - y, dx - x) for dy, dx in doors_coords])
            )
            closest_door_pos = (
                int(doors_coords[closest_door_idx][0]),
                int(doors_coords[closest_door_idx][1]),
            )
            closed_door_in_fov = (
                abs(closest_door_pos[0] - y) <= 8 and abs(closest_door_pos[1] - x) <= 8
            )

        # Altars in FOV
        if curr_char == "_":
            self.known_altar_pos = (y, x)
        else:
            altar_coords = np.argwhere(chars == ord("_"))
            if len(altar_coords) > 0:
                self.known_altar_pos = (
                    int(altar_coords[0][0]),
                    int(altar_coords[0][1]),
                )

        closest_drawbridge = CastleDrawbridgeSolver.detect_drawbridge(chars, message)
        dungeon = DungeonView(
            tile_type=tile_type,
            in_shop=("shop" in message.lower()),
            in_temple=("temple" in message.lower()),
            is_dark_level=(branch_name == "mines"),
            dungeon_branch=branch_name,
            adjacent_closed_door=adj_door,
            adjacent_open_door=False,
            has_closed_door=has_closed_door,
            closed_door_in_fov=closed_door_in_fov,
            closest_door_pos=closest_door_pos,
            door_is_locked=door_is_locked,
            adjacent_fountain=adj_fountain,
            fountain_in_fov=fountain_in_fov,
            closest_fountain_pos=closest_fountain_pos,
            standing_on_fountain=(
                not fountain_vanished
                and (
                    getattr(self, "known_fountain_pos", None) == (y, x)
                    or curr_char == "{"
                )
            ),
            adjacent_altar=adj_altar,
            standing_on_altar=(self.known_altar_pos == (y, x) or curr_char == "_"),
            adjacent_trap=adj_trap,
            standing_on_trap=(curr_char == "^"),
            can_forge_excalibur=can_forge,
            is_sokoban=(branch_name == "sokoban"),
            has_boulders=bool(chars is not None and np.any(chars == ord("0"))),
            drawbridge_in_fov=(closest_drawbridge is not None),
            closest_drawbridge_pos=closest_drawbridge,
        )

        # Register inventory items in Epistemic POMDP engine
        for it in inventory_items:
            self.epistemic.get_or_create(
                uid=it.name,
                name=it.name,
                item_class=it.category,
                slot_letter=it.slot,
            )
        untested_count = self.epistemic.count_untested_buc()

        can_wear_armor = True
        if inv_view.has_unworn_armor:
            arm_slot = inv_view.get_unworn_armor_slot()
            for it in inv_view:
                if it.slot == arm_slot:
                    can_wear_armor = self.epistemic.can_safely_wear(
                        it.name, max_cursed_prob=0.15
                    )
                    break

        can_quaff_heal = True
        heal_slot = inv_view.get_healing_slot()
        if heal_slot:
            for it in inv_view:
                if it.slot == heal_slot:
                    can_quaff_heal = self.epistemic.can_safely_quaff(it.name)
                    break

        epistemic_view = EpistemicView(
            untested_buc_count=untested_count,
            has_untested_items=(untested_count > 0),
            can_safely_wear_armor=can_wear_armor,
            can_safely_quaff_healing=can_quaff_heal,
            items_belief=self.epistemic.beliefs,
        )

        agenda_view = self.agenda.create_view()

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

        dungeon.can_harvest_poison = self.poison_solver.can_harvest_from_corpses(
            corpse_list,
            has_poison_res=getattr(self, "has_poison_res", False),
            hostile_count=combat.hostile_count_fov,
            adjacent_hostile=combat.adjacent_hostile,
        )

        obs = Observation(
            chars=chars,
            glyphs=glyphs,
            hero=hero,
            inventory=inv_view,
            status=status,
            combat=combat,
            spatial=spatial,
            dungeon=dungeon,
            epistemic=epistemic_view,
            agenda=agenda_view,
            corpses=corpse_list,
            message=message,
            raw_obs=raw_obs,
        )
        self.agenda.evaluate_milestones(obs)
        return obs

    def _dismiss_more(
        self,
        raw_obs: dict[str, Any],
        term: bool,
        trunc: bool,
        is_intermediate: bool = False,
    ) -> tuple[dict[str, Any], bool, bool]:
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
                # NetHack ONLY prompts this when prayer timeout has not elapsed or deity is angry.
                # Answering 'n' aborts the unsafe prayer and completely avoids divine smiting!
                raw_obs, _, term, trunc, _ = self.env.step(
                    self.char_to_act.get("n", space_idx)
                )
            elif (
                any(
                    phrase in msg.lower()
                    for phrase in ("eat it?", "eat that?", "eat one?")
                )
                or ("dip" in msg.lower() and "fountain" in msg.lower())
            ):
                raw_obs, _, term, trunc, _ = self.env.step(
                    self.char_to_act.get("y", space_idx)
                )
            elif not is_intermediate and any(
                phrase in msg.lower()
                for phrase in (
                    "eat what?",
                    "what do you want to eat",
                    "drop what?",
                    "what do you want to drop",
                    "throw what?",
                    "what do you want to throw",
                    "dip what?",
                    "what do you want to dip",
                    "zap what?",
                    "what do you want to zap",
                    "read what?",
                    "what do you want to read",
                    "wear what?",
                    "what do you want to wear",
                    "take off what?",
                    "what do you want to take off",
                    "what do you want to put on",
                    "in what direction",
                )
            ):
                if (
                    "what do you want to dip" in msg.lower()
                    or "dip into what" in msg.lower()
                ):
                    self.known_fountain_pos = None
                esc_idx = self.char_to_act.get("\x1b", 38)
                raw_obs, _, term, trunc, _ = self.env.step(esc_idx)
            elif (
                "(y/n)" in msg
                or "[yn" in msg
                or "creatures vanquished" in msg.lower()
                or "really attack" in msg.lower()
                or "really quit" in msg.lower()
                or "possessions identified" in msg.lower()
            ):
                if "really attack" in msg.lower() or "peaceful" in msg.lower():
                    if getattr(self, "_last_attempted_dir", None) is not None:
                        py, px = getattr(self, "_prev_hero_pos", (0, 0))
                        dy, dx = self._last_attempted_dir
                        self.peaceful_positions.add((py + dy, px + dx))
                raw_obs, _, term, trunc, _ = self.env.step(
                    self.char_to_act.get("n", space_idx)
                )
            elif any(
                phrase in msg.lower()
                for phrase in (
                    "who are you",
                    "what is your name",
                    "call this",
                    "call a ",
                    "call an ",
                    "call the ",
                    "what do you want to call",
                    "hello stranger",
                )
            ):
                if (
                    any(
                        phrase in msg.lower()
                        for phrase in (
                            "who are you",
                            "what is your name",
                            "hello stranger",
                        )
                    )
                    and getattr(self, "_last_attempted_dir", None) is not None
                ):
                    py, px = getattr(self, "_prev_hero_pos", (0, 0))
                    dy, dx = self._last_attempted_dir
                    self.peaceful_positions.add((py + dy, px + dx))
                esc_idx = self.char_to_act.get("\x1b", 38)
                raw_obs, _, term, trunc, _ = self.env.step(esc_idx)
            else:
                break
            if term or trunc:
                break
        return raw_obs, term, trunc

    def _step_sequence(
        self, action_indices: list[int]
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        term = False
        trunc = False
        info: dict[str, Any] = {}
        raw_obs = getattr(self, "_last_raw_obs", {})
        for idx, act in enumerate(action_indices):
            if term or trunc:
                break
            raw_obs, r, term, trunc, info = self.env.step(act)
            total_reward += float(r)
            is_intermediate = idx < len(action_indices) - 1
            raw_obs, term, trunc = self._dismiss_more(
                raw_obs, term, trunc, is_intermediate=is_intermediate
            )
            if term or trunc:
                break
        self._last_raw_obs = raw_obs
        obs = self._extract_obs(raw_obs)
        self._last_obs = obs

        msg = obs.message.lower()
        all_doors = self._get_all_doors_mask(obs)
        if (
            any(
                f"{npc} hits" in msg
                for npc in (
                    "watchman",
                    "shopkeeper",
                    "watch captain",
                    "priest",
                    "priestess",
                    "guard",
                )
            )
            and getattr(self, "_last_attempted_dir", None) is not None
        ):
            py, px = getattr(self, "_prev_hero_pos", (0, 0))
            dy, dx = self._last_attempted_dir
            target_tile = (py + dy, px + dx)
            self.hostile_npc_positions.add(target_tile)
            self.peaceful_positions.discard(target_tile)
        if (
            "really attack" in msg
            or "who are you" in msg
            or "hello stranger" in msg
            or "follow me" in msg
        ) and getattr(self, "_last_attempted_dir", None) is not None:
            py, px = getattr(self, "_prev_hero_pos", (0, 0))
            dy, dx = self._last_attempted_dir
            self.peaceful_positions.add((py + dy, px + dx))
        if (
            "cannot pass through" in msg
            or "it's a wall" in msg
            or "it's solid stone" in msg
            or "cannot move there" in msg
        ) and getattr(self, "_last_attempted_dir", None) is not None:
            py, px = getattr(self, "_prev_hero_pos", (0, 0))
            dy, dx = self._last_attempted_dir
            target_tile = (py + dy, px + dx)
            if (
                0 <= target_tile[0] < 21
                and 0 <= target_tile[1] < 79
                and not all_doors[target_tile[0], target_tile[1]]
            ):
                self.blocked_tiles.add(target_tile)
        if "you see no door there" in msg or "cannot open that" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.non_door_tiles.add((py + dy, px + dx))
        if "locked" in msg or "won't open" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.locked_doors.add((py + dy, px + dx))
        # Clear locked/blocked status when door opens or gives way
        if (
            any(
                w in msg
                for w in (
                    "gives way",
                    "crash!",
                    "smashes",
                    "breaks",
                    "destroyed",
                    "door opens",
                )
            )
            and getattr(self, "_last_attempted_dir", None) is not None
        ):
            py, px = getattr(self, "_prev_hero_pos", (0, 0))
            dy, dx = self._last_attempted_dir
            target_tile = (py + dy, px + dx)
            self.locked_doors.discard(target_tile)
            self.blocked_tiles.discard(target_tile)
        if "iron door" in msg:
            if getattr(self, "_last_attempted_dir", None) is not None:
                py, px = getattr(self, "_prev_hero_pos", (0, 0))
                dy, dx = self._last_attempted_dir
                self.blocked_tiles.add((py + dy, px + dx))
        if "feel healthy" in msg:
            self.has_poison_res = True
        if any(
            w in msg
            for w in (
                "wipe out the message",
                "wiped out",
                "rubbed out",
                "scuffed",
                "erased",
                "fades",
                "vanishes",
            )
        ):
            self.elbereth_positions.discard((obs.hero.y, obs.hero.x))

        # Check turn advancement to shield against 0-turn infinite loops
        prev_turn = getattr(self, "_prev_turn", 0)
        curr_turn = obs.hero.turn
        self._prev_turn = curr_turn
        is_step_direction = getattr(self, "_last_action_name", "") in (
            "step_direction",
            "step_to_chokepoint",
            "step_to_frontier",
            "step_to_dead_end",
            "step_away_from_hostile",
            "retreat",
            "open_door",
            "kick_closed_door",
        )
        if curr_turn == prev_turn:
            self.consecutive_zero_turns += 1
            if (
                self.consecutive_zero_turns >= 2
                and is_step_direction
                and getattr(self, "_last_attempted_dir", None) is not None
            ):
                py, px = getattr(self, "_prev_hero_pos", (obs.hero.y, obs.hero.x))
                dy, dx = self._last_attempted_dir
                target_tile = (py + dy, px + dx)
                if 0 <= target_tile[0] < 21 and 0 <= target_tile[1] < 79:
                    self.blocked_tiles.add(target_tile)
        else:
            self.consecutive_zero_turns = 0

        # Shield against impassable obstacles where turn advanced but hero position didn't change
        # ONLY apply to directional steps (step_direction), NOT interaction actions like kicking/opening doors
        if is_step_direction and getattr(self, "_last_attempted_dir", None) is not None:
            py, px = getattr(self, "_prev_hero_pos", (obs.hero.y, obs.hero.x))
            if (obs.hero.y, obs.hero.x) == (py, px):
                if not obs.combat.adjacent_hostile and not obs.combat.adjacent_peaceful:
                    dy, dx = self._last_attempted_dir
                    target_tile = (py + dy, px + dx)
                    if 0 <= target_tile[0] < 21 and 0 <= target_tile[1] < 79:
                        if all_doors[target_tile[0], target_tile[1]]:
                            self.locked_doors.add(target_tile)
                        else:
                            consec_failed = (
                                getattr(self, "_consecutive_failed_steps", 0) + 1
                            )
                            self._consecutive_failed_steps = consec_failed
                            if consec_failed >= 2:
                                self.blocked_tiles.add(target_tile)
            else:
                self._consecutive_failed_steps = 0
        else:
            self._consecutive_failed_steps = 0

        self.blocked_tiles.discard((obs.hero.y, obs.hero.x))
        self.non_door_tiles.discard((obs.hero.y, obs.hero.x))

        return obs, total_reward, bool(term), bool(trunc), info

    def reset(self, seed: int | None = None) -> Observation:
        self.visited.fill(False)
        self.searched_count.fill(0)
        self.peaceful_positions.clear()
        self.blocked_tiles.clear()
        self.non_door_tiles.clear()
        self.failed_wear_slots.clear()
        self.elbereth_positions.clear()
        self.locked_doors.clear()
        self.door_kick_count.clear()
        self.hostile_npc_positions.clear()
        self.looted_tiles.clear()
        self.mines_stairs_positions.clear()
        self._last_descended_stair = None
        self.has_poison_res = False
        self.has_magic_res = False
        self.has_reflection = False
        self.consecutive_passive_waits = 0
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0
        self.consecutive_zero_turns = 0
        self._consecutive_failed_steps = 0
        self._prev_turn = 0
        self.last_prayer_turn = -1000
        self.floor_corpses.clear()
        self.known_stairs_down = None
        self.known_stairs_up = None
        self.stairs_down_discovery_turn = -1
        self.last_target_pos = None
        self.known_chars.fill(0)
        self.epistemic = EpistemicEngine()
        self.agenda = GoalAgenda()
        self.altar_solver.reset()
        self.castle_solver.reset()
        self.known_altar_pos = None
        self.known_fountain_pos = None
        raw_obs, _ = self.env.reset(seed=seed)
        raw_obs, _, _ = self._dismiss_more(raw_obs, False, False)
        self._last_raw_obs = raw_obs
        obs = self._extract_obs(raw_obs)
        self._last_obs = obs
        return obs

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Executes action without safety interception.
        Supports high-level navigation, combat, items, and atomic keyboard commands.
        """
        obs_prev = getattr(self, "_last_obs", None)
        self._prev_hero_pos = (obs_prev.hero.y, obs_prev.hero.x) if obs_prev else (0, 0)
        self._last_attempted_dir = action.direction
        self._last_action_name = action.name
        target_char = "."

        # Break any consecutive 0-turn loop before NLE aborts at 2500
        if getattr(self, "consecutive_zero_turns", 0) >= 4:
            self.consecutive_zero_turns = 0
            act_idx = self.char_to_act.get(".", 0)
            return self._step_sequence([act_idx])

        # Handle composite navigation actions
        if action.name == "step_to_frontier" and obs_prev is not None:
            hero = obs_prev.hero
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            doors_mask = self._get_doors_mask(obs_prev.glyphs)
            all_doors = self._get_all_doors_mask(obs_prev)
            if self.known_stairs_down:
                walkable_nav[self.known_stairs_down[0], self.known_stairs_down[1]] = (
                    True
                )
            target_mask = (walkable & (~self.visited)) | doors_mask
            target_mask[hero.y, hero.x] = False
            frontier = SpatialEngine.find_nearest_frontier(
                (hero.y, hero.x),
                walkable_nav,
                self.visited,
                target_mask=target_mask,
                is_door=all_doors,
            )
            self.last_target_pos = frontier
            if frontier:
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), frontier, walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    return self._step_or_breach(obs_prev, dy, dx)
            # If no reachable frontier, investigate dead ends and perimeter walls
            return self.step(Action(name="step_to_dead_end"))

        elif (
            action.name == "step_to_stairs_down"
            and obs_prev is not None
            and self.known_stairs_down
        ):
            if getattr(obs_prev.hero, "dungeon_branch", "") == "mines":
                if obs_prev.spatial.standing_on_stairs_up:
                    return self.step(Action(name="ascend"))
                elif self.known_stairs_up:
                    return self.step(Action(name="step_to_stairs_up"))
                return self.step(Action(name="step_to_frontier"))

            hero = obs_prev.hero
            self.last_target_pos = self.known_stairs_down
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            all_doors = self._get_all_doors_mask(obs_prev)
            walkable_nav[self.known_stairs_down[0], self.known_stairs_down[1]] = True
            path = SpatialEngine.find_path(
                (hero.y, hero.x),
                self.known_stairs_down,
                walkable_nav,
                is_door=all_doors,
            )
            if path:
                dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                return self._step_or_breach(obs_prev, dy, dx)
            elif (hero.y, hero.x) == self.known_stairs_down:
                return self.step(Action(name="descend"))
            else:
                # Stairs are unreachable directly (behind closed door or secret wall).
                # Continue exploring frontiers or searching perimeter walls to open the route!
                if obs_prev.spatial.has_unvisited_frontier:
                    return self.step(Action(name="step_to_frontier"))
                return self.step(Action(name="step_to_dead_end"))

        elif (
            action.name == "step_to_stairs_up"
            and obs_prev is not None
            and self.known_stairs_up
        ):
            hero = obs_prev.hero
            self.last_target_pos = self.known_stairs_up
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            all_doors = self._get_all_doors_mask(obs_prev)
            walkable_nav[self.known_stairs_up[0], self.known_stairs_up[1]] = True
            path = SpatialEngine.find_path(
                (hero.y, hero.x), self.known_stairs_up, walkable_nav, is_door=all_doors
            )
            if path:
                dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                return self._step_or_breach(obs_prev, dy, dx)
            elif (hero.y, hero.x) == self.known_stairs_up:
                return self.step(Action(name="ascend"))
            else:
                if obs_prev.spatial.has_unvisited_frontier:
                    return self.step(Action(name="step_to_frontier"))
                return self.step(Action(name="step_to_dead_end"))

        elif action.name == "step_to_fountain" and obs_prev is not None:
            hero = obs_prev.hero
            fountain_pos = obs_prev.dungeon.closest_fountain_pos or getattr(
                self, "known_fountain_pos", None
            )
            self.last_target_pos = fountain_pos
            if fountain_pos:
                if (hero.y, hero.x) == fountain_pos:
                    return self.step(Action(name="dip_excalibur"))
                walkable, walkable_nav = self._build_walkable_nav(obs_prev)
                all_doors = self._get_all_doors_mask(obs_prev)
                walkable_nav[fountain_pos[0], fountain_pos[1]] = True
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), fountain_pos, walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    return self._step_or_breach(obs_prev, dy, dx)
                else:
                    return self.step(Action(name="search"))
            return self.step(Action(name="step_to_frontier"))
        elif action.name == "step_to_closed_door" and obs_prev is not None:
            hero = obs_prev.hero
            glyphs = obs_prev.glyphs
            doors_mask = self._get_doors_mask(glyphs)
            all_doors = self._get_all_doors_mask(obs_prev)
            # If already adjacent to a closed door, breach or open it immediately
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = hero.y + dy, hero.x + dx
                if 0 <= ny < 21 and 0 <= nx < 79 and doors_mask[ny, nx]:
                    if (ny, nx) in self.locked_doors and not obs_prev.dungeon.in_shop:
                        return self.step(
                            Action(name="kick_closed_door", direction=(dy, dx))
                        )
                    return self.step(Action(name="open_door", direction=(dy, dx)))

            # Otherwise, navigate towards the nearest closed door
            closed_doors = np.argwhere(doors_mask)
            if len(closed_doors) > 0:
                walkable, walkable_nav = self._build_walkable_nav(obs_prev)
                target = SpatialEngine.find_nearest_target(
                    (hero.y, hero.x),
                    walkable_nav,
                    target_mask=doors_mask,
                    is_door=all_doors,
                )
                self.last_target_pos = target
                if target and target != (-1, -1):
                    path = SpatialEngine.find_path(
                        (hero.y, hero.x), target, walkable_nav, is_door=all_doors
                    )
                    if path:
                        dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                        return self._step_or_breach(obs_prev, dy, dx)
            if obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif action.name == "step_to_dead_end" and obs_prev is not None:
            hero = obs_prev.hero
            chars = obs_prev.chars
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            all_doors = self._get_all_doors_mask(obs_prev)
            dead_ends_mask = self._compute_dead_ends_mask(chars, walkable)

            # Exclude current tile so step_to_dead_end navigates rather than standing still
            step_target_mask = dead_ends_mask.copy()
            step_target_mask[hero.y, hero.x] = False

            # Find nearest reachable target in dead_ends_mask
            target = None
            if np.any(step_target_mask):
                candidates = np.argwhere(step_target_mask)
                min_searches = min(self.searched_count[cy, cx] for cy, cx in candidates)
                min_mask = step_target_mask & (self.searched_count <= min_searches + 2)
                target = SpatialEngine.find_nearest_target(
                    (hero.y, hero.x),
                    walkable_nav,
                    target_mask=min_mask,
                    is_door=all_doors,
                )
                if not target or target == (-1, -1):
                    target = SpatialEngine.find_nearest_target(
                        (hero.y, hero.x),
                        walkable_nav,
                        target_mask=step_target_mask,
                        is_door=all_doors,
                    )

            # If no corridor dead end found, search for any unexhausted wall-adjacent perimeter tile
            if not target or target == (-1, -1):
                wall_adj_mask = np.zeros((21, 79), dtype=bool)
                for cy in range(21):
                    for cx in range(79):
                        if (
                            walkable[cy, cx]
                            and (cy, cx) != (hero.y, hero.x)
                            and self.searched_count[cy, cx] < 10
                        ):
                            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < 21 and 0 <= nx < 79:
                                    ch = (
                                        int(self.known_chars[ny, nx])
                                        if hasattr(self, "known_chars")
                                        and self.known_chars[ny, nx] > 0
                                        else int(chars[ny, nx])
                                    )
                                    if ch in (ord("-"), ord("|"), ord(" "), 0):
                                        wall_adj_mask[cy, cx] = True
                                        break
                if np.any(wall_adj_mask):
                    target = SpatialEngine.find_nearest_target(
                        (hero.y, hero.x),
                        walkable_nav,
                        target_mask=wall_adj_mask,
                        is_door=all_doors,
                    )

            # If still no reachable target found, decay search counts if stagnant (rate limited to 50 turns)
            if not target or target == (-1, -1):
                if (
                    not hasattr(self, "_last_search_decay_turn")
                    or obs_prev.hero.turn - self._last_search_decay_turn >= 50
                ):
                    self._last_search_decay_turn = obs_prev.hero.turn
                    self.searched_count = np.maximum(0, self.searched_count - 10)
                    dead_ends_mask = self._compute_dead_ends_mask(chars, walkable)
                    step_target_mask = dead_ends_mask.copy()
                    step_target_mask[hero.y, hero.x] = False
                    if np.any(step_target_mask):
                        target = SpatialEngine.find_nearest_target(
                            (hero.y, hero.x),
                            walkable_nav,
                            target_mask=step_target_mask,
                            is_door=all_doors,
                        )

            self.last_target_pos = target if (target and target != (-1, -1)) else None

            if target and target != (-1, -1):
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), target, walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    return self._step_or_breach(obs_prev, dy, dx)

            # If no dead end target or wall target is reachable, step to any walkable tile to prevent standing still
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = hero.y + dy, hero.x + dx
                if 0 <= ny < 21 and 0 <= nx < 79 and walkable_nav[ny, nx]:
                    return self._step_or_breach(obs_prev, dy, dx)
            return self.step(Action(name="wait"))

        elif action.name == "step_to" and obs_prev is not None:
            hero = obs_prev.hero
            target = (
                action.target_pos or action.extra.get("target_pos") or action.direction
            )
            if target and isinstance(target, tuple) and len(target) == 2:
                ty, tx = int(target[0]), int(target[1])
                self.last_target_pos = (ty, tx)
                walkable, walkable_nav = self._build_walkable_nav(obs_prev)
                all_doors = self._get_all_doors_mask(obs_prev)
                walkable_nav[ty, tx] = True
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), (ty, tx), walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    return self._step_or_breach(obs_prev, dy, dx)
                elif (hero.y, hero.x) == (ty, tx):
                    return self.step(Action(name="wait"))
                else:
                    return self.step(Action(name="search"))

        elif action.name == "step_to_altar" and obs_prev is not None:
            if self.known_altar_pos:
                return self.step(
                    Action(name="step_to", target_pos=self.known_altar_pos)
                )
            return self.step(Action(name="search"))

        elif action.name == "test_altar_buc" and obs_prev is not None:
            sub_act = self.altar_solver.plan_step(
                obs_prev, self.epistemic, self.known_altar_pos
            )
            if sub_act:
                return self.step(sub_act)
            if (
                obs_prev.spatial.standing_on_stairs_down
                and not obs_prev.status.is_levitating
            ):
                return self.step(Action(name="descend"))
            elif obs_prev.spatial.stairs_down_known:
                return self.step(Action(name="step_to_stairs_down"))
            elif obs_prev.dungeon.adjacent_closed_door:
                return self.step(Action(name="open_door"))
            elif obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif action.name == "harvest_poison_res" and obs_prev is not None:
            sub_act = self.poison_solver.plan_step(obs_prev)
            if sub_act:
                return self.step(sub_act)
            if (
                obs_prev.spatial.standing_on_stairs_down
                and not obs_prev.status.is_levitating
            ):
                return self.step(Action(name="descend"))
            elif obs_prev.spatial.stairs_down_known:
                return self.step(Action(name="step_to_stairs_down"))
            elif obs_prev.dungeon.adjacent_closed_door:
                return self.step(Action(name="open_door"))
            elif obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif action.name == "solve_sokoban" and obs_prev is not None:
            chars = obs_prev.chars
            hero = obs_prev.hero
            boulder_mask = (chars == ord("0"))
            pit_mask = (chars == ord("^"))
            wall_mask = (chars == ord("-")) | (chars == ord("|"))
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            clean_floor = walkable | pit_mask
            push_info = SokobanSolver.find_best_boulder_push(
                (hero.y, hero.x), wall_mask, boulder_mask, pit_mask, clean_floor
            )
            if push_info is not None:
                if push_info["is_ready_to_push"]:
                    return self.step(
                        Action(name="step_direction", direction=push_info["push_dir"])
                    )
                else:
                    all_doors = self._get_all_doors_mask(obs_prev)
                    path = SpatialEngine.find_path(
                        (hero.y, hero.x),
                        push_info["hero_stand_pos"],
                        walkable_nav,
                        is_door=all_doors,
                    )
                    if path:
                        dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                        return self._step_or_breach(obs_prev, dy, dx)
            if obs_prev.spatial.stairs_up_known:
                return self.step(Action(name="step_to_stairs_up"))
            elif obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif action.name == "breach_drawbridge" and obs_prev is not None:
            sub_act = self.castle_solver.plan_step(
                obs_prev, obs_prev.dungeon.closest_drawbridge_pos
            )
            if sub_act:
                return self.step(sub_act)
            if (
                obs_prev.spatial.standing_on_stairs_down
                and not obs_prev.status.is_levitating
            ):
                return self.step(Action(name="descend"))
            elif obs_prev.spatial.stairs_down_known:
                return self.step(Action(name="step_to_stairs_down"))
            elif obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif action.name == "dig_tunnel" and obs_prev is not None:
            hero = obs_prev.hero
            target_pos = (
                action.target_pos
                or action.extra.get("target_pos")
                or self.known_stairs_down
                or self.known_stairs_up
                or self.last_target_pos
            )
            if target_pos is None:
                target_pos = (hero.y, hero.x + 1)
            dig_dir = DiggingRouter.get_cardinal_tunnel_direction(
                (hero.y, hero.x), target_pos
            )
            wand_slot = next(
                (
                    it.slot
                    for it in obs_prev.inventory
                    if "wand of digging" in it.name.lower()
                ),
                None,
            )
            if wand_slot:
                dir_char = DIR_CHARS.get(dig_dir, ".")
                return self._step_sequence(
                    [
                        self.char_to_act.get("z", 0),
                        self.char_to_act.get(wand_slot, 0),
                        self.char_to_act.get(dir_char, 0),
                    ]
                )
            pick_slot = next(
                (
                    it.slot
                    for it in obs_prev.inventory
                    if "pick-axe" in it.name.lower() or "mattock" in it.name.lower()
                ),
                None,
            )
            if pick_slot:
                dir_char = DIR_CHARS.get(dig_dir, ".")
                return self._step_sequence(
                    [
                        self.char_to_act.get("a", 0),
                        self.char_to_act.get(pick_slot, 0),
                        self.char_to_act.get(dir_char, 0),
                    ]
                )
            return self.step(Action(name="step_to_frontier"))

        elif action.name == "step_to_chokepoint" and obs_prev is not None:
            hero = obs_prev.hero
            chars = obs_prev.chars
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            all_doors = self._get_all_doors_mask(obs_prev)
            # Chokepoints are corridor tiles (#) or doorways, excluding blocked tiles and unbreachable locked doors
            chokepoint_mask = (chars == ord("#")) | all_doors
            chokepoint_mask[hero.y, hero.x] = False
            for by, bx in self.blocked_tiles:
                if 0 <= by < 21 and 0 <= bx < 79:
                    chokepoint_mask[by, bx] = False
            for ly, lx in self.locked_doors:
                if (
                    0 <= ly < 21
                    and 0 <= lx < 79
                    and (
                        obs_prev.dungeon.in_shop
                        or self.door_kick_count.get((ly, lx), 0) >= 6
                    )
                ):
                    chokepoint_mask[ly, lx] = False
            chokepoint = SpatialEngine.find_nearest_target(
                (hero.y, hero.x),
                walkable_nav,
                target_mask=chokepoint_mask,
                is_door=all_doors,
            )
            if chokepoint and chokepoint != (-1, -1):
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), chokepoint, walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    return self._step_or_breach(obs_prev, dy, dx)
            return self.step(Action(name="step_away_from_hostile"))

        elif action.name == "step_to_loot" and obs_prev is not None:
            hero = obs_prev.hero
            target_pos = (
                action.target_pos
                or action.extra.get("target_pos")
                or getattr(obs_prev.spatial, "nearby_loot_pos", None)
            )
            if target_pos:
                if (hero.y, hero.x) == target_pos:
                    self.looted_tiles.add(target_pos)
                    if obs_prev.spatial.has_unvisited_frontier:
                        return self.step(Action(name="step_to_frontier"))
                    return self.step(Action(name="step_to_dead_end"))
                walkable, walkable_nav = self._build_walkable_nav(obs_prev)
                all_doors = self._get_all_doors_mask(obs_prev)
                walkable_nav[target_pos[0], target_pos[1]] = True
                path = SpatialEngine.find_path(
                    (hero.y, hero.x), target_pos, walkable_nav, is_door=all_doors
                )
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    if (hero.y + dy, hero.x + dx) == target_pos:
                        self.looted_tiles.add(target_pos)
                    return self._step_or_breach(obs_prev, dy, dx)
            if obs_prev.spatial.has_unvisited_frontier:
                return self.step(Action(name="step_to_frontier"))
            return self.step(Action(name="step_to_dead_end"))

        elif (
            action.name in ("step_away_from_hostile", "retreat")
            and obs_prev is not None
        ):
            hero = obs_prev.hero
            walkable, walkable_nav = self._build_walkable_nav(obs_prev)
            all_doors = self._get_all_doors_mask(obs_prev)
            glyphs = obs_prev.glyphs
            if glyphs is not None:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = hero.y + dy, hero.x + dx
                        if 0 <= ny < 21 and 0 <= nx < 79:
                            g = int(glyphs[ny, nx])
                            if 0 <= g < _MAX_GLYPH and GLYPH_IS_MON_HOSTILE_LUT[g]:
                                walkable_nav[ny, nx] = False

            best_tile = None
            best_dist = -1.0
            best_is_open = False
            closest_pos = obs_prev.combat.closest_hostile_pos
            if closest_pos:
                hy, hx = closest_pos
                curr_dist = math.hypot(hero.y - hy, hero.x - hx)
                dirs = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
                doors_mask = self._get_doors_mask(obs_prev.glyphs)
                for i, (dy, dx) in enumerate(dirs):
                    ny, nx = hero.y + dy, hero.x + dx
                    if 0 <= ny < 21 and 0 <= nx < 79 and walkable_nav[ny, nx]:
                        if i >= 4:
                            # Diagonal doorway restriction
                            if all_doors[hero.y, hero.x] or all_doors[ny, nx]:
                                continue
                            if (
                                not walkable_nav[hero.y, nx]
                                or not walkable_nav[ny, hero.x]
                            ):
                                continue
                        if (ny, nx) in self.locked_doors and obs_prev.dungeon.in_shop:
                            continue
                        d = math.hypot(ny - hy, nx - hx)
                        if d > curr_dist:
                            is_open = not doors_mask[ny, nx]
                            if (is_open and not best_is_open) or (
                                is_open == best_is_open and d > best_dist
                            ):
                                best_dist = d
                                best_tile = (dy, dx)
                                best_is_open = is_open

            if best_tile:
                self.consecutive_passive_waits = 0
                return self._step_or_breach(obs_prev, best_tile[0], best_tile[1])
            elif not obs_prev.combat.closest_hostile_pos:
                self.consecutive_passive_waits = 0
                # No hostile in sight: safely fallback to navigation or search instead of engraving/waiting
                if obs_prev.spatial.stairs_down_known:
                    return self.step(Action(name="step_to_stairs_down"))
                elif obs_prev.spatial.has_unvisited_frontier:
                    return self.step(Action(name="step_to_frontier"))
                else:
                    return self.step(Action(name="step_to_dead_end"))
            else:
                closest_name = getattr(
                    obs_prev.combat, "closest_hostile_name", ""
                ).lower()
                is_passive = (
                    closest_name in ("floating eye", "gas spore")
                    or "mold" in closest_name
                    or "jelly" in closest_name
                    or "sphere" in closest_name
                    or getattr(obs_prev.combat, "gas_spore_in_fov", False)
                    or getattr(obs_prev.combat, "adjacent_floating_eye", False)
                    or getattr(obs_prev.combat, "adjacent_gas_spore", False)
                )
                if is_passive:
                    self.consecutive_passive_waits = (
                        getattr(self, "consecutive_passive_waits", 0) + 1
                    )
                    # If we've already waited once or twice and cannot retreat away:
                    if self.consecutive_passive_waits >= 2:
                        self.consecutive_passive_waits = 0
                        # 1. Try any walkable adjacent tile that is not the monster's tile
                        if closest_pos:
                            hy, hx = closest_pos
                            for dy, dx in (
                                (-1, 0),
                                (1, 0),
                                (0, -1),
                                (0, 1),
                                (-1, -1),
                                (-1, 1),
                                (1, -1),
                                (1, 1),
                            ):
                                ny, nx = hero.y + dy, hero.x + dx
                                if (
                                    0 <= ny < 21
                                    and 0 <= nx < 79
                                    and walkable_nav[ny, nx]
                                    and (ny, nx) != (hy, hx)
                                ):
                                    return self._step_or_breach(obs_prev, dy, dx)

                        # 2. Ranged destruction (safe elimination)
                        if obs_prev.inventory.has_daggers:
                            return self.step(Action(name="throw_dagger"))
                        elif obs_prev.inventory.has_offensive_wand:
                            return self.step(Action(name="zap_offensive_wand"))

                        # 3. Search for secret exit if at dead end
                        if (
                            obs_prev.spatial.standing_on_dead_end
                            and getattr(self, "passive_search_count", 0) < 6
                        ):
                            self.passive_search_count = (
                                getattr(self, "passive_search_count", 0) + 1
                            )
                            return self.step(Action(name="search"))

                        # 4. Emergency strike to break deadlock (strictly prohibited against floating eyes to prevent passive paralysis)
                        if closest_pos:
                            hy, hx = closest_pos
                            is_eye = (
                                self.is_target_floating_eye(obs_prev.glyphs, hy, hx)
                                or "floating eye" in closest_name
                                or getattr(obs_prev.combat, "adjacent_floating_eye", False)
                            )
                            if not is_eye and abs(hy - hero.y) <= 1 and abs(hx - hero.x) <= 1:
                                return self.step(
                                    Action(
                                        name="melee_attack",
                                        direction=(hy - hero.y, hx - hero.x),
                                    )
                                )

                    return self.step(Action(name="wait"))
                is_on_elbereth = (
                    (hero.y, hero.x) in self.elbereth_positions
                    or getattr(obs_prev.combat, "standing_on_elbereth", False)
                    or getattr(obs_prev.spatial, "standing_on_elbereth", False)
                )
                if not is_on_elbereth:
                    return self.step(Action(name="engrave_dust_elbereth"))
                else:
                    if getattr(obs_prev.combat, "adjacent_gas_spore", False) or getattr(
                        obs_prev.combat, "adjacent_floating_eye", False
                    ):
                        return self.step(Action(name="wait"))
                    return self.step(Action(name="melee_attack_hostile"))

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
                            if (ty, tx) in self.peaceful_positions or (
                                ty,
                                tx,
                            ) in self.blocked_tiles:
                                continue
                            g = int(glyphs[ty, tx])
                            if 0 <= g < _MAX_GLYPH and GLYPH_IS_MON_HOSTILE_LUT[g]:
                                if GLYPH_IS_FLOATING_EYE[g] or GLYPH_IS_GAS_SPORE[g]:
                                    continue
                                action = Action(name="melee_attack", direction=(dy, dx))
                                break
            # If no adjacent non-passive monster, do not approach floating eye / gas spore in melee
            if action.direction is None:
                # Cornered last resort: Trapped with adjacent floating eye and cannot retreat (Invariant 14)
                if (
                    getattr(obs_prev.combat, "adjacent_floating_eye", False)
                    and not getattr(obs_prev.combat, "can_retreat", True)
                    and glyphs is not None
                ):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ty, tx = hero.y + dy, hero.x + dx
                            if 0 <= ty < 21 and 0 <= tx < 79:
                                g = int(glyphs[ty, tx])
                                if 0 <= g < _MAX_GLYPH and GLYPH_IS_FLOATING_EYE[g]:
                                    return self.step(
                                        Action(name="melee_attack", direction=(dy, dx))
                                    )

                closest_name = getattr(obs_prev.combat, "closest_hostile_name", "")
                if (
                    closest_name in ("floating eye", "gas spore")
                    or getattr(obs_prev.combat, "adjacent_floating_eye", False)
                    or getattr(obs_prev.combat, "adjacent_gas_spore", False)
                ):
                    return self.step(Action(name="step_away_from_hostile"))
                elif obs_prev.combat.closest_hostile_pos:
                    hy, hx = obs_prev.combat.closest_hostile_pos
                    if (hy, hx) not in self.peaceful_positions:
                        walkable, walkable_nav = self._build_walkable_nav(obs_prev)
                        all_doors = self._get_all_doors_mask(obs_prev)
                        walkable_nav[hy, hx] = True
                        path = SpatialEngine.find_path(
                            (hero.y, hero.x), (hy, hx), walkable_nav, is_door=all_doors
                        )
                        if path:
                            dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                            return self._step_or_breach(obs_prev, dy, dx)

        # Intercept directional steps into closed/locked doors to breach instead of bumping
        if (
            action.name == "step_direction"
            and action.direction
            and obs_prev is not None
        ):
            dy, dx = action.direction
            ny, nx = obs_prev.hero.y + dy, obs_prev.hero.x + dx
            doors_mask = self._get_doors_mask(obs_prev.glyphs)
            if 0 <= ny < 21 and 0 <= nx < 79 and doors_mask[ny, nx]:
                if (ny, nx) in self.locked_doors and not obs_prev.dungeon.in_shop:
                    return self.step(
                        Action(name="kick_closed_door", direction=(dy, dx))
                    )
                return self.step(Action(name="open_door", direction=(dy, dx)))

        # Translate direction or atomic action
        if action.direction is not None and action.direction in DIR_CHARS:
            self._last_attempted_dir = action.direction
            target_char = DIR_CHARS[action.direction]
        elif action.char is not None:
            target_char = action.char
        elif action.name == "descend":
            if obs_prev and getattr(obs_prev.status, "is_levitating", False):
                return self.step(Action(name="wait"))
            if obs_prev and getattr(obs_prev.hero, "dungeon_branch", "") == "mines":
                if obs_prev.spatial.standing_on_stairs_up:
                    return self.step(Action(name="ascend"))
                elif self.known_stairs_up:
                    return self.step(Action(name="step_to_stairs_up"))
                return self.step(Action(name="step_to_frontier"))
            if obs_prev and hasattr(obs_prev, "hero"):
                self._last_descended_stair = (
                    obs_prev.hero.y,
                    obs_prev.hero.x,
                    obs_prev.hero.depth,
                )
            target_char = ">"
        elif action.name == "ascend":
            if obs_prev and getattr(obs_prev.status, "is_levitating", False):
                return self.step(Action(name="wait"))
            if (
                obs_prev
                and getattr(self, "consecutive_zero_turns", 0) >= 2
                and getattr(self, "_last_action_name", "") == "ascend"
            ):
                self.known_stairs_up = None
                return self.step(Action(name="step_to_frontier"))
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
            msg_low = obs.message.lower()
            if "displeased" in msg_low or "anger" in msg_low:
                self.last_prayer_turn = obs.hero.turn + 500
            elif "decide not to pray" in msg_low or "never mind" in msg_low:
                # Cancelled by shield because timeout was still active
                self.last_prayer_turn = obs.hero.turn - 500
            else:
                self.last_prayer_turn = obs.hero.turn
            return obs, r, term, trunc, info
        elif action.name in ("eat_carried_food", "eat_food"):
            if obs_prev and (
                obs_prev.status.encumbrance_level >= EncumbranceState.STRAINED
                or "carrying so much stuff" in getattr(obs_prev, "message", "").lower()
            ):
                unworn = next(
                    (
                        it
                        for it in obs_prev.inventory
                        if it.category == "armor" and not it.is_equipped
                    ),
                    None,
                )
                if unworn:
                    return self.step(Action(name="drop", slot=unworn.slot))
                heavy = next(
                    (
                        it
                        for it in obs_prev.inventory
                        if not it.is_equipped and it.category != "food"
                    ),
                    None,
                )
                if heavy:
                    return self.step(Action(name="drop", slot=heavy.slot))
            slot = action.slot or (
                obs_prev.inventory.get_food_slot() if obs_prev else "a"
            )
            return self._step_sequence(
                [self.char_to_act.get("e", 0), self.char_to_act.get(slot or "a", 0)]
            )
        elif action.name == "eat_floor_corpse":
            hero = obs_prev.hero if obs_prev else None
            if (
                hero
                and (hero.y, hero.x) not in self.floor_corpses
                and self.floor_corpses
            ):
                nearest_pos = min(
                    self.floor_corpses.keys(),
                    key=lambda p: math.hypot(p[0] - hero.y, p[1] - hero.x),
                )
                walkable = build_walkable_mask(obs_prev.raw_obs)
                walkable[nearest_pos[0], nearest_pos[1]] = True
                path = SpatialEngine.find_path((hero.y, hero.x), nearest_pos, walkable)
                if path:
                    dy, dx = path[0][0] - hero.y, path[0][1] - hero.x
                    target_char = DIR_CHARS.get((dy, dx), ".")
                    self._last_attempted_dir = (dy, dx)
                    return self._step_sequence([self.char_to_act.get(target_char, 0)])
            obs, r, term, trunc, info = self._step_sequence([35])
            if hero and (hero.y, hero.x) in self.floor_corpses:
                self.floor_corpses.pop((hero.y, hero.x), None)
            return obs, r, term, trunc, info
        elif action.name in ("quaff_healing", "quaff"):
            slot = action.slot or (
                obs_prev.inventory.get_healing_slot() if obs_prev else "a"
            )
            return self._step_sequence(
                [self.char_to_act.get("q", 0), self.char_to_act.get(slot or "a", 0)]
            )
        elif action.name in ("read_scroll_teleport", "read_scroll"):
            slot = action.slot
            if not slot and action.name == "read_scroll_teleport" and obs_prev:
                slot = obs_prev.inventory.get_scroll_of_teleport_slot()
            slot = slot or "a"
            return self._step_sequence(
                [self.char_to_act.get("r", 0), self.char_to_act.get(slot, 0)]
            )
        elif action.name == "zap_offensive_wand" and obs_prev is not None:
            hero = obs_prev.hero
            slot = action.slot or obs_prev.inventory.get_offensive_wand_slot()
            target_pos = (
                action.target_pos
                or action.extra.get("target_pos")
                or obs_prev.combat.closest_hostile_pos
            )
            if target_pos and target_pos in self.peaceful_positions:
                return self.step(Action(name="wait"))
            if slot and target_pos:
                dy = int(np.sign(target_pos[0] - hero.y))
                dx = int(np.sign(target_pos[1] - hero.x))
                dir_char = DIR_CHARS.get((dy, dx), ".")
                return self._step_sequence(
                    [
                        self.char_to_act.get("z", 0),
                        self.char_to_act.get(slot, 0),
                        self.char_to_act.get(dir_char, 0),
                    ]
                )
            return self.step(Action(name="melee_attack_hostile"))
        elif action.name == "zap_wand_teleport":
            slot = action.slot or (
                obs_prev.inventory.get_wand_of_teleport_slot() if obs_prev else "a"
            )
            return self._step_sequence(
                [
                    self.char_to_act.get("z", 0),
                    self.char_to_act.get(slot or "a", 0),
                    self.char_to_act.get(".", 0),
                ]
            )
        elif action.name == "zap_wand":
            slot = action.slot or "a"
            dir_char = DIR_CHARS.get(action.direction, ".") if action.direction else "."
            return self._step_sequence(
                [
                    self.char_to_act.get("z", 0),
                    self.char_to_act.get(slot, 0),
                    self.char_to_act.get(dir_char, 0),
                ]
            )
        elif action.name == "open_door":
            dir_char = "l"
            found = False
            target_door = None
            if action.direction:
                dir_char = DIR_CHARS.get(action.direction, "l")
                self._last_attempted_dir = action.direction
                if obs_prev:
                    target_door = (
                        obs_prev.hero.y + action.direction[0],
                        obs_prev.hero.x + action.direction[1],
                    )
                found = True
            elif obs_prev:
                hero = obs_prev.hero
                glyphs = obs_prev.glyphs
                doors_mask = self._get_doors_mask(glyphs)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= hero.y + dy < 21 and 0 <= hero.x + dx < 79:
                        if doors_mask[hero.y + dy, hero.x + dx]:
                            dir_char = DIR_CHARS[(dy, dx)]
                            self._last_attempted_dir = (dy, dx)
                            target_door = (hero.y + dy, hero.x + dx)
                            found = True
                            break
            if not found:
                return self.step(Action(name="wait"))
            # If the door is already known to be locked, automatically kick it to breach
            if (
                target_door
                and target_door in self.locked_doors
                and obs_prev
                and not obs_prev.dungeon.in_shop
            ):
                return self.step(
                    Action(name="kick_closed_door", direction=self._last_attempted_dir)
                )
            return self._step_sequence([57, self.char_to_act.get(dir_char, 0)])
        elif action.name == "kick_closed_door":
            dir_char = "l"
            found = False
            target_door = None
            if action.direction:
                dir_char = DIR_CHARS.get(action.direction, "l")
                self._last_attempted_dir = action.direction
                if obs_prev:
                    target_door = (
                        obs_prev.hero.y + action.direction[0],
                        obs_prev.hero.x + action.direction[1],
                    )
                found = True
            elif obs_prev:
                hero = obs_prev.hero
                glyphs = obs_prev.glyphs
                doors_mask = self._get_doors_mask(glyphs)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= hero.y + dy < 21 and 0 <= hero.x + dx < 79:
                        if doors_mask[hero.y + dy, hero.x + dx]:
                            dir_char = DIR_CHARS[(dy, dx)]
                            self._last_attempted_dir = (dy, dx)
                            target_door = (hero.y + dy, hero.x + dx)
                            found = True
                            break
            if not found:
                return self.step(Action(name="wait"))
            if target_door:
                self.door_kick_count[target_door] = (
                    self.door_kick_count.get(target_door, 0) + 1
                )
                if self.door_kick_count[target_door] >= 6:
                    self.blocked_tiles.add(target_door)
            return self._step_sequence([48, self.char_to_act.get(dir_char, 0)])
        elif action.name == "wear_armor":
            slot = action.slot or (
                obs_prev.inventory.get_unworn_armor_slot() if obs_prev else None
            )
            if slot:
                obs, reward, term, trunc, info = self._step_sequence(
                    [self.char_to_act.get("W", 0), self.char_to_act.get(slot, 0)]
                )
                item = next((it for it in obs.inventory if it.slot == slot), None)
                if item is not None and not item.is_equipped:
                    self.failed_wear_slots.add(slot)
                    obs.inventory.failed_armor_slots.add(slot)
                return obs, reward, term, trunc, info
            return self.step(Action(name="wait"))
        elif action.name == "apply_unicorn_horn":
            slot = action.slot or (
                obs_prev.inventory.get_unicorn_horn_slot() if obs_prev else None
            )
            if slot:
                return self._step_sequence(
                    [self.char_to_act.get("a", 0), self.char_to_act.get(slot, 0)]
                )
            return self.step(Action(name="wait"))
        elif (
            action.name in ("throw_dagger", "throw_item", "fire_missile")
            and obs_prev is not None
        ):
            hero = obs_prev.hero
            target_pos = (
                action.target_pos
                or action.extra.get("target_pos")
                or obs_prev.combat.closest_hostile_pos
            )
            if target_pos and target_pos in self.peaceful_positions:
                return self.step(Action(name="wait"))
            slot = action.slot or obs_prev.inventory.get_dagger_slot()
            if target_pos and slot:
                dy = int(np.sign(target_pos[0] - hero.y))
                dx = int(np.sign(target_pos[1] - hero.x))
                dir_char = DIR_CHARS.get((dy, dx), "l")
                return self._step_sequence(
                    [
                        self.char_to_act.get("t", 0),
                        self.char_to_act.get(slot, 0),
                        self.char_to_act.get(dir_char, 0),
                    ]
                )
            return self.step(Action(name="melee_attack_hostile"))

        elif action.name in ("engrave_dust_elbereth", "engrave_elbereth", "engrave"):
            if obs_prev:
                self.elbereth_positions.add((obs_prev.hero.y, obs_prev.hero.x))
            enter_idx = self.char_to_act.get("\r", 19)
            seq = [
                self.char_to_act.get("E", 0),
                self.char_to_act.get("-", 0),
            ]
            for ch in "Elbereth":
                seq.append(self.char_to_act.get(ch, 0))
            seq.append(enter_idx)
            obs, reward, term, trunc, info = self._step_sequence(seq)
            return obs, reward, term, trunc, info

        elif (
            action.name in ("dip_excalibur", "dip_in_fountain") and obs_prev is not None
        ):
            if not obs_prev.dungeon.standing_on_fountain and (
                obs_prev.dungeon.closest_fountain_pos
                or getattr(self, "known_fountain_pos", None)
            ):
                return self.step(Action(name="step_to_fountain"))
            sword_slot = obs_prev.inventory.get_weapon_slot()
            if not sword_slot:
                for it in obs_prev.inventory:
                    if "long sword" in it.name.lower():
                        sword_slot = it.slot
                        break
            if sword_slot:
                dip_idx = getattr(self, "dip_action_idx", 32)
                seq = [
                    dip_idx,
                    self.char_to_act.get(sword_slot, 0),
                ]
                return self._step_sequence(seq)
            return self.step(Action(name="wait"))

        elif action.name == "drop":
            slot = action.slot or "a"
            obs, reward, term, trunc, info = self._step_sequence(
                [self.char_to_act.get("d", 0), self.char_to_act.get(slot, 0)]
            )
            if obs_prev and obs_prev.dungeon.standing_on_altar:
                for it in obs_prev.inventory:
                    if it.slot == slot:
                        self.epistemic.update_from_altar_drop(
                            it.name, obs.hero.is_blind, obs.message
                        )
                        break
            return obs, reward, term, trunc, info

        elif action.name == "pickup":
            target_char = ","
        elif action.name == "pay":
            target_char = "p"

        act_idx = self.char_to_act.get(target_char, self.char_to_act.get(".", 0))
        return self._step_sequence([act_idx])

    def close(self) -> None:
        self.env.close()
