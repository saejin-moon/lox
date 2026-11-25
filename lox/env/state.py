"""
Lazy, Zero-Copy NetHack GameState ("Hard Code Nothing").
Exposes 100% of NLE's raw observation channels and native C symbol tables
(permonst, objclass, glyph classification, BL_MASK conditions) on demand
with zero per-tick computation when not queried.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from nle import nethack as _nle_nh

from lox.env.grid_utils import astar_next_step, bfs_distance_grid
from lox.env.memory import FloorMemory, WorldMemory
from lox.env.types import (
    _ASCII_TRANS_TABLE,
    _CHAR_LUT,
    _CMAP_NAME_BY_GLYPH,
    _CORPSE_MON_BY_GLYPH,
    _FLOOR_GLYPH_FLAGS_LUT,
    _HAS_RUST_FLOOR_MEM,
    _IS_BODY_LUT,
    _IS_BOULDER_GLYPH_LUT,
    _IS_CLOSED_DOOR_GLYPH_LUT,
    _IS_CMAP_LUT,
    _IS_DOORWAY_GLYPH_LUT,
    _IS_DRAWBRIDGE_GLYPH_LUT,
    _IS_EXIT_CMAP_LUT,
    _IS_HAZARD_CMAP_LUT,
    _IS_ICE_GLYPH_LUT,
    _IS_IRON_BARS_OR_TREE_LUT,
    _IS_LAVA_GLYPH_LUT,
    _IS_LOWERED_DRAWBRIDGE_GLYPH_LUT,
    _IS_MONSTER_LUT,
    _IS_OBJECT_LUT,
    _IS_OPEN_DOOR_GLYPH_LUT,
    _IS_PASSABLE_CMAP_LUT,
    _IS_PET_LUT,
    _IS_PORTAL_GLYPH_LUT,
    _IS_SWALLOW_LUT,
    _IS_TRAP_LUT,
    _IS_WALL_GLYPH_LUT,
    _IS_WARNING_LUT,
    _IS_WATER_GLYPH_LUT,
    _MAX_GLYPH,
    _MG_INVIS,
    _MG_PET,
    _MG_RIDDEN,
    _MG_STATUE,
    _MON_INFO_BY_GLYPH,
    _MON_INFO_BY_NAME,
    _OBJ_INFO_BY_GLYPH,
    _OBJ_INFO_BY_NAME,
    _OCLASS_NAMES,
    _OCLASS_TO_CHAR,
    _TRAP_NAME_BY_GLYPH,
    InventoryItem,
    MonsterInfo,
    ObjectInfo,
    VisibleMonster,
    VisibleObject,
    _CallableList,
    _make_mon_info,
    _make_obj_info,
    lookup_object_info_by_name,
    lox_core,
)

nh: Any = _nle_nh


class GameState:
    """
    Zero-copy, lazy game state view passed to the Policy on every step.
    Wraps raw NLE observation arrays and provides on-demand queries for
    blstats, screen grids, inventory, messages, and NLE C symbol tables.
    """

    __slots__ = (
        "_raw_dict",
        "mem",
        "blstats",
        "_blstats_list",
        "glyphs",
        "chars",
        "colors",
        "specials",
        "raw_message",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
        "program_state",
        "internal",
        "screen_descriptions",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "misc",
        "step_count",
        "_message_cache",
        "_inventory_cache",
        "_monsters_cache",
        "_monsters_no_self_cache",
        "_objects_cache",
        "_tty_text_cache",
        "_tty_lines_cache",
        "_glyphs_list_cache",
        "_chars_list_cache",
        "_trap_mask_cache",
        "_trap_grid_cache",
        "_stairs_down_positions_cache",
        "_stairs_up_positions_cache",
        "x",
        "y",
        "pos",
        "score",
        "depth",
        "gold",
        "energy",
        "max_energy",
        "ac",
        "xp_level",
        "turn",
        "hunger_state",
        "carrying_capacity",
        "dungeon_num",
        "level_num",
        "condition_mask",
    )

    def __init__(
        self,
        raw_obs: dict[str, np.ndarray] | tuple[np.ndarray, ...],
        step_count: int = 0,
        mem: WorldMemory | None = None,
        obs_keys: tuple[str, ...] | list[str] | None = None,
        obs_idx_map: dict[str, int] | None = None,
    ):
        if isinstance(raw_obs, (tuple, list)):
            self._raw_dict = None
            if obs_idx_map is not None or obs_keys is not None:
                idx = (
                    obs_idx_map
                    if obs_idx_map is not None
                    else {k: i for i, k in enumerate(obs_keys or ())}
                )
                self.glyphs = raw_obs[idx["glyphs"]]
                self.chars = raw_obs[idx["chars"]]
                self.colors = raw_obs[idx["colors"]]
                self.specials = raw_obs[idx["specials"]]
                self.blstats = raw_obs[idx["blstats"]]
                self.raw_message = raw_obs[idx["message"]]
                self.inv_glyphs = raw_obs[idx["inv_glyphs"]]
                self.inv_strs = raw_obs[idx["inv_strs"]]
                self.inv_letters = raw_obs[idx["inv_letters"]]
                self.inv_oclasses = raw_obs[idx["inv_oclasses"]]
                self.program_state = (
                    raw_obs[idx["program_state"]] if "program_state" in idx else None
                )
                self.internal = raw_obs[idx["internal"]] if "internal" in idx else None
                self.screen_descriptions = (
                    raw_obs[idx["screen_descriptions"]]
                    if "screen_descriptions" in idx
                    else None
                )
                self.tty_chars = (
                    raw_obs[idx["tty_chars"]] if "tty_chars" in idx else None
                )
                self.tty_colors = (
                    raw_obs[idx["tty_colors"]] if "tty_colors" in idx else None
                )
                self.tty_cursor = (
                    raw_obs[idx["tty_cursor"]] if "tty_cursor" in idx else None
                )
                self.misc = raw_obs[idx["misc"]] if "misc" in idx else None
            else:
                self.glyphs = raw_obs[0]
                self.chars = raw_obs[1]
                self.colors = raw_obs[2]
                self.specials = raw_obs[3]
                self.blstats = raw_obs[4]
                self.raw_message = raw_obs[5]
                self.inv_glyphs = raw_obs[6]
                self.inv_strs = raw_obs[7]
                self.inv_letters = raw_obs[8]
                self.inv_oclasses = raw_obs[9]
                self.program_state = raw_obs[10] if len(raw_obs) > 10 else None
                self.internal = raw_obs[11] if len(raw_obs) > 11 else None
                self.screen_descriptions = raw_obs[12] if len(raw_obs) > 12 else None
                self.tty_chars = raw_obs[13] if len(raw_obs) > 13 else None
                self.tty_colors = raw_obs[14] if len(raw_obs) > 14 else None
                self.tty_cursor = raw_obs[15] if len(raw_obs) > 15 else None
                self.misc = raw_obs[16] if len(raw_obs) > 16 else None
        else:
            self._raw_dict = raw_obs
            self.blstats = raw_obs["blstats"]
            self.glyphs = raw_obs["glyphs"]
            self.chars = raw_obs["chars"]
            self.colors = raw_obs["colors"]
            self.specials = raw_obs["specials"]
            self.raw_message = raw_obs["message"]
            self.inv_glyphs = raw_obs["inv_glyphs"]
            self.inv_strs = raw_obs["inv_strs"]
            self.inv_letters = raw_obs["inv_letters"]
            self.inv_oclasses = raw_obs["inv_oclasses"]
            self.program_state = raw_obs.get("program_state")
            self.internal = raw_obs.get("internal")
            self.screen_descriptions = raw_obs.get("screen_descriptions")
            self.tty_chars = raw_obs.get("tty_chars")
            self.tty_colors = raw_obs.get("tty_colors")
            self.tty_cursor = raw_obs.get("tty_cursor")
            self.misc = raw_obs.get("misc")

        self.step_count: int = step_count
        bl = self.blstats.tolist()
        self._blstats_list: list[int] = bl
        n_bl = len(bl)
        self.x: int = bl[0] if n_bl > 0 else 0
        self.y: int = bl[1] if n_bl > 1 else 0
        self.pos: tuple[int, int] = (self.y, self.x)
        self.score: int = bl[9] if n_bl > 9 else 0
        self.depth: int = bl[12] if n_bl > 12 else 0
        self.gold: int = bl[13] if n_bl > 13 else 0
        self.energy: int = bl[14] if n_bl > 14 else 0
        self.max_energy: int = bl[15] if n_bl > 15 else 0
        self.ac: int = bl[16] if n_bl > 16 else 10
        self.xp_level: int = bl[18] if n_bl > 18 else 1
        self.turn: int = bl[20] if n_bl > 20 else 0
        self.hunger_state: int = bl[21] if n_bl > 21 else 1
        self.carrying_capacity: int = bl[22] if n_bl > 22 else 0
        self.dungeon_num: int = bl[23] if n_bl > 23 else 0
        self.level_num: int = bl[24] if n_bl > 24 else 1
        self.condition_mask: int = bl[25] if n_bl > 25 else 0
        self._message_cache: str | None = None
        self._inventory_cache: list[InventoryItem] | None = None
        self._monsters_cache: list[VisibleMonster] | None = None
        self._monsters_no_self_cache: list[VisibleMonster] | None = None
        self._objects_cache: list[VisibleObject] | None = None
        self._tty_text_cache: str | None = None
        self._tty_lines_cache: list[str] | None = None
        self._glyphs_list_cache: list[list[int]] | None = None
        self._chars_list_cache: list[list[int]] | None = None
        self._trap_mask_cache: np.ndarray | None = None
        self._trap_grid_cache: list[list[bool]] | None = None
        self._stairs_down_positions_cache: list[tuple[int, int]] | None = None
        self._stairs_up_positions_cache: list[tuple[int, int]] | None = None

        if mem is None:
            self.mem = WorldMemory()
            self.mem.update(self)
        else:
            self.mem = mem

    @property
    def hp(self) -> int:
        bl = self._blstats_list
        return bl[10] if len(bl) > 10 else 0

    @property
    def max_hp(self) -> int:
        bl = self._blstats_list
        return bl[11] if len(bl) > 11 else 0

    @property
    def raw(self) -> dict[str, Any]:
        if self._raw_dict is None:
            self._raw_dict = {
                "glyphs": self.glyphs,
                "chars": self.chars,
                "colors": self.colors,
                "specials": self.specials,
                "blstats": self.blstats,
                "message": self.raw_message,
                "inv_glyphs": self.inv_glyphs,
                "inv_strs": self.inv_strs,
                "inv_letters": self.inv_letters,
                "inv_oclasses": self.inv_oclasses,
                "program_state": self.program_state,
                "internal": self.internal,
                "screen_descriptions": self.screen_descriptions,
                "tty_chars": self.tty_chars,
                "tty_colors": self.tty_colors,
                "tty_cursor": self.tty_cursor,
                "misc": self.misc,
            }
        return self._raw_dict

    # --- 1-to-1 NLE blstats properties (hot fields cached as plain slot attributes in __init__) ---
    @property
    def strength(self) -> int:
        """NLE_BL_STR25 (3..25 strength scale)."""
        return self._blstats_list[2]

    @property
    def strength_pct(self) -> int:
        """NLE_BL_STR125 (3..125 percentile strength scale)."""
        return self._blstats_list[3]

    @property
    def dex(self) -> int:
        return self._blstats_list[4]

    @property
    def con(self) -> int:
        return self._blstats_list[5]

    @property
    def intel(self) -> int:
        return self._blstats_list[6]

    @property
    def wis(self) -> int:
        return self._blstats_list[7]

    @property
    def cha(self) -> int:
        return self._blstats_list[8]

    @property
    def monster_level(self) -> int:
        return self._blstats_list[17]

    @property
    def xp(self) -> int:
        return self._blstats_list[19]

    @property
    def hunger_name(self) -> str:
        names = (
            "Satiated",
            "Not Hungry",
            "Hungry",
            "Weak",
            "Fainting",
            "Fainted",
            "Starved",
        )
        h = self.hunger_state
        return names[h] if 0 <= h < len(names) else f"Hunger({h})"

    @property
    def encumbrance(self) -> int:
        return self.carrying_capacity

    @property
    def encumbrance_name(self) -> str:
        names = (
            "Unencumbered",
            "Burdened",
            "Stressed",
            "Strained",
            "Overtaxed",
            "Overloaded",
        )
        c = self.carrying_capacity
        return names[c] if 0 <= c < len(names) else f"Encumbrance({c})"

    @property
    def align(self) -> int:
        """-1=Chaotic, 0=Neutral, 1=Lawful."""
        return self._blstats_list[26] if len(self._blstats_list) > 26 else 0

    # --- 1-to-1 NLE Condition Bitmask Queries ---
    @property
    def is_blind(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_BLIND)

    @property
    def is_confused(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_CONF)

    @property
    def is_deaf(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_DEAF)

    @property
    def is_flying(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_FLY)

    @property
    def is_food_poisoned(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_FOODPOIS)

    @property
    def is_hallucinating(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_HALLU)

    @property
    def is_levitating(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_LEV)

    @property
    def is_riding(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_RIDE)

    @property
    def is_slimed(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_SLIME)

    @property
    def is_stoning(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_STONE)

    @property
    def is_strangled(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_STRNGL)

    @property
    def is_stunned(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_STUN)

    @property
    def is_terminally_ill(self) -> bool:
        return bool(self.condition_mask & nh.BL_MASK_TERMILL)

    # --- Lazy Decoded Message & Inventory ---
    @property
    def message(self) -> str:
        if self._message_cache is None:
            raw = bytes(self.raw_message)
            nul = raw.find(b"\x00")
            self._message_cache = (
                (raw[:nul] if nul != -1 else raw)
                .decode("latin1", errors="ignore")
                .strip()
            )
        return self._message_cache

    @property
    def inventory(self) -> list[InventoryItem]:
        if self._inventory_cache is None:
            items: list[InventoryItem] = []
            inv_letters = self.inv_letters
            inv_strs = self.inv_strs
            inv_glyphs = self.inv_glyphs
            inv_oclasses = self.inv_oclasses
            for i in range(len(inv_letters)):
                let_b = int(inv_letters[i])
                if let_b == 0:
                    continue
                letter = chr(let_b)
                raw = bytes(inv_strs[i])
                nul = raw.find(b"\x00")
                desc = (
                    (raw[:nul] if nul != -1 else raw)
                    .decode("latin1", errors="ignore")
                    .strip()
                )
                g = int(inv_glyphs[i])
                oc = int(inv_oclasses[i])
                ch = _OCLASS_TO_CHAR.get(oc, "?")
                sp = desc.find(" ")
                first_word = desc[:sp] if sp != -1 else desc
                qty = int(first_word) if first_word.isdigit() else 1
                o_info = _OBJ_INFO_BY_GLYPH[g] if 0 <= g < _MAX_GLYPH else None
                c_mon = _CORPSE_MON_BY_GLYPH[g] if 0 <= g < _MAX_GLYPH else None
                items.append(
                    InventoryItem(
                        letter=letter,
                        name=desc,
                        glyph=g,
                        oclass=oc,
                        char=ch,
                        quantity=qty,
                        info=o_info,
                        corpse_monster=c_mon,
                    )
                )
            self._inventory_cache = _CallableList(items)
        return self._inventory_cache

    # --- Screen & NLE C Table Queries ---
    def _glyph_int(self, y: int, x: int) -> int:
        gl = self._glyphs_list_cache
        if gl is not None:
            return gl[y][x]
        return int(self.glyphs[y, x])

    def _char_int(self, y: int, x: int) -> int:
        cl = self._chars_list_cache
        if cl is not None:
            return cl[y][x]
        return int(self.chars[y, x])

    def char_at(self, y: int, x: int) -> str:
        if 0 <= y < 21 and 0 <= x < 79:
            c = self._char_int(y, x)
            if (y, x) == (self.y, self.x) and c == ord("@"):
                if getattr(self, "is_on_stairs_down", False):
                    return ">"
                if (
                    self.mem is not None
                    and 0 <= y < self.mem.remembered_chars.shape[0]
                    and 0 <= x < self.mem.remembered_chars.shape[1]
                ):
                    rem_c = int(self.mem.remembered_chars[y, x])
                    if rem_c != 0 and rem_c != ord(" "):
                        return _CHAR_LUT[rem_c] if rem_c < 256 else "?"
            return _CHAR_LUT[c] if c < 256 else "?"
        return " "

    def glyph_at(self, y: int, x: int) -> int:
        if 0 <= y < 21 and 0 <= x < 79:
            return self._glyph_int(y, x)
        return int(nh.NO_GLYPH)

    @staticmethod
    def is_monster_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_MONSTER_LUT[glyph])

    @staticmethod
    def is_pet_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_PET_LUT[glyph])

    @staticmethod
    def is_body_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_BODY_LUT[glyph])

    @staticmethod
    def is_object_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_OBJECT_LUT[glyph])

    @staticmethod
    def is_cmap_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_CMAP_LUT[glyph])

    @staticmethod
    def is_trap_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_TRAP_LUT[glyph])

    @staticmethod
    def is_swallow_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_SWALLOW_LUT[glyph])

    @staticmethod
    def is_warning_glyph(glyph: int) -> bool:
        return 0 <= glyph < _MAX_GLYPH and bool(_IS_WARNING_LUT[glyph])

    @staticmethod
    def monster_info(glyph: int) -> MonsterInfo | None:
        if 0 <= glyph < _MAX_GLYPH:
            return _MON_INFO_BY_GLYPH[glyph] or _CORPSE_MON_BY_GLYPH[glyph]
        return None

    @staticmethod
    def monster_info_by_name(name: str) -> MonsterInfo | None:
        """O(1) lookup of a monster's C `permonst` struct by monster or corpse string."""
        if not name:
            return None
        s = name.lower().strip()
        for prefix in (
            "a ",
            "an ",
            "the ",
            "blessed ",
            "uncursed ",
            "cursed ",
            "partly eaten ",
            "corpse of a ",
            "corpse of an ",
            "corpse of ",
        ):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        if s.endswith(" corpse"):
            s = s[:-7].strip()
        elif s.endswith(" corpses") and len(s) > 8:
            s = s[:-8].strip()
        return _MON_INFO_BY_NAME.get(s)

    @staticmethod
    def object_info(glyph: int) -> ObjectInfo | None:
        if 0 <= glyph < _MAX_GLYPH:
            return _OBJ_INFO_BY_GLYPH[glyph]
        return None

    @staticmethod
    def object_info_by_name(name: str) -> ObjectInfo | None:
        """O(1) lookup of an object's C `objclass` struct by name, class-prefixed phrase, or description."""
        return lookup_object_info_by_name(name)

    @staticmethod
    def cmap_name(glyph: int) -> str | None:
        if 0 <= glyph < _MAX_GLYPH:
            return _CMAP_NAME_BY_GLYPH[glyph]
        return None

    @staticmethod
    def trap_name(glyph: int) -> str | None:
        if 0 <= glyph < _MAX_GLYPH:
            return _TRAP_NAME_BY_GLYPH[glyph]
        return None

    def terrain_at(self, y: int, x: int) -> str:
        """Returns the semantic terrain / feature name at (y, x) (e.g. 'closed door', 'open door', 'staircase down', 'altar')."""
        if 0 <= y < 21 and 0 <= x < 79:
            if (
                y == self.y
                and x == self.x
                and getattr(self, "is_on_stairs_down", False)
            ):
                return "staircase down"
            g = int(self.glyphs[y, x])
            cmap = _CMAP_NAME_BY_GLYPH[g] if 0 <= g < _MAX_GLYPH else None
            if cmap is not None:
                return cmap
            if (
                self.mem is not None
                and hasattr(self.mem, "remembered_glyphs")
                and 0 <= y < self.mem.remembered_glyphs.shape[0]
                and 0 <= x < self.mem.remembered_glyphs.shape[1]
            ):
                rg = int(self.mem.remembered_glyphs[y, x])
                if 0 < rg < _MAX_GLYPH and _CMAP_NAME_BY_GLYPH[rg] is not None:
                    return str(_CMAP_NAME_BY_GLYPH[rg])
            if (
                self.mem is not None
                and 0 <= y < self.mem.remembered_chars.shape[0]
                and 0 <= x < self.mem.remembered_chars.shape[1]
            ):
                rem_c = int(self.mem.remembered_chars[y, x])
                if rem_c == ord(">"):
                    return "staircase down"
                elif rem_c == ord("<"):
                    return "staircase up"
                elif rem_c == ord("_"):
                    return "altar"
                elif rem_c == ord("{"):
                    return "fountain"
                elif rem_c == ord("+"):
                    return "closed door"
                elif rem_c == ord("#"):
                    return "corridor"
                elif rem_c == ord("."):
                    return "floor of a room"
            c = int(self.chars[y, x])
            if c == ord("+"):
                return "closed door"
            if c in (ord("-"), ord("|")) and self.is_cmap_glyph(g):
                return "open door"
            if c == ord(">"):
                return "staircase down"
            if c == ord("<"):
                return "staircase up"
            return _CHAR_LUT[c] if c < 256 else " "
        return " "

    def trap_at(self, y: int, x: int) -> str | None:
        """Returns the specific trap name at (y, x) (e.g. 'arrow trap', 'bear trap') or None."""
        if 0 <= y < 21 and 0 <= x < 79:
            g = self._glyph_int(y, x)
            name = _TRAP_NAME_BY_GLYPH[g] if 0 <= g < _MAX_GLYPH else None
            if name is not None:
                return name
            if (
                self.mem is not None
                and 0 <= y < self.mem.remembered_trap_glyphs.shape[0]
                and 0 <= x < self.mem.remembered_trap_glyphs.shape[1]
            ):
                rg = int(self.mem.remembered_trap_glyphs[y, x])
                if 0 <= rg < _MAX_GLYPH:
                    return _TRAP_NAME_BY_GLYPH[rg]
        return None

    @property
    def trap_mask(self) -> np.ndarray:
        """Vectorized 21x79 boolean mask of visible or remembered traps on the current floor."""
        tm = self._trap_mask_cache
        if tm is not None:
            return tm
        if (
            _HAS_RUST_FLOOR_MEM
            and self.mem is not None
            and self.mem.remembered_trap_glyphs.shape == self.glyphs.shape
            and self.glyphs.dtype == np.int16
            and self.glyphs.flags.c_contiguous
            and self.mem.remembered_trap_glyphs.dtype == np.int32
            and self.mem.remembered_trap_glyphs.flags.c_contiguous
        ):
            tm = lox_core.fast_trap_mask(
                self.mem.remembered_trap_glyphs, self.glyphs, _IS_TRAP_LUT
            )
            self._trap_mask_cache = tm
            return tm
        g_arr = self.glyphs
        clipped = np.clip(g_arr, 0, _MAX_GLYPH - 1)
        mask = _IS_TRAP_LUT[clipped] & (g_arr >= 0) & (g_arr < _MAX_GLYPH)
        if (
            self.mem is not None
            and self.mem.remembered_trap_glyphs.shape == g_arr.shape
        ):
            mask = mask | (self.mem.remembered_trap_glyphs > 0)
        self._trap_mask_cache = mask
        return mask

    def adjacent_closed_doors(self) -> list[tuple[int, int]]:
        """Returns 8-way adjacent `(y, x)` coordinates currently displaying a closed door (`'+'`)."""
        doors: list[tuple[int, int]] = []
        y0, x0 = int(self.y), int(self.x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y0 + dy, x0 + dx
                if 0 <= ny < 21 and 0 <= nx < 79 and self.is_closed_door(ny, nx):
                    doors.append((ny, nx))
        return doors

    def unexplored_frontier_mask(
        self,
        passable_mask: np.ndarray | None = None,
        include_doors: bool = True,
    ) -> np.ndarray:
        """
        Returns boolean (H, W) mask of passable known tiles bordering unvisited walkable
        tiles (or closed doors `'+'` when `include_doors=True`).
        """
        pm = passable_mask if passable_mask is not None else self.passable_mask
        h, w = pm.shape
        if self.mem is None or self.mem.current_floor.visited.shape != pm.shape:
            return pm.copy()
        fl = self.mem.current_floor
        if (
            _HAS_RUST_FLOOR_MEM
            and fl.visited.flags.c_contiguous
            and pm.dtype == np.bool_
            and pm.flags.c_contiguous
            and fl.remembered_chars.dtype == np.uint8
            and fl.remembered_chars.flags.c_contiguous
        ):
            return lox_core.fast_unexplored_frontier_mask(
                fl.visited, pm, fl.remembered_chars, bool(include_doors)
            )
        unvisited_walkable = pm & (~fl.visited)
        boundary = unvisited_walkable.copy()
        neighbor_targets = unvisited_walkable.copy()
        if include_doors:
            neighbor_targets |= fl.remembered_chars == ord("+")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                src_y0 = max(0, -dy)
                src_y1 = min(h, h - dy)
                dst_y0 = max(0, dy)
                dst_y1 = min(h, h + dy)
                src_x0 = max(0, -dx)
                src_x1 = min(w, w - dx)
                dst_x0 = max(0, dx)
                dst_x1 = min(w, w + dx)
                boundary[dst_y0:dst_y1, dst_x0:dst_x1] |= (
                    pm[dst_y0:dst_y1, dst_x0:dst_x1]
                    & neighbor_targets[src_y0:src_y1, src_x0:src_x1]
                )
        return boundary

    def description_at(self, y: int, x: int) -> str:
        """Returns the full English tile description from NLE's C engine for (y, x)."""
        if self.screen_descriptions is not None and 0 <= y < 21 and 0 <= x < 79:
            raw = bytes(self.screen_descriptions[y, x])
            nul = raw.find(b"\x00")
            desc = (
                (raw[:nul] if nul != -1 else raw)
                .decode("latin1", errors="ignore")
                .strip()
            )
            if desc:
                return desc
        return self.terrain_at(y, x)

    def is_closed_door(self, y: int, x: int) -> bool:
        return self.char_at(y, x) == "+" or self.terrain_at(y, x) == "closed door"

    def is_open_door(self, y: int, x: int) -> bool:
        return self.terrain_at(y, x) in ("open door", "doorway")

    def is_stairs_down(self, y: int, x: int) -> bool:
        if self.char_at(y, x) == ">" or "staircase down" in self.terrain_at(y, x):
            return True
        if (
            self.mem is not None
            and 0 <= y < self.mem.remembered_chars.shape[0]
            and 0 <= x < self.mem.remembered_chars.shape[1]
            and self.mem.remembered_chars[y, x] == ord(">")
        ):
            return True
        return False

    def is_stairs_up(self, y: int, x: int) -> bool:
        if self.char_at(y, x) == "<" or "staircase up" in self.terrain_at(y, x):
            return True
        if (
            self.mem is not None
            and 0 <= y < self.mem.remembered_chars.shape[0]
            and 0 <= x < self.mem.remembered_chars.shape[1]
            and self.mem.remembered_chars[y, x] == ord("<")
        ):
            return True
        return False

    def is_altar(self, y: int, x: int) -> bool:
        if self.char_at(y, x) == "_" or self.terrain_at(y, x) == "altar":
            return True
        if (
            self.mem is not None
            and 0 <= y < self.mem.remembered_chars.shape[0]
            and 0 <= x < self.mem.remembered_chars.shape[1]
            and self.mem.remembered_chars[y, x] == ord("_")
        ):
            return True
        return False

    def is_fountain(self, y: int, x: int) -> bool:
        if self.char_at(y, x) == "{" or self.terrain_at(y, x) == "fountain":
            return True
        if (
            self.mem is not None
            and 0 <= y < self.mem.remembered_chars.shape[0]
            and 0 <= x < self.mem.remembered_chars.shape[1]
            and self.mem.remembered_chars[y, x] == ord("{")
        ):
            return True
        return False

    def is_sink(self, y: int, x: int) -> bool:
        return self.terrain_at(y, x) == "sink"

    def is_trap(self, y: int, x: int) -> bool:
        if 0 <= y < 21 and 0 <= x < 79:
            grid = self._trap_grid_cache
            if grid is None:
                grid = self.trap_mask.tolist()
                self._trap_grid_cache = grid
            return grid[y][x]
        return False

    def _feature_positions(
        self, char_ord: int, include_self_on_down: bool = False
    ) -> list[tuple[int, int]]:
        mask = self.chars == char_ord
        if self.mem is not None:
            rem_c = self.mem.current_floor.remembered_chars
            if rem_c.shape == self.chars.shape:
                mask = mask | (rem_c == char_ord)
        ys, xs = np.where(mask)
        coords = list(zip(ys.tolist(), xs.tolist()))
        if (
            include_self_on_down
            and getattr(self, "is_on_stairs_down", False)
            and (self.y, self.x) not in coords
        ):
            coords.append((self.y, self.x))
        return coords

    @property
    def stairs_down_positions(self) -> list[tuple[int, int]]:
        """All visible or remembered staircase-down coordinates on the current floor."""
        cached = self._stairs_down_positions_cache
        if cached is None:
            cached = self._feature_positions(ord(">"), include_self_on_down=True)
            self._stairs_down_positions_cache = cached
        return cached

    @property
    def stairs_up_positions(self) -> list[tuple[int, int]]:
        """All visible or remembered staircase-up coordinates on the current floor."""
        cached = self._stairs_up_positions_cache
        if cached is None:
            cached = self._feature_positions(ord("<"))
            self._stairs_up_positions_cache = cached
        return cached

    @property
    def altar_positions(self) -> list[tuple[int, int]]:
        """All visible or remembered altar coordinates on the current floor."""
        return self._feature_positions(ord("_"))

    @property
    def fountain_positions(self) -> list[tuple[int, int]]:
        """All visible or remembered fountain coordinates on the current floor."""
        return self._feature_positions(ord("{"))

    @property
    def portal_positions(self) -> list[tuple[int, int]]:
        """All visible or remembered magic portal coordinates on the current floor."""
        mask = _IS_PORTAL_GLYPH_LUT[self.glyphs]
        if (
            self.mem is not None
            and hasattr(self.mem, "remembered_glyphs")
            and self.mem.remembered_glyphs.shape == self.glyphs.shape
        ):
            rg = self.mem.remembered_glyphs
            valid_rg = np.where((rg > 0) & (rg < _MAX_GLYPH), rg, 0)
            mask = mask | _IS_PORTAL_GLYPH_LUT[valid_rg]
        ys, xs = np.where(mask)
        return list(zip(ys.tolist(), xs.tolist()))

    @property
    def stairs_down_pos(self) -> tuple[int, int] | None:
        """Finds the coordinates of the staircase down on the current screen, if visible."""
        stairs = np.argwhere(self.chars == ord(">"))
        if len(stairs) > 0:
            return (int(stairs[0, 0]), int(stairs[0, 1]))
        if getattr(self, "is_on_stairs_down", False):
            return (self.y, self.x)
        if self.mem is not None:
            rem = np.argwhere(self.mem.remembered_chars == ord(">"))
            if len(rem) > 0:
                return (int(rem[0, 0]), int(rem[0, 1]))
        return None

    @property
    def stairs_up_pos(self) -> tuple[int, int] | None:
        """Finds the coordinates of the staircase up on the current screen, if visible."""
        stairs = np.argwhere(self.chars == ord("<"))
        if len(stairs) > 0:
            return (int(stairs[0, 0]), int(stairs[0, 1]))
        if self.mem is not None:
            rem = np.argwhere(self.mem.remembered_chars == ord("<"))
            if len(rem) > 0:
                return (int(rem[0, 0]), int(rem[0, 1]))
        return None

    # --- Full 24x80 Terminal & Text Queries ---
    @property
    def tty_lines(self) -> list[str]:
        if self._tty_lines_cache is None:
            lines = []
            if self.tty_chars is not None:
                for r in range(len(self.tty_chars)):
                    line = (
                        bytes(self.tty_chars[r])
                        .translate(_ASCII_TRANS_TABLE)
                        .decode("latin1")
                        .rstrip()
                    )
                    lines.append(line)
            self._tty_lines_cache = lines
        return self._tty_lines_cache

    @property
    def tty_text(self) -> str:
        if self._tty_text_cache is None:
            self._tty_text_cache = "\n".join(self.tty_lines)
        return self._tty_text_cache

    # --- 1-to-1 NLE Internal & Program State Properties ---
    @property
    def deepest_level_reached(self) -> int:
        if self.internal is not None and len(self.internal) > 0:
            return int(self.internal[0])
        return self.depth

    @property
    def is_on_stairs_down(self) -> bool:
        if self.internal is not None and len(self.internal) > 4:
            return bool(self.internal[4])
        sd = self.stairs_down_pos
        return sd is not None and sd == self.pos

    @property
    def internal_hunger(self) -> int:
        """Raw internal integer nutrition counter from NetHack's C engine."""
        if self.internal is not None and len(self.internal) > 7:
            return int(self.internal[7])
        return 900

    @property
    def waiting_for_space(self) -> bool:
        if self.internal is not None and len(self.internal) > 3:
            return bool(self.internal[3])
        return False

    @property
    def in_menu(self) -> bool:
        """True when NetHack is displaying an interactive selection/pager menu (distinct from `--More--`)."""
        if not self.waiting_for_space or self.tty_chars is None:
            return False
        lines = self.tty_lines
        if lines and "--more--" in lines[0].lower():
            return False
        joined = "\n".join(lines[:24]).lower()
        return "(end)" in joined or bool(
            re.search(r"\(\d+\s+of\s+\d+\)", joined)
            or (
                lines
                and any(
                    k in lines[0].lower()
                    for k in ("pick up what", "what would you like", "for what")
                )
            )
        )

    @property
    def is_in_moveloop(self) -> bool:
        if self.program_state is not None and len(self.program_state) > 3:
            return bool(self.program_state[3])
        return True

    @property
    def in_moveloop(self) -> bool:
        return self.is_in_moveloop

    @property
    def is_in_yn(self) -> bool:
        if self.internal is not None and len(self.internal) > 1:
            return bool(self.internal[1])
        if self.program_state is not None and len(self.program_state) > 1:
            return bool(self.program_state[1])
        return False

    def _has_more_on_topline(self) -> bool:
        """True when row 0 of `tty_chars` currently shows a `--More--` pager."""
        tty_c = self.tty_chars
        if tty_c is not None and len(tty_c) > 0:
            row0 = bytes(tty_c[0]).decode("latin1", errors="ignore").lower()
            if "--more--" in row0:
                return True
        return "--more--" in self.message.lower()

    def _prompt_text_lower(self) -> str:
        msg = self.message.strip().lower()
        if msg:
            return msg
        tty_c = self.tty_chars
        if tty_c is not None and len(tty_c) > 0:
            return bytes(tty_c[0]).decode("latin1", errors="ignore").strip().lower()
        return ""

    @property
    def in_getobj_prompt(self) -> bool:
        """True when NetHack asks for an inventory letter (`What do you want to ... [?*]` etc.)."""
        if (
            self.internal is not None
            and len(self.internal) > 3
            and bool(self.internal[3])
            and self._has_more_on_topline()
        ):
            return False
        if not self.is_in_yn:
            return False
        msg = self._prompt_text_lower()
        return (
            "[?*" in msg
            or "?*]" in msg
            or "what do you want to " in msg
            or "for what" in msg
        )

    @property
    def in_getdir_prompt(self) -> bool:
        """True when NetHack asks for a direction (`In what direction?` etc.)."""
        if (
            self.internal is not None
            and len(self.internal) > 3
            and bool(self.internal[3])
            and self._has_more_on_topline()
        ):
            return False
        if not self.is_in_yn:
            return False
        msg = self._prompt_text_lower()
        return "in what direction" in msg or "direction?" in msg

    @property
    def in_yn_prompt(self) -> bool:
        if (
            self.internal is not None
            and len(self.internal) > 3
            and bool(self.internal[3])
            and self._has_more_on_topline()
        ):
            return False
        if self.in_getobj_prompt or self.in_getdir_prompt:
            return False
        if self.internal is not None and len(self.internal) > 1:
            return bool(self.internal[1])
        if self.is_in_yn:
            return True
        msg = self.message.strip().lower()
        if msg.endswith((") n", ") y", ") q", "] n", "] y", "] q")):
            return False
        return (
            "[yn]" in msg or "[ynq]" in msg or "(yn)" in msg or "really attack" in msg
        )

    @property
    def is_in_getlin(self) -> bool:
        if self.internal is not None and len(self.internal) > 2:
            return bool(self.internal[2])
        if self.program_state is not None and len(self.program_state) > 2:
            return bool(self.program_state[2])
        return False

    @property
    def in_text_prompt(self) -> bool:
        if self.internal is not None and len(self.internal) > 2:
            return bool(self.internal[2])
        if self.is_in_getlin:
            return True
        msg = self.message.lower()
        return any(
            p in msg
            for p in (
                "call a ",
                "what do you want to call",
                "name a ",
                "what do you want to name",
            )
        )

    @property
    def in_naming_prompt(self) -> bool:
        return self.in_text_prompt

    @property
    def turn_advanced(self) -> bool:
        """Returns True if this step advanced the in-game clock (consumed >= 1 turn)."""
        if self.mem is None:
            return True
        return not getattr(self.mem, "last_action_zero_turn", False)

    @property
    def consecutive_stalls(self) -> int:
        """Returns the number of consecutive steps that failed to advance the in-game clock."""
        if self.mem is None:
            return 0
        return int(getattr(self.mem, "consecutive_stalls", 0))

    @property
    def time_delta(self) -> int:
        """Returns in-game turns elapsed on this step (1 if turn advanced, 0 if stalled)."""
        return 1 if self.turn_advanced else 0

    @property
    def is_game_over(self) -> bool:
        if self.program_state is not None and len(self.program_state) > 0:
            return bool(self.program_state[0])
        return False

    def visible_monsters(self, exclude_self: bool = True) -> list[VisibleMonster]:
        """Returns all monsters currently visible on the 21x79 screen."""
        if self._monsters_cache is None:
            mons: list[VisibleMonster] = []
            glyphs = self.glyphs
            chars = self.chars
            colors = self.colors
            specials = self.specials
            mon_mask = _IS_MONSTER_LUT[glyphs]
            ys, xs = np.nonzero(mon_mask)
            for i in range(len(ys)):
                yi = int(ys[i])
                xi = int(xs[i])
                g = int(glyphs[yi, xi])
                c = int(chars[yi, xi])
                ch = _CHAR_LUT[c] if c < 256 else "?"
                col = int(colors[yi, xi])
                spec = int(specials[yi, xi])
                is_pet = bool(_IS_PET_LUT[g]) or bool(spec & _MG_PET)
                is_statue = bool(spec & _MG_STATUE)
                is_ridden = bool(spec & _MG_RIDDEN)
                is_invis = bool(spec & _MG_INVIS)
                m_info = _MON_INFO_BY_GLYPH[g]
                name = m_info.name if m_info else ch
                mons.append(
                    VisibleMonster(
                        y=yi,
                        x=xi,
                        glyph=g,
                        char=ch,
                        color=col,
                        is_pet=is_pet,
                        is_statue=is_statue,
                        is_ridden=is_ridden,
                        is_invisible=is_invis,
                        name=name,
                        info=m_info,
                    )
                )
            self._monsters_cache = mons
        if exclude_self:
            if self._monsters_no_self_cache is None:
                hy, hx = self.y, self.x
                self._monsters_no_self_cache = [
                    m for m in self._monsters_cache if not (m.y == hy and m.x == hx)
                ]
            return self._monsters_no_self_cache
        return list(self._monsters_cache)

    def visible_objects(self) -> list[VisibleObject]:
        """Returns all objects and corpses currently visible on the 21x79 screen."""
        if self._objects_cache is None:
            objs: list[VisibleObject] = []
            glyphs = self.glyphs
            chars = self.chars
            colors = self.colors
            obj_mask = _IS_OBJECT_LUT[glyphs]
            ys, xs = np.nonzero(obj_mask)
            for i in range(len(ys)):
                yi = int(ys[i])
                xi = int(xs[i])
                g = int(glyphs[yi, xi])
                c = int(chars[yi, xi])
                ch = _CHAR_LUT[c] if c < 256 else "?"
                col = int(colors[yi, xi])
                is_corpse = bool(_IS_BODY_LUT[g])
                corpse_mon = _CORPSE_MON_BY_GLYPH[g] if is_corpse else None
                o_info = _OBJ_INFO_BY_GLYPH[g]
                if is_corpse and corpse_mon:
                    name = f"corpse of {corpse_mon.name}"
                elif o_info and o_info.name:
                    name = o_info.name
                else:
                    name = ch
                objs.append(
                    VisibleObject(
                        y=yi,
                        x=xi,
                        glyph=g,
                        char=ch,
                        color=col,
                        is_corpse=is_corpse,
                        corpse_monster=corpse_mon,
                        name=name,
                        info=o_info,
                    )
                )

            # Include under-hero objects/corpses if hero stepped onto an object tile
            hy, hx = int(self.y), int(self.x)
            if not any(o.y == hy and o.x == hx for o in objs):
                has_mem_obj = (
                    self.mem is not None
                    and 0 <= hy < self.mem.current_floor.remembered_obj_glyphs.shape[0]
                    and 0 <= hx < self.mem.current_floor.remembered_obj_glyphs.shape[1]
                    and int(self.mem.current_floor.remembered_obj_glyphs[hy, hx]) > 0
                )
                msg_low = str(self.message).lower() if hasattr(self, "message") else ""
                if has_mem_obj and self.mem is not None:
                    rem_g = int(self.mem.current_floor.remembered_obj_glyphs[hy, hx])
                    if 0 <= rem_g < _MAX_GLYPH and _IS_OBJECT_LUT[rem_g]:
                        is_corpse = bool(_IS_BODY_LUT[rem_g])
                        corpse_mon = _CORPSE_MON_BY_GLYPH[rem_g] if is_corpse else None
                        o_info = _OBJ_INFO_BY_GLYPH[rem_g]
                        if is_corpse and corpse_mon:
                            name = f"corpse of {corpse_mon.name}"
                        elif o_info and o_info.name:
                            name = o_info.name
                        else:
                            name = "corpse" if is_corpse else "?"
                        ch = "%" if is_corpse else (o_info.char if o_info else "?")
                        col = o_info.color if o_info else 7
                        objs.append(
                            VisibleObject(
                                y=hy,
                                x=hx,
                                glyph=rem_g,
                                char=ch,
                                color=col,
                                is_corpse=is_corpse,
                                corpse_monster=corpse_mon,
                                name=name,
                                info=o_info,
                            )
                        )
                elif any(
                    tok in msg_low
                    for tok in (
                        "you see here",
                        "objects here",
                        "things that are here",
                    )
                ):
                    is_corpse = "corpse" in msg_low
                    parsed_name = ""
                    for prefix in (
                        "you see here ",
                        "objects here: ",
                        "things that are here: ",
                    ):
                        if prefix in msg_low:
                            idx = msg_low.find(prefix) + len(prefix)
                            parsed_name = str(self.message)[idx:].split(".")[0].strip()
                            break
                    if not parsed_name:
                        parsed_name = "corpse" if is_corpse else "item"
                    corpse_mon = (
                        self.monster_info_by_name(parsed_name) if is_corpse else None
                    )
                    o_info = (
                        None if is_corpse else self.object_info_by_name(parsed_name)
                    )
                    objs.append(
                        VisibleObject(
                            y=hy,
                            x=hx,
                            glyph=0,
                            char="%" if is_corpse else (o_info.char if o_info else "?"),
                            color=o_info.color if o_info else 7,
                            is_corpse=is_corpse,
                            corpse_monster=corpse_mon,
                            name=parsed_name,
                            info=o_info,
                        )
                    )

            self._objects_cache = objs
        return self._objects_cache

    def objects_here(self) -> list[VisibleObject]:
        """Returns all visible objects on the hero's current tile."""
        return [o for o in self.visible_objects() if o.y == self.y and o.x == self.x]

    def adjacent_monsters(self, exclude_pets: bool = True) -> list[VisibleMonster]:
        """Returns all visible monsters in the 8 Chebyshev adjacent squares around the hero."""
        return [
            m
            for m in self.visible_monsters()
            if (not exclude_pets or not m.is_pet)
            and max(abs(m.y - self.y), abs(m.x - self.x)) == 1
        ]

    def adjacent_hostiles(self) -> list[VisibleMonster]:
        """Deprecated alias for adjacent_monsters(exclude_pets=True)."""
        return self.adjacent_monsters(exclude_pets=True)

    @property
    def kv(self) -> dict[str, Any]:
        """Arbitrary episode-persistent key-value storage synced across floor transitions and shards."""
        return self.mem.kv if self.mem is not None else {}

    @property
    def recent_messages(self) -> list[tuple[int, int, str]]:
        """Recent game log messages as (turn, dlvl, message) tuples."""
        return list(self.mem.recent_messages) if self.mem is not None else []

    @property
    def level_exits(self) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
        """Remembered stairs, ladders, and portal coordinates keyed by `(dungeon_num, level_num)`."""
        return self.mem.level_exits if self.mem is not None else {}

    @property
    def worn_oprop_mask(self) -> int:
        """Bitwise OR of C `objclass.oc_oprop` across all worn or wielded inventory items."""
        mask = 0
        for it in self.inventory:
            if (it.is_worn or it.is_wielded) and it.info is not None:
                mask |= int(getattr(it.info, "oprop", 0) or 0)
        return mask

    def _effective_cmap_glyph(self, y: int, x: int) -> int:
        g = int(self.glyphs[y, x])
        if 0 <= g < _MAX_GLYPH and _IS_CMAP_LUT[g]:
            return g
        if (
            self.mem is not None
            and hasattr(self.mem, "remembered_glyphs")
            and 0 <= y < self.mem.remembered_glyphs.shape[0]
            and 0 <= x < self.mem.remembered_glyphs.shape[1]
        ):
            rg = int(self.mem.remembered_glyphs[y, x])
            if 0 < rg < _MAX_GLYPH:
                return rg
        return g

    def is_water(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is water, pool, or moat according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        eg = self._effective_cmap_glyph(y, x)
        return 0 <= eg < _MAX_GLYPH and bool(_IS_WATER_GLYPH_LUT[eg])

    def is_lava(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is molten lava according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        eg = self._effective_cmap_glyph(y, x)
        return 0 <= eg < _MAX_GLYPH and bool(_IS_LAVA_GLYPH_LUT[eg])

    def is_ice(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is ice according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        eg = self._effective_cmap_glyph(y, x)
        return 0 <= eg < _MAX_GLYPH and bool(_IS_ICE_GLYPH_LUT[eg])

    def is_iron_bars(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is iron bars or tree according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        eg = self._effective_cmap_glyph(y, x)
        return 0 <= eg < _MAX_GLYPH and bool(_IS_IRON_BARS_OR_TREE_LUT[eg])

    def is_boulder(self, y: int, x: int) -> bool:
        """Returns True if (y, x) contains a boulder according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        g = int(self.glyphs[y, x])
        return 0 <= g < _MAX_GLYPH and bool(_IS_BOULDER_GLYPH_LUT[g])

    @property
    def boulder_positions(self) -> list[tuple[int, int]]:
        """Returns (y, x) coordinates of all visible boulders on the current floor."""
        ys, xs = np.where(_IS_BOULDER_GLYPH_LUT[self.glyphs])
        return list(zip(ys.tolist(), xs.tolist()))

    @property
    def passable_mask(self) -> np.ndarray:
        """Returns boolean mask of known walkable tiles (including traps/ice/lowered drawbridges)."""
        if self.mem is not None:
            return self.mem.passable_mask
        c = self.chars
        return (
            (c == ord("."))
            | (c == ord("#"))
            | (c == ord(">"))
            | (c == ord("<"))
            | (c == ord("_"))
            | (c == ord("{"))
            | (c == ord("\\"))
            | (c == ord("^"))
        )

    @property
    def safe_passable_mask(self) -> np.ndarray:
        """Returns boolean mask of known walkable tiles excluding remembered traps and terrain hazards."""
        if self.mem is not None:
            return self.mem.safe_passable_mask
        return self.passable_mask & (~_IS_HAZARD_CMAP_LUT[self.glyphs])

    def ascii_map(self) -> str:
        """Renders the current 21x79 `chars` grid as a 21-line ASCII string."""
        lines = []
        for r in range(21):
            line = (
                bytes(self.chars[r])
                .translate(_ASCII_TRANS_TABLE)
                .decode("latin1")
                .rstrip()
            )
            lines.append(line or " ")
        return "\n".join(lines)

    def bfs_distances(
        self,
        passable_mask: np.ndarray,
        start: tuple[int, int] | None = None,
        allow_diagonal: bool = True,
        no_diag: np.ndarray | None = None,
    ) -> np.ndarray:
        """Runs Rust-accelerated BFS distance calculation from `start` (defaults to `(self.y, self.x)`)."""
        sy, sx = start if start is not None else (self.y, self.x)
        p = (
            passable_mask
            if isinstance(passable_mask, np.ndarray) and passable_mask.dtype == np.bool_
            else np.asarray(passable_mask, dtype=np.bool_)
        )
        nd = (
            None
            if no_diag is None
            else (
                no_diag
                if isinstance(no_diag, np.ndarray) and no_diag.dtype == np.bool_
                else np.asarray(no_diag, dtype=np.bool_)
            )
        )

        return bfs_distance_grid(
            p,
            int(sy),
            int(sx),
            bool(allow_diagonal),
            no_diag=nd,
        )

    def path_step(
        self,
        passable_mask: np.ndarray,
        target: tuple[int, int],
        start: tuple[int, int] | None = None,
        allow_diagonal: bool = True,
        no_diag: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """Returns the next `(dy, dx)` step from `start` toward `target` across `passable_mask`."""
        sy, sx = start if start is not None else (self.y, self.x)
        p = (
            passable_mask
            if isinstance(passable_mask, np.ndarray) and passable_mask.dtype == np.bool_
            else np.asarray(passable_mask, dtype=np.bool_)
        )
        nd = (
            None
            if no_diag is None
            else (
                no_diag
                if isinstance(no_diag, np.ndarray) and no_diag.dtype == np.bool_
                else np.asarray(no_diag, dtype=np.bool_)
            )
        )

        return astar_next_step(
            p,
            int(sy),
            int(sx),
            int(target[0]),
            int(target[1]),
            bool(allow_diagonal),
            no_diag=nd,
        )


__all__ = [
    "FloorMemory",
    "GameState",
    "InventoryItem",
    "MonsterInfo",
    "ObjectInfo",
    "VisibleMonster",
    "VisibleObject",
    "WorldMemory",
    "astar_next_step",
    "bfs_distance_grid",
    "lookup_object_info_by_name",
    "lox_core",
    "_ASCII_TRANS_TABLE",
    "_CHAR_LUT",
    "_CMAP_NAME_BY_GLYPH",
    "_CORPSE_MON_BY_GLYPH",
    "_CallableList",
    "_FLOOR_GLYPH_FLAGS_LUT",
    "_HAS_RUST_FLOOR_MEM",
    "_IS_BODY_LUT",
    "_IS_BOULDER_GLYPH_LUT",
    "_IS_CLOSED_DOOR_GLYPH_LUT",
    "_IS_CMAP_LUT",
    "_IS_DOORWAY_GLYPH_LUT",
    "_IS_DRAWBRIDGE_GLYPH_LUT",
    "_IS_EXIT_CMAP_LUT",
    "_IS_HAZARD_CMAP_LUT",
    "_IS_ICE_GLYPH_LUT",
    "_IS_IRON_BARS_OR_TREE_LUT",
    "_IS_LAVA_GLYPH_LUT",
    "_IS_LOWERED_DRAWBRIDGE_GLYPH_LUT",
    "_IS_MONSTER_LUT",
    "_IS_OBJECT_LUT",
    "_IS_OPEN_DOOR_GLYPH_LUT",
    "_IS_PASSABLE_CMAP_LUT",
    "_IS_PET_LUT",
    "_IS_PORTAL_GLYPH_LUT",
    "_IS_SWALLOW_LUT",
    "_IS_TRAP_LUT",
    "_IS_WALL_GLYPH_LUT",
    "_IS_WARNING_LUT",
    "_IS_WATER_GLYPH_LUT",
    "_MAX_GLYPH",
    "_MG_INVIS",
    "_MG_PET",
    "_MG_RIDDEN",
    "_MG_STATUE",
    "_MON_INFO_BY_GLYPH",
    "_MON_INFO_BY_NAME",
    "_OBJ_INFO_BY_GLYPH",
    "_OBJ_INFO_BY_NAME",
    "_OCLASS_NAMES",
    "_OCLASS_TO_CHAR",
    "_TRAP_NAME_BY_GLYPH",
    "_make_mon_info",
    "_make_obj_info",
]
