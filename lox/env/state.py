"""
Lazy, Zero-Copy NetHack GameState ("Hard Code Nothing").
Exposes 100% of NLE's raw observation channels and native C symbol tables
(permonst, objclass, glyph classification, BL_MASK conditions) on demand
with zero per-tick computation when not queried.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nle import nethack as nh

from lox.env.grid_utils import astar_next_step, bfs_distance_grid

# Pre-build 1-to-1 C symbol table lookup caches from NLE's native C structs
# (Zero hardcoded game heuristics: purely mirrors NLE's C permonst & objclass arrays in Python for O(1) access)
_MAX_GLYPH: int = nh.MAX_GLYPH


@dataclass(frozen=True, slots=True)
class MonsterInfo:
    """1-to-1 mirror of NLE's C `permonst` struct for a monster glyph."""

    mon_id: int
    name: str
    symbol: str
    level: int
    speed: int
    ac: int
    mr: int
    align: int
    difficulty: int
    nutrition: int
    weight: int
    size: int
    resistances_mask: int
    conveys_mask: int
    flags1: int
    flags2: int
    flags3: int
    attacks: tuple[tuple[int, int, int, int], ...] = ()
    geno: int = 0

    # --- 1-to-1 NLE C `monflag.h` bitmask properties ---
    @property
    def is_acidic(self) -> bool:
        return bool(self.flags1 & 0x08000000)

    @property
    def is_poisonous(self) -> bool:
        return bool(self.flags1 & 0x10000000)

    @property
    def is_undead(self) -> bool:
        return bool(self.flags2 & 0x00000002)

    @property
    def is_were(self) -> bool:
        return bool(self.flags2 & 0x00000004)

    @property
    def is_human(self) -> bool:
        return bool(self.flags2 & 0x00000008)

    @property
    def is_elf(self) -> bool:
        return bool(self.flags2 & 0x00000010)

    @property
    def is_dwarf(self) -> bool:
        return bool(self.flags2 & 0x00000020)

    @property
    def is_gnome(self) -> bool:
        return bool(self.flags2 & 0x00000040)

    @property
    def is_orc(self) -> bool:
        return bool(self.flags2 & 0x00000080)

    @property
    def is_demon(self) -> bool:
        return bool(self.flags2 & 0x00000100)

    @property
    def is_giant(self) -> bool:
        return bool(self.flags2 & 0x00002000)

    @property
    def is_shapeshifter(self) -> bool:
        return bool(self.flags2 & 0x00004000)

    @property
    def is_domestic(self) -> bool:
        return bool(self.flags2 & 0x00400000)

    @property
    def conveys_fire_res(self) -> bool:
        return bool(self.conveys_mask & 0x01)

    @property
    def conveys_cold_res(self) -> bool:
        return bool(self.conveys_mask & 0x02)

    @property
    def conveys_sleep_res(self) -> bool:
        return bool(self.conveys_mask & 0x04)

    @property
    def conveys_disint_res(self) -> bool:
        return bool(self.conveys_mask & 0x08)

    @property
    def conveys_shock_res(self) -> bool:
        return bool(self.conveys_mask & 0x10)

    @property
    def conveys_poison_res(self) -> bool:
        return bool(self.conveys_mask & 0x20)

    @property
    def conveys_stone_flag(self) -> bool:
        return bool(self.conveys_mask & 0x80)

    @property
    def conveys_see_invis(self) -> bool:
        return bool(self.flags1 & 0x01000000)

    @property
    def conveys_teleport(self) -> bool:
        return bool(self.flags1 & 0x02000000)

    @property
    def conveys_teleport_control(self) -> bool:
        return bool(self.flags1 & 0x04000000)


# 1-to-1 NetHack object class ASCII character symbols and names
_OCLASS_TO_CHAR: dict[int, str] = {
    int(getattr(nh, "WEAPON_CLASS", 2)): ")",
    int(getattr(nh, "ARMOR_CLASS", 3)): "[",
    int(getattr(nh, "RING_CLASS", 4)): "=",
    int(getattr(nh, "AMULET_CLASS", 5)): '"',
    int(getattr(nh, "TOOL_CLASS", 6)): "(",
    int(getattr(nh, "FOOD_CLASS", 7)): "%",
    int(getattr(nh, "POTION_CLASS", 8)): "!",
    int(getattr(nh, "SCROLL_CLASS", 9)): "?",
    int(getattr(nh, "SPBOOK_CLASS", 10)): "+",
    int(getattr(nh, "WAND_CLASS", 11)): "/",
    int(getattr(nh, "COIN_CLASS", 12)): "$",
    int(getattr(nh, "GEM_CLASS", 13)): "*",
    int(getattr(nh, "ROCK_CLASS", 14)): "`",
    int(getattr(nh, "BALL_CLASS", 15)): "0",
    int(getattr(nh, "CHAIN_CLASS", 16)): "_",
    int(getattr(nh, "VENOM_CLASS", 17)): ".",
}

_OCLASS_NAMES: dict[int, str] = {
    int(getattr(nh, "WEAPON_CLASS", 2)): "weapon",
    int(getattr(nh, "ARMOR_CLASS", 3)): "armor",
    int(getattr(nh, "RING_CLASS", 4)): "ring",
    int(getattr(nh, "AMULET_CLASS", 5)): "amulet",
    int(getattr(nh, "TOOL_CLASS", 6)): "tool",
    int(getattr(nh, "FOOD_CLASS", 7)): "food",
    int(getattr(nh, "POTION_CLASS", 8)): "potion",
    int(getattr(nh, "SCROLL_CLASS", 9)): "scroll",
    int(getattr(nh, "SPBOOK_CLASS", 10)): "spellbook",
    int(getattr(nh, "WAND_CLASS", 11)): "wand",
    int(getattr(nh, "COIN_CLASS", 12)): "coin",
    int(getattr(nh, "GEM_CLASS", 13)): "gem",
    int(getattr(nh, "ROCK_CLASS", 14)): "rock",
    int(getattr(nh, "BALL_CLASS", 15)): "ball",
    int(getattr(nh, "CHAIN_CLASS", 16)): "chain",
    int(getattr(nh, "VENOM_CLASS", 17)): "venom",
}


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """1-to-1 mirror of NLE's C `objclass` struct for an object glyph."""

    obj_id: int
    name: str
    descr: str | None
    oclass: int
    oclass_name: str
    char: str
    weight: int
    cost: int
    delay: int
    color: int
    prob: int
    oprop: int = 0


class _CallableList(list):
    """List subclass that returns itself when called `xs()`, allowing both property and method syntax."""

    def __call__(self):
        return self


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """Single item in the player's current NetHack inventory."""

    letter: str
    name: str
    glyph: int
    oclass: int
    char: str
    quantity: int
    info: ObjectInfo | None = None
    corpse_monster: MonsterInfo | None = None

    @property
    def is_corpse(self) -> bool:
        return self.corpse_monster is not None or (
            0 <= self.glyph < _MAX_GLYPH and bool(_IS_BODY_LUT[self.glyph])
        )

    @property
    def is_weapon(self) -> bool:
        return self.oclass == 2

    @property
    def is_armor(self) -> bool:
        return self.oclass == 3

    @property
    def is_ring(self) -> bool:
        return self.oclass == 4

    @property
    def is_amulet(self) -> bool:
        return self.oclass == 5

    @property
    def is_tool(self) -> bool:
        return self.oclass == 6

    @property
    def is_food(self) -> bool:
        return self.oclass == 7

    @property
    def is_potion(self) -> bool:
        return self.oclass == 8

    @property
    def is_scroll(self) -> bool:
        return self.oclass == 9

    @property
    def is_spellbook(self) -> bool:
        return self.oclass == 10

    @property
    def is_wand(self) -> bool:
        return self.oclass == 11

    @property
    def is_coin(self) -> bool:
        return self.oclass == 12

    @property
    def is_gem(self) -> bool:
        return self.oclass == 13

    @property
    def is_worn(self) -> bool:
        return "(being worn)" in self.name

    @property
    def is_wielded(self) -> bool:
        return "(weapon in hand)" in self.name or "(wielded)" in self.name

    @property
    def is_quivered(self) -> bool:
        return "(in quiver)" in self.name

    @property
    def is_blessed(self) -> bool:
        return "blessed" in self.name

    @property
    def is_uncursed(self) -> bool:
        return "uncursed" in self.name

    @property
    def is_cursed(self) -> bool:
        return "cursed" in self.name

    @property
    def oclass_name(self) -> str:
        return _OCLASS_NAMES.get(self.oclass, "unknown")


@dataclass(frozen=True, slots=True)
class VisibleMonster:
    """A monster currently visible on the 21x79 screen grid."""

    y: int
    x: int
    glyph: int
    char: str
    color: int
    is_pet: bool
    is_statue: bool
    is_ridden: bool
    is_invisible: bool
    name: str
    info: MonsterInfo | None

    def is_adjacent_to(self, y: int, x: int) -> bool:
        """Returns True if this monster is Chebyshev adjacent (within 1 tile in any of 8 directions) to (y, x)."""
        return max(abs(self.y - y), abs(self.x - x)) == 1


@dataclass(frozen=True, slots=True)
class VisibleObject:
    """An object or corpse currently visible on the 21x79 screen grid."""

    y: int
    x: int
    glyph: int
    char: str
    color: int
    is_corpse: bool
    corpse_monster: MonsterInfo | None
    name: str
    info: ObjectInfo | None

    @property
    def is_food(self) -> bool:
        return self.is_corpse or (self.info is not None and self.info.oclass == 7)

    @property
    def is_weapon(self) -> bool:
        return self.info is not None and self.info.oclass == 2

    @property
    def is_armor(self) -> bool:
        return self.info is not None and self.info.oclass == 3

    @property
    def is_potion(self) -> bool:
        return self.info is not None and self.info.oclass == 8

    @property
    def is_scroll(self) -> bool:
        return self.info is not None and self.info.oclass == 9

    @property
    def is_wand(self) -> bool:
        return self.info is not None and self.info.oclass == 11


# Build O(1) lookup tables directly from NLE's C `permonst`, `objclass`, and `symdef` tables at import time
_MON_INFO_BY_GLYPH: list[MonsterInfo | None] = [None] * _MAX_GLYPH
_OBJ_INFO_BY_GLYPH: list[ObjectInfo | None] = [None] * _MAX_GLYPH
_CORPSE_MON_BY_GLYPH: list[MonsterInfo | None] = [None] * _MAX_GLYPH
_CMAP_NAME_BY_GLYPH: list[str | None] = [None] * _MAX_GLYPH
_TRAP_NAME_BY_GLYPH: list[str | None] = [None] * _MAX_GLYPH

_IS_MONSTER_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_PET_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_BODY_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_OBJECT_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_CMAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_TRAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_SWALLOW_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_WARNING_LUT = np.zeros(_MAX_GLYPH, dtype=bool)

# Precomputed character lookup table and fast ASCII translation table
_CHAR_LUT: tuple[str, ...] = tuple(
    chr(c) if 32 <= c <= 126 else "?" for c in range(256)
)
_ASCII_TRANS_TABLE: bytes = bytes(b if 32 <= b <= 126 else 32 for b in range(256))
_MG_PET: int = int(getattr(nh, "MG_PET", 0))
_MG_RIDDEN: int = int(getattr(nh, "MG_RIDDEN", 0))
_MG_STATUE: int = int(getattr(nh, "MG_STATUE", 0))
_MG_INVIS: int = int(getattr(nh, "MG_INVIS", 0))


_MON_INFO_BY_NAME: dict[str, MonsterInfo] = {}
_OBJ_INFO_BY_NAME: dict[str, ObjectInfo] = {}


def _make_mon_info(mid: int) -> MonsterInfo | None:
    if not (0 <= mid < nh.NUMMONS):
        return None
    try:
        pm = nh.permonst(mid)
        cs = nh.class_sym.from_mlet(pm.mlet)
        msym = cs.sym if cs and cs.sym else "?"
        mattk_raw = getattr(pm, "mattk", ())
        attacks = tuple(
            (int(a.aatyp), int(a.adtyp), int(a.damn), int(a.damd))
            for a in mattk_raw
            if int(getattr(a, "aatyp", 0)) != 0 or int(getattr(a, "adtyp", 0)) != 0
        )
        info = MonsterInfo(
            mon_id=mid,
            name=pm.mname or "",
            symbol=msym,
            level=int(pm.mlevel),
            speed=int(pm.mmove),
            ac=int(pm.ac),
            mr=int(pm.mr),
            align=int(getattr(pm, "maligntyp", 0)),
            difficulty=int(pm.difficulty),
            nutrition=int(pm.cnutrit),
            weight=int(pm.cwt),
            size=int(pm.msize),
            resistances_mask=int(pm.mresists),
            conveys_mask=int(pm.mconveys),
            flags1=int(pm.mflags1),
            flags2=int(pm.mflags2),
            flags3=int(pm.mflags3),
            attacks=attacks,
            geno=int(getattr(pm, "geno", 0)),
        )
        if info.name:
            _MON_INFO_BY_NAME[info.name.lower()] = info
        return info
    except Exception:
        return None


def _make_obj_info(oid: int) -> ObjectInfo | None:
    if not (0 <= oid < nh.NUM_OBJECTS):
        return None
    try:
        oc = nh.objclass(oid)
        oclass = ord(oc.oc_class) if isinstance(oc.oc_class, str) else int(oc.oc_class)
        name = nh.OBJ_NAME(oc)
        descr = nh.OBJ_DESCR(oc)
        info = ObjectInfo(
            obj_id=oid,
            name=name or "",
            descr=descr,
            oclass=oclass,
            oclass_name=_OCLASS_NAMES.get(oclass, "unknown"),
            char=_OCLASS_TO_CHAR.get(oclass, "?"),
            weight=int(oc.oc_weight),
            cost=int(oc.oc_cost),
            delay=int(oc.oc_delay),
            color=int(oc.oc_color),
            prob=int(oc.oc_prob),
            oprop=int(getattr(oc, "oc_oprop", 0)),
        )
        if info.name:
            _OBJ_INFO_BY_NAME[info.name.lower()] = info
        if info.descr:
            _OBJ_INFO_BY_NAME[info.descr.lower()] = info
        return info
    except Exception:
        return None


# Precompute symdef explanations for all MAXPCHARS
_SYMDEF_EXPLANATIONS: list[str] = []
for _idx in range(getattr(nh, "MAXPCHARS", 96)):
    try:
        _sd = nh.symdef.from_idx(_idx)
        _SYMDEF_EXPLANATIONS.append(_sd.explanation)
    except Exception:
        _SYMDEF_EXPLANATIONS.append("")

for _g in range(_MAX_GLYPH):
    if nh.glyph_is_monster(_g):
        _IS_MONSTER_LUT[_g] = True
        if nh.glyph_is_pet(_g):
            _IS_PET_LUT[_g] = True
        _mid = nh.glyph_to_mon(_g)
        _MON_INFO_BY_GLYPH[_g] = _make_mon_info(_mid)
    elif nh.glyph_is_body(_g):
        _IS_BODY_LUT[_g] = True
        _IS_OBJECT_LUT[_g] = True
        _mid = nh.glyph_to_mon(_g)
        _CORPSE_MON_BY_GLYPH[_g] = _make_mon_info(_mid)
    elif nh.glyph_is_object(_g):
        _IS_OBJECT_LUT[_g] = True
        _oid = nh.glyph_to_obj(_g)
        _OBJ_INFO_BY_GLYPH[_g] = _make_obj_info(_oid)
    elif nh.glyph_is_cmap(_g):
        _IS_CMAP_LUT[_g] = True
        _cid = nh.glyph_to_cmap(_g)
        if 0 <= _cid < len(_SYMDEF_EXPLANATIONS):
            _CMAP_NAME_BY_GLYPH[_g] = _SYMDEF_EXPLANATIONS[_cid]
    elif nh.glyph_is_swallow(_g):
        _IS_SWALLOW_LUT[_g] = True
    elif nh.glyph_is_warning(_g):
        _IS_WARNING_LUT[_g] = True

    if nh.glyph_is_trap(_g):
        _IS_TRAP_LUT[_g] = True
        try:
            _tid = nh.glyph_to_trap(_g)
            if 0 <= _tid < len(_SYMDEF_EXPLANATIONS):
                _TRAP_NAME_BY_GLYPH[_g] = _SYMDEF_EXPLANATIONS[_tid]
            else:
                _TRAP_NAME_BY_GLYPH[_g] = "trap"
        except Exception:
            _TRAP_NAME_BY_GLYPH[_g] = "trap"

_IS_WALL_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_OPEN_DOOR_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_DOORWAY_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_IRON_BARS_OR_TREE_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_WATER_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_LAVA_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_ICE_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_DRAWBRIDGE_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_LOWERED_DRAWBRIDGE_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_BOULDER_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)

for _g in range(_MAX_GLYPH):
    _cn = _CMAP_NAME_BY_GLYPH[_g]
    if _cn == "wall":
        _IS_WALL_GLYPH_LUT[_g] = True
    elif _cn == "open door":
        _IS_OPEN_DOOR_GLYPH_LUT[_g] = True
        _IS_DOORWAY_GLYPH_LUT[_g] = True
    elif _cn == "closed door":
        _IS_DOORWAY_GLYPH_LUT[_g] = True
    if _cn in ("iron bars", "tree"):
        _IS_IRON_BARS_OR_TREE_LUT[_g] = True
    if _cn is not None and ("water" in _cn or "pool" in _cn or "moat" in _cn):
        _IS_WATER_GLYPH_LUT[_g] = True
    if _cn is not None and "lava" in _cn:
        _IS_LAVA_GLYPH_LUT[_g] = True
    if _cn == "ice":
        _IS_ICE_GLYPH_LUT[_g] = True
    if _cn is not None and "drawbridge" in _cn:
        _IS_DRAWBRIDGE_GLYPH_LUT[_g] = True
    if _cn == "lowered drawbridge":
        _IS_LOWERED_DRAWBRIDGE_GLYPH_LUT[_g] = True
    if _OBJ_INFO_BY_GLYPH[_g] is not None and _OBJ_INFO_BY_GLYPH[_g].name == "boulder":
        _IS_BOULDER_GLYPH_LUT[_g] = True


@dataclass
class FloorMemory:
    """Zero-heuristic spatial bookkeeping for a single (dungeon_num, level_num) floor."""

    visited: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.bool_)
    )
    visit_count: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.int32)
    )
    remembered_chars: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.uint8)
    )
    searched_counts: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.int16)
    )
    door_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.bool_)
    )
    remembered_obj_glyphs: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.int32)
    )
    locked_doors: set[tuple[int, int]] = field(default_factory=set)

    @property
    def passable_mask(self) -> np.ndarray:
        """Returns boolean mask of known walkable tiles (floor, corridor, open door, stairs, altar, fountain, throne)."""
        c = self.remembered_chars
        return (
            (c == ord("."))
            | (c == ord("#"))
            | (c == ord(">"))
            | (c == ord("<"))
            | (c == ord("_"))
            | (c == ord("{"))
            | (c == ord("\\"))
        )

    def ensure_shape(self, min_h: int, min_w: int) -> None:
        """Dynamically resizes spatial grids if game coordinates exceed initial bounds (e.g. Craftax 48x48)."""
        cur_h, cur_w = self.visited.shape
        if cur_h < min_h or cur_w < min_w:
            new_h = max(cur_h, min_h)
            new_w = max(cur_w, min_w)

            new_visited = np.zeros((new_h, new_w), dtype=np.bool_)
            new_visited[:cur_h, :cur_w] = self.visited
            self.visited = new_visited

            new_visit_count = np.zeros((new_h, new_w), dtype=np.int32)
            new_visit_count[:cur_h, :cur_w] = self.visit_count
            self.visit_count = new_visit_count

            new_remembered = np.zeros((new_h, new_w), dtype=np.uint8)
            new_remembered[:cur_h, :cur_w] = self.remembered_chars
            self.remembered_chars = new_remembered

            new_searched = np.zeros((new_h, new_w), dtype=np.int16)
            new_searched[:cur_h, :cur_w] = self.searched_counts
            self.searched_counts = new_searched

            new_door_mask = np.zeros((new_h, new_w), dtype=np.bool_)
            new_door_mask[:cur_h, :cur_w] = self.door_mask
            self.door_mask = new_door_mask

            new_obj_glyphs = np.zeros((new_h, new_w), dtype=np.int32)
            new_obj_glyphs[:cur_h, :cur_w] = self.remembered_obj_glyphs
            self.remembered_obj_glyphs = new_obj_glyphs


class WorldMemory:
    """
    Zero-heuristic, deterministic spatial and episodic memory.
    Maintained by NetHackHarness across steps and replays.
    Tracks per-floor exploration grids, remembered static tiles,
    locked door coordinates, position history, and inter-floor stair graph.
    """

    __slots__ = (
        "floors",
        "pos_history",
        "current_floor_key",
        "stair_graph",
        "kv",
        "recent_messages",
        "prev_turn",
        "consecutive_stalls",
        "last_action_zero_turn",
    )

    def __init__(self) -> None:
        self.floors: dict[tuple[int, int], FloorMemory] = {}
        self.pos_history: collections.deque[tuple[int, int, int, int]] = (
            collections.deque(maxlen=32)
        )
        self.current_floor_key: tuple[int, int] = (0, 1)
        self.stair_graph: dict[
            tuple[int, int, int, int], tuple[int, int, int, int]
        ] = {}
        self.kv: dict[str, Any] = {}
        self.recent_messages: collections.deque[tuple[int, int, str]] = (
            collections.deque(maxlen=25)
        )
        self.prev_turn: int = -1
        self.consecutive_stalls: int = 0
        self.last_action_zero_turn: bool = False

    def reset(self) -> None:
        self.floors.clear()
        self.pos_history.clear()
        self.current_floor_key = (0, 1)
        self.stair_graph.clear()
        self.kv.clear()
        self.recent_messages.clear()
        self.prev_turn = -1
        self.consecutive_stalls = 0
        self.last_action_zero_turn = False

    def get_floor(
        self, dnum: int, dlvl: int, shape: tuple[int, int] = (21, 79)
    ) -> FloorMemory:
        key = (int(dnum), int(dlvl))
        if key not in self.floors:
            self.floors[key] = FloorMemory(
                visited=np.zeros(shape, dtype=np.bool_),
                visit_count=np.zeros(shape, dtype=np.int32),
                remembered_chars=np.zeros(shape, dtype=np.uint8),
                searched_counts=np.zeros(shape, dtype=np.int16),
                door_mask=np.zeros(shape, dtype=np.bool_),
                remembered_obj_glyphs=np.zeros(shape, dtype=np.int32),
            )
        return self.floors[key]

    @property
    def current_floor(self) -> FloorMemory:
        return self.get_floor(self.current_floor_key[0], self.current_floor_key[1])

    @property
    def visited(self) -> np.ndarray:
        return self.current_floor.visited

    @property
    def visit_count(self) -> np.ndarray:
        return self.current_floor.visit_count

    @property
    def remembered_chars(self) -> np.ndarray:
        return self.current_floor.remembered_chars

    @property
    def door_mask(self) -> np.ndarray:
        return self.current_floor.door_mask

    @property
    def passable_mask(self) -> np.ndarray:
        return self.current_floor.passable_mask

    @property
    def locked_doors(self) -> set[tuple[int, int]]:
        return self.current_floor.locked_doors

    @property
    def searched_counts(self) -> np.ndarray:
        return self.current_floor.searched_counts

    @property
    def is_oscillating(self) -> bool:
        if len(self.pos_history) < 12:
            return False
        recent = list(self.pos_history)[-12:]
        f0 = (recent[0][0], recent[0][1])
        if any((p[0], p[1]) != f0 for p in recent):
            return False
        unique_tiles = {(p[2], p[3]) for p in recent}
        return len(unique_tiles) <= 3

    def update(self, state: GameState, action_name: str = "") -> None:
        try:
            dnum = int(state.dungeon_num)
            dlvl = int(state.level_num)
            sy = int(state.y)
            sx = int(state.x)
        except Exception:
            return

        # Track in-game turn advancement (zero-turn action awareness)
        try:
            curr_turn = int(getattr(state, "turns", getattr(state, "turn", 0)))
            step_cnt = getattr(state, "step_count", 0)
            if self.prev_turn >= 0 and curr_turn == self.prev_turn and step_cnt > 0:
                self.consecutive_stalls += 1
                self.last_action_zero_turn = True
            else:
                self.consecutive_stalls = 0
                self.last_action_zero_turn = False
            self.prev_turn = curr_turn
        except Exception:
            pass

        gh, gw = 21, 79
        if hasattr(state, "chars") and hasattr(state.chars, "shape"):
            gh, gw = state.chars.shape
        elif hasattr(state, "map") and hasattr(state.map, "shape"):
            gh, gw = state.map.shape
        else:
            gh, gw = max(21, sy + 1), max(79, sx + 1)

        self.current_floor_key = (dnum, dlvl)
        floor = self.get_floor(dnum, dlvl, shape=(gh, gw))
        floor.ensure_shape(gh, gw)

        if 0 <= sy < floor.visited.shape[0] and 0 <= sx < floor.visited.shape[1]:
            floor.visited[sy, sx] = True
            floor.visit_count[sy, sx] += 1
            self.pos_history.append((dnum, dlvl, sy, sx))

        # Update searched counts if SEARCH was executed
        if action_name and "search" in action_name.lower():
            floor.searched_counts[
                max(0, sy - 1) : min(floor.visited.shape[0], sy + 2),
                max(0, sx - 1) : min(floor.visited.shape[1], sx + 2),
            ] += 1

        # Track inter-floor stair graph transitions
        if len(self.pos_history) >= 2:
            prev = self.pos_history[-2]
            now = self.pos_history[-1]
            if (prev[0], prev[1]) != (now[0], now[1]):
                self.stair_graph[prev] = now
                self.stair_graph[now] = prev

        if hasattr(state, "message") and state.message:
            msg_str = str(state.message).strip()
            if msg_str:
                self.recent_messages.append(
                    (int(getattr(state, "turns", 0)), int(dlvl), msg_str)
                )

        # Update remembered terrain chars and doorway mask using C engine cmap glyph lookup
        try:
            glyphs = state.glyphs
            chars = state.chars
            is_cmap = _IS_CMAP_LUT[glyphs]

            # 1. Update remembered_chars for cmap terrain
            mask = is_cmap & (chars != 0) & (chars != ord(" "))
            floor.remembered_chars[mask] = chars[mask]

            # Normalize open doors to '.' so they are never confused with walls ('-' or '|')
            open_door_mask = is_cmap & _IS_OPEN_DOOR_GLYPH_LUT[glyphs]
            floor.remembered_chars[open_door_mask] = ord(".")

            # Normalize iron bars and trees to '|' so they are NEVER treated as walkable '#' corridors
            impassable_cmap = is_cmap & _IS_IRON_BARS_OR_TREE_LUT[glyphs]
            floor.remembered_chars[impassable_cmap] = ord("|")

            # Normalize ice and lowered drawbridge to '.' so they are walkable in passable_mask
            walkable_special = is_cmap & (
                _IS_ICE_GLYPH_LUT[glyphs] | _IS_LOWERED_DRAWBRIDGE_GLYPH_LUT[glyphs]
            )
            floor.remembered_chars[walkable_special] = ord(".")

            # 2. Update door_mask
            doorway_mask = is_cmap & _IS_DOORWAY_GLYPH_LUT[glyphs]
            floor.door_mask[doorway_mask] = True
            # Clear doorway mask where cmap is visible but no longer an intact doorway
            floor.door_mask[is_cmap & (~_IS_DOORWAY_GLYPH_LUT[glyphs])] = False

            # 3. For tiles with visible objects, monsters, or hero where remembered_chars == 0, mark as floor '.'
            h_curr, w_curr = floor.remembered_chars.shape
            grid_y, grid_x = np.ogrid[:h_curr, :w_curr]
            hero_mask = (grid_y == sy) & (grid_x == sx)
            non_cmap_floor = (
                _IS_OBJECT_LUT[glyphs] | _IS_MONSTER_LUT[glyphs]
            ) | hero_mask
            uninit_occupied = non_cmap_floor & (floor.remembered_chars == 0)
            floor.remembered_chars[uninit_occupied] = ord(".")

            msg_low = str(state.message).lower() if hasattr(state, "message") else ""
            if (
                0 <= sy < floor.remembered_chars.shape[0]
                and 0 <= sx < floor.remembered_chars.shape[1]
            ):
                if (
                    getattr(state, "is_on_stairs_down", False)
                    or "staircase down" in msg_low
                ):
                    floor.remembered_chars[sy, sx] = ord(">")
                elif "staircase up" in msg_low:
                    floor.remembered_chars[sy, sx] = ord("<")
                elif "there is an altar" in msg_low or "altar of" in msg_low:
                    floor.remembered_chars[sy, sx] = ord("_")
                elif "there is a fountain" in msg_low:
                    floor.remembered_chars[sy, sx] = ord("{")

            # 4. Update remembered_obj_glyphs
            floor.remembered_obj_glyphs[is_cmap] = 0
            obj_mask = _IS_OBJECT_LUT[glyphs]
            floor.remembered_obj_glyphs[obj_mask] = glyphs[obj_mask]

            if (
                0 <= sy < floor.remembered_obj_glyphs.shape[0]
                and 0 <= sx < floor.remembered_obj_glyphs.shape[1]
            ):
                act_upper = action_name.upper() if action_name else ""
                msg_low = (
                    str(state.message).lower() if hasattr(state, "message") else ""
                )
                if (
                    "you see no objects here" in msg_low
                    or "feel no objects here" in msg_low
                ):
                    floor.remembered_obj_glyphs[sy, sx] = 0
                elif any(tok in act_upper for tok in ("PICKUP", "EAT", "COMMA")):
                    if not any(
                        abort_tok in msg_low
                        for abort_tok in (
                            "never mind",
                            "don't have",
                            "nothing here",
                            "there are ",
                            "you see here",
                            "objects here",
                        )
                    ):
                        floor.remembered_obj_glyphs[sy, sx] = 0
                elif (
                    len(self.pos_history) >= 2
                    and self.pos_history[-1] != self.pos_history[-2]
                ):
                    if not any(
                        tok in msg_low
                        for tok in (
                            "you see here",
                            "objects here",
                            "things that are here",
                            "you feel here",
                            "there is a ",
                            "there is an ",
                            "there are ",
                        )
                    ):
                        floor.remembered_obj_glyphs[sy, sx] = 0
        except Exception:
            pass

        # Update locked doors: prune opened/destroyed doors from memory if recorded by policy
        try:
            for dy_door, dx_door in list(floor.locked_doors):
                if 0 <= dy_door < 21 and 0 <= dx_door < 79:
                    c = state.chars[dy_door, dx_door]
                    if c != 0 and c != ord(" ") and c != ord("+"):
                        floor.locked_doors.discard((dy_door, dx_door))
        except Exception:
            pass


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
    )

    def __init__(
        self,
        raw_obs: dict[str, np.ndarray] | tuple[np.ndarray, ...],
        step_count: int = 0,
        mem: WorldMemory | None = None,
        obs_keys: tuple[str, ...] | list[str] | None = None,
    ):
        if isinstance(raw_obs, (tuple, list)):
            self._raw_dict = None
            if obs_keys is not None:
                idx = {k: i for i, k in enumerate(obs_keys)}
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
        self._blstats_list: list[int] = self.blstats.tolist()
        self._message_cache: str | None = None
        self._inventory_cache: list[InventoryItem] | None = None
        self._monsters_cache: list[VisibleMonster] | None = None
        self._monsters_no_self_cache: list[VisibleMonster] | None = None
        self._objects_cache: list[VisibleObject] | None = None
        self._tty_text_cache: str | None = None
        self._tty_lines_cache: list[str] | None = None

        if mem is None:
            self.mem = WorldMemory()
            self.mem.update(self)
        else:
            self.mem = mem

    @property
    def raw(self) -> dict[str, np.ndarray]:
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

    # --- 1-to-1 NLE blstats properties (O(1) Python list indexing) ---
    @property
    def x(self) -> int:
        return self._blstats_list[0]

    @property
    def y(self) -> int:
        return self._blstats_list[1]

    @property
    def pos(self) -> tuple[int, int]:
        return (self._blstats_list[1], self._blstats_list[0])

    @property
    def strength_pct(self) -> int:
        return self._blstats_list[2]

    @property
    def strength(self) -> int:
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
    def score(self) -> int:
        return self._blstats_list[9]

    @property
    def hp(self) -> int:
        return self._blstats_list[10]

    @property
    def max_hp(self) -> int:
        return self._blstats_list[11]

    @property
    def depth(self) -> int:
        return self._blstats_list[12]

    @property
    def gold(self) -> int:
        return self._blstats_list[13]

    @property
    def energy(self) -> int:
        return self._blstats_list[14]

    @property
    def max_energy(self) -> int:
        return self._blstats_list[15]

    @property
    def ac(self) -> int:
        return self._blstats_list[16]

    @property
    def monster_level(self) -> int:
        return self._blstats_list[17]

    @property
    def xp_level(self) -> int:
        return self._blstats_list[18]

    @property
    def xp(self) -> int:
        return self._blstats_list[19]

    @property
    def turn(self) -> int:
        return self._blstats_list[20]

    @property
    def hunger_state(self) -> int:
        """0=Satiated, 1=Not Hungry, 2=Hungry, 3=Weak, 4=Fainting, 5=Fainted, 6=Starved."""
        return self._blstats_list[21]

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
    def carrying_capacity(self) -> int:
        """0=Unencumbered, 1=Burdened, 2=Stressed, 3=Strained, 4=Overtaxed, 5=Overloaded."""
        return self._blstats_list[22]

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
    def dungeon_num(self) -> int:
        """0=Dungeons of Doom, 1=Gehennom, 2=Gnomish Mines, 3=Quest, 4=Sokoban, 5=Fort Ludios, 6=Vlad's Tower, 7=Elemental Planes."""
        return self._blstats_list[23]

    @property
    def level_num(self) -> int:
        return self._blstats_list[24]

    @property
    def condition_mask(self) -> int:
        return self._blstats_list[25] if len(self._blstats_list) > 25 else 0

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
    def char_at(self, y: int, x: int) -> str:
        if 0 <= y < 21 and 0 <= x < 79:
            c = int(self.chars[y, x])
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
            return int(self.glyphs[y, x])
        return nh.GLYPH_NOTHING

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
        """O(1) lookup of an object's C `objclass` struct by name or description."""
        if not name:
            return None
        s = name.lower().strip()
        for prefix in ("a ", "an ", "the ", "blessed ", "uncursed ", "cursed "):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        return _OBJ_INFO_BY_NAME.get(s)

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
            cmap = _CMAP_NAME_BY_GLYPH[g]
            if cmap is not None:
                return cmap
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
            g = int(self.glyphs[y, x])
            return _TRAP_NAME_BY_GLYPH[g]
        return None

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
            return bool(_IS_TRAP_LUT[int(self.glyphs[y, x])])
        return False

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
    def is_in_moveloop(self) -> bool:
        if self.program_state is not None and len(self.program_state) > 3:
            return bool(self.program_state[3])
        return True

    @property
    def in_moveloop(self) -> bool:
        return self.is_in_moveloop

    @property
    def is_in_yn(self) -> bool:
        if self.program_state is not None and len(self.program_state) > 1:
            return bool(self.program_state[1])
        return False

    @property
    def in_yn_prompt(self) -> bool:
        if self.is_in_yn:
            return True
        msg = self.message.lower()
        return (
            "[yn]" in msg or "[ynq]" in msg or "(yn)" in msg or "really attack" in msg
        )

    @property
    def is_in_getlin(self) -> bool:
        if self.program_state is not None and len(self.program_state) > 2:
            return bool(self.program_state[2])
        return False

    @property
    def in_text_prompt(self) -> bool:
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
                if has_mem_obj:
                    rem_g = int(self.mem.current_floor.remembered_obj_glyphs[hy, hx])
                    if _IS_OBJECT_LUT[rem_g]:
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

    def is_water(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is water, pool, or moat according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        return bool(_IS_WATER_GLYPH_LUT[int(self.glyphs[y, x])])

    def is_lava(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is molten lava according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        return bool(_IS_LAVA_GLYPH_LUT[int(self.glyphs[y, x])])

    def is_ice(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is ice according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        return bool(_IS_ICE_GLYPH_LUT[int(self.glyphs[y, x])])

    def is_iron_bars(self, y: int, x: int) -> bool:
        """Returns True if (y, x) is iron bars or tree according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        return bool(_IS_IRON_BARS_OR_TREE_LUT[int(self.glyphs[y, x])])

    def is_boulder(self, y: int, x: int) -> bool:
        """Returns True if (y, x) contains a boulder according to the C engine symbol table."""
        if not (0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1]):
            return False
        return bool(_IS_BOULDER_GLYPH_LUT[int(self.glyphs[y, x])])

    @property
    def boulder_positions(self) -> list[tuple[int, int]]:
        """Returns (y, x) coordinates of all visible boulders on the current floor."""
        ys, xs = np.where(_IS_BOULDER_GLYPH_LUT[self.glyphs])
        return list(zip(ys.tolist(), xs.tolist()))

    @property
    def passable_mask(self) -> np.ndarray:
        """Returns boolean mask of known walkable tiles (floor, corridor, open door, stairs, altar, fountain, throne)."""
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
        )

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
        """Runs Numba-accelerated BFS distance calculation from `start` (defaults to `(self.y, self.x)`)."""
        sy, sx = start if start is not None else (self.y, self.x)
        p = (
            passable_mask
            if isinstance(passable_mask, np.ndarray) and passable_mask.dtype == np.bool_
            else np.asarray(passable_mask, dtype=np.bool_)
        )
        if no_diag is None and self.mem is not None:
            cur_f = getattr(self.mem, "current_floor", None)
            if (
                cur_f is not None
                and hasattr(cur_f, "door_mask")
                and cur_f.door_mask.shape == p.shape
            ):
                no_diag = cur_f.door_mask

        return bfs_distance_grid(
            p,
            int(sy),
            int(sx),
            bool(allow_diagonal),
            no_diag=no_diag,
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
        if no_diag is None and self.mem is not None:
            cur_f = getattr(self.mem, "current_floor", None)
            if (
                cur_f is not None
                and hasattr(cur_f, "door_mask")
                and cur_f.door_mask.shape == p.shape
            ):
                no_diag = cur_f.door_mask

        return astar_next_step(
            p,
            int(sy),
            int(sx),
            int(target[0]),
            int(target[1]),
            bool(allow_diagonal),
            no_diag=no_diag,
        )
