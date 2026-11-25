"""
NetHack 3.6.6 C Symbol Table Types & O(1) Lookup Tables ("Hard Code Nothing").
Pure 1-to-1 mirrors of NLE's C `permonst`, `objclass`, `symdef`, and glyph LUTs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from nle import nethack as _nle_nh

nh: Any = _nle_nh

lox_core: Any = None
try:
    import lox_core as _lox_core_mod  # type: ignore[import-not-found]

    lox_core = _lox_core_mod
    _HAS_RUST_FLOOR_MEM = hasattr(lox_core, "fast_update_floor_memory")
except Exception:
    lox_core = None
    _HAS_RUST_FLOOR_MEM = False

# Pre-build 1-to-1 C symbol table lookup caches from NLE's native C structs
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

_SPBOOK_OCLASS: int = int(getattr(nh, "SPBOOK_CLASS", 10))


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


_RE_BLESSED = re.compile(r"\bblessed\b")
_RE_UNCURSED = re.compile(r"\buncursed\b")
_RE_CURSED = re.compile(r"(?<!un)\bcursed\b")
_RE_ENCHANT = re.compile(r"(?<!\w)([+-]\d+)\b")
_RE_WIELDED = re.compile(r"\(weapon in (?:hands?|(?:left|right) hand)\)|\(wielded\)")
_RE_LEAD_ENCHANT = re.compile(r"^[+-]\d+\s+")
_RE_LEAD_QTY = re.compile(r"^\d+\s+")


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
        return (
            "(being worn)" in self.name
            or "(on right hand)" in self.name
            or "(on left hand)" in self.name
        )

    @property
    def is_wielded(self) -> bool:
        return bool(_RE_WIELDED.search(self.name))

    @property
    def is_quivered(self) -> bool:
        return "(in quiver)" in self.name

    @property
    def is_blessed(self) -> bool:
        return "blessed" in self.name and bool(_RE_BLESSED.search(self.name))

    @property
    def is_uncursed(self) -> bool:
        return "uncursed" in self.name and bool(_RE_UNCURSED.search(self.name))

    @property
    def is_cursed(self) -> bool:
        return "cursed" in self.name and bool(_RE_CURSED.search(self.name))

    @property
    def is_for_sale(self) -> bool:
        return "(unpaid" in self.name or "for sale" in self.name

    @property
    def enchantment(self) -> int | None:
        if "+" not in self.name and "-" not in self.name:
            return None
        m = _RE_ENCHANT.search(self.name)
        return int(m.group(1)) if m else None

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

    @property
    def is_ring(self) -> bool:
        return self.info is not None and self.info.oclass == 4

    @property
    def is_amulet(self) -> bool:
        return self.info is not None and self.info.oclass == 5

    @property
    def is_tool(self) -> bool:
        return self.info is not None and self.info.oclass == 6

    @property
    def is_spellbook(self) -> bool:
        return self.info is not None and self.info.oclass == 10

    @property
    def is_coin(self) -> bool:
        return self.info is not None and self.info.oclass == 12

    @property
    def is_gem(self) -> bool:
        return self.info is not None and self.info.oclass == 13

    @property
    def is_boulder(self) -> bool:
        return self.info is not None and self.info.name == "boulder"


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
_OBJ_INFO_BY_QUALIFIED_NAME: dict[str, ObjectInfo] = {}

_CLASS_OF_PHRASES: dict[int, tuple[str, ...]] = {
    int(getattr(nh, "RING_CLASS", 4)): ("ring of", "rings of"),
    int(getattr(nh, "AMULET_CLASS", 5)): ("amulet of", "amulets of"),
    int(getattr(nh, "POTION_CLASS", 8)): ("potion of", "potions of"),
    int(getattr(nh, "SCROLL_CLASS", 9)): (
        "scroll of",
        "scrolls of",
        "scroll labeled",
        "scrolls labeled",
    ),
    int(getattr(nh, "SPBOOK_CLASS", 10)): ("spellbook of", "spellbooks of"),
    int(getattr(nh, "WAND_CLASS", 11)): ("wand of", "wands of"),
}


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


def _index_bare_obj_name(key: str, info: ObjectInfo) -> None:
    existing = _OBJ_INFO_BY_NAME.get(key)
    if existing is None:
        _OBJ_INFO_BY_NAME[key] = info
    elif existing.oclass == _SPBOOK_OCLASS and info.oclass != _SPBOOK_OCLASS:
        # Never let a spellbook shadow an earlier potion/scroll/ring/wand of the same base name
        _OBJ_INFO_BY_NAME[key] = info


def _make_obj_info(oid: int) -> ObjectInfo | None:
    if not (0 <= oid < nh.NUM_OBJECTS):
        return None
    try:
        oc = nh.objclass(oid)
        oclass = ord(oc.oc_class) if isinstance(oc.oc_class, str) else int(oc.oc_class)
        name = nh.OBJ_NAME(oc)
        descr = nh.OBJ_DESCR(oc)
        oclass_name = _OCLASS_NAMES.get(oclass, "unknown")
        info = ObjectInfo(
            obj_id=oid,
            name=name or "",
            descr=descr,
            oclass=oclass,
            oclass_name=oclass_name,
            char=_OCLASS_TO_CHAR.get(oclass, "?"),
            weight=int(oc.oc_weight),
            cost=int(oc.oc_cost),
            delay=int(oc.oc_delay),
            color=int(oc.oc_color),
            prob=int(oc.oc_prob),
            oprop=int(getattr(oc, "oc_oprop", 0)),
        )
        if info.name:
            low_name = info.name.lower()
            _index_bare_obj_name(low_name, info)
            _OBJ_INFO_BY_QUALIFIED_NAME[f"{oclass_name} of {low_name}"] = info
            for phrase in _CLASS_OF_PHRASES.get(oclass, ()):
                _OBJ_INFO_BY_QUALIFIED_NAME[f"{phrase} {low_name}"] = info
        if info.descr:
            low_descr = info.descr.lower()
            _index_bare_obj_name(low_descr, info)
            _OBJ_INFO_BY_QUALIFIED_NAME[f"{low_descr} {oclass_name}"] = info
            for phrase in _CLASS_OF_PHRASES.get(oclass, ()):
                _OBJ_INFO_BY_QUALIFIED_NAME[f"{phrase} {low_descr}"] = info
        return info
    except Exception:
        return None


def lookup_object_info_by_name(name: str) -> ObjectInfo | None:
    """O(1) lookup of an object's C `objclass` struct by name, class-prefixed phrase, or description."""
    if not name:
        return None
    s = name.lower().strip()
    # Strip leading quantity, articles, BUC status, and enchantment prefixes iteratively
    changed = True
    while changed and s:
        changed = False
        m_qty = _RE_LEAD_QTY.match(s)
        if m_qty:
            s = s[m_qty.end() :].strip()
            changed = True
        for prefix in (
            "a ",
            "an ",
            "the ",
            "blessed ",
            "uncursed ",
            "cursed ",
            "rusty ",
            "corroded ",
            "burnt ",
            "rotted ",
            "thoroughly ",
            "very ",
            "diluted ",
            "partially used ",
        ):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
                changed = True
        m_ench = _RE_LEAD_ENCHANT.match(s)
        if m_ench:
            s = s[m_ench.end() :].strip()
            changed = True
    # Strip trailing parenthetical annotations like "(weapon in hand)" or "(0:4)"
    paren_idx = s.find(" (")
    if paren_idx != -1:
        s = s[:paren_idx].strip()

    hit = _OBJ_INFO_BY_QUALIFIED_NAME.get(s)
    if hit is not None:
        return hit
    hit = _OBJ_INFO_BY_NAME.get(s)
    if hit is not None:
        return hit
    # Try simple singularization (e.g. "daggers" -> "dagger", "rations" -> "ration")
    if s.endswith("s") and len(s) > 3:
        sing = s[:-1]
        hit = _OBJ_INFO_BY_QUALIFIED_NAME.get(sing) or _OBJ_INFO_BY_NAME.get(sing)
        if hit is not None:
            return hit
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
        _TRAP_NAME_BY_GLYPH[_g] = _CMAP_NAME_BY_GLYPH[_g] or "trap"

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
_IS_PORTAL_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_PASSABLE_CMAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_HAZARD_CMAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_EXIT_CMAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_CLOSED_DOOR_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)

_WALKABLE_CMAP_NAMES = frozenset(
    {
        "doorway",
        "open door",
        "floor of a room",
        "dark part of a room",
        "corridor",
        "lit corridor",
        "staircase up",
        "staircase down",
        "ladder up",
        "ladder down",
        "altar",
        "grave",
        "opulent throne",
        "sink",
        "fountain",
        "ice",
        "lowered drawbridge",
        "air",
        "cloud",
        "poison cloud",
        "magic portal",
        "vibrating square",
    }
)
_EXIT_CMAP_NAMES = frozenset(
    {"staircase up", "staircase down", "ladder up", "ladder down", "magic portal"}
)

for _g in range(_MAX_GLYPH):
    _cn = _CMAP_NAME_BY_GLYPH[_g]
    if _cn == "wall":
        _IS_WALL_GLYPH_LUT[_g] = True
    elif _cn == "open door":
        _IS_OPEN_DOOR_GLYPH_LUT[_g] = True
        _IS_DOORWAY_GLYPH_LUT[_g] = True
    elif _cn == "closed door":
        _IS_DOORWAY_GLYPH_LUT[_g] = True
        _IS_CLOSED_DOOR_GLYPH_LUT[_g] = True
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
    if _cn is not None and "magic portal" in _cn:
        _IS_PORTAL_GLYPH_LUT[_g] = True
    if _IS_CMAP_LUT[_g]:
        _cid_g = nh.glyph_to_cmap(_g)
        if _cid_g > 0 and ((_cn in _WALKABLE_CMAP_NAMES) or _IS_TRAP_LUT[_g]):
            _IS_PASSABLE_CMAP_LUT[_g] = True
        if _cn in _EXIT_CMAP_NAMES:
            _IS_EXIT_CMAP_LUT[_g] = True
        if (
            _IS_TRAP_LUT[_g]
            or _IS_WATER_GLYPH_LUT[_g]
            or _IS_LAVA_GLYPH_LUT[_g]
            or _cn in ("raised drawbridge", "poison cloud")
        ):
            _IS_HAZARD_CMAP_LUT[_g] = True
    _obj_g = _OBJ_INFO_BY_GLYPH[_g]
    if _obj_g is not None and _obj_g.name == "boulder":
        _IS_BOULDER_GLYPH_LUT[_g] = True

# Bit-packed uint16 flags LUT consumed by Rust `lox_core.fast_update_floor_memory`
_FLOOR_GLYPH_FLAGS_LUT = (
    _IS_CMAP_LUT.astype(np.uint16)
    | (_IS_OPEN_DOOR_GLYPH_LUT.astype(np.uint16) << 1)
    | (_IS_IRON_BARS_OR_TREE_LUT.astype(np.uint16) << 2)
    | ((_IS_ICE_GLYPH_LUT | _IS_LOWERED_DRAWBRIDGE_GLYPH_LUT).astype(np.uint16) << 3)
    | (_IS_DOORWAY_GLYPH_LUT.astype(np.uint16) << 4)
    | (_IS_OBJECT_LUT.astype(np.uint16) << 5)
    | (_IS_MONSTER_LUT.astype(np.uint16) << 6)
    | (_IS_TRAP_LUT.astype(np.uint16) << 7)
    | (_IS_PASSABLE_CMAP_LUT.astype(np.uint16) << 8)
    | (_IS_HAZARD_CMAP_LUT.astype(np.uint16) << 9)
)
