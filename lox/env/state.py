"""
Lazy, Zero-Copy NetHack GameState ("Hard Code Nothing").
Exposes 100% of NLE's raw observation channels and native C symbol tables
(permonst, objclass, glyph classification, BL_MASK conditions) on demand
with zero per-tick computation when not queried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from nle import nethack as nh

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


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """1-to-1 mirror of NLE's C `objclass` struct for an object glyph."""

    obj_id: int
    oclass: int
    weight: int
    cost: int
    delay: int
    color: int
    prob: int


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


@dataclass(frozen=True, slots=True)
class VisibleMonster:
    """A monster currently visible on the 21x79 screen grid."""

    y: int
    x: int
    glyph: int
    char: str
    color: int
    is_pet: bool
    is_peaceful: bool
    info: MonsterInfo | None


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
    info: ObjectInfo | None


# Build O(1) lookup tables directly from NLE's C `permonst` and `objclass` tables at import time
_MON_INFO_BY_GLYPH: list[MonsterInfo | None] = [None] * _MAX_GLYPH
_OBJ_INFO_BY_GLYPH: list[ObjectInfo | None] = [None] * _MAX_GLYPH
_CORPSE_MON_BY_GLYPH: list[MonsterInfo | None] = [None] * _MAX_GLYPH

_IS_MONSTER_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_PET_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_BODY_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_OBJECT_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_CMAP_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_SWALLOW_LUT = np.zeros(_MAX_GLYPH, dtype=bool)
_IS_WARNING_LUT = np.zeros(_MAX_GLYPH, dtype=bool)


def _make_mon_info(mid: int) -> MonsterInfo | None:
    if not (0 <= mid < nh.NUMMONS):
        return None
    try:
        pm = nh.permonst(mid)
        mlet = chr(pm.mlet) if 32 <= pm.mlet <= 126 else "?"
        return MonsterInfo(
            mon_id=mid,
            name=pm.mname,
            symbol=mlet,
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
        )
    except Exception:
        return None


def _make_obj_info(oid: int) -> ObjectInfo | None:
    if not (0 <= oid < nh.NUM_OBJECTS):
        return None
    try:
        oc = nh.objclass(oid)
        return ObjectInfo(
            obj_id=oid,
            oclass=int(oc.oc_class),
            weight=int(oc.oc_weight),
            cost=int(oc.oc_cost),
            delay=int(oc.oc_delay),
            color=int(oc.oc_color),
            prob=int(oc.oc_prob),
        )
    except Exception:
        return None


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
    elif nh.glyph_is_swallow(_g):
        _IS_SWALLOW_LUT[_g] = True
    elif nh.glyph_is_warning(_g):
        _IS_WARNING_LUT[_g] = True


# Generic Numba-accelerated grid math utilities (pure algorithms, zero game rules)
@numba.njit(cache=True)
def bfs_distance_grid(
    passable: np.ndarray, sy: int, sx: int, allow_diagonal: bool = True
) -> np.ndarray:
    """
    Computes shortest-path step distances from (sy, sx) across any boolean `passable` (21, 79) mask.
    Unreachable tiles have distance -1.
    """
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if not (0 <= sy < h and 0 <= sx < w):
        return dist
    dist[sy, sx] = 0
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    qy[tail] = sy
    qx[tail] = sx
    tail += 1

    if allow_diagonal:
        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        n_dirs = 8
    else:
        dys = (-1, 1, 0, 0, 0, 0, 0, 0)
        dxs = (0, 0, -1, 1, 0, 0, 0, 0)
        n_dirs = 4

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        head += 1
        cd = dist[cy, cx] + 1
        for i in range(n_dirs):
            ny = cy + dys[i]
            nx = cx + dxs[i]
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1 and passable[ny, nx]:
                dist[ny, nx] = cd
                qy[tail] = ny
                qx[tail] = nx
                tail += 1
    return dist


@numba.njit(cache=True)
def astar_next_step(
    passable: np.ndarray,
    sy: int,
    sx: int,
    ty: int,
    tx: int,
    allow_diagonal: bool = True,
) -> tuple[int, int]:
    """
    Returns the first step delta `(dy, dx)` along the shortest path from `(sy, sx)` to `(ty, tx)`
    on `passable`. Returns `(0, 0)` if already at target or unreachable.
    """
    if sy == ty and sx == tx:
        return (0, 0)
    h, w = passable.shape
    if not (0 <= sy < h and 0 <= sx < w and 0 <= ty < h and 0 <= tx < w):
        return (0, 0)

    # Run BFS backwards from target (ty, tx) so first step from (sy, sx) is O(1) neighbor lookup
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[ty, tx] = 0
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    qy[tail] = ty
    qx[tail] = tx
    tail += 1

    if allow_diagonal:
        dys = (-1, 1, 0, 0, -1, -1, 1, 1)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1)
        n_dirs = 8
    else:
        dys = (-1, 1, 0, 0, 0, 0, 0, 0)
        dxs = (0, 0, -1, 1, 0, 0, 0, 0)
        n_dirs = 4

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        head += 1
        if cy == sy and cx == sx:
            break
        cd = dist[cy, cx] + 1
        for i in range(n_dirs):
            ny = cy + dys[i]
            nx = cx + dxs[i]
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1:
                if passable[ny, nx] or (ny == sy and nx == sx):
                    dist[ny, nx] = cd
                    qy[tail] = ny
                    qx[tail] = nx
                    tail += 1

    best_d = 999999
    best_dy = 0
    best_dx = 0
    for i in range(n_dirs):
        ny = sy + dys[i]
        nx = sx + dxs[i]
        if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] != -1:
            if dist[ny, nx] < best_d:
                best_d = dist[ny, nx]
                best_dy = dys[i]
                best_dx = dxs[i]
    return (best_dy, best_dx)


class GameState:
    """
    Zero-copy, lazy game state view passed to the Policy on every step.
    Wraps raw NLE observation arrays and provides on-demand queries for
    blstats, screen grids, inventory, messages, and NLE C symbol tables.
    """

    __slots__ = (
        "raw",
        "blstats",
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
        "step_count",
        "_message_cache",
        "_inventory_cache",
        "_monsters_cache",
        "_objects_cache",
    )

    def __init__(self, raw_obs: dict[str, np.ndarray], step_count: int = 0):
        self.raw = raw_obs
        self.blstats: np.ndarray = raw_obs["blstats"]
        self.glyphs: np.ndarray = raw_obs["glyphs"]
        self.chars: np.ndarray = raw_obs["chars"]
        self.colors: np.ndarray = raw_obs["colors"]
        self.specials: np.ndarray = raw_obs["specials"]
        self.raw_message: np.ndarray = raw_obs["message"]
        self.inv_glyphs: np.ndarray = raw_obs["inv_glyphs"]
        self.inv_strs: np.ndarray = raw_obs["inv_strs"]
        self.inv_letters: np.ndarray = raw_obs["inv_letters"]
        self.inv_oclasses: np.ndarray = raw_obs["inv_oclasses"]
        self.program_state: np.ndarray | None = raw_obs.get("program_state")
        self.step_count: int = step_count

        self._message_cache: str | None = None
        self._inventory_cache: list[InventoryItem] | None = None
        self._monsters_cache: list[VisibleMonster] | None = None
        self._objects_cache: list[VisibleObject] | None = None

    # --- 1-to-1 NLE blstats properties (O(1) array indexing) ---
    @property
    def x(self) -> int:
        return int(self.blstats[0])

    @property
    def y(self) -> int:
        return int(self.blstats[1])

    @property
    def pos(self) -> tuple[int, int]:
        return (int(self.blstats[1]), int(self.blstats[0]))

    @property
    def strength_pct(self) -> int:
        return int(self.blstats[2])

    @property
    def strength(self) -> int:
        return int(self.blstats[3])

    @property
    def dex(self) -> int:
        return int(self.blstats[4])

    @property
    def con(self) -> int:
        return int(self.blstats[5])

    @property
    def intel(self) -> int:
        return int(self.blstats[6])

    @property
    def wis(self) -> int:
        return int(self.blstats[7])

    @property
    def cha(self) -> int:
        return int(self.blstats[8])

    @property
    def score(self) -> int:
        return int(self.blstats[9])

    @property
    def hp(self) -> int:
        return int(self.blstats[10])

    @property
    def max_hp(self) -> int:
        return int(self.blstats[11])

    @property
    def depth(self) -> int:
        return int(self.blstats[12])

    @property
    def gold(self) -> int:
        return int(self.blstats[13])

    @property
    def energy(self) -> int:
        return int(self.blstats[14])

    @property
    def max_energy(self) -> int:
        return int(self.blstats[15])

    @property
    def ac(self) -> int:
        return int(self.blstats[16])

    @property
    def monster_level(self) -> int:
        return int(self.blstats[17])

    @property
    def xp_level(self) -> int:
        return int(self.blstats[18])

    @property
    def xp(self) -> int:
        return int(self.blstats[19])

    @property
    def turn(self) -> int:
        return int(self.blstats[20])

    @property
    def hunger_state(self) -> int:
        """0=Satiated, 1=Not Hungry, 2=Hungry, 3=Weak, 4=Fainting, 5=Fainted, 6=Starved."""
        return int(self.blstats[21])

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
        return int(self.blstats[22])

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
        return int(self.blstats[23])

    @property
    def level_num(self) -> int:
        return int(self.blstats[24])

    @property
    def condition_mask(self) -> int:
        return int(self.blstats[25]) if len(self.blstats) > 25 else 0

    @property
    def align(self) -> int:
        """-1=Chaotic, 0=Neutral, 1=Lawful."""
        return int(self.blstats[26]) if len(self.blstats) > 26 else 0

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
            raw_bytes = bytes(self.raw_message).split(b"\x00", 1)[0]
            self._message_cache = raw_bytes.decode("latin1", errors="ignore").strip()
        return self._message_cache

    @property
    def inventory(self) -> list[InventoryItem]:
        if self._inventory_cache is None:
            items: list[InventoryItem] = []
            for let_b, str_arr, g_val, oclass_b in zip(
                self.inv_letters, self.inv_strs, self.inv_glyphs, self.inv_oclasses
            ):
                if let_b == 0:
                    continue
                letter = chr(int(let_b))
                desc = (
                    bytes(str_arr)
                    .split(b"\x00", 1)[0]
                    .decode("latin1", errors="ignore")
                    .strip()
                )
                g = int(g_val)
                oc = int(oclass_b)
                ch = chr(oc) if 32 <= oc <= 126 else "?"
                qty = 1
                first_word = desc.split(" ", 1)[0] if desc else ""
                if first_word.isdigit():
                    qty = int(first_word)
                items.append(
                    InventoryItem(
                        letter=letter,
                        name=desc,
                        glyph=g,
                        oclass=oc,
                        char=ch,
                        quantity=qty,
                    )
                )
            self._inventory_cache = _CallableList(items)
        return self._inventory_cache

    # --- Screen & NLE C Table Queries ---
    def char_at(self, y: int, x: int) -> str:
        if 0 <= y < 21 and 0 <= x < 79:
            c = int(self.chars[y, x])
            return chr(c) if 32 <= c <= 126 else " "
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
    def object_info(glyph: int) -> ObjectInfo | None:
        if 0 <= glyph < _MAX_GLYPH:
            return _OBJ_INFO_BY_GLYPH[glyph]
        return None

    def visible_monsters(self, exclude_self: bool = True) -> list[VisibleMonster]:
        """Returns all monsters currently visible on the 21x79 screen."""
        if self._monsters_cache is None:
            mons: list[VisibleMonster] = []
            g_grid = np.clip(self.glyphs, 0, _MAX_GLYPH - 1)
            mon_mask = _IS_MONSTER_LUT[g_grid]
            ys, xs = np.nonzero(mon_mask)
            hy, hx = self.y, self.x
            for y_val, x_val in zip(ys, xs):
                yi, xi = int(y_val), int(x_val)
                if exclude_self and yi == hy and xi == hx:
                    continue
                g = int(self.glyphs[yi, xi])
                c = int(self.chars[yi, xi])
                ch = chr(c) if 32 <= c <= 126 else "?"
                col = int(self.colors[yi, xi])
                spec = int(self.specials[yi, xi])
                is_pet = bool(_IS_PET_LUT[g]) or bool(spec & nh.MG_PET)
                is_peaceful = bool(spec & getattr(nh, "MG_PEACEFUL", 0))
                mons.append(
                    VisibleMonster(
                        y=yi,
                        x=xi,
                        glyph=g,
                        char=ch,
                        color=col,
                        is_pet=is_pet,
                        is_peaceful=is_peaceful,
                        info=_MON_INFO_BY_GLYPH[g],
                    )
                )
            self._monsters_cache = mons
        return self._monsters_cache

    def visible_objects(self) -> list[VisibleObject]:
        """Returns all objects and corpses currently visible on the 21x79 screen."""
        if self._objects_cache is None:
            objs: list[VisibleObject] = []
            g_grid = np.clip(self.glyphs, 0, _MAX_GLYPH - 1)
            obj_mask = _IS_OBJECT_LUT[g_grid]
            ys, xs = np.nonzero(obj_mask)
            for y_val, x_val in zip(ys, xs):
                yi, xi = int(y_val), int(x_val)
                g = int(self.glyphs[yi, xi])
                c = int(self.chars[yi, xi])
                ch = chr(c) if 32 <= c <= 126 else "?"
                col = int(self.colors[yi, xi])
                is_corpse = bool(_IS_BODY_LUT[g])
                objs.append(
                    VisibleObject(
                        y=yi,
                        x=xi,
                        glyph=g,
                        char=ch,
                        color=col,
                        is_corpse=is_corpse,
                        corpse_monster=_CORPSE_MON_BY_GLYPH[g] if is_corpse else None,
                        info=_OBJ_INFO_BY_GLYPH[g],
                    )
                )
            self._objects_cache = objs
        return self._objects_cache

    def ascii_map(self) -> str:
        """Renders the current 21x79 `chars` grid as a 21-line ASCII string."""
        lines = []
        for r in range(21):
            row_bytes = self.chars[r]
            line = "".join(
                chr(int(b)) if 32 <= int(b) <= 126 else " " for b in row_bytes
            )
            lines.append(line.rstrip() or " ")
        return "\n".join(lines)

    def bfs_distances(
        self,
        passable_mask: np.ndarray,
        start: tuple[int, int] | None = None,
        allow_diagonal: bool = True,
    ) -> np.ndarray:
        """Runs Numba-accelerated BFS distance calculation from `start` (defaults to `(self.y, self.x)`)."""
        sy, sx = start if start is not None else (self.y, self.x)
        return bfs_distance_grid(
            np.asarray(passable_mask, dtype=np.bool_),
            int(sy),
            int(sx),
            bool(allow_diagonal),
        )

    def path_step(
        self,
        passable_mask: np.ndarray,
        target: tuple[int, int],
        start: tuple[int, int] | None = None,
        allow_diagonal: bool = True,
    ) -> tuple[int, int]:
        """Returns the next `(dy, dx)` step from `start` toward `target` across `passable_mask`."""
        sy, sx = start if start is not None else (self.y, self.x)
        return astar_next_step(
            np.asarray(passable_mask, dtype=np.bool_),
            int(sy),
            int(sx),
            int(target[0]),
            int(target[1]),
            bool(allow_diagonal),
        )
