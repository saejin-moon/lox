"""
Zero-Heuristic Deterministic Spatial & Episodic Memory (`FloorMemory`, `WorldMemory`).
Maintained by `NetHackHarness` across steps and replays.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from lox.env.types import (
    _FLOOR_GLYPH_FLAGS_LUT,
    _HAS_RUST_FLOOR_MEM,
    _IS_CLOSED_DOOR_GLYPH_LUT,
    _IS_CMAP_LUT,
    _IS_DOORWAY_GLYPH_LUT,
    _IS_EXIT_CMAP_LUT,
    _IS_HAZARD_CMAP_LUT,
    _IS_MONSTER_LUT,
    _IS_OBJECT_LUT,
    _IS_PASSABLE_CMAP_LUT,
    _IS_TRAP_LUT,
    _MAX_GLYPH,
    lox_core,
)

if TYPE_CHECKING:
    from lox.env.state import GameState

# Matches NetHack inventory-pickup confirmations like "e - a morning star." or "a - 3 food rations."
_RE_PICKUP_MSG = re.compile(r"(?:^|\|\s*)[a-zA-Z]\s*-\s+")


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
    remembered_glyphs: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.int16)
    )
    remembered_colors: np.ndarray = field(
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
    remembered_trap_glyphs: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.int32)
    )
    _passable_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.bool_), repr=False
    )
    _safe_passable_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 79), dtype=np.bool_), repr=False
    )
    _passable_dirty: bool = True
    locked_doors: set[tuple[int, int]] = field(default_factory=set)

    def _recompute_passability(self) -> None:
        c = self.remembered_chars
        rg = self.remembered_glyphs
        char_pass = (
            (c == ord("."))
            | (c == ord("#"))
            | (c == ord(">"))
            | (c == ord("<"))
            | (c == ord("_"))
            | (c == ord("{"))
            | (c == ord("\\"))
            | (c == ord("^"))
        )
        if rg.shape == c.shape:
            # Align strictly with Rust `fast_update_floor_memory`:
            # treat unseen stone glyph 2359 (`S_stone`) as unset when a passable char is known.
            has_rg = (rg > 0) & (rg != 2359)
            valid_rg = np.where(has_rg & (rg < _MAX_GLYPH), rg, 0)
            glyph_pass = _IS_PASSABLE_CMAP_LUT[valid_rg]
            glyph_haz = _IS_HAZARD_CMAP_LUT[valid_rg]
            self._passable_mask = np.where(has_rg, glyph_pass, char_pass)
            self._safe_passable_mask = (
                self._passable_mask
                & (~(has_rg & glyph_haz))
                & (self.remembered_trap_glyphs == 0)
            )
        else:
            self._passable_mask = char_pass
            self._safe_passable_mask = char_pass & (self.remembered_trap_glyphs == 0)
        self._passable_dirty = False

    @property
    def passable_mask(self) -> np.ndarray:
        """Returns read-only view of known walkable tiles (use `copy_passable_mask()` to mutate)."""
        if (
            self._passable_dirty
            or self._passable_mask.shape != self.remembered_chars.shape
        ):
            self._recompute_passability()
        v = self._passable_mask.view()
        v.setflags(write=False)
        return v

    def copy_passable_mask(self) -> np.ndarray:
        """Returns a mutable copy of `passable_mask`."""
        return self.passable_mask.copy()

    @property
    def safe_passable_mask(self) -> np.ndarray:
        """Returns read-only view of known walkable tiles excluding remembered traps and terrain hazards."""
        if (
            self._passable_dirty
            or self._safe_passable_mask.shape != self.remembered_chars.shape
        ):
            self._recompute_passability()
        v = self._safe_passable_mask.view()
        v.setflags(write=False)
        return v

    def copy_safe_passable_mask(self) -> np.ndarray:
        """Returns a mutable copy of `safe_passable_mask`."""
        return self.safe_passable_mask.copy()

    def ensure_shape(self, min_h: int, min_w: int) -> None:
        """Dynamically resizes spatial grids if coordinates exceed initial bounds."""
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

            new_rem_glyphs = np.zeros((new_h, new_w), dtype=np.int16)
            rem_gh, rem_gw = self.remembered_glyphs.shape
            new_rem_glyphs[: min(rem_gh, new_h), : min(rem_gw, new_w)] = (
                self.remembered_glyphs[: min(rem_gh, new_h), : min(rem_gw, new_w)]
            )
            self.remembered_glyphs = new_rem_glyphs

            new_rem_colors = np.zeros((new_h, new_w), dtype=np.uint8)
            rem_ch, rem_cw = self.remembered_colors.shape
            new_rem_colors[: min(rem_ch, new_h), : min(rem_cw, new_w)] = (
                self.remembered_colors[: min(rem_ch, new_h), : min(rem_cw, new_w)]
            )
            self.remembered_colors = new_rem_colors

            new_searched = np.zeros((new_h, new_w), dtype=np.int16)
            new_searched[:cur_h, :cur_w] = self.searched_counts
            self.searched_counts = new_searched

            new_door_mask = np.zeros((new_h, new_w), dtype=np.bool_)
            new_door_mask[:cur_h, :cur_w] = self.door_mask
            self.door_mask = new_door_mask

            new_obj_glyphs = np.zeros((new_h, new_w), dtype=np.int32)
            new_obj_glyphs[:cur_h, :cur_w] = self.remembered_obj_glyphs
            self.remembered_obj_glyphs = new_obj_glyphs

            new_trap_glyphs = np.zeros((new_h, new_w), dtype=np.int32)
            new_trap_glyphs[:cur_h, :cur_w] = self.remembered_trap_glyphs
            self.remembered_trap_glyphs = new_trap_glyphs

            self._passable_mask = np.zeros((new_h, new_w), dtype=np.bool_)
            self._safe_passable_mask = np.zeros((new_h, new_w), dtype=np.bool_)
            self._passable_dirty = True


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
        "_cached_floor_key",
        "_cached_floor",
        "stair_graph",
        "level_exits",
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
        self._cached_floor_key: tuple[int, int] | None = None
        self._cached_floor: FloorMemory | None = None
        self.stair_graph: dict[
            tuple[int, int, int, int], tuple[int, int, int, int]
        ] = {}
        self.level_exits: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
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
        self._cached_floor_key = None
        self._cached_floor = None
        self.stair_graph.clear()
        self.level_exits.clear()
        self.kv.clear()
        self.recent_messages.clear()
        self.prev_turn = -1
        self.consecutive_stalls = 0
        self.last_action_zero_turn = False

    def get_floor(
        self, dnum: int, dlvl: int, shape: tuple[int, int] = (21, 79)
    ) -> FloorMemory:
        key = (int(dnum), int(dlvl))
        floor = self.floors.get(key)
        if floor is None:
            floor = FloorMemory(
                visited=np.zeros(shape, dtype=np.bool_),
                visit_count=np.zeros(shape, dtype=np.int32),
                remembered_chars=np.zeros(shape, dtype=np.uint8),
                remembered_glyphs=np.zeros(shape, dtype=np.int16),
                remembered_colors=np.zeros(shape, dtype=np.uint8),
                searched_counts=np.zeros(shape, dtype=np.int16),
                door_mask=np.zeros(shape, dtype=np.bool_),
                remembered_obj_glyphs=np.zeros(shape, dtype=np.int32),
                remembered_trap_glyphs=np.zeros(shape, dtype=np.int32),
            )
            self.floors[key] = floor
        return floor

    @property
    def current_floor(self) -> FloorMemory:
        cfk = self.current_floor_key
        cf = self._cached_floor
        if cf is not None and self._cached_floor_key == cfk:
            return cf
        cf = self.get_floor(cfk[0], cfk[1])
        self._cached_floor_key = cfk
        self._cached_floor = cf
        return cf

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
    def remembered_glyphs(self) -> np.ndarray:
        return self.current_floor.remembered_glyphs

    @property
    def remembered_colors(self) -> np.ndarray:
        return self.current_floor.remembered_colors

    @property
    def door_mask(self) -> np.ndarray:
        return self.current_floor.door_mask

    @property
    def passable_mask(self) -> np.ndarray:
        return self.current_floor.passable_mask

    @property
    def safe_passable_mask(self) -> np.ndarray:
        return self.current_floor.safe_passable_mask

    def copy_passable_mask(self) -> np.ndarray:
        return self.current_floor.copy_passable_mask()

    def copy_safe_passable_mask(self) -> np.ndarray:
        return self.current_floor.copy_safe_passable_mask()

    @property
    def locked_doors(self) -> set[tuple[int, int]]:
        return self.current_floor.locked_doors

    @property
    def searched_counts(self) -> np.ndarray:
        return self.current_floor.searched_counts

    @property
    def remembered_obj_glyphs(self) -> np.ndarray:
        return self.current_floor.remembered_obj_glyphs

    @property
    def remembered_trap_glyphs(self) -> np.ndarray:
        return self.current_floor.remembered_trap_glyphs

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

    def update(
        self,
        state: GameState,
        action_name: str = "",
        messages: list[str] | None = None,
    ) -> None:
        try:
            dnum = int(state.dungeon_num)
            dlvl = int(state.level_num)
            sy = int(state.y)
            sx = int(state.x)
        except Exception:
            return

        # Track in-game turn advancement (zero-turn action awareness)
        try:
            curr_turn = int(state.turn)
            step_cnt = state.step_count
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
        else:
            st_map = getattr(state, "map", None)
            if st_map is not None and hasattr(st_map, "shape"):
                gh, gw = st_map.shape
            else:
                gh, gw = max(21, sy + 1), max(79, sx + 1)

        self.current_floor_key = (dnum, dlvl)
        floor = self.get_floor(dnum, dlvl, shape=(gh, gw))
        self._cached_floor_key = self.current_floor_key
        self._cached_floor = floor
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

        turn_now = int(getattr(state, "turn", getattr(state, "step_count", 0)) or 0)
        if messages:
            for m in messages:
                m_str = str(m or "").strip()
                if m_str and (
                    not self.recent_messages or self.recent_messages[-1][2] != m_str
                ):
                    self.recent_messages.append((turn_now, int(dlvl), m_str))
        elif hasattr(state, "message") and state.message:
            msg_str = str(state.message).strip()
            if msg_str:
                self.recent_messages.append((turn_now, int(dlvl), msg_str))

        # Update remembered terrain chars, glyphs, colors, doorway mask, objects, traps, and passability
        try:
            glyphs = state.glyphs
            chars = state.chars
            colors = getattr(state, "colors", None)
            if not isinstance(colors, np.ndarray) or colors.shape != glyphs.shape:
                colors = np.zeros(glyphs.shape, dtype=np.uint8)
            elif colors.dtype != np.uint8 or not colors.flags.c_contiguous:
                colors = np.ascontiguousarray(colors, dtype=np.uint8)
            on_down = bool(getattr(state, "is_on_stairs_down", False))
            if (
                _HAS_RUST_FLOOR_MEM
                and isinstance(glyphs, np.ndarray)
                and isinstance(chars, np.ndarray)
                and glyphs.dtype == np.int16
                and chars.dtype == np.uint8
                and glyphs.flags.c_contiguous
                and chars.flags.c_contiguous
            ):
                if floor._passable_mask.shape != floor.remembered_chars.shape:
                    floor._passable_mask = np.zeros_like(
                        floor.remembered_chars, dtype=np.bool_
                    )
                if floor._safe_passable_mask.shape != floor.remembered_chars.shape:
                    floor._safe_passable_mask = np.zeros_like(
                        floor.remembered_chars, dtype=np.bool_
                    )
                lox_core.fast_update_floor_memory(
                    glyphs,
                    chars,
                    colors,
                    _FLOOR_GLYPH_FLAGS_LUT,
                    floor.remembered_chars,
                    floor.remembered_glyphs,
                    floor.remembered_colors,
                    floor.door_mask,
                    floor.remembered_obj_glyphs,
                    floor.remembered_trap_glyphs,
                    floor._passable_mask,
                    floor._safe_passable_mask,
                    int(sy),
                    int(sx),
                    on_down,
                )
                floor._passable_dirty = False
            else:
                is_cmap = _IS_CMAP_LUT[glyphs]

                # 1. Update lossless remembered_chars, remembered_glyphs, and remembered_colors for cmap terrain
                mask = is_cmap & (chars != 0) & (chars != ord(" "))
                floor.remembered_chars[mask] = chars[mask]
                if floor.remembered_glyphs.shape == glyphs.shape:
                    floor.remembered_glyphs[is_cmap] = glyphs[is_cmap]
                    floor.remembered_colors[is_cmap] = colors[is_cmap]

                # 2. Update door_mask
                doorway_mask = is_cmap & _IS_DOORWAY_GLYPH_LUT[glyphs]
                floor.door_mask[doorway_mask] = True
                floor.door_mask[is_cmap & (~_IS_DOORWAY_GLYPH_LUT[glyphs])] = False

                # 3. For tiles with visible objects, monsters, or hero where remembered_chars == 0 or '+', mark as floor '.'
                #    and clear stale closed-door or unseen stone glyphs so passability stays consistent with Rust.
                h_curr, w_curr = floor.remembered_chars.shape
                non_cmap_floor = _IS_OBJECT_LUT[glyphs] | _IS_MONSTER_LUT[glyphs]
                uninit_or_closed = non_cmap_floor & (
                    (floor.remembered_chars == 0) | (floor.remembered_chars == ord("+"))
                )
                floor.remembered_chars[uninit_or_closed] = ord(".")
                if floor.remembered_glyphs.shape == glyphs.shape:
                    rg = floor.remembered_glyphs
                    valid_rg = np.where((rg > 0) & (rg < _MAX_GLYPH), rg, 0)
                    stale_rg = non_cmap_floor & (
                        (rg == 2359) | _IS_CLOSED_DOOR_GLYPH_LUT[valid_rg]
                    )
                    rg[stale_rg] = 0
                if 0 <= sy < h_curr and 0 <= sx < w_curr:
                    if floor.remembered_chars[sy, sx] in (0, ord("+")):
                        floor.remembered_chars[sy, sx] = ord(".")
                    if floor.remembered_glyphs.shape == glyphs.shape:
                        cur_rg = int(floor.remembered_glyphs[sy, sx])
                        if cur_rg == 2359 or (
                            0 <= cur_rg < _MAX_GLYPH
                            and bool(_IS_CLOSED_DOOR_GLYPH_LUT[cur_rg])
                        ):
                            floor.remembered_glyphs[sy, sx] = 0
                    if on_down:
                        floor.remembered_chars[sy, sx] = ord(">")

                # 4. Update remembered_obj_glyphs & remembered_trap_glyphs
                floor.remembered_obj_glyphs[is_cmap] = 0
                obj_mask = _IS_OBJECT_LUT[glyphs]
                floor.remembered_obj_glyphs[obj_mask] = glyphs[obj_mask]

                floor.remembered_trap_glyphs[is_cmap & (~_IS_TRAP_LUT[glyphs])] = 0
                trap_mask = _IS_TRAP_LUT[glyphs]
                floor.remembered_trap_glyphs[trap_mask] = glyphs[trap_mask]
                floor._passable_dirty = True

            # Record remembered exits (stairs/ladders/portals) for unvisited branch graph
            if isinstance(glyphs, np.ndarray) and glyphs.ndim == 2:
                if (
                    _HAS_RUST_FLOOR_MEM
                    and glyphs.dtype == np.int16
                    and glyphs.flags.c_contiguous
                ):
                    exits_list = lox_core.fast_extract_exits(glyphs, _IS_EXIT_CMAP_LUT)
                    if exits_list:
                        f_exits = self.level_exits.setdefault((dnum, dlvl), {})
                        for ey, ex, eg in exits_list:
                            f_exits[(ey, ex)] = int(eg)
                else:
                    exit_ys, exit_xs = np.where(_IS_EXIT_CMAP_LUT[glyphs])
                    if len(exit_ys) > 0:
                        f_exits = self.level_exits.setdefault((dnum, dlvl), {})
                        for ey, ex in zip(exit_ys.tolist(), exit_xs.tolist()):
                            f_exits[(ey, ex)] = int(glyphs[ey, ex])
                if on_down and 0 <= sy < glyphs.shape[0] and 0 <= sx < glyphs.shape[1]:
                    f_exits = self.level_exits.setdefault((dnum, dlvl), {})
                    f_exits.setdefault((sy, sx), int(glyphs[sy, sx]))
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

        # Clear under-hero remembered object glyph when NLE reports tile cleared, eaten, or picked up
        try:
            msg_candidates = list(messages or ()) or (
                [state.message] if getattr(state, "message", "") else []
            )
            if (
                0 <= sy < floor.remembered_obj_glyphs.shape[0]
                and 0 <= sx < floor.remembered_obj_glyphs.shape[1]
                and floor.remembered_obj_glyphs[sy, sx] > 0
                and msg_candidates
            ):
                msg_raw = " | ".join(str(m or "") for m in msg_candidates)
                msg_join = msg_raw.lower()
                has_remaining_here = any(
                    p in msg_join
                    for p in ("you see here", "objects here:", "things that are here:")
                )
                cleared_msg = (
                    any(
                        p in msg_join
                        for p in (
                            "you see no objects here",
                            "there is nothing here",
                            "you finish eating",
                            "you stop eating",
                            "you pick up",
                            "gold piece",
                        )
                    )
                    or bool(_RE_PICKUP_MSG.search(msg_raw))
                    or (
                        action_name.upper() == "PICKUP"
                        and "can't carry" not in msg_join
                        and "cannot carry" not in msg_join
                        and "too heavy" not in msg_join
                    )
                )
                if cleared_msg and not has_remaining_here:
                    floor.remembered_obj_glyphs[sy, sx] = 0
        except Exception:
            pass
