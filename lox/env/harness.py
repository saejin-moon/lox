"""
Ultra-Fast 1-to-1 NetHack Harness ("Hard Code Nothing").
Connects the policy directly to NLE's C engine (`nle.nethack.Nethack`)
without gym wrapper bloat or hardcoded heuristics.
Supports 275,000 steps/sec deterministic C-level replay from recorded action bytes.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from nle import nethack as nh

from lox.env.actions import normalize_action_to_bytes
from lox.env.state import (
    _FLOOR_GLYPH_FLAGS_LUT,
    _HAS_RUST_FLOOR_MEM,
    _IS_CMAP_LUT,
    _IS_DOORWAY_GLYPH_LUT,
    _IS_OBJECT_LUT,
    _IS_TRAP_LUT,
    GameState,
    WorldMemory,
    astar_next_step,
    bfs_distance_grid,
    lox_core,
)

DEFAULT_NETHACKOPTIONS: tuple[str, ...] = tuple(
    ["noautopickup"]
    + [
        o
        for o in nh.NETHACKOPTIONS
        if o != "autopickup"
        and o != "noautopickup"
        and not str(o).startswith(("pickup_types:", "pickup_burden:"))
    ]
)

if not _HAS_RUST_FLOOR_MEM:
    # Pre-warm Numba JIT kernels once at module import only when Rust fallback is active
    _DUMMY_MASK = np.ones((4, 4), dtype=np.bool_)
    bfs_distance_grid(_DUMMY_MASK, 0, 0, True)
    astar_next_step(_DUMMY_MASK, 0, 0, 2, 2, True)

ALL_OBSERVATION_KEYS: tuple[str, ...] = (
    "glyphs",
    "chars",
    "colors",
    "specials",
    "blstats",
    "message",
    "inv_glyphs",
    "inv_strs",
    "inv_letters",
    "inv_oclasses",
    "program_state",
    "internal",
    "tty_chars",
    "tty_colors",
    "tty_cursor",
    "screen_descriptions",
    "misc",
)
LEAN_OBSERVATION_KEYS: tuple[str, ...] = (
    "glyphs",
    "chars",
    "colors",
    "specials",
    "blstats",
    "message",
    "inv_glyphs",
    "inv_strs",
    "inv_letters",
    "inv_oclasses",
    "program_state",
    "internal",
    "tty_chars",
    "tty_colors",
    "tty_cursor",
)

ASCII_SPACE = ord(" ")
ASCII_ESC = 27
ASCII_CR = ord("\r")
ASCII_N = ord("n")


def policy_handles_menus(
    policy_code: str = "", skills: dict[str, str] | None = None
) -> bool:
    """True when policy or any skill module explicitly references `in_menu`."""
    if "in_menu" in (policy_code or ""):
        return True
    if skills:
        return any("in_menu" in str(src or "") for src in skills.values())
    return False


class NetHackHarness:
    """
    Direct, high-throughput NLE harness.
    Maps policy actions 1-to-1 to C `nethack.step(byte)` calls and records
    exact action bytes for deterministic 14ms replay.
    """

    def __init__(
        self,
        role: str = "val-hum-law-fem",
        max_steps: int = 25000,
        options: tuple[str, ...] | None = None,
        auto_dismiss_more: bool = True,
        lean_obs: bool = True,
        max_zero_turn_stalls: int = 100,
        allow_menu_linger: bool = True,
    ):
        self.role = role
        self.max_steps = max_steps
        self.auto_dismiss_more = auto_dismiss_more
        self.allow_menu_linger = allow_menu_linger
        self.lean_obs = lean_obs
        self.max_zero_turn_stalls = max_zero_turn_stalls
        self.options = options if options is not None else DEFAULT_NETHACKOPTIONS
        self._obs_keys = list(
            LEAN_OBSERVATION_KEYS if lean_obs else ALL_OBSERVATION_KEYS
        )
        self._idx_map = {k: i for i, k in enumerate(self._obs_keys)}
        self._prog_state_idx = self._idx_map["program_state"]
        self._msg_idx = self._idx_map["message"]
        self._internal_idx = self._idx_map["internal"]

        self.nh = nh.Nethack(
            observation_keys=self._obs_keys,
            options=self.options,
            playername=f"Agent-{self.role}",
            ttyrec=None,
            wizard=False,
            spawn_monsters=True,
            scoreprefix=None,
            fix_moon_phase=True,
        )

        self.step_count: int = 0
        self.current_seed: int = 0
        self.action_bytes = bytearray()
        self.step_byte_offsets: list[int] = []
        self.prompt_aborts: int = 0
        self.prompt_linger_steps: int = 0
        self.max_prompt_linger: int = 3
        self.consecutive_zero_turn_steps: int = 0
        self._prev_turn: int = -1
        self._last_raw_tuple: tuple[np.ndarray, ...] = ()
        self.done: bool = False
        self.mem: WorldMemory = WorldMemory()

    @property
    def steps_taken(self) -> int:
        return self.step_count

    def _tuple_to_dict(
        self, obs_tuple: tuple[np.ndarray, ...]
    ) -> dict[str, np.ndarray]:
        return {k: obs_tuple[i] for k, i in self._idx_map.items()}

    def _in_moveloop(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        return bool(obs_tuple[self._prog_state_idx][3])

    def _is_game_over(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        return bool(obs_tuple[self._prog_state_idx][0] == 1)

    def _extract_raw_message(self, obs_tuple: tuple[np.ndarray, ...]) -> str:
        return (
            bytes(obs_tuple[self._msg_idx])
            .split(b"\x00", 1)[0]
            .decode("latin1", errors="ignore")
            .strip()
        )

    def _waiting_for_space(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        return bool(obs_tuple[self._internal_idx][3] == 1)

    def _in_yn(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        return bool(obs_tuple[self._internal_idx][1] == 1)

    def _in_getlin(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        return bool(obs_tuple[self._internal_idx][2] == 1)

    def _has_more_banner(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        """True when row 0 of `tty_chars` currently shows `--More--`."""
        tty_idx = self._idx_map.get("tty_chars")
        if tty_idx is None or tty_idx >= len(obs_tuple):
            return False
        tty_chars = obs_tuple[tty_idx]
        if tty_chars is None or len(tty_chars) == 0:
            return False
        row0 = bytes(tty_chars[0]).decode("latin1", errors="ignore").lower()
        return "--more--" in row0

    def _in_menu(self, obs_tuple: tuple[np.ndarray, ...]) -> bool:
        """True when NetHack pauses for an interactive menu/pager rather than a plain `--More--` prompt."""
        if not self._waiting_for_space(obs_tuple):
            return False
        tty_idx = self._idx_map.get("tty_chars")
        if tty_idx is None or tty_idx >= len(obs_tuple):
            return False
        tty_chars = obs_tuple[tty_idx]
        if tty_chars is None or len(tty_chars) == 0:
            return False
        row0 = bytes(tty_chars[0]).decode("latin1", errors="ignore").lower()
        if "--more--" in row0:
            return False
        if any(k in row0 for k in ("pick up what", "what would you like", "for what")):
            return True
        joined = "\n".join(
            bytes(tty_chars[r]).decode("latin1", errors="ignore")
            for r in range(min(22, len(tty_chars)))
        ).lower()
        return "(end)" in joined or bool(re.search(r"\(\d+\s+of\s+\d+\)", joined))

    def _settle_prompts(
        self,
        obs_tuple: tuple[np.ndarray, ...],
        done: bool,
        messages: list[str] | None = None,
    ) -> tuple[tuple[np.ndarray, ...], bool]:
        """
        Matches standard NLE prompt completion when NetHack pauses on `--More--`
        (xwaitforspace), outside `in_moveloop`, or in lingering `yn_function`/`getlin`/menu prompts.
        """
        if not self.auto_dismiss_more:
            return obs_tuple, done
        safety = 0
        while (
            not done
            and (
                self._waiting_for_space(obs_tuple)
                or not self._in_moveloop(obs_tuple)
                or self._in_yn(obs_tuple)
                or self._in_getlin(obs_tuple)
            )
            and safety < 64
        ):
            safety += 1
            menu_linger = self.allow_menu_linger and self._in_menu(obs_tuple)
            if self._is_game_over(obs_tuple):
                b = ASCII_SPACE
            elif self._waiting_for_space(obs_tuple) and self._has_more_banner(
                obs_tuple
            ):
                # Clear `--More--` before checking `in_yn` because NLE can keep
                # internal[1] (in_yn) set while waiting on a `--More--` pager.
                b = ASCII_SPACE
            elif self._in_yn(obs_tuple) or self._in_getlin(obs_tuple) or menu_linger:
                # Allow interactive prompt/menu to be returned to policy for resolution
                if self.prompt_linger_steps < self.max_prompt_linger:
                    self.prompt_linger_steps += 1
                    break
                raw_m_l = self._extract_raw_message(obs_tuple).lower()
                is_getobj_or_dir = self._in_yn(obs_tuple) and (
                    "[?*" in raw_m_l
                    or "?*]" in raw_m_l
                    or "what do you want to " in raw_m_l
                    or "for what" in raw_m_l
                    or "in what direction" in raw_m_l
                    or "direction?" in raw_m_l
                )
                b = (
                    ASCII_ESC
                    if (self._in_getlin(obs_tuple) or menu_linger or is_getobj_or_dir)
                    else ASCII_N
                )
                self.prompt_aborts += 1
                self.prompt_linger_steps = 0
            elif self._waiting_for_space(obs_tuple):
                b = ASCII_SPACE
            else:
                b = ASCII_SPACE
            self.action_bytes.append(b)
            try:
                obs_tuple, done = self.nh.step(b)
            except RuntimeError as err:
                if "Called step on finished NetHack" in str(err):
                    self.done = True
                    return obs_tuple, True
                raise
            if messages is not None:
                raw_m = self._extract_raw_message(obs_tuple)
                if raw_m and (not messages or messages[-1] != raw_m):
                    messages.append(raw_m)
        return obs_tuple, done

    def reset(self, seed: int = 42) -> GameState:
        """Resets the C NetHack engine with deterministic seed `(seed, seed, reseed=False, seed)`."""
        self.current_seed = int(seed)
        self.nh.set_initial_seeds(
            self.current_seed, self.current_seed, False, self.current_seed
        )
        self.action_bytes.clear()
        self.step_count = 0
        self.prompt_aborts = 0
        self.prompt_linger_steps = 0
        self.consecutive_zero_turn_steps = 0
        self._prev_turn = -1
        self.done = False
        self.mem.reset()

        obs_tuple = self.nh.reset(None, options=None)
        for _ in range(100):
            if self._in_moveloop(obs_tuple) and not self._waiting_for_space(obs_tuple):
                break
            self.action_bytes.append(ASCII_SPACE)
            try:
                obs_tuple, self.done = self.nh.step(ASCII_SPACE)
            except RuntimeError as err:
                if "Called step on finished NetHack" in str(err):
                    self.done = True
                    break
                raise
            if self.done:
                break

        self._last_raw_tuple = obs_tuple
        self.step_byte_offsets = [len(self.action_bytes)]
        return self.wrap_state(obs_tuple)

    def wrap_state(self, obs_tuple: tuple[np.ndarray, ...] | None = None) -> GameState:
        """Wraps a raw NLE C observation tuple into a zero-copy lazy GameState."""
        if obs_tuple is None:
            obs_tuple = self._last_raw_tuple
        self._last_raw_tuple = obs_tuple
        state = GameState(
            obs_tuple,
            step_count=self.step_count,
            mem=self.mem,
            obs_keys=self._obs_keys,
            obs_idx_map=self._idx_map,
        )
        self.mem.update(state)
        return state

    def _extract_turn(self, obs_tuple: tuple[np.ndarray, ...]) -> int:
        blstats_arr = (
            obs_tuple[self._idx_map["blstats"]]
            if "blstats" in self._idx_map
            else (obs_tuple[4] if len(obs_tuple) > 4 else None)
        )
        if blstats_arr is not None and len(blstats_arr) > nh.NLE_BL_TIME:
            return int(blstats_arr[nh.NLE_BL_TIME])
        return -1

    def step(self, policy_action: Any) -> tuple[GameState, bool, dict[str, Any]]:
        """
        Executes a 1-to-1 policy action (or multi-keystroke sequence) on the C NetHack instance.
        Returns `(next_state, done, info)`.
        """
        if self.done:
            return (
                GameState(
                    self._last_raw_tuple,
                    step_count=self.step_count,
                    obs_keys=self._obs_keys,
                    obs_idx_map=self._idx_map,
                ),
                True,
                {"action_name": "DONE", "is_ascended": False},
            )

        bytes_seq, action_name = normalize_action_to_bytes(policy_action)
        obs_tuple = self._last_raw_tuple
        done = False
        messages: list[str] = []
        seq_aborted = False
        is_extcmd = bool(bytes_seq and bytes_seq[0] == ord("#"))
        extcmd_cr_idx = (
            bytes_seq.index(ASCII_CR)
            if is_extcmd and ASCII_CR in bytes_seq
            else (
                bytes_seq.index(ord("\n"))
                if is_extcmd and ord("\n") in bytes_seq
                else -1
            )
        )
        is_digit_repeat = bool(
            bytes_seq and len(bytes_seq) > 1 and ord("0") <= bytes_seq[0] <= ord("9")
        )

        for idx, b in enumerate(bytes_seq):
            turn_before = self._extract_turn(obs_tuple)
            self.action_bytes.append(b)
            try:
                obs_tuple, done = self.nh.step(b)
            except RuntimeError as err:
                if "Called step on finished NetHack" in str(err):
                    self.done = True
                    done = True
                    break
                raise
            raw_m = self._extract_raw_message(obs_tuple)
            if raw_m and (not messages or messages[-1] != raw_m):
                messages.append(raw_m)
            if done:
                break
            if idx < len(bytes_seq) - 1:
                # Auto-dismiss transient `--More--` between sequence bytes
                more_guard = 0
                while (
                    not done
                    and self._waiting_for_space(obs_tuple)
                    and self._has_more_banner(obs_tuple)
                    and more_guard < 8
                ):
                    more_guard += 1
                    self.action_bytes.append(ASCII_SPACE)
                    obs_tuple, done = self.nh.step(ASCII_SPACE)
                    raw_m2 = self._extract_raw_message(obs_tuple)
                    if raw_m2 and (not messages or messages[-1] != raw_m2):
                        messages.append(raw_m2)
                if done:
                    break
                if self.allow_menu_linger and self._in_menu(obs_tuple):
                    seq_aborted = True
                    break
                # Stop remaining keys of a multi-byte command only when the preceding byte
                # returned cleanly to idle moveloop in 0 elapsed turns (rejected command/prompt).
                in_extcmd_header = is_extcmd and (
                    extcmd_cr_idx < 0 or idx < extcmd_cr_idx
                )
                in_repeat_header = is_digit_repeat and ord("0") <= b <= ord("9")
                turn_after = self._extract_turn(obs_tuple)
                if (
                    not in_extcmd_header
                    and not in_repeat_header
                    and turn_before >= 0
                    and turn_after == turn_before
                    and self._in_moveloop(obs_tuple)
                    and not self._in_yn(obs_tuple)
                    and not self._in_getlin(obs_tuple)
                    and not self._waiting_for_space(obs_tuple)
                ):
                    seq_aborted = True
                    break

        if not done and (
            self._waiting_for_space(obs_tuple)
            or not self._in_moveloop(obs_tuple)
            or self._in_yn(obs_tuple)
            or self._in_getlin(obs_tuple)
        ):
            obs_tuple, done = self._settle_prompts(obs_tuple, done, messages=messages)
            raw_m = self._extract_raw_message(obs_tuple)
            if raw_m and (not messages or messages[-1] != raw_m):
                messages.append(raw_m)

        if (
            self._in_moveloop(obs_tuple)
            and not self._in_yn(obs_tuple)
            and not self._in_getlin(obs_tuple)
            and not (self.allow_menu_linger and self._in_menu(obs_tuple))
        ):
            self.prompt_linger_steps = 0

        self.step_count += 1
        self.step_byte_offsets.append(len(self.action_bytes))

        # Check in-game turn advancement (zero-turn action awareness)
        stalled_zero_turn = False
        blstats_arr = (
            obs_tuple[self._idx_map["blstats"]]
            if "blstats" in self._idx_map
            else (obs_tuple[4] if len(obs_tuple) > 4 else None)
        )
        curr_turn = (
            int(blstats_arr[nh.NLE_BL_TIME])
            if blstats_arr is not None and len(blstats_arr) > nh.NLE_BL_TIME
            else self.step_count
        )
        if self._prev_turn >= 0 and curr_turn == self._prev_turn:
            self.consecutive_zero_turn_steps += 1
            if (
                self.consecutive_zero_turn_steps >= self.max_zero_turn_stalls
                and not done
            ):
                done = True
                stalled_zero_turn = True
        else:
            self.consecutive_zero_turn_steps = 0
        self._prev_turn = curr_turn

        truncated = self.step_count >= self.max_steps
        if truncated and not done:
            done = True

        self.done = bool(done)
        self._last_raw_tuple = obs_tuple
        state = GameState(
            obs_tuple,
            step_count=self.step_count,
            mem=self.mem,
            obs_keys=self._obs_keys,
            obs_idx_map=self._idx_map,
        )
        state._message_cache = " | ".join(messages) if messages else ""
        self.mem.update(state, action_name=action_name, messages=messages)

        is_ascended = False
        if self.done:
            try:
                is_ascended = bool(self.nh.how_done() == nh.ASCENDED)
            except Exception:
                is_ascended = False

        return (
            state,
            self.done,
            {
                "action_name": action_name,
                "truncated": truncated,
                "is_ascended": is_ascended,
                "stalled_zero_turn": stalled_zero_turn,
                "seq_aborted": seq_aborted,
            },
        )

    def replay_bytes(
        self,
        seed: int,
        recorded_bytes: bytes | bytearray | None = None,
        stop_at_byte_idx: int | None = None,
        action_bytes: bytes | bytearray | None = None,
    ) -> GameState:
        """
        Replays `recorded_bytes` directly in C at ~275,000 steps/sec (~14ms per 4,000 turns)
        to reconstruct the exact deterministic `GameState` at any point in a run.
        """
        buf = bytes(
            recorded_bytes if recorded_bytes is not None else (action_bytes or b"")
        )
        self.current_seed = int(seed)
        self.nh.set_initial_seeds(
            self.current_seed, self.current_seed, False, self.current_seed
        )
        self.action_bytes.clear()
        self.step_count = 0
        self.prompt_aborts = 0
        self.step_byte_offsets = [0]
        self.done = False
        self.mem.reset()

        obs_tuple = self.nh.reset(None, options=None)
        limit = (
            len(buf) if stop_at_byte_idx is None else min(len(buf), stop_at_byte_idx)
        )
        bl_idx = self._idx_map["blstats"]
        glyphs_idx = self._idx_map["glyphs"]
        chars_idx = self._idx_map["chars"]
        colors_idx = self._idx_map.get("colors")
        step_ctr = 0
        for i in range(limit):
            b = int(buf[i])
            self.action_bytes.append(b)
            try:
                obs_tuple, self.done = self.nh.step(b)
            except RuntimeError as err:
                if "Called step on finished NetHack" in str(err):
                    self.done = True
                    break
                raise
            if self.done:
                break
            bl = obs_tuple[bl_idx]
            dnum, dlvl, by, bx = int(bl[23]), int(bl[24]), int(bl[1]), int(bl[0])
            self.mem.pos_history.append((dnum, dlvl, by, bx))
            fl = self.mem.get_floor(dnum, dlvl)
            if 0 <= by < 21 and 0 <= bx < 79:
                fl.visited[by, bx] = True
                fl.visit_count[by, sx_clamped := bx] += 1
                if fl.remembered_chars[by, sx_clamped] == 0:
                    fl.remembered_chars[by, sx_clamped] = ord(".")
            if b == ord("s"):
                fl.searched_counts[
                    max(0, by - 1) : min(21, by + 2), max(0, bx - 1) : min(79, bx + 2)
                ] += 1
            floor_changed = False
            if len(self.mem.pos_history) >= 2:
                prev = self.mem.pos_history[-2]
                now = self.mem.pos_history[-1]
                if (prev[0], prev[1]) != (now[0], now[1]):
                    floor_changed = True
                    self.mem.stair_graph[prev] = now
                    self.mem.stair_graph[now] = prev

            in_moveloop = self._in_moveloop(obs_tuple)
            need_floor_mem = (
                i >= limit - 48 or floor_changed or (in_moveloop and (i & 31) == 0)
            )
            if need_floor_mem and (in_moveloop or floor_changed):
                glyphs = obs_tuple[glyphs_idx]
                chars = obs_tuple[chars_idx]
                colors = (
                    obs_tuple[colors_idx]
                    if colors_idx is not None and colors_idx < len(obs_tuple)
                    else np.zeros_like(chars, dtype=np.uint8)
                )
                if (
                    _HAS_RUST_FLOOR_MEM
                    and isinstance(glyphs, np.ndarray)
                    and isinstance(chars, np.ndarray)
                    and isinstance(colors, np.ndarray)
                    and glyphs.dtype == np.int16
                    and chars.dtype == np.uint8
                    and colors.dtype == np.uint8
                    and glyphs.flags.c_contiguous
                    and chars.flags.c_contiguous
                    and colors.flags.c_contiguous
                ):
                    if fl._passable_mask.shape != fl.remembered_chars.shape:
                        fl._passable_mask = np.zeros_like(
                            fl.remembered_chars, dtype=np.bool_
                        )
                    if fl._safe_passable_mask.shape != fl.remembered_chars.shape:
                        fl._safe_passable_mask = np.zeros_like(
                            fl.remembered_chars, dtype=np.bool_
                        )
                    lox_core.fast_update_floor_memory(
                        glyphs,
                        chars,
                        colors,
                        _FLOOR_GLYPH_FLAGS_LUT,
                        fl.remembered_chars,
                        fl.remembered_glyphs,
                        fl.remembered_colors,
                        fl.door_mask,
                        fl.remembered_obj_glyphs,
                        fl.remembered_trap_glyphs,
                        fl._passable_mask,
                        fl._safe_passable_mask,
                        int(by),
                        int(bx),
                        False,
                    )
                    fl._passable_dirty = False
                else:
                    is_cmap = _IS_CMAP_LUT[glyphs]
                    cmap_mask = is_cmap & (chars != 0) & (chars != ord(" "))
                    fl.remembered_chars[cmap_mask] = chars[cmap_mask]
                    fl.remembered_glyphs[is_cmap] = glyphs[is_cmap]
                    fl.remembered_colors[is_cmap] = colors[is_cmap]
                    doorway_mask = is_cmap & _IS_DOORWAY_GLYPH_LUT[glyphs]
                    fl.door_mask[doorway_mask] = True
                    fl.door_mask[is_cmap & (~_IS_DOORWAY_GLYPH_LUT[glyphs])] = False
                    fl.remembered_obj_glyphs[is_cmap] = 0
                    obj_mask = _IS_OBJECT_LUT[glyphs]
                    fl.remembered_obj_glyphs[obj_mask] = glyphs[obj_mask]
                    fl.remembered_trap_glyphs[is_cmap & (~_IS_TRAP_LUT[glyphs])] = 0
                    trap_mask = _IS_TRAP_LUT[glyphs]
                    fl.remembered_trap_glyphs[trap_mask] = glyphs[trap_mask]
                    fl._passable_dirty = True

            if in_moveloop and not self._waiting_for_space(obs_tuple):
                step_ctr += 1
                self.step_byte_offsets.append(len(self.action_bytes))
            if self.done:
                break

        if not self.done and (
            self._waiting_for_space(obs_tuple) or not self._in_moveloop(obs_tuple)
        ):
            obs_tuple, self.done = self._settle_prompts(obs_tuple, self.done)

        self._last_raw_tuple = obs_tuple
        self.step_count = step_ctr
        state = GameState(
            obs_tuple,
            step_count=step_ctr,
            mem=self.mem,
            obs_keys=self._obs_keys,
            obs_idx_map=self._idx_map,
        )
        self.mem.update(state)
        return state

    def close(self) -> None:
        try:
            self.nh.close()
        except Exception:
            pass
