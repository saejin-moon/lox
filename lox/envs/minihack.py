"""
LOX MiniHack Adapter: Clean, high-throughput Gymnasium MiniHack wrapper.
Translates atomic Behavior Tree actions into gym actions and extracts standardized Observations.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import gymnasium
import minihack  # noqa: F401

from lox.core.types import Observation, HeroState, HungerState, Action
from lox.envs.base import EnvironmentAdapter


# Direction offsets mapped to standard NetHack direction characters
CHAR_DIR_MAP: dict[tuple[int, int], str] = {
    (-1, 0): "k",  # North
    (0, 1): "l",   # East
    (1, 0): "j",   # South
    (0, -1): "h",  # West
    (-1, 1): "u",  # North-East
    (1, 1): "n",   # South-East
    (1, -1): "b",  # South-West
    (-1, -1): "y", # North-West
}


class MiniHackAdapter(EnvironmentAdapter):
    """High-speed gymnasium adapter for MiniHack environments."""

    def __init__(self, task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0"):
        self.task_id = task
        self.env = gymnasium.make(
            task,
            observation_keys=["glyphs", "chars", "blstats", "message"],
        )
        self.visited = np.zeros((21, 79), dtype=bool)
        self.turns_on_level = 0
        self.last_depth = 1

        # Build dynamic character-to-action-index lookup table from unwrapped env
        unwrapped = getattr(self.env, "unwrapped", self.env)
        actions = getattr(unwrapped, "actions", [])
        self.char_to_idx: dict[str, int] = {}
        for idx, act in enumerate(actions):
            self.char_to_idx[chr(int(act))] = idx

    def _extract_obs(self, raw_obs: dict[str, Any]) -> Observation:
        blstats = raw_obs["blstats"]
        x, y = int(blstats[0]), int(blstats[1])
        hp, max_hp = int(blstats[9]), int(blstats[10])
        depth = int(blstats[11])
        turn = int(blstats[19])
        hunger_raw = int(blstats[20])
        hunger = HungerState(min(4, max(0, hunger_raw)))

        if depth != self.last_depth:
            self.turns_on_level = 0
            self.visited.fill(False)
            self.last_depth = depth
        else:
            self.turns_on_level += 1

        self.visited[y, x] = True

        hero = HeroState(
            y=y,
            x=x,
            hp=hp,
            max_hp=max_hp,
            depth=depth,
            turn=turn,
            turns_on_level=self.turns_on_level,
            hunger_state=hunger,
        )

        msg_bytes = raw_obs.get("message", b"")
        if isinstance(msg_bytes, np.ndarray):
            message = "".join(chr(c) for c in msg_bytes if c > 0).strip()
        elif isinstance(msg_bytes, bytes):
            message = msg_bytes.decode("ascii", errors="ignore").strip()
        else:
            message = str(msg_bytes).strip()

        return Observation(
            chars=raw_obs["chars"],
            glyphs=raw_obs["glyphs"],
            hero=hero,
            message=message,
            raw_obs=raw_obs,
        )

    def reset(self, seed: int | None = None) -> Observation:
        self.visited.fill(False)
        self.turns_on_level = 0
        self.last_depth = 1
        raw_obs, _ = self.env.reset(seed=seed)
        return self._extract_obs(raw_obs)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        target_char = "s"  # default safe search/rest
        if action.direction is not None and action.direction in CHAR_DIR_MAP:
            target_char = CHAR_DIR_MAP[action.direction]
        elif action.char is not None:
            target_char = action.char
        elif action.name == "search":
            target_char = "s"
        elif action.name == "descend":
            target_char = ">"
        elif action.name == "ascend":
            target_char = "<"

        act_idx = self.char_to_idx.get(target_char)
        if act_idx is None:
            # Fallback if target char not in restricted action set
            act_idx = self.char_to_idx.get("s", 0)

        raw_obs, reward, terminated, truncated, info = self.env.step(act_idx)
        obs = self._extract_obs(raw_obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
