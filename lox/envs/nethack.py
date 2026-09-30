"""
LOX 2.0 NetHack Adapter: High-performance, clean NLE wrapper.
Directly consumes structured NLE observation channels (blstats, inv_strs, inv_letters)
with zero regex scraping and automatic --More-- prompt dismissal.
Equipped with hardcoded safety interlocks (floating eye lockout, corpse rot guard, prayer timer).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import gymnasium as gym
import nle
from nle import nethack

from lox.core.types import Observation, HeroState, HungerState, Item, Action
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


class NetHackAdapter(EnvironmentAdapter):
    """Clean NetHack adapter for NetHackChallenge-v0 and NetHackScore-v0."""

    def __init__(self, env_id: str = "NetHackChallenge-v0", role: str = "valkyrie"):
        self.env_id = env_id
        self.role = role
        self.env = gym.make(env_id, character=role)
        self.visited = np.zeros((21, 79), dtype=bool)
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0

        # Safety interlock tracking
        self.last_prayer_turn = -1000
        self.floor_corpses: dict[tuple[int, int], int] = {}  # (y, x) -> drop_turn

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
        dnum = int(bl[23])

        if depth != self.last_depth or dnum != self.last_dnum:
            self.turns_on_level = 0
            self.visited.fill(False)
            self.floor_corpses.clear()
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
        )

        inventory: list[Item] = []
        inv_letters = raw_obs.get("inv_letters", [])
        inv_strs = raw_obs.get("inv_strs", [])
        inv_oclasses = raw_obs.get("inv_oclasses", [])

        for letter, desc_bytes, oclass in zip(inv_letters, inv_strs, inv_oclasses):
            if letter > 0:
                slot = chr(int(letter))
                desc = self._decode_message(desc_bytes)
                cat = OCLASS_MAP.get(int(oclass), "unknown")
                inventory.append(Item(slot=slot, name=desc, category=cat))

        message = self._decode_message(raw_obs.get("message", ""))

        # Track fresh corpses from kill message
        if "you kill the" in message.lower() or "you destroy the" in message.lower():
            self.floor_corpses[(y, x)] = turn

        return Observation(
            chars=raw_obs["chars"],
            glyphs=raw_obs["glyphs"],
            hero=hero,
            inventory=inventory,
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
            elif "(y/n)" in msg or "Really quit?" in msg:
                raw_obs, _, term, trunc, _ = self.env.step(self.char_to_act.get("n", space_idx))
            else:
                break
        return raw_obs, term, trunc

    def reset(self, seed: int | None = None) -> Observation:
        self.visited.fill(False)
        self.turns_on_level = 0
        self.last_depth = 1
        self.last_dnum = 0
        self.last_prayer_turn = -1000
        self.floor_corpses.clear()
        raw_obs, _ = self.env.reset(seed=seed)
        raw_obs, _, _ = self._dismiss_more(raw_obs, False, False)
        return self._extract_obs(raw_obs)

    def is_target_floating_eye(self, glyphs: np.ndarray, y: int, x: int) -> bool:
        """Floating Eye Safety Interlock: Monster 28 is floating eye; melee strictly locked out."""
        if not (0 <= y < 21 and 0 <= x < 79):
            return False
        g = int(glyphs[y, x])
        if nethack.glyph_is_monster(g):
            mon_id = nethack.glyph_to_mon(g)
            return mon_id == 28
        return False

    def can_safely_pray(self, turn: int) -> bool:
        """Prayer Safety Interlock: Cooldown of >= 350 turns without divine retribution."""
        return (turn - self.last_prayer_turn) >= 350

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        target_char = "."

        # Check melee attack interlock for floating eyes
        if action.name == "melee_attack_hostile" and action.direction is not None:
            dy, dx = action.direction
            # Lookup target tile
            hero = getattr(self, "_last_hero", None)
            if hero is not None and hasattr(self, "_last_glyphs"):
                ty = hero.y + dy
                tx = hero.x + dx
                if self.is_target_floating_eye(self._last_glyphs, ty, tx):
                    # SAFETY INTERLOCK: Replace suicide melee attack with search
                    action = Action(name="search")

        if action.direction is not None and action.direction in DIR_CHARS:
            target_char = DIR_CHARS[action.direction]
        elif action.char is not None:
            target_char = action.char
        elif action.name == "descend":
            target_char = ">"
        elif action.name == "ascend":
            target_char = "<"
        elif action.name == "search":
            target_char = "s"
        elif action.name in ("wait", "rest"):
            target_char = "."
        elif action.name == "pray":
            raw_obs, reward, term, trunc, info = self.env.step(self.pray_action_idx)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            self.last_prayer_turn = obs.hero.turn
            return obs, float(reward), bool(term), bool(trunc), info
        elif action.name == "eat_carried_food" and action.slot:
            act_e = self.char_to_act.get("e", 0)
            raw_obs, reward, term, trunc, info = self.env.step(act_e)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            act_slot = self.char_to_act.get(action.slot, 0)
            raw_obs, r2, term, trunc, info = self.env.step(act_slot)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "kick_closed_door" and action.direction:
            dir_char = DIR_CHARS.get(action.direction, "l")
            raw_obs, reward, term, trunc, info = self.env.step(48)  # Command.KICK
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            act_dir = self.char_to_act.get(dir_char, 0)
            raw_obs, r2, term, trunc, info = self.env.step(act_dir)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "open_door" and action.direction:
            dir_char = DIR_CHARS.get(action.direction, "l")
            raw_obs, reward, term, trunc, info = self.env.step(57)  # Command.OPEN
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            act_dir = self.char_to_act.get(dir_char, 0)
            raw_obs, r2, term, trunc, info = self.env.step(act_dir)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "engrave_elbereth":
            # Command.ENGRAVE (36) -> fingertip '-' (106) -> space -> letters -> enter (19)
            raw_obs, reward, term, trunc, info = self.env.step(36)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            raw_obs, r2, term, trunc, info = self.env.step(106)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            raw_obs, _, term, trunc, _ = self.env.step(self.char_to_act.get(" ", 107))
            for c in "Elbereth":
                raw_obs, _, term, trunc, _ = self.env.step(self.char_to_act.get(c, 19))
            raw_obs, _, term, trunc, info = self.env.step(19)  # ENTER
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "eat_floor_corpse":
            # Command.EAT (35) -> 'y' to confirm eating from floor
            raw_obs, reward, term, trunc, info = self.env.step(35)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            raw_obs, r2, term, trunc, info = self.env.step(self.char_to_act.get("y", 0))
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "dip_excalibur" and action.direction:
            # Command.DIP (32) -> longsword slot -> fountain direction
            dir_char = DIR_CHARS.get(action.direction, "l")
            raw_obs, reward, term, trunc, info = self.env.step(32)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            slot_act = self.char_to_act.get(action.slot or "a", 0)
            raw_obs, r2, term, trunc, info = self.env.step(slot_act)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            act_dir = self.char_to_act.get(dir_char, 0)
            raw_obs, r3, term, trunc, info = self.env.step(act_dir)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2 + r3), bool(term), bool(trunc), info
        elif action.name == "wield_weapon" and action.slot:
            # Command.WIELD (102) -> item slot
            raw_obs, reward, term, trunc, info = self.env.step(102)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            raw_obs, r2, term, trunc, info = self.env.step(self.char_to_act.get(action.slot, 0))
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "wear_armor" and action.slot:
            # Command.WEAR (99) -> item slot
            raw_obs, reward, term, trunc, info = self.env.step(99)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            raw_obs, r2, term, trunc, info = self.env.step(self.char_to_act.get(action.slot, 0))
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info
        elif action.name == "quaff_healing" and action.slot:
            act_q = self.char_to_act.get("q", 0)
            raw_obs, reward, term, trunc, info = self.env.step(act_q)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            act_slot = self.char_to_act.get(action.slot, 0)
            raw_obs, r2, term, trunc, info = self.env.step(act_slot)
            raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
            obs = self._extract_obs(raw_obs)
            return obs, float(reward + r2), bool(term), bool(trunc), info

        act_idx = self.char_to_act.get(target_char, self.char_to_act.get(".", 0))
        raw_obs, reward, term, trunc, info = self.env.step(act_idx)
        raw_obs, term, trunc = self._dismiss_more(raw_obs, term, trunc)
        obs = self._extract_obs(raw_obs)

        self._last_hero = obs.hero
        if obs.glyphs is not None:
            self._last_glyphs = obs.glyphs

        return obs, float(reward), bool(term), bool(trunc), info

    def close(self) -> None:
        self.env.close()
