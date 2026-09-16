"""
Workers layer: Action dispatcher translating high-level tactical primitives
into physical NetHack Learning Environment (NLE) keystrokes and action indices.
"""

from dataclasses import dataclass
from typing import Any
import gymnasium as gym
from nle import nethack

from corp.planner.htn import Task, PrimitiveTask


@dataclass(slots=True, frozen=True)
class ActionMapping:
    """Action index mapping and direction lookup tables."""
    # Cardinal & diagonal direction delta (dr, dc) -> direction char & action index
    DELTA_TO_DIR_CHAR: dict[tuple[int, int], str] = None
    DIR_CHAR_TO_INDEX: dict[str, int] = None
    CHAR_TO_ACTION_INDEX: dict[str, int] = None


def build_action_tables(env: gym.Env) -> tuple[dict[tuple[int, int], int], dict[str, int], dict[str, int]]:
    """
    Constructs fast O(1) lookup tables from the active environment's action tuple.
    """
    actions = env.unwrapped.actions
    char_to_index: dict[str, int] = {}
    for i, a in enumerate(actions):
        if hasattr(a, "value") and 0 <= a.value <= 255:
            # Map character if printable ASCII
            if 32 <= a.value <= 126:
                char_to_index[chr(a.value)] = i
            char_to_index[f"raw_{a.value}"] = i

    # Direction characters in NetHack:
    # k: North (-1, 0), l: East (0, 1), j: South (1, 0), h: West (0, -1)
    # u: NE (-1, 1), n: SE (1, 1), b: SW (1, -1), y: NW (-1, -1)
    delta_to_char = {
        (-1, 0): "k",
        (0, 1): "l",
        (1, 0): "j",
        (0, -1): "h",
        (-1, 1): "u",
        (1, 1): "n",
        (1, -1): "b",
        (-1, -1): "y",
    }

    delta_to_index: dict[tuple[int, int], int] = {}
    for delta, ch in delta_to_char.items():
        if ch in char_to_index:
            delta_to_index[delta] = char_to_index[ch]

    name_to_index: dict[str, int] = {
        "N": char_to_index.get("k", 0),
        "E": char_to_index.get("l", 1),
        "S": char_to_index.get("j", 2),
        "W": char_to_index.get("h", 3),
        "NE": char_to_index.get("u", 4),
        "SE": char_to_index.get("n", 5),
        "SW": char_to_index.get("b", 6),
        "NW": char_to_index.get("y", 7),
        "WAIT": char_to_index.get(".", 18),
        "SEARCH": char_to_index.get("s", 75),
        "MORE": 19,
        "UP": char_to_index.get("<", 16),
        "DOWN": char_to_index.get(">", 17),
        "PICKUP": char_to_index.get(",", 61),
        "EAT": char_to_index.get("e", 35),
        "QUAFF": char_to_index.get("q", 64),
        "READ": char_to_index.get("r", 67),
        "WIELD": char_to_index.get("w", 102),
        "WEAR": char_to_index.get("W", 99),
        "TAKEOFF": char_to_index.get("T", 88),
        "PRAY": 62, # Command.PRAY
        "ENGRAVE": char_to_index.get("E", 36),
        "FIGHT": char_to_index.get("F", 39),
        "FIRE": char_to_index.get("f", 40),
        "THROW": char_to_index.get("t", 91),
        "APPLY": char_to_index.get("a", 24),
        "DROP": char_to_index.get("d", 33),
        "OPEN": char_to_index.get("o", 57),
        "CLOSE": char_to_index.get("c", 30),
        "KICK": 48, # Command.KICK
        "SPACE": char_to_index.get(" ", 107),
    }

    return delta_to_index, name_to_index, char_to_index


class ActionDispatcher:
    """
    Translates tactical PrimitiveTask instances into concrete NLE steps,
    handling multi-keystroke sequences (e.g. eating, wielding, engraving).
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self.delta_to_index, self.name_to_index, self.char_to_index = build_action_tables(env)

    def delta_to_action(self, dr: int, dc: int) -> int:
        """Converts (dr, dc) coordinate step into NLE action index."""
        # Normalize to -1, 0, 1
        sign_r = (dr > 0) - (dr < 0)
        sign_c = (dc > 0) - (dc < 0)
        return self.delta_to_index.get((sign_r, sign_c), self.name_to_index.get("WAIT", 18))

    def dispatch(
        self,
        task: Task,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Executes a primitive or compound task sequence on the environment.
        Returns the final (obs, reward, terminated, truncated, info).
        """
        name = task.name.upper()
        args = task.args

        if name == "STEP":
            if "delta" in args:
                dr, dc = args["delta"]
                action_idx = self.delta_to_action(dr, dc)
            elif "dir" in args:
                action_idx = self.name_to_index.get(args["dir"].upper(), 18)
            else:
                action_idx = self.name_to_index["WAIT"]
            return self.env.step(action_idx)

        elif name == "MELEE_ATTACK":
            if "delta" in args:
                dr, dc = args["delta"]
                action_idx = self.delta_to_action(dr, dc)
            elif "dir" in args:
                action_idx = self.name_to_index.get(args["dir"].upper(), 18)
            else:
                action_idx = self.name_to_index["WAIT"]
            return self.env.step(action_idx)

        elif name == "SEARCH":
            return self.env.step(self.name_to_index["SEARCH"])

        elif name == "WAIT":
            return self.env.step(self.name_to_index["WAIT"])

        elif name == "DESCEND":
            return self.env.step(self.name_to_index["DOWN"])

        elif name == "ASCEND":
            return self.env.step(self.name_to_index["UP"])

        elif name == "PICKUP":
            return self.env.step(self.name_to_index["PICKUP"])

        elif name == "EAT":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["EAT"], slot)

        elif name == "QUAFF":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["QUAFF"], slot)

        elif name == "READ":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["READ"], slot)

        elif name == "WIELD":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["WIELD"], slot)

        elif name == "WEAR":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["WEAR"], slot)

        elif name == "TAKEOFF":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["TAKEOFF"], slot)

        elif name == "DROP":
            slot = args.get("slot", "")
            return self._execute_slot_command(self.name_to_index["DROP"], slot)

        elif name == "PRAY":
            # Command.PRAY (62) triggers "Are you sure you want to pray? [yn]"
            obs, r, term, trunc, info = self.env.step(self.name_to_index["PRAY"])
            if term or trunc:
                return obs, r, term, trunc, info
            # Send 'y' confirmation
            y_idx = self.char_to_index.get("y", self.name_to_index.get("NW", 7))
            return self.env.step(y_idx)

        elif name == "ENGRAVE_DUST":
            text = args.get("text", "Elbereth")
            # 1. E
            obs, r, term, trunc, info = self.env.step(self.name_to_index["ENGRAVE"])
            if term or trunc:
                return obs, r, term, trunc, info
            # 2. '-' for fingers
            dash_idx = self.char_to_index.get("-", self.name_to_index["WAIT"])
            obs, r, term, trunc, info = self.env.step(dash_idx)
            if term or trunc:
                return obs, r, term, trunc, info
            # 3. Type text
            for ch in text:
                ch_idx = self.char_to_index.get(ch, self.name_to_index["SPACE"])
                obs, r, term, trunc, info = self.env.step(ch_idx)
                if term or trunc:
                    return obs, r, term, trunc, info
            # 4. Return/enter
            enter_idx = self.name_to_index.get("MORE", 19)
            return self.env.step(enter_idx)

        elif name == "OPEN":
            dr, dc = args.get("delta", (0, 0))
            dir_act = self.delta_to_action(dr, dc)
            obs, r, term, trunc, info = self.env.step(self.name_to_index["OPEN"])
            if term or trunc:
                return obs, r, term, trunc, info
            return self.env.step(dir_act)

        elif name == "KICK":
            dr, dc = args.get("delta", (0, 0))
            dir_act = self.delta_to_action(dr, dc)
            obs, r, term, trunc, info = self.env.step(self.name_to_index["KICK"])
            if term or trunc:
                return obs, r, term, trunc, info
            return self.env.step(dir_act)

        # Fallback to WAIT
        return self.env.step(self.name_to_index["WAIT"])

    def _execute_slot_command(
        self, cmd_idx: int, slot: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Executes a command key followed by an inventory slot letter."""
        obs, r, term, trunc, info = self.env.step(cmd_idx)
        if term or trunc or not slot:
            return obs, r, term, trunc, info
        slot_ch = slot[0]
        slot_idx = self.char_to_index.get(slot_ch)
        if slot_idx is not None:
            return self.env.step(slot_idx)
        # If unknown slot key, cancel with space or esc
        esc_idx = self.char_to_index.get(" ", self.name_to_index["SPACE"])
        return self.env.step(esc_idx)
