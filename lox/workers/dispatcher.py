"""
Workers layer: Action dispatcher translating high-level tactical primitives
into physical NetHack Learning Environment (NLE) keystrokes and action indices.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import gymnasium as gym
from nle import nethack

from lox.planner.htn import Task, PrimitiveTask


@dataclass
class MicroActionFiber:
    """
    A cooperative execution fiber representing a sequence of micro-actions for a Task.
    Allows stepping one micro-action at a time, inspecting intermediate observations,
    detecting emergent hazards, and interrupting early.
    """
    task: Task
    action_sequence: list[int] = field(default_factory=list)
    cursor: int = 0
    interrupted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return self.interrupted or self.cursor >= len(self.action_sequence)

    def next_action(self) -> int | None:
        if self.is_finished:
            return None
        act = self.action_sequence[self.cursor]
        self.cursor += 1
        return act

    def interrupt(self):
        self.interrupted = True

    def append(self, action: int):
        self.action_sequence.append(action)

    def remaining_actions(self) -> list[int]:
        return self.action_sequence[self.cursor:]



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
        "DIP": 32,  # Command.DIP
        "ENHANCE": 37,  # Command.ENHANCE
        "TWOWEAPON": 95,  # Command.TWOWEAPON
        "OFFER": 56,      # Command.OFFER
        "PAY": char_to_index.get("p", 60),      # Command.PAY
        "ZAP": char_to_index.get("z", 105),    # Command.ZAP
        "SPACE": char_to_index.get(" ", 107),
        "LOOK_HERE": char_to_index.get(":", 51),
        "ESC": char_to_index.get("\x1b", 38),
        "ESCAPE": char_to_index.get("\x1b", 38),
    }

    # Dynamically verify action indices if present in environment
    for i, a in enumerate(actions):
        a_name = getattr(a, "name", "")
        if a_name:
            name_to_index[a_name] = i

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

    def create_fiber(self, task: Task) -> MicroActionFiber:
        """
        Constructs a cooperative MicroActionFiber for the given tactical task.
        """
        name = task.name.upper()
        args = task.args
        actions: list[int] = []

        if name == "STEP":
            if "delta" in args:
                dr, dc = args["delta"]
                actions.append(self.delta_to_action(dr, dc))
            elif "dir" in args:
                actions.append(self.name_to_index.get(args["dir"].upper(), 18))
            else:
                actions.append(self.name_to_index["WAIT"])

        elif name == "MELEE_ATTACK":
            if "delta" in args:
                dr, dc = args["delta"]
                actions.append(self.delta_to_action(dr, dc))
            elif "dir" in args:
                actions.append(self.name_to_index.get(args["dir"].upper(), 18))
            else:
                actions.append(self.name_to_index["WAIT"])

        elif name == "SEARCH":
            actions.append(self.name_to_index["SEARCH"])

        elif name == "WAIT":
            actions.append(self.name_to_index["WAIT"])

        elif name == "DESCEND":
            actions.append(self.name_to_index["DOWN"])

        elif name == "ASCEND":
            actions.append(self.name_to_index["UP"])

        elif name == "PICKUP":
            actions.append(self.name_to_index["PICKUP"])

        elif name in ("EAT", "QUAFF", "READ", "WIELD", "WEAR", "TAKEOFF", "DROP", "QUIVER", "REMOVE"):
            actions.append(self.name_to_index[name])
            slot = args.get("slot", "")
            if slot:
                slot_idx = self.char_to_index.get(slot[0])
                if slot_idx is not None:
                    actions.append(slot_idx)
                else:
                    actions.append(self.char_to_index.get(" ", self.name_to_index["SPACE"]))

        elif name == "PRAY":
            actions.append(self.name_to_index["PRAY"])

        elif name in ("ENGRAVE", "ENGRAVE_DUST"):
            text = args.get("text", "Elbereth")
            slot = args.get("slot")
            actions.append(self.name_to_index["ENGRAVE"])
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            else:
                actions.append(self.char_to_index.get("-", self.name_to_index["WAIT"]))
            for ch in text:
                actions.append(self.char_to_index.get(ch, self.name_to_index["SPACE"]))
            actions.append(self.name_to_index.get("MORE", 19))

        elif name == "LOOK_HERE":
            actions.append(self.name_to_index.get("LOOK_HERE", 51))

        elif name == "ENHANCE":
            actions.append(self.name_to_index.get("ENHANCE", 37))

        elif name in ("ESC", "ESCAPE"):
            actions.append(self.name_to_index.get("ESC", 38))

        elif name in ("OPEN", "CLOSE", "CLOSE_DOOR", "KICK", "FORCE", "FORCE_LOCK", "FIGHT", "FORCE_FIGHT", "JUMP"):
            base_cmd = (
                "CLOSE" if name == "CLOSE_DOOR"
                else "FORCE" if name == "FORCE_LOCK"
                else "FIGHT" if name == "FORCE_FIGHT"
                else name
            )
            actions.append(self.name_to_index[base_cmd])
            dr, dc = args.get("delta", (0, 0))
            actions.append(self.delta_to_action(dr, dc))

        elif name == "DIP":
            actions.append(self.name_to_index.get("DIP", 32))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])

        elif name == "FIRE":
            actions.append(self.name_to_index.get("FIRE", 40))
            dr, dc = args.get("delta", (0, 0))
            actions.append(self.delta_to_action(dr, dc))

        elif name in ("LOOT", "LOOT_CONTAINER"):
            actions.append(self.name_to_index.get("LOOT", 52))
            dr, dc = args.get("delta", (0, 0))
            if dr != 0 or dc != 0:
                actions.append(self.delta_to_action(dr, dc))
            else:
                actions.append(self.char_to_index.get(".", self.name_to_index.get("WAIT", 18)))

        elif name == "UNTRAP":
            actions.append(self.name_to_index.get("UNTRAP", 96))
            dr, dc = args.get("delta", (0, 0))
            actions.append(self.delta_to_action(dr, dc) if (dr != 0 or dc != 0) else self.delta_to_action(0, 0))

        elif name == "SIT":
            actions.append(self.name_to_index.get("SIT", 86))

        elif name in ("TURN", "TURN_UNDEAD"):
            actions.append(self.name_to_index.get("TURN", 94))

        elif name == "TWOWEAPON":
            actions.append(self.name_to_index.get("TWOWEAPON", 95))

        elif name in ("SWAP", "SWAP_WEAPON"):
            actions.append(self.name_to_index.get("SWAP", 87))

        elif name == "WIPE":
            actions.append(self.name_to_index.get("WIPE", 103))

        elif name == "ALTAR_TEST":
            actions.append(self.name_to_index.get("DROP", 33))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            actions.append(self.name_to_index.get("PICKUP", 61))

        elif name == "OFFER":
            actions.append(self.name_to_index.get("OFFER", 56))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])

        elif name == "PAY":
            actions.append(self.name_to_index.get("PAY", 60))

        elif name == "RUB":
            actions.append(self.name_to_index.get("APPLY", 24))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])

        elif name in ("ZAP", "ZAP_WAND"):
            actions.append(self.name_to_index.get("ZAP", 105))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            if "delta" in args:
                dr, dc = args["delta"]
                actions.append(self.delta_to_action(dr, dc))
            elif "dir" in args:
                d = args["dir"]
                if d in self.char_to_index:
                    actions.append(self.char_to_index[d])
                elif d.upper() in self.name_to_index:
                    actions.append(self.name_to_index[d.upper()])

        elif name in ("CAST", "CAST_SPELL"):
            actions.append(self.name_to_index.get("CAST", 28))
            spell = args.get("spell", "")
            if spell and spell[0] in self.char_to_index:
                actions.append(self.char_to_index[spell[0]])
            dr, dc = args.get("delta", (0, 0))
            if dr != 0 or dc != 0:
                actions.append(self.delta_to_action(dr, dc))

        elif name == "APPLY":
            actions.append(self.name_to_index.get("APPLY", 24))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            dr, dc = args.get("delta", (0, 0))
            if dr != 0 or dc != 0:
                actions.append(self.delta_to_action(dr, dc))

        elif name == "PUTON":
            actions.append(self.name_to_index.get("PUTON", 63))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            hand = args.get("hand", "r")
            actions.append(self.char_to_index.get(hand, self.char_to_index.get("r")))

        elif name == "THROW":
            actions.append(self.name_to_index.get("THROW", 91))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            dr, dc = args.get("delta", (0, 0))
            actions.append(self.delta_to_action(dr, dc))

        elif name == "ENGRAVE_WAND":
            actions.append(self.name_to_index.get("ENGRAVE", 36))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])

        elif name == "RUB":
            actions.append(self.name_to_index.get("RUB", 71))
            slot = args.get("slot", "")
            if slot and slot[0] in self.char_to_index:
                actions.append(self.char_to_index[slot[0]])
            gem_slot = args.get("gem_slot", "")
            if gem_slot and gem_slot[0] in self.char_to_index:
                actions.append(self.char_to_index[gem_slot[0]])

        elif name == "CHAT":
            actions.append(self.name_to_index.get("CHAT", 29))
            dr, dc = args.get("delta", (0, 0))
            actions.append(self.delta_to_action(dr, dc))

        else:
            actions.append(self.name_to_index.get("WAIT", 18))

        return MicroActionFiber(task=task, action_sequence=actions)

    def _step_action(
        self,
        action_idx: int,
        step_callback: Callable | None = None,
        fiber: MicroActionFiber | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any], bool]:
        """
        Executes a single primitive action index on the environment, invoking the
        cooperative step_callback if provided. Returns (obs, reward, term, trunc, info, should_abort).
        """
        obs, r, term, trunc, info = self.env.step(action_idx)
        should_abort = term or trunc
        if not should_abort and step_callback is not None:
            cb_result = step_callback(obs, r, term, trunc, info)
            if cb_result is False:
                should_abort = True
                if fiber is not None:
                    fiber.interrupt()
        return obs, r, term, trunc, info, should_abort

    def execute_fiber(
        self,
        fiber: MicroActionFiber,
        step_callback: Callable[[dict[str, Any], float, bool, bool, dict[str, Any]], bool | None] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Executes actions in a MicroActionFiber sequentially until finished, interrupted, or terminal.
        """
        last_result = ({}, 0.0, False, False, {})
        while not fiber.is_finished:
            act = fiber.next_action()
            if act is None:
                break
            obs, r, term, trunc, info, should_abort = self._step_action(act, step_callback, fiber)
            last_result = (obs, r, term, trunc, info)
            if should_abort:
                fiber.interrupt()
                break
        return last_result

    def dispatch(
        self,
        task: Task,
        step_callback: Callable[[dict[str, Any], float, bool, bool, dict[str, Any]], bool | None] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Executes a primitive or compound task sequence on the environment.
        Supports cooperative step_callback on every intermediate keystroke
        for early interrupt when critical hazards or combat emergencies occur.
        """
        name = task.name.upper()
        args = task.args

        # Interactive / dynamic tasks requiring mid-sequence observation inspection:
        if name == "ENHANCE":
            fiber = self.create_fiber(task)
            enhance_idx = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(enhance_idx, step_callback, fiber)
            if abort:
                return obs, r, term, trunc, info

            # Check if any skill option is selectable on screen
            tty = obs.get("tty_chars")
            chosen_letter = None
            if tty is not None:
                import re
                lines = ["".join(chr(c) for c in row) for row in tty]
                for line in lines:
                    m = re.search(r'([a-zA-Z])\s+-\s+([a-zA-Z\s]+)\[', line)
                    if m:
                        chosen_letter = m.group(1)
                        break

            if chosen_letter:
                let_idx = self.char_to_index.get(chosen_letter)
                if let_idx is not None:
                    fiber.append(let_idx)
                    obs, r, term, trunc, info, abort = self._step_action(let_idx, step_callback, fiber)
                    if abort:
                        return obs, r, term, trunc, info

            # ESC dismisses any open menu window immediately and returns to dungeon map
            esc_idx = self.name_to_index.get("ESC", 38)
            fiber.append(esc_idx)
            obs, r, term, trunc, info, _ = self._step_action(esc_idx, step_callback, fiber)
            return obs, r, term, trunc, info

        elif name == "ALTAR_TEST":
            fiber = self.create_fiber(task)
            slot = args.get("slot", "")
            drop_cmd = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(drop_cmd, step_callback, fiber)
            if abort or not slot:
                return obs, r, term, trunc, info

            slot_idx = fiber.next_action()
            if slot_idx is None:
                space_idx = self.char_to_index.get(" ", self.name_to_index["SPACE"])
                obs, r, term, trunc, info, _ = self._step_action(space_idx, step_callback, fiber)
                return obs, r, term, trunc, info

            obs, r, term, trunc, info, abort = self._step_action(slot_idx, step_callback, fiber)
            drop_msg = info.get("full_message", "")
            if abort or "cannot drop" in drop_msg.lower() or "don't have" in drop_msg.lower():
                info["altar_flash"] = drop_msg
                return obs, r, term, trunc, info

            pickup_cmd = fiber.next_action()
            if pickup_cmd is None:
                pickup_cmd = self.name_to_index.get("PICKUP", 61)
            obs, r, term, trunc, info, _ = self._step_action(pickup_cmd, step_callback, fiber)
            info["altar_flash"] = drop_msg
            return obs, r, term, trunc, info

        elif name == "ENGRAVE_WAND":
            fiber = self.create_fiber(task)
            slot = args.get("slot", "")
            engrave_cmd = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(engrave_cmd, step_callback, fiber)
            if abort or not slot:
                return obs, r, term, trunc, info

            slot_idx = fiber.next_action()
            if slot_idx is not None:
                obs, r, term, trunc, info, abort = self._step_action(slot_idx, step_callback, fiber)
                if abort:
                    return obs, r, term, trunc, info

            full_msg = info.get("full_message", "").lower()
            if "what do you want to write" in full_msg:
                more_idx = self.name_to_index.get("MORE", 19)
                fiber.append(more_idx)
                obs, r, term, trunc, info, _ = self._step_action(more_idx, step_callback, fiber)
            return obs, r, term, trunc, info

        elif name == "THROW":
            fiber = self.create_fiber(task)
            slot = args.get("slot", "")
            throw_cmd = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(throw_cmd, step_callback, fiber)
            if abort or not slot:
                return obs, r, term, trunc, info

            slot_idx = fiber.next_action()
            if slot_idx is not None:
                obs, r, term, trunc, info, abort = self._step_action(slot_idx, step_callback, fiber)
                if abort:
                    return obs, r, term, trunc, info
                msg = bytes(obs.get("message", b"")).decode("latin-1", errors="replace").lower()
                if "don't have that object" in msg or "never mind" in msg:
                    space_idx = self.char_to_index.get(" ", self.name_to_index.get("WAIT", 107))
                    obs, r, term, trunc, info, _ = self._step_action(space_idx, step_callback, fiber)
                    return obs, r, term, trunc, info

            dir_act = fiber.next_action()
            if dir_act is not None:
                obs, r, term, trunc, info, _ = self._step_action(dir_act, step_callback, fiber)
            return obs, r, term, trunc, info

        elif name == "PUTON":
            fiber = self.create_fiber(task)
            slot = args.get("slot", "")
            hand = args.get("hand", "r")
            puton_cmd = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(puton_cmd, step_callback, fiber)
            if abort or not slot:
                return obs, r, term, trunc, info

            slot_idx = fiber.next_action()
            if slot_idx is not None:
                obs, r, term, trunc, info, abort = self._step_action(slot_idx, step_callback, fiber)
                if abort:
                    return obs, r, term, trunc, info
                msg = info.get("full_message", "").lower()
                if "which ring-finger" in msg or "right or left" in msg:
                    hand_idx = fiber.next_action()
                    if hand_idx is None:
                        hand_idx = self.char_to_index.get(hand, self.char_to_index.get("r"))
                    if hand_idx is not None:
                        obs, r, term, trunc, info, _ = self._step_action(hand_idx, step_callback, fiber)
            return obs, r, term, trunc, info

        elif name == "CHAT":
            fiber = self.create_fiber(task)
            chat_cmd = fiber.next_action()
            obs, r, term, trunc, info, abort = self._step_action(chat_cmd, step_callback, fiber)
            if abort:
                return obs, r, term, trunc, info

            dir_act = fiber.next_action()
            if dir_act is not None:
                obs, r, term, trunc, info, abort = self._step_action(dir_act, step_callback, fiber)
                if abort:
                    return obs, r, term, trunc, info

            msg = info.get("full_message", "").lower()
            if "how much will you give" in msg or "how much" in msg:
                amount = str(args.get("amount", "400"))
                for dig in amount:
                    d_idx = self.char_to_index.get(dig)
                    if d_idx is not None:
                        fiber.append(d_idx)
                        obs, r, term, trunc, info, abort = self._step_action(d_idx, step_callback, fiber)
                        if abort:
                            return obs, r, term, trunc, info
                enter_idx = self.name_to_index.get("MORE", 19)
                fiber.append(enter_idx)
                obs, r, term, trunc, info, _ = self._step_action(enter_idx, step_callback, fiber)
            return obs, r, term, trunc, info

        # All static multi-step and single-step tasks:
        fiber = self.create_fiber(task)
        return self.execute_fiber(fiber, step_callback=step_callback)

    def _execute_slot_command(
        self,
        cmd_idx: int,
        slot: str,
        step_callback: Callable | None = None,
        fiber: MicroActionFiber | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Executes a command key followed by an inventory slot letter."""
        obs, r, term, trunc, info, abort = self._step_action(cmd_idx, step_callback, fiber)
        if abort or not slot:
            return obs, r, term, trunc, info
        slot_ch = slot[0]
        slot_idx = self.char_to_index.get(slot_ch)
        if slot_idx is not None:
            obs, r, term, trunc, info, _ = self._step_action(slot_idx, step_callback, fiber)
            return obs, r, term, trunc, info
        esc_idx = self.char_to_index.get(" ", self.name_to_index["SPACE"])
        obs, r, term, trunc, info, _ = self._step_action(esc_idx, step_callback, fiber)
        return obs, r, term, trunc, info

