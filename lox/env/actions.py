"""
1-to-1 NetHack Action Mapping ("Hard Code Nothing").
Maps policy yields directly to NLE native Command/Compass/Text byte values.
Supports single atomic actions, raw ASCII bytes/characters, and multi-key sequences.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from nle.nethack.actions import (
    Command,
    CompassDirection,
    CompassDirectionLonger,
    MiscAction,
    MiscDirection,
    TextCharacters,
)


class Action(enum.IntEnum):
    """Complete 1-to-1 enumeration of NLE native actions (byte values sent to NetHack C engine)."""

    # Compass Directions (vi-keys)
    N = CompassDirection.N.value  # ord('k')
    E = CompassDirection.E.value  # ord('l')
    S = CompassDirection.S.value  # ord('j')
    W = CompassDirection.W.value  # ord('h')
    NE = CompassDirection.NE.value  # ord('u')
    SE = CompassDirection.SE.value  # ord('n')
    SW = CompassDirection.SW.value  # ord('b')
    NW = CompassDirection.NW.value  # ord('y')

    # Full-word Compass Aliases
    NORTH = CompassDirection.N.value
    EAST = CompassDirection.E.value
    SOUTH = CompassDirection.S.value
    WEST = CompassDirection.W.value
    NORTHEAST = CompassDirection.NE.value
    SOUTHEAST = CompassDirection.SE.value
    SOUTHWEST = CompassDirection.SW.value
    NORTHWEST = CompassDirection.NW.value

    # Run / Longer Compass Directions (Shift + vi-keys)
    RUN_N = CompassDirectionLonger.N.value
    RUN_E = CompassDirectionLonger.E.value
    RUN_S = CompassDirectionLonger.S.value
    RUN_W = CompassDirectionLonger.W.value
    RUN_NE = CompassDirectionLonger.NE.value
    RUN_SE = CompassDirectionLonger.SE.value
    RUN_SW = CompassDirectionLonger.SW.value
    RUN_NW = CompassDirectionLonger.NW.value

    # Vertical & Misc Directions
    UP = MiscDirection.UP.value  # ord('<')
    DOWN = MiscDirection.DOWN.value  # ord('>')
    ASCEND = MiscDirection.UP.value  # Alias for UP
    DESCEND = MiscDirection.DOWN.value  # Alias for DOWN
    WAIT = MiscDirection.WAIT.value  # ord('.')
    MORE = MiscAction.MORE.value  # ord('\r')

    # All Native NLE Commands
    EXTCMD = Command.EXTCMD.value  # ord('#')
    EXTLIST = Command.EXTLIST.value  # M('?')
    ADJUST = Command.ADJUST.value  # M('a')
    ANNOTATE = Command.ANNOTATE.value  # M('A')
    APPLY = Command.APPLY.value  # ord('a')
    ATTRIBUTES = Command.ATTRIBUTES.value  # C('x')
    AUTOPICKUP = Command.AUTOPICKUP.value  # ord('@')
    CALL = Command.CALL.value  # ord('C')
    CAST = Command.CAST.value  # ord('Z')
    CHAT = Command.CHAT.value  # M('c')
    CLOSE = Command.CLOSE.value  # ord('c')
    CONDUCT = Command.CONDUCT.value  # M('C')
    DIP = Command.DIP.value  # M('d')
    DROP = Command.DROP.value  # ord('d')
    DROPTYPE = Command.DROPTYPE.value  # ord('D')
    EAT = Command.EAT.value  # ord('e')
    ENGRAVE = Command.ENGRAVE.value  # ord('E')
    ENHANCE = Command.ENHANCE.value  # M('e')
    ESC = Command.ESC.value  # C('[') == 27
    FIGHT = Command.FIGHT.value  # ord('F')
    FIRE = Command.FIRE.value  # ord('f')
    FORCE = Command.FORCE.value  # M('f')
    GLANCE = Command.GLANCE.value  # ord(';')
    HISTORY = Command.HISTORY.value  # ord('V')
    INVENTORY = Command.INVENTORY.value  # ord('i')
    INVENTTYPE = Command.INVENTTYPE.value  # ord('I')
    INVOKE = Command.INVOKE.value  # M('i')
    JUMP = Command.JUMP.value  # M('j')
    KICK = Command.KICK.value  # C('d') == 4
    KNOWN = Command.KNOWN.value  # ord('\\')
    KNOWNCLASS = Command.KNOWNCLASS.value  # ord('`')
    LOOK = Command.LOOK.value  # ord(':')
    LOOT = Command.LOOT.value  # M('l')
    MONSTER = Command.MONSTER.value  # M('m')
    MOVE = Command.MOVE.value  # ord('m')
    MOVEFAR = Command.MOVEFAR.value  # ord('M')
    OFFER = Command.OFFER.value  # M('o')
    OPEN = Command.OPEN.value  # ord('o')
    OPTIONS = Command.OPTIONS.value  # ord('O')
    OVERVIEW = Command.OVERVIEW.value  # C('o')
    PAY = Command.PAY.value  # ord('p')
    PICKUP = Command.PICKUP.value  # ord(',')
    PRAY = Command.PRAY.value  # M('p')
    PUTON = Command.PUTON.value  # ord('P')
    QUAFF = Command.QUAFF.value  # ord('q')
    QUIT = Command.QUIT.value  # M('q')
    QUIVER = Command.QUIVER.value  # ord('Q')
    READ = Command.READ.value  # ord('r')
    REDRAW = Command.REDRAW.value  # C('r')
    REMOVE = Command.REMOVE.value  # ord('R')
    RIDE = Command.RIDE.value  # M('R')
    RUB = Command.RUB.value  # M('r')
    RUSH = Command.RUSH.value  # ord('g')
    RUSH2 = Command.RUSH2.value  # ord('G')
    SAVE = Command.SAVE.value  # ord('S')
    SEARCH = Command.SEARCH.value  # ord('s')
    SEEALL = Command.SEEALL.value  # ord('*')
    SEEAMULET = Command.SEEAMULET.value  # ord('"')
    SEEARMOR = Command.SEEARMOR.value  # ord('[')
    SEEGOLD = Command.SEEGOLD.value  # ord('$')
    SEERINGS = Command.SEERINGS.value  # ord('=')
    SEESPELLS = Command.SEESPELLS.value  # ord('+')
    SEETOOLS = Command.SEETOOLS.value  # ord('(')
    SEETRAP = Command.SEETRAP.value  # ord('^')
    SEEWEAPON = Command.SEEWEAPON.value  # ord(')')
    SIT = Command.SIT.value  # M('s')
    SWAP = Command.SWAP.value  # ord('x')
    TAKEOFF = Command.TAKEOFF.value  # ord('T')
    TAKEOFFALL = Command.TAKEOFFALL.value  # ord('A')
    TELEPORT = Command.TELEPORT.value  # C('t')
    THROW = Command.THROW.value  # ord('t')
    TIP = Command.TIP.value  # M('T')
    TRAVEL = Command.TRAVEL.value  # ord('_')
    TURN = Command.TURN.value  # M('t')
    TWOWEAPON = Command.TWOWEAPON.value  # ord('X')
    UNTRAP = Command.UNTRAP.value  # M('u')
    VERSION = Command.VERSION.value  # M('v')
    VERSIONSHORT = Command.VERSIONSHORT.value
    WEAR = Command.WEAR.value  # ord('W')
    WHATDOES = Command.WHATDOES.value  # ord('&')
    WHATIS = Command.WHATIS.value  # ord('/')
    WIELD = Command.WIELD.value  # ord('w')
    WIPE = Command.WIPE.value  # M('w')
    ZAP = Command.ZAP.value  # ord('z')

    # Text / Prompt Characters
    PLUS = TextCharacters.PLUS.value
    MINUS = TextCharacters.MINUS.value
    SPACE = TextCharacters.SPACE.value
    APOS = TextCharacters.APOS.value
    QUOTE = TextCharacters.QUOTE.value
    DOLLAR = TextCharacters.DOLLAR.value
    NUM_0 = TextCharacters.NUM_0.value
    NUM_1 = TextCharacters.NUM_1.value
    NUM_2 = TextCharacters.NUM_2.value
    NUM_3 = TextCharacters.NUM_3.value
    NUM_4 = TextCharacters.NUM_4.value
    NUM_5 = TextCharacters.NUM_5.value
    NUM_6 = TextCharacters.NUM_6.value
    NUM_7 = TextCharacters.NUM_7.value
    NUM_8 = TextCharacters.NUM_8.value
    NUM_9 = TextCharacters.NUM_9.value


# Lookup dicts for fast 1-to-1 resolution
ACTION_BY_NAME: dict[str, int] = {a.name: int(a.value) for a in Action}
ACTION_BY_NAME.update({a.name.lower(): int(a.value) for a in Action})

ACTION_BY_CHAR: dict[int, str] = {int(a.value): a.name for a in Action}

# Standard coordinate delta (dy, dx) -> Action
DIR_DELTAS: dict[tuple[int, int], Action] = {
    (-1, 0): Action.N,
    (0, 1): Action.E,
    (1, 0): Action.S,
    (0, -1): Action.W,
    (-1, 1): Action.NE,
    (1, 1): Action.SE,
    (1, -1): Action.SW,
    (-1, -1): Action.NW,
    (0, 0): Action.WAIT,
}


@dataclass(frozen=True, slots=True)
class RawAction:
    """Explicit single-byte NetHack keystroke action."""

    byte_val: int
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or ACTION_BY_CHAR.get(self.byte_val, f"KEY_{self.byte_val}")


@dataclass(frozen=True, slots=True)
class ActionSequence:
    """Ordered sequence of NetHack keystrokes executed across consecutive C step calls."""

    bytes_seq: tuple[int, ...]
    label: str = ""

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        parts = [
            ACTION_BY_CHAR.get(b, chr(b) if 32 <= b <= 126 else f"\\x{b:02x}")
            for b in self.bytes_seq
        ]
        return "seq(" + ",".join(parts) + ")"


def key(ch: str | int | Action) -> RawAction:
    """Constructs a 1-to-1 single-keystroke action from a character, Action enum, or byte int."""
    if isinstance(ch, Action):
        return RawAction(int(ch.value), ch.name)
    if isinstance(ch, int):
        return RawAction(ch & 0xFF, ACTION_BY_CHAR.get(ch & 0xFF, f"KEY_{ch & 0xFF}"))
    if isinstance(ch, str):
        if len(ch) == 1:
            v = ord(ch) & 0xFF
            return RawAction(v, ACTION_BY_CHAR.get(v, ch))
        if ch in ACTION_BY_NAME:
            v = ACTION_BY_NAME[ch]
            return RawAction(v, ACTION_BY_CHAR.get(v, ch.upper()))
    raise ValueError(f"Cannot convert {ch!r} to a single NetHack keystroke.")


def seq(*steps: str | int | Action | RawAction, label: str = "") -> ActionSequence:
    """
    Constructs a multi-keystroke sequence (e.g. `seq(Action.EAT, 'f')` or `seq(Action.ENGRAVE, '-', 'Elbereth\\r')`).
    Strings longer than 1 character (that are not Action enum names) are expanded character-by-character.
    """
    out: list[int] = []
    for item in steps:
        if isinstance(item, Action):
            out.append(int(item.value))
        elif isinstance(item, RawAction):
            out.append(item.byte_val)
        elif isinstance(item, int):
            out.append(item & 0xFF)
        elif isinstance(item, str):
            if len(item) > 1 and item in ACTION_BY_NAME:
                out.append(ACTION_BY_NAME[item])
            else:
                for ch in item:
                    out.append(ord(ch) & 0xFF)
        elif (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and all(isinstance(x, int) for x in item)
        ):
            # Support (dy, dx) direction delta tuple
            act = DIR_DELTAS.get((int(item[0]), int(item[1])), Action.WAIT)
            out.append(int(act.value))
        else:
            raise ValueError(f"Unsupported element in seq(): {item!r}")
    return ActionSequence(tuple(out), label=label)


def normalize_action_to_bytes(yielded: Any) -> tuple[tuple[int, ...], str]:
    """
    Normalizes any policy yield value into a tuple of raw NetHack byte(s) and a human-readable action name.
    Supports:
    - `Action.N` / `int`
    - `RawAction`
    - `ActionSequence`
    - `(dy, dx)` direction delta tuple (e.g. `(-1, 0)` -> `Action.N`)
    - `str` (either an Action name like `'SEARCH'`, a single char `'k'`, or a multi-char string `'ef'`)
    - `list` / `tuple` of keystrokes
    """
    if yielded is None:
        return (int(Action.WAIT.value),), "WAIT"
    if isinstance(yielded, Action):
        return (int(yielded.value),), yielded.name
    if isinstance(yielded, RawAction):
        return (yielded.byte_val,), yielded.name
    if isinstance(yielded, ActionSequence):
        if not yielded.bytes_seq:
            return (int(Action.WAIT.value),), "WAIT"
        return yielded.bytes_seq, yielded.name
    if isinstance(yielded, int):
        b = yielded & 0xFF
        return (b,), ACTION_BY_CHAR.get(b, f"KEY_{b}")
    if isinstance(yielded, str):
        if not yielded:
            return (int(Action.WAIT.value),), "WAIT"
        if len(yielded) == 1:
            b = ord(yielded) & 0xFF
            return (b,), ACTION_BY_CHAR.get(b, yielded)
        if yielded in ACTION_BY_NAME:
            b = ACTION_BY_NAME[yielded]
            return (b,), ACTION_BY_CHAR.get(b, yielded.upper())
        s = seq(yielded)
        return s.bytes_seq, s.name
    if isinstance(yielded, (tuple, list)):
        if len(yielded) == 2 and all(isinstance(x, (int, float)) for x in yielded):
            dy, dx = int(yielded[0]), int(yielded[1])
            if (-1 <= dy <= 1) and (-1 <= dx <= 1):
                act = DIR_DELTAS.get((dy, dx), Action.WAIT)
                return (int(act.value),), act.name
        s = seq(*yielded)
        return s.bytes_seq, s.name
    return (int(Action.WAIT.value),), "WAIT"
