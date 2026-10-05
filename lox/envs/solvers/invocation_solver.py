"""
LOX Invocation Ritual Solver (Section 6.3).
Automates the mandatory NetHack endgame Invocation ritual at the Vibrating Square.
Tracks the 3 Invocation Tools (Bell of Opening, Book of the Dead, Candelabrum of Abernathy + 7 candles),
identifies the Vibrating Square coordinates on the penultimate Gehennom floor,
and executes the ritual sequence:
1. Apply Candelabrum of Abernathy (attaches candles & lights them)
2. Apply Bell of Opening (rings the Bell)
3. Read the Book of the Dead (opens the staircase down to Moloch's Sanctum)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from lox.core.types import Action, Observation


class InvocationSolver:
    """Stateful macro solver for executing the Invocation ritual."""

    STATES = (
        "SEEK_TOOLS",
        "NAVIGATE_TO_SQUARE",
        "LIGHT_CANDELABRUM",
        "RING_BELL",
        "READ_BOOK",
        "SANCTUM_OPENED",
    )

    def __init__(self):
        self.state: str = "SEEK_TOOLS"
        self.vibrating_square_pos: tuple[int, int] | None = None
        self.candelabrum_lit: bool = False
        self.bell_rung: bool = False
        self.book_read: bool = False

    def reset(self) -> None:
        self.state = "SEEK_TOOLS"
        self.vibrating_square_pos = None
        self.candelabrum_lit = False
        self.bell_rung = False
        self.book_read = False

    @staticmethod
    def detect_vibrating_square(
        chars: np.ndarray | None, message: str = ""
    ) -> tuple[int, int] | None:
        """
        Detects the vibrating square on the floor.
        In NetHack, stepping on the vibrating square produces the message:
        'You feel a strange vibration beneath your feet.'
        Or on some displays it is marked with '~' glyph.
        """
        if "vibration beneath your feet" in message.lower() or "strange vibration" in message.lower():
            return None  # Pos determined by hero's current position

        if chars is not None:
            v_coords = np.argwhere(chars == ord("~"))
            if len(v_coords) > 0:
                return (int(v_coords[0][0]), int(v_coords[0][1]))
        return None

    def has_all_invocation_tools(self, obs: Observation) -> bool:
        """Verifies possession of Bell of Opening, Book of the Dead, and Candelabrum."""
        inv = obs.inventory
        return bool(
            inv.has_bell_of_opening
            and inv.has_book_of_the_dead
            and inv.has_candelabrum
        )

    def plan_step(self, obs: Observation) -> Action | None:
        """
        Calculates the next Action to advance the Invocation ritual.
        """
        hero = obs.hero
        inv = obs.inventory
        hy, hx = hero.y, hero.x
        msg = obs.message.lower()

        # Check message feedback for state transitions
        if "strange vibration" in msg:
            self.vibrating_square_pos = (hy, hx)
        if "candelabrum glows" in msg or "burns with a pure" in msg or "candles are lit" in msg:
            self.candelabrum_lit = True
        if "bell rings" in msg or "bell issues a" in msg or "resonant chime" in msg:
            self.bell_rung = True
        if "stairs appear" in msg or "a stairway appears" in msg or "staircase opens" in msg or "sanctum" in msg:
            self.book_read = True
            self.state = "SANCTUM_OPENED"
            return Action(name="descend")

        if not self.has_all_invocation_tools(obs):
            self.state = "SEEK_TOOLS"
            return None

        # Detect or navigate to vibrating square
        if self.vibrating_square_pos is None:
            detected = self.detect_vibrating_square(obs.chars, obs.message)
            if detected:
                self.vibrating_square_pos = detected

        # If not standing on vibrating square, step toward it
        if self.vibrating_square_pos is not None and (hy, hx) != self.vibrating_square_pos:
            self.state = "NAVIGATE_TO_SQUARE"
            return Action(name="step_to", target_pos=self.vibrating_square_pos)

        # Standing on Vibrating Square! Execute the 3-step ritual sequence:
        # Step 1: Light Candelabrum
        if not self.candelabrum_lit:
            slot = inv.get_candelabrum_slot()
            if slot:
                self.state = "LIGHT_CANDELABRUM"
                return Action(name="apply_unicorn_horn", slot=slot)  # 'a' + slot

        # Step 2: Ring Bell of Opening
        if not self.bell_rung:
            slot = inv.get_bell_slot()
            if slot:
                self.state = "RING_BELL"
                return Action(name="apply_unicorn_horn", slot=slot)  # 'a' + slot

        # Step 3: Read Book of the Dead
        if not self.book_read:
            slot = inv.get_book_slot()
            if slot:
                self.state = "READ_BOOK"
                return Action(name="read_scroll", slot=slot)  # 'r' + slot

        return Action(name="descend")
