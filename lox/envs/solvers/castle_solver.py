"""
LOX Castle Drawbridge Solver:
Automates safe drawbridge breaching at the Castle (Depths 25-29).
Prevents crushed-by-drawbridge instadeaths and moat drowning by:
1. Detecting drawbridge coordinates and surrounding moat.
2. Aligning hero at a safe distance (distance >= 2, directly orthogonal).
3. Zapping Wand of Striking at the closed drawbridge to shatter it into safe rubble.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from lox.core.types import Action, Observation


class CastleDrawbridgeSolver:
    """Stateful solver for safely breaching Castle drawbridges."""

    def __init__(self):
        self.state: str = "IDLE"  # "IDLE", "ALIGNING", "BREACHING", "DONE"
        self.drawbridge_pos: tuple[int, int] | None = None

    def reset(self) -> None:
        self.state = "IDLE"
        self.drawbridge_pos = None

    def plan_step(
        self, obs: Observation, drawbridge_pos: tuple[int, int] | None = None
    ) -> Action | None:
        """
        Calculates the next Action to position at distance 2 and zap wand of striking.
        """
        hero = obs.hero
        hy, hx = hero.y, hero.x

        if drawbridge_pos is not None:
            self.drawbridge_pos = drawbridge_pos

        if self.drawbridge_pos is None:
            return None

        dy, dx = self.drawbridge_pos[0] - hy, self.drawbridge_pos[1] - hx
        dist = abs(dy) + abs(dx)

        # Danger: If adjacent to closed drawbridge (dist == 1), step back immediately!
        if dist == 1:
            step_back_dir = (-dy, -dx)
            return Action(name="step_direction", direction=step_back_dir)

        # Ideal firing position: distance == 2 in a straight orthogonal line
        if (abs(dy) == 2 and dx == 0) or (abs(dx) == 2 and dy == 0):
            # Fire wand of striking directly at drawbridge
            strike_dir = (1 if dy > 0 else (-1 if dy < 0 else 0), 1 if dx > 0 else (-1 if dx < 0 else 0))
            if obs.inventory.has_offensive_wand or getattr(obs.inventory, "has_wand_of_striking", False):
                self.state = "DONE"
                return Action(name="zap_offensive_wand", direction=strike_dir)
            else:
                return Action(name="wait")

        # Otherwise step toward orthogonal stand position at distance 2
        target_y = self.drawbridge_pos[0] + (2 if dy > 0 else (-2 if dy < 0 else 0))
        target_x = self.drawbridge_pos[1] + (2 if dx > 0 else (-2 if dx < 0 else 0))
        return Action(name="step_to", target_pos=(target_y, target_x))

    @staticmethod
    def detect_drawbridge(chars: np.ndarray | None, message: str = "") -> tuple[int, int] | None:
        """Finds coordinates of a closed drawbridge near moat/water."""
        if chars is None:
            return None
        # In Castle, moat is '}' (pool/moat). A bridge/door '#' or '+' adjacent to water '}'
        water_mask = (chars == ord("}"))
        if not np.any(water_mask):
            return None
        for wy, wx in np.argwhere(water_mask):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = wy + dy, wx + dx
                if 0 <= ny < 21 and 0 <= nx < 79:
                    if chars[ny, nx] in (ord("#"), ord("+")):
                        return (int(ny), int(nx))
        return None
