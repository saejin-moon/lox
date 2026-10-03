"""
LOX 2.0 Altar BUC Solver:
Automates multi-turn batch item beatitude testing on altars.
Sequence:
1. Navigate to altar tile.
2. Drop unconfirmed inventory items.
3. AltarListener captures flash messages and collapses BUC distribution.
4. Pick up all items back into inventory.
"""
from __future__ import annotations

from typing import Any
from lox.core.types import Action, Observation
from lox.core.epistemic import EpistemicEngine


class AltarBUCSolver:
    """Stateful macro solver for testing inventory BUC beatitude on altars."""

    def __init__(self):
        self.state: str = "IDLE"  # "IDLE", "NAVIGATING", "DROPPING", "PICKING_UP", "DONE"
        self.drop_queue: list[str] = []
        self.dropped_count: int = 0
        self.altar_pos: tuple[int, int] | None = None

    def reset(self) -> None:
        self.state = "IDLE"
        self.drop_queue.clear()
        self.dropped_count = 0
        self.altar_pos = None

    def plan_step(
        self,
        obs: Observation,
        epistemic: EpistemicEngine,
        known_altar_pos: tuple[int, int] | None = None,
    ) -> Action | None:
        """
        Calculates the next atomic Action to advance the altar testing procedure.
        Returns None when procedure is complete or cannot proceed.
        """
        hero = obs.hero
        hy, hx = hero.y, hero.x

        # Find altar position
        if known_altar_pos is not None:
            self.altar_pos = known_altar_pos
        elif obs.dungeon.standing_on_altar:
            self.altar_pos = (hy, hx)

        if self.altar_pos is None and not obs.dungeon.standing_on_altar and not obs.dungeon.adjacent_altar:
            return None

        # 1. Navigate to Altar
        if not obs.dungeon.standing_on_altar:
            self.state = "NAVIGATING"
            if self.altar_pos:
                return Action(name="step_to", target_pos=self.altar_pos)
            return Action(name="step_to_altar")

        # Standing on altar!
        if self.state in ("IDLE", "NAVIGATING"):
            # Collect items with non-zero BUC entropy
            self.drop_queue.clear()
            for item in obs.inventory:
                # Exclude currently equipped items to avoid un-wielding in hostile zones
                if item.is_equipped:
                    continue
                belief = epistemic.get_or_create(
                    uid=item.name,
                    name=item.name,
                    item_class=item.category,
                    slot_letter=item.slot,
                )
                if belief.entropy_buc() > 0.01:
                    self.drop_queue.append(item.slot)

            if not self.drop_queue:
                self.state = "DONE"
                return None

            self.state = "DROPPING"
            self.dropped_count = 0

        # 2. Dropping untested items
        if self.state == "DROPPING":
            if self.drop_queue:
                slot = self.drop_queue.pop(0)
                self.dropped_count += 1
                return Action(name="drop", slot=slot)
            else:
                self.state = "DONE"
                return Action(name="pickup", char=",")

        return None
