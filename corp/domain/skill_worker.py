"""
Domain Layer: Skill Worker.
Monitors skill advancement readiness via game messages and experience level milestones,
dispatching Task("ENHANCE") to advance combat and spell proficiencies.
"""

from corp.env.blstats import BottomLineStats
from corp.planner.htn import Task


class SkillWorker:
    """
    Manages character skill advancement via the NetHack #enhance command.
    Ensures weapons and spells are promoted to Skilled/Expert to gain
    to-hit, damage, and spellcasting success bonuses.
    """

    def __init__(self):
        self.last_checked_xl: int = 1
        self.pending_enhance: bool = False
        self.last_enhance_turn: int = 0

    def reset(self) -> None:
        """Resets worker state for a new episode."""
        self.last_checked_xl = 1
        self.pending_enhance = False
        self.last_enhance_turn = 0

    def evaluate_skill_turn(
        self,
        blstats: BottomLineStats,
        message: str = "",
    ) -> Task | None:
        """
        Determines if #enhance should be executed.
        Triggers when:
        1. Game message contains 'feel more confident in your' (skill ready to advance).
        2. Hero gained a new experience level (grants new skill slots), checked at most once every 50 turns.
        """
        msg_lower = message.lower()
        if "feel more confident in your" in msg_lower or "feel you could be more skilled" in msg_lower:
            self.pending_enhance = True

        # Check on XL increase
        if blstats.experience_level > self.last_checked_xl:
            self.last_checked_xl = blstats.experience_level
            self.pending_enhance = True

        # Throttle enhance checks: at most once every 50 turns unless explicitly notified by message
        if self.pending_enhance and (blstats.turn - self.last_enhance_turn >= 50 or "feel more confident" in msg_lower):
            self.pending_enhance = False
            self.last_enhance_turn = blstats.turn
            return Task("ENHANCE", is_primitive=True)

        return None
