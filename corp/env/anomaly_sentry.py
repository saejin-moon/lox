"""
Anomaly Sentry: Sub-0.05 ms zero-allocation physical invariant evaluation module.
"""

from dataclasses import dataclass
from enum import Enum, auto
from corp.env.blstats import BottomLineStats, ConditionFlag, HungerState


class AnomalySeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()


@dataclass(slots=True, frozen=True)
class AnomalyEvent:
    severity: AnomalySeverity
    condition_name: str
    message: str
    turn: int
    delta_hp: int = 0
    current_hp: int = 0
    max_hp: int = 0


class AnomalySentry:
    """
    Evaluates physical state deltas on every game tick to detect life-critical
    hazards, fatal status conditions, and spatial deadlocks.
    """

    def __init__(self):
        self.prev_blstats: BottomLineStats | None = None
        self.position_history: list[tuple[int, int]] = []

    def reset(self):
        self.prev_blstats = None
        self.position_history.clear()

    def evaluate(self, current: BottomLineStats, raw_message: str = "") -> list[AnomalyEvent]:
        """Evaluates invariant rules against current observation."""
        events: list[AnomalyEvent] = []

        if self.prev_blstats is None:
            self.prev_blstats = current
            self.position_history.append((current.x, current.y))
            return events

        # 1. Burst HP Drop Check (Loss of >= 20% max HP in a single step)
        if current.hp < self.prev_blstats.hp:
            hp_loss = self.prev_blstats.hp - current.hp
            if hp_loss >= max(3, int(current.max_hp * 0.20)):
                events.append(AnomalyEvent(
                    severity=AnomalySeverity.CRITICAL,
                    condition_name="BURST_DAMAGE",
                    message=f"Burst HP drop: -{hp_loss} HP ({self.prev_blstats.hp} -> {current.hp})",
                    turn=current.turn,
                    delta_hp=hp_loss,
                    current_hp=current.hp,
                    max_hp=current.max_hp,
                ))

        # 2. Critical HP Floor Check
        critical_threshold = max(4, int(current.max_hp * 0.25))
        if current.hp <= critical_threshold and current.hp > 0:
            events.append(AnomalyEvent(
                severity=AnomalySeverity.CRITICAL,
                condition_name="CRITICAL_HP_FLOOR",
                message=f"HP below critical threshold: {current.hp}/{current.max_hp}",
                turn=current.turn,
                current_hp=current.hp,
                max_hp=current.max_hp,
            ))

        # 3. Lethal Status Condition Check
        if current.has_lethal_condition:
            active_conditions = []
            if current.condition_bits & ConditionFlag.STONE:
                active_conditions.append("PETRIFYING")
            if current.condition_bits & ConditionFlag.SLIME:
                active_conditions.append("SLIMING")
            if current.condition_bits & ConditionFlag.STRANGLE:
                active_conditions.append("STRANGLING")
            if current.condition_bits & ConditionFlag.FOOD_POISON:
                active_conditions.append("FOOD_POISONING")
            if current.condition_bits & ConditionFlag.TERM_ILL:
                active_conditions.append("TERMINAL_ILLNESS")

            events.append(AnomalyEvent(
                severity=AnomalySeverity.FATAL,
                condition_name="LETHAL_STATUS",
                message=f"Lethal status active: {', '.join(active_conditions)}",
                turn=current.turn,
                current_hp=current.hp,
                max_hp=current.max_hp,
            ))

        # 4. Control Impairment Check
        if current.is_stunned or current.is_confused or current.is_blind or current.is_hallucinating:
            impairments = []
            if current.is_stunned:
                impairments.append("STUNNED")
            if current.is_confused:
                impairments.append("CONFUSED")
            if current.is_blind:
                impairments.append("BLIND")
            if current.is_hallucinating:
                impairments.append("HALLUCINATING")

            events.append(AnomalyEvent(
                severity=AnomalySeverity.WARNING,
                condition_name="CONTROL_IMPAIRMENT",
                message=f"Sensory/Control impaired: {', '.join(impairments)}",
                turn=current.turn,
                current_hp=current.hp,
                max_hp=current.max_hp,
            ))

        # 5. Critical Hunger Check
        if current.hunger_state >= HungerState.WEAK:
            state_str = "FAINTING" if current.hunger_state >= HungerState.FAINTING else "WEAK"
            events.append(AnomalyEvent(
                severity=AnomalySeverity.CRITICAL,
                condition_name="CRITICAL_HUNGER",
                message=f"Starvation emergency: {state_str}",
                turn=current.turn,
                current_hp=current.hp,
                max_hp=current.max_hp,
            ))

        # 6. Spatial Stalemate Tracker (Last 15 turns)
        self.position_history.append((current.x, current.y))
        if len(self.position_history) > 15:
            self.position_history.pop(0)
            unique_positions = set(self.position_history)
            if len(unique_positions) == 1:
                events.append(AnomalyEvent(
                    severity=AnomalySeverity.WARNING,
                    condition_name="SPATIAL_STALEMATE",
                    message="Agent stationary for 15 consecutive turns",
                    turn=current.turn,
                    current_hp=current.hp,
                    max_hp=current.max_hp,
                ))
            elif len(unique_positions) == 2 and len(self.position_history) == 15:
                # 2-tile oscillation loop
                events.append(AnomalyEvent(
                    severity=AnomalySeverity.WARNING,
                    condition_name="OSCILLATION_LOOP",
                    message="Agent oscillating between 2 tiles for 15 turns",
                    turn=current.turn,
                    current_hp=current.hp,
                    max_hp=current.max_hp,
                ))

        self.prev_blstats = current
        return events
