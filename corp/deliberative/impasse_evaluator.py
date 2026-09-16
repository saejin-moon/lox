"""
Holistic Cognitive Impasse Evaluator: Gating System 2 Deliberative LLM Triggers.
Replaces artificial timers/rate-limiters with multi-dimensional heuristic assessment
of dungeon topology, macro-stagnation horizons, latent inventory affordances,
and acute survival urgency.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import NormalizedItem
from corp.domain.navigation_manager import LevelMap


@dataclass(slots=True)
class ImpasseEvaluation:
    """Diagnostic outcome of a holistic cognitive impasse evaluation."""
    should_trigger: bool
    score: float
    stagnation_score: float
    topology_score: float
    affordance_score: float
    dilemma_score: float
    urgency_multiplier: float
    diagnosis: str


class HolisticImpasseEvaluator:
    """
    Evaluates whether the game state represents a true macro-strategic impasse
    requiring an LLM reasoner, or a minor micro-oscillation resolvable by System 1.
    """

    ESCAPE_TOOL_KEYWORDS = [
        "pick-axe",
        "pick",
        "blindfold",
        "towel",
        "horn",
        "mirror",
        "wand of digging",
        "wand of teleport",
        "wand of striking",
        "scroll of teleport",
        "potion of levitation",
    ]

    UNIDENTIFIED_KEYWORDS = [
        "scroll",
        "potion",
        "wand",
        "spellbook",
        "ring",
        "amulet",
    ]

    PEACEFUL_NAMES = [
        "shopkeeper",
        "priest",
        "watchman",
        "guard",
        "oracle",
    ]

    def __init__(
        self,
        threshold: float = 1.0,
        stagnation_window: int = 50,
        min_stall_turns: int = 15,
    ):
        self.threshold = threshold
        self.stagnation_window = stagnation_window
        self.min_stall_turns = min_stall_turns
        # Ring buffer of progress frames: (turn, depth, score, explored_count)
        self.progress_history: deque[tuple[int, int, int, int]] = deque(maxlen=stagnation_window)
        self.stall_counters: dict[str, int] = {}
        self.last_depth = 1

    def reset(self):
        """Clears episode history."""
        self.progress_history.clear()
        self.stall_counters.clear()
        self.last_depth = 1

    def record_step(self, blstats: BottomLineStats, lvl_map: LevelMap | None = None):
        """Records a step observation for sliding-window macro-progress analysis."""
        explored = int(np.count_nonzero(lvl_map.visited > 0)) if lvl_map is not None else 0
        self.progress_history.append((blstats.turn, blstats.depth, blstats.score, explored))
        self.last_depth = blstats.depth

    def record_stall(self, task_name: str) -> int:
        """Increments consecutive stall count for a given task."""
        count = self.stall_counters.get(task_name, 0) + 1
        self.stall_counters[task_name] = count
        return count

    def clear_stall(self, task_name: str):
        """Resets consecutive stall count when progress occurs."""
        self.stall_counters.pop(task_name, None)

    def evaluate(
        self,
        blstats: BottomLineStats,
        chars: np.ndarray,
        lvl_map: LevelMap | None,
        inv_items: list[NormalizedItem],
        failed_task: str,
        message: str = "",
        adjacent_monster_names: list[str] | None = None,
        prayer_cooldown_remaining: int = 0,
    ) -> ImpasseEvaluation:
        """
        Computes the continuous composite cognitive impasse score S_deliberate.
        Returns detailed diagnostic verdict.
        """
        py, px = blstats.y, blstats.x
        consecutive_stalls = self.stall_counters.get(failed_task, 1)

        # Micro-oscillation Fast Reject: Stalls below min_stall_turns never trigger LLM
        if consecutive_stalls < self.min_stall_turns:
            return ImpasseEvaluation(
                should_trigger=False,
                score=0.0,
                stagnation_score=0.0,
                topology_score=0.0,
                affordance_score=0.0,
                dilemma_score=0.0,
                urgency_multiplier=1.0,
                diagnosis=f"Micro-oscillation ({consecutive_stalls}/{self.min_stall_turns} stalls) - System 1 symbolic perturbation active",
            )

        # -------------------------------------------------------------
        # 1. Macro-Stagnation Horizon (Phi_stagnation)
        # -------------------------------------------------------------
        phi_stagnation = 0.0
        if len(self.progress_history) >= min(20, self.stagnation_window):
            oldest = self.progress_history[0]
            newest = self.progress_history[-1]
            turn_delta = newest[0] - oldest[0]
            depth_delta = abs(newest[1] - oldest[1])
            score_delta = newest[2] - oldest[2]
            explored_delta = newest[3] - oldest[3]

            if turn_delta >= 20 and depth_delta == 0 and score_delta <= 0 and explored_delta <= 1:
                phi_stagnation = 1.0
            elif turn_delta >= 15 and depth_delta == 0 and explored_delta <= 3:
                phi_stagnation = 0.5

        # -------------------------------------------------------------
        # 2. Global Topological & Frontier Exhaustion (Phi_topology)
        # -------------------------------------------------------------
        phi_topology = 0.0
        if lvl_map is not None:
            # Check for unopened doors
            has_unopened_doors = bool(np.any((chars == ord("+")) & (lvl_map.visited == 0)))
            # Check for unvisited walkable tiles
            has_unvisited_tiles = bool(np.any(lvl_map.walkable & (lvl_map.visited == 0)))
            # Check for unsearched perimeter walls
            is_wall = (chars == ord("|")) | (chars == ord("-")) | (chars == 0) | (chars == ord(" "))
            has_wall_neighbor = np.zeros_like(is_wall)
            has_wall_neighbor[1:, :] |= is_wall[:-1, :]
            has_wall_neighbor[:-1, :] |= is_wall[1:, :]
            has_wall_neighbor[:, 1:] |= is_wall[:, :-1]
            has_wall_neighbor[:, :-1] |= is_wall[:, 1:]
            has_unsearched_walls = bool(np.any(lvl_map.walkable & (lvl_map.searched < 12) & has_wall_neighbor))

            # True topological exhaustion: No doors, no frontiers, all walls searched, stairs down missing
            if not has_unopened_doors and not has_unvisited_tiles and not has_unsearched_walls:
                phi_topology = 1.0
            elif not has_unopened_doors and not has_unvisited_tiles:
                # Frontiers explored, only wall searching remains
                phi_topology = 0.4
            else:
                # Normal frontiers or doors still exist: System 1 should explore them
                phi_topology = 0.0

        # -------------------------------------------------------------
        # 3. Latent Inventory Affordances (Phi_affordance)
        # -------------------------------------------------------------
        num_unidentified = 0
        num_tools = 0
        has_cursed_equipment = False

        for item in inv_items:
            raw_desc = item.raw_str.lower()
            if any(k in raw_desc for k in self.UNIDENTIFIED_KEYWORDS):
                num_unidentified += 1
            if any(k in raw_desc for k in self.ESCAPE_TOOL_KEYWORDS):
                num_tools += 1
            if item.equipped and item.buc_state == "CURSED":
                has_cursed_equipment = True

        prayer_ready = prayer_cooldown_remaining <= 0
        phi_affordance = min(
            1.0,
            (0.30 * min(num_unidentified, 3)) + (0.40 * min(num_tools, 2)) + (0.30 if prayer_ready else 0.0),
        )

        # -------------------------------------------------------------
        # 4. Qualitative Tactical Dilemma (Phi_dilemma)
        # -------------------------------------------------------------
        phi_dilemma = 0.0
        msg_lower = message.lower()

        # Check for peaceful blocking entity
        if adjacent_monster_names:
            for mname in adjacent_monster_names:
                if any(p in mname.lower() for p in self.PEACEFUL_NAMES):
                    phi_dilemma = max(phi_dilemma, 0.9)

        if "really attack" in msg_lower or "peaceful" in msg_lower:
            phi_dilemma = max(phi_dilemma, 0.9)

        if has_cursed_equipment:
            phi_dilemma = max(phi_dilemma, 0.6)

        # -------------------------------------------------------------
        # 5. Acute Survival Urgency Multiplier (U_urgency)
        # -------------------------------------------------------------
        urgency = 1.0
        if blstats.hunger_state >= HungerState.WEAK:
            urgency += 1.5
        elif blstats.hunger_state >= HungerState.HUNGRY:
            urgency += 0.3

        hp_ratio = float(blstats.hp) / float(max(1, blstats.max_hp))
        if hp_ratio <= 0.25:
            urgency += 1.2
        elif hp_ratio <= 0.50:
            urgency += 0.5

        if blstats.is_blind or blstats.is_confused or blstats.is_stunned:
            urgency += 0.5

        # -------------------------------------------------------------
        # Composite Score Calculation
        # -------------------------------------------------------------
        w_stagnation = 0.35
        w_topology = 0.35
        w_affordance = 0.30
        w_dilemma = 0.55

        base_score = (
            (w_stagnation * phi_stagnation)
            + (w_topology * phi_topology)
            + (w_affordance * phi_affordance)
            + (w_dilemma * phi_dilemma)
        )
        total_score = base_score * urgency

        should_trigger = total_score >= self.threshold

        diagnosis_parts = []
        if phi_stagnation > 0:
            diagnosis_parts.append(f"Stagnation={phi_stagnation:.1f}")
        if phi_topology > 0:
            diagnosis_parts.append(f"TopologyExhausted={phi_topology:.1f}")
        if phi_affordance > 0:
            diagnosis_parts.append(f"Affordance={phi_affordance:.1f}")
        if phi_dilemma > 0:
            diagnosis_parts.append(f"Dilemma={phi_dilemma:.1f}")
        diagnosis_parts.append(f"Urgency={urgency:.1f}x")
        diagnosis_str = f"Score={total_score:.2f} ({', '.join(diagnosis_parts)})"

        return ImpasseEvaluation(
            should_trigger=should_trigger,
            score=total_score,
            stagnation_score=phi_stagnation,
            topology_score=phi_topology,
            affordance_score=phi_affordance,
            dilemma_score=phi_dilemma,
            urgency_multiplier=urgency,
            diagnosis=diagnosis_str,
        )
