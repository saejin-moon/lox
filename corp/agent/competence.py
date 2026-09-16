"""
Agent Layer: Meta-Cognitive Competence Discovery Engine.
Autonomously measures persona capability profiles and calculates the competence score
C(theta) to select the optimal archetype for Mode B (MODE_COMPETENCE_SELECTION).
"""

from dataclasses import dataclass, field
import numpy as np

from corp.planner.persona import PersonaTraitVector


@dataclass
class PersonaPerformanceRecord:
    """Historical telemetry for an evaluated persona configuration."""
    role: str
    trait_vector: PersonaTraitVector
    survival_turns: list[int] = field(default_factory=list)
    dungeon_depths: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    fatal_errors: int = 0
    total_episodes: int = 0


class CompetenceEngine:
    """
    Measures capability profiles across character archetypes:
    C(theta) = w1 * (MedianDepth / D_max) + w2 * (MeanTurns / T_max) - w3 * (Violations / N_total)
    """

    def __init__(
        self,
        w_depth: float = 0.50,
        w_turns: float = 0.35,
        w_errors: float = 0.15,
        max_depth: float = 50.0,
        max_turns: float = 50000.0,
    ):
        self.w_depth = w_depth
        self.w_turns = w_turns
        self.w_errors = w_errors
        self.max_depth = max_depth
        self.max_turns = max_turns
        self.records: dict[str, PersonaPerformanceRecord] = {}

    def record_episode(
        self,
        role: str,
        trait_vector: PersonaTraitVector,
        turns: int,
        depth: int,
        score: int,
        had_fatal_error: bool = False,
    ):
        """Records an episode outcome for the given persona."""
        role_key = role.lower()
        if role_key not in self.records:
            self.records[role_key] = PersonaPerformanceRecord(
                role=role_key,
                trait_vector=trait_vector,
            )

        rec = self.records[role_key]
        rec.survival_turns.append(turns)
        rec.dungeon_depths.append(depth)
        rec.scores.append(score)
        rec.total_episodes += 1
        if had_fatal_error:
            rec.fatal_errors += 1

    def compute_competence(self, role: str) -> float:
        """
        Computes normalized competence score C(theta) in [0.0, 1.0].
        """
        role_key = role.lower()
        if role_key not in self.records or self.records[role_key].total_episodes == 0:
            return 0.5 # Prior expectation

        rec = self.records[role_key]
        med_depth = float(np.median(rec.dungeon_depths))
        mean_turns = float(np.mean(rec.survival_turns))
        error_rate = float(rec.fatal_errors) / float(rec.total_episodes)

        norm_depth = min(1.0, med_depth / self.max_depth)
        norm_turns = min(1.0, mean_turns / self.max_turns)

        competence = (
            (self.w_depth * norm_depth)
            + (self.w_turns * norm_turns)
            - (self.w_errors * error_rate)
        )
        return float(np.clip(competence, 0.0, 1.0))

    def select_optimal_persona(self) -> str:
        """
        Selects the persona archetype with the highest proven competence score.
        Defaults to 'valkyrie' (the gold standard melee tank) if unmeasured.
        """
        if not self.records:
            return "valkyrie"

        best_role = "valkyrie"
        best_score = -1.0
        for role in self.records:
            score = self.compute_competence(role)
            if score > best_score:
                best_score = score
                best_role = role

        return best_role
