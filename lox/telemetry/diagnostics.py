"""
LOX Empirical Root Cause Diagnostic Engine: Multi-dimensional episode autopsy & failure classification.
Transforms raw tick telemetry, action streams, inventory, and death circumstances into high-precision
failure archetypes to eliminate superficial final-hit combat misclassification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FailureArchetype(str, Enum):
    """Canonical failure modes prioritized by causal precedence."""

    STALL_SECRET_DOOR = "STALL_SECRET_DOOR"
    STARVATION_FAINTING = "STARVATION_FAINTING"
    PING_PONG_OSCILLATION = "PING_PONG_OSCILLATION"
    ARMOR_DEFICIT = "ARMOR_DEFICIT"
    PREMATURE_PRAYER = "PREMATURE_PRAYER"
    PASSIVE_HAZARD_PARALYSIS = "PASSIVE_HAZARD_PARALYSIS"
    COMBAT_TACTICAL_SWARM = "COMBAT_TACTICAL_SWARM"
    COMBAT_FAST_PREDATOR = "COMBAT_FAST_PREDATOR"
    COMBAT_GENERAL = "COMBAT_GENERAL"
    MAX_TURNS_REACHED = "MAX_TURNS_REACHED"
    ABORTED_ZERO_PROGRESS = "ABORTED_ZERO_PROGRESS"


@dataclass(slots=True)
class EpisodeDiagnostic:
    """Comprehensive diagnostic profile for a single episode."""

    episode_id: str
    run_id: str
    depth: int
    max_depth: int
    turns: int
    killer: str
    ac_at_death: int
    primary_archetype: FailureArchetype
    confidence: float
    explanation: str
    turns_fainting: int = 0
    turns_weak: int = 0
    is_oscillating: bool = False
    has_body_armor: bool = False
    secret_door_searches: int = 0


@dataclass(slots=True)
class BatchDiagnosticSummary:
    """Aggregated empirical root cause metrics across an evaluation batch."""

    total_episodes: int
    avg_depth: float
    max_depth: int
    avg_turns: float
    archetype_counts: dict[str, int] = field(default_factory=dict)
    archetype_percentages: dict[str, float] = field(default_factory=dict)
    fainting_episode_count: int = 0
    exploration_stall_count: int = 0
    armor_deficit_count: int = 0
    oscillation_count: int = 0

    def format_markdown_table(self) -> str:
        """Formats a clean markdown table suitable for direct injection into LLM prompts."""
        lines = [
            "### Empirical Failure Archetype Breakdown",
            "| Root Cause Archetype | Episodes | % of Batch | Key Causal Factor |",
            "| :--- | :---: | :---: | :--- |",
        ]
        descriptions = {
            FailureArchetype.STALL_SECRET_DOOR.value: "Stalled on early level (>500t) without finding stairs; dead-end search loop",
            FailureArchetype.STARVATION_FAINTING.value: "Entered FAINTING (unconscious); killed while helpless by low-tier entities",
            FailureArchetype.ARMOR_DEFICIT.value: "Died at Depth >= 4 with AC >= 5; lacked body armor or unworn armor in inventory",
            FailureArchetype.PING_PONG_OSCILLATION.value: "Stuck in 2-tile ping-pong position oscillation for >= 40 turns",
            FailureArchetype.PREMATURE_PRAYER.value: "Prayed on 'HUNGRY' or HP>50%; prayer on cooldown during fatal emergency",
            FailureArchetype.PASSIVE_HAZARD_PARALYSIS.value: "Paralyzed by floating eye or killed by gas spore area blast",
            FailureArchetype.COMBAT_TACTICAL_SWARM.value: "Surrounded by >= 2 hostiles in open room without corridor retreat",
            FailureArchetype.COMBAT_FAST_PREDATOR.value: "Killed by predator with speed > 12 (ants, bees, foxes) in open area",
            FailureArchetype.COMBAT_GENERAL.value: "Legitimate combat fatality against high-tier monster at depth",
            FailureArchetype.MAX_TURNS_REACHED.value: "Exceeded 25,000 turn episode ceiling alive",
            FailureArchetype.ABORTED_ZERO_PROGRESS.value: "Aborted due to repeated zero-turn stalls",
        }
        for arch, count in sorted(
            self.archetype_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = self.archetype_percentages.get(arch, 0.0)
            desc = descriptions.get(arch, "Combat or environmental mortality")
            lines.append(f"| `{arch}` | {count} | {pct:.1f}% | {desc} |")
        return "\n".join(lines)


class RootCauseClassifier:
    """Classifies episodes into root cause archetypes from raw telemetry."""

    FAST_PREDATORS = {
        "soldier ant",
        "giant ant",
        "killer bee",
        "queen bee",
        "fox",
        "coyote",
        "dingo",
        "jaguar",
        "giant bat",
        "wolf",
    }

    PASSIVE_HAZARDS = {
        "floating eye",
        "gas spore",
        "brown mold",
        "yellow mold",
        "green slime",
        "freezing sphere",
        "flaming sphere",
        "shocking sphere",
    }

    @classmethod
    def classify(
        cls,
        ep_summary: dict[str, Any],
        ticks: list[dict[str, Any]] | None = None,
    ) -> EpisodeDiagnostic:
        ticks = ticks or []
        depth = int(ep_summary.get("final_depth") or ep_summary.get("depth") or 1)
        max_d = int(ep_summary.get("max_depth") or depth)
        turns = int(ep_summary.get("ep_turns") or ep_summary.get("turns") or 0)
        killer = str(ep_summary.get("killer") or "").lower()
        ac = int(
            ep_summary.get("ac_at_death")
            if ep_summary.get("ac_at_death") is not None
            else 10
        )
        inv_str = str(ep_summary.get("inventory_at_death") or "").lower()
        death_reason = str(ep_summary.get("death_reason") or "").lower()

        # Check ticks for rich state telemetry
        turns_fainting = sum(1 for t in ticks if t.get("hunger") == "FAINTING")
        turns_weak = sum(1 for t in ticks if t.get("hunger") == "WEAK")
        searches_count = int(ep_summary.get("searches") or 0)

        # Detect position oscillation in last 60 ticks
        is_oscillating = False
        if len(ticks) >= 30:
            recent_positions = [(t.get("y", -1), t.get("x", -1)) for t in ticks[-40:]]
            unique_pos = set(recent_positions)
            if len(unique_pos) <= 2 and len(recent_positions) >= 30:
                is_oscillating = True

        # Check body armor
        has_body_armor = any(
            piece in inv_str
            for piece in ("mail", "suit", "cuirass", "plate", "jacket", "leather armor")
            if "(being worn)" in inv_str or "armor" in inv_str
        )
        if ac <= 4:
            has_body_armor = True

        # 1. Max turns reached or aborted
        if "maxturnsreached" in death_reason or turns >= 24990:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.MAX_TURNS_REACHED,
                confidence=1.0,
                explanation=f"Reached turn limit ({turns} turns)",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        if "aborted" in death_reason:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.ABORTED_ZERO_PROGRESS,
                confidence=1.0,
                explanation="Aborted due to repeated zero-turn stall",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 2. Starvation / Fainting Check
        if (
            turns_fainting > 0
            or "starvation" in death_reason
            or "faint" in death_reason
        ):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.STARVATION_FAINTING,
                confidence=0.95,
                explanation=f"Fainted from hunger ({turns_fainting} ticks fainting); killed by {killer or 'starvation'} while unconscious",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 3. Position Oscillation Ping-Pong
        if is_oscillating:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.PING_PONG_OSCILLATION,
                confidence=0.90,
                explanation="Trapped in 2-tile ping-pong oscillation for >= 30 turns prior to death",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=True,
                has_body_armor=has_body_armor,
            )

        # 4. Level Exploration / Secret Door Stall Check
        turns_dl1 = int(ep_summary.get("turns_dl1") or 0)
        turns_dl2 = int(ep_summary.get("turns_dl2") or 0)
        if (
            (turns_dl1 >= 500 and max_d == 1)
            or (turns_dl2 >= 600 and max_d <= 2)
            or (turns >= 1200 and max_d <= 2 and searches_count >= 15)
        ):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.STALL_SECRET_DOOR,
                confidence=0.90,
                explanation=f"Stalled on DL {max_d} for {turns} turns searching dead ends without descending",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
                secret_door_searches=searches_count,
            )

        # 5. Passive Hazard Paralysis / Blast
        if any(h in killer for h in cls.PASSIVE_HAZARDS):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.PASSIVE_HAZARD_PARALYSIS,
                confidence=0.95,
                explanation=f"Hit or caught in blast of passive hazard ({killer})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 6. Armor Deficit at Depth >= 4
        if depth >= 4 and ac >= 5:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.ARMOR_DEFICIT,
                confidence=0.85,
                explanation=f"Died on DL {depth} with dangerous AC {ac} (unequipped body armor)",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 7. Fast Predator Attack
        if any(p in killer for p in cls.FAST_PREDATORS):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.COMBAT_FAST_PREDATOR,
                confidence=0.85,
                explanation=f"Killed by high-speed predator ({killer}, speed > 12)",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 8. Combat Swarm Check
        if ticks:
            last_tick = ticks[-1]
            hostiles_fov = int(last_tick.get("hostiles_in_fov") or 0)
            adj_count = (
                len(str(last_tick.get("adjacent_monsters") or "").split(","))
                if last_tick.get("adjacent_monsters")
                else 0
            )
            if hostiles_fov >= 2 or adj_count >= 2:
                return EpisodeDiagnostic(
                    episode_id=str(ep_summary.get("episode_id", "")),
                    run_id=str(ep_summary.get("run_id", "")),
                    depth=depth,
                    max_depth=max_d,
                    turns=turns,
                    killer=killer,
                    ac_at_death=ac,
                    primary_archetype=FailureArchetype.COMBAT_TACTICAL_SWARM,
                    confidence=0.80,
                    explanation=f"Overwhelmed by swarm ({hostiles_fov} hostiles in FOV, killer: {killer})",
                    turns_fainting=turns_fainting,
                    turns_weak=turns_weak,
                    is_oscillating=is_oscillating,
                    has_body_armor=has_body_armor,
                )

        # 9. General Combat Mortality
        return EpisodeDiagnostic(
            episode_id=str(ep_summary.get("episode_id", "")),
            run_id=str(ep_summary.get("run_id", "")),
            depth=depth,
            max_depth=max_d,
            turns=turns,
            killer=killer or "combat",
            ac_at_death=ac,
            primary_archetype=FailureArchetype.COMBAT_GENERAL,
            confidence=0.75,
            explanation=f"Combat mortality against {killer or 'hostile monster'} on DL {depth}",
            turns_fainting=turns_fainting,
            turns_weak=turns_weak,
            is_oscillating=is_oscillating,
            has_body_armor=has_body_armor,
        )

    @classmethod
    def summarize_batch(
        cls, diagnostics: list[EpisodeDiagnostic]
    ) -> BatchDiagnosticSummary:
        total = len(diagnostics)
        if total == 0:
            return BatchDiagnosticSummary(0, 0.0, 0, 0.0)

        avg_depth = float(np.mean([d.depth for d in diagnostics]))
        max_depth = int(np.max([d.max_depth for d in diagnostics]))
        avg_turns = float(np.mean([d.turns for d in diagnostics]))

        counts: dict[str, int] = {}
        for d in diagnostics:
            name = d.primary_archetype.value
            counts[name] = counts.get(name, 0) + 1

        percentages = {k: (v / total) * 100.0 for k, v in counts.items()}

        fainting_count = sum(
            1
            for d in diagnostics
            if d.turns_fainting > 0
            or d.primary_archetype == FailureArchetype.STARVATION_FAINTING
        )
        stall_count = sum(
            1
            for d in diagnostics
            if d.primary_archetype == FailureArchetype.STALL_SECRET_DOOR
        )
        armor_count = sum(
            1
            for d in diagnostics
            if d.primary_archetype == FailureArchetype.ARMOR_DEFICIT
        )
        osc_count = sum(
            1
            for d in diagnostics
            if d.is_oscillating
            or d.primary_archetype == FailureArchetype.PING_PONG_OSCILLATION
        )

        return BatchDiagnosticSummary(
            total_episodes=total,
            avg_depth=avg_depth,
            max_depth=max_depth,
            avg_turns=avg_turns,
            archetype_counts=counts,
            archetype_percentages=percentages,
            fainting_episode_count=fainting_count,
            exploration_stall_count=stall_count,
            armor_deficit_count=armor_count,
            oscillation_count=osc_count,
        )
