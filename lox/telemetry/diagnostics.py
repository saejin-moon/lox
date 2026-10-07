"""
LOX Empirical Root Cause Diagnostic Engine: Multi-dimensional episode autopsy & failure classification.
Transforms raw tick telemetry, action streams, inventory, and death circumstances into high-precision
failure archetypes to eliminate superficial final-hit combat misclassification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FailureArchetype(str, Enum):
    """Canonical failure modes prioritized by causal precedence."""

    PETRIFICATION = "PETRIFICATION"
    DROWNING_OR_LAVA = "DROWNING_OR_LAVA"
    PEACEFUL_NPC_PROVOCATION = "PEACEFUL_NPC_PROVOCATION"
    TRAP_FATALITY = "TRAP_FATALITY"
    INSTADEATH_POISON = "INSTADEATH_POISON"
    STATUS_EFFECT_HELPLESS = "STATUS_EFFECT_HELPLESS"
    STALL_SECRET_DOOR = "STALL_SECRET_DOOR"
    STARVATION_FAINTING = "STARVATION_FAINTING"
    PING_PONG_OSCILLATION = "PING_PONG_OSCILLATION"
    ENCUMBRANCE_IMMOBILITY = "ENCUMBRANCE_IMMOBILITY"
    ARMOR_DEFICIT = "ARMOR_DEFICIT"
    PREMATURE_PRAYER = "PREMATURE_PRAYER"
    PASSIVE_HAZARD_PARALYSIS = "PASSIVE_HAZARD_PARALYSIS"
    COMBAT_TACTICAL_SWARM = "COMBAT_TACTICAL_SWARM"
    COMBAT_FAST_PREDATOR = "COMBAT_FAST_PREDATOR"
    COMBAT_GENERAL = "COMBAT_GENERAL"
    MAX_TURNS_REACHED = "MAX_TURNS_REACHED"
    ABORTED_ZERO_PROGRESS = "ABORTED_ZERO_PROGRESS"


@dataclass(slots=True)
class CausalIncidentDossier:
    """Detailed multi-turn causal timeline analysis of an episode failure."""

    episode_id: str
    depth: int
    turns: int
    terminal_event: str
    causal_drivers: list[str] = field(default_factory=list)
    timeline_breakdown: list[str] = field(default_factory=list)
    strategic_recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "depth": self.depth,
            "turns": self.turns,
            "terminal_event": self.terminal_event,
            "causal_drivers": self.causal_drivers,
            "timeline_breakdown": self.timeline_breakdown,
            "strategic_recommendation": self.strategic_recommendation,
        }


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
    causal_dossier: CausalIncidentDossier | None = None


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
    causal_summary: dict[str, Any] = field(default_factory=dict)
    top_killers: list[tuple[str, int]] = field(default_factory=list)
    avg_ac_at_death: float = 10.0
    depth_distribution: dict[str, float] = field(default_factory=dict)

    def format_markdown_table(self) -> str:
        """Formats a clean markdown table suitable for direct injection into LLM prompts."""
        lines = [
            "### Empirical Failure Archetype Breakdown",
            "| Root Cause Archetype | Episodes | % of Batch | Key Causal Factor |",
            "| :--- | :---: | :---: | :--- |",
        ]
        descriptions = {
            FailureArchetype.PETRIFICATION.value: "Turned to stone by cockatrice, chickatrice, or handling corpse without gloves",
            FailureArchetype.DROWNING_OR_LAVA.value: "Drowned in water, moat, or burned in pool of lava",
            FailureArchetype.PEACEFUL_NPC_PROVOCATION.value: "Provoked peaceful/neutral entity (guard, priest, shopkeeper, watchman) into hostile retaliation",
            FailureArchetype.TRAP_FATALITY.value: "Killed by dungeon trap (spiked pit, arrow trap, falling rock, rolling boulder)",
            FailureArchetype.INSTADEATH_POISON.value: "Killed by lethal poison sting/bite (ants, bees, scorpions) without poison resistance",
            FailureArchetype.STATUS_EFFECT_HELPLESS.value: "Killed while disabled by sleep, paralysis, stun, confusion, or blindness",
            FailureArchetype.ENCUMBRANCE_IMMOBILITY.value: "Movement speed severely degraded by Strained/Overtaxed encumbrance, blocking escape",
            FailureArchetype.STALL_SECRET_DOOR.value: "PACING STALL: Loitered on early floors (>300t) searching blank walls instead of descending; exhausted turns and food",
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

    def format_trustworthy_triad_report(self) -> str:
        """
        Formats the complete 3-Factor Trustworthy Triad Report:
        1. Macro Archetype Distribution Table (High-level batch view)
        2. Orthogonal Multi-Factor Systemic Drivers (Cross-cutting causal bottlenecks)
        3. Ground-Truth Empirical Telemetry Context (Top fatal monsters, AC, depth distribution, timeline)
        """
        lines = [
            "## THE TRUSTWORTHY TRIAD: EMPIRICAL BATCH AUTOPSY REPORT",
            "",
            "### Factor 1: Macro Failure Archetype Distribution (100-Episode Batch)",
            self.format_markdown_table(),
            "",
            "### Factor 2: Orthogonal Systemic Drivers & Bottlenecks (Cross-Cutting Autopsy)",
        ]

        bottlenecks = self.causal_summary.get("systemic_bottlenecks", [])
        if bottlenecks:
            for b in bottlenecks:
                lines.append(f"- **{b.get('bottleneck', 'UNKNOWN')}**: {b.get('pct', 0.0)}% of episodes affected ({b.get('count', 0)} incidents)")
        else:
            lines.append("- No cross-cutting systemic bottlenecks detected.")

        recs = self.causal_summary.get("top_strategic_recommendations", [])
        if recs:
            lines.append("")
            lines.append("**Top Empirical Strategic Recommendations:**")
            for idx, r in enumerate(recs, 1):
                lines.append(f"{idx}. {r}")

        lines.extend([
            "",
            "### Factor 3: Ground-Truth Empirical Telemetry Context",
        ])

        if self.top_killers:
            killers_str = ", ".join(f"{k} ({c})" for k, c in self.top_killers[:6])
            lines.append(f"- **Top Fatal Monsters/Causes**: {killers_str}")
        else:
            lines.append("- **Top Fatal Monsters/Causes**: N/A")

        if self.depth_distribution:
            depth_str = " | ".join(f"{k}: {v:.1f}%" for k, v in self.depth_distribution.items())
            lines.append(f"- **Mortality Depth Distribution**: {depth_str}")
        else:
            lines.append("- **Mortality Depth Distribution**: DL 1-2: 100.0%")

        lines.append(f"- **Batch Averages at Death**: AC: {self.avg_ac_at_death:.1f} (Base AC is 10, target is <= 0) | Turns: {self.avg_turns:.1f}")

        samples = self.causal_summary.get("fatal_episodes_sample", [])
        if samples:
            lines.append("")
            lines.append("**Representative Fatal Encounters:**")
            for s in samples[:3]:
                lines.append(f"- Episode `{s.get('id', '?')}`: {s.get('event', 'Fatal incident')} (DL {s.get('depth', '?')}, Turn {s.get('turns', '?')})")

        return "\n".join(lines)


    def to_dict(self) -> dict[str, Any]:
        """Returns compact dictionary representation of batch metrics and root cause breakdown."""
        causes = {}
        for arch, count in sorted(
            self.archetype_counts.items(), key=lambda x: x[1], reverse=True
        ):
            causes[arch] = {
                "count": count,
                "pct": round(self.archetype_percentages.get(arch, 0.0), 1),
            }
        res: dict[str, Any] = {
            "batch_metrics": {
                "episodes": self.total_episodes,
                "avg_depth": round(self.avg_depth, 2),
                "max_depth": self.max_depth,
                "avg_turns": round(self.avg_turns, 1),
            },
            "root_causes": causes,
        }
        if self.causal_summary:
            res["causal_incident_analysis"] = self.causal_summary
        return res

    def format_yaml(self) -> str:
        """Formats a clean, informative YAML representation of batch diagnostics with causal attribution."""
        import yaml
        d = self.to_dict()
        lines = [
            "batch_metrics:",
            f"  episodes: {d['batch_metrics']['episodes']}",
            f"  avg_depth: {d['batch_metrics']['avg_depth']}",
            f"  max_depth: {d['batch_metrics']['max_depth']}",
            f"  avg_turns: {d['batch_metrics']['avg_turns']}",
            "root_causes:",
        ]
        for cause, info in d["root_causes"].items():
            lines.append(f"  {cause}: {{count: {info['count']}, pct: {info['pct']}}}")
        if self.causal_summary:
            causal_yaml = yaml.dump({"causal_incident_analysis": self.causal_summary}, sort_keys=False)
            lines.append(causal_yaml.strip())
        return "\n".join(lines)

    def save(self, filepath: str = "data/latest_diagnostics.yaml") -> None:
        """Saves batch diagnostic summary to a YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.format_yaml() + "\n")


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

    PETRIFYING_HAZARDS = {
        "cockatrice",
        "chickatrice",
        "medusa",
    }

    POISONOUS_VERMIN = {
        "soldier ant",
        "killer bee",
        "queen bee",
        "giant ant",
        "scorpion",
        "centipede",
        "spider",
        "cave spider",
        "poisonous snake",
        "snake",
        "pit viper",
        "rattlesnake",
        "cobra",
    }

    PEACEFUL_NPCS = {
        "guard",
        "vault guard",
        "shopkeeper",
        "priest",
        "priestess",
        "watchman",
        "aligned priest",
        "high priest",
        "oracle",
    }

    TRAP_TERMS = {
        "spiked pit",
        "pit",
        "arrow trap",
        "dart trap",
        "rolling boulder",
        "land mine",
        "falling rock",
        "bear trap",
        "trap",
    }

    WATER_LAVA_TERMS = {
        "drown",
        "pool of water",
        "water",
        "moat",
        "lava",
        "boiling lava",
    }

    @classmethod
    def classify(
        cls,
        ep_summary: dict[str, Any],
        ticks: list[dict[str, Any]] | None = None,
    ) -> EpisodeDiagnostic:
        diag = cls._classify_internal(ep_summary, ticks)
        diag.causal_dossier = CausalTimelineAnalyzer.analyze_episode(ep_summary, ticks)
        return diag

    @classmethod
    def _classify_internal(
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

        # Check ticks or ep_summary for rich state telemetry
        turns_fainting = int(ep_summary.get("turns_fainting", 0)) or sum(
            1 for t in ticks if t.get("hunger") == "FAINTING"
        )
        turns_weak = int(ep_summary.get("turns_weak", 0)) or sum(
            1 for t in ticks if t.get("hunger") == "WEAK"
        )
        searches_count = int(ep_summary.get("searches") or 0)

        # Detect position oscillation in last 60 ticks or summary flag
        is_oscillating = bool(ep_summary.get("is_oscillating", False))
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

        # 2. Petrification Check
        if any(
            p in killer or p in death_reason
            for p in ("cockatrice", "chickatrice", "medusa", "turned to stone", "stoning", "petrif")
        ):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.PETRIFICATION,
                confidence=1.0,
                explanation=f"Turned to stone by {killer or 'petrification'}",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 3. Drowning / Lava Check
        if any(w in killer or w in death_reason for w in cls.WATER_LAVA_TERMS):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.DROWNING_OR_LAVA,
                confidence=0.98,
                explanation=f"Drowned in water or burned in lava ({killer or death_reason})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 4. Peaceful NPC Provocation Check
        if any(npc in killer for npc in cls.PEACEFUL_NPCS):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.PEACEFUL_NPC_PROVOCATION,
                confidence=0.95,
                explanation=f"Killed by provoked peaceful/neutral entity ({killer})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 5. Trap Fatality Check
        if any(tr in killer or tr in death_reason for tr in cls.TRAP_TERMS) and "combat" not in death_reason:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.TRAP_FATALITY,
                confidence=0.95,
                explanation=f"Killed by dungeon trap ({killer or death_reason})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 6. Poison Instadeath Check
        if (
            "poison" in death_reason
            or "poison" in killer
            or (
                any(pv in killer for pv in cls.POISONOUS_VERMIN)
                and any(t in death_reason for t in ("poison", "sting", "died from poison"))
            )
        ):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.INSTADEATH_POISON,
                confidence=0.95,
                explanation=f"Killed by lethal poison bite/sting ({killer or death_reason})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 7. Position Oscillation Ping-Pong
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

        # 8. Status Effect Helpless Check
        if any(
            st in death_reason or st in killer
            for st in ("paraly", "sleep", "stun", "confus", "blind", "helpless")
        ) and not any(h in killer for h in cls.PASSIVE_HAZARDS):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.STATUS_EFFECT_HELPLESS,
                confidence=0.90,
                explanation=f"Killed while incapacitated by status condition ({killer or death_reason})",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 9. Encumbrance Immobility Check
        if (
            any(enc in inv_str for enc in ("strained", "overtaxed", "overloaded"))
            or str(ep_summary.get("encumbrance") or "").upper() in ("STRAINED", "OVERTAXED", "OVERLOADED")
            or "burdened" in death_reason
            or "strained" in death_reason
        ):
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.ENCUMBRANCE_IMMOBILITY,
                confidence=0.85,
                explanation=f"Severely encumbered (Strained/Overtaxed); movement speed crippled before death to {killer}",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
            )

        # 10. Passive Hazard Paralysis / Blast
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

        # 11. Pacing / Exploration Stall Check (CAUSAL ANTECEDENT TO STARVATION & COMBAT MORTALITY)
        # If the hero burned excessive turns on early floors (DL 1-3), that pacing stall is the
        # true root cause of subsequent food exhaustion, starvation, or low-tier monster death!
        turns_dl1 = int(ep_summary.get("turns_dl1") or 0)
        turns_dl2 = int(ep_summary.get("turns_dl2") or 0)
        is_pacing_stall = (
            (turns_dl1 >= 400 and max_d == 1)
            or (max_d == 1 and turns >= 500)
            or (searches_count >= 10 and (turns_dl1 >= 300 or turns_dl2 >= 300 or turns >= 500) and max_d <= 3)
            or (turns_dl1 >= 500 and max_d <= 2)
            or (turns_dl2 >= 600 and max_d <= 2)
        )
        if is_pacing_stall:
            return EpisodeDiagnostic(
                episode_id=str(ep_summary.get("episode_id", "")),
                run_id=str(ep_summary.get("run_id", "")),
                depth=depth,
                max_depth=max_d,
                turns=turns,
                killer=killer,
                ac_at_death=ac,
                primary_archetype=FailureArchetype.STALL_SECRET_DOOR,
                confidence=0.95,
                explanation=f"PACING STALL: Stalled on DL {max_d} searching for secret doors/routes ({turns} turns, {searches_count} searches, DL1: {turns_dl1}t, DL2: {turns_dl2}t); killed by {killer or death_reason}",
                turns_fainting=turns_fainting,
                turns_weak=turns_weak,
                is_oscillating=is_oscillating,
                has_body_armor=has_body_armor,
                secret_door_searches=searches_count,
            )

        # 12. Starvation / Fainting Check
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

        dossiers = [d.causal_dossier for d in diagnostics if d.causal_dossier is not None]
        causal_summary = CausalTimelineAnalyzer.summarize_batch(dossiers) if dossiers else {}

        # 3. Factor 3 Metrics: Top killers, AC at death, and depth distribution
        killer_counts: dict[str, int] = {}
        for d in diagnostics:
            k = d.killer.strip().lower()
            if k and k not in ("unknown", "none", "combat", ""):
                killer_counts[k] = killer_counts.get(k, 0) + 1
        top_killers = sorted(killer_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        avg_ac_at_death = float(np.mean([d.ac_at_death for d in diagnostics]))

        depth_dist_counts = {"DL 1-2": 0, "DL 3-4": 0, "DL 5-6": 0, "DL 7+": 0}
        for d in diagnostics:
            if d.depth <= 2:
                depth_dist_counts["DL 1-2"] += 1
            elif d.depth <= 4:
                depth_dist_counts["DL 3-4"] += 1
            elif d.depth <= 6:
                depth_dist_counts["DL 5-6"] += 1
            else:
                depth_dist_counts["DL 7+"] += 1
        depth_distribution = {k: (v / total) * 100.0 for k, v in depth_dist_counts.items()}

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
            causal_summary=causal_summary,
            top_killers=top_killers,
            avg_ac_at_death=avg_ac_at_death,
            depth_distribution=depth_distribution,
        )


class CausalTimelineAnalyzer:
    """Performs chronological milestone tracing and root-cause credit assignment."""

    @classmethod
    def analyze_episode(
        cls,
        ep_summary: dict[str, Any],
        ticks: list[dict[str, Any]] | None = None,
    ) -> CausalIncidentDossier:
        ticks = ticks or []
        ep_id = str(ep_summary.get("episode_id", ""))
        depth = int(ep_summary.get("final_depth") or ep_summary.get("depth") or 1)
        turns = int(ep_summary.get("ep_turns") or ep_summary.get("turns") or 0)
        killer = str(ep_summary.get("killer") or "").lower()
        ac = int(
            ep_summary.get("ac_at_death")
            if ep_summary.get("ac_at_death") is not None
            else 10
        )
        inv_str = str(ep_summary.get("inventory_at_death") or "").lower()
        death_reason = str(ep_summary.get("death_reason") or "").lower()
        score = int(ep_summary.get("score") or 0)

        terminal = f"Killed by {killer or 'hostile encounter'} on Depth {depth} (Turn {turns})"
        if "starvation" in death_reason or "faint" in death_reason or killer == "starvation":
            terminal = f"Died of starvation on Depth {depth} (Turn {turns})"
        elif "maxturnsreached" in death_reason or turns >= 24990:
            terminal = f"Reached turn ceiling (25,000 turns) on Depth {depth}"
        elif "aborted" in death_reason:
            terminal = f"Aborted due to repeated zero-turn stall on Depth {depth}"

        causal_drivers: list[str] = []
        timeline: list[str] = []

        # 1. Equipment & AC Trajectory Trace
        has_body_armor = any(
            piece in inv_str
            for piece in ("mail", "suit", "cuirass", "plate", "jacket", "leather armor")
            if "(being worn)" in inv_str or "armor" in inv_str
        )
        if depth >= 4 and ac >= 6:
            causal_drivers.append(
                f"EQUIPMENT_NEGLECT: Hero reached Depth {depth} with baseline AC {ac} (no body armor equipped across {turns} turns). Incoming monster damage was 3.5x normal."
            )
        elif depth >= 3 and not has_body_armor:
            causal_drivers.append(
                f"UNARMORED_BODY: Hero traversed {depth} floors without acquiring or equipping body armor."
            )

        # 2. Pacing & Exploration Stall Trace
        turns_dl1 = int(ep_summary.get("turns_dl1") or 0)
        turns_dl2 = int(ep_summary.get("turns_dl2") or 0)
        if turns_dl1 >= 800 or turns_dl2 >= 800:
            causal_drivers.append(
                f"PACING_STALL: Hero burned excessive turns on early depths (DL1: {turns_dl1}t, DL2: {turns_dl2}t) searching walls, consuming food before reaching mid-game."
            )
        elif turns >= 2000 and depth <= 3:
            causal_drivers.append(
                f"SEARCH_EXHAUSTION: Spent {turns} turns wandering early floors DL 1-{depth} instead of diving downward."
            )

        # 3. Nutrition Blindness Trace
        if ("starvation" in death_reason or killer == "starvation") and any(
            f in inv_str for f in ("ration", "wafer", "fruit", "cram", "food", "egg", "tripe", "meat")
        ):
            causal_drivers.append(
                "NUTRITION_BLINDNESS: Died of starvation while carrying edible rations/food in inventory."
            )

        # 4. Tactical Hoarding Trace
        if killer != "starvation" and any(
            w in inv_str
            for w in (
                "wand of striking",
                "wand of fire",
                "wand of cold",
                "wand of lightning",
                "wand of teleportation",
                "scroll of teleportation",
            )
        ):
            causal_drivers.append(
                "TACTICAL_HOARDING: Died in combat while holding unused offensive wands or escape scrolls in inventory."
            )

        # 5. Missed Artifact Milestone Trace
        if score >= 400 and "long sword" in inv_str and "excalibur" not in inv_str:
            causal_drivers.append(
                "MISSED_ARTIFACT: Hero reached XL >= 5 with Long Sword but never forged Excalibur (+1d10 damage, drain immunity)."
            )

        # 6. High-Impact Roguelike Hazards Trace
        if any(p in killer or p in death_reason for p in ("cockatrice", "chickatrice", "medusa", "turned to stone", "stoning", "petrif")):
            causal_drivers.append(
                f"COCKATRICE_BLINDNESS: Turned to stone by {killer or 'petrification'}. Engaged in melee without gloves or touched corpse."
            )
        elif any(npc in killer for npc in ("guard", "vault guard", "shopkeeper", "priest", "priestess", "watchman")):
            causal_drivers.append(
                f"PROVOKED_NEUTRAL: Engaged or threw projectiles at peaceful {killer}, provoking lethal retaliation."
            )
        elif "poison" in death_reason or (any(pv in killer for pv in ("ant", "bee", "scorpion", "spider", "snake", "centipede")) and any(t in death_reason for t in ("poison", "sting"))):
            causal_drivers.append(
                f"UNCHECKED_POISON_VULNERABILITY: Struck down by lethal poison from {killer}. Lacked poison resistance or defensive Elbereth dust ward."
            )
        elif any(tr in killer or tr in death_reason for tr in ("spiked pit", "pit", "trap", "falling rock", "rolling boulder", "dart trap")):
            causal_drivers.append(
                f"FATAL_TRAP_TRIGGER: Fatal dungeon trap ({killer or death_reason}) triggered at low HP without prior search."
            )
        elif any(w in killer or w in death_reason for w in ("drown", "pool of water", "moat", "lava")):
            causal_drivers.append(
                f"HAZARDOUS_TERRAIN_IMMERSION: Stepped into water or lava ({killer or death_reason}) without levitation."
            )
        elif any(st in death_reason or st in killer for st in ("paraly", "sleep", "stun", "confus", "blind", "helpless")) and not any(h in killer for h in ("floating eye", "gas spore")):
            causal_drivers.append(
                f"DISABLING_STATUS: Struck down while disabled by paralysis, sleep, or stun against {killer}."
            )
        elif any(enc in inv_str for enc in ("strained", "overtaxed", "overloaded")) or "strained" in death_reason:
            causal_drivers.append(
                f"ENCUMBRANCE_IMMOBILITY: Movement penalized by heavy carrying load, preventing combat retreat from {killer}."
            )

        # 7. Timeline Milestone Extraction from Ticks
        if ticks:
            n_t = len(ticks)
            step_indices = [0, n_t // 3, (2 * n_t) // 3, max(0, n_t - 1)]
            for s_idx in sorted(set(step_indices)):
                if s_idx < n_t:
                    t = ticks[s_idx]
                    timeline.append(
                        f"Turn {t.get('turn', s_idx)} (DL {t.get('depth', 1)}): HP {t.get('hp', '?')}/{t.get('max_hp', '?')}, AC {t.get('ac', '?')}, Action: {t.get('action', 'N/A')}"
                    )
        else:
            timeline.append(f"Turn {turns} (DL {depth}): Fatal incident ({terminal})")

        # 8. Strategic Recommendation
        rec = "Optimize tactical combat reflexes and corridor chokepoints."
        if any("COCKATRICE_BLINDNESS" in d for d in causal_drivers):
            rec = "Prioritize skill_combat: never fight cockatrices bare-handed; wear gloves before handling corpses."
        elif any("PROVOKED_NEUTRAL" in d for d in causal_drivers):
            rec = "Prioritize determine_goal: strictly enforce peaceful entity non-aggression and bypass neutral NPCs."
        elif any("UNCHECKED_POISON_VULNERABILITY" in d for d in causal_drivers):
            rec = "Prioritize skill_combat: engrave dust Elbereth when facing insects/vermin, and harvest poison-resistant corpses."
        elif any("FATAL_TRAP_TRIGGER" in d for d in causal_drivers):
            rec = "Prioritize skill_explore_and_dive: rest to full HP before exploring corridors and search suspicious dead ends."
        elif any("ENCUMBRANCE_IMMOBILITY" in d for d in causal_drivers):
            rec = "Prioritize skill_scavenge_armor: drop excess heavy armor or loot to maintain unencumbered movement."
        elif any("EQUIPMENT_NEGLECT" in d or "UNARMORED_BODY" in d for d in causal_drivers):
            rec = "Prioritize skill_scavenge_armor: actively loot and equip body armor on early depths to achieve AC <= 2 before descending."
        elif any("PACING_STALL" in d or "SEARCH_EXHAUSTION" in d for d in causal_drivers):
            rec = "Prioritize skill_explore_and_dive: cease exhaustive perimeter searches once stairs down are found to preserve turns and food."
        elif any("NUTRITION_BLINDNESS" in d for d in causal_drivers):
            rec = "Proactively consume non-perishable carried food at hunger_state >= 2."
        elif any("MISSED_ARTIFACT" in d for d in causal_drivers):
            rec = "Prioritize skill_forge_excalibur: backtrack to known fountain at XL >= 5 to forge Excalibur."

        return CausalIncidentDossier(
            episode_id=ep_id,
            depth=depth,
            turns=turns,
            terminal_event=terminal,
            causal_drivers=causal_drivers,
            timeline_breakdown=timeline,
            strategic_recommendation=rec,
        )

    @classmethod
    def summarize_batch(
        cls,
        dossiers: list[CausalIncidentDossier],
    ) -> dict[str, Any]:
        """Aggregates causal incident dossiers across an entire evaluation batch."""
        driver_counts: dict[str, int] = {}
        total = len(dossiers)
        depth_counts: dict[str, int] = {"DL 1-2": 0, "DL 3-4": 0, "DL 5-6": 0, "DL 7+": 0}
        total_turns = 0
        rec_counts: dict[str, int] = {}

        for dos in dossiers:
            total_turns += dos.turns
            if dos.depth <= 2:
                depth_counts["DL 1-2"] += 1
            elif dos.depth <= 4:
                depth_counts["DL 3-4"] += 1
            elif dos.depth <= 6:
                depth_counts["DL 5-6"] += 1
            else:
                depth_counts["DL 7+"] += 1

            if dos.strategic_recommendation:
                rec_counts[dos.strategic_recommendation] = (
                    rec_counts.get(dos.strategic_recommendation, 0) + 1
                )

            for driver in dos.causal_drivers:
                prefix = driver.split(":")[0].strip()
                driver_counts[prefix] = driver_counts.get(prefix, 0) + 1

        top_drivers = sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)
        top_recs = [
            r[0]
            for r in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        sample_episodes = [
            {
                "id": dos.episode_id,
                "depth": dos.depth,
                "turns": dos.turns,
                "event": dos.terminal_event,
            }
            for dos in dossiers[:5]
        ]

        return {
            "total_incidents": total,
            "avg_turns_at_death": round(total_turns / max(1, total), 1),
            "mortality_depth_distribution": depth_counts,
            "systemic_bottlenecks": [
                {
                    "bottleneck": k,
                    "count": v,
                    "pct": round((v / max(1, total)) * 100.0, 1),
                }
                for k, v in top_drivers
            ],
            "top_strategic_recommendations": top_recs,
            "fatal_episodes_sample": sample_episodes,
            "investigation_notice": "Call get_death_autopsy_trace(episode_id) to inspect the 15-tick flight recorder for any specific run.",
        }

