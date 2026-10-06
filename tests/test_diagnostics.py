"""
Unit tests for the LOX Empirical Root Cause Diagnostic Engine.
Validates multi-dimensional failure classification and batch reporting against synthetic telemetry.
"""

from __future__ import annotations

import os

from lox.telemetry.diagnostics import (
    EpisodeDiagnostic,
    FailureArchetype,
    RootCauseClassifier,
)


def test_classify_fainting_starvation():
    ep = {
        "episode_id": "test_ep_01",
        "final_depth": 2,
        "max_depth": 2,
        "ep_turns": 2400,
        "killer": "newt",
        "ac_at_death": 6,
        "death_reason": "Killed in combat",
    }
    ticks = [
        {"turn": 2390, "hunger": "NORMAL"},
        {"turn": 2395, "hunger": "WEAK"},
        {"turn": 2399, "hunger": "FAINTING"},
        {"turn": 2400, "hunger": "FAINTING"},
    ]
    diag = RootCauseClassifier.classify(ep, ticks=ticks)
    assert diag.primary_archetype == FailureArchetype.STARVATION_FAINTING
    assert diag.turns_fainting == 2
    assert "Fainted from hunger" in diag.explanation


def test_classify_stall_secret_door():
    ep = {
        "episode_id": "test_ep_02",
        "final_depth": 1,
        "max_depth": 1,
        "ep_turns": 1850,
        "turns_dl1": 1850,
        "killer": "grid bug",
        "ac_at_death": 6,
        "searches": 20,
        "death_reason": "Killed in combat",
    }
    diag = RootCauseClassifier.classify(ep, ticks=[])
    assert diag.primary_archetype == FailureArchetype.STALL_SECRET_DOOR
    assert "Stalled on DL 1" in diag.explanation


def test_classify_oscillation():
    ep = {
        "episode_id": "test_ep_03",
        "final_depth": 2,
        "max_depth": 2,
        "ep_turns": 800,
        "turns_dl1": 200,
        "turns_dl2": 600,
        "killer": "jackal",
        "ac_at_death": 5,
        "death_reason": "Killed in combat",
    }
    # 40 ticks alternating between (10, 15) and (10, 16)
    ticks = [
        {"y": 10, "x": 15 if i % 2 == 0 else 16, "hunger": "NORMAL"} for i in range(40)
    ]
    diag = RootCauseClassifier.classify(ep, ticks=ticks)
    assert diag.primary_archetype == FailureArchetype.PING_PONG_OSCILLATION
    assert diag.is_oscillating is True


def test_classify_armor_deficit():
    ep = {
        "episode_id": "test_ep_04",
        "final_depth": 6,
        "max_depth": 6,
        "ep_turns": 1900,
        "turns_dl1": 150,
        "turns_dl2": 150,
        "killer": "hill orc",
        "ac_at_death": 6,
        "death_reason": "Killed in combat",
    }
    diag = RootCauseClassifier.classify(ep, ticks=[])
    assert diag.primary_archetype == FailureArchetype.ARMOR_DEFICIT
    assert "dangerous AC 6" in diag.explanation


def test_classify_fast_predator():
    ep = {
        "episode_id": "test_ep_05",
        "final_depth": 3,
        "max_depth": 3,
        "ep_turns": 600,
        "killer": "soldier ant",
        "ac_at_death": 2,
        "death_reason": "Killed in combat",
    }
    diag = RootCauseClassifier.classify(ep, ticks=[])
    assert diag.primary_archetype == FailureArchetype.COMBAT_FAST_PREDATOR
    assert "speed > 12" in diag.explanation


def test_classify_passive_hazard():
    ep = {
        "episode_id": "test_ep_06",
        "final_depth": 4,
        "max_depth": 4,
        "ep_turns": 1100,
        "killer": "floating eye",
        "ac_at_death": 2,
        "death_reason": "Killed in combat",
    }
    diag = RootCauseClassifier.classify(ep, ticks=[])
    assert diag.primary_archetype == FailureArchetype.PASSIVE_HAZARD_PARALYSIS


def test_summarize_batch():
    diags = [
        EpisodeDiagnostic(
            episode_id=f"e_{i}",
            run_id="run_01",
            depth=1 if i < 4 else 6,
            max_depth=1 if i < 4 else 6,
            turns=1500,
            killer="test",
            ac_at_death=6,
            primary_archetype=FailureArchetype.STALL_SECRET_DOOR
            if i < 4
            else FailureArchetype.ARMOR_DEFICIT,
            confidence=0.9,
            explanation="test",
        )
        for i in range(10)
    ]
    summary = RootCauseClassifier.summarize_batch(diags)
    assert summary.total_episodes == 10
    assert summary.archetype_counts[FailureArchetype.STALL_SECRET_DOOR.value] == 4
    assert summary.archetype_counts[FailureArchetype.ARMOR_DEFICIT.value] == 6
    assert (
        summary.archetype_percentages[FailureArchetype.STALL_SECRET_DOOR.value] == 40.0
    )

    table_md = summary.format_markdown_table()
    assert "Empirical Failure Archetype Breakdown" in table_md
    assert "`STALL_SECRET_DOOR`" in table_md
    assert "`ARMOR_DEFICIT`" in table_md

    # Test to_dict and YAML serialization
    d = summary.to_dict()
    assert d["batch_metrics"]["episodes"] == 10
    assert d["root_causes"]["ARMOR_DEFICIT"]["count"] == 6

    yaml_str = summary.format_yaml()
    assert "batch_metrics:" in yaml_str
    assert "ARMOR_DEFICIT: {count: 6, pct: 60.0}" in yaml_str
    assert "STALL_SECRET_DOOR: {count: 4, pct: 40.0}" in yaml_str

    # Test YAML file saving
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        y_path = os.path.join(tmpdir, "diag.yaml")
        summary.save(y_path)
        assert os.path.exists(y_path)
        assert "batch_metrics:" in open(y_path).read()


def test_causal_timeline_analyzer():
    from lox.telemetry.diagnostics import CausalTimelineAnalyzer

    ep_summary = {
        "episode_id": "test_ep_01",
        "depth": 6,
        "turns": 3500,
        "killer": "soldier ant",
        "ac_at_death": 7,
        "inventory_at_death": "a +1 long sword (weapon in hand), a +3 small shield (being worn)",
        "death_reason": "killed by a soldier ant",
        "score": 600,
        "turns_dl1": 1200,
        "turns_dl2": 900,
    }
    ticks = [
        {"turn": 100, "depth": 1, "hp": 15, "max_hp": 15, "ac": 7, "action": "search"},
        {"turn": 1500, "depth": 2, "hp": 22, "max_hp": 22, "ac": 7, "action": "search"},
        {"turn": 3400, "depth": 6, "hp": 35, "max_hp": 35, "ac": 7, "action": "melee_attack_hostile"},
    ]

    dossier = CausalTimelineAnalyzer.analyze_episode(ep_summary, ticks)
    assert dossier.episode_id == "test_ep_01"
    assert dossier.depth == 6
    assert any("EQUIPMENT_NEGLECT" in d for d in dossier.causal_drivers)
    assert any("PACING_STALL" in d for d in dossier.causal_drivers)
    assert any("MISSED_ARTIFACT" in d for d in dossier.causal_drivers)
    assert "Prioritize skill_scavenge_armor" in dossier.strategic_recommendation

    # Test batch summarization
    batch_summary = CausalTimelineAnalyzer.summarize_batch([dossier])
    assert batch_summary["total_incidents"] == 1
    assert any(b["bottleneck"] == "EQUIPMENT_NEGLECT" for b in batch_summary["systemic_bottlenecks"])


def test_classify_new_archetypes():
    from lox.telemetry.diagnostics import CausalTimelineAnalyzer

    # 1. Petrification
    ep_pet = {
        "episode_id": "ep_pet",
        "depth": 8,
        "turns": 2000,
        "killer": "chickatrice",
        "death_reason": "turned to stone",
    }
    diag_pet = RootCauseClassifier.classify(ep_pet)
    assert diag_pet.primary_archetype == FailureArchetype.PETRIFICATION
    assert any("COCKATRICE_BLINDNESS" in d for d in diag_pet.causal_dossier.causal_drivers)

    # 2. Poison Instadeath
    ep_poi = {
        "episode_id": "ep_poi",
        "depth": 3,
        "turns": 1100,
        "killer": "killer bee",
        "death_reason": "died of poison from a killer bee's sting",
    }
    diag_poi = RootCauseClassifier.classify(ep_poi)
    assert diag_poi.primary_archetype == FailureArchetype.INSTADEATH_POISON
    assert any("UNCHECKED_POISON_VULNERABILITY" in d for d in diag_poi.causal_dossier.causal_drivers)

    # 3. Peaceful NPC Provocation
    ep_peace = {
        "episode_id": "ep_peace",
        "depth": 5,
        "turns": 1500,
        "killer": "vault guard",
        "death_reason": "killed by a guard",
    }
    diag_peace = RootCauseClassifier.classify(ep_peace)
    assert diag_peace.primary_archetype == FailureArchetype.PEACEFUL_NPC_PROVOCATION
    assert any("PROVOKED_NEUTRAL" in d for d in diag_peace.causal_dossier.causal_drivers)

    # 4. Trap Fatality
    ep_trap = {
        "episode_id": "ep_trap",
        "depth": 4,
        "turns": 900,
        "killer": "spiked pit",
        "death_reason": "fell into a spiked pit",
    }
    diag_trap = RootCauseClassifier.classify(ep_trap)
    assert diag_trap.primary_archetype == FailureArchetype.TRAP_FATALITY
    assert any("FATAL_TRAP_TRIGGER" in d for d in diag_trap.causal_dossier.causal_drivers)

    # 5. Drowning or Lava
    ep_drown = {
        "episode_id": "ep_drown",
        "depth": 7,
        "turns": 2500,
        "killer": "pool of water",
        "death_reason": "drowned in a pool of water",
    }
    diag_drown = RootCauseClassifier.classify(ep_drown)
    assert diag_drown.primary_archetype == FailureArchetype.DROWNING_OR_LAVA
    assert any("HAZARDOUS_TERRAIN_IMMERSION" in d for d in diag_drown.causal_dossier.causal_drivers)

    # 6. Status Effect Helpless
    ep_status = {
        "episode_id": "ep_status",
        "depth": 4,
        "turns": 1200,
        "killer": "orc",
        "death_reason": "killed while sleeping",
    }
    diag_status = RootCauseClassifier.classify(ep_status)
    assert diag_status.primary_archetype == FailureArchetype.STATUS_EFFECT_HELPLESS
    assert any("DISABLING_STATUS" in d for d in diag_status.causal_dossier.causal_drivers)

    # 7. Encumbrance Immobility
    ep_enc = {
        "episode_id": "ep_enc",
        "depth": 3,
        "turns": 800,
        "killer": "rothe",
        "inventory_at_death": "a plate mail (strained)",
        "death_reason": "killed in combat",
    }
    diag_enc = RootCauseClassifier.classify(ep_enc)
    assert diag_enc.primary_archetype == FailureArchetype.ENCUMBRANCE_IMMOBILITY
    assert any("ENCUMBRANCE_IMMOBILITY" in d for d in diag_enc.causal_dossier.causal_drivers)

