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

    # Test YAML and JSON serialization
    yaml_str = summary.format_yaml()
    assert "batch_metrics:" in yaml_str
    assert "ARMOR_DEFICIT: {count: 6, pct: 60.0}" in yaml_str
    assert "STALL_SECRET_DOOR: {count: 4, pct: 40.0}" in yaml_str

    json_str = summary.format_json()
    assert '"episodes": 10' in json_str
    assert '"ARMOR_DEFICIT"' in json_str

    # Test file saving
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        y_path = os.path.join(tmpdir, "diag.yaml")
        j_path = os.path.join(tmpdir, "diag.json")
        summary.save(y_path)
        summary.save(j_path)
        assert os.path.exists(y_path)
        assert os.path.exists(j_path)
        assert "batch_metrics:" in open(y_path).read()
        assert '"batch_metrics"' in open(j_path).read()
