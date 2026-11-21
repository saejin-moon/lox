"""Domain-scoped tests for LOX."""

from __future__ import annotations

from lox.ui.console import ConsoleUI


def test_compact_console_ui_rendering(capsys):
    console = ConsoleUI(use_color=False)
    console.banner(
        model="google/gemma-4-31b-it",
        provider="openrouter",
        db_path="data/lox.duckdb",
        workers=8,
        role="val-hum-law-fem",
    )
    console.gen_header(
        gen=1, mode="FUNC_MOD_ADD", target_episodes=1000, target_fns=["combat"]
    )
    console.tool_call(
        1, "query_db", "SELECT * FROM runs WHERE is_outlier_good = TRUE", "5 rows", 12.4
    )
    console.hypothesis_box(
        {
            "mode": "FUNC_MOD_ADD",
            "target_functions": ["combat"],
            "observation_from_db": "Outliers use stairs to heal",
            "mechanism": "Retreat to '<' when HP < 30%",
        }
    )
    console.progress(
        triggered_done=1000,
        triggered_target=1000,
        raw_done=1250,
        steps_per_sec=45000.0,
        avg_depth=4.25,
        max_depth=11,
        outliers=18,
    )
    console.batch_summary(
        {
            "episodes_recorded": 1000,
            "episodes_discarded": 250,
            "avg_depth": 4.25,
            "p90_depth": 7.0,
            "max_depth": 11,
            "avg_score": 1420.5,
            "avg_turns": 1850.0,
            "outlier_count": 18,
            "steps_per_sec": 45000.0,
            "wall_sec": 41.2,
        },
        {
            "passed": True,
            "mean_delta": 0.85,
            "ci_lower": 0.42,
            "ci_upper": 1.28,
            "wilcoxon_p": 0.00012,
            "wins": 410,
            "losses": 190,
            "ties": 400,
            "reason": "Significant gain on 1000 triggered seeds",
        },
    )
    captured = capsys.readouterr().out
    assert "LOX" in captured
    assert "Hard Code Nothing" in captured
    assert "FUNC_MOD_ADD" in captured
    assert "[T1]" in captured
    assert "PROMOTED"
