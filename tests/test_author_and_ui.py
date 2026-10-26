"""
Tests for AuthorAgent response extraction, macro invocation inside Policy,
GameState C-table lookups & status flags, and compact ConsoleUI rendering.
"""

from __future__ import annotations

from lox.author.agent import AuthorAgent
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_and_macros
from lox.ui.console import ConsoleUI


def test_author_yaml_and_code_extraction():
    sample_response = """
<think>Analyzing outlier runs...</think>
Here is my hypothesis based on the outlier good runs:

```yaml
mode: FUNC_MOD_ADD
target_functions:
  - handle_combat
  - flee_corridor
observation_from_db: "Outlier runs on Dlvl 6+ all triggered flee_corridor when HP < 40%."
mechanism: "Retreat toward corridor '#' tiles when surrounded by >= 2 hostiles."
```

And here is the updated implementation:

```python
def handle_combat(self, state):
    state = yield Action.WAIT
    return state

def flee_corridor(self, state):
    state = yield Action.SEARCH
    return state
```
"""
    hyp, code = AuthorAgent.extract_yaml_and_python(sample_response)
    assert hyp["mode"] == "FUNC_MOD_ADD"
    assert hyp["target_functions"] == ["handle_combat", "flee_corridor"]
    assert "Outlier runs" in hyp["observation_from_db"]
    assert "def handle_combat" in code
    assert "def flee_corridor" in code


def test_macros_called_from_policy_and_traced():
    macros_code = """
def macro_safe_search(state):
    \"\"\"Yields a search action and returns updated state.\"\"\"
    state = yield Action.SEARCH
    return state

def macro_is_low_hp(state, threshold=0.35):
    return (state.hp / max(1, state.max_hp)) < threshold
"""
    policy_code = """
class Policy:
    def run(self, state):
        while True:
            if macro_is_low_hp(state, 0.5):
                state = yield Action.WAIT
            else:
                state = yield from macro_safe_search(state)
"""
    compiled = compile_policy_and_macros(
        policy_code=policy_code, macros_code=macros_code
    )
    harness = NetHackHarness(max_steps=30)
    try:
        st = harness.reset(seed=77)
        runner = compiled.create_runner()
        for _ in range(5):
            act, active_fn = runner.next_action(st)
            st, done, _ = harness.step(act)
            if done:
                break
        assert runner.tracer.fn_calls.get("macro_is_low_hp", 0) >= 1
        assert runner.tracer.fn_calls.get("macro_safe_search", 0) >= 1
        assert runner.tracer.fn_turns.get("macro_safe_search", 0) >= 1
    finally:
        harness.close()


def test_gamestate_c_tables_and_status_properties():
    harness = NetHackHarness(max_steps=20)
    try:
        st = harness.reset(seed=99)
        assert isinstance(st.hunger_name, str)
        assert isinstance(st.encumbrance_name, str)
        assert st.is_blind is False
        assert st.is_confused is False
        assert st.is_stunned is False
        assert st.is_hallucinating is False
        assert st.is_food_poisoned is False
        assert st.is_levitating is False

        # Test static glyph lookup helpers
        hero_g = st.glyph_at(st.y, st.x)
        assert st.is_monster_glyph(hero_g) or st.is_cmap_glyph(hero_g) or hero_g >= 0
        minfo = st.monster_info(hero_g)
        if minfo is not None:
            assert minfo.name != ""
            assert minfo.speed >= 0
    finally:
        harness.close()


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
    assert "PROMOTED" in captured
