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
    assert "PROMOTED"


def test_query_db_untruncated_long_strings_for_small_rowsets(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    db_file = str(tmp_path / "test_untruncated.duckdb")
    con = init_db(db_file)
    long_digest = "EVENT_START " + ("A" * 2500) + " EVENT_END"
    con.execute(
        "INSERT INTO expert_runs (run_id, turns, depth, score, summary_digest) VALUES (?, ?, ?, ?, ?)",
        ["exp_test_01", 100, 3, 200, long_digest],
    )
    con.close()

    tools = AuthorToolRegistry(db_path=db_file)
    out, count_str = tools.query_db(
        "SELECT summary_digest FROM expert_runs WHERE run_id = 'exp_test_01'"
    )
    assert count_str == "1 rows"
    # Verify the 2500-char string was NOT truncated down to 700 chars
    assert "EVENT_START" in out
    assert "EVENT_END" in out
    assert len(out) >= 2500


def test_dry_run_policy_behavioral_critic_detects_oscillation(tmp_path):
    from lox.author.tools import AuthorToolRegistry

    tools = AuthorToolRegistry(db_path=str(tmp_path / "db.duckdb"))
    tools.current_policy_code = ""

    # Policy that walks 5 steps East then endlessly ping-pongs North/South
    oscillating_code = """class Policy:
    def __init__(self):
        self.step_idx = 0
    def run(self, state):
        while True:
            self.step_idx += 1
            if self.step_idx <= 4:
                state = yield Action.E
            elif self.step_idx % 2 == 1:
                state = yield Action.N
            else:
                state = yield Action.S
"""
    out, status = tools.dry_run_policy(code=oscillating_code, max_turns=70, seed=42)
    assert "BEHAVIORAL WARNING" in out
    assert "oscillation detected" in out


def test_dry_run_policy_pilot_evaluation_and_revision_credits(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.runner.pool import PersistentEvalPool

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=40)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "db.duckdb"))
        baseline_runs = [
            {"seed": 42, "max_depth": 1, "score": 10},
            {"seed": 43, "max_depth": 1, "score": 15},
            {"seed": 44, "max_depth": 2, "score": 50},
        ]
        tools.configure_generation(
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=3,
            pilot_episodes=3,
        )

        # 1. Syntax error should NOT burn pilot revision credits
        bad_code = "def syntax_err(\n"
        err_out, err_status = tools.dry_run_policy(bad_code)
        assert err_status == "COMPILE_ERROR"
        assert tools.pilot_runs_used == 0

        # 2. Valid code runs pilot batch, increments pilot_runs_used to 1, and reports paired metrics
        valid_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        out, status = tools.dry_run_policy(valid_policy, max_turns=20)
        assert tools.pilot_runs_used == 1
        assert "PILOT BATCH 1/3 REPORT" in out
        assert "Paired vs Baseline:" in out
        assert "Revisions Remaining: 2 left" in out
    finally:
        pool.close()


def test_dry_run_policy_replay_run_id_fast_forward(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db
    from lox.env.actions import Action

    db_file = str(tmp_path / "replay_test.duckdb")
    con = init_db(db_file)
    # Simulate a run that took 10 Action.WAIT steps on seed 99
    sim_bytes = bytes([int(Action.WAIT)] * 10)
    con.execute(
        "INSERT INTO runs (run_id, batch_id, seed, steps, action_bytes) VALUES (?, ?, ?, ?, ?)",
        ["run_rec_99", "b0", 99, 10, sim_bytes],
    )
    con.close()

    tools = AuthorToolRegistry(db_path=db_file)
    test_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
"""
    out, status = tools.dry_run_policy(
        code=test_code,
        seed=1,  # seed=1 is ignored because replay_run_id provides seed=99
        max_turns=15,
        replay_run_id="run_rec_99",
        replay_steps=5,
    )
    assert status.startswith("PASSED")
    assert "Dry-Run PASSED" in out
