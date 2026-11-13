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


def test_author_agent_auto_pilot_dispatch(tmp_path, monkeypatch):
    from lox.db.schema import init_db
    from lox.runner.pool import PersistentEvalPool

    db_path = str(tmp_path / "test_ap.duckdb")
    con = init_db(db_path)
    con.close()

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=20)
    try:
        agent = AuthorAgent(provider="openrouter", db_path=db_path)
        baseline_runs = [
            {"seed": 10, "max_depth": 0, "score": 0},
            {"seed": 11, "max_depth": 0, "score": 0},
        ]

        turn_count = 0

        def mock_call_llm_api(messages, allow_tools=True):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                # LLM outputs YAML + code directly without calling dry_run_policy
                return {
                    "role": "assistant",
                    "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
observation_from_db: "Baseline exploration"
mechanism: "Wait actions"
```

```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
```""",
                    "tool_calls": [],
                }
            else:
                # LLM responds after pilot feedback
                return {
                    "role": "assistant",
                    "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
observation_from_db: "Revised exploration"
mechanism: "Wait and search"
```

```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
```""",
                    "tool_calls": [],
                }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm_api)

        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT",
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=3,
            pilot_episodes=2,
            max_tool_turns=4,
        )

        # Auto-pilot should have automatically executed the candidate code
        assert agent.tools.pilot_runs_used >= 1
        assert "class Policy" in proposal.full_policy_code
    finally:
        pool.close()


def test_real_state_macro_verification(tmp_path):
    from lox.author.tools import AuthorToolRegistry

    tools = AuthorToolRegistry(db_path=str(tmp_path / "test.duckdb"))

    # 1. Valid macro should execute on real states and return PASSED (macro)
    valid_macro = """
def get_walkable_mask(state):
    return state.chars == ord('.')

def step_wait(state):
    yield Action.WAIT

def is_safe_corpse(obj):
    return obj.is_corpse and 'poison' not in obj.name.lower()

def execute_move(state, dy, dx):
    target_y = state.y + dy
    return DIR_DELTAS.get((dy, dx), Action.WAIT)

def conditional_generator(state):
    for o in state.visible_objects():
        if o.is_corpse:
            yield seq(Action.EAT, 'y')
"""
    msg, status = tools.dry_run_policy(code=valid_macro, is_macro=True)
    assert status == "PASSED (macro)"
    assert "Real-State Macro Verification PASSED" in msg

    # 2. Broken macro with invalid attribute lookup should fail with RUNTIME_ERROR
    broken_macro = """
def bad_macro(state):
    return state.non_existent_attribute_xyz
"""
    msg, status = tools.dry_run_policy(code=broken_macro, is_macro=True)
    assert status == "RUNTIME_ERROR"
    assert "non_existent_attribute_xyz" in msg

    # 3. Invalid generator yield should be caught as INVALID_YIELD
    bad_generator = """
def bad_gen(state):
    yield 123456789  # Not a valid Action / sequence
"""
    msg, status = tools.dry_run_policy(code=bad_generator, is_macro=True)
    assert status == "INVALID_YIELD"


def test_gen0_full_write_pilot_verification(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.runner.pool import PersistentEvalPool

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=15)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "gen0.duckdb"))
        # In Gen 0 FULL_WRITE, baseline_runs is empty
        tools.configure_generation(
            pool=pool,
            baseline_runs=[],
            max_pilot_runs=3,
            pilot_episodes=2,
        )

        valid_starter_policy = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        msg, status = tools.dry_run_policy(valid_starter_policy)
        assert status.startswith("PASSED")
        assert "Gen 0 Baseline:" in status
        assert "GEN 0 BASELINE VERIFIED" in msg
        assert tools.last_passed_code == valid_starter_policy
        assert tools.last_pilot_improved is True
    finally:
        pool.close()


def test_scientific_investigation_protocol_and_failure_hints(tmp_path, monkeypatch):
    from lox.author.prompts import build_user_prompt
    from lox.author.tools import AuthorToolRegistry
    from lox.runner.pool import PersistentEvalPool

    # 1. Prompt check: FUNC_MOD_ADD mandates empirical research on Turn 1
    prompt = build_user_prompt(
        mode="FUNC_MOD_ADD",
        generation=2,
        current_policy="class Policy:\n    pass\n",
        macros_docs="",
        batch_digest="Sample digest",
    )
    assert "MANDATORY Empirical Research (Turn 1)" in prompt
    assert "DO NOT OUTPUT CODE ON TURN 1" in prompt
    assert "query_wiki" in prompt
    assert "query_db" in prompt

    # 2. Pilot failure report includes diagnostic research recommendations
    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "diag.duckdb"))
        # Baseline with higher depth so candidate policy fails
        baseline_runs = [
            {"seed": 100, "max_depth": 5, "score": 500},
            {"seed": 101, "max_depth": 5, "score": 500},
        ]
        tools.configure_generation(
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=2,
            pilot_episodes=2,
        )

        failing_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        msg, status = tools.dry_run_policy(failing_code)
        assert status.startswith("PILOT_FAILED")
        assert "MANDATORY INVESTIGATION PROTOCOL" in msg
        assert "query_wiki" in msg
        assert "query_db" in msg
        assert tools.last_pilot_improved is False

        # 3. Agent synthesize sends scientific investigation guidance on pilot failure
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "diag.duckdb"),
            pilot_fallback="abort",
        )
        received_messages = []

        def mock_call_llm(messages, allow_tools=True):
            received_messages.extend(messages)
            if len(received_messages) <= 3:
                # Turn 1: LLM emits code that fails
                return {
                    "role": "assistant",
                    "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'test'\nmechanism: 'test'\n```\n```python\n{failing_code}```",
                    "tool_calls": [],
                }
            else:
                # Turn 2: LLM receives failure guidance
                return {
                    "role": "assistant",
                    "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'fixed'\nmechanism: 'fixed'\n```\n```python\n{failing_code}```",
                    "tool_calls": [],
                }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)
        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy=failing_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=2,
            pilot_episodes=2,
        )
        assert proposal.aborted is True
        assert "PILOT_GATE_ABORT" in proposal.abort_reason

        user_msgs = [m["content"] for m in received_messages if m.get("role") == "user"]
        failure_guidance = [m for m in user_msgs if "🛑 PILOT REVISION FAILED" in m]
        assert len(failure_guidance) > 0, (
            "Agent must send scientific investigation guidance on pilot failure"
        )
        assert "DO NOT BLINDLY GUESS CODE EDITS!" in failure_guidance[0]
        assert "query_wiki" in failure_guidance[0]
        assert "query_db" in failure_guidance[0]
    finally:
        pool.close()


def test_pilot_fallback_best_does_not_abort_on_non_negative_delta(
    tmp_path, monkeypatch
):
    from lox.runner.pool import PersistentEvalPool

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "fallback.duckdb"),
            pilot_fallback="best",
        )
        neutral_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""

        def mock_call_llm(messages, allow_tools=True):
            return {
                "role": "assistant",
                "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'test'\nmechanism: 'test'\n```\n```python\n{neutral_code}```",
                "tool_calls": [],
            }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)
        # Baseline with equal performance so delta is non-negative (>= 0.0)
        baseline_runs = [
            {"seed": 100, "max_depth": 1, "score": 0},
            {"seed": 101, "max_depth": 1, "score": 0},
        ]
        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy=neutral_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=1,
            pilot_episodes=2,
        )
        assert proposal.aborted is False
        assert proposal.full_policy_code != ""
    finally:
        pool.close()


def test_pass4_system_prompt_restructuring_and_turn1_gating(tmp_path, monkeypatch):
    """
    Pass 4 Verification:
    1. System prompt contains Empirical Differential Diagnostics with expert_ticks queries,
       Pre-Submission Invariant Checklist, and pruned action working set routing to query_policy_docs("actions").
    2. User prompt enforces Binding Constraint rule and Causal Hypothesis schema across modes.
    3. extract_yaml_and_python enforces Causal Hypothesis schema fields.
    4. AuthorAgent.synthesize intercepts Turn-1 code generation when zero tools were called
       and enforces Phase 1 research without executing auto-pilot.
    """
    from lox.author.agent import AuthorAgent
    from lox.author.prompts import build_system_prompt, build_user_prompt
    from lox.author.tools import AuthorToolRegistry

    # 1. System Prompt Verification
    sys_prompt = build_system_prompt(str(tmp_path / "test.duckdb"))
    assert "Empirical Differential Diagnostics" in sys_prompt
    assert "expert_ticks" in sys_prompt
    assert "champion_runs" in sys_prompt
    assert "SELECT 'champion_failures' AS source" in sys_prompt
    assert "Pre-Submission Invariant Checklist" in sys_prompt
    assert "Generator Invariant" in sys_prompt
    assert "Passability" in sys_prompt
    assert 'query_policy_docs("actions")' in sys_prompt

    # Verify query_policy_docs("actions") tool route
    tools = AuthorToolRegistry(db_path=str(tmp_path / "test.duckdb"))
    action_docs, summary = tools.query_policy_docs("actions")
    assert "Action.SEARCH" in action_docs
    assert "Action.KICK" in action_docs
    assert "Action.PRAY" in action_docs
    assert summary == "action docs"

    # 2. User Prompt Verification across modes
    for mode in ("FUNC_MOD_ADD", "STRUCTURAL_REFACTOR", "FULL_WRITE"):
        user_prompt = build_user_prompt(
            mode=mode,
            generation=1,
            current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT\n",
            macros_docs="",
            batch_digest="Batch summary: 75% died to starvation.",
        )
        assert (
            "The Binding Constraint Rule" in user_prompt
            or "binding_constraint" in user_prompt
        )
        assert "binding_constraint:" in user_prompt
        assert "verified_c_rule:" in user_prompt
        assert "state_persistence:" in user_prompt
        assert "mechanism:" in user_prompt
        assert "target_functions:" in user_prompt

    # 3. Hypothesis Extraction Schema Enforcement
    sample_llm_output = """
```yaml
mode: FUNC_MOD_ADD
target_functions: [explore_step]
observation_from_db: "70% died from wall stalling"
mechanism: "Check get_passable_mask"
```
```python
def explore_step(self, state):
    state = yield Action.SEARCH
    return state
```
"""
    hyp, code = AuthorAgent.extract_yaml_and_python(sample_llm_output)
    assert hyp["target_functions"] == ["explore_step"]
    assert hyp["binding_constraint"] == "70% died from wall stalling"
    assert "verified_c_rule" in hyp
    assert hyp["mechanism"] == "Check get_passable_mask"
    assert "state_persistence" in hyp
    assert "def explore_step" in code

    # 4. Turn-1 Mandatory Investigation Gating in AuthorAgent.synthesize
    agent = AuthorAgent(provider="openrouter", db_path=str(tmp_path / "test.duckdb"))
    turn_counter = 0
    received_user_prompts = []

    def mock_call_llm_turn1_gate(messages, allow_tools=True):
        nonlocal turn_counter
        turn_counter += 1
        for m in messages:
            if m.get("role") == "user":
                received_user_prompts.append(m.get("content", ""))

        if turn_counter == 1:
            # Emits code immediately on Turn 1 without calling query_db or query_wiki
            return {
                "role": "assistant",
                "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
binding_constraint: "Stalling"
verified_c_rule: "C door rule"
mechanism: "Kick doors"
state_persistence: "self.tried"
```
```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
```""",
                "tool_calls": [],
            }
        else:
            # Turn 2: returns a response after gating
            return {
                "role": "assistant",
                "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
binding_constraint: "Stalling"
verified_c_rule: "C door rule"
mechanism: "Kick doors"
state_persistence: "self.tried"
```
```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
```""",
                "tool_calls": [],
            }

    monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm_turn1_gate)

    proposal = agent.synthesize(
        mode="FUNC_MOD_ADD",
        generation=1,
        current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT\n",
        max_tool_turns=3,
    )

    # Verify that Turn 1 Phase 1 enforcement intercepted Turn 1 and auto-pilot was blocked on Turn 1
    assert any("🛑 PHASE 1 ENFORCEMENT" in p for p in received_user_prompts)
    assert turn_counter >= 2
    assert "class Policy" in proposal.full_policy_code


def test_pilot_stops_immediately_after_pass(tmp_path):
    """Verify that dry_run_policy returns cached pass without burning extra pilot runs."""
    from lox.author.tools import AuthorToolRegistry
    from lox.lang.validator import is_ast_equivalent
    from lox.runner.pool import PersistentEvalPool

    c1 = "class Policy:\n    def run(self, state):\n        while True:\n            state = yield Action.WAIT\n"
    c2 = "class Policy:\n    def run(self, state):\n        # Added comment\n        while True:\n            state = yield Action.WAIT\n"
    assert is_ast_equivalent(c1, c2)

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "stop_after_pass.duckdb"))
        tools.configure_generation(
            pool=pool,
            baseline_runs=[],
            max_pilot_runs=3,
            pilot_episodes=2,
        )

        msg, status = tools.dry_run_policy(c1)
        assert status.startswith("PASSED")
        assert tools.last_pilot_improved is True
        assert tools.pilot_runs_used == 1

        # Calling again with AST-equivalent code must return cached pass without burning worker runs
        msg2, status2 = tools.dry_run_policy(c2)
        assert status2.startswith("PASSED (cached")
        assert tools.pilot_runs_used == 1
    finally:
        pool.close()


def test_synthesis_logs_persistence(tmp_path):
    """Verify that BatchDatabaseLogger persists and retrieves synthesis_logs with reasoning and diagnostics."""
    from lox.db.logger import BatchDatabaseLogger

    db_p = str(tmp_path / "test_syn.duckdb")
    logger = BatchDatabaseLogger(db_path=db_p)

    logger.save_synthesis_log(
        log_id="syn_001_1_0_test",
        generation=1,
        batch_id="gen_001_func_mod_add",
        mode="FUNC_MOD_ADD",
        turn_index=1,
        turn_type="RESEARCH_TOOL",
        repair_idx=0,
        model="test-model",
        thinking_content="I should check expert runs for staircase descent.",
        message_content="Querying expert runs.",
        tool_calls_json='[{"name": "query_db", "args": {"sql": "SELECT 1"}}]',
        tool_results_json='[{"name": "query_db", "output": "1"}]',
        candidate_code="def explore(self, state):\n    return Action.WAIT\n",
        skill_name="navigation",
        skill_code="",
        hypothesis_yaml="mode: FUNC_MOD_ADD\ntarget_functions: [explore]\n",
        status="TOOL_CALLS",
        error_diagnostic="",
        prompt_sent="Investigate the failure causes.",
        prompt_tokens=150,
        completion_tokens=50,
        reasoning_tokens=25,
        wall_ms=120.5,
    )

    logs = logger.get_recent_synthesis_logs(generation=1)
    assert len(logs) == 1
    row = logs[0]
    assert row["log_id"] == "syn_001_1_0_test"
    assert row["generation"] == 1
    assert row["mode"] == "FUNC_MOD_ADD"
    assert row["turn_type"] == "RESEARCH_TOOL"
    assert (
        row["thinking_content"] == "I should check expert runs for staircase descent."
    )
    assert row["reasoning_tokens"] == 25
    assert row["wall_ms"] == 120.5


def test_in_turn_repair_caret_and_prompts():
    """Verify that format_code_with_line_numbers formats caret pointers and build_repair_prompt provides category guidance."""
    from lox.author.agent import AuthorAgent
    from lox.lang.validator import format_code_with_line_numbers

    code = "def explore(self, state):\n    if state.hp < 5\n        return Action.WAIT"
    formatted = format_code_with_line_numbers(code, highlight_line=2, col_offset=20)
    assert ">  2 |     if state.hp < 5" in formatted
    assert "^" in formatted

    agent = AuthorAgent(provider="openrouter")
    unreach_p = agent.build_repair_prompt(
        status="UNREACHABLE_ERROR",
        diag="Method ['stealth_step'] was never called by run",
        mode="FUNC_MOD_ADD",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "UNREACHABLE METHOD IN POLICY" in unreach_p
    assert "The updated caller method" in unreach_p
    assert "def run(self, state):" in unreach_p

    syntax_p = agent.build_repair_prompt(
        status="COMPILE_ERROR",
        diag="SyntaxError: invalid syntax",
        mode="FUNC_MOD_ADD",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "PYTHON SYNTAX OR INDENTATION ERROR" in syntax_p
    assert "4 spaces per indentation level" in syntax_p


def test_summarizer_level_stall_vs_livelock():
    """Verify that build_batch_contrast_digest attributes level stalls to stair descent rather than innocent prompt handlers."""
    from lox.db.summarizer import build_batch_contrast_digest

    batch_stats = {
        "batch_id": "gen_018_champ",
        "avg_depth": 1.1,
        "p90_depth": 2.0,
        "max_depth": 6,
        "episodes_recorded": 100,
    }
    # 50 runs end in max_steps, but explored multiple tiles (recurrence < 0.85)
    runs = [
        {
            "run_id": f"r_{i}",
            "seed": i,
            "max_depth": 1,
            "steps": 5000,
            "turns": 5000,
            "score": 100,
            "end_reason": "max_steps",
            "milestones": {
                "stagnation_ratio": 0.8,
                "recurrence_rate": 0.35,  # Low recurrence = wandering across level
            },
        }
        for i in range(50)
    ]
    fn_stats = [
        {
            "function_name": "handle_prompt",
            "trigger_rate": 0.9,
            "total_turns_active": 30000,
            "avg_depth_when_triggered": 1.0,
            "death_rate_while_active": 0.0,
            "ochiai_good_run_score": 0.1,
            "liblit_failure_score": 0.0,
            "stagnation_turns": 25000,
            "stagnation_share": 0.5,
            "livelock_score": 0.45,
        }
    ]

    digest = build_batch_contrast_digest(batch_stats, runs, fn_stats)
    assert "PRIMARY BOTTLENECK: LEVEL EXPLORATION & STAIRCASE DESCENT FAILURE" in digest
    assert "staircase location & descent" in digest
