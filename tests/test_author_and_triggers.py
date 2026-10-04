from lox.author.agent import AuthorAgent
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType


def test_dynamic_triggers():
    engine = DynamicTriggerEngine(stall_threshold=150, cluster_threshold=3)
    recorder = FlightRecorder()

    # Normal turn -> no trigger
    trig, msg = engine.check_turn(turns_on_level=50, depth=1, has_frontier=True)
    assert trig == TriggerType.NONE

    # Stall turn -> STALL trigger fires
    trig, msg = engine.check_turn(turns_on_level=155, depth=1, has_frontier=False)
    assert trig == TriggerType.STALL
    assert "pacing stall" in msg

    # Starvation crisis -> STARVATION trigger fires
    trig, msg = engine.check_turn(
        turns_on_level=10, depth=1, hunger_state="WEAK", food_count=0
    )
    assert trig == TriggerType.STARVATION
    assert "Starvation crisis" in msg

    # Depth breakthrough -> MILESTONE trigger fires
    trig, msg = engine.check_turn(turns_on_level=10, depth=2)
    assert trig == TriggerType.MILESTONE
    assert "Depth 2" in msg

    # Record 3 identical deaths
    recorder.record_death("You faint from lack of food.")
    recorder.record_death("The jackal bites!")
    recorder.record_death("You faint from lack of food.")
    recorder.record_death("You faint from lack of food.")

    trig, msg = engine.check_death_cluster(recorder)
    assert trig == TriggerType.CLUSTER_DEATH
    assert "lack of food" in msg


def test_author_agent_mock_synthesis():
    agent = AuthorAgent(provider="mock")
    current_code = """
def explore():
    step_to_frontier()

plan = [explore]
"""
    trigger_reason = "Death cluster: 3/5 died of starvation."
    autopsy = "### Autopsy: Fainted on DL1"

    new_code, tree, error = agent.synthesize_policy(
        current_code, trigger_reason, autopsy
    )
    assert error is None
    assert tree is not None
    assert "class Agent:" in new_code
    assert "yield pray()" in new_code
    assert "yield melee_attack_hostile()" in new_code


def test_extract_code_unclosed_block():
    agent = AuthorAgent(provider="mock")
    # Truncated response with no closing backticks
    raw_unclosed = """Here is the evolved policy:
```python
def combat():
    if adjacent_hostile:
        melee_attack_hostile()

plan = [combat]
"""
    extracted = agent.extract_code(raw_unclosed)
    assert "plan = [combat]" in extracted
    assert "Here is the evolved policy" not in extracted


def test_tool_type_coercion_and_shielding(tmp_path):
    import duckdb

    from lox.author.tools import DuckDBToolRegistry

    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE episodes (
            run_id VARCHAR, episode_id VARCHAR, depth INTEGER, score INTEGER,
            turns INTEGER, death_reason VARCHAR, solved BOOLEAN, wall_sec FLOAT,
            role VARCHAR, gold INTEGER, max_depth INTEGER, steps INTEGER,
            attacks INTEGER, descents INTEGER, searches INTEGER, eats INTEGER,
            prayers INTEGER, death_category VARCHAR
        )
    """)
    con.execute(
        "INSERT INTO episodes VALUES ('r1', 'e1', 1, 10, 50, 'starved', false, 1.0, 'valkyrie', 0, 1, 50, 0, 0, 0, 0, 0, 'hunger')"
    )
    con.close()

    tools = DuckDBToolRegistry(db_path=db_path)
    # Passing string window should not raise TypeError
    tax = tools.get_death_taxonomy(window="10")  # type: ignore
    assert "Top Fatalities" in tax
    assert "starved" in tax

    # Passing string depth should not raise TypeError
    pacing = tools.get_floor_pacing_stats(depth="1")  # type: ignore
    assert "Floor Pacing Stats" in pacing

    # Shielded tool execution in AuthorAgent
    agent = AuthorAgent(provider="mock", db_path=db_path)
    res = agent._execute_tool("get_death_taxonomy", {"window": "5"})
    assert "Top Fatalities" in res
