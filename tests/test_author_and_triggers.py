import pytest
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType
from lox.author.agent import AuthorAgent


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
    trig, msg = engine.check_turn(turns_on_level=10, depth=1, hunger_state="WEAK", food_count=0)
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

    new_code, tree, error = agent.synthesize_policy(current_code, trigger_reason, autopsy)
    assert error is None
    assert tree is not None
    assert "emergency_recovery" in new_code
    assert "combat" in new_code
