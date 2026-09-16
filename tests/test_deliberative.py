"""
Unit tests for Phase 4: Deliberative LLM Core, ResponseParser,
AutopsyEngine, and DeadlockResolver.
"""

import pytest
import numpy as np

from corp.deliberative.schemas import HTNGraphPatch, AutopsyReport, SubTaskSpec, NogoodClause
from corp.deliberative.response_parser import ResponseParser, LLMParseError
from corp.deliberative.providers.mock_provider import MockProvider
from corp.deliberative.autopsy_engine import AutopsyEngine
from corp.deliberative.deadlock_resolver import DeadlockResolver
from corp.env.flight_recorder import FlightRecorderRingBuffer
from corp.env.blstats import BottomLineStats
from corp.planner.nogood import NogoodStore
from corp.planner.predicates import PredicateBit


def test_response_parser_dual_phase():
    raw_response = """
    <think>
    The agent encountered an oscillation between coordinates (10, 10) and (10, 11).
    There is a locked door at (10, 12) blocking forward pathing.
    We need to inject a kick action or search for secret corridors.
    </think>
    ```json
    {
      "deadlock_cause": "Locked door blocking corridor route",
      "confidence": 0.95,
      "abandon_current_macro": false,
      "injected_subtasks": [
        {"task_name": "APPLY", "target_slot": "k", "target_direction": "l"},
        {"task_name": "STEP", "target_direction": "l"}
      ],
      "new_invariants": ["no_unidentified_scrolls_read"]
    }
    ```
    """

    thinking, patch = ResponseParser.parse_dual_phase(raw_response, HTNGraphPatch)
    assert "oscillation" in thinking
    assert patch.deadlock_cause == "Locked door blocking corridor route"
    assert patch.confidence == 0.95
    assert len(patch.injected_subtasks) == 2
    assert patch.injected_subtasks[0].task_name == "APPLY"
    assert patch.injected_subtasks[0].target_direction == "l"


@pytest.mark.asyncio
async def test_autopsy_engine_synthesis():
    # 1. Setup simulated flight recorder with death trace
    recorder = FlightRecorderRingBuffer(capacity=20)
    raw_bl = np.zeros(27, dtype=np.int64)
    raw_bl[10] = 0   # 0 HP (Dead!)
    raw_bl[11] = 16  # Max HP
    raw_bl[20] = 105 # Turn 105
    blstats = BottomLineStats.from_blstats(raw_bl)

    recorder.record(
        turn=105,
        action="MELEE_ATTACK",
        blstats=blstats,
        message="You hit the floating eye. You cannot move! The jackal bites! You die...",
    )

    # 2. Mock LLM Response with structured autopsy report
    canned_autopsy = """
    <think>
    The player melee attacked a floating eye without blindness or reflection.
    This caused passive gaze paralysis, rendering the player helpless while jackals attacked.
    The fatal action was MELEE_ATTACK when adjacent to floating eye.
    </think>
    ```json
    {
      "death_cause": "Passive gaze paralysis followed by jackal melee",
      "lethal_turn": 105,
      "causal_chain": [
        "Closed distance with floating eye",
        "Executed melee attack without reflection or blindness",
        "Paralyzed for 50 turns",
        "Killed by adjacent jackals"
      ],
      "counterfactual_fix": "Used ranged dagger throwing or bypassed floating eye",
      "nogood": {
        "trigger_predicates": {
          "adjacent_floating_eye": true,
          "is_blind": false,
          "reflection_active": false
        },
        "fatal_action": "MELEE_ATTACK",
        "derived_constraint": "Never melee attack floating eye without blindness or reflection"
      }
    }
    ```
    """

    provider = MockProvider(canned_response=canned_autopsy)
    nogood_store = NogoodStore()
    engine = AutopsyEngine(llm_provider=provider, nogood_store=nogood_store)

    report, entry = await engine.execute_autopsy(
        recorder=recorder,
        death_message="Killed by a jackal while helpless.",
        generation=2,
    )

    assert report.lethal_turn == 105
    assert entry.forbidden_action == "MELEE_ATTACK"
    assert entry.generation == 2
    assert entry.mask & PredicateBit.ADJACENT_FLOATING_EYE

    # Verify active Nogood store immediately blocks this action!
    state_mask = PredicateBit.ADJACENT_FLOATING_EYE
    forbidden, reason = nogood_store.is_forbidden(state_mask, "MELEE_ATTACK")
    assert forbidden is True


@pytest.mark.asyncio
async def test_deadlock_resolver():
    canned_patch = """
    <think>
    The agent is stuck on a door. We should schedule search actions.
    </think>
    {
      "deadlock_cause": "Dead-end corridor requiring secret door search",
      "confidence": 0.88,
      "abandon_current_macro": false,
      "injected_subtasks": [
        {"task_name": "SEARCH"},
        {"task_name": "SEARCH"},
        {"task_name": "STEP", "target_direction": "h"}
      ],
      "new_invariants": []
    }
    """
    provider = MockProvider(canned_response=canned_patch)
    resolver = DeadlockResolver(llm_provider=provider)

    raw_bl = np.zeros(27, dtype=np.int64)
    raw_bl[10] = 15
    raw_bl[11] = 15
    blstats = BottomLineStats.from_blstats(raw_bl)

    patch = await resolver.resolve_deadlock(
        failed_task="ROUTE_TO_FRONTIER",
        blstats=blstats,
        cycle_detected=True,
    )

    assert patch.confidence == 0.88
    tasks = DeadlockResolver.patch_to_tasks(patch)
    assert len(tasks) == 3
    assert tasks[0].name == "SEARCH"
    assert tasks[0].is_primitive is True
    assert tasks[2].name == "STEP"
    assert tasks[2].args["direction"] == "h"
