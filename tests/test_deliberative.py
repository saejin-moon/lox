"""
Unit tests for Phase 4: Deliberative LLM Core, ResponseParser,
AutopsyEngine, and DeadlockResolver.
"""

import os

import pytest
import numpy as np

from corp.deliberative.schemas import HTNGraphPatch, AutopsyReport, SubTaskSpec, NogoodClause
from corp.deliberative.response_parser import ResponseParser, LLMParseError
from corp.deliberative.providers.mock_provider import MockProvider
from corp.deliberative.providers.openrouter import OpenRouterProvider
from corp.deliberative.providers.gemini import GeminiProvider
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


def test_provider_initialization():
    gemini = GeminiProvider(api_key="test_gemini_key")
    assert gemini.model == os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
    assert "generativelanguage.googleapis.com" in str(gemini.client.base_url)
    # native genai path: thinking-enabled author path (R3)
    if gemini._genai_client is not None:
        assert gemini.thinking_level == os.environ.get("GEMINI_THINKING_LEVEL", "high")

    openrouter = OpenRouterProvider(api_key="test_openrouter_key")
    assert "openrouter.ai" in str(openrouter.client.base_url)


def test_throttler_hierarchical_escalation():
    from corp.deliberative.throttler import LLMRateThrottler, ThrottlerConfig
    throttler = LLMRateThrottler(ThrottlerConfig(escalation_threshold=2, min_wall_seconds=0.0))

    # First stall attempt: Should reject LLM and enforce symbolic resolution
    throttler.record_stall("NAVIGATE")
    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=10, task_name="NAVIGATE")
    assert not allowed
    assert "hierarchical_escalation_required" in reason

    # Second stall attempt: Meets threshold, allowed
    throttler.record_stall("NAVIGATE")
    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=11, task_name="NAVIGATE")
    assert allowed
    assert "granted" in reason


def test_throttler_quotas_and_cooldown():
    from corp.deliberative.throttler import LLMRateThrottler, ThrottlerConfig
    config = ThrottlerConfig(
        max_in_game_per_episode=2,
        min_wall_seconds=0.5,
        min_turn_gap=50,
        escalation_threshold=1,
    )
    throttler = LLMRateThrottler(config)

    # 1. First query allowed
    throttler.record_stall("NAVIGATE")
    allowed, _ = throttler.should_allow_query("deadlock_cycle", current_turn=100, task_name="NAVIGATE")
    assert allowed
    throttler.record_query_dispatched("deadlock_cycle", current_turn=100)

    # 2. Turn cooldown active
    throttler.record_stall("NAVIGATE")
    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=120, task_name="NAVIGATE")
    assert not allowed
    assert "turn_cooldown_active" in reason

    # 3. Turn cooldown passed (turn 200), but wall cooldown active (< 0.5s)
    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=200, task_name="NAVIGATE")
    assert not allowed
    assert "wall_cooldown_active" in reason

    # 4. Sleep past wall cooldown
    import time
    time.sleep(0.55)
    allowed, _ = throttler.should_allow_query("deadlock_cycle", current_turn=200, task_name="NAVIGATE")
    assert allowed
    throttler.record_query_dispatched("deadlock_cycle", current_turn=200)

    # 5. Episode quota exhausted (2 out of 2)
    time.sleep(0.55)
    throttler.record_stall("NAVIGATE")
    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=350, task_name="NAVIGATE")
    assert not allowed
    assert "in_game_quota_exceeded" in reason

    # 6. Autopsy is NOT starved by in-game quota
    allowed, reason = throttler.should_allow_query("autopsy", current_turn=350)
    assert allowed
    assert "autopsy_granted" in reason


def test_throttler_state_cache():
    from corp.deliberative.throttler import LLMRateThrottler
    from corp.deliberative.schemas import HTNGraphPatch
    throttler = LLMRateThrottler()

    dummy_patch = HTNGraphPatch(
        deadlock_cause="Test obstacle",
        confidence=1.0,
        abandon_current_macro=False,
        injected_subtasks=[],
    )
    s_hash = 123456789
    throttler.store_cached_patch(s_hash, dummy_patch)
    assert throttler.get_cached_patch(s_hash) is dummy_patch

    allowed, reason = throttler.should_allow_query("deadlock_cycle", current_turn=10, state_hash=s_hash)
    assert not allowed
    assert "cached_patch_available" in reason


