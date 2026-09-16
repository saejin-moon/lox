"""
Integration and unit tests for CORPAgent autonomous game loop and CompetenceEngine.
"""

import pytest
import numpy as np

from corp.env.nle_wrapper import make_env
from corp.agent.corp_agent import CORPAgent, EpisodeResult
from corp.agent.competence import CompetenceEngine
from corp.planner.persona import PersonaTraitVector
from corp.deliberative.providers.mock_provider import MockProvider


def test_competence_engine():
    engine = CompetenceEngine()
    valk_traits = PersonaTraitVector(0.9, 0.3, 0.2, 0.4, 0.8)
    wiz_traits = PersonaTraitVector(0.3, 0.4, 0.95, 0.5, 0.8)

    # Valkyrie survives deep into dungeon
    for _ in range(5):
        engine.record_episode("valkyrie", valk_traits, turns=15000, depth=15, score=4000, had_fatal_error=False)

    # Wizard dies early
    for _ in range(5):
        engine.record_episode("wizard", wiz_traits, turns=1200, depth=3, score=500, had_fatal_error=True)

    valk_comp = engine.compute_competence("valkyrie")
    wiz_comp = engine.compute_competence("wizard")

    assert valk_comp > wiz_comp
    assert engine.select_optimal_persona() == "valkyrie"


def test_corp_agent_reset_and_persona():
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())

    obs, info = agent.reset()
    assert agent.persona is not None
    assert 0.0 <= agent.persona.resilience <= 1.0
    assert 0.0 <= agent.persona.mana <= 1.0
    assert agent.current_blstats is not None
    assert agent.current_blstats.turn == 1


def test_corp_agent_steps():
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    initial_turn = agent.current_blstats.turn
    # Execute 10 autonomous cognitive steps
    for _ in range(10):
        if agent.is_terminal:
            break
        obs, reward, terminated, truncated, info = agent.step()

    # Verify game advanced
    assert agent.current_blstats.turn > initial_turn


def test_corp_agent_bounded_episode():
    env = make_env()
    agent = CORPAgent(
        env=env,
        llm_provider=MockProvider(),
        enable_deliberative_autopsy=True,
    )

    result = agent.run_episode(max_steps=50)
    assert isinstance(result, EpisodeResult)
    assert result.turns >= 1
    assert result.max_depth >= 1
    assert result.steps_per_second > 100.0  # Verify sub-10ms latency per cognitive cycle
