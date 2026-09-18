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

    # Verify agent executed steps without error
    assert agent.step_counter > 0
    assert agent.max_turn_reached >= initial_turn


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
    assert result.steps_per_second > 100.0  # Verify sub-10ms latency per cognitive cycle


def test_corp_agent_priority_emergency_healing():
    import dataclasses
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    # Artificially set agent state to low HP (4 / 16)
    agent.current_blstats = dataclasses.replace(agent.current_blstats, hp=4, max_hp=16)

    # Insert a healing potion into inventory tracker
    inv_strs = np.zeros((55, 80), dtype=np.uint8)
    inv_letters = np.zeros(55, dtype=np.uint8)
    inv_glyphs = np.zeros(55, dtype=np.int32)

    potion_str = b"x - an uncursed potion of extra healing"
    inv_strs[0, :len(potion_str)] = [c for c in potion_str]
    inv_letters[0] = ord("x")
    import nle
    inv_glyphs[0] = nle.nethack.GLYPH_OBJ_OFF + 50
    env.inventory_tracker.synchronize(inv_strs, inv_letters, inv_glyphs, turn=10)

    # Place an adjacent monster right next to hero
    py, px = agent.current_blstats.y, agent.current_blstats.x
    agent.current_glyphs[py, px + 1] = nle.nethack.GLYPH_MON_OFF + 1  # jackal adjacent

    # Agent selects action: Priority 0.5 must quaff the potion rather than blindly attacking!
    chosen = agent.select_action()
    assert chosen.name == "QUAFF"
    assert chosen.args["slot"] == "x"


def test_corp_agent_bicameral_htn_persona_modulation():
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    # Verify HTN methods are registered for SURVIVE_AND_ASCEND
    assert "SURVIVE_AND_ASCEND" in agent.planner.methods
    assert len(agent.planner.methods["SURVIVE_AND_ASCEND"]) >= 7

    # Verify decomposition works cleanly
    action = agent.select_action()
    assert action is not None
    assert action.is_primitive is True


def test_anti_stall_guard_and_blocked_tiles():
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    agent.reset()

    # 1. Verify Anti-Stall Guard triggers on 4+ zero-turn steps
    agent.consecutive_zero_turn_steps = 5
    action = agent.select_action()
    assert action.name == "ESCAPE"

    agent.consecutive_zero_turn_steps = 6
    action = agent.select_action()
    assert action.name == "WAIT"

    # 2. Verify blocked_tiles persistence across update_map calls
    lvl_map = agent.nav_mgr.get_or_create_level(agent.current_blstats)
    test_pos = (5, 8)
    lvl_map.blocked_tiles.add(test_pos)

    # Chars has floor '.' at test_pos
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[test_pos[0], test_pos[1]] = ord(".")
    agent.nav_mgr.update_map(chars, agent.current_blstats)

    # Walkable MUST stay False because test_pos is in blocked_tiles!
    assert lvl_map.walkable[test_pos[0], test_pos[1]] == False

