import pytest
from lox.envs.nethack import NetHackAdapter
from lox.core.types import Action


def test_tactical_primitives_execution():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=101)

    initial_turn = obs.hero.turn
    assert initial_turn >= 1

    # 1. Test engrave_elbereth sequence (advances turn)
    obs, reward, term, trunc, _ = adapter.step(Action(name="engrave_elbereth"))
    assert obs.hero.turn > initial_turn

    # 2. Test kick closed door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(Action(name="kick_closed_door", direction=(0, 1)))
    assert obs.hero.hp > 0

    # 3. Test open door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(Action(name="open_door", direction=(0, 1)))
    assert obs.hero.turn >= initial_turn

    # 4. Test eat floor corpse (safe fallback when empty)
    obs, reward, term, trunc, _ = adapter.step(Action(name="eat_floor_corpse"))
    assert obs is not None

    adapter.close()
