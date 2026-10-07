"""
Shared Pytest Fixtures and Configuration for LOX Test Suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from lox.core.types import (
    CombatView,
    DungeonView,
    EpistemicView,
    HeroState,
    HeroStatus,
    InventoryView,
    Observation,
    RunMemory,
    SpatialView,
)


@pytest.fixture
def dummy_hero_state() -> HeroState:
    """Returns a basic HeroState for testing."""
    return HeroState(
        y=10,
        x=10,
        hp=16,
        max_hp=16,
        energy=5,
        max_energy=5,
        ac=6,
        level=1,
        depth=1,
        turn=10,
    )


@pytest.fixture
def dummy_observation(dummy_hero_state: HeroState) -> Observation:
    """Returns a minimal valid Observation for testing."""
    return Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=dummy_hero_state,
        status=HeroStatus(),
        combat=CombatView(),
        spatial=SpatialView(),
        dungeon=DungeonView(),
        epistemic=EpistemicView(),
        inventory=InventoryView(),
        memory=RunMemory(),
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically assigns markers based on directory hierarchy."""
    for item in items:
        path_str = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path_str:
            item.add_marker(pytest.mark.integration)
            if "/envs/" in path_str or "scientific" in path_str:
                item.add_marker(pytest.mark.gym)
