"""
ARCHIVED TESTS: Legacy conveyor-corpse detour tests.

Archived because these tests assert that the agent detours from stairs / descent
to chase corpses up to radius 15. This was falsified and proven a net negative
by the food-security A/B test (2026-09-19) documented in AGENTS.md Rule 14:
"Starvation: #pray at WEAK+ (no food) when safe — resets nutrition to 900.
Descent is never blocked by hunger. A/B-proven negative result: food-chasing at ANY
radius is a net negative ... descent is never blocked by hunger."
"""
import numpy as np
import pytest
from nle import nethack

from lox.domain.navigation_manager import NavigationManager
from lox.env.blstats import BottomLineStats


def test_conveyor_corpse_overrides_descent_archived():
    """Archived: previously forced agent to detour toward conveyor corpse rather than descending."""
    pass


def test_navigation_conveyor_corpse_pathing_archived():
    """Archived: previously forced agent without poison resistance to prioritize conveyor pathing."""
    pass
