"""
LOX-ψ Navigation: Special Dungeon Features & POI Routing.

Manages navigation to non-standard environmental features:
1. Fountains (safe dipping protocol for Excalibur at XL >= 5).
2. Altars (epistemic BUC identification and sacrifice).
3. Thrones & Chests.
"""
from __future__ import annotations

from typing import Any, Tuple, Optional
import numpy as np

from lox.domain.navigation.level_map import LevelMap
from lox.env.blstats import BottomLineStats


class FeatureNavigator:
    """Finds paths to points of interest on the level."""

    def __init__(self, config: Any = None):
        self.cfg = config

    def find_nearest_fountain(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        dist_grid: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Returns the closest reachable unexhausted fountain."""
        if not lvl.fountains:
            return None
        valid = [
            f for f in lvl.fountains
            if f not in lvl.blocked_tiles and dist_grid[f[0], f[1]] >= 0
        ]
        if not valid:
            return None
        valid.sort(key=lambda f: dist_grid[f[0], f[1]])
        return valid[0]

    def find_nearest_altar(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        dist_grid: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Returns the closest reachable altar."""
        if not lvl.altars:
            return None
        valid = [
            a for a in lvl.altars
            if a not in lvl.blocked_tiles and dist_grid[a[0], a[1]] >= 0
        ]
        if not valid:
            return None
        valid.sort(key=lambda a: dist_grid[a[0], a[1]])
        return valid[0]
