"""
Unit tests for modularized navigation components:
SearchPolicy, StairRouter, and FeatureNavigator.
"""
import numpy as np
import pytest

from lox.domain.navigation.search_policy import SearchPolicy
from lox.domain.navigation.stair_routing import StairRouter
from lox.domain.navigation.feature_navigation import FeatureNavigator
from lox.domain.navigation.level_map import LevelMap
from lox.env.blstats import BottomLineStats


def make_blstats(xl=1, hp=16, dnum=0, dlevel=1):
    raw = np.zeros(27, dtype=np.int64)
    raw[0], raw[1] = 10, 10
    raw[10], raw[11] = hp, hp
    raw[12] = dlevel
    raw[18] = xl
    raw[23] = dnum
    raw[24] = dlevel
    return BottomLineStats.from_blstats(raw)


class TestSearchPolicy:
    def test_damage_abort(self):
        sp = SearchPolicy()
        sp._current_search_spot = (5, 5)
        sp._current_search_burst = 3

        # First check establishes baseline HP = 20
        assert not sp.check_damage_abort(20)

        # HP drops to 15 -> aborts active burst!
        assert sp.check_damage_abort(15)
        assert sp._current_search_spot is None
        assert sp._current_search_burst == 0

    def test_suppress_perimeter_when_frontiers_exist(self):
        sp = SearchPolicy()
        lvl = LevelMap(dnum=0, dlevel=1)
        chars = np.full((21, 79), ord("."), dtype=np.uint8)
        # Create a perimeter wall
        chars[4, 5:10] = ord("-")
        lvl.walkable[5:10, 5:10] = True
        dist_grid = np.zeros((21, 79), dtype=np.int32)

        # When has_unvisited_frontiers is True, room perimeter candidate is suppressed!
        cand = sp.find_best_search_candidate(
            5, 5, lvl, chars, dist_grid, has_unvisited_frontiers=True
        )
        assert cand is None


class TestStairRouter:
    def test_retreat_from_mines_underleveled(self):
        sr = StairRouter()
        # In Gnomish Mines (dnum == 2), XL 3 < 5 -> retreat!
        bl = make_blstats(xl=3, dnum=2)
        assert sr.should_retreat_from_mines(bl)

        # High enough level (XL 6) -> safe to explore Mines
        bl_high = make_blstats(xl=6, dnum=2)
        assert not sr.should_retreat_from_mines(bl_high)

    def test_select_safe_stairs(self):
        sr = StairRouter()
        lvl = LevelMap(dnum=0, dlevel=2)
        lvl.all_stairs_down = {(10, 20), (10, 50)}
        bl = make_blstats(xl=3, dnum=0, dlevel=2)
        chars = np.full((21, 79), ord("."), dtype=np.uint8)

        stair = sr.select_safe_down_stairs(lvl, bl, chars)
        assert stair in lvl.all_stairs_down


class TestFeatureNavigator:
    def test_find_fountain(self):
        fn = FeatureNavigator()
        lvl = LevelMap(dnum=0, dlevel=1)
        lvl.fountains.add((8, 8))
        dist_grid = np.full((21, 79), -1, dtype=np.int32)
        dist_grid[8, 8] = 5

        fountain = fn.find_nearest_fountain(5, 5, lvl, dist_grid)
        assert fountain == (8, 8)

    def test_unreachable_fountain_ignored(self):
        fn = FeatureNavigator()
        lvl = LevelMap(dnum=0, dlevel=1)
        lvl.fountains.add((8, 8))
        dist_grid = np.full((21, 79), -1, dtype=np.int32)  # unreachable

        fountain = fn.find_nearest_fountain(5, 5, lvl, dist_grid)
        assert fountain is None
