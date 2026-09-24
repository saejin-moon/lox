"""
Performance-regression guards for the two safe optimizations:
  * GoalInterpreter loads failure counters ONLY when the program references the
    failures_in_10_episodes_ge predicate (avoids a per-process DuckDB query).
  * ItemBeliefState.buc_determined is exactly equivalent to p_buc.max() >= 0.999
    but uses scalar indexing (measured hot path).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from lox.env.blstats import BottomLineStats
from lox.policy.goal_interpreter import GoalInterpreter, _program_uses_failure_predicate
from lox.policy.program import PolicyProgram


def make_blstats(turn=10):
    return BottomLineStats(
        x=10, y=10, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=16, max_hp=16,
        depth=2, gold=0, energy=10, max_energy=10, ac=8, monster_level=1,
        experience=2, turn=turn, hunger_state=1, encumbrance=0,
        dungeon_number=0, level_number=2, condition_bits=0, alignment=0)


class TestFailureCounterGating:
    def _write_program(self, tmp_path, conditions: dict) -> str:
        base = PolicyProgram.load()
        d = base.to_dict()
        d["strategy_plan"][0]["when"] = conditions.get("goal_when", "(true)")
        if "rule_when" in conditions:
            d["tactic_rules"] = [{"when": conditions["rule_when"], "do": "retreat"}]
        if "nogood_when" in conditions:
            d["nogoods"] = [{"when": conditions["nogood_when"], "cause": "x"}]
        path = str(tmp_path / "p.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        return path

    def test_predicate_detection(self, tmp_path):
        assert not _program_uses_failure_predicate(
            PolicyProgram.load(self._write_program(tmp_path, {})))
        assert _program_uses_failure_predicate(PolicyProgram.load(self._write_program(
            tmp_path, {"goal_when": "(failures_in_10_episodes_ge 2 descend)"})))
        assert _program_uses_failure_predicate(PolicyProgram.load(self._write_program(
            tmp_path, {"rule_when": "(failures_in_10_episodes_ge 1 enter_sokoban)"})))
        assert _program_uses_failure_predicate(PolicyProgram.load(self._write_program(
            tmp_path, {"nogood_when": "(failures_in_10_episodes_ge 3 descend)"})))

    def test_unused_predicate_never_queries_duckdb(self, tmp_path, monkeypatch):
        path = self._write_program(tmp_path, {})
        prog = PolicyProgram.load(path)
        assert not _program_uses_failure_predicate(prog)
        called = []
        monkeypatch.setattr("lox.policy.failure_counters.load_failure_counters",
                            lambda *a, **k: called.append(1) or {"descend": 5})
        interp = GoalInterpreter(failure_counters=None, program_path=path)
        ctx = interp._ctx(make_blstats(), "valkyrie")
        assert ctx["failures"] == {} and ctx["failures_total"] == 0
        assert called == []  # no DuckDB access

    def test_used_predicate_loads_and_injects(self, tmp_path, monkeypatch):
        path = self._write_program(tmp_path, {"goal_when": "(failures_in_10_episodes_ge 2 descend)"})
        assert _program_uses_failure_predicate(PolicyProgram.load(path))
        monkeypatch.setattr("lox.policy.failure_counters.load_failure_counters",
                            lambda *a, **k: {"descend": 5})
        interp = GoalInterpreter(failure_counters=None, program_path=path)
        ctx = interp._ctx(make_blstats(), "valkyrie")
        assert ctx["failures"] == {"descend": 5} and ctx["failures_total"] == 5

    def test_explicit_counters_still_win(self, tmp_path):
        path = self._write_program(tmp_path, {})
        interp = GoalInterpreter(failure_counters={"x": 1}, program_path=path)
        assert interp._ctx(make_blstats(), "valkyrie")["failures"] == {"x": 1}


class TestScanCache:
    def _mgr_and_grid(self, mon_x=11):
        import nle.nethack as nethack
        from lox.domain.combat_manager import TacticalCombatManager
        from tests.test_shop_and_dungeon_graph import make_blstats
        mgr = TacticalCombatManager()
        glyphs = np.full((21, 79), nethack.NO_GLYPH, dtype=np.int32)
        glyphs[10, mon_x] = nethack.GLYPH_MON_OFF + 1  # jackal
        return mgr, glyphs, make_blstats(x=10, y=10, turn=50)

    def test_hit_returns_equal_copy(self):
        mgr, glyphs, bs = self._mgr_and_grid()
        a = mgr.scan_monsters(glyphs, bs)
        b = mgr.scan_monsters(glyphs, bs)
        assert [m.name for m in a] == [m.name for m in b]
        assert a is not b  # copy: caller mutation cannot corrupt the cache
        assert a == b

    def test_inplace_glyph_mutation_invalidates(self):
        import nle.nethack as nethack
        mgr, glyphs, bs = self._mgr_and_grid(mon_x=11)
        assert len(mgr.scan_monsters(glyphs, bs)) == 1
        glyphs.fill(nethack.NO_GLYPH)  # monster removed in place, same turn
        assert mgr.scan_monsters(glyphs, bs) == []

    def test_hero_move_invalidates(self):
        import nle.nethack as nethack
        from lox.domain.combat_manager import TacticalCombatManager
        from tests.test_shop_and_dungeon_graph import make_blstats
        mgr = TacticalCombatManager()
        glyphs = np.full((21, 79), nethack.NO_GLYPH, dtype=np.int32)
        glyphs[10, 11] = nethack.GLYPH_MON_OFF + 1
        bs_adj = make_blstats(x=10, y=10, turn=50)
        bs_far = make_blstats(x=10, y=1, turn=50)   # same turn, moved away
        assert mgr.scan_monsters(glyphs, bs_adj)[0].is_adjacent is True
        assert mgr.scan_monsters(glyphs, bs_far)[0].is_adjacent is False

    def test_reset_clears_cache(self):
        mgr, glyphs, bs = self._mgr_and_grid()
        mgr.scan_monsters(glyphs, bs)
        mgr.reset()
        assert getattr(mgr, "_scan_cache", None) is None


class TestBucDetermined:
    @pytest.mark.parametrize("p", [
        [0.10, 0.80, 0.10], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.999, 0.0005, 0.0005], [0.998, 0.001, 0.001], [0.0, 0.0, 0.0],
    ])
    def test_equivalence_to_numpy_max(self, p):
        from lox.epistemic.belief_state import ItemBeliefState
        b = ItemBeliefState(uid="u", p_buc=np.array(p, dtype=np.float64))
        assert b.buc_determined == bool(b.p_buc.max() >= 0.999)

    def test_random_equivalence(self):
        from lox.epistemic.belief_state import ItemBeliefState
        rng = np.random.default_rng(0)
        for _ in range(2000):
            p = rng.random(3)
            if rng.random() < 0.3:
                p = np.zeros(3); p[rng.integers(0, 3)] = 1.0
            b = ItemBeliefState(uid="u", p_buc=p)
            assert b.buc_determined == bool(p.max() >= 0.999)

    def test_nonstandard_shape_falls_back(self):
        from lox.epistemic.belief_state import ItemBeliefState
        b = ItemBeliefState(uid="u", p_buc=np.array([0.2, 0.8], dtype=np.float64))
        assert b.buc_determined == bool(np.array([0.2, 0.8]).max() >= 0.999)


class TestNumbaAStarEquivalence:
    def test_astar_path_found(self):
        from lox.navigation.astar import GridAStar
        walkable = np.zeros((21, 79), dtype=bool)
        walkable[2:10, 2:10] = True
        path = GridAStar.find_path((2, 2), (8, 8), walkable)
        assert path is not None
        assert len(path) == 6
        assert path[-1].row == 8 and path[-1].col == 8

    def test_astar_hazard_avoidance(self):
        from lox.navigation.astar import GridAStar
        walkable = np.zeros((21, 79), dtype=bool)
        walkable[5:8, 5:8] = True
        hazards = np.zeros((21, 79), dtype=float)
        hazards[6, 6] = 500.0
        path = GridAStar.find_path((5, 6), (7, 6), walkable, hazard_costs=hazards)
        assert path is not None
        coords = [(n.row, n.col) for n in path]
        assert (6, 6) not in coords


class TestMonsterLUTCorrectness:
    def test_lut_matches_nethack_permonst(self):
        from nle import nethack
        from lox.domain.combat.threat_scan import _PM_NAME, _PM_SPEED, _PM_LEVEL, _PM_AC
        for mid in range(getattr(nethack, "NUMMONS", 381)):
            pm = nethack.permonst(mid)
            assert _PM_NAME[mid] == pm.mname.lower()
            assert _PM_SPEED[mid] == pm.mmove
            assert _PM_LEVEL[mid] == pm.mlevel
            assert _PM_AC[mid] == pm.ac


class TestNumbaFrontierBFSEquivalence:
    def test_frontier_found(self):
        from lox.navigation.frontier import FrontierExplorer
        walkable = np.zeros((21, 79), dtype=bool)
        unmapped = np.zeros((21, 79), dtype=bool)
        walkable[5:10, 5:10] = True
        unmapped[5:10, 10] = True  # unmapped border to the east

        frontier = FrontierExplorer.find_nearest_frontier((5, 5), walkable, unmapped)
        assert frontier is not None
        # Frontier tile must be within the walkable room adjacent to unmapped
        assert frontier[0] in range(5, 10)
        assert frontier[1] == 9

    def test_unreachable_frontier_returns_none(self):
        from lox.navigation.frontier import FrontierExplorer
        walkable = np.zeros((21, 79), dtype=bool)
        unmapped = np.zeros((21, 79), dtype=bool)
        walkable[2, 2] = True
        # Frontier far away and disconnected
        walkable[15, 15] = True
        unmapped[15, 16] = True

        frontier = FrontierExplorer.find_nearest_frontier((2, 2), walkable, unmapped)
        assert frontier is None


class TestNumbaDistanceGridEquivalence:
    def test_bfs_distances_open_room(self):
        from lox.domain.navigation.exploration import ExplorationMixin
        from lox.domain.navigation.level_map import LevelMap
        mixin = ExplorationMixin()
        lvl = LevelMap(dnum=0, dlevel=1)
        lvl.walkable[5:8, 5:8] = True

        dist_grid = mixin._compute_bfs_distances(6, 6, lvl)
        assert dist_grid[6, 6] == 0
        assert dist_grid[6, 7] == 1
        assert dist_grid[5, 5] == 1
        assert dist_grid[0, 0] == -1  # unwalkable

    def test_bfs_corner_cutting_interlock(self):
        from lox.domain.navigation.exploration import ExplorationMixin
        from lox.domain.navigation.level_map import LevelMap
        mixin = ExplorationMixin()
        lvl = LevelMap(dnum=0, dlevel=1)
        # Create an L-shaped corner with diagonal blocked
        lvl.walkable[5, 5] = True
        lvl.walkable[6, 6] = True
        lvl.walkable[5, 6] = False
        lvl.walkable[6, 5] = False

        dist_grid = mixin._compute_bfs_distances(5, 5, lvl)
        assert dist_grid[5, 5] == 0
        # (6, 6) should be unreachable because diagonal corner cutting is forbidden!
        assert dist_grid[6, 6] == -1


class TestNumbaBresenhamLOSEquivalence:
    def test_clear_line_of_sight(self):
        from lox.domain.navigation.stepping import SteppingMixin
        walkable = np.ones((21, 79), dtype=bool)
        assert SteppingMixin.has_line_of_sight(5, 5, 5, 15, walkable)
        assert SteppingMixin.has_line_of_sight(5, 5, 10, 10, walkable)

    def test_blocked_line_of_sight(self):
        from lox.domain.navigation.stepping import SteppingMixin
        walkable = np.ones((21, 79), dtype=bool)
        # Wall blocking horizontal line
        walkable[5, 10] = False
        assert not SteppingMixin.has_line_of_sight(5, 5, 5, 15, walkable)
        # Endpoint itself does not have to be walkable (e.g. targeting a wall or monster on unwalkable tile)
        # but intermediate tiles must be walkable
        walkable[5, 10] = True
        assert SteppingMixin.has_line_of_sight(5, 5, 5, 15, walkable)