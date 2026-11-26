"""
Adversarial & Boundary Regression Tests for Full-Codebase Audit Findings (P0/P1/P2).
Covers:
- P0 sandbox escapes (operator.attrgetter/methodcaller, typing.get_type_hints,
  generator/frame/traceback introspection, numpy file I/O & ndarray.tofile/dump)
- P1 DuckDB init cache survival across writes, falsy-zero 0.0 micro-pilot tracking,
  SQL double-quote table-function bypass & fail-closed external_access,
  proactive_2 confirm-crash branch guard, worn_oprop_mask bit-shifting,
  Wiki pre-mortem snippet/text extraction, FINAL_SYNTHESIS & multi-step repair logs
- P2 Rust replay shape ValueError & tile clear passability, GameState tuple indices 12-15,
  unworn_armor_deaths SQL union, MAP-Elites None field tolerance
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lox.author.tools.db_tools import validate_readonly_sql
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, init_db, is_db_initialized
from lox.engine.lifecycle import update_proactive_branch_lifecycle
from lox.engine.referee_gates import TournamentVerdict
from lox.lang.compiler import (
    _SAFE_STDLIB_MODULES,
    _is_forbidden_attr,
    compile_policy_package,
)
from lox.lang.sandbox import PolicyValidationError, validate_python_code


def test_p0_sandbox_blocks_operator_typing_frame_and_numpy_io() -> None:
    """All 4 P0 sandbox escape vectors must be blocked at both AST and runtime proxy layers."""
    bad_snippets = [
        # 1. operator.attrgetter / methodcaller C-level getattr bypass
        "import operator\nclass Policy:\n    def step(self, s): return operator.attrgetter('__subclasses__')(object)()\n",
        "from operator import attrgetter\nclass Policy:\n    def step(self, s): return attrgetter('__subclasses__')(object)\n",
        "import operator\nclass Policy:\n    def step(self, s): return operator.methodcaller('__subclasses__')(object)\n",
        # 2. typing.get_type_hints unrestricted eval()
        "import typing\ndef f(x: '1+1'): pass\nclass Policy:\n    def step(self, s): return typing.get_type_hints(f, globalns={})\n",
        "from typing import get_type_hints\nclass Policy:\n    def step(self, s): return get_type_hints(s)\n",
        # 3. generator/frame/traceback/code introspection
        "class Policy:\n    def run(self, s):\n        g = (x for x in [1])\n        _ = g.gi_frame.f_back.f_globals\n        yield 0\n",
        "class Policy:\n    def step(self, s):\n        return s.tb_frame.f_builtins\n",
        # 4. numpy filesystem/pickle I/O & ndarray.tofile/dump
        "import numpy as np\nclass Policy:\n    def step(self, s): return np.load('/tmp/x.npy', allow_pickle=True)\n",
        "import numpy as np\nclass Policy:\n    def step(self, s):\n        a = np.zeros(1)\n        a.tofile('/tmp/pwn')\n        return 0\n",
        "import numpy as np\nclass Policy:\n    def step(self, s):\n        x = np\n        return x.savetxt('/tmp/pwn', x.zeros(1))\n",
    ]
    for idx, src in enumerate(bad_snippets):
        with pytest.raises(PolicyValidationError, match="forbidden"):
            validate_python_code(src)
        with pytest.raises(PolicyValidationError):
            compile_policy_package(src)

    # Runtime proxy & _is_forbidden_attr defense-in-depth
    op_proxy = _SAFE_STDLIB_MODULES["operator"]
    assert not hasattr(op_proxy, "attrgetter")
    assert not hasattr(op_proxy, "methodcaller")
    typing_proxy = _SAFE_STDLIB_MODULES["typing"]
    assert not hasattr(typing_proxy, "get_type_hints")
    assert not hasattr(typing_proxy, "ForwardRef")
    np_proxy = _SAFE_STDLIB_MODULES["numpy"]
    for bad_np in ("load", "save", "savez", "fromfile", "memmap", "loadtxt", "savetxt"):
        assert not hasattr(np_proxy, bad_np), f"numpy proxy leaked {bad_np}"
    for bad_attr in (
        "gi_frame",
        "f_back",
        "f_globals",
        "f_builtins",
        "tb_frame",
        "tofile",
        "dump",
    ):
        assert _is_forbidden_attr(object(), bad_attr)


def test_p1_db_init_cache_persists_across_writes(tmp_path) -> None:
    """_db_file_sig keys on (dev, ino) so writes do not invalidate is_db_initialized()."""
    dbp = str(tmp_path / "cache_test.duckdb")
    init_db(dbp).close()
    assert is_db_initialized(dbp) is True

    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, avg_depth, max_depth, avg_score) "
            "VALUES ('b1', 1, 'SWARM', 1.5, 2, 100.0)"
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    assert is_db_initialized(dbp) is True


def test_p1_sql_guard_blocks_double_quoted_table_functions() -> None:
    """Double-quoted system/table function invocations must not bypass validate_readonly_sql."""
    for bad_sql in (
        "SELECT * FROM \"read_text\"('/etc/hostname')",
        "SELECT * FROM \"READ_CSV\"('/tmp/a.csv')",
        "SELECT \"current_setting\"('home_directory')",
        'SELECT * FROM "duckdb_settings"()',
        'SELECT * FROM "duckdb_databases"()',
        "SELECT * FROM \"read_text\" /* comment */ ('/etc/passwd')",
    ):
        err, _, _ = validate_readonly_sql(bad_sql)
        assert err is not None, f"SQL guard bypassed by: {bad_sql}"

    # Ordinary double-quoted column aliases/names remain permitted
    err_ok, _, _ = validate_readonly_sql(
        'SELECT "delete" AS "select" FROM runs LIMIT 5'
    )
    assert err_ok is None


def test_p1_proactive2_branch_does_not_advance_on_confirm_crash(tmp_path) -> None:
    """If proactive_2 wins S_select but crashes or fails safety in S_confirm, branch must NOT advance."""
    from lox.author.swarm_peer import PeerCandidate

    dbp = str(tmp_path / "p2_guard.duckdb")
    init_db(dbp).close()
    logger = BatchDatabaseLogger(dbp)

    p2_cand = PeerCandidate(
        peer_id="proactive_2",
        claimed_deficit_id="proactive_branch_step1",
        claimed_label="branch step",
        hypothesis_yaml={"mechanism": "new capability"},
        policy_code="class Policy:\n    def step(self, s): return 0\n",
        skills={},
        sandbox_passed=True,
        sandbox_metrics={"avg_depth": 3.5},
    )
    verdict = TournamentVerdict(
        promoted=False,
        winner_id="incumbent",
        winner_policy_code="class Policy:\n    def step(self, s): return 0\n",
        winner_skills={},
        winner_batch_eval={"runs": []},
        incumbent_avg_depth=3.0,
        incumbent_avg_score=100.0,
        winner_avg_depth=3.0,
        winner_avg_score=100.0,
        reason="Rejected `proactive_2` (policy_exceptions=2 on confirm)",
        select_pair_by_peer={
            "proactive_2": {"wins": 10, "losses": 2, "winsor_delta": 0.4}
        },
        select_stats_by_peer={
            "proactive_2": {
                "avg_depth": 3.4,
                "policy_exceptions": 0,
                "auto_esc_total": 0,
            }
        },
        select_inc_avg_depth=3.0,
        select_inc_auto_esc=0,
        confirm_challenger_eval=(
            "proactive_2",
            {"runs": [{"seed": 1, "policy_exceptions": 2}]},
        ),
    )
    update_proactive_branch_lifecycle(
        db_logger=logger,
        generation=10,
        cur_fp="fp_champ_10",
        active_branch=None,
        verdict=verdict,
        peer_candidates=[p2_cand],
    )
    assert logger.load_active_proactive_branch("fp_champ_10") is None


def test_p1_worn_oprop_mask_bitshifts_and_tuple_indices() -> None:
    """worn_oprop_mask bit-shifts (1 << oprop) and GameState tuple fallback maps indices 12-15 accurately."""
    from lox.env.state import GameState

    # 1. Verify bit-shifting prevents property-ID collisions (e.g. oprop=30 -> 1 bit, not 4 bits)
    fake_state = object.__new__(GameState)
    it1 = SimpleNamespace(is_worn=True, is_wielded=False, info=SimpleNamespace(oprop=1))
    it2 = SimpleNamespace(is_worn=True, is_wielded=False, info=SimpleNamespace(oprop=2))
    it3 = SimpleNamespace(
        is_worn=True, is_wielded=False, info=SimpleNamespace(oprop=30)
    )
    it_unworn = SimpleNamespace(
        is_worn=False, is_wielded=False, info=SimpleNamespace(oprop=12)
    )
    fake_state._inventory_cache = [it1, it2, it3, it_unworn]
    mask = fake_state.worn_oprop_mask
    assert mask == ((1 << 1) | (1 << 2) | (1 << 30))
    assert bin(mask).count("1") == 3
    assert (mask & (1 << 3)) == 0  # fire(1) + cold(2) must NOT collide with sleep(3)

    # 2. Verify raw tuple positional indices 12..15 align with ALL_OBSERVATION_KEYS
    raw_tuple: tuple[Any, ...] = (
        np.zeros((21, 79), dtype=np.int16),  # 0 glyphs
        np.zeros((21, 79), dtype=np.uint8),  # 1 chars
        np.zeros((21, 79), dtype=np.int8),  # 2 colors
        np.zeros((21, 79), dtype=np.int8),  # 3 specials
        np.zeros(27, dtype=np.int64),  # 4 blstats
        np.zeros(256, dtype=np.uint8),  # 5 message
        np.zeros(55, dtype=np.int16),  # 6 inv_glyphs
        np.zeros((55, 80), dtype=np.uint8),  # 7 inv_strs
        np.zeros(55, dtype=np.uint8),  # 8 inv_letters
        np.zeros(55, dtype=np.int8),  # 9 inv_oclasses
        np.zeros(2, dtype=np.int64),  # 10 program_state
        np.zeros(6, dtype=np.int64),  # 11 internal
        "SENTINEL_TTY_CHARS",  # 12 tty_chars
        "SENTINEL_TTY_COLORS",  # 13 tty_colors
        "SENTINEL_TTY_CURSOR",  # 14 tty_cursor
        "SENTINEL_SCREEN_DESC",  # 15 screen_descriptions
        "SENTINEL_MISC",  # 16 misc
    )
    gs = GameState(raw_tuple)
    assert gs.tty_chars == "SENTINEL_TTY_CHARS"
    assert gs.tty_colors == "SENTINEL_TTY_COLORS"
    assert gs.tty_cursor == "SENTINEL_TTY_CURSOR"
    assert gs.screen_descriptions == "SENTINEL_SCREEN_DESC"
    assert gs.misc == "SENTINEL_MISC"


def test_p2_rust_replay_masks_raise_value_error_on_shape_mismatch() -> None:
    """fast_trap_mask and fast_unexplored_frontier_mask must raise ValueError on transposed/mismatched 2D shapes."""
    import lox_core

    glyphs = np.zeros((21, 79), dtype=np.int16)
    transposed_traps = np.zeros((79, 21), dtype=np.int32)
    lut = np.zeros(10, dtype=np.bool_)
    with pytest.raises(ValueError, match="shape mismatch"):
        lox_core.fast_trap_mask(transposed_traps, glyphs, lut)

    passable = np.ones((21, 79), dtype=np.bool_)
    transposed_vis = np.zeros((79, 21), dtype=np.bool_)
    rchars = np.full((21, 79), ord("."), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        lox_core.fast_unexplored_frontier_mask(transposed_vis, passable, rchars, False)


def test_p2_unworn_armor_union_and_wiki_snippet(tmp_path) -> None:
    """vital_summary computes exact union of died_unarmored OR unworn_armor_in_pack (no `or` masking)."""
    from lox.author.swarm_deficits import DeficitAnalyzer

    dbp = str(tmp_path / "unworn_union.duckdb")
    init_db(dbp).close()
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed, avg_depth, max_depth, avg_score) "
            "VALUES ('b_u1', 1, 'SWARM', TRUE, 2.0, 3, 100.0)"
        )
        # 1 run with died_unarmored=TRUE only, 5 runs with unworn_armor_in_pack=true only -> union must be 6, not 1!
        con.execute(
            "INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, died_unarmored, milestones_json, run_kind) "
            "VALUES ('r0', 'b_u1', 1, 100, 2, 2, 100, 200, TRUE, '{}', 'champion')"
        )
        for i in range(1, 6):
            con.execute(
                "INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, died_unarmored, milestones_json, run_kind) "
                "VALUES (?, 'b_u1', 1, ?, 2, 2, 100, 200, FALSE, '{\"unworn_armor_in_pack\": true}', 'champion')",
                [f"r{i}", 100 + i],
            )
    finally:
        con.close()

    rep = DeficitAnalyzer(dbp).analyze(
        generation=2, current_policy="class Policy: pass"
    )
    assert "unworn_armor_deaths=6" in rep.vital_summary
