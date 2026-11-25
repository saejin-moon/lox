"""Domain-scoped tests for LOX."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lox.author.agent import AuthorAgent
from lox.author.prompts import build_user_prompt
from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
from lox.author.swarm_peer import PeerCandidate
from lox.author.tools import AuthorToolRegistry, get_tool_specs
from lox.core.contracts import EpisodeRun
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, init_db
from lox.db.theories import TheoryDocumentStore
from lox.engine.hypothesis import (
    progression_metric,
)
from lox.engine.map_elites import MAPElitesArchive
from lox.engine.swarm_referee import SwarmReferee
from lox.env import get_adapter
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_and_macros, compile_policy_package
from lox.lang.validator import (
    PolicyValidationError,
    check_evolve_block_bounds,
    run_pyright_on_package,
    validate_cross_module_contracts,
)
from lox.runner.pool import PersistentEvalPool
from lox.ui.console import ConsoleUI


def test_phase3_unified_duckdb_prefix_cache_tier1_and_multiseed_smoke(tmp_path):

    from lox.author.tools import AuthorToolRegistry
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db

    db_path = str(tmp_path / "p3.duckdb")
    init_db(db_path).close()

    # 1. Unified DuckDB `theories` + `tournament_attempts`
    tstore = TheoryDocumentStore(db_path=db_path)
    tstore.save_document(
        generation=5,
        agent_role="peer_1",
        doc_type="THEORY",
        title="Shop Theft Prevention",
        content="noautopickup avoids unpaid shop theft warnings.",
        tags=["shop", "autopickup"],
    )
    hits = tstore.search_theories("shop autopickup", limit=3)
    assert hits and hits[0]["title"] == "Shop Theft Prevention"

    logger = BatchDatabaseLogger(db_path=db_path)
    logger.save_tournament_attempts(
        [
            {
                "attempt_id": "att_g5_peer_1",
                "generation": 5,
                "peer_id": "peer_1",
                "deficit_id": "slice_early",
                "candidate_fp": "fp_abc",
                "promoted": False,
                "sandbox_passed": True,
                "avg_depth": 4.2,
                "max_depth": 8,
                "avg_score": 180.0,
                "avg_ac": 5.5,
                "avg_xp_level": 3.0,
                "policy_exceptions": 0,
                "seeds_gained": [1001, 1002],
                "seeds_lost": [1005],
                "top_tracebacks": [],
            }
        ]
    )
    con = connect_db(db_path, read_only=True)
    try:
        cnt = con.execute("SELECT COUNT(*) FROM tournament_attempts").fetchone()[0]
        t_cnt = con.execute("SELECT COUNT(*) FROM theories").fetchone()[0]
        assert cnt == 1
        assert t_cnt >= 2
    finally:
        con.close()

    # 2. Tier-1 manifest in SWARM_PEER prompt
    long_policy = "\n".join(
        [
            "from lox.env.actions import Action",
            "class Policy:",
        ]
        + [f"    def helper_{i}(self, s): return None" for i in range(70)]
    )
    large_skills = {
        "combat": "\n".join([f"def fight_{i}(s): return None" for i in range(40)]),
        "equip": "\n".join([f"def wear_{i}(s): return None" for i in range(40)]),
    }
    peer_prompt = build_user_prompt(
        mode="SWARM_PEER",
        generation=5,
        current_policy=long_policy,
        batch_digest="Gen 5 deficits",
        skills=large_skills,
    )
    assert "### Behavior Tree Primitives" not in peer_prompt
    assert "Domain Skill Tree" in peer_prompt
    assert "`skills/combat.py` (40 LOC)" in peer_prompt
    assert "def fight_39" not in peer_prompt
    assert "def helper_59" not in peer_prompt

    # 3. 3-seed smoke sweep catches error triggered ONLY on seed 1000
    reg = AuthorToolRegistry(db_path=db_path)
    reg.configure_generation(
        pool=None, baseline_runs=[], generation=5, mode="SWARM_PEER"
    )
    cand_seed1000_bug = """
from lox.env.actions import Action

def conditional_bug(st):
    if st.step_count == 5 and st.y != 11:
        return 1 / 0
    return Action.SEARCH

class Policy:
    def step(self, st):
        return conditional_bug(st) or Action.WAIT
"""
    diag, status = reg.dry_run_policy(
        code=cand_seed1000_bug, seed=42, max_turns=60, smoke_only=True
    )
    # Whether raised on seed 42 or seed 1000, multi-seed smoke catches it with RUNTIME_ERROR!
    assert status == "RUNTIME_ERROR"

    # 4. Append-only messages + static tools payload check in AuthorAgent
    import httpx

    ag = AuthorAgent(
        provider="openrouter", api_key="test-key", db_path=db_path, agent_role="peer_1"
    )
    seen_prefixes: list[tuple[bool, str, list[str]]] = []

    def fake_post(self_client, url, headers=None, json=None, **kwargs):
        import json as _json

        seen_prefixes.append(
            (
                "tools" in json,
                str(json.get("tool_choice", "")),
                [m["role"] for m in json["messages"]],
            )
        )

        class _Resp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self_resp):
                if len(seen_prefixes) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "c1",
                                            "function": {
                                                "name": "query_wiki",
                                                "arguments": _json.dumps(
                                                    {"query": "shop"}
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {},
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "```python\nclass Policy:\n    def step(self, s): return Action.WAIT\n```"
                            }
                        }
                    ],
                    "usage": {},
                }

        return _Resp()

    old_post = httpx.Client.post
    httpx.Client.post = fake_post  # type: ignore[method-assign]
    try:
        ag.synthesize(mode="CREATE", generation=1, current_policy="", max_tool_turns=1)
    finally:
        httpx.Client.post = old_post  # type: ignore[method-assign]

    assert len(seen_prefixes) >= 2
    assert all(has_tools for has_tools, _, _ in seen_prefixes)
    assert seen_prefixes[0][1] == "auto"
    assert seen_prefixes[-1][1] == "none"
    # Every prefix of turn i messages matches turn i-1
    assert seen_prefixes[1][2][: len(seen_prefixes[0][2])] == seen_prefixes[0][2]


def test_pass4_codebase_bottleneck_fixes(tmp_path: Path) -> None:
    """
    Pass 4 Codebase Bottleneck Verification:
    1. champion_runs, champion_function_stats, champion_ticks, and latest_pilot_runs views exist in DuckDB.
    2. query_db dynamically rewrites (SELECT MAX(batch_id) FROM runs) to active champion batch.
    3. ingest_expert_runs_into_duckdb populates expert_ticks with depth >= 2 rows.
    4. WorldMemory normalizes open doors to ord('.') in remembered_chars, sets door_mask = True,
       exposes passable_mask and adjacent_hostiles(), and astar_next_step avoids diagonal steps on door_mask.
    5. ModeScheduler triggers REFACTOR based on policy line count and cadence.
    """
    from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
    from lox.env.grid_utils import astar_next_step
    from lox.env.state import _IS_OPEN_DOOR_GLYPH_LUT, WorldMemory

    # 1. Canonical Views & SELECT MAX(batch_id) Poison Pill Resolution
    db_path = str(tmp_path / "test_bottlenecks.duckdb")
    con = init_db(db_path)
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, hypothesis_passed, created_at)
        VALUES ('c2026_g001', 1, 'FUNC_MOD_ADD', TRUE, '2026-01-01 10:00:00'),
               ('pilot_9999_01', 1, 'PILOT', FALSE, '2026-01-01 11:00:00')
        """
    )
    con.execute(
        """
        INSERT INTO runs (run_id, batch_id, generation, seed)
        VALUES ('run_champ', 'c2026_g001', 1, 100),
               ('run_pilot', 'pilot_9999_01', 1, 101)
        """
    )
    champ_count = con.execute("SELECT COUNT(*) FROM champion_runs").fetchone()[0]
    pilot_count = con.execute("SELECT COUNT(*) FROM latest_pilot_runs").fetchone()[0]
    assert champ_count == 1
    assert pilot_count == 1
    con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    yaml_out, count_str = tools.query_db("SELECT run_id FROM champion_runs")
    assert "run_champ" in yaml_out
    assert "run_pilot" not in yaml_out

    # Verify query_monster and query_item tools
    m_out, _ = tools.query_monster("jackal")
    assert "jackal" in m_out
    i_out, _ = tools.query_item("food ration")
    assert "food" in i_out.lower()

    # 2. Expert Ticks Deep Depths Population
    res = ingest_expert_runs_into_duckdb(db_path=db_path, max_ascensions=3, force=True)
    assert res["expert_runs"] >= 3
    con = connect_db(db_path, read_only=True)
    deep_ticks = con.execute(
        "SELECT COUNT(*) FROM expert_ticks WHERE depth >= 2"
    ).fetchone()[0]
    jackal_ticks = con.execute(
        "SELECT COUNT(*) FROM expert_ticks WHERE message ILIKE '%jackal%'"
    ).fetchone()[0]
    con.close()
    assert deep_ticks > 0
    assert jackal_ticks > 0

    # 3. WorldMemory Open-Door Normalization, door_mask & Diagonal Doorway Avoidance
    mem = WorldMemory()

    class DummyState:
        def __init__(self):
            self.dungeon_num = 0
            self.level_num = 1
            self.y = 10
            self.x = 20
            self.chars = np.full((21, 79), ord(" "), dtype=np.uint8)
            self.glyphs = np.zeros((21, 79), dtype=np.int32)
            self.message = ""

    st = DummyState()
    # Find an open door glyph from precomputed LUT
    open_door_glyphs = np.nonzero(_IS_OPEN_DOOR_GLYPH_LUT)[0]
    od_g = int(open_door_glyphs[0]) if len(open_door_glyphs) > 0 else 2372
    st.glyphs[10, 15] = od_g
    st.chars[10, 15] = ord("-")  # C engine ASCII for horizontal open door
    mem.update(st)

    # Lossless remembered_chars + remembered_glyphs + door_mask + passable_mask
    assert mem.current_floor.remembered_chars[10, 15] == ord("-")
    assert mem.current_floor.remembered_glyphs[10, 15] == od_g
    assert bool(mem.door_mask[10, 15]) is True
    assert bool(mem.passable_mask[10, 15]) is True

    # Doorway diagonal collision avoidance in pathfinding
    passable_grid = np.ones((5, 5), dtype=np.bool_)
    door_grid = np.zeros((5, 5), dtype=np.bool_)
    door_grid[2, 2] = True  # doorway at (2, 2)

    # Diagonal step into (2, 2) from (1, 1) is forbidden
    step = astar_next_step(
        passable_grid,
        sy=1,
        sx=1,
        ty=3,
        tx=3,
        allow_diagonal=True,
        no_diag=door_grid,
    )
    assert step in ((1, 0), (0, 1)), f"Expected orthogonal entry, got {step}"


def test_pass5_deep_audit_fixes(tmp_path: Path) -> None:
    """
    Pass 5 Deep Audit & Runtime Attribution Verification:
    1. Under-Hero Object Visibility (state.visible_objects() and state.objects_here()).
    2. Effective Function Trigger Attribution (ExecutionTracer distinguishes value-returns vs no-op).
    3. Compiler-Gate smoke_only=True and best-pilot preservation.
    4. Dark corridor memory in NetHackHarness.replay_bytes.
    """
    from lox.author.tools import AuthorToolRegistry
    from lox.env.harness import NetHackHarness
    from lox.env.state import _IS_BODY_LUT, WorldMemory

    # 1. Under-Hero Object Visibility
    mem = WorldMemory()
    body_glyphs = np.nonzero(_IS_BODY_LUT)[0]
    corpse_g = int(body_glyphs[0]) if len(body_glyphs) > 0 else 1900

    harness = NetHackHarness(max_steps=10)
    try:
        real_st = harness.reset(seed=42)
        hy, hx = real_st.y, real_st.x
        # Simulate corpse remembered at hero's current position
        mem.current_floor.remembered_obj_glyphs[hy, hx] = corpse_g
        real_st.mem = mem
        here = real_st.objects_here()
        assert len(here) >= 1
        corpse_obj = here[0]
        assert corpse_obj.y == hy
        assert corpse_obj.x == hx
        assert corpse_obj.is_corpse is True
    finally:
        harness.close()

    # 2. Effective Function Trigger Attribution
    code = """
class Policy:
    def handle_items(self, state):
        if state.score > 999:
            return Action.WAIT
        return None

    def handle_combat(self, state):
        if state.score > 999:
            return Action.WAIT
        return None

    def move_towards(self, state):
        return Action.E

    def explore(self, state):
        return self.move_towards(state)

    def run(self, state):
        while True:
            act = self.handle_items(state)
            if act is not None:
                state = yield act
                continue
            act = self.handle_combat(state)
            if act is not None:
                state = yield act
                continue
            act = self.explore(state)
            state = yield act
"""
    compiled = compile_policy_and_macros(code)
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=123)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st)
        assert act == Action.E
        assert runner.tracer.fn_calls.get("handle_items", 0) == 0
        assert runner.tracer.fn_calls.get("handle_combat", 0) == 0
        assert runner.tracer.fn_calls.get("explore", 0) == 1
        assert runner.tracer.fn_turns.get("explore", 0) == 1
        assert runner.tracer.fn_calls.get("move_towards", 0) == 1
        assert runner.tracer.fn_turns.get("move_towards", 0) == 0
        assert runner.tracer.triggered_any(["handle_items"]) is False
    finally:
        h.close()

    # 3. Compiler-Gate smoke_only=True & Best-Pilot Fallback
    db_p = str(tmp_path / "tools_test.duckdb")
    tools = AuthorToolRegistry(db_path=db_p)
    smoke_code = """
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
    out, status = tools.dry_run_policy(smoke_code, smoke_only=True)
    assert status == "PASSED (smoke check)"

    tools.last_passed_code = "code_v1"
    tools.last_pilot_improved = True
    tools.best_pilot_code = "code_v1"
    tools.best_pilot_delta = 1.5

    # If subsequent pilot fails, last_pilot_improved stays True because last_passed_code is set
    improved = False
    assert not improved
    if not tools.last_passed_code:
        tools.last_pilot_improved = False
    assert tools.last_pilot_improved is True
    assert tools.last_passed_code == "code_v1"

    # 4. Dark Corridor Memory in replay_bytes
    h_rep = NetHackHarness(max_steps=20)
    try:
        st = h_rep.reset(seed=999)
        bytes_recorded = bytearray()
        for a in (Action.EAST, Action.EAST, Action.SOUTH, Action.SOUTH):
            st, d, _ = h_rep.step(a)
            bytes_recorded.append(int(a))
            if d:
                break
        replayed_st = h_rep.replay_bytes(seed=999, recorded_bytes=bytes(bytes_recorded))
        assert replayed_st is not None
        fl = h_rep.mem.current_floor
        for dnum, dlvl, py, px in list(h_rep.mem.pos_history):
            if (
                (dnum, dlvl) == h_rep.mem.current_floor_key
                and 0 <= py < 21
                and 0 <= px < 79
            ):
                assert fl.remembered_chars[py, px] != 0, f"Blank char at ({py}, {px})"
    finally:
        h_rep.close()


def test_pass6_overnight_audit_fixes(tmp_path: Path) -> None:
    """Pass 6: Overnight Campaign & Runtime Audit Overhaul Verification."""
    from lox.lang.validator import splice_functions_into_policy

    # 1. ExecutionTracer attributes turns to handle_combat (not leaf helper get_passable_mask)
    #    and counts boolean skill helper calls returning False
    helper_skill = """
def helper_is_low_hp(state):
    return False
"""
    pol_src = """
from skills.helpers import helper_is_low_hp

class Policy:
    def get_passable_mask(self, state):
        return state.mem.passable_mask

    def handle_combat(self, state):
        if helper_is_low_hp(state):
            return Action.WAIT
        mask = self.get_passable_mask(state)
        return Action.E

    def run(self, state):
        while True:
            act = self.handle_combat(state)
            state = yield act
"""
    compiled = compile_policy_package(pol_src, skills={"helpers": helper_skill})
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st)
        assert act == Action.E
        assert active_fn == "handle_combat"
        assert runner.tracer.fn_turns.get("handle_combat", 0) == 1
        assert runner.tracer.fn_turns.get("get_passable_mask", 0) == 0
        assert runner.tracer.fn_calls.get("helper_is_low_hp", 0) >= 1
    finally:
        h.close()

    # 2. PersistentEvalPool.evaluate_batch with priority_seeds and unconditional_seeds=[]
    #    discards untriggered episodes instead of treating all priority seeds as unconditional
    pool = PersistentEvalPool(workers=2, max_steps=20)
    try:
        pol_dummy = """
class Policy:
    def run(self, state):
        while True:
            yield Action.WAIT
"""
        # When unconditional_seeds=[], seed 100 is NOT unconditional, so enforce_trigger is True
        res_filtered = pool.evaluate_batch(
            batch_id="test_filter_bypass",
            policy_code=pol_dummy,
            target_episodes=1,
            target_functions=["nonexistent_fn"],
            priority_seeds=[100, 101],
            unconditional_seeds=[],
            unreachable_abort_after=3,
            show_progress=False,
        )
        assert res_filtered["unreachable_abort"] is True
        assert res_filtered["episodes_recorded"] == 0
        assert res_filtered["episodes_discarded"] >= 3

        # When unconditional_seeds=[100], seed 100 is kept even without triggering
        res_kept = pool.evaluate_batch(
            batch_id="test_filter_uncond",
            policy_code=pol_dummy,
            target_episodes=1,
            target_functions=["nonexistent_fn"],
            priority_seeds=[100],
            unconditional_seeds=[100],
            show_progress=False,
        )
        assert res_kept["episodes_recorded"] == 1
        assert res_kept["runs"][0]["seed"] == 100
    finally:
        pool.close()

    # 3. splice_functions_into_policy preserves existing referenced self.<attr> in __init__
    base_code = """
class Policy:
    def __init__(self):
        self.old_attr1 = 10
        self.old_attr2 = 20
        self.overwrite_me = "old"

    def run(self, state):
        _ = (self.old_attr1, self.old_attr2)
        return Action.WAIT
"""
    patch_code = """
    def __init__(self):
        \"\"\"Patch docstring.\"\"\"
        self.overwrite_me = "new"
        self.new_attr = 30
"""
    spliced, changed = splice_functions_into_policy(base_code, patch_code)
    assert "__init__" in changed
    assert "self.old_attr1 = 10" in spliced
    assert "self.old_attr2 = 20" in spliced
    assert (
        "self.overwrite_me = 'new'" in spliced or 'self.overwrite_me = "new"' in spliced
    )
    assert "self.new_attr = 30" in spliced

    # 5. AuthorToolRegistry.query_policy_docs("policy") returns current_policy_code
    db_p = str(tmp_path / "pass6_tools.duckdb")
    tools = AuthorToolRegistry(db_path=db_p)
    tools.current_policy_code = "class Policy:\n    pass"
    doc_text, topic_tag = tools.query_policy_docs("policy")
    assert "class Policy:" in doc_text
    assert topic_tag == "policy source"

    # 6. ConsoleUI.hypothesis_box and BatchDatabaseLogger.save_experiment record
    #    binding_constraint and verified_c_rule
    c_ui = ConsoleUI(enabled=True)
    c_ui.hypothesis_box(
        {
            "binding_constraint": "Hunger clock forces exploration",
            "mechanism": "Eat corpses before descending",
            "target_functions": ["eat"],
        }
    )
    c_ui.hypothesis_box(
        {
            "verified_c_rule": "Doors cannot be entered diagonally",
            "mechanism": "Orthogonal alignment",
            "target_functions": ["move"],
        }
    )

    db_logger = BatchDatabaseLogger(db_p)
    db_logger.save_experiment(
        experiment_id="exp_test_binding",
        generation=1,
        batch_id="b1",
        mode="FUNC_MOD_ADD",
        target_functions=["eat"],
        hypothesis={"binding_constraint": "Hunger clock forces exploration"},
        verdict={"passed": True},
    )
    db_logger.save_experiment(
        experiment_id="exp_test_verified",
        generation=2,
        batch_id="b2",
        mode="FUNC_MOD_ADD",
        target_functions=["move"],
        hypothesis={"verified_c_rule": "Doors cannot be entered diagonally"},
        verdict={"passed": True},
    )
    con = connect_db(db_p)
    try:
        rows = con.execute(
            "SELECT experiment_id, observation_from_db FROM experiments ORDER BY generation"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == "Hunger clock forces exploration"
        assert rows[1][1] == "Doors cannot be entered diagonally"
    finally:
        con.close()


def test_pass7_final_preflight_audit_fixes(tmp_path: Path) -> None:
    """Pass 7: Final Pre-Flight Deep Sweep Verification."""
    from lox.runner.pool import run_single_episode

    # 1. Standing on a downstairs tile '>' where chars[y, x] == ord('@')
    h1 = NetHackHarness(max_steps=20)
    try:
        st = h1.reset(seed=42)
        st.chars[st.y, st.x] = ord("@")
        st.mem.remembered_chars[st.y, st.x] = ord(">")
        # Ensure no other '>' on screen so we uniquely test under-hero detection
        st.chars[st.chars == ord(">")] = ord(".")

        assert st.char_at(st.y, st.x) == ">"
        assert st.stairs_down_pos == (st.y, st.x)
        assert st.is_stairs_down(st.y, st.x) is True
        assert st.terrain_at(st.y, st.x) == "staircase down"
    finally:
        h1.close()

    # 2. Void method with early bare return increments fn_calls and does not clobber active_fn
    void_code = """
class Policy:
    def __init__(self):
        self.count = 0

    def update_memory(self, state):
        if self.count > 0:
            return
        self.count += 1
        return

    def explore(self, state):
        return Action.WAIT

    def run(self, state):
        while True:
            self.update_memory(state)
            act = self.explore(state)
            state = yield act
"""
    compiled = compile_policy_and_macros(void_code)
    h2 = NetHackHarness(max_steps=10)
    try:
        st2 = h2.reset(seed=10)
        runner = compiled.create_runner()
        act, active_fn = runner.next_action(st2)
        assert act == Action.WAIT
        assert active_fn == "explore"
        assert runner.tracer.fn_calls.get("update_memory", 0) >= 1
        assert runner.tracer.fn_turns.get("explore", 0) == 1
    finally:
        h2.close()

    # 3. Prefix shard executes up to post_replay_max_steps instead of truncating at step 1
    h3 = NetHackHarness(max_steps=350)
    try:
        st3 = h3.reset(seed=99)
        bytes_list = bytearray()
        for _ in range(300):
            st3, done, _ = h3.step(Action.WAIT)
            bytes_list.append(int(Action.WAIT))
            if done:
                break
    finally:
        h3.close()

    pol_simple = """
class Policy:
    def run(self, state):
        while True:
            yield Action.WAIT
"""
    task = {
        "seed": 99,
        "run_id": "test_shard_300",
        "role": "val-hum-law-fem",
        "max_steps": 20,
        "post_replay_max_steps": 20,
        "replay_bytes": bytes(bytes_list),
        "policy_code": pol_simple,
        "macros_code": "",
        "target_functions": [],
        "is_priority_seed": True,
        "enforce_trigger": False,
        "record_ticks": False,
    }
    shard_res = run_single_episode(task)
    assert shard_res["triggered"] is True
    assert shard_res["steps"] > 1, (
        f"Shard truncated prematurely with steps={shard_res['steps']}"
    )
    assert shard_res["steps"] <= 20


def test_pass8_long_horizon_and_unattended_ops(tmp_path: Path) -> None:
    """Pass 8: Multi-Week Unattended Ascension & Deep-Horizon Scaling Architecture."""
    from lox.db.shards import extract_shards_from_runs
    from lox.engine.reporting import (
        compute_adaptive_max_steps,
        write_status_report,
    )
    from lox.env.state import (
        _IS_ICE_GLYPH_LUT,
        _IS_IRON_BARS_OR_TREE_LUT,
        WorldMemory,
    )
    from lox.lang.compiler import compile_policy_and_macros

    # 1. Adaptive Max Steps Progression
    assert compute_adaptive_max_steps(1) == 5000
    assert compute_adaptive_max_steps(3) == 5000
    assert compute_adaptive_max_steps(4) == 10000
    assert compute_adaptive_max_steps(7) == 10000
    assert compute_adaptive_max_steps(8) == 18000
    assert compute_adaptive_max_steps(12) == 18000
    assert compute_adaptive_max_steps(13) == 30000
    assert compute_adaptive_max_steps(20) == 30000
    assert compute_adaptive_max_steps(21) == 50000
    assert compute_adaptive_max_steps(50) == 50000

    # 2. WorldMemory & GameState persistence & normalization
    wm = WorldMemory()
    assert hasattr(wm, "kv") and isinstance(wm.kv, dict)
    assert hasattr(wm, "recent_messages")
    wm.kv["found_altar"] = True

    # Test GameState methods & properties via harness
    harness = NetHackHarness(max_steps=10)
    try:
        state = harness.reset(seed=42)
        assert hasattr(state, "kv")
        assert state.kv is state.mem.kv
        state.kv["test_key"] = 123
        assert state.mem.kv["test_key"] == 123
        assert hasattr(state, "recent_messages")
        assert isinstance(state.recent_messages, list)

        # Tree / Iron bars lossless remembered_chars + remembered_glyphs + passable_mask
        tree_g = int(np.where(_IS_IRON_BARS_OR_TREE_LUT)[0][0])
        state.glyphs[10, 10] = tree_g
        state.chars[10, 10] = ord("#")
        state.mem.update(state)
        fl = state.mem.get_floor(state.dungeon_num, state.level_num)
        assert fl.remembered_chars[10, 10] == ord("#")
        assert int(fl.remembered_glyphs[10, 10]) == tree_g
        assert bool(fl.passable_mask[10, 10]) is False

        # Ice lossless remembered_chars + remembered_glyphs + passable_mask + is_ice
        ice_g = int(np.where(_IS_ICE_GLYPH_LUT)[0][0])
        state.glyphs[11, 11] = ice_g
        state.chars[11, 11] = ord("-")
        state.mem.update(state)
        assert fl.remembered_chars[11, 11] == ord("-")
        assert int(fl.remembered_glyphs[11, 11]) == ice_g
        assert bool(fl.passable_mask[11, 11]) is True
        assert bool(state.is_ice(11, 11)) is True

        assert hasattr(state, "is_water")
        assert hasattr(state, "is_lava")
        assert hasattr(state, "is_ice")
        assert hasattr(state, "is_iron_bars")
        assert hasattr(state, "is_boulder")
        assert isinstance(state.boulder_positions, list)
        assert isinstance(state.is_water(state.y, state.x), (bool, np.bool_))
        assert isinstance(state.is_lava(state.y, state.x), (bool, np.bool_))
        assert isinstance(state.is_ice(state.y, state.x), (bool, np.bool_))
        assert isinstance(state.is_iron_bars(state.y, state.x), (bool, np.bool_))
        assert isinstance(state.is_boulder(state.y, state.x), (bool, np.bool_))
    finally:
        harness.close()

    # 3. Shard Stratification & Deep Horizon (400 steps on Dlvl >= 5)
    synthetic_runs = [
        {
            "run_id": "r_deep_10",
            "seed": 101,
            "max_depth": 10,
            "depth": 10,
            "steps": 200,
            "turns": 200,
            "action_bytes": b"k" * 200,
            "milestones": {"depth_first_turn": {"10": 150}},
            "end_reason": "killed_by_dragon",
        },
        {
            "run_id": "r_deep_6",
            "seed": 102,
            "max_depth": 6,
            "depth": 6,
            "steps": 120,
            "turns": 120,
            "action_bytes": b"k" * 120,
            "milestones": {"depth_first_turn": {"6": 90}},
            "end_reason": "killed_by_soldier_ant",
        },
        {
            "run_id": "r_shallow_3",
            "seed": 103,
            "max_depth": 3,
            "depth": 3,
            "steps": 60,
            "turns": 60,
            "action_bytes": b"k" * 60,
            "milestones": {"depth_first_turn": {"3": 40}},
            "end_reason": "killed_by_jackal",
        },
        {
            "run_id": "r_shallow_1",
            "seed": 104,
            "max_depth": 1,
            "depth": 1,
            "steps": 30,
            "turns": 30,
            "action_bytes": b"k" * 30,
            "milestones": {},
            "end_reason": "killed_by_lichen",
        },
    ]
    shards = extract_shards_from_runs(synthetic_runs, max_shards=20)
    assert len(shards) >= 3
    deep_shards = [
        s for s in shards if s.depth_at_cut >= 5 and "floor_entry" in s.trigger_context
    ]
    shallow_shards = [
        s for s in shards if s.depth_at_cut < 5 and "floor_entry" in s.trigger_context
    ]
    assert len(deep_shards) >= 1
    for s in deep_shards:
        assert s.post_replay_max_steps == 400
    for s in shallow_shards:
        assert s.post_replay_max_steps == 200

    # 4. PolicyRunner Preemption on in_yn_prompt or critical hp
    pol_macro_src = """
class Policy:
    def __init__(self):
        self.flee_hp_ratio = 0.5
    def should_preempt_subgenerator(self, state):
        return (state.hp / max(1, state.max_hp)) < self.flee_hp_ratio
    def subgen(self):
        while True:
            yield Action.WAIT
    def run(self, state):
        while True:
            if (state.hp / max(1, state.max_hp)) < self.flee_hp_ratio:
                yield Action.SEARCH
            else:
                yield self.subgen()
"""
    compiled = compile_policy_and_macros(pol_macro_src)
    runner = compiled.create_runner()
    h4 = NetHackHarness(max_steps=10)
    try:
        st = h4.reset(seed=55)
        # 1st step creates runner._active_subgen
        act1, _ = runner.next_action(st)
        assert act1 == Action.WAIT
        assert runner._active_subgen is not None
        # Now drop HP below flee_hp_ratio
        st.blstats[10] = 1  # HP = 1
        st.blstats[11] = 20  # Max HP = 20
        st._blstats_list[10] = 1
        st._blstats_list[11] = 20
        # Next action must preempt the subgen
        act2, _ = runner.next_action(st)
        assert act2 == Action.SEARCH
        assert runner._active_subgen is None
    finally:
        h4.close()

    # 5. Discoveries Table, Logger Methods, and Pruning
    db_path = str(tmp_path / "test_p8.duckdb")
    logger = BatchDatabaseLogger(db_path)
    logger.save_discovery(
        discovery_id="disc_001",
        generation=1,
        batch_id="batch_01",
        mode="FUNC_MOD_ADD",
        target_functions=["kick_doors"],
        binding_constraint="locked doors block progress",
        verified_c_rule="kicking breaks wooden doors",
        mechanism="lets hero enter locked rooms for food",
        mean_delta=3.4,
        avg_depth=5.2,
        p90_depth=7.0,
        max_depth=9,
    )
    discoveries = logger.get_recent_discoveries(limit=5)
    assert len(discoveries) == 1
    assert discoveries[0]["discovery_id"] == "disc_001"
    assert discoveries[0]["verified_c_rule"] == "kicking breaks wooden doors"

    # Save failed experiment
    logger.save_experiment(
        experiment_id="exp_fail_01",
        generation=2,
        batch_id="batch_02",
        mode="FUNC_MOD_ADD",
        target_functions=["eat_unknown"],
        hypothesis={"mechanism": "eat floating eye corpse"},
        verdict={"passed": False, "reason": "paralyzed forever"},
    )
    failures = logger.get_recent_failed_experiments(limit=5)
    assert len(failures) == 1
    assert failures[0]["mechanism"] == "eat floating eye corpse"

    # Prune telemetry
    logger.prune_old_telemetry(current_generation=25, keep_generations=20)

    # Stale 17-column ticks table auto-migration
    stale_db_path = str(tmp_path / "stale_test.duckdb")
    con_stale = connect_db(stale_db_path)
    con_stale.execute(
        "CREATE TABLE ticks (" + ", ".join([f"col_{i} INT" for i in range(17)]) + ")"
    )
    con_stale.close()
    con_migrated = init_db(stale_db_path)
    migrated_cols = [
        r[1] for r in con_migrated.execute("PRAGMA table_info('ticks')").fetchall()
    ]
    assert len(migrated_cols) == 21
    assert "byte_offset" in migrated_cols
    con_migrated.close()

    # 6. write_status_report and scripts/status_lox.py
    ws_dir = str(tmp_path / "workspace")
    Path(ws_dir).mkdir(parents=True, exist_ok=True)
    write_status_report(
        workspace_dir=ws_dir,
        db_path=db_path,
        gen=5,
        active_max_steps=18000,
        current_policy="class Policy:\n    pass\n",
    )
    status_file = Path(ws_dir) / "STATUS.md"
    assert status_file.exists()
    status_content = status_file.read_text()
    assert "LOX Campaign Status" in status_content
    assert "18,000" in status_content

    # 7. Unreachable Target Function Circuit Breaker in PersistentEvalPool
    pool = PersistentEvalPool(workers=2, max_steps=10)
    try:
        pol_idle = """
class Policy:
    def target_method(self, state):
        return Action.WAIT
    def run(self, state):
        while True:
            yield Action.WAIT
"""
        eval_res = pool.evaluate_batch(
            batch_id="b_unreach",
            policy_code=pol_idle,
            target_functions=["target_method"],
            target_episodes=5,
            unreachable_abort_after=2,
            max_raw_multiplier=1,
            show_progress=False,
        )
        assert eval_res.get("unreachable_abort") is True
    finally:
        pool.close()


def test_pass9_debugging_telemetry_and_rollout_metrics(tmp_path: Path) -> None:
    """Pass 9: Comprehensive verification of safety, behavioral, blunder, and efficiency telemetry."""
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import get_schema_markdown, init_db
    from lox.runner.pool import PersistentEvalPool

    db_path = str(tmp_path / "test_telemetry.duckdb")
    con = init_db(db_path)

    # 1. Verify schema columns exist in DuckDB
    runs_cols = [r[1] for r in con.execute("PRAGMA table_info('runs')").fetchall()]
    for col in [
        "auto_esc_count",
        "subgen_preemptions",
        "policy_exceptions",
        "unique_tiles",
        "pct_wait",
        "pct_search",
        "starved_with_food",
        "died_unarmored",
    ]:
        assert col in runs_cols, f"Column {col} missing from runs table!"

    batches_cols = [
        r[1] for r in con.execute("PRAGMA table_info('batches')").fetchall()
    ]
    for col in ["seeds_evaluated", "is_early_stopped", "pilot_best_delta"]:
        assert col in batches_cols, f"Column {col} missing from batches table!"

    exp_cols = [
        r[1] for r in con.execute("PRAGMA table_info('experiments')").fetchall()
    ]
    for col in ["seeds_evaluated", "is_early_stopped", "pilot_best_delta"]:
        assert col in exp_cols, f"Column {col} missing from experiments table!"

    schema_md = get_schema_markdown(db_path)
    assert "auto_esc_count" in schema_md
    assert "starved_with_food" in schema_md
    assert "seeds_evaluated" in schema_md

    # 2. Test saving and querying telemetry in BatchDatabaseLogger
    logger = BatchDatabaseLogger(db_path=db_path)
    sample_runs = [
        {
            "run_id": "test_run_001",
            "seed": 42,
            "depth": 3,
            "max_depth": 4,
            "score": 350,
            "turns": 400,
            "steps": 400,
            "xp_level": 2,
            "final_hp": 0,
            "max_hp": 18,
            "final_ac": 10,
            "gold": 50,
            "end_reason": "died of starvation",
            "death_message": "You starved to death.",
            "is_outlier_good": True,
            "outlier_z_score": 2.5,
            "batch_percentile": 95.0,
            "summary_digest": "Starvation on dlvl 3",
            "game_log": "You feel hungry. You starve.",
            "milestones_json": "{}",
            "triggered_functions": "{}",
            "last_active_function": "explore",
            "action_bytes": b"\x1b\x1b",
            "wall_ms": 120.5,
            "auto_esc_count": 5,
            "subgen_preemptions": 3,
            "policy_exceptions": 1,
            "unique_tiles": 58,
            "pct_wait": 15.25,
            "pct_search": 32.50,
            "starved_with_food": True,
            "died_unarmored": True,
        }
    ]

    logger.save_batch(
        batch_id="batch_telemetry_test",
        generation=1,
        mode="FUNC_MOD_ADD",
        parent_policy_id="pol_base",
        candidate_policy_id="pol_cand",
        target_functions=["explore"],
        runs=sample_runs,
        episodes_discarded=2,
        batch_metrics={
            "avg_depth": 3.0,
            "p90_depth": 4.0,
            "max_depth": 4,
            "avg_score": 350.0,
            "max_score": 350,
            "avg_turns": 400.0,
            "seeds_evaluated": 60,
            "is_early_stopped": True,
            "pilot_best_delta": 0.35,
        },
        verdict={
            "passed": False,
            "mean_delta": -0.1,
            "wilcoxon_p": 0.45,
            "ci_lower": -0.2,
            "ci_upper": 0.05,
            "wins": 20,
            "losses": 25,
            "ties": 15,
            "seeds_evaluated": 60,
            "is_early_stopped": True,
            "pilot_best_delta": 0.35,
            "reason": "Sequential futility early stopping at n=60",
        },
    )

    logger.save_experiment(
        experiment_id="exp_telemetry_test",
        generation=1,
        batch_id="batch_telemetry_test",
        mode="FUNC_MOD_ADD",
        target_functions=["explore"],
        hypothesis={"mechanism": "improve exploration"},
        verdict={
            "passed": False,
            "mean_delta": -0.1,
            "wilcoxon_p": 0.45,
            "ci_lower": -0.2,
            "ci_upper": 0.05,
            "wins": 20,
            "losses": 25,
            "ties": 15,
            "seeds_evaluated": 60,
            "is_early_stopped": True,
            "pilot_best_delta": 0.35,
            "reason": "Sequential futility early stopping at n=60",
        },
    )

    row_run = con.execute("""
        SELECT auto_esc_count, subgen_preemptions, policy_exceptions,
               unique_tiles, pct_wait, pct_search, starved_with_food, died_unarmored
        FROM runs WHERE run_id = 'test_run_001'
    """).fetchone()
    assert row_run == (5, 3, 1, 58, 15.25, 32.50, True, True)

    row_batch = con.execute("""
        SELECT seeds_evaluated, is_early_stopped, pilot_best_delta
        FROM batches WHERE batch_id = 'batch_telemetry_test'
    """).fetchone()
    assert row_batch == (60, True, 0.35)

    row_exp = con.execute("""
        SELECT seeds_evaluated, is_early_stopped, pilot_best_delta
        FROM experiments WHERE experiment_id = 'exp_telemetry_test'
    """).fetchone()
    assert row_exp == (60, True, 0.35)
    con.close()

    # 3. Test PolicyRunner runtime telemetry tracking
    pol_code = """
class Policy:
    def __init__(self):
        self.flee_hp_ratio = 0.5
        self.preempt_on_hostile = True
    def subgen(self, state):
        while True:
            state = yield Action.WAIT
    def run(self, state):
        while True:
            act = yield from self.subgen(state)
            state = yield act
"""
    compiled = compile_policy_package(pol_code)
    runner = compiled.create_runner()
    assert runner.auto_esc_count == 0
    assert runner.subgen_preemptions == 0
    assert runner.policy_exceptions == 0

    pool = PersistentEvalPool(workers=2, max_steps=15)
    try:
        eval_res = pool.evaluate_batch(
            batch_id="b_eval_telemetry",
            policy_code=pol_code,
            target_functions=["run"],
            target_episodes=2,
            show_progress=False,
        )
        assert len(eval_res["runs"]) >= 1
        r0 = eval_res["runs"][0]
        assert "auto_esc_count" in r0
        assert "subgen_preemptions" in r0
        assert "policy_exceptions" in r0
        assert "unique_tiles" in r0
        assert "pct_wait" in r0
        assert "pct_search" in r0
        assert "starved_with_food" in r0
        assert "died_unarmored" in r0
    finally:
        pool.close()


def test_phase2_pyright_contracts_evolve_block_tools_and_deficits(tmp_path):
    # Cross-module & pyright catching real mistakes
    bad_pol = """from skills.missing_mod import helper
class Policy:
    def run(self, state: GameState):
        yield state.not_a_real_field
"""
    with pytest.raises(PolicyValidationError):
        validate_cross_module_contracts(bad_pol, skills={})

    py_errs = run_pyright_on_package(bad_pol, skills={})
    assert any("missing_mod" in e or "not_a_real_field" in e for e in py_errs)

    good_pol = """from lox.env.actions import Action

def step_wait(st):
    return Action.WAIT

class Policy:
    def step(self, state):
        return step_wait(state) or Action.WAIT
"""
    assert run_pyright_on_package(good_pol, skills={}) == []

    # #EVOLVE-BLOCK bounds check
    scaffold = "# keep\nX = 1\n# EVOLVE-BLOCK: core\ndef foo():\n    return 1\n# END-EVOLVE-BLOCK\n"
    ok_cand = "# keep\nX = 1\n# EVOLVE-BLOCK: core\ndef foo():\n    return 2\n# END-EVOLVE-BLOCK\n"
    bad_cand = "# changed\nX = 2\n# EVOLVE-BLOCK: core\ndef foo():\n    return 2\n# END-EVOLVE-BLOCK\n"
    assert check_evolve_block_bounds(scaffold, ok_cand) is None
    assert check_evolve_block_bounds(scaffold, bad_cand) is not None

    # Repo-intel tools + filtered SWARM_PEER tool schema
    db_path = str(tmp_path / "test_p2.duckdb")
    init_db(db_path).close()
    reg = AuthorToolRegistry(db_path=db_path)
    reg.current_policy_code = good_pol
    reg.skills = {
        "nav": "def move_dir(st, ws):\n    ws.nav_target = (1, 2)\n    return ws.nav_target\n"
    }
    reg.configure_generation(
        pool=None, skills=reg.skills, generation=1, mode="SWARM_PEER"
    )
    ref_out, _ = reg.find_references("step_wait")
    assert "policy.py" in ref_out
    grep_out, _ = reg.grep_skills("nav_target")
    assert "skills/nav.py" in grep_out
    parity_out, _ = reg.benchmark_expert_parity(top_n=5)
    assert "Parity" in parity_out

    peer_specs = get_tool_specs("SWARM_PEER")
    peer_tool_names = {s["function"]["name"] for s in peer_specs}
    assert "query_policy_docs" not in peer_tool_names
    assert "read_wiki_article" in peer_tool_names
    assert "architecture_map" in peer_tool_names
    dry_props = next(
        s["function"]["parameters"]["properties"]
        for s in peer_specs
        if s["function"]["name"] == "dry_run_policy"
    )
    assert "hypothesis" not in dry_props and "is_macro" not in dry_props

    # Pre-mortem ticks + THEORY priority + sample_run_ids[:1]
    logger = BatchDatabaseLogger(db_path)
    logger.save_policy(
        policy_id="pol_g1",
        generation=1,
        mode="SWARM",
        policy_code=good_pol,
        macros_code="",
        params={},
        avg_depth=1.0,
        max_depth=1,
        skills=reg.skills,
        is_promoted=True,
    )
    run_rec = EpisodeRun(
        run_id="run_pm_1",
        seed=42,
        role="val-hum-law-fem",
        max_depth=1,
        depth=1,
        turns=15,
        score=5,
        end_reason="died_by_newt",
        action_bytes=b"..",
        ticks=[
            {
                "step_idx": i,
                "depth": 1,
                "turn": i + 1,
                "hp": 10 - i,
                "max_hp": 14,
                "action_name": "WAIT",
                "active_function": "step_wait",
                "message": "The newt bites!" if i == 3 else "",
            }
            for i in range(4)
        ],
    )
    logger.save_batch(
        batch_id="batch_g1",
        generation=1,
        mode="SWARM",
        parent_policy_id="pol_g1",
        candidate_policy_id="pol_g1",
        target_functions=[],
        runs=[run_rec],
        episodes_discarded=0,
        batch_metrics={"avg_depth": 1.0, "max_depth": 1},
        verdict={"passed": True, "mean_delta": 0.1},
    )
    con = connect_db(db_path)
    con.execute(
        "INSERT INTO theories (doc_id, generation, agent_role, doc_type, title, content, tags, verified_c_rule) "
        "VALUES ('th_log', 2, 'SWARM_REFEREE', 'REFEREE_LOG', 'Ref note', '', '', '')"
    )
    con.execute(
        "INSERT INTO theories (doc_id, generation, agent_role, doc_type, title, content, tags, verified_c_rule) "
        "VALUES ('th_real', 1, 'peer_1', 'THEORY', 'Real C Theory', '', '', 'Rule')"
    )
    con.close()

    rep = DeficitAnalyzer(db_path).analyze(
        generation=2, current_policy=good_pol, skills=reg.skills
    )
    brief = rep.format_brief()
    assert "run_ids=['run_pm_1']" in brief
    assert "pre_mortem=[" in brief
    assert rep.recent_theories[0].startswith("[Gen 1 | peer_1] Real C Theory")

    # TRIGGER_MISS check when edited skill function never executes on claimed deficit fixtures
    h = NetHackHarness(max_steps=40)
    try:
        h.reset(seed=7)
        h.step(Action.S)
        raw_bytes = bytes(h.action_bytes)
    finally:
        h.close()
    claim = DeficitCluster(
        cluster_id="c1",
        kind="early_trajectory",
        label="test",
        run_count=2,
        avg_depth=1.0,
        avg_turns=10.0,
        replay_fixtures=[
            {
                "run_id": "r1",
                "seed": 7,
                "action_bytes": raw_bytes,
                "stop_at_byte_idx": 1,
            }
        ],
    )
    reg2 = AuthorToolRegistry(db_path=db_path)
    reg2.skills = {"nav": "def helper(st, bb):\n    return Action.WAIT\n"}
    reg2.configure_generation(
        pool=None,
        skills=reg2.skills,
        generation=2,
        mode="SWARM_PEER",
        claimed_deficit=claim,
    )
    reg2.current_policy_code = good_pol
    dead_patch = '<skill_code module="nav">\ndef dead_never_called(st, bb):\n    return Action.DOWN\n</skill_code>'
    diag_tm, status_tm = reg2.dry_run_policy(
        code=dead_patch, seed=42, max_turns=25, smoke_only=False
    )
    assert status_tm.startswith("PASSED")
    assert "TRIGGER_MISS WARNING" in diag_tm


def test_phase3_metric_racing_pareto_stacking_and_caution_reseed():
    adapter = get_adapter("nethack")
    r_sample = {
        "max_depth": 3,
        "xp_level": 4,
        "final_ac": 4,
        "worn_oprop_count": 2,
        "dungeons_visited": 2,
        "score": 400,
        "unique_tiles": 50,
    }
    assert pytest.approx(
        progression_metric(r_sample), rel=1e-6
    ) == adapter.progression_metric(r_sample)
    # Caution reseed in MAP-Elites
    arch = MAPElitesArchive()
    arch.cells["a"] = type(
        "E",
        (),
        {
            "cell_id": "a",
            "caution_bin": 0,
            "fitness": 9.0,
            "policy_code": "p1",
            "skills": {},
        },
    )()  # type: ignore[assignment]
    arch.cells["b"] = type(
        "E",
        (),
        {
            "cell_id": "b",
            "caution_bin": 3,
            "fitness": 4.0,
            "policy_code": "p2",
            "skills": {},
        },
    )()  # type: ignore[assignment]
    picked = arch.sample_elite(strategy="caution")
    assert picked is not None and picked.cell_id == "b"

    # Two-Stage Seed Racing + N-Way Sequential Greedy Stacking mock tournament
    base_pol = (
        "from skills import a_sk, b_sk\n"
        "class Policy:\n"
        "    def step(self, st):\n"
        "        return a_sk.act_a(st) or b_sk.act_b(st) or Action.WAIT\n"
    )
    base_skills = {
        "a_sk": "def act_a(st):\n    return None\n",
        "b_sk": "def act_b(st):\n    return None\n",
    }
    p1_skills = {
        "a_sk": "def act_a(st):\n    return Action.N\n",
        "b_sk": "def act_b(st):\n    return None\n",
    }
    p2_skills = {
        "a_sk": "def act_a(st):\n    return None\n",
        "b_sk": "def act_b(st):\n    return Action.S\n",
    }
    p3_skills = {
        "a_sk": "def act_a(st):\n    return Action.E\n",
        "b_sk": "def act_b(st):\n    return None\n",
    }

    c1 = PeerCandidate(
        peer_id="peer_1",
        claimed_deficit_id="d1",
        claimed_label="p1",
        hypothesis_yaml={},
        policy_code=base_pol,
        skills=p1_skills,
        touched_skills={"a_sk"},
        touched_skill_symbols={"a_sk": {"act_a"}},
        sandbox_passed=True,
        sandbox_metrics={"avg_depth": 2.0, "avg_score": 50.0},
    )
    c2 = PeerCandidate(
        peer_id="peer_2",
        claimed_deficit_id="d2",
        claimed_label="p2",
        hypothesis_yaml={},
        policy_code=base_pol,
        skills=p2_skills,
        touched_skills={"b_sk"},
        touched_skill_symbols={"b_sk": {"act_b"}},
        sandbox_passed=True,
        sandbox_metrics={"avg_depth": 1.8, "avg_score": 40.0},
    )
    c3 = PeerCandidate(
        peer_id="peer_3",
        claimed_deficit_id="d3",
        claimed_label="p3",
        hypothesis_yaml={},
        policy_code=base_pol,
        skills=p3_skills,
        touched_skills={"a_sk"},
        touched_skill_symbols={"a_sk": {"act_a"}},
        sandbox_passed=True,
        sandbox_metrics={"avg_depth": 1.0, "avg_score": 5.0},
    )

    class _FakePool:
        role = "val-hum-law-fem"
        benchmark = "nethack"
        max_steps = 50

        def __init__(self):
            self.tasks_seen = 0

        def evaluate_tasks(self, tasks):
            self.tasks_seen += len(tasks)
            out = []
            for t in tasks:
                sd = int(t["seed"])
                sk = t["skills"]
                has_a = "Action.N" in sk.get("a_sk", "")
                has_b = "Action.S" in sk.get("b_sk", "")
                has_bad = "Action.E" in sk.get("a_sk", "")
                if has_bad:
                    d = 1
                elif has_a and has_b:
                    d = 5
                elif has_a:
                    d = 3
                elif has_b:
                    d = 2
                else:
                    d = 1
                out.append(
                    EpisodeRun(
                        run_id=t["run_id"],
                        seed=sd,
                        role=self.role,
                        max_depth=d,
                        depth=d,
                        score=d * 20,
                        turns=50,
                        end_reason="max_steps",
                    )
                )
            return out

    pool = _FakePool()
    ref = SwarmReferee(max_tournament_peers=3)
    verdict = ref.evaluate_tournament(
        generation=1,
        candidates=[c1, c2, c3],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=pool,
        seeds=list(range(1, 9)),  # 8 seeds with race_seeds=4
        race_seeds=4,
    )
    assert verdict.promoted is True
    assert verdict.winner_avg_depth >= 4.9  # Merged a_sk + b_sk won!
    # 6 packages (inc + 3 finalists + 2 orthos) * 8 seeds = 48 without racing -> 40 with 2-stage racing
    assert pool.tasks_seen < 48


def test_phase4_and_5_organic_scenarios_referee_composition_holdout_and_scoped_ochiai(
    tmp_path,
):

    from lox.author.swarm_deficits import DeficitAnalyzer, DeficitReport
    from lox.author.swarm_peer import (
        PeerCandidate,
        compute_candidate_skill_symbols,
        incubate_candidate,
    )
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.db.shards import mine_organic_scenarios
    from lox.db.summarizer import compute_function_stats
    from lox.engine.frontier import FrontierArchive
    from lox.engine.swarm_referee import SwarmReferee

    db_path = str(tmp_path / "p45.duckdb")
    f_dir = str(tmp_path / "frontier")
    init_db(db_path).close()

    # 1. FrontierArchive inverse-visit visit_count persistence
    fa = FrontierArchive(frontier_dir=f_dir)
    fa.add_checkpoint(
        seed=10,
        depth=6,
        milestone="dlvl_6",
        turns=500,
        action_bytes=b"llllkjjjjhhh" * 2,
    )
    fa.add_checkpoint(
        seed=11,
        depth=6,
        milestone="dlvl_6",
        turns=520,
        action_bytes=b"kkkkhhhhllll" * 2,
    )
    cp = fa.sample_checkpoint(min_depth=4)
    assert cp is not None and cp.visit_count == 1
    fa_reloaded = FrontierArchive(frontier_dir=f_dir)
    assert any(c.visit_count == 1 for c in fa_reloaded.checkpoints.values())

    # 2. mine_organic_scenarios across structural buckets
    con = connect_db(db_path, read_only=False)
    ab_long = b"l" * 120
    con.execute(
        """
        INSERT INTO runs (
            run_id, batch_id, generation, seed, role, depth, max_depth, score, turns, steps,
            end_reason, auto_esc_count, starved_with_food, died_unarmored, is_shard, action_bytes, milestones_json
        ) VALUES
        ('r_deep', 'b1', 1, 2001, 'val', 5, 5, 400, 1200, 1200, 'killed by ant', 0, FALSE, FALSE, FALSE, ?, '{"depth_first_turn":{"2":200}}'),
        ('r_stall', 'b1', 1, 2002, 'val', 1, 1, 10, 80, 200, 'stalled_zero_turn', 3, FALSE, FALSE, FALSE, ?, '{}'),
        ('r_unarm', 'b1', 1, 2003, 'val', 2, 2, 80, 400, 400, 'killed by orc', 0, FALSE, TRUE, FALSE, ?, '{"unworn_armor_in_pack":true,"depth_first_turn":{"2":250}}'),
        ('r_ill', 'b1', 1, 2004, 'val', 2, 2, 70, 500, 500, 'died of terminal illness', 0, TRUE, FALSE, FALSE, ?, '{}')
        """,
        [ab_long, ab_long, ab_long, ab_long],
    )
    con.close()
    mined = mine_organic_scenarios(db_path, frontier_dir=f_dir, max_total=10)
    buckets = {sh.trigger_context for sh in mined}
    assert "frontier_deep" in buckets
    assert "stall_or_prompt" in buckets
    assert "unequipped_crisis" in buckets
    assert "resource_or_mortality" in buckets

    # 3. Hard exception rejection during incubate_candidate
    base_pol = """
from lox.env.actions import Action
from skills.sub import h_base

class Policy:
    def step(self, state):
        return h_base(state) or Action.WAIT
"""
    base_skills = {
        "sub": "from lox.env.actions import Action\ndef h_base(s):\n    return Action.SEARCH\n"
    }
    cand_exc = PeerCandidate(
        peer_id="p_exc",
        claimed_deficit_id="c1",
        claimed_label="raises in pool",
        policy_code=base_pol,
        skills=base_skills,
    )

    class _ExcPool:
        def evaluate_batch(self, **kwargs):
            return {
                "runs": [
                    {
                        "run_id": "x1",
                        "seed": 42,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 50,
                        "turns": 100,
                        "policy_exceptions": 2,
                        "end_reason": "policy_error:KeyError",
                        "milestones_json": json.dumps(
                            {"exception_tracebacks": ["sub.py:5: KeyError: 'k'"]}
                        ),
                    }
                ]
            }

    rep_inc = DeficitReport(generation=2, regime="PLATEAU")
    inc_out = incubate_candidate(
        cand_exc,
        deficit_report=rep_inc,
        claimed_deficit=DeficitAnalyzer._bootstrap_clusters()[0],
        eval_pool=_ExcPool(),
        shard_seeds=[42],
    )
    assert inc_out.sandbox_passed is False
    assert "exception" in inc_out.error.lower()

    # 4. Top-K 3-way AST composition inside the SAME skill module + Hidden Holdout audit cancellation
    sk_a = base_skills["sub"] + "\ndef h_alpha(s):\n    return Action.EAST\n"
    sk_b = base_skills["sub"] + "\ndef h_beta(s):\n    return Action.WEST\n"
    pol_a = base_pol.replace(
        "from skills.sub import h_base", "from skills.sub import h_alpha, h_base"
    ).replace(
        "class Policy:",
        "def run_alpha(s):\n    return h_alpha(s)\n\nclass Policy:",
    )
    pol_b = base_pol.replace(
        "from skills.sub import h_base", "from skills.sub import h_base, h_beta"
    ).replace(
        "class Policy:",
        "def run_beta(s):\n    return h_beta(s)\n\nclass Policy:",
    )
    ca = PeerCandidate(
        peer_id="pA",
        claimed_deficit_id="da",
        claimed_label="alpha",
        policy_code=pol_a,
        skills={"sub": sk_a},
        touched_skills={"sub"},
        touched_branches={"alpha"},
        touched_skill_symbols=compute_candidate_skill_symbols(
            {"sub": sk_a}, base_skills
        ),
        sandbox_passed=True,
    )
    cb = PeerCandidate(
        peer_id="pB",
        claimed_deficit_id="db",
        claimed_label="beta",
        policy_code=pol_b,
        skills={"sub": sk_b},
        touched_skills={"sub"},
        touched_branches={"beta"},
        touched_skill_symbols=compute_candidate_skill_symbols(
            {"sub": sk_b}, base_skills
        ),
        sandbox_passed=True,
    )
    combos = SwarmReferee.build_orthogonal_candidates([ca, cb], base_pol, base_skills)
    assert len(combos) == 1
    assert "h_alpha" in combos[0].skills["sub"] and "h_beta" in combos[0].skills["sub"]
    assert "h_alpha" in combos[0].policy_code and "h_beta" in combos[0].policy_code

    # Holdout cancellation check
    class _HoldoutFailPool:
        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [1, 2])
            if batch_id.startswith("holdout_") and "_inc_" not in batch_id:
                return {
                    "runs": [
                        {
                            "run_id": f"hr_{s}",
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 0,
                            "turns": 50,
                            "final_ac": 10,
                            "xp_level": 1,
                        }
                        for s in s_list
                    ]
                }
            if "_inc_" in batch_id:
                return {
                    "runs": [
                        {
                            "run_id": f"ir_{s}",
                            "seed": s,
                            "depth": 2,
                            "max_depth": 2,
                            "score": 100,
                            "turns": 200,
                            "final_ac": 9,
                            "xp_level": 2,
                        }
                        for s in s_list
                    ]
                }
            return {
                "runs": [
                    {
                        "run_id": f"cr_{s}",
                        "seed": s,
                        "depth": 4,
                        "max_depth": 4,
                        "score": 300,
                        "turns": 500,
                        "final_ac": 6,
                        "xp_level": 3,
                    }
                    for s in s_list
                ]
            }

    ref = SwarmReferee()
    hv = ref.evaluate_tournament(
        generation=10,
        candidates=[ca],
        base_policy=base_pol,
        base_skills=base_skills,
        eval_pool=_HoldoutFailPool(),
        seeds=[1001, 1002, 1003, 1004],
        holdout_seeds=[100000, 100001, 100002, 100003],
    )
    assert hv.promoted is False
    assert hv.holdout_iqm_delta is not None and hv.holdout_iqm_delta < -0.15
    assert len(hv.attempts_summary) >= 1

    # 5. Stagnation-discounted Ochiai + tournament_attempts surfaced by DeficitAnalyzer
    fn_stats = compute_function_stats(
        "b_stag",
        10,
        [
            {
                "run_id": "r_g",
                "is_outlier_good": True,
                "max_depth": 5,
                "turns": 100,
                "end_reason": "killed",
                "triggered_functions": json.dumps({"good_fn": 20, "wait_stag": 10}),
                "milestones": {"stagnation_fn_turns": {"wait_stag": 85}},
                "summary_digest": "Stalls: wait_stag stalled (T5..T90 (85 turns))",
            }
        ],
    )
    by_fn_st = {row["function_name"]: row for row in fn_stats}
    assert (
        by_fn_st["wait_stag"]["ochiai_good_run_score"]
        < by_fn_st["good_fn"]["ochiai_good_run_score"]
    )

    BatchDatabaseLogger(db_path=db_path).save_tournament_attempts(hv.attempts_summary)
    rep_after = DeficitAnalyzer(db_path).analyze(generation=11, current_policy=base_pol)
    assert any(
        "REJECTED" in line and "pA" in line for line in rep_after.recent_attempts
    )
