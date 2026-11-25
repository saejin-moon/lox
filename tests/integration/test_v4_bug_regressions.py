"""
Comprehensive v4 regression test suite covering security, boundary, truthiness,
author/peer/lifecycle, env, lang, Rust, and DB fixes from BUGS.md:
- Sandbox & SQL Security: C-1, C-2, H-9
- Boundary & Truthiness: C-3, C-4, H-14, M-10, M-11, M-16, M-17
- Generator & Compiler: C-7, C-8, H-29, H-30, H-31, H-32, H-33, H-34
- Env & Rust: H-19, H-20, H-21, H-22, H-23, H-24, H-27, M-14, M-18, M-19
- Author, Peer, & Lifecycle: C-6, H-2..H-13, M-2, M-3, M-6, M-8, M-9, M-12, M-13, M-35, M-38
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lox.author.agent import AuthorAgent
from lox.author.llm_client import extract_usage_tokens, resolve_api_key
from lox.author.parsing import safe_as_int
from lox.author.swarm_peer import PeerCandidate, compute_candidate_footprint
from lox.author.tools import AuthorToolRegistry, validate_readonly_sql
from lox.core.contracts import BatchEvaluation, EpisodeRun
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import compute_llm_cost, init_db
from lox.engine.hypothesis import (
    compute_confidence_sequence,
    evaluate_tournament_pair,
)
from lox.engine.lifecycle import update_near_misses, update_stepping_stone_lifecycle
from lox.engine.map_elites import MAPElitesArchive
from lox.engine.referee_gates import (
    TournamentVerdict,
    _binomial_sign_p,
    _evaluate_confirmation,
    _passes_safety_invariants,
)
from lox.engine.swarm_referee import SwarmReferee
from lox.env.actions import ACTION_BY_NAME, Action, key, normalize_action_to_bytes, seq
from lox.env.benchmarks.nethack_scenarios import (
    PROMPT_TORTURE_SUITES,
    NetHackScenarioHarness,
)
from lox.env.harness import NetHackHarness
from lox.env.state import GameState
from lox.env.types import InventoryItem
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import PolicyValidationError, validate_python_code

# =============================================================================
# 1. Sandbox & SQL Security (C-1, C-2, H-9)
# =============================================================================


@pytest.mark.parametrize(
    "malicious_code",
    [
        "from lox.lang.validator import os\nclass Policy:\n    def step(self, s): return Action.WAIT\n",
        "from dataclasses import sys\nclass Policy:\n    def step(self, s): return Action.WAIT\n",
        "import random\nclass Policy:\n    def step(self, s):\n        _ = random._os\n        return Action.WAIT\n",
        "class Policy:\n    def step(self, s):\n        _ = dataclasses.sys.modules['os'].name\n        return Action.WAIT\n",
        "class Policy:\n    def step(self, s):\n        r = random\n        return r._os.name\n",
        "import re\nclass Policy:\n    def step(self, s):\n        x = re\n        return x._parser.__name__\n",
        "class Policy:\n    def step(self, s):\n        return enum.sys.modules\n",
    ],
)
def test_c1_sandbox_blocks_module_escapes(malicious_code: str):
    with pytest.raises(PolicyValidationError):
        validate_python_code(malicious_code)


def test_c1_runtime_getattr_and_stdlib_proxy_isolation():
    code = """
class Policy:
    def step(self, state):
        g = getattr(key, "__" + "globals__")
        return Action.WAIT
"""
    compiled = compile_policy_package(code)
    runner = compiled.create_runner()
    harness = NetHackHarness(max_steps=5)
    try:
        st = harness.reset(seed=42)
        with pytest.raises(AttributeError):
            runner.next_action(st)
    finally:
        harness.close()

    # Even if an expression evaded AST name tracking (e.g. dynamic getattr or dict lookup),
    # runtime stdlib module proxies strip ModuleType/private attrs completely while keeping safe APIs.
    proxy_escape = """
import random as rnd
import dataclasses as dc
class Policy:
    def step(self, state):
        mods = [rnd, dc, re, collections, np]
        _ = getattr(mods[1], "s" + "ys")
        return Action.WAIT
"""
    compiled_pe = compile_policy_package(proxy_escape)
    r_pe = compiled_pe.create_runner()
    h_pe = NetHackHarness(max_steps=5)
    try:
        st_pe = h_pe.reset(seed=42)
        with pytest.raises(AttributeError):
            r_pe.next_action(st_pe)
    finally:
        h_pe.close()

    # Safe stdlib features (dataclasses, numpy, np.linalg, collections.abc, collections.Counter, submodule imports) work cleanly.
    safe_stdlib = """
import collections.abc
import collections.abc as cabc
from collections.abc import Callable
import numpy.linalg
from numpy.linalg import norm
@dataclasses.dataclass
class Tracker:
    cnt: int = 0
class Policy:
    def step(self, state):
        _ = np.linalg.norm(np.array([3.0, 4.0]))
        _ = numpy.linalg.norm(np.array([3.0, 4.0]))
        _ = norm(np.array([3.0, 4.0]))
        assert callable(collections.Counter)
        assert Callable is not None and cabc.Callable is Callable
        assert isinstance(Tracker(), Tracker)
        return Action.WAIT
"""
    compiled_ok = compile_policy_package(safe_stdlib)
    r_ok = compiled_ok.create_runner()
    h_ok = NetHackHarness(max_steps=5)
    try:
        st_ok = h_ok.reset(seed=42)
        act, _ = r_ok.next_action(st_ok)
        assert act == Action.WAIT
    finally:
        h_ok.close()

    for bad_submod_code in (
        "import collections.sys\nclass Policy:\n    def step(self, state):\n        return Action.WAIT\n",
        "from numpy.testing import assert_allclose\nclass Policy:\n    def step(self, state):\n        return Action.WAIT\n",
    ):
        with pytest.raises(PolicyValidationError):
            validate_python_code(bad_submod_code)

    tamper_code = """
class Policy:
    def step(self, state):
        _lox_tracer.max_loop_ticks = 10**12
        return Action.WAIT
"""
    with pytest.raises(PolicyValidationError):
        validate_python_code(tamper_code)


@pytest.mark.parametrize(
    "bad_sql",
    [
        "SELECT 1 -- '\n; DROP TABLE runs -- '",
        "SELECT /* comment ; */ 1; DELETE FROM runs",
        "SELECT * FROM read_text('.env')",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "COPY (SELECT 1) TO '/tmp/pwn.csv'",
        "ATTACH '/tmp/x.db' AS x",
        "PRAGMA database_list",
        "SET enable_external_access = true",
        "CALL pragma_version()",
    ],
)
def test_c2_sql_guard_blocks_injections_and_file_reads(bad_sql: str, tmp_path):
    err, _, _ = validate_readonly_sql(bad_sql)
    assert err is not None, f"SQL guard failed to block: {bad_sql}"

    db_path = str(tmp_path / "sec.duckdb")
    con = init_db(db_path)
    con.close()
    reg = AuthorToolRegistry(db_path=db_path)
    out, status = reg.query_db(bad_sql)
    assert status == "error"
    assert "Error" in out


def test_h9_read_skill_path_traversal_and_snapshot_isolation(tmp_path):
    reg = AuthorToolRegistry(db_path=str(tmp_path / "t.duckdb"), workspace_dir=tmp_path)
    out, status = reg.read_skill("../../../lox/author/agent")
    assert status == "error"
    assert "Invalid skill name" in out

    # Snapshot isolation: when self.skills is populated, never fall back to global disk skills
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "secret_disk.py").write_text("X = 1\n", encoding="utf-8")
    reg.skills = {"world": "A = 1\n"}
    out2, status2 = reg.read_skill("secret_disk")
    assert status2 == "not_found"


# =============================================================================
# 2. Boundary & Truthiness Regressions (C-3, C-4, H-14, M-10, M-11, M-16, M-17)
# =============================================================================


def test_c3_exact_zero_p_values_and_large_n_binomial():
    # hw_mcnemar_p == 0.0 MUST reject (never coerced to 1.0 by `or 1.0`)
    ok, reason = _passes_safety_invariants(
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "run_count": 32,
            "cvar25_prog": 2.0,
        },
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "run_count": 32,
            "cvar25_prog": 2.0,
        },
        has_base=True,
        pair_stats={"hw_worsened": 20, "hw_improved": 0, "hw_mcnemar_p": 0.0},
        stage="confirm",
    )
    assert not ok
    assert reason == "horizon_waste_regression"

    # sign_perm_p == 0.0 MUST promote (never coerced to 1.0 by `or 1.0`)
    beats, safe, rej, _ = _evaluate_confirmation(
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "run_count": 32,
            "cvar25_prog": 3.0,
            "ast_nodes": 100,
        },
        {
            "wins": 20,
            "losses": 0,
            "ties": 12,
            "paired_count": 32,
            "mean_delta": 0.5,
            "winsor_delta": 0.4,
            "sign_perm_p": 0.0,
            "hw_worsened": 0,
            "hw_improved": 0,
            "hw_mcnemar_p": 1.0,
        },
        {
            "policy_exceptions": 0,
            "auto_esc_total": 0,
            "run_count": 32,
            "cvar25_prog": 2.5,
            "ast_nodes": 100,
        },
        has_base=True,
        alpha_cutoff=0.05,
    )
    assert beats and safe and rej == "promoted"

    # Large n >= 1025 binomial must NOT raise OverflowError
    p_big = _binomial_sign_p(800, 700)
    assert 0.0 <= p_big < 0.01


def test_c4_bootstrap_failure_never_promotes():
    ref = SwarmReferee()

    class _CrashingPool:
        benchmark = "nethack"

        def evaluate_batch(self, **kwargs):
            seeds = kwargs.get("priority_seeds", [1, 2])
            return BatchEvaluation(
                {
                    "batch_id": kwargs.get("batch_id", "b"),
                    "runs": [
                        {
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 0,
                            "turns": 1,
                            "policy_exceptions": 2,
                            "end_reason": "crash",
                        }
                        for s in seeds
                    ],
                }
            )

    cand = PeerCandidate(
        peer_id="reactive_1",
        claimed_deficit_id="boot",
        claimed_label="boot",
        policy_code="class Policy:\n    def step(self, s): return Action.WAIT\n",
        sandbox_passed=True,
    )
    verdict = ref.evaluate_tournament(
        generation=1,
        candidates=[cand],
        base_policy="",
        base_skills={},
        eval_pool=_CrashingPool(),
        seeds=[1, 2, 3, 4],
    )
    assert verdict.promoted is False


def test_h14_m10_m11_m16_m17_truthiness_and_boundaries():
    # H-14: avg_ac == 0.0 must be preserved (better armor than AC 10!)
    f_ac0 = MAPElitesArchive.compute_fitness(
        {"avg_depth": 2.0, "max_depth": 3, "avg_ac": 0.0}
    )
    f_ac10 = MAPElitesArchive.compute_fitness(
        {"avg_depth": 2.0, "max_depth": 3, "avg_ac": 10.0}
    )
    assert f_ac0 > f_ac10 + 1000.0

    # M-10: seed == 0 shards pair properly
    pair = evaluate_tournament_pair(
        [
            {
                "is_shard": True,
                "shard_id": "",
                "seed": 0,
                "max_depth": 3,
                "depth": 3,
                "score": 500,
            }
        ],
        [
            {
                "is_shard": True,
                "shard_id": "",
                "seed": 0,
                "max_depth": 1,
                "depth": 1,
                "score": 100,
            }
        ],
    )
    assert pair["paired_count"] == 1
    assert pair["wins"] == 1

    # M-11: n=3 identical deltas do NOT collapse to zero-width CI
    mean_d, lo, hi = compute_confidence_sequence(np.array([0.01, 0.01, 0.01]))
    assert hi > mean_d > lo

    # M-16 & M-17: explicit None coalescing + final_hp/final_ac aliases
    ep = EpisodeRun(
        {"max_depth": None, "depth": 4, "is_shard": None, "final_hp": 15, "final_ac": 0}
    )
    assert ep["max_depth"] == 4
    assert ep["is_shard"] is False
    assert ep["hp"] == 15
    assert ep["ac"] == 0


# =============================================================================
# 3. Env, Rust, & Lang Regressions (H-20..H-24, H-27, H-29, H-30, C-7, C-8, M-18, M-35)
# =============================================================================


def test_env_and_rust_fixes():
    # H-23: ACTION_BY_NAME includes aliases & numpy integers work
    assert ACTION_BY_NAME["NORTH"] == Action.N
    assert ACTION_BY_NAME["ASCEND"] == Action.UP
    assert normalize_action_to_bytes(np.int64(Action.N)) == ((int(Action.N),), "N")
    assert key(np.int32(ord("y"))).byte_val == ord("y")
    assert seq(np.int64(Action.N), np.int64(Action.S)).bytes_seq == (
        int(Action.N),
        int(Action.S),
    )

    # H-22: class-qualified lookup & enchantment stripping without spellbook shadowing
    p_heal = GameState.object_info_by_name("potion of healing")
    assert p_heal is not None and p_heal.oclass_name == "potion"
    sword = GameState.object_info_by_name("+2 long sword")
    assert sword is not None and sword.oclass_name == "weapon"

    # M-18: two-handed & two-weapon wielded regex
    it_2h = InventoryItem(
        letter="a",
        name="a +0 two-handed sword (weapon in hands)",
        glyph=0,
        oclass=2,
        char=")",
        quantity=1,
    )
    it_off = InventoryItem(
        letter="b",
        name="a +0 silver saber (weapon in left hand)",
        glyph=0,
        oclass=2,
        char=")",
        quantity=1,
    )
    assert it_2h.is_wielded is True
    assert it_off.is_wielded is True

    # H-27: Rust shape mismatch raises ValueError instead of panicking
    import lox_core

    mask = np.ones((21, 79), dtype=np.bool_)
    bad_nd = np.zeros((5, 5), dtype=np.bool_)
    with pytest.raises(ValueError):
        lox_core.fast_bfs_distance_grid(mask, 1, 1, True, bad_nd)
    with pytest.raises(ValueError):
        lox_core.fast_astar_next_step(mask, 1, 1, 2, 2, True, bad_nd)

    # H-24: prompt_torture scenarios trigger their real NetHack prompts
    for sc_name, expected_attr in (
        ("prompt_torture_yn_confirm", "in_yn_prompt"),
        ("prompt_torture_call_name", "in_text_prompt"),
        ("prompt_torture_direction_count", "in_getdir_prompt"),
        ("prompt_torture_multipage_menu", "in_menu"),
    ):
        assert sc_name in PROMPT_TORTURE_SUITES
        h = NetHackScenarioHarness(scenario=sc_name, max_steps=10)
        try:
            st = h.reset(seed=42)
            assert getattr(st, expected_attr) is True, (
                f"{sc_name} did not trigger {expected_attr}"
            )
        finally:
            h.close()


def test_c7_c8_h29_h30_lang_fixes():
    # H-29 & H-30: `from __future__ import annotations` + AssertionError/LookupError allowed
    code = """from __future__ import annotations
class Policy:
    def _helper(self, state: object):
        yield Action.NORTH
        return Action.SOUTH

    def step(self, state: object):
        try:
            assert state is not None
        except AssertionError:
            return Action.WAIT
        return self._helper(state)
"""
    compiled = compile_policy_package(code)
    runner = compiled.create_runner()
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        a1, _ = runner.next_action(st)
        assert a1 == Action.NORTH
        # C-8: Resuming subgen honors StopIteration.value (`Action.SOUTH`) and preserves `_mode == "step"`
        a2, _ = runner.next_action(st)
        assert a2 == Action.SOUTH
        assert runner._mode == "step"
    finally:
        h.close()


# =============================================================================
# 4. Author, Peer, Cost, & Lifecycle Regressions (C-6, H-4, H-5, H-11, M-3, M-4, M-8, M-9, M-12, M-35)
# =============================================================================


def test_author_llm_parsing_and_cost_fixes(monkeypatch, tmp_path):
    # H-4: safe usage extraction on None / null nested dicts
    assert extract_usage_tokens(None) == (0, 0, 0)
    assert extract_usage_tokens(
        {
            "prompt_tokens": 10,
            "completion_tokens": None,
            "completion_tokens_details": None,
        }
    ) == (10, 0, 0)

    # H-5: provider-first API key resolution
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-key")
    assert resolve_api_key("openai") == "oa-key"
    assert resolve_api_key("gemini") == "gm-key"
    assert resolve_api_key("openrouter") == "or-key"

    # H-11: safe_as_int handles None / malformed strings
    assert safe_as_int("not-an-int", 7) == 7
    assert safe_as_int(None, 3) == 3

    # M-35: compute_llm_cost longest-key matching & empty model fallback
    assert compute_llm_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
    assert compute_llm_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert compute_llm_cost("", 1_000_000, 0) == pytest.approx(0.15)

    # M-3: run_tool_session with max_tool_turns=0 sends 1 clean request with empty tool_specs & no budget nudge
    db_path = str(tmp_path / "ts.duckdb")
    con = init_db(db_path)
    con.close()
    agent = AuthorAgent(
        provider="openrouter", model="test", api_key="k", db_path=db_path
    )
    seen_calls: list[dict[str, Any]] = []

    def _fake_call(messages, allow_tools=True, tool_specs=None):
        seen_calls.append(
            {
                "messages": list(messages),
                "allow_tools": allow_tools,
                "tool_specs": tool_specs,
            }
        )
        return {"content": '[{"peer_id": "reactive_1"}]', "_usage": None}

    agent._call_llm_api = _fake_call  # type: ignore[method-assign]
    out = agent.run_tool_session(
        "sys", "usr", max_tool_turns=0, require_json=True, expected_json_type="array"
    )
    assert "reactive_1" in out
    assert len(seen_calls) == 1
    assert seen_calls[0]["allow_tools"] is False
    assert seen_calls[0]["tool_specs"] == []
    assert len(seen_calls[0]["messages"]) == 2


def test_c6_non_mock_swarm_peer_auto_dry_run_passes_cleanly(tmp_path):
    db_path = str(tmp_path / "c6.duckdb")
    con = init_db(db_path)
    con.close()
    agent = AuthorAgent(
        provider="openrouter", model="test", api_key="k", db_path=db_path
    )
    agent.tools.enforce_peer_research_gate = False
    calls = 0

    def _fake_llm(messages, allow_tools=True, tool_specs=None):
        nonlocal calls
        calls += 1
        return {
            "content": (
                "```yaml\nmechanism: test step\ntarget_functions: [step]\n```\n"
                "<policy_code>\nclass Policy:\n    def step(self, state):\n        return Action.SEARCH\n</policy_code>"
            ),
            "_usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

    agent._call_llm_api = _fake_llm  # type: ignore[method-assign]
    prop = agent.synthesize(
        mode="SWARM_PEER", generation=2, current_policy="", max_tool_turns=5
    )
    assert calls == 1, (
        "C-6 regression: passing peer dry-run should break on Turn 1 without false PILOT REVISION FAILED"
    )
    assert "Action.SEARCH" in prop.full_policy_code


def test_m4_m8_m9_m12_footprint_and_lifecycle(tmp_path):
    # M-4: compute_candidate_footprint detects clobbered protected functions across Option-A skills
    base_pol = "from skills.combat import fight\nclass Policy:\n    def step(self, s): return fight(s)\n"
    base_sk = {"combat": "def fight(s): return Action.WAIT\n"}
    new_pol = "class Policy:\n    def step(self, s): return Action.SEARCH\n"
    _, _, clobbers = compute_candidate_footprint(
        new_pol,
        {},
        base_pol,
        base_sk,
        protected_branches=[{"fn": "fight", "ochiai": 0.9}],
    )
    assert clobbers is True

    # M-9: near_misses cleared immediately upon promotion
    nm = [{"peer_id": "reactive_1", "generation": 1}]
    dummy_verdict = TournamentVerdict(
        promoted=True,
        winner_id="reactive_1",
        winner_policy_code=new_pol,
        winner_skills={},
        winner_batch_eval=BatchEvaluation({"runs": []}),
        incumbent_avg_depth=1.0,
        incumbent_avg_score=0.0,
        winner_avg_depth=2.0,
        winner_avg_score=100.0,
    )
    update_near_misses(
        nm,
        generation=2,
        verdict=dummy_verdict,
        peer_candidates=[],
        current_policy=base_pol,
        initial_skills=base_sk,
    )
    assert nm == []

    # M-8: stepping stone revision_count carried forward and retired at max_revisions=2
    db_path = str(tmp_path / "m8.duckdb")
    con = init_db(db_path)
    con.close()
    logger = BatchDatabaseLogger(db_path)
    stone0 = {
        "stone_id": "stone_g1_reactive_1",
        "parent_policy_id": "fp123",
        "generation": 1,
        "source_peer_id": "reactive_1",
        "source_run_kind": "select",
        "source_batch_id": "b1",
        "revision_count": 1,  # already revised once!
        "status": "active",
        "policy_code": new_pol,
        "skills": {},
        "mechanism": "m",
        "target_functions": ["step"],
        "wins": 5,
        "losses": 1,
        "iqm_delta": 0.3,
        "winsor_delta": 0.3,
        "wilcoxon_p": 0.08,
        "sign_p": 0.08,
        "reject_reason": "below_sign_perm_alpha",
        "seeds_gained": [1, 2],
        "seeds_lost": [3],
    }
    logger.save_stepping_stone(stone0)
    rej_verdict = TournamentVerdict(
        promoted=False,
        winner_id="incumbent",
        winner_policy_code=base_pol,
        winner_skills=base_sk,
        winner_batch_eval=BatchEvaluation({"runs": []}),
        incumbent_avg_depth=1.0,
        incumbent_avg_score=0.0,
        winner_avg_depth=1.0,
        winner_avg_score=0.0,
        qualifying_stepping_stone={
            **stone0,
            "stone_id": "stone_g2_reactive_1",
            "revision_count": 0,
        },
    )
    update_stepping_stone_lifecycle(
        db_logger=logger,
        generation=2,
        cur_fp="fp123",
        active_stone=stone0,
        verdict=rej_verdict,
        max_revisions=2,
    )
    # Stone hit revision 2/2 -> MUST be retired (None active) instead of respawning at rev 0!
    assert (
        logger.load_active_stepping_stone(parent_policy_id="fp123", max_revisions=2)
        is None
    )


def test_h2_h13_m12_m14_repair_reset_empty_finalists_and_blocked_until(tmp_path):
    from lox.lang.starter import build_clean_option_a_starter
    from scripts.reset_clean import reset_clean

    # H-2: SKILL_SYNTAX_ERROR triggers LLM repair call (never dead-looping without calling LLM)
    db_path = str(tmp_path / "h2.duckdb")
    init_db(db_path).close()
    agent = AuthorAgent(
        provider="openrouter", model="test", api_key="k", db_path=db_path
    )
    agent.tools.enforce_peer_research_gate = False
    llm_calls = 0

    def _fake_repair_llm(messages, allow_tools=True, tool_specs=None):
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return {
                "content": (
                    "```yaml\nmechanism: bad skill syntax\ntarget_functions: [step]\n```\n"
                    '<skill_code module="broken">\ndef bad(:\n    pass\n</skill_code>\n'
                    "<policy_code>\nclass Policy:\n    def step(self, state): return Action.SEARCH\n</policy_code>"
                ),
                "_usage": {"prompt_tokens": 5, "completion_tokens": 10},
            }
        return {
            "content": (
                "```yaml\nmechanism: fixed skill syntax\ntarget_functions: [step]\n```\n"
                '<skill_code module="broken">\ndef ok(state):\n    return Action.SEARCH\n</skill_code>\n'
                "<policy_code>\nfrom skills.broken import ok\nclass Policy:\n    def step(self, state): return ok(state)\n</policy_code>"
            ),
            "_usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

    agent._call_llm_api = _fake_repair_llm  # type: ignore[method-assign]
    prop = agent.synthesize(
        mode="SWARM_PEER", generation=2, current_policy="", max_tool_turns=3
    )
    assert llm_calls == 2, (
        f"H-2 regression: expected 2 LLM calls (initial + repair), got {llm_calls}"
    )
    assert "from skills.broken import ok" in prop.full_policy_code

    # H-13: reset_campaign_state clears stepping_stones and proactive_branches while keeping experts
    db_rc = str(tmp_path / "rc.duckdb")
    con = init_db(db_rc)
    con.execute(
        "INSERT INTO stepping_stones (stone_id, generation, parent_policy_id, source_peer_id, policy_code, status) "
        "VALUES ('st1', 1, 'fp', 'reactive_1', 'class Policy: pass', 'active')"
    )
    con.execute(
        "INSERT INTO proactive_branches (branch_id, base_champion_fp, generation_started, last_generation, policy_code, status) "
        "VALUES ('br1', 'fp', 1, 1, 'class Policy: pass', 'active')"
    )
    con.execute(
        "INSERT INTO expert_runs (run_id, source, role, max_depth, turns) VALUES ('ex1', 'nao', 'Val', 15, 3000)"
    )
    con.close()
    reset_clean(
        db_path=db_rc,
        policy_path=str(tmp_path / "ws" / "policy.py"),
        macros_path=str(tmp_path / "ws" / "macros.py"),
        frontier_dir=str(tmp_path / "fr"),
    )
    con = init_db(db_rc)
    try:
        assert con.execute("SELECT COUNT(*) FROM stepping_stones").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM proactive_branches").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM expert_runs").fetchone()[0] == 1
    finally:
        con.close()

    # M-12: empty finalists still records failed peer candidates into verdict.attempts_summary
    base_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s): return Action.WAIT\n"
    failed_cand = PeerCandidate(
        peer_id="reactive_1",
        claimed_deficit_id="c1",
        claimed_label="crash",
        policy_code=base_pol,
        skills={},
        sandbox_passed=False,
        error="compile_failed",
    )

    class _MinPool:
        benchmark = "nethack"

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"r_{s}",
                        "seed": s,
                        "depth": 1,
                        "max_depth": 1,
                        "score": 0,
                        "turns": 10,
                    }
                    for s in (priority_seeds or [1])
                ],
            }

    verdict = SwarmReferee().evaluate_tournament(
        generation=1,
        candidates=[failed_cand],
        base_policy=base_pol,
        base_skills={},
        eval_pool=_MinPool(),
        seeds=[1, 2],
    )
    assert verdict.promoted is False
    assert any(a.get("peer_id") == "reactive_1" for a in verdict.attempts_summary)

    # M-14: starter navigation/core.step_toward respects ws.blocked_until without mutating ws.passable_mask
    _, skills = build_clean_option_a_starter()
    world_ns: dict[str, Any] = {}
    nav_ns: dict[str, Any] = {}
    exec(compile(skills["world"], "<world>", "exec"), world_ns)
    exec(compile(skills["navigation/core"], "<navigation/core>", "exec"), nav_ns)
    h = NetHackHarness(max_steps=10)
    try:
        st = h.reset(seed=42)
        ws = world_ns["WorldState"]()
        ws.observe(st)
        # Find an adjacent passable tile and block it in ws.blocked_until
        adj_passable = [
            (st.y + dy, st.x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy != 0 or dx != 0)
            and 0 <= st.y + dy < 21
            and 0 <= st.x + dx < 79
            and bool(ws.passable_mask[st.y + dy, st.x + dx])
        ]
        assert len(adj_passable) >= 2
        # Block every adjacent passable tile except one via the per-floor runtime blacklist
        fc = ws.floor(st)
        open_tile = adj_passable[-1]
        blocked_set = set(adj_passable[:-1])
        for bt in blocked_set:
            fc.blacklist(bt, int(st.turn) + 50)
        mask_before = ws.passable_mask.copy()
        # Route toward a distant passable tile: A* must never step onto a blacklisted tile
        far = next(
            (int(y), int(x))
            for y, x in zip(*np.nonzero(ws.passable_mask), strict=True)
            if max(abs(int(y) - st.y), abs(int(x) - st.x)) >= 3
        )
        act = nav_ns["step_toward"](st, ws, far)
        if act is not None:
            assert ws.pending_move_target not in blocked_set
            assert ws.pending_move_target == open_tile
        # ws.passable_mask itself is never mutated by blacklist masking
        assert np.array_equal(ws.passable_mask, mask_before)
    finally:
        h.close()


def test_audit_p1_p2_guardrail1_logger_deficits_experts_grid_theories_and_peer_merge(
    tmp_path,
):
    """Verifies late-audit findings:
    - grid_utils.bfs_distance_grid / astar_next_step re-raise ValueError on shape mismatch
    - logger.py preserves 0/0.0 for ac, avg_ac, final_ac, hunger_state, seed, wilcoxon_p, sign_p
    - summarizer.py preserves ac=0 and handles None tick fields
    - swarm_deficits.py preserves c_ac == 0.0
    - download_experts.py preserves hp=0 and ac=0
    - TheoryDocumentStore re-initializes replaced DB via is_db_initialized
    - parse_peer_output merges partial <skill_code> updates cleanly
    """
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.swarm_peer import parse_peer_output
    from lox.datasets.download_experts import parse_dumplog_to_expert_run
    from lox.db.schema import connect_db
    from lox.db.summarizer import summarize_episode_ticks
    from lox.db.theories import TheoryDocumentStore
    from lox.env.grid_utils import astar_next_step, bfs_distance_grid

    # 1. grid_utils does not swallow Rust ValueError into Numba
    mask = np.ones((21, 79), dtype=np.bool_)
    bad_nd = np.zeros((5, 5), dtype=np.bool_)
    with pytest.raises(ValueError):
        bfs_distance_grid(mask, 1, 1, True, bad_nd)
    with pytest.raises(ValueError):
        astar_next_step(mask, 1, 1, 2, 2, True, bad_nd)

    # 2. logger.py preserves exact 0 / 0.0 and tolerates None
    dbp = str(tmp_path / "g1_logger.duckdb")
    init_db(dbp).close()
    logger = BatchDatabaseLogger(dbp)
    logger.save_tournament_attempts(
        [
            {
                "attempt_id": "att_zero_ac",
                "generation": 1,
                "peer_id": "reactive_1",
                "deficit_id": "d1",
                "candidate_fp": "fp0",
                "promoted": True,
                "sandbox_passed": True,
                "avg_depth": 2.0,
                "max_depth": 2,
                "avg_score": 100.0,
                "avg_ac": 0.0,
                "avg_xp_level": 2.0,
                "policy_exceptions": 0,
            }
        ]
    )
    logger.save_stepping_stone(
        {
            "stone_id": "st_zero_p",
            "generation": 1,
            "parent_policy_id": "fp0",
            "source_peer_id": "reactive_1",
            "policy_code": "class Policy: pass",
            "wilcoxon_p": 0.0,
            "sign_p": 0.0,
        }
    )
    logger.save_runs_and_ticks(
        batch_id="b_zero_ac",
        generation=1,
        run_kind="select",
        peer_id="reactive_1",
        runs=[
            {
                "run_id": "r_zero",
                "seed": 0,
                "depth": 2,
                "max_depth": 2,
                "score": 100,
                "turns": 50,
                "steps": 50,
                "final_ac": 0,
                "xp_level": 2,
                "ticks": [
                    {
                        "turn": 1,
                        "step_idx": 1,
                        "depth": 2,
                        "hp": 10,
                        "max_hp": 10,
                        "ac": 0,
                        "hunger_state": 0,
                        "y": 5,
                        "x": 5,
                        "action_name": "WAIT",
                    }
                ],
            }
        ],
    )
    logger.save_batch(
        batch_id="b_nulls",
        generation=1,
        mode="SWARM",
        parent_policy_id="fp0",
        candidate_policy_id="pol_1",
        target_functions=["step"],
        runs=[{"run_id": "rb_0", "seed": 0, "final_ac": None, "max_depth": None}],
        episodes_discarded=0,
        batch_metrics={"avg_depth": None, "max_depth": None},
        verdict={"passed": True, "wilcoxon_p": None, "ci_lower": None},
    )

    con = connect_db(dbp, read_only=True)
    try:
        assert (
            con.execute(
                "SELECT avg_ac FROM tournament_attempts WHERE attempt_id='att_zero_ac'"
            ).fetchone()[0]
            == 0.0
        )
        wp, sp = con.execute(
            "SELECT wilcoxon_p, sign_p FROM stepping_stones WHERE stone_id='st_zero_p'"
        ).fetchone()
        assert wp == 0.0 and sp == 0.0
        seed_v, ac_v = con.execute(
            "SELECT seed, final_ac FROM runs WHERE batch_id='b_zero_ac'"
        ).fetchone()
        assert seed_v == 0 and ac_v == 0
        t_ac, t_hs = con.execute(
            "SELECT ac, hunger_state FROM ticks WHERE run_id LIKE 'g1_reactive_1_select_s0%'"
        ).fetchone()
        assert t_ac == 0 and t_hs == 0
        # Verify c_ac == 0.0 preserved in DeficitAnalyzer
        rep = DeficitAnalyzer(dbp).analyze(generation=2)
        assert "AC=0.0" in rep.format_brief()
    finally:
        con.close()

    # 3. summarizer.py preserves ac=0 and None fields across inflection loops
    _, _, ms = summarize_episode_ticks(
        [
            {"turn": i, "depth": 1, "hp": None, "max_hp": None, "ac": 0}
            for i in range(1, 8)
        ]
    )
    assert ms["best_ac"] == 0

    # 4. download_experts preserves hp=0 and ac=0
    exp_run = parse_dumplog_to_expert_run(
        run_id="exp_zero_test",
        meta={
            "name": "hero",
            "role": "Val",
            "hp": 0,
            "maxhp": 12,
            "ac": 0,
            "death": "killed by grid bug",
        },
        dumplog_text="",
    )
    assert exp_run["final_hp"] == 0 and exp_run["final_ac"] == 0

    # 5. TheoryDocumentStore re-initializes if DB file is replaced on disk
    tdb = tmp_path / "theory_reinit.duckdb"
    tstore1 = TheoryDocumentStore(db_path=str(tdb))
    tstore1.save_document(1, "reactive_1", "THEORY", "T1", "C1")
    tdb.unlink()
    duckdb_empty = connect_db(str(tdb), read_only=False)
    duckdb_empty.close()
    tstore2 = TheoryDocumentStore(db_path=str(tdb))
    assert tstore2.search_theories("initialization", limit=5) is not None

    # 6. parse_peer_output merges partial <skill_code> updates cleanly
    base_pol = "from skills.nav import move_ok\nclass Policy:\n    def step(self, s): return move_ok(s)\n"
    base_skills = {
        "nav": "from lox.env.actions import Action\ndef untouched_helper(s):\n    return Action.N\ndef move_ok(s):\n    return Action.WAIT\n"
    }
    raw_resp = (
        "```yaml\nmechanism: move north\ntarget_functions: [move_ok]\n```\n"
        '<skill_code module="nav">\n# PATCH\ndef move_ok(s):\n    return untouched_helper(s)\n</skill_code>\n'
    )
    _hyp, _pol, kept_sk, _pruned = parse_peer_output(raw_resp, base_pol, base_skills)
    assert "def untouched_helper" in kept_sk["nav"]
    assert "return untouched_helper(s)" in kept_sk["nav"]
