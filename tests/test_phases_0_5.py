"""
Comprehensive verification suite covering Phases 0-5 of the LOX Dlvl-10+ Upgrade:
- Phase 0/4: Rust grid extractors, FloorMemory views, Blackboard TTL blacklist, ActionLeaf preempt_if
- Phase 1: CLI args (--refactor-model, --reasoning-effort, --routing, --race-seeds, --scenario-gate),
           WikiEngine AND-tier + untruncated read_article, untruncated diffs/DB LIMIT, prompt docs
- Phase 2: pyright + .pyi stubs, cross-module contracts, #EVOLVE-BLOCK, repo-intel tools,
           pre-mortem tick slices + THEORY priority, SWARM_PEER TRIGGER_MISS
- Phase 3: synchronized progression_metric, 2-stage seed racing (32->64 style), Pareto promotion,
           N-way sequential greedy stacking, MAP-Elites caution reseed
- Phase 5: MiniHack noisy scenarios & real NLE prompt_torture suite, contracts, StrategyStack
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
from lox.author.swarm_peer import PeerCandidate
from lox.author.tools import AuthorToolRegistry, get_tool_specs
from lox.core.contracts import EpisodeRun
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import connect_db, init_db
from lox.engine.hypothesis import progression_metric
from lox.engine.map_elites import MAPElitesArchive
from lox.engine.swarm_referee import SwarmReferee
from lox.env import get_adapter
from lox.env.actions import Action
from lox.env.benchmarks.nethack_scenarios import (
    ALL_SCENARIOS,
    PROMPT_TORTURE_SUITES,
    SCENARIO_ARCHETYPES,
    run_scenario_gate,
)
from lox.env.harness import NetHackHarness
from lox.env.state import FloorMemory
from lox.lang.bt import (
    ActionLeaf,
    BehaviorTree,
    Blackboard,
    PrioritySelector,
)
from lox.lang.compiler import compile_policy_package
from lox.lang.contracts import GoalIntent, SafetyReport, validate_contract_instance
from lox.lang.strategy import (
    StrategyFrame,
    StrategyStack,
    preempt_if,
    until,
    with_cleanup,
)
from lox.lang.validator import (
    PolicyValidationError,
    check_evolve_block_bounds,
    run_pyright_on_package,
    validate_cross_module_contracts,
)


def test_phase0_rust_and_bt_preempt_and_blacklist():
    bb = Blackboard()
    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=42)
        assert isinstance(st.adjacent_closed_doors(), list)
        frontier = st.unexplored_frontier_mask()
        assert frontier.shape == (21, 79) and frontier.dtype == np.bool_

        bb.floor_blacklist_add(st, (5, 5), ttl=3)
        assert bb.floor_is_blacklisted(st, (5, 5)) is True

        # Preempt_if interrupts min_commit immediately
        ticks = {"explore": 0, "escape": 0}
        danger = {"active": False}

        def do_escape(s, _bb):
            if danger["active"]:
                ticks["escape"] += 1
                return Action.WAIT
            return None

        def do_explore(s, _bb):
            ticks["explore"] += 1
            return Action.S

        tree = BehaviorTree(
            PrioritySelector(
                [
                    ActionLeaf(do_escape, name="escape"),
                    ActionLeaf(
                        do_explore,
                        name="explore",
                        min_commit=5,
                        preempt_if=lambda _s, _bb: danger["active"],
                    ),
                ],
                name="root",
            ),
            blackboard=bb,
        )
        tree.tick(st)
        assert ticks["explore"] == 1
        danger["active"] = True
        act = tree.tick(st)
        assert act == Action.WAIT and ticks["escape"] == 1
    finally:
        h.close()

    fm = FloorMemory()
    assert fm.passable_mask.flags.writeable is False
    assert fm.copy_passable_mask().flags.writeable is True


def test_phase1_cli_args_wiki_and_prompts():
    from scripts.run_swarm import parse_args

    args_def = parse_args(["--provider", "openrouter", "--model", "z-ai/glm-5.3-flash"])
    assert args_def.refactor_model == "z-ai/glm-5.3-flash"
    assert args_def.reasoning_effort == "max"
    assert args_def.seeds == 64
    assert args_def.race_seeds == 32
    assert args_def.scenario_gate == "advisory"

    args_custom = parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.3-flash",
            "--refactor-model",
            "anthropic/claude-opus-4.5",
            "--routing",
            "exacto",
            "--race-seeds",
            "16",
            "--scenario-gate",
            "strict",
        ]
    )
    assert args_custom.refactor_model == "anthropic/claude-opus-4.5"
    assert args_custom.routing == "exacto"
    assert args_custom.race_seeds == 16
    assert args_custom.scenario_gate == "strict"

    if os.path.exists(DEFAULT_WIKI_DB):
        eng = WikiEngine(DEFAULT_WIKI_DB)
        hits = eng.search_wiki("Sokoban Level", top_k=3)
        assert hits
        art = eng.read_article("Sokoban")
        assert len(art) > 1500
        assert "Sokoban" in art

    sys_p = build_system_prompt(is_swarm=True)
    assert "preempt_if" in sys_p and "adjacent_closed_doors" in sys_p
    pol_25 = (
        "from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.tree = BehaviorTree(PrioritySelector([\n"
        + "".join(f"            ActionLeaf(f_{i}, name='fn_{i}'),\n" for i in range(25))
        + "        ]), blackboard=Blackboard())\n"
    )
    user_p = build_user_prompt(
        mode="SWARM_PEER",
        generation=3,
        current_policy=pol_25,
        skills={"nav": "def go(): return Action.N\n"},
    )
    assert "fn_24" in user_p  # No silent brs[:20] truncation!


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
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector

def step_wait(st, bb):
    return Action.WAIT

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()
        self.tree = BehaviorTree(PrioritySelector([ActionLeaf(step_wait, name='w')]), blackboard=self.blackboard)
    def step(self, state):
        return self.tree.tick(state) or Action.WAIT
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
        "nav": "def move_dir(st, bb):\n    bb.nav_target = (1, 2)\n    return bb.nav_target\n"
    }
    reg.configure_generation(
        pool=None, skills=reg.skills, generation=1, mode="SWARM_PEER"
    )
    ref_out, _ = reg.find_references("step_wait")
    assert "policy.py" in ref_out
    grep_out, _ = reg.grep_skills("nav_target")
    assert "skills/nav.py" in grep_out
    bb_out, _ = reg.blackboard_map()
    assert "nav_target" in bb_out
    parity_out, _ = reg.benchmark_expert_parity(top_n=5)
    assert "Parity" in parity_out

    peer_specs = get_tool_specs("SWARM_PEER")
    peer_tool_names = {s["function"]["name"] for s in peer_specs}
    assert "query_policy_docs" not in peer_tool_names
    assert "read_wiki_article" in peer_tool_names
    assert "blackboard_map" in peer_tool_names
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


def test_phase5_scenarios_contracts_and_strategy_stack():
    assert len(SCENARIO_ARCHETYPES) == 7
    assert len(PROMPT_TORTURE_SUITES) == 4
    assert len(ALL_SCENARIOS) == 11

    adapter = get_adapter("nethack_scenarios")
    assert adapter.name == "nethack_scenarios"

    # Prompt-aware policy resolves text/yn/getobj/getdir/menu prompts cleanly with 0 aborts/exceptions
    clean_pol = """class Policy:
    def run(self, state):
        while True:
            if (
                state.in_text_prompt
                or state.in_yn_prompt
                or state.in_getobj_prompt
                or state.in_getdir_prompt
                or state.in_menu
            ):
                state = yield Action.ESC
            else:
                state = yield Action.S
"""
    res = run_scenario_gate(clean_pol, max_steps=25)
    assert res["exceptions"] == 0
    assert res["prompt_livelocks"] == 0
    assert res["passed_count"] == len(ALL_SCENARIOS)

    # StrategyStack preemption & cleanup unwind
    cleanup_log: list[str] = []
    bb = Blackboard()
    stack = StrategyStack()
    danger = {"now": False}

    def low_gen(_st, _bb):
        yield Action.N
        yield Action.N

    def hi_gen(_st, _bb):
        yield Action.ESC

    stack.register(
        StrategyFrame(
            name="high_escape",
            priority=100,
            factory=hi_gen,
            preempt_if=preempt_if(lambda _s, _b: danger["now"]),
            until=until(lambda _s, _b: not danger["now"]),
            cleanup=with_cleanup(lambda _s, _b: cleanup_log.append("hi_done")),
        )
    )
    stack.push(
        StrategyFrame(
            name="low_explore",
            priority=10,
            factory=low_gen,
            cleanup=with_cleanup(lambda _s, _b: cleanup_log.append("low_cleaned")),
        )
    )
    assert stack.tick(None, bb) == Action.N
    danger["now"] = True
    assert stack.tick(None, bb) == Action.ESC
    assert stack.preemptions == 1
    assert "low_cleaned" in cleanup_log

    # Contracts check
    gi = GoalIntent(goal_id="descend", priority=1.0, target_pos=(4, 5))
    sr = SafetyReport(safe_to_continue=True, risk_score=0.1)
    assert gi.goal_id == "descend" and sr.safe_to_continue
    assert validate_contract_instance(object()) == []

    # Policy compiling & importing contracts + strategy inside LOX sandbox
    sandbox_pol = """from lox.lang.strategy import StrategyStack, StrategyFrame
from lox.lang.contracts import GoalIntent
class Policy:
    def __init__(self):
        self.stack = StrategyStack()
        self.intent = GoalIntent(goal_id="explore", priority=1.0)
    def step(self, state):
        return Action.WAIT
"""
    compiled = compile_policy_package(sandbox_pol)
    runner = compiled.create_runner()
    assert runner is not None
