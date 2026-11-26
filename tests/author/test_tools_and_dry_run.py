"""Domain-scoped tests for LOX."""

from __future__ import annotations

from unittest.mock import patch

from lox.author.agent import AuthorAgent
from lox.author.swarm_deficits import DeficitCluster
from lox.author.tools import AuthorToolRegistry
from lox.db.schema import init_db
from lox.env.actions import Action
from lox.lang.validator import extract_module_outline


def test_query_db_untruncated_long_strings_for_small_rowsets(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    db_file = str(tmp_path / "test_untruncated.duckdb")
    con = init_db(db_file)
    long_digest = "EVENT_START " + ("A" * 2500) + " EVENT_END"
    con.execute(
        "INSERT INTO expert_runs (run_id, turns, depth, score, summary_digest) VALUES (?, ?, ?, ?, ?)",
        ["exp_test_01", 100, 3, 200, long_digest],
    )
    con.close()

    tools = AuthorToolRegistry(db_path=db_file)
    out, count_str = tools.query_db(
        "SELECT summary_digest FROM expert_runs WHERE run_id = 'exp_test_01'"
    )
    assert count_str == "1 rows"
    # Verify the 2500-char string was NOT truncated down to 700 chars
    assert "EVENT_START" in out
    assert "EVENT_END" in out
    assert len(out) >= 2500


def test_dry_run_policy_behavioral_critic_detects_oscillation(tmp_path):
    from lox.author.tools import AuthorToolRegistry

    tools = AuthorToolRegistry(db_path=str(tmp_path / "db.duckdb"))
    tools.current_policy_code = ""

    # Policy that walks 5 steps East then endlessly ping-pongs North/South
    oscillating_code = """class Policy:
    def __init__(self):
        self.step_idx = 0
    def run(self, state):
        while True:
            self.step_idx += 1
            if self.step_idx <= 4:
                state = yield Action.E
            elif self.step_idx % 2 == 1:
                state = yield Action.N
            else:
                state = yield Action.S
"""
    out, status = tools.dry_run_policy(code=oscillating_code, max_turns=70, seed=42)
    assert "BEHAVIORAL WARNING" in out
    assert "oscillation detected" in out


def test_dry_run_policy_replay_run_id_fast_forward(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    db_file = str(tmp_path / "replay_test.duckdb")
    con = init_db(db_file)
    # Simulate a run that took 10 Action.WAIT steps on seed 99
    sim_bytes = bytes([int(Action.WAIT)] * 10)
    con.execute(
        "INSERT INTO runs (run_id, batch_id, seed, steps, action_bytes) VALUES (?, ?, ?, ?, ?)",
        ["run_rec_99", "b0", 99, 10, sim_bytes],
    )
    con.close()

    tools = AuthorToolRegistry(db_path=db_file)
    test_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
"""
    out, status = tools.dry_run_policy(
        code=test_code,
        seed=1,  # seed=1 is ignored because replay_run_id provides seed=99
        max_turns=15,
        replay_run_id="run_rec_99",
        replay_steps=5,
    )
    assert status.startswith("PASSED")
    assert "Dry-Run PASSED" in out


def test_dry_run_check_does_not_log_false_pilot_failed(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.db.schema import connect_db, init_db

    db_path = str(tmp_path / "dry_run_status.duckdb")
    con = init_db(db_path)
    con.close()

    agent = AuthorAgent(provider="mock", db_path=db_path, agent_role="peer_1")
    agent.provider = "openrouter"
    agent.tools.configure_generation(
        pool=None, baseline_runs=[], generation=1, mode="SWARM_PEER"
    )

    sample_policy = (
        "from lox.env.actions import Action\n"
        "def act_fn(st):\n"
        "    return Action.SEARCH\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return act_fn(state) or Action.WAIT\n"
    )

    import json

    turn_counter = {"n": 0}

    def clean_fake_llm(messages, allow_tools=True):
        if turn_counter["n"] == 0:
            turn_counter["n"] += 1
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_dry_1",
                        "function": {
                            "name": "dry_run_policy",
                            "arguments": json.dumps(
                                {"code": sample_policy, "smoke_only": True}
                            ),
                        },
                    }
                ],
            }
        return {"content": f"```python\n{sample_policy}\n```", "tool_calls": []}

    agent._call_llm_api = clean_fake_llm  # type: ignore[method-assign]
    agent.synthesize(
        mode="SWARM_PEER",
        generation=1,
        current_policy="",
        max_tool_turns=3,
    )

    con = connect_db(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT turn_type, status FROM synthesis_logs WHERE generation = 1"
        ).fetchall()
    finally:
        con.close()

    assert rows
    assert ("DRY_RUN_CHECK", "DRY_RUN_PASSED") in rows
    assert all(st != "PILOT_FAILED" for _, st in rows)

    # Also test auto-pilot verification (LLM returns code directly with no tool_calls)
    agent2 = AuthorAgent(provider="mock", db_path=db_path, agent_role="peer_2")
    agent2.provider = "openrouter"
    agent2.tools.configure_generation(
        pool=None, baseline_runs=[], generation=2, mode="SWARM_PEER"
    )
    agent2._call_llm_api = lambda messages, allow_tools=True: {
        "content": f"```python\n{sample_policy}\n```",
        "tool_calls": [],
    }  # type: ignore[method-assign]
    agent2.synthesize(
        mode="SWARM_PEER",
        generation=2,
        current_policy="",
        max_tool_turns=3,
    )
    con = connect_db(db_path, read_only=True)
    try:
        rows_g2 = con.execute(
            "SELECT turn_type, status FROM synthesis_logs WHERE generation = 2"
        ).fetchall()
    finally:
        con.close()
    assert ("DRY_RUN_CHECK", "DRY_RUN_PASSED") in rows_g2
    assert all(st != "PILOT_FAILED" for _, st in rows_g2)

    # Verify query_db Binder Error includes exact valid column list hint
    bad_msg, bad_stat = agent2.tools.query_db(
        "SELECT nonexistent_col FROM runs ORDER BY created_at DESC"
    )
    assert bad_stat == "sql_error"
    assert "Valid schema columns: runs(" in bad_msg
    assert "created_at" in bad_msg

    # Verify extract_all_skill_codes parses both XML <skill_code> and markdown ### skill: blocks
    mixed_out = """
<skill_code module="nav">
```python
def go(): return None
```
</skill_code>

### skill: combat.py
```python
def hit(): return None
```
"""
    parsed_skills = AuthorAgent.extract_all_skill_codes(mixed_out)
    assert sorted(parsed_skills.keys()) == ["combat", "nav"]
    assert "def go()" in parsed_skills["nav"]
    assert "def hit()" in parsed_skills["combat"]


def test_tools_read_policy_current_and_synthesize_preserves_peer_context(tmp_path):
    db_path = str(tmp_path / "peer_ctx.duckdb")
    init_db(db_path).close()
    agent = AuthorAgent(provider="openrouter", db_path=db_path, api_key="test-key")
    cur_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    deficit = DeficitCluster(
        cluster_id="c1",
        kind="early_trajectory",
        label="test",
    )
    agent.tools.configure_generation(
        pool=None,
        baseline_runs=[],
        generation=2,
        mode="SWARM_PEER",
        claimed_deficit=deficit,
        scenario_gate="strict",
    )
    agent.tools.enforce_peer_research_gate = True
    agent.tools.current_policy_code = cur_pol

    txt, sum_s = agent.tools.read_policy()
    assert "class Policy:" in txt
    assert "current" in sum_s

    with patch.object(
        agent,
        "_call_llm_api",
        return_value={
            "content": f"```python\n{cur_pol}\n```",
            "_reasoning": "",
            "_usage": {},
        },
    ):
        agent.synthesize(
            mode="SWARM_PEER",
            generation=2,
            current_policy=cur_pol,
            max_tool_turns=1,
        )
    assert agent.tools.claimed_deficit is deficit
    assert agent.tools.scenario_gate == "strict"
    assert agent.tools.enforce_peer_research_gate is True


def test_author_tools_symbol_queries(tmp_path):
    db_path = str(tmp_path / "test_sym.duckdb")
    con = init_db(db_path)
    con.close()

    tools = AuthorToolRegistry(db_path=db_path)
    m_out, m_stat = tools.query_monster("floating eye")
    assert "floating eye" in m_out
    assert m_stat == "found"

    i_out, i_stat = tools.query_item("potion of extra healing")
    assert "potion of extra healing" in i_out
    assert i_stat == "found"

    # Test query_db does NOT rewrite SELECT MAX(batch_id)
    # The literal query string should be executed as-is
    res, _ = tools.query_db("SELECT 1 AS num WHERE 1 = (SELECT 1)")
    assert "num: 1" in res


def test_extract_module_outline_and_read_skill_tool(tmp_path):
    skill_code = """
class CombatManager:
    \"\"\"Handles melee and ranged tactical decisions.\"\"\"
    def engage(self, state, target_id):
        \"\"\"Executes attack sequence.\"\"\"
        return 1

def emergency_escape(state):
    \"\"\"Calculates fastest escape path to corridor.\"\"\"
    return (0, 0)
"""

    outline = extract_module_outline(skill_code, module_name="combat")
    assert "# Module: combat" in outline
    assert "class CombatManager:" in outline
    assert "def engage(self, state, target_id):" in outline
    assert "def emergency_escape(state):" in outline

    # Test read_skill tool in AuthorToolRegistry
    tools = AuthorToolRegistry(db_path=":memory:")
    tools.skills["combat"] = skill_code

    content, summary = tools.read_skill("combat")
    assert "class CombatManager" in content
    assert "lines" in summary

    # Test reading non-existent skill
    err_msg, err_summary = tools.read_skill("non_existent")
    assert "not found" in err_msg
    assert err_summary == "not_found"


def test_dry_run_policy_with_candidate_skill(tmp_path):
    tools = AuthorToolRegistry(db_path=":memory:")
    policy_code = """
from skills.combat import defend
from lox.env.actions import Action

class Policy:
    def run(self, state):
        while True:
            state = yield defend(self, state)
"""
    combat_code = """
from lox.env.actions import Action

def defend(self, state):
    return Action.SEARCH
"""
    diag, status = tools.dry_run_policy(
        code=policy_code,
        skill_name="combat",
        skill_code=combat_code,
        smoke_only=True,
    )
    assert "PASSED" in status or "VALID" in status


def test_ide_navigation_tools_read_symbol_ranged_skill_and_architecture_map(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db
    from lox.lang.starter import build_clean_option_a_starter

    dbp = str(tmp_path / "tools.duckdb")
    init_db(dbp).close()
    pol, skills = build_clean_option_a_starter()
    skills["goals/explore"] = (
        '"""Explore goal."""\n'
        "from lox.env.actions import Action\n"
        "import skills.goals.core as goals\n\n"
        "def _exp_step(state, ws):\n"
        "    mem = ws.domain('exp_state', dict)\n"
        "    return Action.SEARCH\n\n"
        "goals.register_goal('explore_floor', priority=10, trigger=lambda s, w: True, step=_exp_step)\n"
    )
    pol = "import skills.goals.explore  # noqa: F401\n" + pol

    reg = AuthorToolRegistry(db_path=dbp)
    reg.configure_generation(
        pool=None,
        baseline_runs=[],
        skills=skills,
        generation=1,
        mode="SWARM_PEER",
    )
    reg.current_policy_code = pol

    sliced, s_tag = reg.read_skill("goals/explore", start_line=4, end_line=7)
    assert "5: def _exp_step" in sliced
    assert "lines 4-7/" in s_tag

    sym_txt, _ = reg.read_symbol("goals/explore._exp_step")
    assert "skills/goals/explore.py:5-7" in sym_txt
    assert "def _exp_step" in sym_txt

    arch_txt, arch_sum = reg.architecture_map()
    assert "explore_floor" in arch_txt
    assert "exp_state" in arch_txt
    assert "1 goals, 1 domains" == arch_sum


def test_read_symbol_resolves_skills_dotted_class_method(tmp_path):
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db
    from lox.lang.starter import build_clean_option_a_starter

    dbp = str(tmp_path / "tools2.duckdb")
    init_db(dbp).close()
    pol, skills = build_clean_option_a_starter()
    reg = AuthorToolRegistry(db_path=dbp)
    reg.configure_generation(
        pool=None,
        baseline_runs=[],
        skills=skills,
        generation=1,
        mode="SWARM_PEER",
    )
    reg.current_policy_code = pol

    txt1, tag1 = reg.read_symbol("skills.goals.core.GoalRunner.tick")
    assert tag1 != "not_found"
    assert "GoalRunner.tick" in txt1
    assert "def tick(" in txt1

    txt2, tag2 = reg.read_symbol("goals/core.GoalRunner.tick")
    assert tag2 != "not_found"
    assert "def tick(" in txt2

    txt3, tag3 = reg.read_symbol("skills.navigation.core.step_toward")
    assert tag3 != "not_found"
    assert "def step_toward(" in txt3


def test_option_a_peer_prompt_micro_pilot_research_gate_and_search_slice(tmp_path):
    """Verifies G16-G25 plateau breakthrough: clean Option-A peer prompt, paired CRN micro-pilot diff,
    1-lookup empirical research gate, and high-search/horizon deficit slice."""
    from lox.author.prompts import build_system_prompt
    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    # 1. Option-A peer system prompt drops contradictory BehaviorTree docs & teaches Zero-Drift + Schema-Safe rules
    peer_sys = build_system_prompt(is_swarm=True, is_peer=True)
    assert "Zero-Drift Guard Scoping" in peer_sys
    assert "Schema-Safe `ws.domain(...)`" in peer_sys
    assert "Paired Micro-Pilot Diff" in peer_sys
    assert "PrioritySelector" not in peer_sys
    assert "ActionLeaf" not in peer_sys
    assert "Blackboard" not in peer_sys

    # 2. Relaxed RESEARCH_REQUIRED gate: 1 query_db lookup unlocks dry_run_policy
    dbp = str(tmp_path / "mp.duckdb")
    con = init_db(dbp)
    con.execute(
        """
        INSERT INTO batches (batch_id, generation, mode, parent_policy_id, candidate_policy_id, avg_depth, max_depth, avg_score, avg_turns, hypothesis_passed)
        VALUES ('b1', 18, 'SWARM', 'p0', 'p18', 2.5, 6, 420.0, 2200.0, TRUE)
        """
    )
    # Seed 101: Dlvl 6 guard (healthy); Seed 202: high-search starvation at Dlvl 3; Seed 203: shallow max_steps confinement
    con.execute(
        """
        INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, score, turns, steps, final_hp, max_hp, final_ac, xp_level, end_reason, pct_wait, pct_search, unique_tiles)
        VALUES
          ('r101', 'b1', 18, 101, 6, 6, 950, 1200, 1200, 18, 22, 2, 4, 'killed', 1.0, 4.0, 320),
          ('r202', 'b1', 18, 202, 3, 3, 410, 6864, 6864, 0, 18, 4, 3, 'starved', 2.0, 44.5, 190),
          ('r203', 'b1', 18, 203, 2, 2, 220, 4000, 4000, 16, 16, 4, 2, 'max_steps', 3.0, 52.0, 120)
        """
    )
    con.close()

    # 3. DeficitAnalyzer isolates pct_search >= 35%, turns >= 1500 into slice_high_search_or_horizon_burn
    #    while keeping Dlvl <= 2 max_steps inside slice_max_steps_confinement
    rep = DeficitAnalyzer(dbp).analyze(
        generation=19, current_policy="class Policy: pass"
    )
    cluster_map = {c.cluster_id: c for c in rep.fail_clusters}
    assert "slice_high_search_or_horizon_burn" in cluster_map
    assert 202 in cluster_map["slice_high_search_or_horizon_burn"].sample_seeds
    assert "slice_max_steps_confinement" in cluster_map
    assert 203 in cluster_map["slice_max_steps_confinement"].sample_seeds

    # 4. Paired CRN Micro-Pilot inside SWARM_PEER dry_run_policy (with exact first-divergence byte & 2-call cap)
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    cand_pol_v1 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    cand_pol_v2 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.E\n"
    )
    cand_pol_v3 = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.W\n"
    )

    class _MockMicroPilotPool:
        max_steps = 8000

        def __init__(self):
            self.calls = 0

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            self.calls += 1
            # Target seed 202 had turns=6864 (>= 4000), so long-horizon cap scales to 6500
            assert kwargs.get("max_steps") == 6500
            runs = []
            for sd in priority_seeds or []:
                if sd == 101:
                    # Call 1 regresses guard seed @ byte 3; Call 2 has non-regressing drift; Call 3 has ZERO DRIFT
                    if self.calls == 1:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 2,
                                "score": 200,
                                "turns": 400,
                                "end_reason": "killed",
                                "pct_search": 5.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcXYZ",
                            }
                        )
                    elif self.calls == 2:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 6,
                                "score": 960,
                                "turns": 1200,
                                "end_reason": "killed",
                                "pct_search": 4.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcdeZ",
                            }
                        )
                    else:
                        runs.append(
                            {
                                "seed": 101,
                                "max_depth": 6,
                                "score": 950,
                                "turns": 1200,
                                "end_reason": "killed",
                                "pct_search": 4.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"abcdef",
                            }
                        )
                elif sd == 202:
                    # Call 3 simulates an over-tightened dead patch (ZERO DRIFT on target)
                    if self.calls == 3:
                        runs.append(
                            {
                                "seed": 202,
                                "max_depth": 3,
                                "score": 410,
                                "turns": 6864,
                                "end_reason": "starved",
                                "pct_search": 44.5,
                                "policy_exceptions": 0,
                                "action_bytes": b"ssssss",
                            }
                        )
                    else:
                        runs.append(
                            {
                                "seed": 202,
                                "max_depth": 5,
                                "score": 760,
                                "turns": 1800,
                                "end_reason": "killed",
                                "pct_search": 12.0,
                                "policy_exceptions": 0,
                                "action_bytes": b"ssssNN",
                            }
                        )
            return {"batch_id": batch_id, "runs": runs}

    mp_pool = _MockMicroPilotPool()
    baseline_runs = [
        {
            "seed": 101,
            "max_depth": 6,
            "score": 950,
            "turns": 1200,
            "end_reason": "killed",
            "pct_search": 4.0,
            "policy_exceptions": 0,
            "action_bytes": b"abcdef",
        },
        {
            "seed": 202,
            "max_depth": 3,
            "score": 410,
            "turns": 6864,
            "end_reason": "starved",
            "pct_search": 44.5,
            "policy_exceptions": 0,
            "action_bytes": b"ssssss",
        },
    ]

    reg = AuthorToolRegistry(db_path=dbp)
    reg.configure_generation(
        pool=mp_pool,
        baseline_runs=baseline_runs,
        generation=19,
        mode="SWARM_PEER",
        peer_guard_seeds=[101],
        peer_target_seeds=[202],
        max_peer_micro_pilots=3,
    )
    reg.current_policy_code = base_pol
    reg.enforce_peer_research_gate = True

    # Blocked before empirical lookup
    _, st0 = reg.dry_run_policy(cand_pol_v1, seed=42, max_turns=15)
    assert st0 == "RESEARCH_REQUIRED"

    # Single query_db call unlocks dry_run_policy
    _, db_st = reg.query_db("SELECT seed, max_depth FROM runs LIMIT 3")
    assert "rows" in db_st

    # Micro-pilot 1: detects GUARD regression @ byte 3 + TARGET rescue
    out1, st1 = reg.dry_run_policy(cand_pol_v1, seed=42, max_turns=15)
    assert st1.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (1/3" in out1
    assert "GUARD seed=101: ⚠️ REGRESSION (DRIFT @ byte 3 -> Dlvl 6->2" in out1
    assert "GUARD DRIFT WARNING" in out1
    assert "TARGET seed=202: ✅ RESCUED/IMPROVED (drift @ byte 4 -> Dlvl 3->5" in out1
    assert mp_pool.calls == 1

    # Micro-pilot 2: non-regressing guard drift explicitly marked OK (do NOT over-tighten)
    out2, st2 = reg.dry_run_policy(cand_pol_v2, seed=42, max_turns=15)
    assert st2.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (2/3" in out2
    assert "GUARD seed=101: OK (non-regressing drift @ byte 5" in out2
    assert "do NOT over-tighten" in out2
    assert "GUARD DRIFT WARNING" not in out2
    assert mp_pool.calls == 2

    # Micro-pilot 3: over-tightened dead patch triggers DEAD-PATCH WARNING + 1 bonus retry (3 -> 4)
    out3, st3 = reg.dry_run_policy(cand_pol_v3, seed=42, max_turns=15)
    assert st3.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (3/3" in out3
    assert "GUARD seed=101: ZERO DRIFT (Dlvl 6" in out3
    assert "DEAD-PATCH WARNING" in out3
    assert "+1 bonus micro-pilot granted" in out3
    assert mp_pool.calls == 3

    # Micro-pilot 4: uses the 1 bonus retry
    out4, st4 = reg.dry_run_policy(
        cand_pol_v3 + "\n# bonus retry\n", seed=42, max_turns=15
    )
    assert st4.startswith("PASSED")
    assert "Paired Micro-Pilot Diff vs Incumbent (4/4" in out4
    assert mp_pool.calls == 4

    # Micro-pilot 5: respects cap 4/4 (does not call eval_pool again)
    out5, st5 = reg.dry_run_policy(
        cand_pol_v3 + "\n# cap check\n", seed=42, max_turns=15
    )
    assert st5.startswith("PASSED")
    assert "cap 4/4 reached" in out5
    assert mp_pool.calls == 4
    reg.close()


def test_micro_pilot_6_probes_net_summary_and_guard_jitter(tmp_path):
    """
    Verifies Gen 52-53 candidate-quality fixes in _run_peer_paired_micro_pilot:
    1. Evaluates up to 6 PROBE seeds and emits Micro-Pilot Net Summary.
    2. Emits GENERALIZATION REGRESSION WARNING when >= 2 PROBE seeds regress.
    3. Emits ZERO-PROBE-DRIFT NOTE when 0/4+ PROBE seeds diverge.
    4. Does NOT falsely flag minor GUARD score jitter at equal depth as REGRESSION.
    """
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    dbp = str(tmp_path / "mp6.duckdb")
    init_db(dbp)
    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.N\n"
    )
    probes = [5000001 + i for i in range(6)]

    class _ProbeTestPool:
        max_steps = 1000

        def __init__(self, mode: str):
            self.mode = mode

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            runs = []
            for s in s_list:
                if s == 10:  # GUARD: equal depth, -120 score jitter (must stay OK!)
                    runs.append(
                        {
                            "run_id": f"mp_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 380,
                            "turns": 500,
                            "pct_search": 10.0,
                            "end_reason": "killed",
                            "action_bytes": b"AB",
                        }
                    )
                elif s == 20:  # TARGET: rescued
                    runs.append(
                        {
                            "run_id": f"mp_{s}",
                            "seed": s,
                            "depth": 4,
                            "max_depth": 4,
                            "score": 800,
                            "turns": 500,
                            "pct_search": 10.0,
                            "end_reason": "killed",
                            "action_bytes": b"XY",
                        }
                    )
                elif self.mode == "regress" and s in probes[:3]:
                    runs.append(
                        {
                            "run_id": f"mp_{s}",
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 100,
                            "turns": 500,
                            "pct_search": 10.0,
                            "end_reason": "killed",
                            "action_bytes": b"ZZ",
                        }
                    )
                else:
                    runs.append(
                        {
                            "run_id": f"mp_{s}",
                            "seed": s,
                            "depth": 3,
                            "max_depth": 3,
                            "score": 500,
                            "turns": 500,
                            "pct_search": 10.0,
                            "end_reason": "killed",
                            "action_bytes": b"AA",
                        }
                    )
            return {"batch_id": batch_id, "runs": runs}

    base_runs = [
        {
            "seed": s,
            "depth": 2 if s == 20 else 3,
            "max_depth": 2 if s == 20 else 3,
            "score": 500,
            "turns": 500,
            "pct_search": 10.0,
            "end_reason": "killed",
            "action_bytes": b"AA",
        }
        for s in [10, 20] + probes
    ]

    reg_warn = AuthorToolRegistry(db_path=dbp)
    reg_warn.current_policy_code = base_pol
    reg_warn.configure_generation(
        pool=_ProbeTestPool("regress"),
        baseline_runs=base_runs,
        peer_guard_seeds=[10],
        peer_target_seeds=[20],
        peer_probe_seeds=probes,
        max_peer_micro_pilots=2,
    )
    out_reg = reg_warn._run_peer_paired_micro_pilot(base_pol, {})
    assert "GUARD seed=10: OK" in out_reg
    assert "GUARD DRIFT WARNING" not in out_reg
    assert "Micro-Pilot Net Summary:" in out_reg
    assert "PROBE diverged=3/6, regressed=3/6" in out_reg
    assert "GENERALIZATION REGRESSION WARNING" in out_reg

    reg_zero = AuthorToolRegistry(db_path=dbp)
    reg_zero.current_policy_code = base_pol
    reg_zero.configure_generation(
        pool=_ProbeTestPool("zero"),
        baseline_runs=base_runs,
        peer_guard_seeds=[10],
        peer_target_seeds=[20],
        peer_probe_seeds=probes,
        max_peer_micro_pilots=2,
    )
    out_zero = reg_zero._run_peer_paired_micro_pilot(base_pol, {})
    assert "PROBE diverged=0/6, regressed=0/6" in out_zero
    assert "ZERO-PROBE-DRIFT NOTE" in out_zero


def test_dry_run_replay_runs_micro_pilot_and_flags_auto_esc_spike(tmp_path):
    """
    Verifies Gen 63-64 fixes:
    1. dry_run_policy(..., replay_run_id=...) in SWARM_PEER mode STILL runs the Paired Micro-Pilot
       instead of silently skipping it.
    2. _run_peer_paired_micro_pilot surfaces auto_esc spikes and flags them as regressions.
    """
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import connect_db, init_db
    from lox.env.actions import Action

    dbp = str(tmp_path / "replay_mp.duckdb")
    init_db(dbp)
    con = connect_db(dbp, read_only=False)
    try:
        con.execute(
            "INSERT INTO batches (batch_id, generation, mode, hypothesis_passed) VALUES ('b64', 64, 'SWARM', TRUE)"
        )
        con.execute(
            """
            INSERT INTO runs (run_id, batch_id, generation, seed, depth, max_depth, turns, score, end_reason, action_bytes, is_shard)
            VALUES ('run_r64', 'b64', 64, 42, 1, 1, 10, 0, 'killed', ?, FALSE)
            """,
            [bytes([int(Action.SEARCH)] * 6)],
        )
    finally:
        con.close()

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.SEARCH\n"
    )

    class _EscPool:
        max_steps = 1000

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [])
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"e_{s}",
                        "seed": s,
                        "depth": 3,
                        "max_depth": 3,
                        "score": 500,
                        "turns": 500,
                        "pct_search": 10.0,
                        "auto_esc_count": 4 if s == 5000001 else 0,
                        "end_reason": "killed",
                        "action_bytes": b"XY",
                    }
                    for s in s_list
                ],
            }

    reg = AuthorToolRegistry(db_path=dbp)
    reg.current_policy_code = base_pol
    reg.configure_generation(
        pool=_EscPool(),
        mode="SWARM_PEER",
        baseline_runs=[
            {
                "seed": s,
                "depth": 3,
                "max_depth": 3,
                "score": 500,
                "turns": 500,
                "pct_search": 10.0,
                "auto_esc_count": 0,
                "end_reason": "killed",
                "action_bytes": b"AA",
            }
            for s in (10, 20, 5000001)
        ],
        peer_guard_seeds=[10],
        peer_target_seeds=[20],
        peer_probe_seeds=[5000001],
        max_peer_micro_pilots=4,
    )
    out_txt, status = reg.dry_run_policy(
        code=base_pol,
        max_turns=10,
        replay_run_id="run_r64",
        replay_steps=3,
    )
    assert "PASSED" in status
    assert "Paired Micro-Pilot Diff vs Incumbent" in out_txt
    assert "PROBE seed=5000001: ⚠️ REGRESSED" in out_txt
    assert "auto_esc=0->4" in out_txt


def test_micro_pilot_logs_queryable_ticks_and_live_terminal_state(tmp_path):
    """
    Verifies:
    1. `_run_peer_paired_micro_pilot` calls `evaluate_batch` with `record_ticks=True`
       and persists both `micro_pilot` and `micro_pilot_baseline` ticks into unified
       `runs` + `ticks`, immediately queryable via `query_db` on `peer_ticks`.
    2. Smoke replay preserves the last live state (`Dlvl > 0, HP > 0`) instead of
       printing `Dlvl=0, Pos=(0,0), HP=0/0, Turn=0` when the terminal frame zeroes blstats.
    """
    from lox.author.tools import AuthorToolRegistry
    from lox.db.schema import init_db

    dbp = str(tmp_path / "mp_ticks.duckdb")
    init_db(dbp).close()

    base_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, s):\n"
        "        return Action.SEARCH\n"
    )
    seen_record_ticks: list[bool] = []

    class _TickPool:
        max_steps = 500

        def evaluate_batch(
            self, batch_id, priority_seeds=None, record_ticks=False, **kwargs
        ):
            seen_record_ticks.append(bool(record_ticks))
            s_list = list(priority_seeds or [])
            is_inc = "_inc_" in str(batch_id)
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"r_{s}",
                        "seed": s,
                        "depth": 2 if is_inc else 3,
                        "max_depth": 2 if is_inc else 3,
                        "score": 100 if is_inc else 250,
                        "turns": 50,
                        "pct_search": 5.0,
                        "end_reason": "killed",
                        "action_bytes": b"AA" if is_inc else b"AB",
                        "ticks": [
                            {
                                "turn": t,
                                "step_idx": t,
                                "depth": 1 if is_inc else 2,
                                "hp": 14,
                                "max_hp": 16,
                                "action_name": "SEARCH" if is_inc else "DOWN",
                                "active_function": "step",
                                "message": "base_msg" if is_inc else "cand_msg",
                            }
                            for t in (1, 2, 3)
                        ],
                    }
                    for s in s_list
                ],
            }

    reg = AuthorToolRegistry(db_path=dbp)
    reg.current_policy_code = base_pol
    reg.configure_generation(
        pool=_TickPool(),
        generation=67,
        mode="SWARM_PEER",
        baseline_runs=[],
        peer_guard_seeds=[10],
        peer_target_seeds=[20],
        peer_probe_seeds=[30],
        max_peer_micro_pilots=4,
        peer_id="peer_2",
    )
    cand_pol = base_pol.replace("Action.SEARCH", "Action.WAIT")
    out_txt, status = reg.dry_run_policy(code=cand_pol, seed=42, max_turns=10)
    assert "PASSED" in status
    assert seen_record_ticks and all(seen_record_ticks)
    assert "Queryable Ticks Logged" in out_txt
    assert "peer_id = 'peer_2'" in out_txt
    assert "Status=ALIVE" in out_txt

    # Peer can immediately query its own micro-pilot & baseline ticks via raw SQL
    sql_out, sql_st = reg.query_db(
        "SELECT run_kind, seed, COUNT(*) AS n_ticks FROM peer_ticks "
        "WHERE generation = 67 AND peer_id = 'peer_2' AND attempt_num = 1 "
        "GROUP BY run_kind, seed ORDER BY run_kind, seed"
    )
    assert "6 rows" in sql_st, (sql_out, sql_st)
    assert "micro_pilot" in sql_out
    assert "micro_pilot_baseline" in sql_out


def test_v3_micro_pilot_best_recovery_and_dead_patch_hygiene(tmp_path):
    """Verifies Component 4: best_peer_micro_pilot tracks best passing attempt and SwarmPeerAgent auto-recovers it when final draft degrades."""
    from unittest.mock import patch

    from lox.author.agent import SynthesisProposal
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import SwarmPeerAgent
    from lox.db.schema import init_db

    dbp = str(tmp_path / "mp_hygiene.duckdb")
    init_db(dbp).close()

    base_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.SEARCH\n"
    good_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.E\n"
    bad_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.W\n"

    peer = SwarmPeerAgent(
        peer_id="reactive_1",
        provider="openrouter",
        model="test-model",
        db_path=dbp,
        max_tool_turns=22,
    )

    def _fake_synth(**kwargs):
        # Simulate attempt 1 good (+1.5 depth, diverged) followed by attempt 2 degraded (-1.0 depth, 0 wins)
        peer.author.tools.best_peer_micro_pilot = {
            "attempt": 1,
            "code": good_pol,
            "skills": {},
            "avg_depth_delta": 1.5,
            "wins": 3,
            "losses": 0,
            "ties": 1,
            "diverged": 3,
            "probe_diverged": 2,
            "policy_exceptions": 0,
            "guard_regressions": 0,
        }
        peer.author.tools.last_peer_micro_pilot = {
            "attempt": 2,
            "code": bad_pol,
            "skills": {},
            "avg_depth_delta": -1.0,
            "wins": 0,
            "losses": 3,
            "ties": 1,
            "diverged": 3,
            "probe_diverged": 2,
            "policy_exceptions": 0,
            "guard_regressions": 1,
        }
        return SynthesisProposal(
            mode="SWARM_PEER",
            hypothesis={"mechanism": "test_recovery"},
            raw_code_patch=bad_pol,
            full_policy_code=bad_pol,
            full_macros_code="",
            target_functions=["step"],
        )

    rep = DeficitReport(generation=10, regime="PLATEAU", d_succ_seeds=[1, 2])
    cl = DeficitCluster("c1", "death", "test", 5, [3, 4], "ev")
    with patch.object(peer.author, "synthesize", side_effect=_fake_synth):
        cand = peer.run_cycle(
            generation=10,
            deficit_report=rep,
            claimed_deficit=cl,
            current_policy=base_pol,
            current_skills={},
            eval_pool=None,
            track="reactive",
        )
    assert cand.sandbox_passed is True
    assert "Action.E" in cand.policy_code

    # Dead patch (0 diverged on both best & last) is rejected cleanly before tournament
    def _fake_dead(**kwargs):
        dead_rec = {
            "attempt": 1,
            "code": bad_pol,
            "skills": {},
            "avg_depth_delta": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 4,
            "diverged": 0,
            "probe_diverged": 0,
            "policy_exceptions": 0,
            "guard_regressions": 0,
        }
        peer.author.tools.best_peer_micro_pilot = dead_rec
        peer.author.tools.last_peer_micro_pilot = dead_rec
        return SynthesisProposal(
            mode="SWARM_PEER",
            hypothesis={"mechanism": "dead"},
            raw_code_patch=bad_pol,
            full_policy_code=bad_pol,
            full_macros_code="",
            target_functions=["step"],
        )

    with patch.object(peer.author, "synthesize", side_effect=_fake_dead):
        cand_dead = peer.run_cycle(
            generation=10,
            deficit_report=rep,
            claimed_deficit=cl,
            current_policy=base_pol,
            current_skills={},
            eval_pool=None,
            track="reactive",
        )
    assert cand_dead.sandbox_passed is False
    assert "0-byte behavioral divergence" in cand_dead.error

    # If the peer revises its final submission AFTER a failed/dead micro-pilot, hygiene must NOT clobber it!
    fixed_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.N\n"

    def _fake_fixed_after_dead(**kwargs):
        peer.author.tools.best_peer_micro_pilot = {
            "attempt": 1,
            "code": bad_pol,
            "skills": {},
            "avg_depth_delta": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 4,
            "diverged": 0,
            "probe_diverged": 0,
            "policy_exceptions": 0,
            "guard_regressions": 0,
        }
        peer.author.tools.last_peer_micro_pilot = (
            peer.author.tools.best_peer_micro_pilot
        )
        return SynthesisProposal(
            mode="SWARM_PEER",
            hypothesis={"mechanism": "fixed_in_final_turn"},
            raw_code_patch=fixed_pol,
            full_policy_code=fixed_pol,
            full_macros_code="",
            target_functions=["step"],
        )

    with patch.object(peer.author, "synthesize", side_effect=_fake_fixed_after_dead):
        cand_fixed = peer.run_cycle(
            generation=10,
            deficit_report=rep,
            claimed_deficit=cl,
            current_policy=base_pol,
            current_skills={},
            eval_pool=None,
            track="reactive",
        )
    assert cand_fixed.sandbox_passed is True
    assert "Action.N" in cand_fixed.policy_code


def test_query_wiki_and_read_wiki_article_char_caps(tmp_path):
    """Verifies Guidebook-sized exact/#1 wiki hits and default read_wiki_article tool reads stay bounded."""
    import sqlite3

    from lox.author.tools import AuthorToolRegistry

    wdb = str(tmp_path / "mini_wiki.db")
    con = sqlite3.connect(wdb)
    try:
        con.execute(
            "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, content TEXT)"
        )
        con.execute(
            "CREATE VIRTUAL TABLE wiki_fts USING fts5(title, content, content='articles', content_rowid='id')"
        )
        con.execute(
            "CREATE TABLE redirects (alias TEXT PRIMARY KEY, target_title TEXT)"
        )
        big_text = "Guidebook entry about hunger and stamina. " * 2500  # ~105k chars
        sec_text = "Secondary encyclopedia reference about food. " * 400  # ~17k chars
        con.execute(
            "INSERT INTO articles (id, title, content) VALUES (1, 'NetHack 3.6.6 Guidebook', ?)",
            (big_text,),
        )
        con.execute(
            "INSERT INTO articles (id, title, content) VALUES (2, 'HungerGuide', ?)",
            (sec_text,),
        )
        con.execute(
            "INSERT INTO wiki_fts (rowid, title, content) VALUES (1, 'NetHack 3.6.6 Guidebook', ?)",
            (big_text,),
        )
        con.execute(
            "INSERT INTO wiki_fts (rowid, title, content) VALUES (2, 'HungerGuide', ?)",
            (sec_text,),
        )
        con.commit()
    finally:
        con.close()

    reg = AuthorToolRegistry(db_path=str(tmp_path / "t.duckdb"), wiki_db_path=wdb)
    q_out, _ = reg.query_wiki("NetHack 3.6.6 Guidebook", top_k=2)
    assert len(q_out) < 12000
    assert "Showing 8000/" in q_out
    assert "read_wiki_article(" in q_out

    r_out, _ = reg.read_wiki_article("NetHack 3.6.6 Guidebook")
    assert len(r_out) < 17000
    assert "TRUNCATED:" in r_out
    assert "offset=16000" in r_out


def test_post_incubation_auto_recover_rescues_winning_micro_pilot(tmp_path):
    """Verifies Fix 4: if an un-piloted late-turn edit fails incubation, SwarmPeerAgent recovers a strictly winning best_peer_micro_pilot."""
    from unittest.mock import patch

    from lox.author.agent import SynthesisProposal
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import SwarmPeerAgent
    from lox.db.schema import init_db

    dbp = str(tmp_path / "post_rec.duckdb")
    init_db(dbp).close()

    base_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.WAIT\n"
    win_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.E\n"
    bad_final_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.W\n"

    peer = SwarmPeerAgent(peer_id="proactive_2", provider="openrouter", db_path=dbp)
    rep = DeficitReport(generation=87, regime="PLATEAU", d_succ_seeds=[10])
    cl = DeficitCluster(
        cluster_id="proactive_sheltered_branch",
        kind="proactive_branch",
        label="step 4/4",
        sample_seeds=[20],
    )

    def _fake_synth(**kwargs):
        peer.author.tools.best_peer_micro_pilot = {
            "attempt": 2,
            "code": win_pol,
            "skills": {},
            "avg_depth_delta": 0.643,
            "wins": 3,
            "losses": 0,
            "ties": 11,
            "diverged": 4,
            "probe_diverged": 3,
            "policy_exceptions": 0,
            "guard_regressions": 0,
        }
        peer.author.tools.last_peer_micro_pilot = (
            peer.author.tools.best_peer_micro_pilot
        )
        return SynthesisProposal(
            mode="SWARM_PEER",
            hypothesis={"mechanism": "late_unpiloted_rewrite"},
            raw_code_patch=bad_final_pol,
            full_policy_code=bad_final_pol,
            full_macros_code="",
            target_functions=["step"],
        )

    def _fake_incubate(cand, **kwargs):
        if "Action.W" in cand.policy_code:
            cand.sandbox_passed = False
            cand.error = "Incubation paired depth regression (depth_delta=-2.62)"
        else:
            cand.sandbox_passed = True
            cand.error = ""
            cand.sandbox_metrics = {"avg_depth": 4.6, "depth_delta": 0.6}
        return cand

    with (
        patch.object(peer.author, "synthesize", side_effect=_fake_synth),
        patch("lox.author.swarm_peer.incubate_candidate", side_effect=_fake_incubate),
    ):
        res = peer.run_cycle(
            generation=87,
            deficit_report=rep,
            claimed_deficit=cl,
            current_policy=base_pol,
            current_skills={},
            eval_pool=None,
            track="proactive_branch",
        )
    assert res.sandbox_passed is True
    assert "Action.E" in res.policy_code
