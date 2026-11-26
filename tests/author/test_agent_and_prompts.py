"""Domain-scoped tests for LOX."""

from __future__ import annotations

from unittest.mock import patch

from lox.author.agent import AuthorAgent, SynthesisProposal
from lox.author.prompts import build_user_prompt
from lox.author.swarm_deficits import DeficitCluster
from lox.author.tools import AuthorToolRegistry


def test_author_yaml_and_code_extraction():
    sample_response = """
<think>Analyzing outlier runs...</think>
Here is my hypothesis based on the outlier good runs:

```yaml
mode: FUNC_MOD_ADD
target_functions:
  - handle_combat
  - flee_corridor
observation_from_db: "Outlier runs on Dlvl 6+ all triggered flee_corridor when HP < 40%."
mechanism: "Retreat toward corridor '#' tiles when surrounded by >= 2 hostiles."
```

And here is the updated implementation:

```python
def handle_combat(self, state):
    state = yield Action.WAIT
    return state

def flee_corridor(self, state):
    state = yield Action.SEARCH
    return state
```
"""
    hyp, code = AuthorAgent.extract_yaml_and_python(sample_response)
    assert hyp["mode"] == "FUNC_MOD_ADD"
    assert hyp["target_functions"] == ["handle_combat", "flee_corridor"]
    assert "Outlier runs" in hyp["observation_from_db"]
    assert "def handle_combat" in code
    assert "def flee_corridor" in code


def test_in_turn_repair_caret_and_prompts():
    """Verify that format_code_with_line_numbers formats caret pointers and build_repair_prompt provides category guidance."""
    from lox.author.agent import AuthorAgent
    from lox.lang.validator import format_code_with_line_numbers

    code = "def explore(self, state):\n    if state.hp < 5\n        return Action.WAIT"
    formatted = format_code_with_line_numbers(code, highlight_line=2, col_offset=20)
    assert ">  2 |     if state.hp < 5" in formatted
    assert "^" in formatted

    agent = AuthorAgent(provider="openrouter")
    unreach_p = agent.build_repair_prompt(
        status="UNREACHABLE_ERROR",
        diag="Method ['stealth_step'] was never called by run",
        mode="SWARM_PEER",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "stealth_step" in unreach_p
    assert "SWARM PEER PATCH FAILED VERIFICATION" in unreach_p
    assert "<skill_code" in unreach_p

    syntax_p = agent.build_repair_prompt(
        status="COMPILE_ERROR",
        diag="SyntaxError: invalid syntax",
        mode="WRITE",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "PYTHON SYNTAX OR INDENTATION ERROR" in syntax_p
    assert "<skill_code>" in syntax_p


def test_agent_64k_max_tokens_and_wait_dominant_rejection():
    from lox.author.swarm_deficits import DeficitReport
    from lox.author.swarm_peer import PeerCandidate, incubate_candidate

    agent = AuthorAgent(provider="mock")
    assert agent.max_tokens == 65536

    pol_zombie = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.did = False\n"
        "    def step(self, state):\n"
        "        if not self.did:\n"
        "            self.did = True\n"
        "            return Action.SEARCH\n"
        "        return Action.WAIT\n"
    )
    cand = incubate_candidate(
        PeerCandidate(
            peer_id="p_zombie",
            claimed_deficit_id="c1",
            claimed_label="zombie wait",
            policy_code=pol_zombie,
        ),
        deficit_report=DeficitReport(generation=1, regime="BOOTSTRAP"),
        claimed_deficit=DeficitCluster(cluster_id="c1", kind="test", label="test"),
    )
    assert cand.sandbox_passed is False
    assert "WAIT-dominant deadlock" in cand.error


def test_context_scoped_user_prompt_includes_skill_registry():
    combat_code = "def attack(state):\n    '''Attacks monster.'''\n    pass\n"
    prompt = build_user_prompt(
        mode="SWARM_PEER",
        generation=1,
        current_policy="class Policy: pass",
        skills={"combat": combat_code},
    )

    assert "Domain Skill Tree" in prompt
    assert "skills/combat.py" in prompt
    assert "attack" in prompt
    assert "read_skill" in prompt


def test_pass2_bug3_synthesis_proposal_skill_code_wiring() -> None:
    """Pass 2 Bug 3: SynthesisProposal must preserve skill_code extracted from AuthorAgent."""
    hyp = {
        "mode": "FUNC_MOD_ADD",
        "target_functions": ["handle_combat"],
        "skill_module": "combat_tactics",
    }
    content = """
Here is my plan to improve combat survivability:

```python:combat_tactics
def evade_trap(state):
    return Action.WAIT
```

```python
def handle_combat(self, state):
    return self.explore_step(state)
```
"""
    skill = AuthorAgent.extract_skill_code(content, hyp)
    assert "def evade_trap(state):" in skill

    # Test hypothesis with direct skill_code key
    hyp_direct = {
        "mode": "FUNC_MOD_ADD",
        "skill_module": "trap_nav",
        "skill_code": "def avoid_trap(): pass",
    }
    skill_direct = AuthorAgent.extract_skill_code("", hyp_direct)
    assert skill_direct == "def avoid_trap(): pass"

    proposal = SynthesisProposal(
        mode="FUNC_MOD_ADD",
        hypothesis=hyp,
        raw_code_patch="def handle_combat(self, state): pass",
        full_policy_code="class Policy: pass",
        full_macros_code="",
        target_functions=["handle_combat"],
        skill_code=skill,
    )
    assert proposal.skill_code == skill


def test_gen41_xml_bundle_and_partial_skill_patch_preservation(tmp_path):
    from lox.author.agent import AuthorAgent
    from lox.author.tools import (
        OPENAI_TOOL_SPECS,
        merge_skill_module_update,
    )

    # 1. OpenAI schema omits untyped additionalProperties `skills` dict
    dry_run_spec = next(
        s for s in OPENAI_TOOL_SPECS if s["function"]["name"] == "dry_run_policy"
    )
    props = dry_run_spec["function"]["parameters"]["properties"]
    assert "skills" not in props
    assert "<skill_code" in props["code"]["description"]

    # 2. Partial # PATCH update to existing skill preserves untouched exports (_delta_of)
    existing_shared = (
        "from lox.env.actions import Action\n\n"
        "def _delta_of(a, b):\n"
        "    return (b[0] - a[0], b[1] - a[1])\n\n"
        "def choose_move(state):\n"
        "    return Action.WAIT\n"
    )
    partial_patch = (
        "# PATCH\n"
        "def choose_move(state):\n"
        "    dx, dy = _delta_of((0, 0), (1, 0))\n"
        "    return Action.SEARCH if dx == 1 and dy == 0 else Action.WAIT\n"
    )
    merged = merge_skill_module_update(existing_shared, partial_patch)
    assert "def _delta_of(a, b):" in merged
    assert "return Action.SEARCH" in merged

    # 3. dry_run_policy unpacks XML bundle (<skill_code> + <policy_code>) and keeps _delta_of
    base_policy = (
        "from lox.env.actions import Action\n"
        "from skills.shared import _delta_of, choose_move\n\n"
        "def dispatch_move(state):\n"
        "    return choose_move(state)\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        _ = _delta_of((0, 0), (0, 1))\n"
        "        return dispatch_move(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "xml.duckdb"))
    tools.configure_generation(
        mode="SWARM_PEER",
        generation=41,
        skills={"shared": existing_shared},
    )
    tools.current_policy_code = base_policy

    xml_skill_only = (
        '<skill_code module="shared">\n'
        "# PATCH: update choose_move without deleting _delta_of\n"
        "def choose_move(state):\n"
        "    dx, _ = _delta_of((0, 0), (1, 0))\n"
        "    return Action.SEARCH if dx == 1 else Action.WAIT\n"
        "</skill_code>\n"
    )
    res_x, sum_x = tools.dry_run_policy(
        code=xml_skill_only,
        max_turns=20,
        smoke_only=True,
    )
    assert sum_x.startswith("PASSED"), res_x
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]
    assert "Action.SEARCH" in tools.last_passed_skills["shared"]

    # 4. XML bundle with both <skill_code> and <policy_code> splice
    xml_both = (
        '<skill_code module="shared">\n'
        "def choose_move(state):\n"
        "    return Action.SEARCH\n"
        "</skill_code>\n"
        "<policy_code>\n"
        "def dispatch_move(state):\n"
        "    _tag = 'mv2'\n"
        "    return choose_move(state)\n"
        "</policy_code>\n"
    )
    res_b, sum_b = tools.dry_run_policy(code=xml_both, max_turns=20, smoke_only=True)
    assert sum_b.startswith("PASSED"), res_b
    assert "mv2" in tools.last_passed_code
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]

    # 5. AuthorAgent._execute_tool_call rescues <skill_code> emitted in assistant_text
    agent = AuthorAgent(db_path=str(tmp_path / "xml.duckdb"), provider="mock")
    agent.tools = tools
    out = agent._execute_tool_call(
        turn_idx=1,
        name="dry_run_policy",
        args={"code": "", "max_turns": 20},
        assistant_text=xml_skill_only,
    )
    assert "SMOKE" in out or "Compile" in out or "PASSED" in out or "Pilot" in out
    assert "def _delta_of(a, b):" in tools.last_passed_skills["shared"]


def test_gen41_prompt_grounding_and_xml_instructions():
    from lox.author.prompts import build_system_prompt, build_user_prompt

    sys_prompt = build_system_prompt(is_swarm=True)
    assert "champion_runs" in sys_prompt
    assert "champion_ticks" in sys_prompt
    assert "champion_function_stats" in sys_prompt

    usr_prompt = build_user_prompt(
        mode="SWARM_PEER",
        generation=41,
        current_policy="class Policy:\n    pass\n",
        batch_digest="Gen 40 summary",
    )
    assert "query_wiki" in usr_prompt
    assert "query_theories" in usr_prompt
    assert "Recent Experiment & Rejection Memory" in usr_prompt
    assert '<skill_code module="' in usr_prompt
    assert "<policy_code>" in usr_prompt
    assert "skills={{" not in usr_prompt


def test_prompt_docs_split_schema_noop_diff_and_map_elites(tmp_path):
    from lox.author.prompts import build_system_prompt
    from lox.engine.map_elites import MAPElitesArchive

    sys_swarm = build_system_prompt(is_swarm=True)
    assert "3 Strict Generator Rules" not in sys_swarm
    assert "state.in_menu" in sys_swarm
    assert "no_diag=state.mem.door_mask" in sys_swarm
    assert "`experiments`" not in sys_swarm
    assert "register_goal" in sys_swarm

    base_pol = (
        "from lox.env.actions import Action\n\n"
        "def leaf_wait(state):\n"
        "    return Action.WAIT\n\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return leaf_wait(state) or Action.WAIT\n"
    )
    tools = AuthorToolRegistry(db_path=str(tmp_path / "noop.duckdb"))
    tools.configure_generation(mode="SWARM_PEER", generation=62, skills={})
    tools.current_policy_code = base_pol

    # <hypothesis> wrapper is stripped and AST-identical candidate triggers NOOP_DIFF
    msg_noop, st_noop = tools.dry_run_policy(
        code="<hypothesis>\nhypothesis: test\n</hypothesis>\n<policy_code>\n"
        + base_pol
        + "\n</policy_code>\n",
        smoke_only=False,
    )
    assert st_noop == "NOOP_DIFF", msg_noop

    # MAPElitesArchive v2 isolates avg_depth & armor/XL vitals without max_depth aliasing
    cid_pct, d_b, v_b, t_b = MAPElitesArchive.compute_cell_id(
        {
            "avg_depth": 3.0,
            "max_depth": 9,
            "avg_ac": 5.0,
            "avg_xp_level": 3.0,
            "avg_unique_tiles": 300.0,
            "avg_turns": 1500.0,
        }
    )
    assert cid_pct.startswith("cell_v2_")
    assert (
        d_b == 3
    )  # avg_depth 3.0 in [2.5, 4.0) -> d3 (ignores single outlier max_depth 9)
    assert v_b == 2  # (10-5) + 1.5*(3-1) = 8.0 in [7.0, 12.0) -> v2
    assert t_b == 2  # 300/1500 = 0.20 in [0.16, 0.30) -> t2


def test_g17_fixes_glm5_medium_reasoning_export_preserve_and_no_nested_resplit():
    from unittest.mock import MagicMock

    from lox.author.agent import AuthorAgent
    from lox.author.tools import merge_skill_module_update
    from lox.lang.modularize import maybe_auto_modularize

    # 1. glm-5 with reasoning_effort="medium" sends "medium" (no silent "high" override) and caps max_tokens=32768
    captured_payloads = []

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok", "reasoning": "short"}}],
                "usage": {},
            }

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured_payloads.append(json)
            return _FakeResp()

    agent_med = AuthorAgent(
        provider="openrouter",
        model="z-ai/glm-5.3-flash:nitro",
        api_key="sk-test-key",
        reasoning_effort="medium",
    )
    with patch("httpx.Client", _FakeClient):
        agent_med._call_llm_api([{"role": "user", "content": "hi"}], allow_tools=False)
    assert captured_payloads[-1]["reasoning"] == {"effort": "medium"}
    assert captured_payloads[-1]["max_tokens"] == 32768

    agent_max = AuthorAgent(
        provider="openrouter",
        model="z-ai/glm-5.3-flash:nitro",
        api_key="sk-test-key",
        reasoning_effort="max",
    )
    with patch("httpx.Client", _FakeClient):
        agent_max._call_llm_api([{"role": "user", "content": "hi"}], allow_tools=False)
    assert captured_payloads[-1]["reasoning"] == {"effort": "max"}
    assert captured_payloads[-1]["max_tokens"] == 65536
    MagicMock()

    # 2. merge_skill_module_update preserves facade-exported constants/helpers when peer adds new helpers
    existing_sub = (
        "from lox.env.actions import Action\n"
        "_INF = 10 ** 8\n"
        "def _mdiff(m):\n"
        "    return 0\n"
        "def melee_step(state, bb):\n"
        "    return Action.E\n"
    )
    patch_with_new_helpers = (
        "def _hold_msg(state):\n"
        "    return ''\n"
        "def _last_move_delta(state):\n"
        "    return None\n"
        "def melee_step(state, bb):\n"
        "    return None if _hold_msg(state) else Action.N\n"
    )
    merged_sub = merge_skill_module_update(existing_sub, patch_with_new_helpers)
    assert "_INF = 10 ** 8" in merged_sub
    assert "def _mdiff(m):" in merged_sub
    assert "def _hold_msg(state):" in merged_sub

    # 3. Nested submodule under a Modular facade is NOT recursively split below 1.5x threshold
    pol = "import skills.exploration as exploration\nclass Policy:\n    def step(self, s): return exploration.go(s)\n"
    sub_350 = "\n".join(f"def _f_{i}():\n    return {i}" for i in range(180))
    ch, _, _, _ = maybe_auto_modularize(
        pol,
        {
            "exploration": '"""Modular facade for `exploration`."""\nfrom skills.exploration_nav import go\n',
            "exploration_nav": f"def go(s):\n    return None\n{sub_350}\n",
        },
        verify_seeds=[1000],
        verify_steps=10,
    )
    assert ch is False


def test_v3_agent_probe_preservation_repair_prompt_and_tool_compaction(tmp_path):
    """Verifies Component 1 & 4: synthesize preserves peer_probe_seeds, build_repair_prompt uses Option A, and _compact_tool_messages shrinks older tool outputs."""
    from unittest.mock import patch

    from lox.author.agent import AuthorAgent

    # 1. build_repair_prompt uses Option-A (self.ws / self.goals.tick), not legacy self.tree.tick
    agent = AuthorAgent(provider="openrouter", db_path=str(tmp_path / "ag.duckdb"))
    rep_p = agent.build_repair_prompt(
        status="COMPILE_ERROR",
        diag="AttributeError: bad attr",
        mode="SWARM_PEER",
        repair_idx=1,
        max_repair_turns=3,
    )
    assert "self.tree.tick" not in rep_p
    assert "self.goals.tick" in rep_p or "self.ws" in rep_p

    # 2. Rolling tool-output compaction keeps last 4 tool turns verbatim and compacts older large tool outputs
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "content": f"HDR_{i}_" + ("X" * 1200),
            }
        )
    compacted = AuthorAgent._compact_tool_messages(
        msgs, keep_recent_turns=4, max_old_chars=600, head_chars=350
    )
    tool_msgs = [m for m in compacted if m.get("role") == "tool"]
    assert len(tool_msgs) == 6
    assert "compacted earlier" in tool_msgs[0]["content"]
    assert "compacted earlier" in tool_msgs[1]["content"]
    assert len(tool_msgs[0]["content"]) < 500
    # Last 4 tool turns stay verbatim
    for tm in tool_msgs[2:]:
        assert "compacted earlier" not in tm["content"]
        assert len(tm["content"]) > 1000

    # 3. synthesize() in SWARM_PEER preserves peer_probe_seeds, peer_id, baseline_policy_code, baseline_skills
    base_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.E\n"
    agent.tools.configure_generation(
        mode="SWARM_PEER",
        generation=5,
        skills={"goals/explore": "def go(s): return None\n"},
        peer_guard_seeds=[10, 11],
        peer_target_seeds=[20, 21],
        peer_probe_seeds=[30, 31, 32, 33, 34, 35],
        peer_id="reactive_1",
    )
    agent.tools.baseline_policy_code = base_pol
    agent.tools.baseline_skills = {"goals/explore": "def go(s): return None\n"}

    seen_probes: list[list[int]] = []

    def _fake_api(messages, allow_tools=True, **kwargs):
        seen_probes.append(list(agent.tools.peer_probe_seeds))
        return {
            "role": "assistant",
            "content": (
                "```yaml\nobservation_from_db: test\nmechanism: ok\n```\n"
                "<policy_code>\n" + base_pol + "\n</policy_code>"
            ),
            "tool_calls": [],
        }

    with patch.object(agent, "_call_llm_api", side_effect=_fake_api):
        agent.synthesize(
            mode="SWARM_PEER",
            generation=5,
            current_policy=base_pol,
            skills={"goals/explore": "def go(s): return None\n"},
            max_tool_turns=2,
        )
    assert seen_probes and seen_probes[0] == [30, 31, 32, 33, 34, 35]
    assert agent.tools.peer_id == "reactive_1"
    assert agent.tools.baseline_policy_code == base_pol


def test_openrouter_cost_tool_arg_compaction_and_budget_bypass_fix(tmp_path):
    """Verifies live usage.cost extraction, stale tool_calls compaction, and post-pilot budget bypass fix."""
    import json
    from unittest.mock import patch

    from lox.author.agent import AuthorAgent
    from lox.author.llm_client import compact_tool_messages, extract_usage_cost

    # 1. Exact OpenRouter usage.cost extraction preserves 0.0 and ignores bool/missing
    assert extract_usage_cost({"cost": 0.0042, "prompt_tokens": 100}) == 0.0042
    assert extract_usage_cost({"cost": 0.0}) == 0.0
    assert extract_usage_cost({"total_cost": 0.015}) == 0.015
    assert extract_usage_cost({"cost": True}) is None
    assert extract_usage_cost({"prompt_tokens": 100}) is None

    # 2. Old assistant tool_calls arguments > max_old_tool_arg_chars are compacted cleanly as valid JSON
    big_code = "def x():\n    return 1\n" * 120
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_old",
                    "type": "function",
                    "function": {
                        "name": "dry_run_policy",
                        "arguments": json.dumps({"code": big_code}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_old",
            "name": "dry_run_policy",
            "content": "ok",
        },
        {"role": "assistant", "content": "r1"},
        {"role": "tool", "tool_call_id": "tc_1", "name": "query_db", "content": "res1"},
        {"role": "assistant", "content": "r2"},
        {"role": "tool", "tool_call_id": "tc_2", "name": "query_db", "content": "res2"},
    ]
    comp = compact_tool_messages(msgs, keep_recent_turns=2)
    comp_args = json.loads(comp[2]["tool_calls"][0]["function"]["arguments"])
    assert "_compacted" in comp_args
    assert "dry_run_policy" in comp_args["_compacted"]

    # 3. _build_post_tool_budget_notice catches post-pilot SQL loops even when last_dry_run_passed=True
    repilot_msg = AuthorAgent._build_post_tool_budget_notice(
        turn_idx=8,
        effective_max_turns=22,
        turns_since_last_dry_run=3,
        last_dry_run_passed=True,
        last_peer_micro_pilot={
            "diverged": 4,
            "wins": 1,
            "losses": 5,
            "avg_depth_delta": -1.2,
            "guard_regressions": 1,
            "policy_exceptions": 0,
        },
    )
    assert repilot_msg and "Re-Pilot Notice" in repilot_msg

    clean_wrap_msg = AuthorAgent._build_post_tool_budget_notice(
        turn_idx=8,
        effective_max_turns=22,
        turns_since_last_dry_run=3,
        last_dry_run_passed=True,
        last_peer_micro_pilot={
            "diverged": 6,
            "wins": 4,
            "losses": 0,
            "avg_depth_delta": 0.5,
            "guard_regressions": 0,
            "policy_exceptions": 0,
        },
    )
    assert clean_wrap_msg and "already passed cleanly" in clean_wrap_msg

    # 4. End-to-end save_synthesis_log receives exact OpenRouter usage.cost instead of rate-table fallback
    dbp = str(tmp_path / "cost_log.duckdb")
    agent = AuthorAgent(provider="openrouter", db_path=dbp)
    pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, s):\n        return Action.E\n"

    def _fake_api_cost(messages, allow_tools=True, **kwargs):
        return {
            "role": "assistant",
            "content": (
                "```yaml\nobservation_from_db: ok\nmechanism: ok\n```\n"
                "<policy_code>\n" + pol + "\n</policy_code>"
            ),
            "tool_calls": [],
            "_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "cost": 0.00777,
            },
        }

    with patch.object(agent, "_call_llm_api", side_effect=_fake_api_cost):
        agent.synthesize(
            mode="SWARM_PEER",
            generation=7,
            current_policy=pol,
            max_tool_turns=2,
        )
    from lox.db.schema import connect_db

    con = connect_db(dbp, read_only=True)
    try:
        rows = con.execute(
            "SELECT cost_usd FROM synthesis_logs WHERE generation = 7"
        ).fetchall()
    finally:
        con.close()
    assert any(abs(float(r[0] or 0.0) - 0.00777) < 1e-9 for r in rows)
