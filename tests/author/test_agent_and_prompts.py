"""Domain-scoped tests for LOX."""

from __future__ import annotations

from unittest.mock import patch

from lox.author.agent import AuthorAgent, SynthesisProposal
from lox.author.prompts import build_user_prompt
from lox.author.swarm_deficits import DeficitCluster
from lox.author.tools import AuthorToolRegistry
from lox.db.schema import init_db
from lox.runner.pool import PersistentEvalPool


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


def test_author_agent_auto_pilot_dispatch(tmp_path, monkeypatch):

    db_path = str(tmp_path / "test_ap.duckdb")
    con = init_db(db_path)
    con.close()

    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=20)
    try:
        agent = AuthorAgent(provider="openrouter", db_path=db_path)
        baseline_runs = [
            {"seed": 10, "max_depth": 0, "score": 0},
            {"seed": 11, "max_depth": 0, "score": 0},
        ]

        turn_count = 0

        def mock_call_llm_api(messages, allow_tools=True):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                # LLM outputs YAML + code directly without calling dry_run_policy
                return {
                    "role": "assistant",
                    "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
observation_from_db: "Baseline exploration"
mechanism: "Wait actions"
```

```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
```""",
                    "tool_calls": [],
                }
            else:
                # LLM responds after pilot feedback
                return {
                    "role": "assistant",
                    "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
observation_from_db: "Revised exploration"
mechanism: "Wait and search"
```

```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
```""",
                    "tool_calls": [],
                }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm_api)

        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT",
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=3,
            pilot_episodes=2,
            max_tool_turns=4,
        )

        # Auto-pilot should have automatically executed the candidate code
        assert agent.tools.pilot_runs_used >= 1
        assert "class Policy" in proposal.full_policy_code
    finally:
        pool.close()


def test_scientific_investigation_protocol_and_failure_hints(tmp_path, monkeypatch):
    from lox.author.prompts import build_user_prompt

    # 1. Prompt check: FUNC_MOD_ADD mandates empirical research on Turn 1
    prompt = build_user_prompt(
        mode="FUNC_MOD_ADD",
        generation=2,
        current_policy="class Policy:\n    pass\n",
        macros_docs="",
        batch_digest="Sample digest",
    )
    assert "MANDATORY Empirical Research (Turn 1)" in prompt
    assert "DO NOT OUTPUT CODE ON TURN 1" in prompt
    assert "query_wiki" in prompt
    assert "query_db" in prompt

    # 2. Pilot failure report includes diagnostic research recommendations
    pool = PersistentEvalPool(workers=2, role="val-hum-law-fem", max_steps=10)
    try:
        tools = AuthorToolRegistry(db_path=str(tmp_path / "diag.duckdb"))
        # Baseline with higher depth so candidate policy fails
        baseline_runs = [
            {"seed": 100, "max_depth": 5, "score": 500},
            {"seed": 101, "max_depth": 5, "score": 500},
        ]
        tools.configure_generation(
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=2,
            pilot_episodes=2,
        )

        failing_code = """class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
"""
        msg, status = tools.dry_run_policy(failing_code)
        assert status.startswith("PILOT_FAILED")
        assert "PILOT BATCH" in msg
        assert "End Reasons:" in msg
        assert "MANDATORY INVESTIGATION PROTOCOL" not in msg
        assert tools.last_pilot_improved is False

        # 3. Agent synthesize sends scientific investigation guidance on pilot failure
        agent = AuthorAgent(
            provider="openrouter",
            db_path=str(tmp_path / "diag.duckdb"),
            pilot_fallback="abort",
        )
        received_messages = []

        def mock_call_llm(messages, allow_tools=True):
            received_messages.extend(messages)
            if len(received_messages) <= 3:
                # Turn 1: LLM emits code that fails
                return {
                    "role": "assistant",
                    "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'test'\nmechanism: 'test'\n```\n```python\n{failing_code}```",
                    "tool_calls": [],
                }
            else:
                # Turn 2: LLM receives failure guidance
                return {
                    "role": "assistant",
                    "content": f"```yaml\nmode: FUNC_MOD_ADD\ntarget_functions: [run]\nobservation_from_db: 'fixed'\nmechanism: 'fixed'\n```\n```python\n{failing_code}```",
                    "tool_calls": [],
                }

        monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm)
        proposal = agent.synthesize(
            mode="FUNC_MOD_ADD",
            generation=2,
            current_policy=failing_code,
            pool=pool,
            baseline_runs=baseline_runs,
            max_pilot_runs=2,
            pilot_episodes=2,
        )
        assert proposal.aborted is True
        assert "PILOT_GATE_ABORT" in proposal.abort_reason

        user_msgs = [m["content"] for m in received_messages if m.get("role") == "user"]
        failure_guidance = [m for m in user_msgs if "🛑 PILOT REVISION FAILED" in m]
        assert len(failure_guidance) > 0, (
            "Agent must send scientific investigation guidance on pilot failure"
        )
        assert "DO NOT BLINDLY GUESS CODE EDITS!" in failure_guidance[0]
        assert "query_wiki" in failure_guidance[0]
        assert "query_db" in failure_guidance[0]
    finally:
        pool.close()


def test_pass4_system_prompt_restructuring_and_turn1_gating(tmp_path, monkeypatch):
    """
    Pass 4 Verification:
    1. System prompt contains Empirical Differential Diagnostics with expert_ticks queries,
       Pre-Submission Invariant Checklist, and pruned action working set routing to query_policy_docs("actions").
    2. User prompt enforces Binding Constraint rule and Causal Hypothesis schema across modes.
    3. extract_yaml_and_python enforces Causal Hypothesis schema fields.
    4. AuthorAgent.synthesize intercepts Turn-1 code generation when zero tools were called
       and enforces Phase 1 research without executing auto-pilot.
    """
    from lox.author.agent import AuthorAgent
    from lox.author.prompts import build_system_prompt, build_user_prompt

    # 1. System Prompt Verification
    sys_prompt = build_system_prompt(str(tmp_path / "test.duckdb"))
    assert "Empirical Differential Diagnostics" in sys_prompt
    assert "expert_ticks" in sys_prompt
    assert "champion_runs" in sys_prompt
    assert "SELECT 'champion_failures' AS source" in sys_prompt
    assert "Pre-Submission Invariant Checklist" in sys_prompt
    assert "Generator Invariant" in sys_prompt
    assert "Passability" in sys_prompt
    assert 'query_policy_docs("actions")' in sys_prompt

    # Verify query_policy_docs("actions") tool route
    tools = AuthorToolRegistry(db_path=str(tmp_path / "test.duckdb"))
    action_docs, summary = tools.query_policy_docs("actions")
    assert "Action.SEARCH" in action_docs
    assert "Action.KICK" in action_docs
    assert "Action.PRAY" in action_docs
    assert summary == "action docs"

    # 2. User Prompt Verification across modes
    for mode in ("FUNC_MOD_ADD", "STRUCTURAL_REFACTOR", "FULL_WRITE"):
        user_prompt = build_user_prompt(
            mode=mode,
            generation=1,
            current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT\n",
            macros_docs="",
            batch_digest="Batch summary: 75% died to starvation.",
        )
        assert (
            "The Binding Constraint Rule" in user_prompt
            or "binding_constraint" in user_prompt
        )
        assert "binding_constraint:" in user_prompt
        assert "verified_c_rule:" in user_prompt
        assert "state_persistence:" in user_prompt
        assert "mechanism:" in user_prompt
        assert "target_functions:" in user_prompt

    # 3. Hypothesis Extraction Schema Enforcement
    sample_llm_output = """
```yaml
mode: FUNC_MOD_ADD
target_functions: [explore_step]
observation_from_db: "70% died from wall stalling"
mechanism: "Check get_passable_mask"
```
```python
def explore_step(self, state):
    state = yield Action.SEARCH
    return state
```
"""
    hyp, code = AuthorAgent.extract_yaml_and_python(sample_llm_output)
    assert hyp["target_functions"] == ["explore_step"]
    assert hyp["binding_constraint"] == "70% died from wall stalling"
    assert "verified_c_rule" in hyp
    assert hyp["mechanism"] == "Check get_passable_mask"
    assert "state_persistence" in hyp
    assert "def explore_step" in code

    # 4. Turn-1 Mandatory Investigation Gating in AuthorAgent.synthesize
    agent = AuthorAgent(provider="openrouter", db_path=str(tmp_path / "test.duckdb"))
    turn_counter = 0
    received_user_prompts = []

    def mock_call_llm_turn1_gate(messages, allow_tools=True):
        nonlocal turn_counter
        turn_counter += 1
        for m in messages:
            if m.get("role") == "user":
                received_user_prompts.append(m.get("content", ""))

        if turn_counter == 1:
            # Emits code immediately on Turn 1 without calling query_db or query_wiki
            return {
                "role": "assistant",
                "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
binding_constraint: "Stalling"
verified_c_rule: "C door rule"
mechanism: "Kick doors"
state_persistence: "self.tried"
```
```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.WAIT
```""",
                "tool_calls": [],
            }
        else:
            # Turn 2: returns a response after gating
            return {
                "role": "assistant",
                "content": """```yaml
mode: FUNC_MOD_ADD
target_functions: [run]
binding_constraint: "Stalling"
verified_c_rule: "C door rule"
mechanism: "Kick doors"
state_persistence: "self.tried"
```
```python
class Policy:
    def run(self, state):
        while True:
            state = yield Action.SEARCH
```""",
                "tool_calls": [],
            }

    monkeypatch.setattr(agent, "_call_llm_api", mock_call_llm_turn1_gate)

    proposal = agent.synthesize(
        mode="FUNC_MOD_ADD",
        generation=1,
        current_policy="class Policy:\n    def run(self, state):\n        state = yield Action.WAIT\n",
        max_tool_turns=3,
    )

    # Verify that Turn 1 Phase 1 enforcement intercepted Turn 1 and auto-pilot was blocked on Turn 1
    assert any("🛑 PHASE 1 ENFORCEMENT" in p for p in received_user_prompts)
    assert turn_counter >= 2
    assert "class Policy" in proposal.full_policy_code


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
        mode="FUNC_MOD_ADD",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "UNREACHABLE METHOD IN POLICY" in unreach_p
    assert "The updated caller method" in unreach_p
    assert "def run(self, state):" in unreach_p

    syntax_p = agent.build_repair_prompt(
        status="COMPILE_ERROR",
        diag="SyntaxError: invalid syntax",
        mode="FUNC_MOD_ADD",
        repair_idx=1,
        max_repair_turns=5,
    )
    assert "PYTHON SYNTAX OR INDENTATION ERROR" in syntax_p
    assert "4 spaces per indentation level" in syntax_p


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
        mode="FUNC_MOD_ADD",
        generation=1,
        current_policy="class Policy: pass",
        macros_docs="### Macros: None",
        skills={"combat": combat_code},
    )

    assert "Hierarchical Skill Registry" in prompt
    assert "# Module: combat" in prompt
    assert "def attack(state):" in prompt
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
        "def choose_move(state, bb):\n"
        "    return Action.WAIT\n"
    )
    partial_patch = (
        "# PATCH\n"
        "def choose_move(state, bb):\n"
        "    dx, dy = _delta_of((0, 0), (1, 0))\n"
        "    return Action.SEARCH if dx == 1 and dy == 0 else Action.WAIT\n"
    )
    merged = merge_skill_module_update(existing_shared, partial_patch)
    assert "def _delta_of(a, b):" in merged
    assert "return Action.SEARCH" in merged

    # 3. dry_run_policy unpacks XML bundle (<skill_code> + <policy_code>) and keeps _delta_of
    base_policy = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n"
        "from skills.shared import _delta_of, choose_move\n\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(choose_move, name='mv')], name='root')\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(root=build_tree(self.blackboard), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        _ = _delta_of((0, 0), (0, 1))\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
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
        "def choose_move(state, bb):\n"
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
        "def choose_move(state, bb):\n"
        "    return Action.SEARCH\n"
        "</skill_code>\n"
        "<policy_code>\n"
        "def build_tree(bb):\n"
        "    return PrioritySelector([ActionLeaf(choose_move, name='mv2')], name='root')\n"
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
    assert 'ActionLeaf(fn, name="...", min_commit=0)' in sys_swarm

    base_pol = (
        "from lox.env.actions import Action\n"
        "from lox.lang.bt import Blackboard, ActionLeaf, PrioritySelector, BehaviorTree\n\n"
        "def leaf_wait(state, bb):\n"
        "    return Action.WAIT\n\n"
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.blackboard = Blackboard()\n"
        "        self.tree = BehaviorTree(PrioritySelector([ActionLeaf(leaf_wait, name='w')]), blackboard=self.blackboard)\n"
        "    def step(self, state):\n"
        "        return self.tree.tick(state) or Action.WAIT\n"
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
