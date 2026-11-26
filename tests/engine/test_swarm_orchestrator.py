"""Domain-scoped tests for LOX."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster
from lox.author.swarm_peer import PeerCandidate
from lox.author.tools import AuthorToolRegistry
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.schema import connect_db, init_db
from lox.db.theories import TheoryDocumentStore
from lox.lang.validator import (
    PolicyValidationError,
    validate_cross_module_contracts,
)


def test_swarm_campaign_orchestrator_end_to_end(tmp_path):
    workspace = tmp_path / "workspace"
    frontier = tmp_path / "frontier"
    db_path = str(tmp_path / "swarm_test.duckdb")

    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator

    orchestrator = SwarmCampaignOrchestrator(
        db_path=db_path,
        workspace_dir=str(workspace),
        frontier_dir=str(frontier),
        provider="mock",
        num_workers=2,
        turns_per_worker=1,
        seeds_per_batch=2,
    )
    try:
        history = orchestrator.run(generations=1)
        assert len(history) == 1
        assert history[0]["generation"] == 1
        assert history[0]["avg_depth"] >= 1.0

        con = duckdb.connect(db_path, read_only=True)
        batches_cnt = con.execute("SELECT count(*) FROM batches").fetchone()[0]
        champ_runs_cnt = con.execute(
            "SELECT count(*) FROM runs WHERE run_kind = 'champion'"
        ).fetchone()[0]
        peer_runs_cnt = con.execute("SELECT count(*) FROM peer_runs").fetchone()[0]
        all_ticks_cnt = con.execute("SELECT count(*) FROM all_ticks").fetchone()[0]
        assert batches_cnt == 1
        assert champ_runs_cnt == 2
        assert peer_runs_cnt >= 2
        assert all_ticks_cnt > 0
        con.close()
    finally:
        orchestrator.close()


def test_run_swarm_clean_resets_campaign_while_preserving_experts(tmp_path):
    import subprocess
    import sys

    db_path = tmp_path / "clean_test.duckdb"
    workspace = tmp_path / "workspace"
    frontier = tmp_path / "frontier"
    (workspace / "skills").mkdir(parents=True)
    (workspace / "skills" / "legacy.py").write_text("x = 1\n")
    (workspace / "policy.py").write_text("from skills.legacy import x\n")
    (workspace / "macros.py").write_text("def m(): pass\n")
    (workspace / "STATUS.md").write_text("stale status\n")
    frontier.mkdir(parents=True)
    (frontier / "stale_cp.json").write_text("{}")

    store = TheoryDocumentStore(db_path=str(workspace / "theories.lance"))
    store.save_document(
        generation=99,
        agent_role="stale_worker",
        doc_type="THEORY",
        title="Stale Theory",
        content="Should be removed by --clean",
    )

    con = init_db(str(db_path))
    con.execute(
        "INSERT INTO policies (policy_id, generation, mode, policy_code, is_promoted) "
        "VALUES ('pol_stale', 99, 'WRITE', 'stale', TRUE)"
    )
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, candidate_policy_id, episodes_recorded, avg_depth, max_depth) "
        "VALUES ('b_stale', 99, 'WRITE', 'pol_stale', 1, 1.0, 1)"
    )
    con.execute(
        "INSERT INTO expert_runs (run_id, source, role, max_depth, turns) "
        "VALUES ('exp_keep_1', 'nao', 'Val', 20, 5000)"
    )
    con.close()

    repo_root = Path(__file__).resolve().parents[2]
    res = subprocess.run(
        [
            sys.executable,
            "scripts/run_swarm.py",
            "--clean",
            "--provider",
            "mock",
            "--generations",
            "1",
            "--workers",
            "1",
            "--turns-per-worker",
            "1",
            "--seeds",
            "1",
            "--db-path",
            str(db_path),
            "--workspace-dir",
            str(workspace),
            "--frontier-dir",
            str(frontier),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert (workspace / "STATUS.md").exists()
    status_txt = (workspace / "STATUS.md").read_text(encoding="utf-8")
    assert "stale status" not in status_txt
    assert "LOX Campaign Status Dashboard" in status_txt
    assert not (workspace / "skills" / "legacy.py").exists()
    assert not (frontier / "stale_cp.json").exists()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        stale_pol = con.execute(
            "SELECT COUNT(*) FROM policies WHERE policy_id = 'pol_stale'"
        ).fetchone()[0]
        stale_batch = con.execute(
            "SELECT COUNT(*) FROM batches WHERE batch_id = 'b_stale'"
        ).fetchone()[0]
        kept_expert = con.execute(
            "SELECT COUNT(*) FROM expert_runs WHERE run_id = 'exp_keep_1'"
        ).fetchone()[0]
        fresh_batches = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        fresh_policies = con.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        assert stale_pol == 0
        assert stale_batch == 0
        assert kept_expert == 1
        assert fresh_batches == 1
        assert fresh_policies == 1
    finally:
        con.close()


def test_climbing_regime_does_not_throttle_peers_to_two(tmp_path, monkeypatch):
    from unittest.mock import patch

    from lox.author.swarm_deficits import DeficitReport
    from lox.db.schema import init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator

    dbp = str(tmp_path / "climb.duckdb")
    init_db(dbp).close()
    ws = tmp_path / "ws"
    fr = tmp_path / "fr"
    orch = SwarmCampaignOrchestrator(
        db_path=dbp,
        workspace_dir=str(ws),
        frontier_dir=str(fr),
        provider="mock",
        num_workers=4,
        seeds_per_batch=4,
        race_seeds=2,
    )
    seen_peers: list[str] = []

    class _FakePool:
        benchmark = "nethack"
        max_steps = 100

        def evaluate_batch(self, batch_id, priority_seeds=None, **kwargs):
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 100,
                        "turns": 100,
                        "final_ac": 6,
                        "xp_level": 2,
                        "end_reason": "killed",
                    }
                    for s in (priority_seeds or [1])
                ],
            }

        def close(self):
            pass

    orch.eval_pool = _FakePool()

    def _fake_analyze(self, generation, current_policy="", skills=None):
        return DeficitReport(
            generation=generation,
            regime="CLIMBING",
            fail_clusters=[
                DeficitCluster("c1", "k1", "l1", run_count=5, sample_seeds=[100]),
                DeficitCluster("c2", "k2", "l2", run_count=5, sample_seeds=[101]),
            ],
        )

    def _fake_synth(
        self,
        generation,
        current_policy,
        current_skills,
        deficit_report,
        claimed_deficit,
        **kwargs,
    ):
        seen_peers.append(self.peer_id)
        return PeerCandidate(
            peer_id=self.peer_id,
            claimed_deficit_id=claimed_deficit.cluster_id,
            claimed_label=claimed_deficit.label,
            policy_code=current_policy,
            skills=dict(current_skills),
            sandbox_passed=False,
            error="stop",
        )

    try:
        with (
            patch("lox.author.swarm_deficits.DeficitAnalyzer.analyze", _fake_analyze),
            patch(
                "lox.author.swarm_peer.SwarmPeerAgent.run_cycle",
                _fake_synth,
            ),
        ):
            orch.run_generation(6)
    finally:
        orch.close()

    assert len(seen_peers) == 4, seen_peers


def test_swarm_orchestrator_permission_error_fast_fail_and_all_peer_crash_abort(
    tmp_path, monkeypatch
):
    from unittest.mock import patch

    from lox.db.schema import init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator

    db_p = str(tmp_path / "perm.duckdb")
    ws_p = tmp_path / "ws"
    ws_p.mkdir()
    fr_p = str(tmp_path / "frontier")
    init_db(db_p)

    # Seed Gen 1 policy so Gen 2 exercises peer generation path without Gen 1 empty-incumbent fallback
    (ws_p / "policy.py").write_text(
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n",
        encoding="utf-8",
    )

    class _NoRolloutPool:
        def evaluate_batch(self, batch_id="", priority_seeds=None, **kwargs):
            if str(batch_id).startswith("incubate_inc_"):
                return {
                    "batch_id": batch_id,
                    "runs": [
                        {
                            "run_id": f"inc_{s}",
                            "seed": s,
                            "depth": 1,
                            "max_depth": 1,
                            "score": 10,
                            "turns": 50,
                        }
                        for s in (priority_seeds or [10000, 10001])
                    ],
                }
            raise AssertionError(
                f"Zombie tournament/incumbent evaluation `{batch_id}` should not run after peer failure!"
            )

        def close(self):
            pass

    monkeypatch.setattr(
        "lox.engine.swarm_orchestrator.PersistentEvalPool",
        lambda *a, **k: _NoRolloutPool(),
    )

    # 1. PermissionError in any peer aborts campaign immediately and writes campaign_errors.log
    orch = SwarmCampaignOrchestrator(
        db_path=db_p,
        workspace_dir=str(ws_p),
        frontier_dir=fr_p,
        provider="mock",
        model="mock-model",
        num_workers=2,
        seeds_per_batch=4,
    )
    with patch(
        "lox.engine.swarm_orchestrator.SwarmPeerAgent.run_cycle",
        side_effect=PermissionError("HTTP 402 Insufficient credits"),
    ):
        with pytest.raises(PermissionError, match="Insufficient credits"):
            orch.run(generations=3)

    err_log = (ws_p / "campaign_errors.log").read_text(encoding="utf-8")
    assert "PermissionError" in err_log
    con = connect_db(db_p, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM policies").fetchone()[0] == 0
    finally:
        con.close()

    # 2. HTTP 402 string wrapped in RuntimeError also escalates to PermissionError immediately
    orch2 = SwarmCampaignOrchestrator(
        db_path=db_p,
        workspace_dir=str(ws_p),
        frontier_dir=fr_p,
        provider="mock",
        model="mock-model",
        num_workers=2,
        seeds_per_batch=4,
    )
    with patch(
        "lox.engine.swarm_orchestrator.SwarmPeerAgent.run_cycle",
        side_effect=RuntimeError("OpenRouter returned HTTP 402: out of credits"),
    ):
        with pytest.raises(PermissionError):
            orch2.run(generations=2)

    # 3. Non-PermissionError crash across ALL peers aborts generation and breaks campaign after 3 failures
    orch3 = SwarmCampaignOrchestrator(
        db_path=db_p,
        workspace_dir=str(ws_p),
        frontier_dir=fr_p,
        provider="mock",
        model="mock-model",
        num_workers=2,
        seeds_per_batch=4,
    )
    with patch(
        "lox.engine.swarm_orchestrator.SwarmPeerAgent.run_cycle",
        side_effect=ValueError("Transient upstream outage"),
    ) as m_peer:
        hist = orch3.run(generations=6)
    assert hist == []
    assert m_peer.call_count <= 3 * orch3.num_workers


def test_skill_aware_refactor_thresholds_split_merge_and_swarm_refactor_slot(tmp_path):
    import json

    from lox.author.tools import merge_skill_module_update
    from lox.db.schema import init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.lang.modularize import maybe_auto_modularize

    # 1. DeficitAnalyzer flags skill bloat at >=250 / >=300 LOC
    dbp = str(tmp_path / "refactor_thresh.duckdb")
    init_db(dbp)
    thin_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.E\n"
    )
    bloated_skill = "\n".join([f"def h_{i}():\n    return {i}" for i in range(160)])
    rep = DeficitAnalyzer(dbp).analyze(
        generation=16, current_policy=thin_pol, skills={"exploration": bloated_skill}
    )
    assert rep.bloat_warning is True
    assert any("exploration" in note for note in rep.skill_bloat_notes)

    # 3. merge_skill_module_update preserves intentional split introducing a new skills.* import
    split_parent = "from skills.exploration_nav import h_0\ndef h_1():\n    return 1\n"
    merged = merge_skill_module_update(bloated_skill, split_parent)
    assert "from skills.exploration_nav import h_0" in merged
    assert "def h_150" not in merged

    # 4. Option A: maybe_auto_modularize deterministically splits a bloated skill >= 450 LOC
    ws = tmp_path / "ws_ref"
    (ws / "skills").mkdir(parents=True, exist_ok=True)
    skill_pol = (
        "import skills.exploration as exploration\n"
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return exploration.search_step(state) or exploration.explore_step(state) or Action.E\n"
    )
    bloated_real_skill = (
        "from lox.env.actions import Action\n"
        "STEP_CONST = 1\n\n"
        "def _shared_ok(state):\n"
        "    return state.hp >= STEP_CONST\n\n"
        + "\n".join(
            f"def _pad_a_{i}():\n    return {i if i == 0 else f'_pad_a_{i - 1}()'}\n"
            for i in range(120)
        )
        + "\n".join(
            f"def _pad_b_{i}():\n    return {i if i == 0 else f'_pad_b_{i - 1}()'}\n"
            for i in range(120)
        )
        + "def search_step(state):\n"
        "    if _shared_ok(state) and _pad_a_119() >= 0 and state.step_count % 2 == 0:\n"
        "        return Action.SEARCH\n"
        "    return None\n\n"
        "def explore_step(state):\n"
        "    if _shared_ok(state) and _pad_b_119() >= 0:\n"
        "        return Action.E\n"
        "    return None\n"
    )
    changed, _new_pol, new_skills, mod_msg = maybe_auto_modularize(
        skill_pol,
        {"exploration": bloated_real_skill},
        verify_seeds=[1000, 1001],
        verify_steps=40,
    )
    assert changed is True, mod_msg
    assert len(new_skills["exploration"].splitlines()) < 30
    assert len(new_skills) >= 3
    assert "skill/exploration" in mod_msg

    # 5. SwarmCampaignOrchestrator applies Option A bit-identical skill split on disk & DuckDB
    (ws / "policy.py").write_text(skill_pol, encoding="utf-8")
    (ws / "skills" / "exploration.py").write_text(bloated_real_skill, encoding="utf-8")
    orch = SwarmCampaignOrchestrator(
        db_path=dbp,
        workspace_dir=str(ws),
        frontier_dir=str(tmp_path / "fr_ref"),
        provider="mock",
        num_workers=2,
        eval_workers=2,
        seeds_per_batch=2,
    )
    try:
        res = orch.run_generation(16)
        assert res["promoted"] is False
        disk_skills = {
            str(p.relative_to(ws / "skills"))
            .removesuffix(".py")
            .replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in (ws / "skills").rglob("*.py")
            if p.name != "__init__.py"
        }
        assert len(disk_skills) >= 2
        assert "exploration/core" in disk_skills
        assert len(disk_skills["exploration/core"].splitlines()) < len(
            bloated_real_skill.splitlines()
        )
        con = connect_db(dbp, read_only=True)
        try:
            row = con.execute(
                "SELECT skills_json FROM policies WHERE generation=16"
            ).fetchone()
            assert row is not None and len(json.loads(row[0])) == len(disk_skills)
        finally:
            con.close()
    finally:
        orch.close()

    # 6. validate_cross_module_contracts & dry_run_policy accept facade re-exports (e.g. from split skills)

    facade_skills = {
        **new_skills,
        "combat": (
            "from skills.exploration import search_step\n"
            "def _can_fight():\n"
            "    return True\n"
            "def attack_or_search(state):\n"
            "    return search_step(state) if _can_fight() else None\n"
        ),
    }
    validate_cross_module_contracts(skill_pol, facade_skills)

    # Arity check still fires transitively through the facade
    bad_arity_skills = {
        **new_skills,
        "combat": (
            "from skills.exploration import search_step\n"
            "def attack_or_search(state):\n"
            "    return search_step(state, 123)\n"
        ),
    }
    try:
        validate_cross_module_contracts(skill_pol, bad_arity_skills)
        raise AssertionError("Expected transitive arity PolicyValidationError")
    except PolicyValidationError as exc:
        assert "search_step" in str(exc)

    reg = AuthorToolRegistry(db_path=dbp)
    reg.configure_generation(generation=16, mode="SWARM_PEER", skills=facade_skills)
    reg.current_policy_code = skill_pol
    reg.enforce_peer_research_gate = False
    _out_f, st_f = reg.dry_run_policy(
        '<skill_code module="combat">\ndef attack_or_search(state):\n    return search_step(state) if (_can_fight() and state.hp > 0) else None\n</skill_code>',
        seed=42,
        max_turns=20,
    )
    assert "PASSED" in st_f or st_f == "PASSED"


def test_phase1_cli_args_wiki_and_prompts():
    from scripts.run_swarm import parse_args

    args_def = parse_args(["--provider", "openrouter", "--model", "z-ai/glm-5.3-flash"])
    assert args_def.expert_model == "thinkingmachines/inkling-2-small:floor"
    assert args_def.reasoning_effort == "medium"
    assert args_def.seeds == 64
    assert args_def.race_seeds == 32
    assert args_def.scenario_gate == "off"

    args_non_or = parse_args(["--provider", "gemini", "--model", "gemini-2.5-flash"])
    assert args_non_or.expert_model == "gemini-2.5-flash"

    args_custom = parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.3-flash",
            "--expert-model",
            "anthropic/claude-opus-4.5",
            "--routing",
            "exacto",
            "--race-seeds",
            "16",
            "--scenario-gate",
            "strict",
        ]
    )
    assert args_custom.expert_model == "anthropic/claude-opus-4.5"
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
    assert "adjacent_closed_doors" in sys_p
    assert "register_goal" in sys_p
    pol_25 = (
        "import skills.goals.core as goals\n"
        "from lox.env.actions import Action\n"
        + "".join(
            f"goals.register_goal('fn_{i}', priority={i}, trigger=lambda s, w: False, step=lambda s, w: Action.WAIT)\n"
            for i in range(25)
        )
        + "class Policy:\n    def step(self, state): return Action.WAIT\n"
    )
    user_p = build_user_prompt(
        mode="SWARM_PEER",
        generation=3,
        current_policy=pol_25,
        skills={"nav": "def go(): return Action.N\n"},
    )
    assert "fn_24" in user_p  # No silent brs[:20] truncation!


def test_dual_track_1_revision_plus_4_fresh_and_directive_planner(
    tmp_path, monkeypatch
):
    """
    Verifies:
    1. `SwarmMetaAgent.plan_peer_directives` returns orthogonal directives for all Fresh peers
       (and falls back cleanly in mock mode).
    2. When an active stepping stone exists for the current incumbent, `SwarmCampaignOrchestrator`
       spawns 1 Revision (`peer_rev`, starting from stepping stone code) + 4 Fresh (`peer_1..4`)
       = 5 concurrent peers, and retires the stone after 2 revision attempts.
    """
    from unittest.mock import patch

    from lox.author.directives import SwarmDirectivePlanner
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.engine.swarm_referee import _package_fingerprint

    db_p = str(tmp_path / "dual_track.duckdb")
    ws_p = tmp_path / "ws"
    ws_p.mkdir()
    fr_p = str(tmp_path / "frontier")
    init_db(db_p).close()

    inc_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.WAIT\n"
    )
    stone_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    (ws_p / "policy.py").write_text(inc_pol, encoding="utf-8")
    inc_fp = _package_fingerprint(inc_pol, {})

    logger = BatchDatabaseLogger(db_path=db_p)
    logger.save_stepping_stone(
        {
            "stone_id": "stone_g66_peer_2",
            "generation": 66,
            "parent_policy_id": inc_fp,
            "source_peer_id": "peer_2",
            "source_run_kind": "select",
            "source_batch_id": "sel_g66_peer_2",
            "policy_code": stone_pol,
            "skills": {},
            "mechanism": "shop escape",
            "target_functions": ["explore"],
            "wins": 7,
            "losses": 0,
            "iqm_delta": 0.20,
            "winsor_delta": 0.20,
            "wilcoxon_p": 0.008,
            "sign_p": 0.008,
            "reject_reason": "cut_in_select_stage_by_stack",
            "seeds_gained": [101, 102],
            "seeds_lost": [103],
        }
    )

    # 1. Verify Directive Planner mock fallback for Layout A (2R + 2P)
    meta = SwarmDirectivePlanner(provider="mock", model="test-model", db_path=db_p)
    dummy_rep = DeficitReport(
        generation=67,
        incumbent_avg_depth=3.5,
        incumbent_max_depth=6,
        regime="PLATEAU",
        fail_clusters=[
            DeficitCluster("c1", "death", "Starvation", 10, [101], "ev1"),
            DeficitCluster("c2", "loop", "Search loop", 12, [102], "ev2"),
        ],
    )
    dirs = meta.plan_peer_directives(
        generation=67,
        fresh_peer_ids=["reactive_1", "reactive_2", "proactive_1", "proactive_2"],
        deficit_report=dummy_rep,
        active_stepping_stone=logger.load_active_stepping_stone(inc_fp),
        peer_roles={
            "reactive_1": "revision",
            "reactive_2": "reactive",
            "proactive_1": "proactive_capability",
            "proactive_2": "proactive_branch",
        },
    )
    assert [d["peer_id"] for d in dirs] == [
        "reactive_1",
        "reactive_2",
        "proactive_1",
        "proactive_2",
    ]
    assert dirs[1]["exclude_overlap"] and "stepping-stone" in dirs[1]["exclude_overlap"]
    assert dirs[2]["track"] == "proactive_capability"
    assert dirs[3]["track"] == "proactive_branch"

    # 2. Verify SwarmOrchestrator launches Layout A: 2 Reactive + 2 Proactive
    #    where `reactive_1` picks up the active stepping stone (`stone_pol`)!
    class _StubPool:
        benchmark = "nethack"

        def evaluate_batch(self, batch_id="", priority_seeds=None, **kwargs):
            return {
                "batch_id": batch_id,
                "runs": [
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": 2,
                        "max_depth": 2,
                        "score": 50,
                        "turns": 50,
                        "end_reason": "killed",
                        "ticks": [
                            {
                                "turn": 1,
                                "step_idx": 1,
                                "depth": 2,
                                "hp": 10,
                                "max_hp": 10,
                            }
                        ],
                    }
                    for s in (priority_seeds or [1, 2])
                ],
            }

        def close(self):
            pass

    monkeypatch.setattr(
        "lox.engine.swarm_orchestrator.PersistentEvalPool",
        lambda *a, **k: _StubPool(),
    )
    orch = SwarmCampaignOrchestrator(
        db_path=db_p,
        workspace_dir=str(ws_p),
        frontier_dir=fr_p,
        provider="mock",
        model="mock-model",
        num_workers=4,
        seeds_per_batch=4,
    )
    seen_starts: dict[str, tuple[str, bool, str]] = {}

    def _fake_peer_cycle(
        self_peer,
        generation,
        deficit_report,
        claimed_deficit,
        current_policy,
        current_skills,
        **kwargs,
    ):
        seen_starts[self_peer.peer_id] = (
            current_policy,
            kwargs.get("stepping_stone") is not None,
            str(kwargs.get("track", "")),
        )
        return PeerCandidate(
            peer_id=self_peer.peer_id,
            claimed_deficit_id=claimed_deficit.cluster_id,
            claimed_label=claimed_deficit.label,
            policy_code=inc_pol,
            skills={},
            sandbox_passed=False,
            error="noop_test",
            track=str(kwargs.get("track", "reactive")),
        )

    try:
        with (
            patch(
                "lox.author.swarm_deficits.DeficitAnalyzer.analyze",
                lambda *a, **k: dummy_rep,
            ),
            patch(
                "lox.author.swarm_peer.SwarmPeerAgent.run_cycle",
                _fake_peer_cycle,
            ),
        ):
            orch.run_generation(67)
    finally:
        orch.close()

    assert set(seen_starts.keys()) == {
        "reactive_1",
        "reactive_2",
        "proactive_1",
        "proactive_2",
    }
    assert seen_starts["reactive_1"] == (stone_pol, True, "revision")
    assert seen_starts["reactive_2"] == (inc_pol, False, "reactive")
    assert seen_starts["proactive_1"] == (inc_pol, False, "proactive_capability")
    assert seen_starts["proactive_2"] == (inc_pol, False, "proactive_branch")

    # After 1 revision attempt, revision_count incremented to 1 (still active for 1 more try)
    st_after = logger.load_active_stepping_stone(inc_fp, max_revisions=2)
    assert st_after is not None and st_after["revision_count"] == 1


def test_v3_proactive_branch_advancement_4gen_reset_and_graduation(
    tmp_path, monkeypatch
):
    """
    Verifies Component 3:
    1. `proactive_2` advances its sheltered branch on non-catastrophic S_select without polluting Champion.
    2. On Gen 2..3, `proactive_2` continues from the saved `proactive_head` code.
    3. At step 4 without graduating, branch resets cleanly to Champion.
    4. When `proactive_2` wins strict tournament promotion, branch status becomes 'graduated' and workspace updates.
    """
    from unittest.mock import patch

    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate
    from lox.db.logger import BatchDatabaseLogger
    from lox.db.schema import connect_db, init_db
    from lox.engine.swarm_orchestrator import SwarmCampaignOrchestrator
    from lox.engine.swarm_referee import _package_fingerprint

    db_p = str(tmp_path / "pbranch.duckdb")
    ws_p = tmp_path / "ws"
    ws_p.mkdir()
    init_db(db_p).close()

    inc_pol = "from lox.env.actions import Action\nclass Policy:\n    def step(self, state):\n        return Action.WAIT\n"

    def br_step_pol(k: int) -> str:
        return (
            f"from lox.env.actions import Action\n"
            f"class Policy:\n"
            f"    def step(self, state):\n"
            f"        _v = {k}\n"
            f"        return Action.SEARCH\n"
        )

    (ws_p / "policy.py").write_text(inc_pol, encoding="utf-8")
    inc_fp = _package_fingerprint(inc_pol, {})

    dummy_rep = DeficitReport(
        generation=1,
        incumbent_avg_depth=3.0,
        incumbent_max_depth=5,
        regime="PLATEAU",
        fail_clusters=[DeficitCluster("c1", "loop", "Loop", 10, [101], "ev")],
    )

    promote_p2_mode = {"win": False}

    class _BranchPool:
        benchmark = "nethack"

        def evaluate_batch(self, batch_id="", priority_seeds=None, **kwargs):
            s_list = list(priority_seeds or [1, 2, 3, 4])
            is_p2 = "proactive_2" in str(batch_id)
            runs = []
            for idx, s in enumerate(s_list):
                if is_p2 and promote_p2_mode["win"]:
                    d, sc = 5, 1200
                elif is_p2:
                    # Non-catastrophic S_select: 2W/2L around champion depth=3.0 (winsor_delta~0.0, avg_depth=3.0)
                    d = 4 if idx % 2 == 0 else 2
                    sc = 700 if idx % 2 == 0 else 400
                else:
                    d, sc = 3, 500
                runs.append(
                    {
                        "run_id": f"{batch_id}_{s}",
                        "seed": s,
                        "depth": d,
                        "max_depth": d,
                        "score": sc,
                        "turns": 500,
                        "end_reason": "killed",
                    }
                )
            return {"batch_id": batch_id, "runs": runs}

        def close(self):
            pass

    monkeypatch.setattr(
        "lox.engine.swarm_orchestrator.PersistentEvalPool",
        lambda *a, **k: _BranchPool(),
    )
    orch = SwarmCampaignOrchestrator(
        db_path=db_p,
        workspace_dir=str(ws_p),
        frontier_dir=str(tmp_path / "fr"),
        provider="mock",
        model="mock-model",
        num_workers=4,
        seeds_per_batch=8,
    )
    assert orch.referee.max_tournament_peers >= 4

    p2_seen_starts: list[str] = []

    def _fake_peer_cycle(
        self_peer,
        generation,
        deficit_report,
        claimed_deficit,
        current_policy,
        current_skills,
        **kwargs,
    ):
        if self_peer.peer_id == "proactive_2":
            p2_seen_starts.append(current_policy)
            return PeerCandidate(
                peer_id="proactive_2",
                claimed_deficit_id=claimed_deficit.cluster_id,
                claimed_label=claimed_deficit.label,
                hypothesis_yaml={"mechanism": f"branch_step_g{generation}"},
                policy_code=br_step_pol(generation),
                skills={},
                sandbox_passed=True,
                sandbox_metrics={"avg_depth": 3.0},
                track="proactive_branch",
            )
        return PeerCandidate(
            peer_id=self_peer.peer_id,
            claimed_deficit_id=claimed_deficit.cluster_id,
            claimed_label=claimed_deficit.label,
            policy_code=inc_pol,
            skills={},
            sandbox_passed=False,
            error="skip_other",
            track=str(kwargs.get("track", "reactive")),
        )

    logger = BatchDatabaseLogger(db_path=db_p)
    try:
        with (
            patch(
                "lox.author.swarm_deficits.DeficitAnalyzer.analyze",
                lambda *a, **k: dummy_rep,
            ),
            patch(
                "lox.author.swarm_peer.SwarmPeerAgent.run_cycle",
                _fake_peer_cycle,
            ),
        ):
            # Gens 1..3 advance sheltered branch without modifying workspace Champion
            for g in (1, 2, 3):
                r = orch.run_generation(g)
                assert r["promoted"] is False
                assert (ws_p / "policy.py").read_text(encoding="utf-8") == inc_pol
                br = logger.load_active_proactive_branch(inc_fp)
                assert br is not None and br["steps_taken"] == g
            # Gen 2 started from Gen 1's branch head, not champion
            assert "_v = 1" in p2_seen_starts[1]
            assert "_v = 2" in p2_seen_starts[2]

            # Gen 4 hits max_steps=4 -> resets cleanly to Champion (no active branch left)
            r4 = orch.run_generation(4)
            assert r4["promoted"] is False
            assert logger.load_active_proactive_branch(inc_fp) is None

            # Gen 5 starts fresh from Champion, then Gen 6 graduates proactive_2 to Champion!
            r5 = orch.run_generation(5)
            assert r5["promoted"] is False
            assert p2_seen_starts[4] == inc_pol
            br5 = logger.load_active_proactive_branch(inc_fp)
            assert br5 is not None and br5["steps_taken"] == 1

            promote_p2_mode["win"] = True
            r6 = orch.run_generation(6)
            assert r6["promoted"] is True
            assert r6["winner_id"] == "proactive_2"
            assert "_v = 6" in (ws_p / "policy.py").read_text(encoding="utf-8")
            con = connect_db(db_p, read_only=True)
            try:
                st_row = con.execute(
                    "SELECT status FROM proactive_branches WHERE branch_id = ?",
                    [br5["branch_id"]],
                ).fetchone()
                assert st_row is not None and st_row[0] == "graduated"
            finally:
                con.close()
    finally:
        orch.close()


def test_expert_model_always_routes_proactive_2(tmp_path):
    """Verifies `--expert-model` defaults cleanly on OpenRouter and ALWAYS routes `proactive_2` (even at plateau=0)."""
    from unittest.mock import patch

    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.author.swarm_peer import PeerCandidate
    from lox.engine.swarm_orchestrator import (
        DEFAULT_EXPERT_MODEL,
        SwarmCampaignOrchestrator,
    )

    orch = SwarmCampaignOrchestrator(
        db_path=str(tmp_path / "exp_route.duckdb"),
        workspace_dir=str(tmp_path / "ws"),
        frontier_dir=str(tmp_path / "fr"),
        provider="openrouter",
        model="z-ai/glm-5.3-flash:nitro",
        seeds_per_batch=8,
        num_workers=4,
    )
    assert orch.expert_model == DEFAULT_EXPERT_MODEL

    seen_models: dict[str, str] = {}
    rep = DeficitReport(
        generation=1,
        regime="CLIMBING",
        plateau_streak=0,
        bloat_warning=False,
        d_succ_seeds=[11, 12],
        fail_clusters=[
            DeficitCluster(
                cluster_id="c1", kind="early_mortality", label="c1", sample_seeds=[21]
            )
        ],
    )

    def _cap_run_cycle(self_peer, **kw):
        seen_models[self_peer.peer_id] = self_peer.model
        return PeerCandidate(
            peer_id=self_peer.peer_id,
            claimed_deficit_id="c1",
            claimed_label="c1",
            policy_code=str(kw.get("current_policy") or ""),
            skills=dict(kw.get("current_skills") or {}),
            sandbox_passed=False,
            error="stop",
        )

    try:
        with (
            patch.object(orch.deficit_analyzer, "analyze", return_value=rep),
            patch.object(
                orch.directive_planner,
                "plan_peer_directives",
                return_value=[],
            ),
            patch.object(
                orch.eval_pool,
                "evaluate_batch",
                return_value={"runs": [{"seed": 11, "depth": 2, "score": 50}]},
            ),
            patch(
                "lox.engine.swarm_orchestrator.SwarmPeerAgent.run_cycle",
                new=_cap_run_cycle,
            ),
            patch.object(
                orch.referee,
                "evaluate_tournament",
                side_effect=RuntimeError("stop_after_peers"),
            ),
        ):
            try:
                orch.run_generation(1)
            except RuntimeError as exc:
                assert "stop_after_peers" in str(exc)
    finally:
        orch.close()

    assert seen_models.get("proactive_2") == DEFAULT_EXPERT_MODEL
    assert seen_models.get("reactive_1") == "z-ai/glm-5.3-flash:nitro"
    assert seen_models.get("reactive_2") == "z-ai/glm-5.3-flash:nitro"
    assert seen_models.get("proactive_1") == "z-ai/glm-5.3-flash:nitro"


def test_cooldown_rotation_in_directives_and_lifecycle(tmp_path):
    """Verifies Fix 3: cooled-down clusters rotate behind untouched high-signal clusters."""
    from lox.author.directives import SwarmDirectivePlanner
    from lox.author.swarm_deficits import DeficitCluster, DeficitReport
    from lox.engine.lifecycle import build_layout_a_assignments

    cl_stale = DeficitCluster(
        cluster_id="slice_high_search_or_horizon_burn",
        kind="search_or_horizon_loop",
        label="high search",
        run_count=20,
        sample_seeds=[1, 2],
    )
    cl_stalled = DeficitCluster(
        cluster_id="slice_stalled",
        kind="stalled_trajectory",
        label="zero-turn stalls",
        run_count=16,
        sample_seeds=[3, 4],
    )
    cl_eye = DeficitCluster(
        cluster_id="slice_recurring_outcome_1",
        kind="recurring_terminal_state",
        label="recurring phrase `floating eye`",
        run_count=9,
        sample_seeds=[5, 6],
    )
    rep = DeficitReport(
        generation=88,
        regime="PLATEAU",
        fail_clusters=[cl_stale, cl_stalled, cl_eye],
        cooled_down_deficits={"slice_high_search_or_horizon_burn": 3},
    )

    planner = SwarmDirectivePlanner(
        provider="mock", db_path=str(tmp_path / "cd.duckdb")
    )
    dirs = planner.plan_peer_directives(
        generation=88,
        fresh_peer_ids=["reactive_1", "reactive_2", "proactive_1", "proactive_2"],
        deficit_report=rep,
        peer_roles={
            "reactive_1": "reactive",
            "reactive_2": "reactive",
            "proactive_1": "proactive_capability",
            "proactive_2": "proactive_branch",
        },
    )
    by_p = {d["peer_id"]: d for d in dirs}
    assert by_p["reactive_1"]["target_cluster_id"] == "slice_stalled"
    assert by_p["reactive_2"]["target_cluster_id"] == "slice_recurring_outcome_1"

    assigns = build_layout_a_assignments(
        generation=88,
        roster_specs=[("reactive_1", "reactive"), ("reactive_2", "reactive")],
        clusters=rep.fail_clusters,
        orthogonal_clusters=rep.fail_clusters,
        deficit_report=rep,
        current_policy="class Policy: pass",
        initial_skills={},
        probe_seeds=[100, 101],
        planned_directives={},
        active_stone=None,
        active_branch=None,
        plateau_streak=5,
        near_misses=[],
    )
    assigned_cids = [a[1].cluster_id for a in assigns]
    assert assigned_cids == ["slice_stalled", "slice_recurring_outcome_1"]
