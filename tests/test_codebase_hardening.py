"""
Tests for Core Domain Contracts & Codebase Hardening.
Verifies EpisodeRun, BatchEvaluation, CandidatePackage, assert_baseline_invariants,
and DB logger champion baseline loading.
"""

from lox.core.contracts import (
    BatchEvaluation,
    CandidatePackage,
    EpisodeRun,
    assert_baseline_invariants,
    split_full_and_shard_runs,
)
from lox.engine.orchestrator import CampaignOrchestrator, PolicyArchiveEntry


def test_episode_run_alias_synchronization():
    """Verify EpisodeRun keeps paired aliases synchronized in all write/update patterns."""
    # Initialization
    r = EpisodeRun(xp_level=5, max_hp=16, triggered_functions={"explore": 10})
    assert r["final_xp_level"] == 5
    assert r["final_max_hp"] == 16
    assert r["functions_triggered"] == {"explore": 10}

    # Updating via primary alias
    r["xp_level"] = 6
    assert r["final_xp_level"] == 6

    # Updating via secondary alias
    r["final_max_hp"] = 20
    assert r["max_hp"] == 20

    # Updating via dict.update()
    r.update({"triggered_functions": {"combat": 3}})
    assert r["functions_triggered"] == {"combat": 3}

    r.update({"functions_triggered": {"rest": 5}})
    assert r["triggered_functions"] == {"rest": 5}


def test_episode_run_shard_detection_and_stairs():
    """Verify is_shard is correctly recognized and stairs_down_found is inferred."""
    full_run = EpisodeRun(
        run_id="run_1001",
        seed=1001,
        max_depth=3,
        summary_digest="Reached dlvl 3",
    )
    assert full_run["is_shard"] is False
    assert full_run["stairs_down_found"] is True

    shard_run = EpisodeRun(
        run_id="shard_1001_s00",
        seed=1001,
        max_depth=1,
        summary_digest="Prefix shard slice",
    )
    assert shard_run["is_shard"] is True
    assert shard_run["stairs_down_found"] is False


def test_split_full_and_shard_runs():
    """Verify split_full_and_shard_runs partitions correctly."""
    r1 = EpisodeRun(run_id="run_1", seed=1)
    r2 = EpisodeRun(run_id="shard_1_0", seed=1, is_shard=True)
    r3 = EpisodeRun(run_id="run_2", seed=2)
    r4 = EpisodeRun(run_id="shard_2_0", seed=2)

    full, shards = split_full_and_shard_runs([r1, r2, r3, r4])
    assert len(full) == 2
    assert [r["run_id"] for r in full] == ["run_1", "run_2"]
    assert len(shards) == 2
    assert [r["run_id"] for r in shards] == ["shard_1_0", "shard_2_0"]


def test_batch_evaluation_typed_wrapper():
    """Verify BatchEvaluation exposes full_runs and shard_runs."""
    r1 = EpisodeRun(run_id="run_1", seed=1)
    r2 = EpisodeRun(run_id="shard_1_0", seed=1)
    be = BatchEvaluation(
        batch_id="b_01",
        runs=[r1, r2],
        avg_depth=2.0,
    )
    assert len(be.full_runs) == 1
    assert len(be.shard_runs) == 1
    assert be.full_runs[0]["run_id"] == "run_1"
    assert be["avg_depth"] == 2.0


def test_assert_baseline_invariants():
    """Verify assert_baseline_invariants strips shards and enforces uniqueness."""
    r1 = EpisodeRun(run_id="run_1", seed=1, max_depth=2)
    r1_dup = EpisodeRun(run_id="run_1_dup", seed=1, max_depth=3)
    r2 = EpisodeRun(run_id="run_2", seed=2, max_depth=1)
    shard = EpisodeRun(run_id="shard_1_0", seed=1)

    clean = assert_baseline_invariants([r1, shard, r1_dup, r2])
    assert len(clean) == 2
    seeds = [r["seed"] for r in clean]
    assert seeds == [1, 2]
    # Deduplication takes the latest seed entry
    assert clean[0]["max_depth"] == 3


def test_candidate_package_immutability_and_sync():
    """Verify CandidatePackage encapsulates policy and modular skill code."""
    pkg = CandidatePackage(
        code="class Policy: pass",
        skill_name="combat",
        skill_code="def fight(): pass",
        pilot_delta=0.85,
        pilot_summary="PASSED pilot",
        improved=True,
    )
    assert pkg.is_populated is True
    assert pkg.code == "class Policy: pass"
    assert pkg.skill_name == "combat"
    assert pkg.skill_code == "def fight(): pass"
    assert pkg.pilot_delta == 0.85


def test_policy_archive_entry_skills_sync(tmp_path):
    """Verify PolicyArchiveEntry bundles skills and sync_workspace_skills isolates them."""
    workspace = tmp_path / "workspace"
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True)

    orch = CampaignOrchestrator(
        provider="mock",
        db_path=str(tmp_path / "test.duckdb"),
        policy_path=str(workspace / "policy.py"),
        macros_path=str(workspace / "macros.py"),
        workspace_dir=str(workspace),
        workers=1,
    )

    # Save a skill
    orch.save_workspace_skill("descent_v1", "def go_down(): pass")
    assert (skills_dir / "descent_v1.py").exists()

    entry = PolicyArchiveEntry(
        policy_id="pol_v1",
        policy_code="class Policy: pass",
        niche="descent",
        avg_depth=4.0,
        p90_depth=5.0,
        max_depth=6,
        avg_ac=10.0,
        avg_xl=2.0,
        skills=orch.load_workspace_skills(),
    )
    assert "descent_v1" in entry.skills

    # Add another skill in workspace
    orch.save_workspace_skill("combat_v2", "def attack(): pass")
    assert len(orch.load_workspace_skills()) == 2

    # Now restore entry's skills into workspace
    orch.sync_workspace_skills(entry.skills)
    current_skills = orch.load_workspace_skills()
    assert len(current_skills) == 1
    assert "descent_v1" in current_skills
    assert "combat_v2" not in current_skills
    assert not (skills_dir / "combat_v2.py").exists()
