"""
S0 tests: the prompt-program A/B harness (AGENT_PLAN §1.6, §2-S0 exit gate).

Runs scripts/run_prompt_ab.run_ab end-to-end with the mock provider over
isolated workdirs (never touches the live program or the real ledger).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from lox.policy.program import PolicyProgram
from lox.policy.prompts import list_prompt_versions


def _ab_args(**overrides):
    """Argparse-equivalent namespace for run_prompt_ab.run_ab."""
    from argparse import Namespace
    base = dict(
        candidates="v1,v5",
        author_mode="agentic",
        sessions=1,
        provider="mock",
        model=None,
        program_path="data/policy_program.json",
        db_path="data/lox_telemetry.duckdb",
        ledger_path="data/revision_ledger.jsonl",
        max_turns=6,
        downstream=False,
        downstream_episodes=2,
        downstream_steps=500,
        workdir=None,
        out=None,
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.mark.skipif(not os.path.exists("data/policy_program.json"),
                    reason="live program missing")
def test_prompt_ab_two_candidates_agentic(tmp_path):
    from run_prompt_ab import run_ab
    out_path = str(tmp_path / "ab.json")
    rc = asyncio.run(run_ab(_ab_args(
        workdir=str(tmp_path / "work"), out=out_path)))
    assert rc == 0
    with open(out_path, encoding="utf-8") as f:
        results = json.load(f)
    assert set(results["arms"]) == {"agentic:v1", "agentic:v5"}
    for arm in results["arms"].values():
        assert 0.0 <= arm["accept_rate"] <= 1.0
        assert "reject_taxonomy" in arm and "sessions" in arm
        assert arm["mean_tool_calls"] >= 1  # the agentic arm gathers evidence


def test_prompt_ab_reject_taxonomy_recorded(tmp_path):
    """A candidate that produces an invalid diff records its validator code."""
    from lox.deliberative.providers.mock_provider import MockProvider
    import run_prompt_ab as ab_mod

    out_path = str(tmp_path / "ab.json")
    provider = MockProvider(canned_response=(
        '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) '
        '(reason "mock: unknown param probe"))\n'
        '(set policy_params.survival.no_such_param 0.5)'))
    orig_get_provider = ab_mod.get_provider
    ab_mod.get_provider = lambda *a, **k: provider
    try:
        rc = asyncio.run(ab_mod.run_ab(
            _ab_args(candidates="v1", workdir=str(tmp_path / "work"), out=out_path)))
    finally:
        ab_mod.get_provider = orig_get_provider
    assert rc == 0
    results = json.load(open(out_path))
    assert results["arms"]["agentic:v1"]["accept_rate"] == 0.0
    assert results["arms"]["agentic:v1"]["reject_taxonomy"] == {"ERR_BOUNDS": 1}


def test_prompt_ab_bundle_arm_has_no_tool_loop(tmp_path):
    from run_prompt_ab import run_ab
    out_path = str(tmp_path / "ab.json")
    asyncio.run(run_ab(_ab_args(
        candidates="v1", author_mode="bundle",
        workdir=str(tmp_path / "work"), out=out_path)))
    results = json.load(open(out_path))
    arm = results["arms"]["bundle:v1"]
    assert arm["mean_tool_calls"] == 0
    assert arm["sessions"][0]["program_path"] is not None


def test_prompt_ab_candidates_default_to_all_versions():
    versions = list_prompt_versions()
    assert len(versions) >= 5  # the S0 seed campaign (v1..v5)
    assert all(v.startswith("v") for v in versions)


def test_prompt_ab_isolated_program_copies(tmp_path):
    """Sessions run from the SAME parent program and never mutate the live file."""
    from run_prompt_ab import run_ab
    live_before = PolicyProgram.load().version
    workdir = str(tmp_path / "work")
    asyncio.run(run_ab(_ab_args(candidates="v1", sessions=2,
                                workdir=workdir,
                                out=str(tmp_path / "ab.json"))))
    assert PolicyProgram.load().version == live_before  # untouched
    # each session got its own program copy + ledger
    session_dirs = sorted(os.listdir(os.path.join(workdir, "agentic_v1")))
    assert session_dirs == ["s0", "s1"]
