"""
Per-candidate evidence tagging: run ↔ candidate lineage registry + candidate-scoped
report bundles (pre-S2 plumbing).
"""
from __future__ import annotations

import duckdb
import pytest

from lox.evolution.lineage import RunLineage
from lox.policy.report import build_report_bundle

EPISODES = ("episode_id VARCHAR, run_id VARCHAR, total_turns BIGINT, max_depth INTEGER, "
            "final_score BIGINT, death_message VARCHAR, death_category VARCHAR")


@pytest.fixture
def tagged_db(tmp_path):
    """Two candidate batches (each 2 worker runs) + an untagged run."""
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute(f"CREATE TABLE episodes ({EPISODES})")
    rows = [
        ("e1", "POPA_g0_c1_w0", 100, 1, 10, "", "none"),
        ("e2", "POPA_g0_c1_w1", 200, 2, 30, "The rat bites!", "none"),
        ("e3", "POPA_g0_c2_w0", 300, 5, 90, "", "none"),
        ("e4", "untaggedX_w0", 50, 1, 1, "", "none"),
    ]
    for r in rows:
        con.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?)", r)
    con.close()
    rg = RunLineage(db)
    rg.register(run_id="POPA_g0_c1", domain="nethack", program_version=7,
                population_id="POPA", candidate_id="c1", generation=0, parent_id="c0")
    rg.register(run_id="POPA_g0_c2", domain="nethack", program_version=7,
                population_id="POPA", candidate_id="c2", generation=0, parent_id="c0")
    return db


class TestRunLineage:
    def test_register_lookup_and_worker_strip(self, tagged_db):
        rg = RunLineage(tagged_db)
        lin = rg.lookup("POPA_g0_c1")
        assert lin["candidate_id"] == "c1" and lin["population_id"] == "POPA"
        assert lin["generation"] == 0 and lin["parent_id"] == "c0"
        # a worker run id resolves to its batch base
        assert rg.lookup("POPA_g0_c1_w7")["run_id"] == "POPA_g0_c1"

    def test_lookup_unknown_is_none(self, tagged_db):
        assert RunLineage(tagged_db).lookup("nope") is None

    def test_runs_for_candidate(self, tagged_db):
        rg = RunLineage(tagged_db)
        assert rg.runs_for_candidate("c1") == ["POPA_g0_c1"]
        assert rg.runs_for_candidate("c1", population_id="POPA") == ["POPA_g0_c1"]
        assert rg.runs_for_candidate("c1", population_id="OTHER") == []

    def test_candidates_listing(self, tagged_db):
        cands = RunLineage(tagged_db).candidates("POPA")
        assert {c["candidate_id"] for c in cands} == {"c1", "c2"}

    def test_idempotent_register(self, tagged_db):
        rg = RunLineage(tagged_db)
        rg.register(run_id="POPA_g0_c1", candidate_id="c1", population_id="POPA",
                    generation=1)
        assert rg.lookup("POPA_g0_c1")["generation"] == 1
        assert len(rg.candidates("POPA")) == 2   # upsert, not duplicate


class TestCandidateScopedBundle:
    def test_candidate_scope_and_lineage(self, tagged_db):
        b = build_report_bundle(tagged_db, candidate_id="c1")
        assert b["scope"]["source"] == "candidate"
        assert b["scope"]["run_id"] == "POPA_g0_c1"
        assert b["scope"]["lineage"]["candidate_id"] == "c1"
        assert b["scope"]["lineage"]["population_id"] == "POPA"
        assert b["batch"]["episodes"] == 2        # only c1's worker runs

    def test_other_candidate_is_isolated(self, tagged_db):
        b = build_report_bundle(tagged_db, candidate_id="c2")
        assert b["batch"]["episodes"] == 1
        assert b["scope"]["lineage"]["candidate_id"] == "c2"

    def test_unknown_candidate_degrades(self, tagged_db):
        b = build_report_bundle(tagged_db, candidate_id="ghost")
        assert b["scope"]["source"] == "candidate_missing"
        assert b["batch"]["episodes"] == 0

    def test_untagged_latest_has_no_lineage(self, tagged_db):
        # latest run (untaggedX) resolves normally, no lineage key
        b = build_report_bundle(tagged_db)
        assert b["scope"]["run_id"] == "untaggedX"
        assert "lineage" not in b["scope"]