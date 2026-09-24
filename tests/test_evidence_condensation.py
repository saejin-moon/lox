"""
Evidence-audit + condensation tests (S1.5).

Guards the three evidence-quality properties:
  1. SCOPE — aggregates are scoped to ONE run; previous run carried for deltas.
  2. CONDENSATION — episode digests surface systemic pathology (stall/action mix)
     that a last-N-tick window cannot; trajectory slices are run-length encoded.
  3. NORMALIZATION — death taxonomy is low-cardinality (category-first, first
     sentence, melee bucketing), not 486 raw strings.
Plus token-footprint guards (compact manifest, bounded transcript).
"""
from __future__ import annotations

import os

import duckdb
import pytest

from lox.policy.author_tools import read_trajectory, MAX_TOOL_OUTPUT_CHARS
from lox.policy.author_agent import _trim_transcript, MAX_TRANSCRIPT_CHARS
from lox.policy.manifest import build_manifest
from lox.policy.program import PolicyProgram
from lox.policy.report import (
    _normalize_cause, build_episode_digests, build_report_bundle,
    build_trajectory_slices,
)

EPISODES = ("episode_id VARCHAR, run_id VARCHAR, total_turns BIGINT, max_depth INTEGER, "
            "final_score BIGINT, death_message VARCHAR, death_category VARCHAR, "
            "final_hp INTEGER")
TICKS = ("episode_id VARCHAR, step BIGINT, turn BIGINT, depth INTEGER, hp INTEGER, "
         "max_hp INTEGER, ac INTEGER, hunger_state INTEGER, action_name VARCHAR, "
         "x INTEGER, y INTEGER, condition_bits BIGINT")
GOALS = ("ts DOUBLE, run_id VARCHAR, episode_id VARCHAR, depth INTEGER, dnum INTEGER, "
         "turn INTEGER, goal VARCHAR, event VARCHAR, steps_in_goal INTEGER")
EVENTS = ("run_id VARCHAR, episode_id VARCHAR, step BIGINT, turn BIGINT, depth INTEGER, "
          "kind VARCHAR, code VARCHAR, detail VARCHAR")


@pytest.fixture
def stall_db(tmp_path):
    """A run whose dominant episode is a 1000-turn SEARCH stall at depth 1 (the
    pathology the old last-N-tick view hides), plus a second run for delta tests."""
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute(f"CREATE TABLE episodes ({EPISODES})")
    con.execute(f"CREATE TABLE ticks ({TICKS})")
    con.execute(f"CREATE TABLE goal_events ({GOALS})")
    con.execute(f"CREATE TABLE events ({EVENTS})")
    con.execute("INSERT INTO episodes VALUES ('ep_stall','runA',1000,1,10,'','none',16)")
    con.execute("INSERT INTO episodes VALUES ('ep_dead','runA',100,3,50,"
                "'The giant rat bites!  The sewer rat hits!','none',0)")
    con.execute("INSERT INTO episodes VALUES ('ep_old','runB',200,5,900,'','none',20)")
    con.execute("""INSERT INTO ticks
        SELECT 'ep_stall', i, i + 1, 1, 16, 16, 9, 1, 'SEARCH', 10, 10, 0 FROM range(1000) t(i)""")
    con.execute("""INSERT INTO ticks
        SELECT 'ep_dead', i, i + 1, 3,
               CASE WHEN i < 90 THEN 16 ELSE GREATEST(0, 16 - (i - 90) * 2) END,
               16, 9, 1, 'STEP', 10, 10, 0 FROM range(100) t(i)""")
    con.execute("INSERT INTO goal_events VALUES (1,'runA','ep_stall',1,0,1,'explore_floor','activated',1)")
    con.execute("INSERT INTO goal_events VALUES (2,'runA','ep_stall',1,0,900,'descend','skipped_budget',500)")
    con.execute("INSERT INTO events VALUES ('runA','ep_stall',1,1,1,'message','stair','You descend.')")
    con.execute("INSERT INTO events VALUES ('runA','ep_stall',2,900,1,'anomaly','CRITICAL_HP_FLOOR','HP low')")
    con.execute("INSERT INTO events VALUES ('runA','ep_dead',3,99,3,'anomaly','BURST_DAMAGE','-6 HP')")
    con.close()
    return db


class TestScope:
    def test_bundle_scoped_to_latest_run_with_prev(self, stall_db):
        b = build_report_bundle(stall_db)
        assert b["scope"]["source"] == "latest_run"
        assert b["scope"]["run_id"] == "runB"          # rowid order → last inserted
        assert b["scope"]["episode_count"] == 1
        assert b["batch"]["episodes"] == 1
        assert b["prev_run"]["run_id"] == "runA"       # delta attribution
        assert b["prev_run"]["episodes"] == 2

    def test_explicit_run_id_respected(self, stall_db):
        b = build_report_bundle(stall_db, run_id="runA")
        assert b["scope"]["run_id"] == "runA" and b["scope"]["source"] == "explicit"
        assert b["batch"]["episodes"] == 2

    def test_goal_stats_scoped_to_run(self, stall_db):
        b = build_report_bundle(stall_db, run_id="runA")
        goals = {g["goal"]: g for g in b["goal_stats"]}
        assert "explore_floor" in goals and "descend" in goals
        assert goals["descend"]["skips_budget"] == 1
        # runB has no goal events → scoping must not leak runA's
        assert build_report_bundle(stall_db, run_id="runB")["goal_stats"] == []

    def test_stall_census_has_real_steps(self, stall_db):
        b = build_report_bundle(stall_db, run_id="runA")
        assert b["stall_census"] and b["stall_census"][0]["median_stall_turns"] == 500


class TestDistribution:
    def test_shape_and_action_mix(self, stall_db):
        from lox.policy.report import build_run_distribution
        d = build_run_distribution(stall_db, run_id="runA", d1_stall_turns=500)
        assert d["episodes"] == 2
        assert d["depth_histogram"] == {"1": 1, "2-3": 1, "4-6": 0, "7-9": 0, "10+": 0}
        assert d["never_left_d1"] == 1
        assert d["long_d1_stalls"] == 1          # ep_stall: 1000 turns at d1 >= 500
        assert d["action_mix"][0]["action"] == "SEARCH"   # 1000 of 1100 ticks
        assert d["action_mix"][0]["pct"] >= 90

    def test_event_census(self, stall_db):
        from lox.policy.report import build_run_distribution
        d = build_run_distribution(stall_db, run_id="runA")
        anoms = {a["code"]: a["count"] for a in d["anomalies"]}
        assert anoms == {"CRITICAL_HP_FLOOR": 1, "BURST_DAMAGE": 1}
        msgs = {m["class"]: m["count"] for m in d["message_classes"]}
        assert msgs == {"stair": 1}

    def test_bundle_carries_distribution(self, stall_db):
        b = build_report_bundle(stall_db, run_id="runA")
        assert b["distribution"]["never_left_d1"] == 1
        assert b["distribution"]["action_mix"]

    def test_digest_and_slice_include_events(self, stall_db):
        dg = build_episode_digests(stall_db, run_id="runA", n=2)
        joined = "\n".join(dg)
        assert "anomalies: CRITICAL_HP_FLOOR×1" in joined
        assert "message classes: stair×1" in joined
        sl = build_trajectory_slices(stall_db, run_id="runA")
        assert any("BURST_DAMAGE" in s["slice"] for s in sl)

    def test_missing_events_table_degrades(self, tmp_path):
        import duckdb as _d
        from lox.policy.report import build_run_distribution
        db = str(tmp_path / "noevents.duckdb")
        con = _d.connect(db)
        con.execute(f"CREATE TABLE episodes ({EPISODES})")
        con.execute("INSERT INTO episodes VALUES ('e','r',10,1,0,'','none',0)")
        con.close()
        d = build_run_distribution(db, run_id="r")
        assert d["episodes"] == 1 and d["anomalies"] == [] and d["action_mix"] == []


class TestNormalization:
    def test_category_preferred(self):
        assert _normalize_cause("The jackal bites!", "starvation") == "starvation"

    def test_first_sentence(self):
        out = _normalize_cause("You faint from lack of food.  The kobold hits!", None)
        assert "faint from lack of food" in out
        assert "kobold" not in out

    def test_melee_bucketed(self):
        assert _normalize_cause("The giant rat bites!  The sewer rat hits!", None) == "melee (bites)"
        assert _normalize_cause("The kobold zombie hits!", None) == "melee (hits)"

    def test_survived(self):
        assert _normalize_cause("Survived", None) == "survived"
        assert _normalize_cause("", None) == "survived"

    def test_taxonomy_low_cardinality(self, stall_db):
        b = build_report_bundle(stall_db, run_id="runA")
        causes = [t["cause"] for t in b["death_taxonomy"]]
        assert causes == ["melee (bites)"]
        assert b["death_taxonomy"][0]["count"] == 1


class TestCondensation:
    def test_digest_surfaces_search_stall(self, stall_db):
        digests = build_episode_digests(stall_db, run_id="runA", n=4)
        assert digests
        joined = "\n".join(digests)
        # the pathology: ~100% SEARCH, huge depth-gain gap — impossible in a last-120 view
        assert "SEARCH" in joined
        assert "longest depth-gain gap" in joined
        stall = next(d for d in digests if "ep_stall" in d)
        assert "95%" in stall or "99%" in stall or "100%" in stall
        assert "explore_floor" in stall and "descend" in stall

    def test_trajectory_slice_is_rle_not_stride(self, stall_db):
        slices = build_trajectory_slices(stall_db, run_id="runA")
        assert slices and slices[0]["cause"] == "melee (bites)"
        # RLE lines carry run counts; no stride/downsample marker
        assert "×" in slices[0]["slice"]
        assert "stride" not in slices[0]["slice"]

    def test_read_trajectory_rle(self, stall_db):
        out = read_trajectory("ep_stall", 0, 0, db_path=stall_db)
        assert "SEARCH" in out and "×" in out
        assert "stride" not in out and "downsampled" not in out

    def test_read_trajectory_large_window_is_bounded(self, stall_db):
        out = read_trajectory("ep_stall", 0, 0, db_path=stall_db)
        assert len(out) <= MAX_TOOL_OUTPUT_CHARS + 200  # MAX_TOOL_OUTPUT_CHARS + truncation note


class TestTokenFootprint:
    def test_compact_manifest_is_much_smaller(self):
        prog = PolicyProgram.load()
        man = build_manifest(prog.version, prog.domain,
                             live_macros={m["name"]: m["body"] for m in prog.macros})
        full, compact = man.render(), man.render(compact=True)
        assert len(compact) < len(full) * 0.7          # measured ~45% smaller
        # the domain line must be parseable as a domain, not "nethack v6" (a live
        # session copied the decorated form into the diff header → ERR_HEADER)
        assert f"domain: {man.domain}" in compact
        assert f"{man.domain} v{man.version}" not in compact
        # compact still carries every symbol name (validity preserved)
        for name in man.predicates:
            assert name in compact
        for path in man.param_leaves:
            assert path.split(".")[-1] in compact

    def test_transcript_trim_bounds_growth(self):
        big = "x" * (MAX_TRANSCRIPT_CHARS * 3)
        trimmed = _trim_transcript(big)
        assert len(trimmed) <= MAX_TRANSCRIPT_CHARS + 200
        assert "elided" in trimmed

    def test_transcript_under_budget_untouched(self):
        assert _trim_transcript("small") == "small"

    def test_bundle_json_compact_fits_author_budget(self, stall_db):
        import json
        b = build_report_bundle(stall_db, run_id="runA")
        assert len(json.dumps(b, separators=(",", ":"))) < 14000