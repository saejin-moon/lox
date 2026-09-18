"""R3 tests: revision ledger, report bundle, reviser, and the unattended loop."""
import json
import os
import tempfile

import pytest

from corp.deliberative.providers.mock_provider import MockProvider
from corp.policy.ledger import RevisionLedger
from corp.policy.manifest import build_manifest
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM
from corp.policy.report import build_report_bundle
from corp.policy.reviser import Reviser, build_author_prompt


class TestLedger:
    def test_append_and_tail(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = RevisionLedger(path)
            led.append(led.revision_entry(revision=2, parent_version=1, author_model="m",
                                          provider="mock", accepted=True, reason="x"))
            led.append(led.revision_entry(revision=2, parent_version=1, author_model="m",
                                          provider="mock", accepted=False,
                                          reject_code="ERR_BOUNDS", gate="bounds"))
            entries = led.all()
            assert len(entries) == 2
            assert entries[0]["accepted"] is True
            assert entries[1]["reject_code"] == "ERR_BOUNDS"
            assert led.tail(1)[0]["type"] == "policy_revision"

    def test_acceptance_rate_per_model(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = RevisionLedger(path)
            for acc in (True, True, False):
                led.append(led.revision_entry(revision=2, parent_version=1,
                                              author_model="small-model", provider="mock",
                                              accepted=acc))
            led.append(led.revision_entry(revision=2, parent_version=1,
                                          author_model="frontier", provider="api",
                                          accepted=True))
            assert led.acceptance_rate("small-model") == pytest.approx(2 / 3)
            assert led.acceptance_rate("frontier") == 1.0


class TestReportBundle:
    def test_missing_db_degrades_gracefully(self):
        bundle = build_report_bundle(db_path="/nonexistent/db.duckdb",
                                     ledger_path="/nonexistent/ledger.jsonl")
        assert bundle["domain"] == "nethack"
        assert bundle["batch"]["episodes"] == 0
        assert bundle["death_taxonomy"] == []
        assert bundle["variance_note"].startswith("NetHackChallenge unseedable")

    def test_prev_revisions_from_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = RevisionLedger(path)
            led.append(led.revision_entry(revision=2, parent_version=1,
                                          author_model="m", provider="mock", accepted=True,
                                          delta={"after": 512.0}))
            bundle = build_report_bundle(db_path="/nonexistent", ledger_path=path)
            assert len(bundle["prev_revisions"]) == 1
            assert bundle["prev_revisions"][0]["version"] == 2
            assert bundle["prev_revisions"][0]["delta_score"] == 512.0


class TestReviser:
    @pytest.mark.asyncio
    async def test_propose_diff_from_prose_output(self):
        provider = MockProvider(canned_response=(
            "Sure! Here's my proposal:\n"
            "(revision 2 (parent 1) (author \"m\") (domain nethack) (reason \"ok\"))\n"
            "(set policy_params.survival.rest_below_frac 0.65)\n"
            "Hope this helps."))
        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        reviser = Reviser(provider)
        text = await reviser.propose_diff(program, build_manifest(1, "nethack"), {})
        assert "(set policy_params.survival.rest_below_frac 0.65)" in text

    @pytest.mark.asyncio
    async def test_repair_retry_on_parse_failure(self):
        # First call emits garbage, second (repair) emits a valid diff
        class GarbageThenValidProvider(MockProvider):
            async def generate_text(self, system_prompt, user_prompt, context=None):
                self._text_call_count += 1
                from corp.deliberative.providers.base import LLMResponse
                if self._text_call_count == 1:
                    raw = "I would propose... (revision 2 (parent 1) oops unbalanced"
                else:
                    raw = ('(revision 2 (parent 1) (author "m") (domain nethack) (reason "fixed"))\n'
                           '(set policy_params.survival.rest_below_frac 0.65)')
                return LLMResponse(thinking_content="", parsed_payload=raw, raw_text=raw)

        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        reviser = Reviser(GarbageThenValidProvider())
        text = await reviser.propose_diff(program, build_manifest(1, "nethack"), {})
        assert "(set policy_params.survival.rest_below_frac 0.65)" in text
        assert reviser.last_meta.get("repaired") is True  # exactly one repair retry

    def test_author_prompt_contains_manifest_and_report(self):
        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        manifest = build_manifest(1, "nethack")
        system, user = build_author_prompt(program, manifest, {"batch": {"episodes": 5}},
                                           rag_slices=["The Castle has a drawbridge."])
        assert "policy_params.survival.rest_below_frac" in user
        assert "hp_frac" in user
        assert '"episodes": 5' in user
        assert "drawbridge" in user
        assert "version 2" in user  # task states the target version


class TestRevisionLoop:
    @pytest.mark.asyncio
    async def test_ten_mock_revisions_unattended(self, tmp_path):
        """R3 acceptance criterion 1 (from AGENT_PLAN §4.5): 10 gated revisions
        unattended with MockProvider; ledger shows accept + reject mix; 0 rejected-diff
        leaks (program on disk always passes validation)."""
        import asyncio
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from run_revision_loop import run_loop, parse_args  # noqa: E402

        program_path = str(tmp_path / "policy_program.json")
        ledger_path = str(tmp_path / "ledger.jsonl")
        PolicyProgram.from_dict(DEFAULT_PROGRAM).save(program_path)

        ns = parse_args.__wrapped__() if hasattr(parse_args, "__wrapped__") else None
        # build args namespace directly (argparse defaults + overrides)
        from argparse import Namespace
        args = Namespace(max_revisions=10, provider="mock", model=None,
                         program_path=program_path, ledger_path=ledger_path,
                         db_path="/nonexistent/telemetry.duckdb", live_gates=False,
                         fixture_shadow=True, quick_batch_episodes=3,
                         quick_batch_steps=5000, no_git=True)
        rc = await run_loop(args)
        assert rc == 0

        led = RevisionLedger(ledger_path)
        revs = [e for e in led.all() if e.get("type") == "policy_revision"]
        assert len(revs) == 10
        accepted = [e for e in revs if e["accepted"]]
        rejected = [e for e in revs if not e["accepted"]]
        assert accepted, "mock canned diffs must include accepted revisions"
        assert rejected, "mock canned diffs must include the out-of-vocab rejection"
        assert all(e["reject_code"] == "ERR_BOUNDS" for e in rejected)

        # 0 rejected-diff leaks: the program on disk is exactly the last accepted version
        final = json.load(open(program_path))
        assert final["version"] == len(accepted) + 1
        last_accepted = accepted[-1]
        assert final["version"] == last_accepted["revision"]
        # every accepted program's params must be within the manifest bounds
        manifest = build_manifest(final["version"], final.get("domain", "nethack"),
                                  live_macros={m["name"]: m["body"] for m in final["macros"]})
        for path, value in final.get("params", {}).items():
            leaf = manifest.leaf(path)
            assert leaf is not None, f"leaked param {path}"
            assert leaf["min"] <= value <= leaf["max"], f"leaked out-of-bounds {path}={value}"