"""
Tests for Two-Tier Error Recovery in Author Agent (Tier 1 programmatic repair & Tier 2 interactive LLM re-prompt).
"""
import pytest
from unittest.mock import AsyncMock

from lox.deliberative.providers.base import LLMResponse
from lox.deliberative.providers.mock_provider import MockProvider as BaseMockProvider
from lox.policy.author_agent import repair_diff_text, author_session
from lox.policy.manifest import build_manifest
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM
from lox.policy.ledger import RevisionLedger

MANIFEST = build_manifest(1, "nethack")


class SequentialMockProvider(BaseMockProvider):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = list(responses)
        self.call_count = 0
        self.model = "mock-repair-model"
        self.last_prompts = []

    async def generate_text(self, system: str, user: str, context: dict | None = None) -> LLMResponse:
        self.last_prompts.append((system, user))
        if self.call_count < len(self.responses):
            resp_text = self.responses[self.call_count]
        else:
            resp_text = self.responses[-1]
        self.call_count += 1
        return LLMResponse(
            thinking_content="",
            parsed_payload=None,
            raw_text=resp_text,
            tokens_in=100,
            tokens_out=50,
        )


def test_repair_diff_text_programmatic():
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.version = 5

    flawed_diff = """```python
revision: 2, parent: 1, author: "flawed-llm", reason: "testing repair"
set hp_emergency = 0.55;
defmacro low_hp(hp): return hp_frac <= hp;
```"""

    repaired = repair_diff_text(flawed_diff, prog, MANIFEST)

    # Markdown fences stripped
    assert "```" not in repaired
    # Revision and parent normalized
    assert "revision: 6, parent: 5" in repaired
    # policy_params prefix added
    assert "set policy_params.hp_emergency = 0.55" in repaired
    # return stripped from defmacro
    assert "defmacro low_hp(hp) = hp_frac <= hp" in repaired
    # semicolons stripped
    assert ";" not in repaired


@pytest.mark.asyncio
async def test_author_session_tier1_auto_repair(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.version = 1
    prog_path = str(tmp_path / "policy_program.json")
    ledger_path = str(tmp_path / "ledger.jsonl")
    ledger = RevisionLedger(ledger_path)

    # Diff has wrong revision header (revision: 99, parent: 98), but is otherwise valid
    flawed_diff = """
revision: 99, parent: 98, author: "tier1-test", reason: "testing tier 1"
goal deprioritize explore_floor: when hunger_ge(2)
"""
    provider = SequentialMockProvider([flawed_diff])

    outcome = await author_session(
        provider=provider,
        program=prog,
        program_path=prog_path,
        ledger=ledger,
        ledger_path=ledger_path,
        commit=False,
        manifest=MANIFEST,
        bundle={},
    )

    assert outcome.accepted, f"Session failed: {outcome.error_code} - {outcome.detail}"
    assert outcome.candidate is not None
    assert outcome.candidate.version == 2


@pytest.mark.asyncio
async def test_author_session_tier2_interactive_llm_repair(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.version = 1
    prog_path = str(tmp_path / "policy_program.json")
    ledger_path = str(tmp_path / "ledger.jsonl")
    ledger = RevisionLedger(ledger_path)

    # 1st response: contains an unknown predicate that Tier 1 cannot fix
    broken_diff = """
revision: 2, parent: 1, author: "tier2-test", reason: "broken diff"
goal prioritize descend: when unknown_magic_predicate
"""

    # 2nd response (Tier 2 repair): fixes the predicate to a valid one
    corrected_diff = """
revision: 2, parent: 1, author: "tier2-test", reason: "fixed predicate"
goal prioritize descend: when stairs_known
"""
    provider = SequentialMockProvider([broken_diff, corrected_diff])

    outcome = await author_session(
        provider=provider,
        program=prog,
        program_path=prog_path,
        ledger=ledger,
        ledger_path=ledger_path,
        commit=False,
        manifest=MANIFEST,
        bundle={},
    )

    assert outcome.accepted, f"Tier 2 repair failed: {outcome.error_code} - {outcome.detail}"
    assert outcome.candidate is not None
    assert outcome.candidate.version == 2
    # Verify provider was called a second time for Tier 2 repair
    assert provider.call_count == 2
    # Verify the repair prompt included the compiler error diagnostic
    repair_prompt = provider.last_prompts[1][1]
    assert "unknown_magic_predicate" in repair_prompt
    assert "DIFF COMPILATION ERROR" in repair_prompt or "FIX COMPILATION FAILURE" in repair_prompt


@pytest.mark.asyncio
async def test_author_session_tier2_outer_repair(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.version = 1
    prog_path = str(tmp_path / "policy_program.json")
    ledger_path = str(tmp_path / "ledger.jsonl")
    ledger = RevisionLedger(ledger_path)

    # 1st response on turn 1 (max_turns=1, so loop must return it)
    broken_diff = """
revision: 2, parent: 1, author: "tier2-outer-test", reason: "broken on final turn"
goal prioritize descend: when unknown_magic_predicate
"""

    # 2nd response to outer Tier 2 repair prompt
    corrected_diff = """
revision: 2, parent: 1, author: "tier2-outer-test", reason: "fixed predicate"
goal prioritize descend: when stairs_known
"""
    provider = SequentialMockProvider([broken_diff, corrected_diff])

    outcome = await author_session(
        provider=provider,
        program=prog,
        program_path=prog_path,
        ledger=ledger,
        ledger_path=ledger_path,
        commit=False,
        manifest=MANIFEST,
        bundle={},
        max_turns=1,
    )

    assert outcome.accepted, f"Outer Tier 2 repair failed: {outcome.error_code} - {outcome.detail}"
    assert outcome.candidate is not None
    assert outcome.candidate.version == 2
    assert provider.call_count == 2
    repair_prompt = provider.last_prompts[1][1]
    assert "FIX COMPILATION FAILURE" in repair_prompt
    assert "unknown_magic_predicate" in repair_prompt
