"""R3 tests: deliberative unification onto the diff contract (AGENT_PLAN step 9)."""
import json
import os
import tempfile

import pytest

from corp.deliberative.deadlock_resolver import DeadlockResolver, emit_deadlock_diff
from corp.deliberative.schemas import HTNGraphPatch, SubTaskSpec
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM


def make_patch(**kw):
    defaults = dict(
        deadlock_cause="Spatial oscillation between two corridor tiles",
        confidence=0.9,
        abandon_current_macro=False,
        injected_subtasks=[SubTaskSpec(task_name="SEARCH")],
        new_invariants=[],
    )
    defaults.update(kw)
    return HTNGraphPatch(**defaults)


class TestPatchToDiff:
    def test_diff_header_and_nogood(self):
        diff = DeadlockResolver.patch_to_diff(make_patch(), parent_version=3,
                                              failed_task="EXPLORE", depth=4, turn=900)
        assert "(revision 4 (parent 3)" in diff
        assert 'author "deadlock-resolver"' in diff
        assert "EXPLORE" in diff
        assert "(depth_le 4)" in diff
        assert "(turn_ge 900)" in diff

    def test_no_depth_clause_when_unknown(self):
        diff = DeadlockResolver.patch_to_diff(make_patch(), parent_version=1)
        assert "depth_le" not in diff


class TestEmitDeadlockDiff:
    def test_acceptance_persists_bumped_program(self, tmp_path):
        path = str(tmp_path / "policy_program.json")
        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        program.save(path)
        diff = DeadlockResolver.patch_to_diff(make_patch(), parent_version=program.version,
                                              failed_task="STEP", depth=2, turn=50)
        assert emit_deadlock_diff(diff, program, program_path=path) is True
        reloaded = PolicyProgram.load(path)
        assert reloaded.version == program.version + 1
        assert reloaded.nogoods[-1]["cause"].startswith("Spatial oscillation")

    def test_zero_rejected_diff_leaks(self, tmp_path):
        """A malformed diff must never rewrite the program file."""
        path = str(tmp_path / "policy_program.json")
        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        program.save(path)
        before = open(path).read()
        bad = "(revision 99 (parent 1) (author \"x\") (domain nethack) (reason \"r\"))\n(nogood add (when (nonsense_predicate)) (cause \"c\"))"
        assert emit_deadlock_diff(bad, program, program_path=path) is False
        assert open(path).read() == before


class TestAgentDiffPath:
    def test_agent_has_emit_deadlock_diff(self):
        """The agent's in-game deliberation path must route through diff emission."""
        from corp.agent.corp_agent import CORPAgent
        assert hasattr(CORPAgent, "_emit_deadlock_diff")
        import inspect
        # source check spans the MRO (behavior-preserving split moved the runner
        # into corp/agent/episode_runner.py)
        src = "\n".join(inspect.getsource(c) for c in CORPAgent.__mro__ if c is not object)
        # plan_queue injection removed; diff emission wired
        assert "plan_queue.extend" not in src
        assert "_emit_deadlock_diff" in src
