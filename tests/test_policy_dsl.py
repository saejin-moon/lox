"""R3 tests: S-expression diff reader (corp/policy/dsl.py, MACRO.md §2–3)."""
import pytest

from corp.policy.dsl import (
    parse_diff, extract_diff_text, render_node, DiffParseError, parse_forms,
)


class TestDiffParsing:
    def test_header_and_set(self):
        d = parse_diff('(revision 2 (parent 1) (author "gpt") (domain nethack) (reason "r"))\n'
                       '(set policy_params.survival.rest_below_frac 0.65)')
        assert d.header.revision == 2 and d.header.parent == 1
        assert d.header.author == "gpt" and d.header.domain == "nethack"
        assert len(d.sets) == 1
        assert d.sets[0].path == "policy_params.survival.rest_below_frac"
        assert d.sets[0].value == 0.65

    def test_rule_add(self):
        d = parse_diff('(revision 2 (parent 1))\n'
                       '(rule add tactic_rules (when (and (in_mines) (hp_frac <= 0.40)))'
                       ' (do retreat) (note "attrition"))')
        r = d.rules[0]
        assert r.action == "add"
        assert r.rule["do"] == "retreat"
        assert r.rule["when"] == ("call", "and", [
            ("call", "in_mines", []),
            ("call", "hp_frac", ["<=", 0.4]),
        ])
        assert r.rule["note"] == "attrition"

    def test_rule_remove_by_index(self):
        d = parse_diff('(revision 2 (parent 1))\n(rule remove tactic_rules 3)')
        assert d.rules[0].action == "remove" and d.rules[0].index == 3

    def test_goal_ops(self):
        d = parse_diff('(revision 2 (parent 1))\n'
                       '(goal prioritize descend (until (depth_ge 9)))\n'
                       '(goal deprioritize enter_sokoban (after (failures_in_10_episodes_ge 2 enter_sokoban)))\n'
                       '(goal set_threshold descend descent.min_hp_frac 0.5)')
        assert [g.kind for g in d.goals] == ["prioritize", "deprioritize", "set_threshold"]
        assert d.goals[2].path == "descent.min_hp_frac"

    def test_nogood_and_defmacro(self):
        d = parse_diff('(revision 2 (parent 1))\n'
                       '(defmacro when_endangered (and (adjacent_hostiles) (not has_healing)))\n'
                       '(nogood add (when (depth_le 5)) (cause "throne shock") (forbid SIT))')
        assert d.defmacros[0].name == "when_endangered"
        assert d.nogoods[0].cause == "throne shock"
        assert d.nogoods[0].forbid == "SIT"

    def test_comments_stripped(self):
        d = parse_diff('(revision 2 (parent 1)) ; header comment\n'
                       '; full-line comment\n(set policy_params.survival.rest_below_frac 0.65) ; trailing')
        assert len(d.sets) == 1

    def test_depth_budget_enforced(self):
        with pytest.raises(DiffParseError) as e:
            parse_diff('(revision 2 (parent 1))\n(rule add tactic_rules '
                       '(when (and (or (and (or (and (lawful) (true)) (true)) (true)) (true)))) (do retreat))')
        assert e.value.code == "ERR_DEPTH_BUDGET"

    def test_unknown_head_rejected(self):
        with pytest.raises(DiffParseError):
            parse_diff('(revision 2 (parent 1))\n(celebrate (true))')

    def test_set_requires_policy_params_prefix(self):
        with pytest.raises(DiffParseError):
            parse_diff('(revision 2 (parent 1))\n(set survival.rest_below_frac 0.65)')

    def test_missing_parent_rejected(self):
        with pytest.raises(DiffParseError):
            parse_diff('(revision 2)\n(set policy_params.survival.rest_below_frac 0.65)')


class TestExtractDiffText:
    def test_plain(self):
        raw = '(revision 2 (parent 1))\n(set policy_params.survival.rest_below_frac 0.65)'
        assert extract_diff_text(raw) == raw

    def test_code_fences_and_prose(self):
        raw = ('Here is my proposed diff:\n```lisp\n'
               '(revision 2 (parent 1))\n(set policy_params.survival.rest_below_frac 0.65)\n'
               '(rule add tactic_rules (when (lawful)) (do retreat))\n```\n'
               'This raises the resting gate because of stall telemetry.')
        out = extract_diff_text(raw)
        assert out.startswith("(revision 2")
        assert "rule add" in out
        assert "stall telemetry" not in out

    def test_missing_revision_form(self):
        with pytest.raises(DiffParseError):
            extract_diff_text("I suggest setting rest_below_frac to 0.65")

    def test_unbalanced_rejected(self):
        with pytest.raises(DiffParseError):
            extract_diff_text("(revision 2 (parent 1)\n(set policy_params.survival.x 1)")


class TestRenderNode:
    def test_roundtrip(self):
        node = ("call", "and", [("call", "in_mines", []), ("call", "hp_frac", ["<=", 0.4])])
        text = render_node(node)
        assert text == "(and (in_mines) (hp_frac <= 0.4))"
        # re-parses to the same structure (modulo 0.4 float identity)
        re_node = parse_forms(text)[0]
        assert re_node == node

    def test_zero_arg_and_bool(self):
        assert render_node(("call", "lawful", [])) == "(lawful)"
        assert render_node(True) == "true" and render_node(False) == "false"