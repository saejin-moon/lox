"""
S1 tests: authoring-tree compiler + byte-equivalent compiled artifacts +
decision-trace equivalence (AGENT_PLAN §1.1, §2-S1).
"""
from __future__ import annotations

import json
import os

import pytest

from lox.policy import compiler
from lox.policy.compiler import (
    compile_and_commit, compile_authoring_tree, write_authoring_tree,
)
from lox.policy.program import DEFAULT_PROGRAM, PolicyProgram

LIVE = {"nethack": "data/policy_program.json",
        "minihack": "data/policy_program_minihack.json"}


def make_blstats(hp=20, max_hp=20, depth=1, ac=9, xl=1, turn=10, hunger=1, dnum=0):
    from lox.env.blstats import BottomLineStats
    return BottomLineStats(
        x=10, y=10, strength_pct=0, strength=16, dexterity=15, constitution=16,
        intelligence=10, wisdom=10, charisma=10, score=100, hp=hp, max_hp=max_hp,
        depth=depth, gold=0, energy=10, max_energy=10, ac=ac, monster_level=1,
        experience=xl, turn=turn, hunger_state=hunger, encumbrance=0,
        dungeon_number=dnum, level_number=1, condition_bits=0, alignment=0,
    )


# ===========================================================================
# Round-trip byte-equivalence (the S1 storage contract)
# ===========================================================================

@pytest.mark.parametrize("domain", ["nethack", "minihack"])
def test_migrated_tree_compiles_byte_equivalent(tmp_path, domain):
    """The canonical migration requirement: tree → compiled artifact is the SAME
    dict (and the same bytes) the executor consumed before the migration."""
    tree = os.path.join("data", "program", domain)  # literal: conftest sandboxes tree_dir_for
    if not os.path.isdir(tree):
        pytest.skip("authoring tree not migrated")
    compiled = compile_authoring_tree(tree)
    live = PolicyProgram.load(LIVE[domain])
    assert compiled.to_dict() == live.to_dict()
    # byte-equivalence: serialized forms are identical
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    compiled.save(str(a))
    live.save(str(b))
    assert a.read_bytes() == b.read_bytes()


def test_default_program_tree_roundtrip(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    compiled = compile_authoring_tree(tree)
    assert compiled.to_dict() == prog.to_dict()
    # a second write is idempotent (stable file bytes)
    files_first = {p: open(os.path.join(tree, p), encoding="utf-8").read()
                   for p in _tree_files(tree)}
    write_authoring_tree(compiled, tree)
    files_second = {p: open(os.path.join(tree, p), encoding="utf-8").read()
                    for p in _tree_files(tree)}
    assert files_first == files_second


def _tree_files(root: str):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(out)


def test_interlock_rules_preserved(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    compiled = compile_authoring_tree(tree)
    original_interlocks = [r for r in prog.tactic_rules if r.get("interlock")]
    compiled_interlocks = [r for r in compiled.tactic_rules if r.get("interlock")]
    assert len(original_interlocks) >= 1
    assert original_interlocks == compiled_interlocks
    # interlock rules are materialized as tree files with the flag
    rule_texts = [open(os.path.join(tree, "rules", f), encoding="utf-8").read()
                  for f in os.listdir(os.path.join(tree, "rules"))]
    assert sum("(interlock true)" in t for t in rule_texts) == len(original_interlocks)


def test_note_and_unless_none_keys_roundtrip(tmp_path):
    """Rules carry explicit None keys (`unless: null`, `note: null`) — the tree must
    preserve them for byte-equivalence (validator _render_rule emits both keys)."""
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.tactic_rules = prog.tactic_rules + [
        {"when": '(monster "jackal")', "do": "retreat", "unless": None, "note": None}]
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    compiled = compile_authoring_tree(tree)
    assert compiled.to_dict() == prog.to_dict()
    assert compiled.tactic_rules[-1] == prog.tactic_rules[-1]


def test_params_nogoods_macros_roundtrip(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    prog.params = {"survival.rest_below_frac": 0.65, "navigation.monster_cost": 1200.0}
    prog.macros = [{"name": "when_endangered",
                    "body": "(and (hp_frac <= 0.4) (not has_healing))"}]
    prog.nogoods = [{"when": "(adjacent_hostiles)", "cause": "death by jackal",
                     "forbid": "MELEE_ATTACK"}]
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    compiled = compile_authoring_tree(tree)
    assert compiled.to_dict() == prog.to_dict()
    assert compiled.params == prog.params
    assert compiled.macros == prog.macros
    assert compiled.nogoods == prog.nogoods


# ===========================================================================
# Fail-loud compiler errors
# ===========================================================================

def test_missing_tree_errors(tmp_path):
    with pytest.raises(ValueError, match="ERR_NO_TREE"):
        compile_authoring_tree(str(tmp_path / "nope"))


def test_malformed_rule_file_errors(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    with open(os.path.join(tree, "rules", "00_broken.sexpr"), "w") as f:
        f.write("(rule add tactic_rules (do retreat))\n")
    with pytest.raises(ValueError, match="ERR_PARSE"):
        compile_authoring_tree(tree)


def test_handler_files_rejected_until_s3(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "nethack")
    write_authoring_tree(prog, tree)
    with open(os.path.join(tree, "handlers", "smoke.sexpr"), "w") as f:
        f.write("(goal-handler smoke)\n")
    with pytest.raises(ValueError, match="ERR_HANDLERS_UNVERIFIED"):
        compile_authoring_tree(tree)


def test_roundtrip_gate_fails_loud(tmp_path, monkeypatch):
    """compile_and_commit refuses to commit when tree ↔ candidate diverge."""
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "nethack")
    live = str(tmp_path / "program.json")

    def truncating_writer(program, tree_dir):
        os.makedirs(tree_dir, exist_ok=True)
        with open(os.path.join(tree_dir, "program.json"), "w") as f:
            json.dump({"version": program.version, "domain": program.domain}, f)
        return []

    monkeypatch.setattr(compiler, "write_authoring_tree", truncating_writer)
    with pytest.raises(ValueError, match="ERR_ROUNDTRIP"):
        compile_and_commit(prog, tree, live)


def test_compile_and_commit_writes_compiled_and_archive(tmp_path):
    prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
    tree = str(tmp_path / "tree")
    live = str(tmp_path / "live.json")
    out = compile_and_commit(prog, tree, live)
    assert json.load(open(out)) == prog.to_dict()
    assert PolicyProgram.load(live).to_dict() == prog.to_dict()
    archives = os.listdir(compiler.ARCHIVE_DIR)  # sandboxed by conftest
    assert len(archives) == 1 and archives[0].endswith(".json")


# ===========================================================================
# Decision-trace equivalence (per-manager: goals + tactics)
# ===========================================================================

@pytest.fixture
def compiled_live_pair(tmp_path):
    """(live program, tree-compiled artifact path) for the live nethack program."""
    if not os.path.isdir("data/program/nethack"):
        pytest.skip("authoring tree not migrated")
    live = PolicyProgram.load(LIVE["nethack"])
    # literal path: conftest sandboxes compiler.tree_dir_for
    compiled = compile_authoring_tree(os.path.join("data", "program", "nethack"))
    path = str(tmp_path / "compiled_nethack.json")
    compiled.save(path)
    return live, path


def test_goal_interpreter_decision_trace_equivalent(compiled_live_pair):
    """Per-goal decision-trace equivalence: the compiled artifact drives the exact
    same (phase, navigation directive) sequence as the pre-migration live program."""
    from lox.policy.goal_interpreter import GoalInterpreter

    _live, compiled_path = compiled_live_pair
    interp_live = GoalInterpreter(failure_counters={}, program_path=LIVE["nethack"])
    interp_comp = GoalInterpreter(failure_counters={}, program_path=compiled_path)
    assert interp_live.program.to_dict() == interp_comp.program.to_dict()

    sweep = [
        make_blstats(),
        make_blstats(depth=2, xl=2, ac=8, turn=200),
        make_blstats(hp=6, max_hp=20, depth=3, xl=3, ac=6, turn=800, hunger=3),
        make_blstats(depth=5, xl=5, ac=4, turn=1500),
        make_blstats(dnum=2, depth=3, xl=2, turn=2500),          # Mines routing
        make_blstats(dnum=3, depth=7, xl=6, turn=4000),          # Sokoban
        make_blstats(hp=3, max_hp=30, depth=8, xl=8, ac=-2, turn=6000, hunger=4),
        make_blstats(depth=14, xl=12, ac=-8, turn=12000),
    ]
    for i, bs in enumerate(sweep):
        phase_live = interp_live.update_state(bs, role="valkyrie")
        phase_comp = interp_comp.update_state(bs, role="valkyrie")
        assert phase_live == phase_comp, f"phase diverged at fixture {i}"
        assert (interp_live.get_navigation_directive(bs)
                == interp_comp.get_navigation_directive(bs)), f"directive diverged at {i}"
        assert (interp_live.should_defer_stairs_for_farming(bs, unvisited_count=5, turns_spent=50)
                == interp_comp.should_defer_stairs_for_farming(bs, 5, 50))
    assert interp_live.program.tactic_rules == interp_comp.program.tactic_rules


def test_tactic_engine_decision_trace_equivalent(compiled_live_pair):
    """Per-manager equivalence for the combat tactic engine: same first-match verb
    (and note/index) across a fixture combat sweep."""
    from lox.policy.tactics import TacticRuleEngine

    live, compiled_path = compiled_live_pair
    compiled = PolicyProgram.load(compiled_path)
    eng_live = TacticRuleEngine(live.tactic_rules)
    eng_comp = TacticRuleEngine(compiled.tactic_rules)
    assert len(eng_live) == len(eng_comp)

    base = {"hp": 20, "max_hp": 20, "has_poison_res": False, "adjacent_hostiles": 1}
    sweep = [
        {**base, "monster_name": "giant spider"},
        {**base, "monster_name": "killer bee"},
        {**base, "monster_name": "giant spider", "has_poison_res": True},
        {**base, "monster_name": "ogre", "hp": 6},
        {**base, "monster_name": "rothe", "hp": 4},
        {**base, "monster_name": "jackal"},
        {**base, "monster_name": ""},
    ]
    for i, ctx in enumerate(sweep):
        assert eng_live.evaluate(ctx) == eng_comp.evaluate(ctx), f"tactic diverged at {i}"


def test_compiled_artifact_is_executor_contract(tmp_path, compiled_live_pair):
    """Zero executor rewrites: PolicyProgram.load + apply_overlay consume the
    compiled artifact exactly like the live file."""
    from lox.policy.config import default_config
    _live, compiled_path = compiled_live_pair
    prog = PolicyProgram.load(compiled_path)
    cfg = default_config()
    prog.apply_overlay(cfg)  # must not raise
    assert prog.version == PolicyProgram.load(LIVE["nethack"]).version