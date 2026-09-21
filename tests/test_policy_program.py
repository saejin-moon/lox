"""
R2 regression tests: Policy Program + Goal Interpreter.
Verifies the condition evaluator, program container, overlay application, and the
GoalInterpreter's transition/directive semantics. The pre-R2 MacroAscensionDirector phase
machine these tests were once run against for decision-trace equivalence was deleted
(R5→R6 checklist item 6); the expected outcomes below are frozen from that equivalence.
"""
import pytest

from corp.policy.goal_state import AscensionPhase
from corp.env.blstats import HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.policy.config import PolicyConfig
from corp.policy.goal_interpreter import GoalInterpreter
from corp.policy.predicates import parse, eval_condition, nethack_bindings, tokenize
from corp.policy.program import PolicyProgram


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def base_ctx(**over):
    ctx = {
        "xl": 1, "depth": 1, "dnum": 0, "lawful": True,
        "has_poison_res": False, "has_reflection": False, "has_excalibur": False,
        "minetown_visited": False, "donations_count": 0, "ac": 10,
        "steps_in_goal": 0, "turn": 100,
    }
    ctx.update(over)
    return ctx


def test_parse_and_eval():
    b = nethack_bindings(base_ctx())
    assert eval_condition("(true)", b) is True
    assert eval_condition("(or (xl_ge 4) (depth_ge 4))", b) is False
    assert eval_condition("(and (lawful) (not has_excalibur))", b) is True
    assert eval_condition("(depth_between 1 3)", b) is True
    assert eval_condition("(not (in_mines))", b) is True


def test_parse_rejects_unknown_symbols_and_depth():
    b = nethack_bindings(base_ctx())
    with pytest.raises(ValueError, match="ERR_UNKNOWN_SYMBOL"):
        eval_condition("(frobnicate 3)", b)
    with pytest.raises(ValueError, match="ERR_DEPTH_BUDGET"):
        eval_condition("(and (or (and (or (true)))))", b)
    with pytest.raises(ValueError):
        parse("(unclosed")


def test_tokenizer_strips_comments():
    assert tokenize("(true) ; comment") == ["(", "true", ")"]


# ---------------------------------------------------------------------------
# Policy program container
# ---------------------------------------------------------------------------

def test_default_program_loads_and_has_thirteen_goals():
    """R6: the ascension stack is a 13-goal mortality-driven plan."""
    prog = PolicyProgram.load("data/policy_program.json")
    assert [g.goal for g in prog.strategy_plan] == [
        "explore_floor", "early_survival_stack", "forge_excalibur",
        "hunt_poison_res", "enter_sokoban", "equip_upgrade", "goto_minetown",
        "survival_intrinsics", "descend", "enter_gehennom", "castle_wishing",
        "vlad_invocation", "ascension_run",
    ]


def test_overlay_applies_and_rejects_unknown(tmp_path):
    prog = PolicyProgram.from_dict({
        "version": 1, "domain": "nethack",
        "params": {"survival.rest_below_frac": 0.65},
        "strategy_plan": [{"goal": "descend"}],
    })
    cfg = PolicyConfig.defaults()
    assert cfg.survival.rest_below_frac == 0.60
    prog.apply_overlay(cfg)
    assert cfg.survival.rest_below_frac == 0.65

    bad = PolicyProgram.from_dict({
        "version": 1, "domain": "nethack",
        "params": {"survival.nonexistent": 1},
        "strategy_plan": [{"goal": "descend"}],
    })
    with pytest.raises(ValueError, match="ERR_UNKNOWN_SYMBOL"):
        bad.apply_overlay(PolicyConfig.defaults())


# ---------------------------------------------------------------------------
# GoalInterpreter transition/directive semantics (frozen from R2 equivalence)
# ---------------------------------------------------------------------------

def make_bl(xl=1, depth=1, dnum=0, dlevel=1, ac=10, alignment=1, turn=100, hp=40, max_hp=40):
    import numpy as np
    from corp.env.blstats import BottomLineStats
    raw = np.zeros(27, dtype=np.int64)
    raw[0], raw[1] = 10, 10
    raw[10], raw[11] = hp, max_hp
    raw[12] = depth
    raw[16] = ac
    raw[18] = xl
    raw[20] = turn
    raw[23] = dnum
    raw[24] = dlevel
    raw[26] = alignment
    return BottomLineStats.from_blstats(raw)


def drive(interpreter, bl, role="valkyrie", **flags):
    """Update the interpreter; return (directive, phase)."""
    interpreter.update_state(bl, role=role, **flags)
    directive = interpreter.get_navigation_directive(bl, None)
    return directive, interpreter.state.current_phase


def test_goal_sequence_transitions():
    gi = GoalInterpreter(PolicyConfig.defaults(), program_path="")

    # 1. Fresh DL1 XL1: explore phase, no directive
    d, p = drive(gi, make_bl())
    assert (d, p) == (None, AscensionPhase.EARLY_EXPLORATION)

    # 2. XL 5 lawful: early_survival_stack activates first (R6 #1 holds until
    #    the AC ladder / exit depth / timeout threshold is met)
    d, p = drive(gi, make_bl(xl=5, depth=3))
    assert p == AscensionPhase.SURVIVAL_STACK

    # 3. AC <= survival target: survival stack completes → forge_excalibur
    d, p = drive(gi, make_bl(xl=5, depth=3, ac=0))
    assert p == AscensionPhase.EXCALIBUR_FORGE

    # 4. Excalibur obtained: forge completes → hunt_poison_res
    inv = InventoryNormalizer()
    excal = NormalizedItem(current_letter="a", raw_str="the blessed Excalibur", buc_state="BLESSED")
    inv.active_items[excal.uid] = excal
    d, p = drive(gi, make_bl(xl=5, depth=3, ac=0), inv_tracker=inv)
    assert p == AscensionPhase.POISON_RES_HUNT

    # 5. Poison resistance message: hunt completes → enter_sokoban, directive ENTER_SOKOBAN
    d, p = drive(gi, make_bl(xl=5, depth=4, ac=0), message="You feel healthy.")
    assert p == AscensionPhase.SOKOBAN_PROGRESSION
    assert d == "ENTER_SOKOBAN"

    # 6. Reflection obtained: sokoban completes → equip_upgrade activates (R6 #2)
    refl = NormalizedItem(current_letter="b", raw_str="an uncursed amulet of reflection", buc_state="UNCURSED")
    inv.active_items[refl.uid] = refl
    d, p = drive(gi, make_bl(xl=6, depth=3, ac=0), inv_tracker=inv)
    assert p == AscensionPhase.EQUIP_UPGRADE

    # 7. AC <= -10: equip completes → goto_minetown, directive GOTO_MINETOWN on DL 2-4 dnum 0
    d, p = drive(gi, make_bl(xl=6, depth=3, ac=-10), inv_tracker=inv)
    assert p == AscensionPhase.MINETOWN_PROTECTION
    assert d == "GOTO_MINETOWN"

    # 8. AC <= 0: minetown completes → survival_intrinsics skipped (DL 3 < 8)
    #    → descend, directive DESCEND
    d, p = drive(gi, make_bl(xl=8, depth=6, ac=-10), inv_tracker=inv)
    assert p == AscensionPhase.DEEP_DESCENT
    assert d == "DESCEND"


def test_mines_and_sokoban_directives():
    gi = GoalInterpreter(PolicyConfig.defaults(), program_path="")

    # Mine's End (dnum 2, dlevel 10) → ASCEND_FROM_MINES regardless of goal
    d, _ = drive(gi, make_bl(xl=7, depth=12, dnum=2, dlevel=10))
    assert d == "ASCEND_FROM_MINES"

    # In Sokoban with reflection: EXIT_SOKOBAN
    inv = InventoryNormalizer()
    refl = NormalizedItem(current_letter="b", raw_str="an uncursed amulet of reflection", buc_state="UNCURSED")
    inv.active_items[refl.uid] = refl
    bl = make_bl(xl=6, depth=8, dnum=3, dlevel=2)
    d, _ = drive(gi, bl, inv_tracker=inv)
    assert d == "EXIT_SOKOBAN"


def test_forge_budget_skips_after_max_steps():
    """forge_excalibur with max_steps 2000 auto-advances to hunt_poison_res.
    (R6: early_survival_stack consumes its own 1500-step budget first, so the
    drive must outlast both budgets.)"""
    gi = GoalInterpreter(PolicyConfig.defaults(), program_path="")
    bl = make_bl(xl=5, depth=3, alignment=0)  # neutral: forge's when-lawful skips it

    for _ in range(3600):
        drive(gi, bl)
    assert gi.state.current_phase == AscensionPhase.POISON_RES_HUNT


def test_stair_deferral():
    gi = GoalInterpreter(PolicyConfig.defaults(), program_path="")
    bl = make_bl(xl=1, depth=1)
    args = (30, 40)  # unvisited_count, turns_spent
    assert gi.should_defer_stairs_for_farming(bl, *args) is True
    bl2 = make_bl(xl=3, depth=1)
    assert gi.should_defer_stairs_for_farming(bl2, *args) is False


def test_goal_events_flush(tmp_path):
    gi = GoalInterpreter(PolicyConfig.defaults())
    bl = make_bl()
    for _ in range(3):
        gi.update_state(bl, role="valkyrie")
    inv = InventoryNormalizer()
    bl5 = make_bl(xl=5, depth=3)
    gi.update_state(bl5, role="valkyrie")          # xl 5 → forge activates (no excalibur yet)
    excal = NormalizedItem(current_letter="a", raw_str="the blessed Excalibur", buc_state="BLESSED")
    inv.active_items[excal.uid] = excal
    gi.update_state(bl5, inv_tracker=inv, role="valkyrie")  # excalibur obtained → forge completes

    db = str(tmp_path / "goal_events.duckdb")
    n = gi.flush_events(db, run_id="testrun", episode_id="ep_test")
    assert n > 0
    import duckdb
    con = duckdb.connect(db, read_only=True)
    rows = con.execute("SELECT goal, event FROM goal_events ORDER BY ts").fetchall()
    con.close()
    assert ("explore_floor", "completed") in rows
    assert ("early_survival_stack", "activated") in rows  # R6: next certified milestone
