"""
Unit tests for EpistemicWorker:
- Altar B.U.C. Testing & Observation Collapse
- Altar Sacrifice (fresh corpse offering)
- Wand Engrave-Testing & Explosion Safety Guard
- Epistemic Navigation Coordination
- Action Dispatcher Multi-Keystroke Handlers
"""

import numpy as np
import pytest

from lox.env.blstats import BottomLineStats, HungerState
from lox.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from lox.epistemic.epistemic_manager import EpistemicManager
from lox.domain.epistemic_worker import EpistemicWorker
from lox.domain.navigation_manager import NavigationManager
from lox.workers.dispatcher import ActionDispatcher
from lox.planner.htn import Task


def make_dummy_blstats(y=10, x=10, hp=20, max_hp=20, turn=100, depth=1, encumbrance=0):
    raw = np.zeros(27, dtype=np.int64)
    raw[0] = x
    raw[1] = y
    raw[10] = hp
    raw[11] = max_hp
    raw[20] = turn
    raw[22] = encumbrance
    raw[23] = depth
    return BottomLineStats.from_blstats(raw)


def test_epistemic_worker_has_untested_items():
    ep_mgr = EpistemicManager()
    worker = EpistemicWorker(ep_mgr)
    inv = InventoryNormalizer()

    # Empty inventory
    assert not worker.has_untested_items(inv)

    # Item with known BUC
    item1 = NormalizedItem(uid="u1", raw_str="an uncursed plate mail", current_letter="a", buc_state="UNCURSED")
    inv.active_items["u1"] = item1
    assert not worker.has_untested_items(inv)

    # Item with UNKNOWN BUC
    item2 = NormalizedItem(uid="u2", raw_str="a helmet", current_letter="b", buc_state="UNKNOWN")
    inv.active_items["u2"] = item2
    assert worker.has_untested_items(inv)

    # Once added to tested_uids, should return False
    worker.tested_uids.add("u2")
    assert not worker.has_untested_items(inv)


def test_epistemic_worker_altar_test_evaluation():
    ep_mgr = EpistemicManager()
    worker = EpistemicWorker(ep_mgr)
    inv = InventoryNormalizer()

    blstats = make_dummy_blstats(y=5, x=5)
    chars = np.full((21, 79), ord("."), dtype=np.int32)
    lvl_altars = {(5, 5)}  # Hero standing on altar

    item = NormalizedItem(uid="helm1", raw_str="an iron skull cap", current_letter="b", buc_state="UNKNOWN")
    inv.active_items["helm1"] = item

    # 1. Standing on altar with unknown armor piece -> ALTAR_TEST task
    task = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
        has_adjacent_hostiles=False,
    )
    assert task is not None
    assert task.name == "ALTAR_TEST"
    assert task.args["slot"] == "b"
    assert task.args["uid"] == "helm1"

    # 2. Hostile combat interlock -> Must return None
    task_combat = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
        has_adjacent_hostiles=True,
    )
    assert task_combat is None


def test_epistemic_worker_altar_test_observation_collapse():
    ep_mgr = EpistemicManager()
    worker = EpistemicWorker(ep_mgr)
    inv = InventoryNormalizer()
    blstats = make_dummy_blstats()

    # Case A: Blessed flash
    item_a = NormalizedItem(uid="ring1", raw_str="a gold ring", current_letter="c", buc_state="UNKNOWN")
    inv.active_items["ring1"] = item_a
    worker.on_altar_test_result("ring1", "An amber flash surrounds the gold ring!", blstats, inv)
    assert item_a.buc_state == "BLESSED"
    assert "ring1" in worker.tested_uids

    # Case B: Cursed flash
    item_b = NormalizedItem(uid="mail1", raw_str="a chain mail", current_letter="d", buc_state="UNKNOWN")
    inv.active_items["mail1"] = item_b
    worker.on_altar_test_result("mail1", "A flash of black light surrounds the chain mail.", blstats, inv)
    assert item_b.buc_state == "CURSED"

    # Case C: Uncursed (no flash)
    item_c = NormalizedItem(uid="boots1", raw_str="iron shoes", current_letter="e", buc_state="UNKNOWN")
    inv.active_items["boots1"] = item_c
    worker.on_altar_test_result("boots1", "You drop the iron shoes.", blstats, inv)
    assert item_c.buc_state == "UNCURSED"


def test_epistemic_worker_altar_sacrifice():
    ep_mgr = EpistemicManager()
    worker = EpistemicWorker(ep_mgr)
    inv = InventoryNormalizer()
    blstats = make_dummy_blstats(y=7, x=7, turn=120)
    chars = np.full((21, 79), ord("."), dtype=np.int32)
    lvl_altars = {(7, 7)}

    # Fresh safe corpse (age = 120 - 100 = 20 <= 50)
    fresh_corpse = NormalizedItem(
        uid="corpse1",
        raw_str="a goblin corpse",
        current_letter="f",
        first_seen_turn=100,
    )
    inv.active_items["corpse1"] = fresh_corpse

    task = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
    )
    assert task is not None
    assert task.name == "OFFER"
    assert task.args["slot"] == "f"

    # Rotten / stale corpse (age = 120 - 50 = 70 > 50) -> Should NOT offer
    stale_corpse = NormalizedItem(
        uid="corpse2",
        raw_str="a giant rat corpse",
        current_letter="g",
        first_seen_turn=50,
    )
    inv.active_items.clear()
    inv.active_items["corpse2"] = stale_corpse

    task_stale = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
    )
    assert task_stale is None

    # Unsafe domestic dog corpse -> Should NOT offer
    dog_corpse = NormalizedItem(
        uid="corpse3",
        raw_str="a puppy corpse",
        current_letter="h",
        first_seen_turn=110,
    )
    inv.active_items.clear()
    inv.active_items["corpse3"] = dog_corpse

    task_dog = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
    )
    assert task_dog is None


def test_epistemic_worker_wand_engrave_testing():
    ep_mgr = EpistemicManager()
    worker = EpistemicWorker(ep_mgr)
    inv = InventoryNormalizer()
    blstats = make_dummy_blstats(y=4, x=4, hp=20, max_hp=20)
    chars = np.full((21, 79), ord("."), dtype=np.int32)
    lvl_altars = set()

    # Uncursed wand candidate
    wand = NormalizedItem(
        uid="w1",
        raw_str="a balsa wand",
        current_letter="z",
        buc_state="UNCURSED",
    )
    inv.active_items["w1"] = wand
    belief = ep_mgr.get_or_create_belief(wand)
    belief.candidate_identities = ["wand of digging", "wand of sleep", "wand of death"]
    belief.collapse_buc("UNCURSED")

    task = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
        has_adjacent_hostiles=False,
        visible_monster_count=0,
    )
    assert task is not None
    assert task.name == "ENGRAVE_WAND"
    assert task.args["slot"] == "z"

    # Observation update
    worker.on_wand_engrave_result("w1", "The gravel flies up!", inv)
    assert "w1" in worker.engraved_wand_uids
    assert belief.is_formally_identified()
    assert belief.candidate_identities == ["wand of digging"]
    assert "wand of digging" in wand.raw_str

    # Cursed wand explosion safety check: cursed wand must NOT be engrave-tested
    cursed_wand = NormalizedItem(
        uid="w2",
        raw_str="an iron wand",
        current_letter="y",
        buc_state="CURSED",
    )
    inv.active_items["w2"] = cursed_wand
    belief_c = ep_mgr.get_or_create_belief(cursed_wand)
    belief_c.collapse_buc("CURSED")

    task_cursed = worker.evaluate_epistemic_turn(
        blstats=blstats,
        chars=chars,
        inv_tracker=inv,
        lvl_altars=lvl_altars,
        has_adjacent_hostiles=False,
        visible_monster_count=0,
    )
    # Cursed wand must be blocked!
    assert task_cursed is None


def test_navigation_altar_targeting():
    nav = NavigationManager()
    chars = np.full((21, 79), ord("."), dtype=np.int32)
    chars[8, 15] = ord("_")  # Altar at (8, 15)
    blstats = make_dummy_blstats(y=5, x=5)

    # With target_altar=True, should path toward (8, 15)
    task = nav.evaluate_navigation_turn(
        chars=chars,
        blstats=blstats,
        target_altar=True,
    )
    assert task is not None
    assert task.name == "STEP"
    dr, dc = task.args["delta"]
    # Path step should move closer to (8, 15)
    assert (dr > 0) or (dc > 0)


class MockAction:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


class MockEnv:
    def __init__(self):
        # Create standard actions
        self.actions = [
            MockAction("DROP", ord("d")),
            MockAction("PICKUP", ord(",")),
            MockAction("OFFER", 239),
            MockAction("ENGRAVE", ord("E")),
            MockAction("MORE", 13),
            MockAction("WAIT", ord(".")),
            MockAction("b", ord("b")),
            MockAction("f", ord("f")),
            MockAction("z", ord("z")),
            MockAction("x", ord("x")),
            MockAction("SPACE", ord(" ")),
        ]
        self.unwrapped = self
        self.history = []

    def step(self, action_idx: int):
        act = self.actions[action_idx]
        self.history.append(act.name)
        obs = {
            "blstats": np.zeros(27, dtype=np.int64),
            "chars": np.zeros((21, 79), dtype=np.int32),
            "glyphs": np.zeros((21, 79), dtype=np.int32),
        }
        info = {
            "blstats": BottomLineStats.from_blstats(obs["blstats"]),
            "full_message": "An amber flash surrounds the item.",
        }
        return obs, 0.0, False, False, info


def test_dispatcher_epistemic_actions():
    mock_env = MockEnv()
    dispatcher = ActionDispatcher(mock_env)

    # 1. ALTAR_TEST
    mock_env.history.clear()
    task = Task("ALTAR_TEST", is_primitive=True, args={"slot": "b"})
    obs, r, term, trunc, info = dispatcher.dispatch(task)
    assert mock_env.history == ["DROP", "b", "PICKUP"]
    assert "altar_flash" in info

    # 2. OFFER
    mock_env.history.clear()
    task = Task("OFFER", is_primitive=True, args={"slot": "f"})
    dispatcher.dispatch(task)
    assert mock_env.history == ["OFFER", "f"]

    # 3. ENGRAVE_WAND
    mock_env.history.clear()
    task = Task("ENGRAVE_WAND", is_primitive=True, args={"slot": "z"})
    dispatcher.dispatch(task)
    assert "ENGRAVE" in mock_env.history
    assert "z" in mock_env.history

