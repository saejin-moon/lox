"""Domain-scoped tests for LOX."""

from __future__ import annotations

import time

import numpy as np

from lox.author.tools import AuthorToolRegistry
from lox.db.schema import init_db
from lox.env.actions import (
    ACTION_BY_CHAR,
    DIR_DELTAS,
    Action,
    extcmd,
    key,
    normalize_action_to_bytes,
    repeat,
    seq,
)
from lox.env.harness import NetHackHarness


def test_one_to_one_action_mapping():
    assert int(Action.N) == ord("k")
    assert int(Action.NORTH) == ord("k")
    assert int(Action.S) == ord("j")
    assert int(Action.WAIT) == ord(".")
    assert int(Action.DOWN) == ord(">")
    assert int(Action.UP) == ord("<")
    assert int(Action.SEARCH) == ord("s")
    assert key("e").byte_val == ord("e")
    assert key(27).byte_val == 27

    # Additional commands from cmdhelp and wizard mode
    assert int(Action.AGAIN) == 1
    assert int(Action.REDO) == 1
    assert int(Action.INTERRUPT) == 3
    assert int(Action.PREVMSG) == 16
    assert int(Action.SUSPEND) == 26
    assert int(Action.HELP) == ord("?")
    assert int(Action.SHELL) == ord("!")
    assert int(Action.WIZDETECT) == 5

    # Sequence resolution
    s = seq(Action.ENGRAVE, "-", "Elbereth\r")
    assert s.bytes_seq[:2] == (ord("E"), ord("-"))
    assert len(s.bytes_seq) == 2 + len("Elbereth\r")

    # Repeat helper
    rep_s = repeat(10, Action.SEARCH)
    assert rep_s.bytes_seq == (ord("1"), ord("0"), ord("s"))

    # Extended command helper
    ext_p = extcmd("pray", "y")
    assert ext_p.bytes_seq[0] == ord("#")
    assert ext_p.bytes_seq[-1] == ord("y")

    # Verify harness.step actually dispatches all keystrokes in a multi-key sequence
    harness = NetHackHarness(max_steps=50)
    try:
        harness.reset(seed=42)
        st, done, _ = harness.step(seq(Action.OPEN, Action.S))
        # Both 'o' (111) and 'j' (106) must be present in action_bytes
        assert ord("o") in harness.action_bytes
        assert ord("j") in harness.action_bytes
        assert "In what direction?" in st.message
    finally:
        harness.close()

    # normalize_action_to_bytes handles Action, str, int, tuple/list, None
    assert normalize_action_to_bytes(Action.WAIT) == ((ord("."),), "WAIT")
    assert normalize_action_to_bytes("s") == ((ord("s"),), "SEARCH")
    assert normalize_action_to_bytes(None) == ((ord("."),), "WAIT")
    raw_seq, name = normalize_action_to_bytes([Action.OPEN, Action.E])
    assert raw_seq == (ord("o"), ord("l"))
    assert "seq(" in name
    assert DIR_DELTAS[(-1, 0)] == Action.N
    assert ACTION_BY_CHAR[ord(".")] == "WAIT"


def test_deterministic_replay_bytes_and_throughput():
    harness = NetHackHarness(max_steps=500)
    try:
        st = harness.reset(seed=12345)
        actions = [
            Action.SEARCH,
            Action.WAIT,
            Action.EAST,
            Action.WEST,
            Action.SOUTH,
            Action.NORTH,
        ]
        t0 = time.perf_counter()
        n_steps = 200
        for i in range(n_steps):
            st, done, info = harness.step(actions[i % len(actions)])
            if done:
                break
        elapsed = max(1e-6, time.perf_counter() - t0)
        steps_per_sec = harness.steps_taken / elapsed
        # Single-core stepping with full GameState wrapper should easily exceed 2,000 steps/sec
        assert steps_per_sec > 2000.0

        recorded_bytes = harness.action_bytes
        final_y, final_x, final_turn, final_hp = st.y, st.x, st.turn, st.hp

        # Replay exact bytes from seed 12345 and verify bit-identical end state
        replay_st = harness.replay_bytes(seed=12345, action_bytes=recorded_bytes)
        assert replay_st.y == final_y
        assert replay_st.x == final_x
        assert replay_st.turn == final_turn
        assert replay_st.hp == final_hp
    finally:
        harness.close()


def test_auto_dismiss_more_stairs_descent_and_replay():
    harness = NetHackHarness(max_steps=500)
    try:
        # Seed 10001 has downstairs visible at reset
        st = harness.reset(seed=10001)
        stairs = st.stairs_down_pos
        if stairs:
            passable = np.isin(
                st.chars,
                [
                    ord("."),
                    ord("#"),
                    ord(">"),
                    ord("<"),
                    ord("+"),
                    ord("-"),
                    ord("|"),
                ],
            )
            while st.pos != stairs and harness.step_count < 150:
                dy, dx = st.path_step(passable, stairs)
                if (dy, dx) == (0, 0):
                    break
                st, done, _ = harness.step(DIR_DELTAS[(dy, dx)])
                if done:
                    break
            if st.pos == stairs:
                st, done, _ = harness.step(Action.DOWN)
                assert st.depth == 2
                assert not st.waiting_for_space
                rec = bytes(harness.action_bytes)
                replay_st = harness.replay_bytes(seed=10001, recorded_bytes=rec)
                assert replay_st.depth == 2
    finally:
        harness.close()


def test_fatal_replay_bytes_and_step_guard():
    """Verify that replaying bytes that end in death cleanly handles finished NetHack without raising RuntimeError."""
    harness = NetHackHarness(max_steps=500)
    try:
        harness.reset(seed=123)
        for _ in range(20):
            st, done, _ = harness.step(seq("#", "p", "r", "a", "y", "\r", "y"))
            if done:
                break
        assert harness.done is True
        fatal_bytes = bytes(harness.action_bytes)

        harness2 = NetHackHarness(max_steps=500)
        try:
            st2 = harness2.replay_bytes(seed=123, recorded_bytes=fatal_bytes)
            assert st2.hp <= 0 or harness2.done is True
            assert harness2.done is True

            st3, done3, info3 = harness2.step(Action.WAIT)
            assert done3 is True
            assert info3["action_name"] == "DONE"
        finally:
            harness2.close()
    finally:
        harness.close()


def test_c_symbol_table_bitflags_and_replay_full_memory():
    """Verify 1-to-1 C permonst/objclass properties, name lookups, and full-grid replay memory."""
    harness = NetHackHarness(max_steps=200)
    try:
        st = harness.reset(seed=42)
        # Verify InventoryItem carries ObjectInfo from C objclass table
        inv = st.inventory
        assert len(inv) > 0
        assert any(it.info is not None and it.info.weight >= 0 for it in inv)

        # Verify 1-to-1 C permonst bitmask properties via monster_info_by_name
        bee = st.monster_info_by_name("a killer bee corpse")
        assert bee is not None
        assert bee.is_poisonous is True
        assert bee.conveys_poison_res is True

        acid_blob = st.monster_info_by_name("acid blob")
        assert acid_blob is not None
        assert acid_blob.is_acidic is True

        zombie = st.monster_info_by_name("human zombie")
        assert zombie is not None
        assert zombie.is_undead is True

        tengu = st.monster_info_by_name("tengu")
        assert tengu is not None
        assert tengu.conveys_teleport is True
        assert tengu.conveys_teleport_control is True

        # Verify object_info_by_name exposes oprop
        speed_boots = st.object_info_by_name("speed boots")
        assert speed_boots is not None
        assert speed_boots.oprop > 0

        # Step a few times and verify replay_bytes reconstructs full room remembered_chars (not just 3x3)
        st, _, _ = harness.step(Action.WAIT)
        live_remembered_count = int(np.count_nonzero(harness.mem.remembered_chars))
        rec_bytes = bytes(harness.action_bytes)

        harness2 = NetHackHarness(max_steps=200)
        try:
            harness2.replay_bytes(seed=42, recorded_bytes=rec_bytes)
            replay_remembered_count = int(
                np.count_nonzero(harness2.mem.remembered_chars)
            )
            assert replay_remembered_count == live_remembered_count
            assert len(harness2.step_byte_offsets) >= 2
        finally:
            harness2.close()
    finally:
        harness.close()


def test_phase1_lossless_memory_traps_prompts_and_seq_abort() -> None:
    """Phase 1: trap labels, safe_passable_mask, exits, oprop, noautopickup, getobj/getdir, seq_aborted."""
    from lox.env.harness import DEFAULT_NETHACKOPTIONS, NetHackHarness
    from lox.env.state import _IS_TRAP_LUT, _TRAP_NAME_BY_GLYPH

    assert "noautopickup" in DEFAULT_NETHACKOPTIONS
    assert "autopickup" not in DEFAULT_NETHACKOPTIONS
    trap_glyphs = [int(g) for g in np.nonzero(_IS_TRAP_LUT)[0]]
    assert len(trap_glyphs) >= 10
    trap_names = [_TRAP_NAME_BY_GLYPH[g] for g in trap_glyphs]
    assert any(
        "trap" in v.lower() or "pit" in v.lower() or "web" in v.lower()
        for v in trap_names
    )
    assert all(
        "wall" not in v.lower() and "doorway" not in v.lower() for v in trap_names
    )

    harness = NetHackHarness(max_steps=50)
    try:
        st = harness.reset(seed=42)
        tg = trap_glyphs[0]
        ty, tx = st.y, st.x
        st.glyphs[ty, tx] = tg
        st.chars[ty, tx] = ord("^")
        st.mem.update(st)
        assert bool(st.passable_mask[ty, tx]) is True
        assert bool(st.safe_passable_mask[ty, tx]) is False
        assert int(st.mem.remembered_glyphs[ty, tx]) == tg
        assert isinstance(st.worn_oprop_mask, int)
        assert isinstance(st.mem.level_exits, dict)

        st._message_cache = "What do you want to wear? [a-z or ?*]"
        orig_internal = st.internal.copy()
        st.internal[1] = 1
        st.internal[2] = 0
        assert st.in_getobj_prompt is True
        assert st.in_getdir_prompt is False
        assert st.in_yn_prompt is False

        st._message_cache = "In what direction?"
        assert st.in_getdir_prompt is True
        assert st.in_getobj_prompt is False
        assert st.in_yn_prompt is False
        st.internal[:] = orig_internal

        # Mid-sequence safety abort: ZAP when pack has no wand aborts before second byte 'z'
        bytes_before = len(harness.action_bytes)
        _, _, info = harness.step(seq(Action.ZAP, key("z")))
        assert info.get("seq_aborted") is True
        assert len(harness.action_bytes) == bytes_before + 1
    finally:
        harness.close()


def test_harness_mid_sequence_more_dismissal_and_prompt_properties():
    harness = NetHackHarness(role="val-hum-law-fem")
    state = harness.reset(seed=42)

    # Check prompt state properties
    assert hasattr(state, "in_moveloop")
    assert hasattr(state, "in_yn_prompt")
    assert hasattr(state, "in_text_prompt")
    assert state.in_moveloop is True
    assert state.in_yn_prompt is False
    assert state.in_text_prompt is False

    # Execute a multi-keystroke sequence (e.g. search 3 times)
    multi_action = seq(Action.SEARCH, Action.SEARCH, Action.SEARCH)
    next_state, done, info = harness.step(multi_action)
    assert next_state is not None
    assert harness.step_count == 1
    # Check that bytes are recorded in action_bytes
    assert len(harness.action_bytes) >= 3

    # Check deterministic replay matches position and step count
    replayed = harness.replay_bytes(seed=42, recorded_bytes=bytes(harness.action_bytes))
    assert replayed.step_count == harness.step_count
    assert replayed.y == next_state.y and replayed.x == next_state.x
    assert (
        replayed.dungeon_num == next_state.dungeon_num
        and replayed.level_num == next_state.level_num
    )


def test_world_memory_and_byte_replay() -> None:
    """3. WorldMemory searched_counts, stair_graph, and exact byte-replay alignment."""
    harness = NetHackHarness(max_steps=200)
    try:
        st0 = harness.reset(seed=12345)
        # Verify searched_counts initially 0
        assert np.all(st0.mem.searched_counts == 0)

        # Execute Action.SEARCH
        st1, _, _ = harness.step(Action.SEARCH)
        # Player position and 8-neighborhood should be 1
        assert st1.mem.searched_counts[st1.y, st1.x] == 1
        assert st1.mem.searched_counts[max(0, st1.y - 1), max(0, st1.x - 1)] == 1

        # Execute multi-byte sequence
        kick_east = seq(Action.KICK, Action.EAST)
        st2, _, _ = harness.step(kick_east)
        cut_offset = len(harness.action_bytes)
        assert cut_offset > 0

        # Simulate stair transition
        st2.mem.pos_history.append((0, 1, 10, 20))

        class MockFloor2State:
            dungeon_num = 0
            level_num = 2
            y = 12
            x = 22

        st2.mem.update(MockFloor2State())
        assert (0, 1, 10, 20) in st2.mem.stair_graph
        assert st2.mem.stair_graph[(0, 1, 10, 20)] == (0, 2, 12, 22)
        assert st2.mem.stair_graph[(0, 2, 12, 22)] == (0, 1, 10, 20)

        # Replay bytes up to cut_offset
        replayed_st = harness.replay_bytes(
            seed=12345,
            recorded_bytes=bytes(harness.action_bytes),
            stop_at_byte_idx=cut_offset,
        )
        assert replayed_st.y == st2.y
        assert replayed_st.x == st2.x
        assert replayed_st.depth == st2.depth
        assert replayed_st.turn == st2.turn
        assert replayed_st.mem.searched_counts[st1.y, st1.x] == 1
    finally:
        harness.close()


def test_more_yn_collision_opt_in_menu_linger_and_hypothesis_summary(tmp_path):
    import numpy as np

    from lox.author.swarm_deficits import DeficitAnalyzer
    from lox.env.actions import Action
    from lox.env.harness import NetHackHarness, policy_handles_menus
    from lox.env.state import GameState

    # 1. Opt-in menu linger detection
    assert policy_handles_menus("def step(self, s): return Action.WAIT", {}) is False
    assert policy_handles_menus("def step(self, s): return s.in_menu", {}) is True
    assert (
        policy_handles_menus("class Policy: pass", {"m": "if s.in_menu: pass"}) is True
    )

    h = NetHackHarness(max_steps=20, allow_menu_linger=False)
    try:
        st0 = h.reset(42)
        raw = list(h._last_raw_tuple)
        obs_keys = list(h._obs_keys)
        idx_map = dict(h._idx_map)
        assert st0.depth == 1
        st1, done1, _ = h.step(Action.INVENTORY)
        assert not done1
        # Unhandled menus are cleanly auto-dismissed when allow_menu_linger=False
        assert st1.in_menu is False
    finally:
        h.close()

    # 2. in_yn_prompt must yield False when `--More--` is on row 0 while waiting_for_space
    idx_int = idx_map["internal"]
    idx_tty = idx_map["tty_chars"]
    raw[idx_int] = raw[idx_int].copy()
    raw[idx_int][1] = 1
    raw[idx_int][3] = 1
    raw[idx_tty] = raw[idx_tty].copy()
    banner = b"You open the tin. It smells like orcs.--More--"
    raw[idx_tty][0, :] = ord(" ")
    raw[idx_tty][0, : len(banner)] = np.frombuffer(banner, dtype=np.uint8)
    collided = GameState(tuple(raw), obs_keys=obs_keys, obs_idx_map=idx_map)
    assert collided.in_yn_prompt is False

    # 3. DeficitAnalyzer._populate_recent_attempts surfaces hypothesis/cluster over binding_constraint
    dbp = str(tmp_path / "rec.duckdb")
    con = init_db(dbp)
    con.execute(
        "INSERT INTO policies (policy_id, generation, mode, policy_code, avg_depth, max_depth, is_promoted) VALUES ('p1', 1, 'SWARM', 'class Policy: pass', 4.5, 9, TRUE)"
    )
    con.execute(
        "INSERT INTO batches (batch_id, generation, mode, parent_policy_id, candidate_policy_id, episodes_recorded, episodes_discarded, avg_depth, max_depth, avg_score, hypothesis_passed) VALUES ('b1', 2, 'SWARM', 'p1', 'p2', 24, 0, 4.75, 11, 700.0, TRUE)"
    )
    con.execute(
        "INSERT INTO synthesis_logs (log_id, generation, agent_role, turn_index, turn_type, hypothesis_yaml) VALUES ('s1', 2, 'peer_1', 3, 'SMOKE_CHECK', 'binding_constraint: long boilerplate\ncluster: slice_stalled\nhypothesis: H-fix-more-yn-loop')"
    )
    rep = DeficitAnalyzer(dbp).analyze(
        generation=3, current_policy="class Policy: pass"
    )
    con.close()
    assert rep.recent_attempts
    assert "H-fix-more-yn-loop" in rep.recent_attempts[0]
    assert "binding_constraint" not in rep.recent_attempts[0]

    # 4. enforce_peer_research_gate requires at least 1 empirical lookup (query_db, query_wiki, or query_theories) before dry_run_policy

    class _FakeTheories:
        def search_theories(self, query, limit=6, doc_type=None):
            return []

    reg = AuthorToolRegistry(db_path=dbp)
    reg.theories_store = _FakeTheories()
    reg.configure_generation(generation=3, mode="SWARM_PEER")
    reg.enforce_peer_research_gate = True
    pol_code = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        return Action.SEARCH\n"
    )
    msg_r1, st_r1 = reg.dry_run_policy(pol_code, seed=42, max_turns=15)
    assert st_r1 == "RESEARCH_REQUIRED"
    assert (
        "query_db" in msg_r1 and "query_wiki" in msg_r1 and "query_theories" in msg_r1
    )

    # Running at least 1 empirical lookup (e.g. query_theories or query_db or query_wiki) unlocks dry_run_policy
    reg.query_theories("armor equip")
    _msg_r2, st_r2 = reg.dry_run_policy(pol_code, seed=42, max_turns=15)
    assert st_r2 != "RESEARCH_REQUIRED"
