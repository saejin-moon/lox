"""
Tests for the 1-to-1 NetHackHarness, Action mapping, lazy GameState queries,
Numba BFS/A* pathfinding, deterministic replay_bytes, and stepping throughput.
"""

from __future__ import annotations

import time
import numpy as np

from lox.env.actions import ACTION_BY_CHAR, DIR_DELTAS, Action, key, normalize_action_to_bytes, seq
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

    # Sequence resolution
    s = seq(Action.ENGRAVE, "-", "Elbereth\r")
    assert s.bytes_seq[:2] == (ord("E"), ord("-"))
    assert len(s.bytes_seq) == 2 + len("Elbereth\r")

    # normalize_action_to_bytes handles Action, str, int, tuple/list, None
    assert normalize_action_to_bytes(Action.WAIT) == ((ord("."),), "WAIT")
    assert normalize_action_to_bytes("s") == ((ord("s"),), "SEARCH")
    assert normalize_action_to_bytes(None) == ((ord("."),), "WAIT")
    raw_seq, name = normalize_action_to_bytes([Action.OPEN, Action.E])
    assert raw_seq == (ord("o"), ord("l"))
    assert "seq(" in name
    assert DIR_DELTAS[(-1, 0)] == Action.N
    assert ACTION_BY_CHAR[ord(".")] == "WAIT"


def test_gamestate_queries_and_pathfinding():
    harness = NetHackHarness(max_steps=200)
    try:
        state = harness.reset(seed=42)
        assert state.depth == 1
        assert state.hp > 0
        assert state.max_hp >= state.hp
        assert 0 <= state.y < 21
        assert 0 <= state.x < 79
        assert state.char_at(state.y, state.x) != ""
        assert len(state.ascii_map().splitlines()) == 21

        # Inventory should be populated from initial reset
        inv = state.inventory()
        assert len(inv) > 0
        assert any(it.letter == "a" for it in inv)

        # Monster and object queries
        mons = state.visible_monsters()
        objs = state.visible_objects()
        assert isinstance(mons, list)
        assert isinstance(objs, list)

        # BFS distances and A* path step
        passable = (
            (state.chars == ord("."))
            | (state.chars == ord("#"))
            | (state.chars == ord(">"))
            | (state.chars == ord("<"))
            | (state.chars == ord("+"))
        )
        dists = state.bfs_distances(passable)
        assert dists.shape == (21, 79)
        assert dists[state.y, state.x] == 0

        # Find any reachable tile at distance >= 1 and verify path_step returns a valid step
        reachable = np.argwhere(dists > 0)
        if len(reachable) > 0:
            ty, tx = int(reachable[0][0]), int(reachable[0][1])
            dy, dx = state.path_step(passable, (ty, tx))
            assert (dy, dx) != (0, 0)
            assert abs(dy) <= 1 and abs(dx) <= 1
    finally:
        harness.close()


def test_deterministic_replay_bytes_and_throughput():
    harness = NetHackHarness(max_steps=500)
    try:
        st = harness.reset(seed=12345)
        actions = [Action.SEARCH, Action.WAIT, Action.EAST, Action.WEST, Action.SOUTH, Action.NORTH]
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
