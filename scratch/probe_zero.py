"""Probe: catch zero-turn decision loops (task + message) in full-budget episodes."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

from corp.env.nle_wrapper import make_env
from corp.agent.corp_agent import CORPAgent
from corp.planner.nogood import NogoodStore
from corp.deliberative.providers.mock_provider import MockProvider


def run_one(seed: int):
    env = make_env(character="val-hum-law-fem")
    agent = CORPAgent(
        env=env,
        nogood_store=NogoodStore(),
        llm_provider=MockProvider(),
        eval_type="debug",
        mode="random",
        seed=seed,
        role="valkyrie",
    )

    zero_tasks = Counter()
    zero_msgs = Counter()
    zero_runs = 0
    cur_zero = 0
    orig_step = agent.step

    def wrapped_step():
        task_before = agent.select_action  # noqa: F841 (not used; see below)
        prev_turn = agent.current_blstats.turn if agent.current_blstats else -1
        # peek at the task that will be selected by patching select_action once
        out = orig_step()
        new_turn = agent.current_blstats.turn if agent.current_blstats else -1
        nonlocal cur_zero, zero_runs
        if new_turn == prev_turn:
            cur_zero += 1
            zero_tasks[out.get("task", "?") if isinstance(out, dict) else "?"] += 1
            msg = agent.current_message[:60] if agent.current_message else ""
            zero_msgs[msg] += 1
        else:
            if cur_zero > 4:
                zero_runs += 1
            cur_zero = 0
        return out

    # capture task names: patch dispatcher instead
    orig_dispatch = agent.dispatcher.dispatch
    last_task = {"name": "?"}

    def wrapped_dispatch(task, *a, **kw):
        last_task["name"] = task.name
        last_task["args"] = dict(task.args) if task.args else {}
        return orig_dispatch(task, *a, **kw)

    agent.dispatcher.dispatch = wrapped_dispatch

    def wrapped_step2():
        prev_turn = agent.current_blstats.turn if agent.current_blstats else -1
        out = orig_step()
        new_turn = agent.current_blstats.turn if agent.current_blstats else -1
        nonlocal cur_zero
        if new_turn == prev_turn:
            cur_zero += 1
            zero_tasks[last_task["name"]] += 1
            msg = agent.current_message[:60] if agent.current_message else "(empty)"
            zero_msgs[f"{last_task['name']} | {msg}"] += 1
        else:
            cur_zero = 0
        return out

    result = agent.run_episode(max_steps=20000)
    # run_episode resets the agent; instead we instrument post-hoc via a second run
    return result, zero_tasks, zero_msgs


def main():
    # Simpler: monkeypatch step wrapper around agent before run
    for seed in (42, 43):
        env = make_env(character="val-hum-law-fem")
        agent = CORPAgent(
            env=env,
            nogood_store=NogoodStore(),
            llm_provider=MockProvider(),
            eval_type="debug",
            mode="random",
            seed=seed,
            role="valkyrie",
        )
        zero_tasks = Counter()
        zero_msgs = Counter()
        last_task = {"name": "?"}
        orig_dispatch = agent.dispatcher.dispatch

        def wrapped_dispatch(task, *a, **kw):
            last_task["name"] = task.name
            return orig_dispatch(task, *a, **kw)

        agent.dispatcher.dispatch = wrapped_dispatch

        orig_step = agent.step
        state = {"prev_turn": -1, "prev_pos": None}

        def wrapped_step():
            prev_turn = agent.current_blstats.turn if agent.current_blstats else -1
            out = orig_step()
            new_turn = agent.current_blstats.turn if agent.current_blstats else -1
            if new_turn == prev_turn and prev_turn >= 0:
                zero_tasks[last_task["name"]] += 1
                msg = agent.current_message[:60] if agent.current_message else "(empty)"
                zero_msgs[f"{last_task['name']} | {msg}"] += 1
                if last_task["name"] == "STEP" and zero_tasks["STEP"] % 200 == 0:
                    bl = agent.current_blstats
                    lvl = agent.nav_mgr.levels.get((bl.dungeon_number, bl.level_number))
                    door_info = ""
                    if lvl is not None:
                        y, x = bl.y, bl.x
                        door_info = f" doors={sorted(lvl.doors)[:6]} att={ {k: v for k, v in list(lvl.door_attempts.items())[:6]} } blk={sorted(lvl.blocked_tiles)[:6]}"
                    print(f"    ZERO-STEP #{zero_tasks['STEP']} pos=({bl.y},{bl.x}) depth={bl.depth} dnum={bl.dungeon_number} chars_here={chr(agent.current_chars[bl.y, bl.x]) if agent.current_chars is not None else '?'}{door_info}", flush=True)
            return out

        agent.step = wrapped_step
        result = agent.run_episode(max_steps=20000)
        print(
            f"seed={seed}: turns={result.turns} depth={result.max_depth} score={result.final_score} "
            f"zero_decisions={sum(zero_tasks.values())}",
            flush=True,
        )
        for t, c in zero_tasks.most_common(5):
            print(f"   task {t}: {c}")
        for m, c in zero_msgs.most_common(5):
            print(f"   {c:6d}x {m}")


if __name__ == "__main__":
    main()
