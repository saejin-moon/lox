#!/usr/bin/env python3
"""
LOX MiniHack Runner: High-throughput policy evaluation on MiniHack mazes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.core.spatial import SpatialEngine
from lox.core.tree import Blackboard
from lox.core.types import Action, Status
from lox.dsl.compiler import compile_policy
from lox.envs.minihack import MiniHackAdapter

WALKABLE_CHARS = {
    ord("."),
    ord("#"),
    ord("+"),
    ord("'"),
    ord("@"),
    ord(">"),
    ord("<"),
    ord("$"),
}


def get_walkable(chars: np.ndarray) -> np.ndarray:
    mask = np.zeros(chars.shape, dtype=bool)
    for c in WALKABLE_CHARS:
        mask |= chars == c
    return mask


def run_minihack_eval(
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    episodes: int = 10,
    max_steps: int = 200,
) -> dict:
    adapter = MiniHackAdapter(task=task)

    def handle_step_to_stairs(bb: Blackboard, args):
        chars = bb.obs.chars
        hero_pos = (bb.obs.hero.y, bb.obs.hero.x)
        walkable = get_walkable(chars)
        stairs_loc = np.argwhere(chars == ord(">"))
        if len(stairs_loc) > 0:
            target = (int(stairs_loc[0, 0]), int(stairs_loc[0, 1]))
            path = SpatialEngine.find_path(hero_pos, target, walkable)
            if path:
                dy = path[0][0] - hero_pos[0]
                dx = path[0][1] - hero_pos[1]
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_step_to_frontier(bb: Blackboard, args):
        chars = bb.obs.chars
        hero_pos = (bb.obs.hero.y, bb.obs.hero.x)
        walkable = get_walkable(chars)
        frontier = SpatialEngine.find_nearest_frontier(
            hero_pos, walkable, adapter.visited
        )
        if frontier is not None:
            path = SpatialEngine.find_path(hero_pos, frontier, walkable)
            if path:
                dy = path[0][0] - hero_pos[0]
                dx = path[0][1] - hero_pos[1]
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    policy_code = """
def reach_goal():
    if stairs_down_known:
        step_to_stairs_down()

def explore():
    step_to_frontier()

plan = [
    reach_goal,
    explore,
]
"""
    tree = compile_policy(
        policy_code,
        action_handlers={
            "step_to_stairs_down": handle_step_to_stairs,
            "step_to_frontier": handle_step_to_frontier,
        },
    )

    solved_count = 0
    total_steps = 0
    success_steps = []
    t0 = time.perf_counter()

    for ep in range(episodes):
        obs = adapter.reset(seed=ep + 1000)
        ep_steps = 0
        ep_solved = False

        while ep_steps < max_steps:
            chars = obs.chars
            stairs_known = bool(np.any(chars == ord(">")))
            memory = {"stairs_down_known": stairs_known, "has_unvisited_frontier": True}

            action = tree.execute(obs, memory=memory) or Action(name="search")
            obs, reward, terminated, truncated, _ = adapter.step(action)
            ep_steps += 1

            if reward > 0.5 or (terminated and not truncated):
                ep_solved = True
                break

        total_steps += ep_steps
        if ep_solved:
            solved_count += 1
            success_steps.append(ep_steps)

        status_str = "SOLVED" if ep_solved else "FAILED"
        print(f"Episode {ep + 1:2d}/{episodes}: {status_str} in {ep_steps:3d} steps")

    elapsed = time.perf_counter() - t0
    adapter.close()

    sps = total_steps / max(1e-4, elapsed)
    success_rate = (solved_count / episodes) * 100.0
    mean_steps = float(np.mean(success_steps)) if success_steps else 0.0

    print("\n" + "=" * 50)
    print(f"MiniHack Evaluation: {task}")
    print(
        f"Episodes: {episodes} | Solved: {solved_count}/{episodes} ({success_rate:.1f}%)"
    )
    print(f"Mean Steps on Success: {mean_steps:.1f}")
    print(f"Total Time: {elapsed:.2f}s | Throughput: {sps:.1f} steps/second")
    print("=" * 50)

    return {
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "sps": sps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="MiniHack-ExploreMaze-Easy-Mapped-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    run_minihack_eval(args.task, args.episodes, args.max_steps)
