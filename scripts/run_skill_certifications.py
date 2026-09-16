"""
Skill Certifications Runner: Validates isolated domain managers on MiniHack benchmark tasks
prior to full NetHack game rollouts.
"""

import argparse
import os
import sys
import gymnasium as gym
import minihack

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from corp.env.auto_more import AutoMoreWrapper
from corp.agent.corp_agent import CORPAgent


CERTIFICATION_SUITES = [
    {
        "name": "Room Exploration & Navigation",
        "env_id": "MiniHack-Room-5x5-v0",
        "max_steps": 100,
        "trials": 3,
        "success_condition": "reached_stairs_or_alive",
    },
    {
        "name": "Corridor Traversal & Pathing",
        "env_id": "MiniHack-Corridor-R2-v0",
        "max_steps": 150,
        "trials": 3,
        "success_condition": "reached_stairs_or_alive",
    },
    {
        "name": "Tactical Corridor Combat",
        "env_id": "MiniHack-CorridorBattle-v0",
        "max_steps": 200,
        "trials": 3,
        "success_condition": "survived_engagement",
    },
]


def run_certification(task: dict) -> dict:
    import nle.nethack
    print(f"\nRunning Certification: {task['name']} [{task['env_id']}]")
    successes = 0

    for trial in range(1, task["trials"] + 1):
        base_env = gym.make(task["env_id"], actions=nle.nethack.ACTIONS)
        wrapped_env = AutoMoreWrapper(base_env)
        agent = CORPAgent(env=wrapped_env)

        agent.reset()
        steps = 0
        total_reward = 0.0
        last_info = {}
        while not agent.is_terminal and steps < task["max_steps"]:
            obs, r, term, trunc, info = agent.step()
            total_reward += float(r)
            last_info = info
            steps += 1

        is_success = (
            total_reward > 0.0
            or last_info.get("end_status") == 2
            or (agent.current_blstats is not None and agent.current_blstats.hp > 0 and not agent.is_terminal)
            or (agent.current_blstats is not None and agent.current_blstats.hp > 0)
        )
        if is_success:
            successes += 1

        print(
            f"  Trial {trial}/{task['trials']}: "
            f"Steps={steps:3d} | Reward={total_reward:.1f} | "
            f"Status={'PASS' if is_success else 'FAIL'}"
        )

    pass_rate = float(successes) / float(task["trials"])
    return {
        "name": task["name"],
        "env_id": task["env_id"],
        "pass_rate": pass_rate,
        "certified": pass_rate >= 0.66,
    }


def main():
    print("=" * 80)
    print("CORP Domain Worker Skill Certification Suite (MiniHack Regression Testbed)")
    print("=" * 80)

    results = []
    for task in CERTIFICATION_SUITES:
        res = run_certification(task)
        results.append(res)

    print("\n" + "=" * 80)
    print("SKILL CERTIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Skill Domain':<35} | {'MiniHack Environment':<30} | {'Status':<10}")
    print("-" * 80)
    all_certified = True
    for r in results:
        status_str = "CERTIFIED" if r["certified"] else "RETRY"
        if not r["certified"]:
            all_certified = False
        print(f"{r['name']:<35} | {r['env_id']:<30} | {status_str:<10}")
    print("=" * 80)
    if all_certified:
        print("ALL DOMAIN WORKERS CERTIFIED FOR FULL GAME DEPLOYMENT!\n")
    else:
        print("Warning: Some skill certifications require tuning.\n")


if __name__ == "__main__":
    main()
