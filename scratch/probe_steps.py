"""Probe: find which actions consume anomalous numbers of env.step calls."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corp.env.nle_wrapper import make_env
from corp.agent.corp_agent import CORPAgent
from corp.planner.nogood import NogoodStore
from corp.deliberative.providers.mock_provider import MockProvider


def main():
    env = make_env(character="val-hum-law-fem")
    agent = CORPAgent(
        env=env,
        nogood_store=NogoodStore(),
        llm_provider=MockProvider(),
        eval_type="debug",
        mode="random",
        seed=47,
        role="valkyrie",
    )

    # Wrap the inner env's step to count calls per turn
    inner = env.env if hasattr(env, "env") else env
    counts = {"n": 0}
    orig_step = inner.step

    def counting_step(action):
        counts["n"] += 1
        return orig_step(action)

    inner.step = counting_step

    # Log per-decision consumption every 50 decisions
    orig_select = agent.select_action
    decisions = {"i": 0, "last": 0}

    def logged_select():
        task = orig_select()
        decisions["i"] += 1
        if decisions["i"] % 50 == 0:
            rate = (counts["n"] - decisions["last"]) / 50.0
            decisions["last"] = counts["n"]
            bl = agent.current_blstats
            print(f"[dec {decisions['i']:5d}] depth={bl.depth} turn={bl.turn} envsteps={counts['n']} avg={rate:.1f}", flush=True)
        return task

    agent.select_action = logged_select
    result = agent.run_episode(max_steps=8000)
    print(f"END: turns={result.turns} depth={result.max_depth} score={result.final_score} total_envsteps={counts['n']}")


if __name__ == "__main__":
    main()
