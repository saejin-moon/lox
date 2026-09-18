"""
Harness to run and benchmark upstream AutoAscend against NetHackChallenge-v0 in modern gymnasium,
providing an empirical baseline for score, depth, and survival turns.

Results are consolidated into DuckDB alongside CORP telemetry so both agents can be
compared apples-to-apples, and optionally exported to structured JSON via --output.
"""

import sys
import os
import time
import argparse
import json
import gymnasium
import gym
import nle
import nle.nethack as nh
import numpy as np

# Ensure autoascend is in sys.path
sys.path.insert(0, "/home/bae/autoascend")
sys.path.insert(0, "/home/bae/corp")

from autoascend import agent as agent_lib
from autoascend.env_wrapper import EnvWrapper
from corp.telemetry import DuckDBConsolidator


class GymFromGymnasium(gym.Env):
    """Bridges modern Gymnasium 5-tuple step/2-tuple reset to legacy gym interface used by AutoAscend.

    NOTE: blstats MUST be truncated to 26 elements. AutoAscend's BLStats class has exactly 26
    fields (its layout predates NLE adding the 27th `alignment` element); passing all 27 raises
    "BLStats.__new__() takes 27 positional arguments but 28 were given". condition_bits (idx 25)
    is included in the 26 - nothing AutoAscend reads is lost.
    """
    def __init__(self, gym_env):
        self.env = gym_env
        self.action_space = gym_env.action_space
        self.observation_space = gym_env.observation_space
        self._actions = getattr(gym_env.unwrapped, 'actions', list(nh.ACTIONS))
        self._steps = 0
        self._turns = 0
        self.last_observation = None
        self._seeds = (0, 0)

    def get_seeds(self):
        return self._seeds

    def seed(self, core=None, disp=None):
        self._seeds = (core or 0, disp or 0)
        return self._seeds

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs['blstats'] = obs['blstats'][:26]  # AutoAscend BLStats arity (see class docstring)
        self._steps = 0
        self._turns = int(obs['blstats'][20])
        self.last_observation = (obs['glyphs'], obs['chars'])
        return obs

    def step(self, action):
        self._steps += 1
        obs, reward, term, trunc, info = self.env.step(action)
        obs['blstats'] = obs['blstats'][:26]  # AutoAscend BLStats arity (see class docstring)
        self._turns = int(obs['blstats'][20])
        self.last_observation = (obs['glyphs'], obs['chars'])
        return obs, reward, term or trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def run_autoascend_episode(role: str = "val", seed: int = 42, step_limit: int = 50000):
    start_time = time.time()

    # Character matching for role
    ROLE_CHARACTER_MAP = {
        "val": "val-hum-law-fem",
        "bar": "bar-hum-neu-mal",
        "sam": "sam-hum-law-mal",
        "pri": "pri-hum-fem-neu",
        "mon": "mon-hum-neu-mal",
    }
    char_str = ROLE_CHARACTER_MAP.get(role.lower(), "val-hum-law-fem")

    base_env = gymnasium.make('NetHackChallenge-v0', character=char_str)
    wrapped = GymFromGymnasium(base_env)
    wrapped.seed(seed, seed)

    env = EnvWrapper(wrapped, visualizer_args=dict(enable=False), step_limit=step_limit, interactive=False)

    try:
        env.main()
    except Exception as e:
        env.end_reason = f"exception: {e}"

    duration = time.time() - start_time
    try:
        summary = env.get_summary()
    except Exception as e:
        # env.main() teardown can null the agent before get_summary when the episode itself crashed
        summary = {
            'score': 0, 'steps': wrapped._steps, 'turns': wrapped._turns, 'level_num': 1,
            'experience_level': 1, 'milestone': 'crashed', 'panic_num': 0,
            'character': char_str, 'end_reason': f"summary-failed: {e}",
        }
    summary['duration'] = duration
    summary['character'] = char_str
    summary['seed'] = seed
    return summary


def store_in_duckdb(summaries: list[dict], db_path: str, role: str, step_limit: int) -> str:
    """Persist AutoAscend episode summaries into DuckDB (episodes table) for unified comparison."""
    run_id = f"aa_{int(time.time())}"
    try:
        consolidator = DuckDBConsolidator(db_path=db_path)
    except Exception as e:
        print(f"[telemetry] DuckDB consolidation unavailable ({e}); skipping persistence")
        return run_id

    import duckdb
    con = duckdb.connect(db_path)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS autoascend_episodes (
                run_id VARCHAR,
                role VARCHAR,
                episode INTEGER,
                seed INTEGER,
                total_turns INTEGER,
                total_steps INTEGER,
                max_depth INTEGER,
                final_score INTEGER,
                ascended BOOLEAN,
                death_message VARCHAR,
                duration_sec DOUBLE
            )
        """)
        for i, s in enumerate(summaries, 1):
            con.execute(
                "INSERT INTO autoascend_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    role,
                    i,
                    int(s.get('seed', 0)),
                    int(s.get('turns', 0)),
                    int(s.get('steps', 0)),
                    int(s.get('level_num', 1)),
                    int(s.get('score', 0)),
                    bool(s.get('ascended', False)),
                    str(s.get('end_reason', 'unknown')),
                    float(s.get('duration', 0.0)),
                ],
            )
        print(f"[telemetry] Stored {len(summaries)} AutoAscend episodes in {db_path} (run_id={run_id})")
    finally:
        con.close()
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoAscend on NetHackChallenge-v0")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--role", type=str, default="val", help="Role (val, bar, sam, pri)")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--step-limit", type=int, default=50000, help="Step budget limit")
    parser.add_argument("--output", type=str, default=None, help="Path to save structured JSON results")
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default="data/corp_telemetry.duckdb",
        help="Path to consolidated DuckDB database for unified CORP/AutoAscend telemetry",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"AutoAscend Baseline Runner: {args.episodes} episodes | Role: {args.role} | Step Limit: {args.step_limit}")
    print("=" * 80)

    results = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        summary = run_autoascend_episode(role=args.role, seed=ep_seed, step_limit=args.step_limit)
        results.append(summary)
        score_val = int(summary.get('score', 0))
        turns_val = int(summary.get('turns', 0))
        steps_val = int(summary.get('steps', 0))
        depth_val = int(summary.get('level_num', 1))
        print(
            f"Ep {i+1:2d}/{args.episodes:2d} | "
            f"Turns: {turns_val:5d} | "
            f"Depth: {depth_val:2d} | "
            f"Score: {score_val:5d} | "
            f"Steps: {steps_val:5d} | "
            f"Death: {summary.get('end_reason', 'unknown')[:35]}"
        )

    mean_score = np.mean([r.get('score', 0) for r in results])
    median_depth = np.median([r.get('level_num', 1) for r in results])
    mean_turns = np.mean([r.get('turns', 0) for r in results])
    ascension_rate = float(sum(1 for r in results if r.get('ascended')) / max(1, len(results)))
    print("=" * 80)
    print(f"AutoAscend Empirical Baseline (Role: {args.role}):")
    print(f"Mean Score: {mean_score:.1f} | Median Depth: {median_depth:.1f} | Mean Turns: {mean_turns:.1f} | Ascension: {ascension_rate*100:.1f}%")
    print("=" * 80)

    # Persist to DuckDB for unified comparison with CORP runs
    store_in_duckdb(results, args.duckdb_path, role=args.role, step_limit=args.step_limit)

    # Optionally export structured JSON
    if args.output:
        payload = {
            "agent": "autoascend",
            "role": args.role,
            "episodes": args.episodes,
            "step_limit": args.step_limit,
            "base_seed": args.seed,
            "median_depth": float(median_depth),
            "mean_score": float(mean_score),
            "mean_turns": float(mean_turns),
            "ascension_rate": ascension_rate,
            "detailed_episodes": results,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
