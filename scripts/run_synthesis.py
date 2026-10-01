#!/usr/bin/env python3
"""
LOX 2.0 Embodied Policy Synthesis Engine: Dynamic Trigger Outer Loop.
Executes real episodes in NetHack or MiniHack, streams flight telemetry to Parquet,
consolidates into DuckDB, detects real pacing stalls and fatalities, invokes the Author Agent,
and evolves the compiled Behavior Tree.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from typing import Any
import duckdb
import numpy as np
import nle.nethack as nh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resource
import signal

def _sigalrm_policy_timeout_handler(signum, frame):
    raise TimeoutError("Policy execution timed out (infinite zero-yield generator loop detected)")

try:
    signal.signal(signal.SIGALRM, _sigalrm_policy_timeout_handler)
except Exception:
    pass

# Expand stack allocation to 256MB and recursion limit to 100k
try:
    resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024 * 1024, resource.RLIM_INFINITY))
except Exception:
    pass
sys.setrecursionlimit(100000)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lox.core.types import Action, HeroState, HungerState, Observation, Status
from lox.core.spatial import SpatialEngine
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter, DIR_CHARS
from lox.envs.minihack import MiniHackAdapter
from lox.author.agent import AuthorAgent
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.consolidator import consolidate_run
from lox.telemetry.tokens import get_token_usage_summary


def run_synthesis_loop(
    provider: str = "mock",
    model: str | None = None,
    api_key: str | None = None,
    env_type: str = "nethack",
    role: str = "valkyrie",
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    max_generations: int = 500,
    eval_episodes: int = 50,
    max_turns: int = 10000,
    policy_path: str = "data/latest_policy.py",
    fresh: bool = False,
    stall_threshold: int = 80,
    cluster_threshold: int = 2,
    db_path: str = "data/lox.duckdb",
    target_depth: float | None = None,
):
    print("=" * 65)
    print("LOX 2.0 Embodied Batched Empirical Policy Synthesis Engine")
    print(f"Provider:    {provider} | Model: {model or 'default'}")
    print(f"Environment: {env_type.upper()} ({role if env_type == 'nethack' else task})")
    print(f"Database:    {db_path}")
    print(f"Config:      {max_generations} gens | {eval_episodes} eps/gen | {max_turns} max turns")
    if target_depth is not None:
        print(f"Target:      Avg Depth >= {target_depth:.1f}")
    print(f"Policy Path: {policy_path} (fresh={fresh})")
    print("=" * 65)

    run_id = f"synth_{provider}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    author = AuthorAgent(provider=provider, model=model, api_key=api_key, db_path=db_path)
    trigger_engine = DynamicTriggerEngine(stall_threshold=stall_threshold, cluster_threshold=cluster_threshold)
    recorder = FlightRecorder(capacity=100)

    # Initialize DuckDB evolved_policies table
    try:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS evolved_policies (
                generation INTEGER,
                run_id VARCHAR,
                avg_depth DOUBLE,
                max_depth INTEGER,
                avg_turns DOUBLE,
                code TEXT,
                timestamp TIMESTAMP
            )
        """)
        con.close()
    except Exception as e:
        print(f"[Warning] Failed to initialize evolved_policies table: {e}")

    # Initialize Environment
    if env_type == "nethack":
        adapter = NetHackAdapter(env_id="NetHackChallenge-v0", role=role)
    else:
        adapter = MiniHackAdapter(task=task)

    # Load or Seed policy
    os.makedirs("data/policies", exist_ok=True)
    if not fresh and os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            current_policy = f.read().strip()
        print(f"\n[Resumed Policy from {policy_path}]")
    else:
        current_policy = """
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # Emergency: self-monitored prayer and healing
            if obs.hero.hp_frac < 0.15 and (obs.hero.turn - self.last_prayer_turn >= 350):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            elif obs.hero.hp_frac < 0.30 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
            elif obs.hero.hunger_state >= HUNGRY and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # Tactical combat
            if obs.combat.adjacent_hostile:
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                elif obs.hero.hp_frac < 0.35 and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # Navigation & Exploration
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.dungeon.adjacent_closed_door:
                obs = yield open_door()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                obs = yield wait()
"""
        with open(policy_path, "w") as f:
            f.write(current_policy.strip() + "\n")
        print("\n[Initialized Seed Policy]:")
    print(current_policy.strip())

    current_tree = compile_policy(current_policy)

    for gen in range(1, max_generations + 1):
        print(f"\n--- Running Generation {gen} Evaluation ({eval_episodes} real episodes) ---")
        gen_start_t = time.perf_counter()
        trigger_fired = False
        trigger_reason = ""
        gen_depths = []
        gen_turns = []
        gen_dir_id = f"{run_id}_g{gen:03d}"
        logger = ParquetLogger(run_id=gen_dir_id, base_dir="data/telemetry", flush_interval=100)

        for ep_idx in range(eval_episodes):
            ep_id = f"{gen_dir_id}_e{ep_idx+1:03d}"
            obs = adapter.reset(seed=(gen * 1000 + ep_idx))
            ep_turns = 0
            death_reason = "active"
            max_depth_reached = obs.hero.depth if obs.hero.depth > 0 else 1
            last_valid_gold = obs.hero.gold if hasattr(obs.hero, "gold") else 0
            last_5_actions: list[str] = []
            turns_dl1 = 0
            turns_dl2 = 0
            turns_mines = 0
            ep_descents = 0
            ep_attacks = 0
            ep_searches = 0
            ep_eats = 0
            ep_prayers = 0
            inventory_at_death_str = ""

            policy_runner = current_tree.create_runner(obs)

            for step in range(max_turns):
                ep_turns += 1
                hero = obs.hero
                hy, hx = hero.y, hero.x
                max_depth_reached = max(max_depth_reached, hero.depth)
                last_valid_gold = hero.gold if hasattr(hero, "gold") else last_valid_gold

                if hero.depth == 1:
                    turns_dl1 += 1
                elif hero.depth == 2:
                    turns_dl2 += 1
                if getattr(hero, "dungeon_branch", "") == "mines":
                    turns_mines += 1

                try:
                    signal.alarm(2)
                    action = policy_runner.send(obs)
                except TimeoutError:
                    print(f"\n[Warning] Policy hung in zero-yield loop on ep {ep_id}, turn {hero.turn}. Auto-recovering via wait().")
                    action = Action(name="wait")
                finally:
                    signal.alarm(0)

                if action is None:
                    action = Action(name="search")

                if action.name == "descend":
                    ep_descents += 1
                elif "attack" in action.name:
                    ep_attacks += 1
                elif action.name == "search":
                    ep_searches += 1
                elif "eat" in action.name:
                    ep_eats += 1
                elif action.name == "pray":
                    ep_prayers += 1

                last_5_actions.append(action.name)
                if len(last_5_actions) > 5:
                    last_5_actions.pop(0)

                tile_type = getattr(obs.dungeon, "tile_type", "room") if hasattr(obs, "dungeon") else "room"
                closest_name = getattr(obs.combat, "closest_hostile_name", "") if hasattr(obs, "combat") else ""
                closest_dist = getattr(obs.combat, "closest_hostile_dist", 99.0) if hasattr(obs, "combat") else 99.0
                hostiles_fov = getattr(obs.combat, "hostile_count_fov", 0) if hasattr(obs, "combat") else 0
                dungeon_branch = getattr(obs.hero, "dungeon_branch", "dungeon") if hasattr(obs, "hero") else "dungeon"

                recorder.record_turn(
                    turn=hero.turn,
                    depth=hero.depth,
                    hp=hero.hp,
                    max_hp=hero.max_hp,
                    hunger=hero.hunger_state.name,
                    pos=(hy, hx),
                    action_name=action.name,
                    message=obs.message,
                    closest_hostile_name=closest_name,
                    closest_hostile_dist=closest_dist,
                    hostiles_in_fov=hostiles_fov,
                    tile_type=tile_type,
                    dungeon_branch=dungeon_branch,
                )

                logger.log_tick(
                    episode_id=ep_id,
                    turn=hero.turn,
                    depth=hero.depth,
                    hp=hero.hp,
                    max_hp=hero.max_hp,
                    hunger=hero.hunger_state.name,
                    y=hy,
                    x=hx,
                    action=action.name,
                    message=obs.message,
                    reward=0.0,
                    closest_hostile_name=closest_name,
                    closest_hostile_dist=closest_dist,
                    hostiles_in_fov=hostiles_fov,
                    tile_type=tile_type,
                    dungeon_branch=dungeon_branch,
                )

                step_res = adapter.step(action)
                if len(step_res) == 5:
                    obs, reward, term, trunc, info = step_res
                    done = bool(term or trunc)
                else:
                    obs, reward, done, info = step_res
                    trunc = (step >= max_turns - 1)
                    term = done and not trunc

                if done:
                    if trunc or step >= max_turns - 1:
                        death_reason = f"MaxTurnsReached (Depth {max_depth_reached})"
                    elif getattr(obs.hero, "hp", 0) <= 0 or term:
                        msg = getattr(obs, "message", "").strip()
                        msg_l = msg.lower()
                        if "starv" in msg_l:
                            death_reason = "Starvation"
                        elif "chok" in msg_l:
                            death_reason = "Choked on food"
                        elif "poison" in msg_l:
                            death_reason = "Poison"
                        elif "petrif" in msg_l:
                            death_reason = "Petrification"
                        elif any(k in msg_l for k in ("bites", "stings", "hits", "kills", "killed", "claws", "shoots", "strikes")):
                            death_reason = msg.split(".")[0][:45] if msg else "Combat fatality"
                        elif msg:
                            death_reason = msg[:45]
                        else:
                            death_reason = "Killed in combat"
                    else:
                        death_reason = info.get("death_reason", "died") if hasattr(info, "get") else "ended"

                    inventory_items = [f"{it.name} ({it.category})" for it in obs.inventory] if hasattr(obs, "inventory") else []
                    inventory_at_death_str = ", ".join(inventory_items[:10])
                    recorder.record_death(death_reason)
                    break

            final_depth = max_depth_reached
            final_score = last_valid_gold + (final_depth * 100)
            gen_depths.append(final_depth)
            gen_turns.append(ep_turns)

            logger.log_episode(
                episode_id=ep_id,
                depth=final_depth,
                score=final_score,
                turns=ep_turns,
                death_reason=death_reason,
                solved=False,
                wall_sec=time.perf_counter() - gen_start_t,
                role=role if env_type == "nethack" else "minihack",
                gold=last_valid_gold,
                max_depth=final_depth,
                steps=ep_turns,
                attacks=ep_attacks,
                descents=ep_descents,
                searches=ep_searches,
                eats=ep_eats,
                prayers=ep_prayers,
                death_category=death_reason[:30],
                inventory_at_death=inventory_at_death_str,
                last_5_actions=" -> ".join(last_5_actions),
                turns_dl1=turns_dl1,
                turns_dl2=turns_dl2,
                turns_mines=turns_mines,
            )

        # Consolidate raw telemetry into DuckDB so LLM tools query live empirical state
        logger.flush()
        consolidate_run(run_id=gen_dir_id, db_path=db_path, telemetry_dir="data/telemetry", cleanup=True)

        avg_d = float(np.mean(gen_depths))
        max_d = int(np.max(gen_depths))
        avg_t = float(np.mean(gen_turns))

        # Query DuckDB for empirical mortality taxonomy across the generation batch
        top_deaths = []
        recent_fatal_samples = []
        try:
            con = duckdb.connect(db_path, read_only=True)
            top_deaths = con.execute(f"""
                SELECT death_reason, count(*) as count, round(count(*) * 100.0 / {eval_episodes}, 1) as pct
                FROM episodes
                WHERE episode_id LIKE '{gen_dir_id}_%'
                GROUP BY death_reason
                ORDER BY count DESC
                LIMIT 5
            """).fetchall()
            recent_fatal_samples = con.execute(f"""
                SELECT death_reason, depth, turns, inventory_at_death, last_5_actions
                FROM episodes
                WHERE episode_id LIKE '{gen_dir_id}_%' AND death_reason NOT IN ('active', 'MaxTurnsReached')
                ORDER BY rowid DESC
                LIMIT 3
            """).fetchall()
            con.close()
        except Exception:
            pass

        death_summary_str = ", ".join(f"{r[0]} ({r[2]}%)" for r in top_deaths) if top_deaths else "None"
        primary_cause = top_deaths[0][0] if top_deaths else "Floor Stagnation"
        primary_pct = top_deaths[0][2] if top_deaths else 0.0

        print(f"\n[Gen {gen} Batch Metrics ({eval_episodes} eps)] Avg Depth: {avg_d:.2f} | Max Depth: {max_d} | Avg Turns: {avg_t:.1f}")
        print(f"[Gen {gen} Ranked Fatalities] {death_summary_str}")

        if target_depth is not None and avg_d >= target_depth:
            print("\n" + "=" * 65)
            print(f"[CAMPAIGN GOAL ACHIEVED] Generation {gen} reached target average depth {avg_d:.2f} >= {target_depth:.2f}!")
            print(f"Max Depth: {max_d} | Average Turns: {avg_t:.1f}")
            print("=" * 65)
            break

        trigger_reason = (
            f"Generation {gen} Empirical Incident Autopsy ({eval_episodes} episodes): "
            f"Avg Depth {avg_d:.2f}, Max Depth {max_d}, Avg Turns {avg_t:.1f}. "
            f"#1 Mortality Bottleneck: '{primary_cause}' ({primary_pct}% of runs). "
            f"Synthesize an evolved policy program that directly mitigates this #1 cause of death, optimizes stair navigation, and breaks through deeper dungeon levels."
        )

        mortality_lines = [
            f"  {idx}. {r[0]}: {r[1]}/{eval_episodes} runs ({r[2]}%)"
            for idx, r in enumerate(top_deaths, 1)
        ]
        status_rep = (
            f"Batch Size: {eval_episodes} real episodes\n"
            f"Average Depth: {avg_d:.2f}\n"
            f"Max Depth: {max_d}\n"
            f"Average Turns Survived: {avg_t:.1f}\n\n"
            f"Ranked Causes of Death (Ranked by Popularity/Frequency):\n"
            + ("\n".join(mortality_lines) if mortality_lines else "  - None recorded")
        )
        if recent_fatal_samples:
            sample_lines = [
                f"  - Incident: \"{s[0]}\" at Depth {s[1]}, Turn {s[2]}\n    Inventory at Death: {s[3] or 'Empty'}\n    Action Sequence: {s[4] or 'N/A'}"
                for s in recent_fatal_samples
            ]
            status_rep += "\n\nRecent Fatal Incident Logs:\n" + "\n".join(sample_lines)

        recent_trajectory = recorder.get_last_10_turns_trajectory()
        if recent_trajectory:
            status_rep += f"\n\nPre-Death Diagnostic Trace:\n{recent_trajectory}"

        print(f"\n[Author Agent] Initiating empirical synthesis session (querying DuckDB & evaluating)...")
        new_code, tree, error = author.synthesize_policy(
            current_policy=current_policy,
            trigger_reason=trigger_reason,
            status_report=status_rep,
            run_id=run_id,
        )

        if error:
            print(f"[Validation Failed] {error}")
            print(f"[Rejected Candidate Code]:\n{new_code.strip()}\n")
        else:
            print(f"[Policy Verified & Compiled! Generation {gen} accepted]")
            current_policy = new_code
            current_tree = tree
            print("\nEvolved Policy Program:")
            print(new_code.strip())

            # Persist latest policy and archive checkpoint
            # Persist latest policy and archive checkpoint
            with open(policy_path, "w") as f:
                f.write(new_code.strip() + "\n")
            ckpt_path = f"data/policies/gen_{gen:04d}.py"
            with open(ckpt_path, "w") as f:
                f.write(new_code.strip() + "\n")
            print(f"[Checkpoint Saved] -> {policy_path} & {ckpt_path}")

            # Log to DuckDB evolved_policies table with retry shield
            for attempt in range(3):
                try:
                    con = duckdb.connect(db_path)
                    con.execute(
                        "INSERT INTO evolved_policies VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [gen, run_id, float(avg_d), int(max_d), float(avg_t), new_code.strip(), datetime.datetime.now()],
                    )
                    con.close()
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[Warning] Could not record evolved policy in DuckDB: {e}")
                    time.sleep(0.5)

    adapter.close()

    # Final report of token expenditure
    token_stats = get_token_usage_summary(run_id=run_id, db_path=db_path)
    print("\n" + "=" * 65)
    print("Synthesis Session Token & Cost Accounting:")
    print(f"- LLM Synthesis Sessions: {token_stats['total_calls']}")
    print(f"- Prompt Tokens:          {token_stats['prompt_tokens']:,}")
    print(f"- Completion Tokens:      {token_stats['completion_tokens']:,}")
    print(f"- Total Tokens:           {token_stats['total_tokens']:,}")
    print(f"- Estimated Cost:         ${token_stats['total_cost_usd']:.4f} USD")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None, help="Explicit API key (overrides environment and .env)")
    parser.add_argument("--env", default="nethack", choices=["nethack", "minihack"], help="Execution environment")
    parser.add_argument("--role", default="valkyrie", help="Hero role for NetHack")
    parser.add_argument("--task", default="MiniHack-ExploreMaze-Easy-Mapped-v0", help="Task for MiniHack")
    parser.add_argument("--generations", type=int, default=500, help="Total synthesis generations")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Real evaluation episodes per generation batch")
    parser.add_argument("--max-turns", type=int, default=10000, help="Max turns per episode")
    parser.add_argument("--policy-path", default="data/latest_policy.py", help="Path to persisted latest policy")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start instead of resuming latest policy")
    parser.add_argument("--stall-threshold", type=int, default=80, help="Turns without progress to trigger stall autopsy")
    parser.add_argument("--cluster-threshold", type=int, default=2, help="Deaths of same cause to trigger cluster autopsy")
    parser.add_argument("--db-path", default="data/lox.duckdb")
    parser.add_argument("--target-depth", type=float, default=None, help="Target average depth to reach before stopping")
    args = parser.parse_args()

    run_synthesis_loop(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        env_type=args.env,
        role=args.role,
        task=args.task,
        max_generations=args.generations,
        eval_episodes=args.eval_episodes,
        max_turns=args.max_turns,
        policy_path=args.policy_path,
        fresh=args.fresh,
        stall_threshold=args.stall_threshold,
        cluster_threshold=args.cluster_threshold,
        db_path=args.db_path,
        target_depth=args.target_depth,
    )
