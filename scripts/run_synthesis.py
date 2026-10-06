#!/usr/bin/env python3
"""
LOX Embodied Policy Synthesis Engine: Dynamic Trigger Outer Loop.
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
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import resource
import signal


def _sigalrm_policy_timeout_handler(signum, frame):
    raise TimeoutError(
        "Policy execution timed out (infinite zero-yield generator loop detected)"
    )


try:
    signal.signal(signal.SIGALRM, _sigalrm_policy_timeout_handler)
except Exception:
    pass

# Expand stack allocation to 256MB and recursion limit to 100k
try:
    resource.setrlimit(
        resource.RLIMIT_STACK, (256 * 1024 * 1024, resource.RLIM_INFINITY)
    )
except Exception:
    pass
sys.setrecursionlimit(100000)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from lox.author.agent import AuthorAgent
from lox.core.spatial import SpatialEngine
from lox.core.types import Action, HungerState
from lox.dsl.compiler import compile_policy
from lox.envs.minihack import MiniHackAdapter
from lox.envs.nethack import NetHackAdapter
from lox.telemetry.consolidator import consolidate_run
from lox.telemetry.diagnostics import RootCauseClassifier
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.tokens import get_token_usage_summary
from lox.telemetry.triggers import DynamicTriggerEngine


def _run_single_episode_worker(payload: dict[str, Any]) -> dict[str, Any]:
    gen = payload["gen"]
    ep_idx = payload["ep_idx"]
    gen_dir_id = payload["gen_dir_id"]
    policy_code = payload["policy_code"]
    env_type = payload["env_type"]
    role = payload["role"]
    task = payload["task"]
    max_turns = payload["max_turns"]
    gen_start_t = payload["gen_start_t"]

    if env_type == "nethack":
        adapter = NetHackAdapter(env_id="NetHackChallenge-v0", role=role)
    else:
        adapter = MiniHackAdapter(task=task)

    tree = compile_policy(policy_code)
    ep_id = f"{gen_dir_id}_e{ep_idx + 1:03d}"
    seed = payload.get("seed", gen * 1000 + ep_idx)
    obs = adapter.reset(seed=seed)
    ep_turns = 0
    death_reason = "active"
    max_depth_reached = obs.hero.depth if obs.hero.depth > 0 else 1
    last_valid_gold = obs.hero.gold if hasattr(obs.hero, "gold") else 0
    last_valid_score = getattr(obs.hero, "score", 0)
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
    last_known_hostile = ""
    last_valid_ac = getattr(obs.hero, "ac", 10)
    last_valid_max_hp = getattr(obs.hero, "max_hp", 15)
    ep_start_time = time.perf_counter()
    MAX_EPISODE_WALL_SEC = 90.0

    recorder = FlightRecorder(capacity=100)
    logger = ParquetLogger(
        run_id=gen_dir_id,
        base_dir="data/telemetry",
        flush_interval=5000,
        file_prefix=f"ep{ep_idx + 1:03d}",
    )

    policy_runner = tree.create_runner(obs)

    for step in range(max_turns):
        if time.perf_counter() - ep_start_time > MAX_EPISODE_WALL_SEC:
            death_reason = f"EpisodeWallTimeout ({int(MAX_EPISODE_WALL_SEC)}s)"
            break
        ep_turns += 1
        hero = obs.hero
        hy, hx = hero.y, hero.x
        max_depth_reached = max(max_depth_reached, hero.depth)
        last_valid_gold = hero.gold if hasattr(hero, "gold") else last_valid_gold
        if hasattr(hero, "score") and hero.score > 0:
            last_valid_score = hero.score

        if hero.depth == 1:
            turns_dl1 += 1
        elif hero.depth == 2:
            turns_dl2 += 1
        if getattr(hero, "dungeon_branch", "") == "mines":
            turns_mines += 1

        try:
            action = policy_runner.send(obs)
        except StopIteration:
            break
        except Exception:
            action = Action(name="wait")

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

        tile_type = (
            getattr(obs.dungeon, "tile_type", "room")
            if hasattr(obs, "dungeon")
            else "room"
        )
        closest_name = (
            getattr(obs.combat, "closest_hostile_name", "")
            if hasattr(obs, "combat")
            else ""
        )
        closest_dist = (
            getattr(obs.combat, "closest_hostile_dist", 99.0)
            if hasattr(obs, "combat")
            else 99.0
        )
        hostiles_fov = (
            getattr(obs.combat, "hostile_count_fov", 0) if hasattr(obs, "combat") else 0
        )
        dungeon_branch = (
            getattr(obs.hero, "dungeon_branch", "dungeon")
            if hasattr(obs, "hero")
            else "dungeon"
        )

        if hasattr(hero, "ac") and hero.ac is not None:
            last_valid_ac = hero.ac
        if hasattr(hero, "max_hp") and hero.max_hp > 0:
            last_valid_max_hp = hero.max_hp
        if closest_name:
            last_known_hostile = closest_name
        elif hasattr(obs, "combat") and getattr(obs.combat, "adjacent_monsters", None):
            last_known_hostile = obs.combat.adjacent_monsters[0]

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

        stairs_y = (
            adapter.known_stairs_down[0]
            if getattr(adapter, "known_stairs_down", None) is not None
            else -1
        )
        stairs_x = (
            adapter.known_stairs_down[1]
            if getattr(adapter, "known_stairs_down", None) is not None
            else -1
        )
        target_y = (
            adapter.last_target_pos[0]
            if getattr(adapter, "last_target_pos", None) is not None
            else -1
        )
        target_x = (
            adapter.last_target_pos[1]
            if getattr(adapter, "last_target_pos", None) is not None
            else -1
        )
        tiles_vis = (
            int(np.sum(adapter.visited))
            if hasattr(adapter, "visited") and adapter.visited is not None
            else 0
        )
        adj_monsters = (
            ",".join(obs.combat.adjacent_monsters)
            if hasattr(obs, "combat") and hasattr(obs.combat, "adjacent_monsters")
            else ""
        )
        act_subroutine = getattr(action, "subroutine", "") or action.name

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
            target_y=target_y,
            target_x=target_x,
            stairs_down_y=stairs_y,
            stairs_down_x=stairs_x,
            stairs_down_turn=getattr(adapter, "stairs_down_discovery_turn", -1),
            tiles_visited_count=tiles_vis,
            unvisited_frontier_count=getattr(
                obs.spatial, "unvisited_frontier_count", 0
            ),
            dead_ends_count=getattr(obs.spatial, "dead_ends_count", 0),
            adjacent_monsters=adj_monsters,
            ac=hero.ac,
            xl=hero.level,
            active_subroutine=act_subroutine,
            food_count=getattr(obs.inventory, "food_count", 0),
            potion_count=getattr(obs.inventory, "potion_count", 0),
            scroll_count=getattr(obs.inventory, "scroll_count", 0),
            dagger_count=getattr(obs.inventory, "dagger_count", 0),
            weapon_in_hand=getattr(obs.inventory, "equipped_weapon_name", ""),
        )

        step_res = adapter.step(action)
        if len(step_res) == 5:
            obs, reward, term, trunc, info = step_res
            done = bool(term or trunc)
        else:
            obs, reward, done, info = step_res
            trunc = step >= max_turns - 1
            term = done and not trunc

        if done:
            end_status = (
                info.get("end_status", None) if isinstance(info, dict) else None
            )
            is_aborted = (
                end_status is not None and getattr(end_status, "name", "") == "ABORTED"
            )

            recent_msgs = [
                s.message for s in list(getattr(recorder, "buffer", []))
            ]
            if getattr(obs.hero, "hp", 0) <= 0:
                msg = getattr(obs, "message", "").strip()
                msg_l = msg.lower()
                is_starving = (
                    "starv" in msg_l
                    or any(
                        "starv" in m.lower() or "faint from lack of food" in m.lower()
                        for m in recent_msgs
                    )
                    or getattr(obs.hero, "hunger_state", None) == HungerState.FAINTING
                )
                if is_starving:
                    death_reason = "Starvation"
                elif "chok" in msg_l or any("chok" in m.lower() for m in recent_msgs):
                    death_reason = "Choked on food"
                elif "poison" in msg_l or any(
                    "poison" in m.lower() for m in recent_msgs
                ):
                    death_reason = "Poison"
                elif (
                    "petrif" in msg_l
                    or "turn to stone" in msg_l
                    or "turning to stone" in msg_l
                    or "slowing sensation" in msg_l
                    or any(
                        "petrif" in m.lower()
                        or "turn to stone" in m.lower()
                        or "turning to stone" in m.lower()
                        or "slowing sensation" in m.lower()
                        for m in recent_msgs
                    )
                ):
                    death_reason = "Petrification"
                elif any(
                    k in msg_l
                    for k in (
                        "bites",
                        "stings",
                        "hits",
                        "kills",
                        "killed",
                        "claws",
                        "shoots",
                        "strikes",
                    )
                ):
                    death_reason = msg.split(".")[0][:45] if msg else "Combat fatality"
                elif msg:
                    death_reason = msg[:45]
                else:
                    death_reason = "Killed in combat"
            elif is_aborted:
                death_reason = f"Aborted / ZeroProgress (Depth {max_depth_reached})"
            elif step >= max_turns - 1 or trunc:
                death_reason = f"MaxTurnsReached (Depth {max_depth_reached})"
            else:
                death_reason = (
                    info.get("death_reason", "ended")
                    if hasattr(info, "get")
                    else "ended"
                )

            inventory_items = (
                [f"{it.name} ({it.category})" for it in obs.inventory]
                if hasattr(obs, "inventory")
                else []
            )
            inventory_at_death_str = ", ".join(inventory_items[:10])
            recorder.record_death(death_reason)
            break

    killer = ""
    if death_reason == "Starvation":
        killer = "starvation"
    elif death_reason == "Choked on food":
        killer = "choking"
    elif death_reason == "Poison":
        killer = "poison"
    elif death_reason == "Petrification":
        killer = "petrification"
    elif (
        obs.hero.is_dead
        or "combat" in death_reason.lower()
        or "fatality" in death_reason.lower()
    ):
        recent_msgs = [
            s.message for s in list(getattr(recorder, "buffer", []))[-30:]
        ]
        if getattr(obs, "message", ""):
            recent_msgs.append(obs.message)
        for cand in reversed(recent_msgs):
            m = cand.lower()
            if "magic missile bounces" in m and "hits you" in m:
                killer = "magic missile bounce"
                break
            if "you've been warned, knave" in m or "drop that gold" in m or "drop that money" in m:
                killer = "guard"
                break
            for hit_verb in (
                " hits you!",
                " touches you!",
                " stings you!",
                " bites you!",
                " strikes you!",
                " crushes you!",
                " zaps a ",
                " bites!",
                " hits!",
                " stings!",
                " claws!",
                " shoots!",
                " strikes!",
                " kicks!",
                " zaps!",
                " explodes!",
                " exploded!",
                " burns!",
                " freezes!",
                " poisons!",
                " thrusts ",
                " thrusts his ",
                " thrusts her ",
                " touches!",
                " crushes!",
                " swings ",
                " bites.",
                " hits.",
                " stings.",
            ):
                if hit_verb in m:
                    part = m.split(hit_verb)[0].strip()
                    part = part.split(".")[-1].split("!")[-1].strip()
                    words = part.split()
                    if words and words[0] in ("the", "a", "an"):
                        killer = " ".join(words[1:])
                    elif words:
                        killer = words[-1]
                    break
            if killer:
                break
            for k in ("killed by a ", "killed by an ", "killed by the "):
                if k in m:
                    killer = m.split(k)[-1].split(".")[0].strip()
                    break
            if killer:
                break

        if not killer:
            adj_mons = [
                m for m in getattr(obs.combat, "adjacent_monsters", [])
                if m not in ("floating eye", "gas spore", "unseen hostile")
            ]
            if adj_mons:
                killer = adj_mons[0]
        if not killer and last_known_hostile:
            if last_known_hostile not in ("floating eye", "gas spore", "unseen hostile"):
                killer = last_known_hostile
        if not killer:
            cname = getattr(obs.combat, "closest_hostile_name", "")
            if cname and cname not in ("floating eye", "gas spore", "unseen hostile"):
                killer = cname
        if not killer and is_starving:
            death_reason = "Starvation"
            killer = "starvation"
    ac_at_death = last_valid_ac
    hp_at_death = getattr(obs.hero, "hp", 0)
    max_hp_at_death = last_valid_max_hp
    excalibur_forged = (
        any("excalibur" in it.name.lower() for it in obs.inventory)
        if hasattr(obs, "inventory")
        else False
    )

    final_depth = max_depth_reached
    final_score = (
        last_valid_score
        if last_valid_score > 0
        else (last_valid_gold + (final_depth * 100))
    )

    try:
        logger.flush_ticks()
    except Exception:
        pass
    try:
        adapter.close()
    except Exception:
        pass

    return {
        "episode_id": ep_id,
        "seed": seed,
        "final_depth": final_depth,
        "final_score": final_score,
        "ep_turns": ep_turns,
        "death_reason": death_reason,
        "wall_sec": time.perf_counter() - gen_start_t,
        "role": role if env_type == "nethack" else "minihack",
        "gold": last_valid_gold,
        "max_depth": final_depth,
        "attacks": ep_attacks,
        "descents": ep_descents,
        "searches": ep_searches,
        "eats": ep_eats,
        "prayers": ep_prayers,
        "death_category": death_reason[:30],
        "inventory_at_death": inventory_at_death_str,
        "last_5_actions": " -> ".join(last_5_actions),
        "turns_dl1": turns_dl1,
        "turns_dl2": turns_dl2,
        "turns_mines": turns_mines,
        "killer": killer,
        "ac_at_death": ac_at_death,
        "hp_at_death": hp_at_death,
        "max_hp_at_death": max_hp_at_death,
        "excalibur_forged": excalibur_forged,
        "turns_fainting": sum(1 for s in recorder.buffer if s.hunger == "FAINTING"),
        "turns_weak": sum(1 for s in recorder.buffer if s.hunger == "WEAK"),
        "is_oscillating": (len(set(s.pos for s in list(recorder.buffer)[-40:])) <= 2)
        if len(recorder.buffer) >= 30
        else False,
        "trajectory": recorder.get_last_10_turns_trajectory(),
    }


def run_synthesis_loop(
    provider: str = "mock",
    model: str | None = None,
    api_key: str | None = None,
    env_type: str = "nethack",
    role: str = "valkyrie",
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    max_generations: int = 500,
    eval_episodes: int = 20,
    max_turns: int = 10000,
    policy_path: str = "data/latest_policy.py",
    fresh: bool = False,
    stall_threshold: int = 80,
    cluster_threshold: int = 2,
    db_path: str = "data/lox.duckdb",
    target_depth: float | None = None,
    workers: int = 10,
    starter_policy: str = "data/modular_starter_policy.py",
    twin_test: bool = True,
    seed_base: int = 42,
):
    print("=" * 65)
    print("LOX Embodied Batched Empirical Policy Synthesis Engine")
    print(f"Provider:    {provider} | Model: {model or 'default'}")
    print(
        f"Environment: {env_type.upper()} ({role if env_type == 'nethack' else task})"
    )
    print(f"Database:    {db_path}")
    print(
        f"Config:      {max_generations} gens | {eval_episodes} eps/gen | {max_turns} max turns"
    )
    print(f"Parallelism: {workers} workers ({min(workers, eval_episodes)} concurrent)")
    print(f"Validation:  Counterfactual Twin Replay = {twin_test} (base_seed={seed_base})")
    if target_depth is not None:
        print(f"Target:      Avg Depth >= {target_depth:.1f}")
    print(f"Policy Path: {policy_path} (fresh={fresh})")
    print(f"Starter Ref: {starter_policy}")
    print("=" * 65)

    run_id = f"synth_{provider}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    author = AuthorAgent(
        provider=provider, model=model, api_key=api_key, db_path=db_path
    )
    DynamicTriggerEngine(
        stall_threshold=stall_threshold, cluster_threshold=cluster_threshold
    )
    FlightRecorder(capacity=100)

    # Initialize DuckDB evolved_policies and meta_experiments tables
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
        con.execute("""
            CREATE TABLE IF NOT EXISTS meta_experiments (
                campaign_id VARCHAR,
                generation INTEGER,
                timestamp TIMESTAMP,
                targeted_skill VARCHAR,
                hypothesis_yaml VARCHAR,
                causal_finding VARCHAR,
                predicted_outcome VARCHAR,
                baseline_avg_depth DOUBLE,
                actual_avg_depth DOUBLE,
                outcome_validated BOOLEAN,
                policy_code TEXT,
                paired_seed_delta DOUBLE,
                seeds_improved INTEGER,
                seeds_regressed INTEGER,
                seeds_unchanged INTEGER,
                ci_95 DOUBLE,
                incident_resolution_rate DOUBLE
            )
        """)
        for col_def in (
            ("paired_seed_delta", "DOUBLE"),
            ("seeds_improved", "INTEGER"),
            ("seeds_regressed", "INTEGER"),
            ("seeds_unchanged", "INTEGER"),
            ("ci_95", "DOUBLE"),
            ("incident_resolution_rate", "DOUBLE"),
        ):
            try:
                con.execute(f"ALTER TABLE meta_experiments ADD COLUMN {col_def[0]} {col_def[1]}")
            except Exception:
                pass
        con.close()
    except Exception as e:
        print(f"[Warning] Failed to initialize DuckDB synthesis tables: {e}")

    # Pre-warm SpatialEngine JIT kernels to populate disk cache before worker forks
    print("[SpatialEngine] Pre-warming Numba JIT spatial kernels...")
    SpatialEngine.warmup()
    print("[SpatialEngine] JIT kernels compiled and ready.")

    # Load or Seed policy
    os.makedirs("data/policies", exist_ok=True)
    if not fresh and os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            current_policy = f.read().strip()
        print(f"\n[Resumed Policy from {policy_path}]")
    else:
        if starter_policy and os.path.exists(starter_policy):
            with open(starter_policy, "r") as f:
                current_policy = f.read().strip()
            print(f"\n[Initialized Seed Policy from Modular Template: {starter_policy}]")
        else:
            current_policy = """
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            if obs.hero.hp_frac < 0.15 and (obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            if obs.combat.adjacent_hostile:
                obs = yield melee_attack_hostile()
                continue
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue
            if obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
                continue
            obs = yield wait()
"""
            print("\n[Initialized Fallback Seed Policy]:")
        with open(policy_path, "w") as f:
            f.write(current_policy.strip() + "\n")
        print("\n[Initialized Seed Policy]:")
    print(current_policy.strip())

    compile_policy(current_policy)

    last_recorded_hypothesis: dict[str, Any] | None = None
    last_baseline_depth: float | None = None
    last_batch_seeds: list[int] = []
    last_batch_results_by_seed: dict[int, dict[str, Any]] = {}
    last_batch_primary_cause: str = ""

    for gen in range(1, max_generations + 1):
        print(
            f"\n--- Running Generation {gen} Evaluation ({eval_episodes} real episodes) ---"
        )
        gen_start_t = time.perf_counter()
        trigger_reason = ""
        gen_depths = []
        gen_turns = []
        gen_scores = []
        gen_dir_id = f"{run_id}_g{gen:03d}"
        logger = ParquetLogger(
            run_id=gen_dir_id, base_dir="data/telemetry", flush_interval=100
        )

        # Seed assignment: Counterfactual Twin Replay or Fresh Batch
        if (
            twin_test
            and last_batch_seeds
            and last_recorded_hypothesis is not None
        ):
            current_seeds = list(last_batch_seeds)
            is_twin_eval = True
            print(
                f"\n[Counterfactual Twin Evaluation] Replaying exact {len(current_seeds)} seeds from Gen {gen - 1} "
                f"to measure hypothesis performance with 0 environment variance!"
            )
        else:
            current_seeds = [seed_base + (gen - 1) * 1000 + i for i in range(eval_episodes)]
            is_twin_eval = False

        payloads = [
            {
                "gen": gen,
                "ep_idx": ep_idx,
                "seed": current_seeds[ep_idx],
                "gen_dir_id": gen_dir_id,
                "policy_code": current_policy,
                "env_type": env_type,
                "role": role,
                "task": task,
                "max_turns": max_turns,
                "gen_start_t": gen_start_t,
            }
            for ep_idx in range(eval_episodes)
        ]

        if workers > 1:
            with mp.Pool(processes=min(workers, eval_episodes)) as pool:
                async_res = pool.map_async(_run_single_episode_worker, payloads)
                try:
                    ep_results = async_res.get(timeout=180.0)
                except (mp.TimeoutError, Exception) as exc:
                    reason = (
                        "WorkerPoolTimeout"
                        if isinstance(exc, mp.TimeoutError)
                        else f"WorkerPoolException: {exc}"[:40]
                    )
                    print(
                        f"\n[Warning] Generation {gen} worker pool encountered {reason}! Terminating worker processes..."
                    )
                    pool.terminate()
                    pool.join()
                    ep_results = [
                        {
                            "episode_id": f"{gen_dir_id}_e{i + 1:03d}",
                            "seed": current_seeds[i],
                            "final_depth": 1,
                            "final_score": 0,
                            "ep_turns": 100,
                            "death_reason": reason,
                            "wall_sec": 180.0,
                            "role": role,
                            "gold": 0,
                            "max_depth": 1,
                            "attacks": 0,
                            "descents": 0,
                            "searches": 0,
                            "eats": 0,
                            "prayers": 0,
                            "death_category": "timeout",
                            "inventory_at_death": "",
                            "last_5_actions": ["wait"],
                            "turns_dl1": 100,
                            "turns_dl2": 0,
                            "turns_mines": 0,
                            "killer": reason,
                            "ac_at_death": 10,
                            "hp_at_death": 0,
                            "max_hp_at_death": 15,
                            "excalibur_forged": False,
                            "trajectory": "",
                        }
                        for i in range(eval_episodes)
                    ]
        else:
            ep_results = [_run_single_episode_worker(p) for p in payloads]

        recent_trajectory = ""
        batch_diagnostics = [RootCauseClassifier.classify(r) for r in ep_results]
        batch_summary = RootCauseClassifier.summarize_batch(batch_diagnostics)

        current_results_by_seed = {}
        for r, diag in zip(ep_results, batch_diagnostics):
            r["root_cause"] = diag.primary_archetype.value
            current_results_by_seed[r.get("seed", 0)] = r

        for r, diag in zip(ep_results, batch_diagnostics):
            gen_depths.append(r["final_depth"])
            gen_scores.append(r["final_score"])
            gen_turns.append(r["ep_turns"])
            if r["death_reason"] != "active":
                recent_trajectory = r["trajectory"]

            logger.log_episode(
                episode_id=r["episode_id"],
                depth=r["final_depth"],
                score=r["final_score"],
                turns=r["ep_turns"],
                death_reason=r["death_reason"],
                solved=False,
                wall_sec=r["wall_sec"],
                role=r["role"],
                gold=r["gold"],
                max_depth=r["max_depth"],
                steps=r["ep_turns"],
                attacks=r["attacks"],
                descents=r["descents"],
                searches=r["searches"],
                eats=r["eats"],
                prayers=r["prayers"],
                death_category=r["death_category"],
                inventory_at_death=r["inventory_at_death"],
                last_5_actions=r["last_5_actions"],
                turns_dl1=r["turns_dl1"],
                turns_dl2=r["turns_dl2"],
                turns_mines=r["turns_mines"],
                killer=r.get("killer", ""),
                ac_at_death=r.get("ac_at_death", 10),
                hp_at_death=r.get("hp_at_death", 0),
                max_hp_at_death=r.get("max_hp_at_death", 0),
                excalibur_forged=r.get("excalibur_forged", False),
                root_cause=diag.primary_archetype.value,
                turns_fainting=diag.turns_fainting,
                has_body_armor=diag.has_body_armor,
                is_oscillating=diag.is_oscillating,
            )

        # Consolidate raw telemetry into DuckDB so LLM tools query live empirical state
        logger.flush()
        consolidate_run(
            run_id=gen_dir_id,
            db_path=db_path,
            telemetry_dir="data/telemetry",
            cleanup=True,
        )

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

        death_summary_str = (
            ", ".join(f"{r[0]} ({r[2]}%)" for r in top_deaths) if top_deaths else "None"
        )
        # Determine dominant root cause archetype from empirical diagnostic engine
        top_arch = (
            max(batch_summary.archetype_counts.items(), key=lambda x: x[1])[0]
            if batch_summary.archetype_counts
            else "COMBAT_GENERAL"
        )
        top_arch_pct = batch_summary.archetype_percentages.get(top_arch, 0.0)

        # Save compact failure summary to YAML
        batch_summary.save("data/latest_diagnostics.yaml")

        print(
            f"\n[Gen {gen} Batch Metrics ({eval_episodes} eps)] Avg Depth: {avg_d:.2f} | Max Depth: {max_d} | Avg Turns: {avg_t:.1f}"
        )
        print(f"[Gen {gen} Ranked Fatalities] {death_summary_str}")
        print(
            f"\n[Gen {gen} Failure Diagnostics (saved to data/latest_diagnostics.yaml)]:\n{batch_summary.format_yaml()}\n"
        )

        # Scientific Method: Evaluate previous generation's hypothesis if one was recorded
        if last_recorded_hypothesis is not None and last_baseline_depth is not None:
            pred = last_recorded_hypothesis.get("predicted_outcome", {})
            direction = (
                pred.get("expected_direction", "increase")
                if isinstance(pred, dict)
                else "increase"
            )

            # Check if this was a paired counterfactual twin run
            paired_delta: float | None = None
            seeds_improved = 0
            seeds_regressed = 0
            seeds_unchanged = 0
            ci_95: float | None = None
            resolution_rate: float | None = None
            failed_seed_count = 0

            if is_twin_eval and last_batch_results_by_seed:
                paired_deltas = []
                for s in current_seeds:
                    if s in last_batch_results_by_seed and s in current_results_by_seed:
                        d_new = current_results_by_seed[s]["final_depth"]
                        d_old = last_batch_results_by_seed[s]["final_depth"]
                        delta_s = d_new - d_old
                        paired_deltas.append(delta_s)
                        if delta_s > 0:
                            seeds_improved += 1
                        elif delta_s < 0:
                            seeds_regressed += 1
                        else:
                            seeds_unchanged += 1

                if paired_deltas:
                    paired_delta = float(np.mean(paired_deltas))
                    std_err = (
                        float(np.std(paired_deltas) / np.sqrt(len(paired_deltas)))
                        if len(paired_deltas) > 1
                        else 0.0
                    )
                    ci_95 = float(1.96 * std_err)

                # Check incident resolution rate on episodes that suffered the primary root cause
                if last_batch_primary_cause:
                    failed_seeds = [
                        s
                        for s, r in last_batch_results_by_seed.items()
                        if r.get("root_cause") == last_batch_primary_cause
                    ]
                    failed_seed_count = len(failed_seeds)
                    if failed_seeds:
                        resolved = sum(
                            1
                            for s in failed_seeds
                            if s in current_results_by_seed
                            and current_results_by_seed[s]["final_depth"]
                            > last_batch_results_by_seed[s]["final_depth"]
                        )
                        resolution_rate = float(resolved / len(failed_seeds))

            if paired_delta is not None:
                if direction == "increase":
                    validated = bool(paired_delta > 0 and (seeds_improved >= seeds_regressed))
                else:
                    validated = bool(paired_delta < 0)
                eval_metric_str = f"Paired Delta: {paired_delta:+.2f} (±{ci_95:.2f} 95% CI) | Improved: {seeds_improved}, Regressed: {seeds_regressed}, Tied: {seeds_unchanged}"
            else:
                delta = avg_d - last_baseline_depth
                validated = (
                    bool(avg_d > last_baseline_depth)
                    if direction == "increase"
                    else bool(avg_d < last_baseline_depth)
                )
                eval_metric_str = f"Batch Avg Delta: {avg_d - last_baseline_depth:+.2f}"

            print("\n[Scientific Method: Hypothesis Validation]")
            print(
                f"  Targeted Skill: {last_recorded_hypothesis.get('targeted_skill', 'N/A')}"
            )
            print(
                f"  Causal Finding: {last_recorded_hypothesis.get('causal_finding', 'N/A')}"
            )
            print(
                f"  Evaluation Mode: {'Counterfactual Twin Replay (0 Variance)' if is_twin_eval else 'Fresh Seed Batch'}"
            )
            print(
                f"  Baseline Depth: {last_baseline_depth:.2f} -> Actual Depth: {avg_d:.2f} ({eval_metric_str})"
            )
            if resolution_rate is not None and last_batch_primary_cause:
                print(
                    f"  '{last_batch_primary_cause}' Resolution Rate: {resolution_rate * 100.0:.1f}% ({failed_seed_count} incident seeds)"
                )
            print(
                f"  Outcome: {'VALIDATED' if validated else 'FALSIFIED'} (Predicted: {pred})"
            )

            try:
                con = duckdb.connect(db_path)
                con.execute(
                    """
                    UPDATE meta_experiments
                    SET actual_avg_depth = ?,
                        outcome_validated = ?,
                        paired_seed_delta = ?,
                        seeds_improved = ?,
                        seeds_regressed = ?,
                        seeds_unchanged = ?,
                        ci_95 = ?,
                        incident_resolution_rate = ?
                    WHERE campaign_id = ? AND generation = ?
                """,
                    [
                        float(avg_d),
                        validated,
                        paired_delta,
                        seeds_improved,
                        seeds_regressed,
                        seeds_unchanged,
                        ci_95,
                        resolution_rate,
                        run_id,
                        gen - 1,
                    ],
                )
                con.close()
            except Exception as e:
                print(f"[Warning] Failed to update meta_experiments validation: {e}")

            last_recorded_hypothesis = None

        if target_depth is not None and avg_d >= target_depth:
            print("\n" + "=" * 65)
            print(
                f"[CAMPAIGN GOAL ACHIEVED] Generation {gen} reached target average depth {avg_d:.2f} >= {target_depth:.2f}!"
            )
            print(f"Max Depth: {max_d} | Average Turns: {avg_t:.1f}")
            print("=" * 65)
            break

        trigger_reason = (
            f"Generation {gen} Empirical Diagnostic Autopsy ({eval_episodes} episodes): "
            f"Avg Depth {avg_d:.2f}, Max Depth {max_d}, Avg Turns {avg_t:.1f}. "
            f"#1 Empirical Root Cause: '{top_arch}' ({top_arch_pct:.1f}% of runs). "
            f"Mitigate this specific failure mode, optimize stair navigation, and break through to Depth 10+."
        )

        # Build comprehensive, pure YAML incident report
        report_data: dict[str, Any] = {
            "generation": gen,
            **batch_summary.to_dict(),
            "ranked_fatalities": [
                {"cause": r[0], "count": r[1], "pct": float(r[2])} for r in top_deaths
            ],
        }
        if recent_fatal_samples:
            report_data["recent_fatal_incidents"] = [
                {
                    "cause": s[0],
                    "depth": s[1],
                    "turn": s[2],
                    "inventory": s[3] or "Empty",
                    "actions": s[4] or "N/A",
                }
                for s in recent_fatal_samples
            ]
        if recent_trajectory:
            report_data["pre_death_trajectory"] = recent_trajectory

        full_report_yaml = yaml.dump(report_data, sort_keys=False)
        with open("data/latest_report.yaml", "w") as f:
            f.write(full_report_yaml)

        status_rep = f"```yaml\n{full_report_yaml.strip()}\n```"

        # Record this batch's seeds, results, and primary failure archetype for the next generation's counterfactual validation
        last_batch_seeds = list(current_seeds)
        last_batch_results_by_seed = dict(current_results_by_seed)
        last_batch_primary_cause = top_arch

        print(
            "\n[Author Agent] Initiating empirical synthesis session (querying DuckDB & evaluating)..."
        )
        new_code, tree, error = author.synthesize_policy(
            current_policy=current_policy,
            trigger_reason=trigger_reason,
            status_report=status_rep,
            run_id=run_id,
            causal_summary=batch_summary.causal_summary,
        )

        if error:
            print(f"[Validation Failed] {error}")
            print(f"[Rejected Candidate Code]:\n{new_code.strip()}\n")
        else:
            print(f"[Policy Verified & Compiled! Generation {gen} accepted]")
            current_policy = new_code
            print("\nEvolved Policy Program:")
            print(new_code.strip())

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
                        [
                            gen,
                            run_id,
                            float(avg_d),
                            int(max_d),
                            float(avg_t),
                            new_code.strip(),
                            datetime.datetime.now(),
                        ],
                    )
                    con.close()
                    break
                except Exception as e:
                    if attempt == 2:
                        print(
                            f"[Warning] Could not record evolved policy in DuckDB: {e}"
                        )
                    time.sleep(0.5)

            # Log hypothesis to DuckDB meta_experiments table
            hypothesis = getattr(author, "last_hypothesis", None)
            if hypothesis:
                last_recorded_hypothesis = hypothesis
                last_baseline_depth = float(avg_d)
                for attempt in range(3):
                    try:
                        con = duckdb.connect(db_path)
                        con.execute(
                            """
                            INSERT INTO meta_experiments (
                                campaign_id, generation, timestamp, targeted_skill,
                                hypothesis_yaml, causal_finding, predicted_outcome,
                                baseline_avg_depth, actual_avg_depth, outcome_validated,
                                policy_code
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                run_id,
                                gen,
                                datetime.datetime.now(),
                                str(hypothesis.get("targeted_skill", "")),
                                yaml.dump(hypothesis),
                                str(hypothesis.get("causal_finding", "")),
                                str(hypothesis.get("predicted_outcome", "")),
                                float(avg_d),
                                None,
                                None,
                                new_code.strip(),
                            ],
                        )
                        con.close()
                        print(
                            f"[Meta-Experiment Logged] Registered scientific hypothesis for Gen {gen} in DuckDB meta_experiments."
                        )
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(
                                f"[Warning] Could not record meta_experiment in DuckDB: {e}"
                            )
                        time.sleep(0.5)
                    time.sleep(0.5)

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
    parser.add_argument(
        "--provider", default="mock", choices=["mock", "gemini", "openrouter"]
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key (overrides environment and .env)",
    )
    parser.add_argument(
        "--env",
        default="nethack",
        choices=["nethack", "minihack"],
        help="Execution environment",
    )
    parser.add_argument("--role", default="valkyrie", help="Hero role for NetHack")
    parser.add_argument(
        "--task",
        default="MiniHack-ExploreMaze-Easy-Mapped-v0",
        help="Task for MiniHack",
    )
    parser.add_argument(
        "--generations", type=int, default=500, help="Total synthesis generations"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Real evaluation episodes per generation batch (default: 20)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10000, help="Max turns per episode"
    )
    parser.add_argument(
        "--policy-path",
        default="data/latest_policy.py",
        help="Path to persisted latest policy",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh start instead of resuming latest policy",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=80,
        help="Turns without progress to trigger stall autopsy",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=int,
        default=2,
        help="Deaths of same cause to trigger cluster autopsy",
    )
    parser.add_argument("--db-path", default="data/lox.duckdb")
    parser.add_argument(
        "--target-depth",
        type=float,
        default=None,
        help="Target average depth to reach before stopping",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel worker processes for episode evaluation",
    )
    parser.add_argument(
        "--starter-policy",
        default="data/modular_starter_policy.py",
        help="Path to modular starter policy template to seed or reset from",
    )
    parser.add_argument(
        "--twin-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable counterfactual seed-pinned twin evaluation for hypothesis validation (default: True)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Deterministic base seed for episode evaluations",
    )
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
        workers=args.workers,
        starter_policy=args.starter_policy,
        twin_test=args.twin_test,
        seed_base=args.seed_base,
    )
