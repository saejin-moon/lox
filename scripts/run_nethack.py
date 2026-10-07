#!/usr/bin/env python3
"""
LOX NetHack Runner & Evaluator.
Executes autonomous runs using compiled Pythonic Behavior Trees,
records flight telemetry to streaming Parquet partitions,
monitors dynamic triggers, and consolidates into DuckDB post-run.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.core.types import Action
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter
from lox.telemetry.consolidator import consolidate_run, safe_duckdb_connect
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType


def run_nethack_eval(
    episodes: int = 10,
    max_turns: int = 10000,
    policy_path: str = "data/latest_policy.py",
    role: str = "valkyrie",
    env_id: str = "NetHackChallenge-v0",
    run_id: str | None = None,
    db_path: str = "data/lox.duckdb",
    telemetry_dir: str = "data/telemetry",
    record_telemetry: bool = True,
) -> dict[str, Any]:
    if run_id is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"nethack_{role}_{ts}"

    print("=" * 65)
    print("LOX NetHack Autonomous Evaluation")
    print(f"Run ID:      {run_id}")
    print(f"Environment: {env_id} (Role: {role})")
    print(f"Episodes:    {episodes} | Max Turns: {max_turns}")
    print(f"Policy:      {policy_path}")
    print(f"Telemetry:   {'Parquet -> DuckDB' if record_telemetry else 'Disabled'}")
    print("=" * 65)

    adapter = NetHackAdapter(env_id=env_id, role=role)
    trigger_engine = DynamicTriggerEngine(stall_threshold=80, cluster_threshold=3)
    flight_recorder = FlightRecorder(capacity=100)

    logger = None
    if record_telemetry:
        logger = ParquetLogger(
            run_id=run_id, base_dir=telemetry_dir, flush_interval=100
        )

    # Load policy from policy_path or fallback to default
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            policy_code = f.read().strip()
        print(f"[Loaded Benchmark Policy from {policy_path}]")
    else:
        policy_code = """
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # Emergency: prayer and healing
            if obs.hero.hp_frac < 0.20 and (obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            elif obs.hero.hp_frac < 0.35 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
            elif obs.hero.hunger_state >= HUNGRY and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # Tactical combat & retreat
            if obs.combat.adjacent_hostile:
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                elif obs.hero.hp_frac < 0.35 and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # Environment & Navigation
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.stairs_down_known and not obs.spatial.has_unvisited_frontier:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                obs = yield wait()
"""
        print("[Using Default Policy]")

    tree = compile_policy(policy_code)

    episode_summaries = []
    t_start = time.perf_counter()

    for ep in range(episodes):
        ep_id = f"ep_{ep + 1:03d}"
        ep_t0 = time.perf_counter()
        obs = adapter.reset(seed=ep + 2000)

        score = 0
        death_reason = "MaxTurnsReached"
        solved = False
        max_depth_reached = obs.hero.depth
        last_valid_turns = obs.hero.turn
        last_valid_max_hp = obs.hero.max_hp
        policy_runner = tree.create_runner(obs)
        last_5_actions: list[str] = []
        turns_dl1 = 0
        turns_dl2 = 0
        turns_mines = 0
        inventory_at_death_str = ""

        # Action and event counters
        total_steps = 0
        total_attacks = 0
        total_descents = 0
        total_searches = 0
        total_eats = 0
        total_prayers = 0

        for turn_idx in range(max_turns):
            hero = obs.hero
            hy, hx = hero.y, hero.x
            max_depth_reached = max(max_depth_reached, hero.depth)
            last_valid_turns = hero.turn
            last_valid_max_hp = hero.max_hp

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
                getattr(obs.combat, "hostile_count_fov", 0)
                if hasattr(obs, "combat")
                else 0
            )
            dungeon_branch = (
                getattr(obs.hero, "dungeon_branch", "dungeon")
                if hasattr(obs, "hero")
                else "dungeon"
            )

            # Track action metrics
            if action.name == "step":
                total_steps += 1
            elif action.name == "melee_attack_hostile":
                total_attacks += 1
            elif action.name == "descend":
                total_descents += 1
                if logger is not None:
                    logger.log_event(
                        ep_id,
                        hero.turn,
                        hero.depth,
                        "descend",
                        message="Hero descended stairs",
                    )
            elif action.name == "search":
                total_searches += 1
            elif action.name == "eat_carried_food":
                total_eats += 1
                if logger is not None:
                    logger.log_event(
                        ep_id, hero.turn, hero.depth, "eat", message="Hero ate food"
                    )
            elif action.name == "pray":
                total_prayers += 1
                if logger is not None:
                    logger.log_event(
                        ep_id, hero.turn, hero.depth, "pray", message="Hero prayed"
                    )

            flight_recorder.record_turn(
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

            # Check dynamic triggers
            food_items = sum(1 for it in obs.inventory if it.category == "food")
            has_frontier = getattr(obs.spatial, "has_unvisited_frontier", True)
            trig, trig_reason = trigger_engine.check_turn(
                turns_on_level=hero.turns_on_level,
                depth=hero.depth,
                hunger_state=hero.hunger_state.name,
                food_count=food_items,
                has_frontier=has_frontier,
            )
            if trig in (TriggerType.STALL, TriggerType.STARVATION):
                if logger is not None:
                    logger.log_event(
                        ep_id, hero.turn, hero.depth, "trigger", message=trig_reason
                    )
                if hero.turns_on_level % 80 == 0:
                    print(
                        f"  [Dynamic Trigger: {trig.name}] {trig_reason} at turn {hero.turn}"
                    )

            # Telemetry tick logging
            if logger is not None:
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
                    if hasattr(obs, "combat")
                    and hasattr(obs.combat, "adjacent_monsters")
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

            # Step environment
            obs, reward, term, trunc, info = adapter.step(action)

            # Check for kill event
            if (
                "you kill" in obs.message.lower()
                or "you destroy" in obs.message.lower()
            ):
                if logger is not None:
                    logger.log_event(
                        ep_id, hero.turn, hero.depth, "combat_kill", message=obs.message
                    )

            if term or trunc:
                end_msg = obs.message.lower()
                if (
                    "died" in end_msg
                    or "killed" in end_msg
                    or "choked" in end_msg
                    or "starved" in end_msg
                ):
                    death_reason = obs.message
                elif hero.hp <= 0:
                    death_reason = "ZeroHP (Killed)"
                else:
                    death_reason = "Terminated"

                inventory_items = (
                    [f"{it.name} ({it.category})" for it in obs.inventory]
                    if hasattr(obs, "inventory")
                    else []
                )
                inventory_at_death_str = ", ".join(inventory_items[:10])

                # Check cluster fatality trigger
                flight_recorder.record_death(death_reason)
                c_trig, c_reason = trigger_engine.check_death_cluster(flight_recorder)
                if c_trig == TriggerType.CLUSTER_DEATH:
                    print(f"  [Notice: Cluster Death Trigger] {c_reason}")
                break

        ep_wall = time.perf_counter() - ep_t0
        final_depth = obs.hero.depth if obs.hero.depth > 0 else max_depth_reached
        final_turns = obs.hero.turn if obs.hero.turn > 0 else last_valid_turns
        final_hp = max(0, obs.hero.hp)
        final_max_hp = obs.hero.max_hp if obs.hero.max_hp > 0 else last_valid_max_hp
        score = obs.hero.gold + (final_depth * 100)

        # Categorize death reason
        low_death = death_reason.lower()
        if "maxturns" in low_death or "timeout" in low_death:
            death_cat = "timeout"
        elif "starv" in low_death or "faint" in low_death or "choke" in low_death:
            death_cat = "starvation"
        elif "kill" in low_death or "died" in low_death or "zerohp" in low_death:
            death_cat = "combat"
        elif solved:
            death_cat = "survived"
        else:
            death_cat = "other"

        if logger is not None:
            logger.log_episode(
                episode_id=ep_id,
                depth=final_depth,
                score=score,
                turns=final_turns,
                death_reason=death_reason,
                solved=solved,
                wall_sec=ep_wall,
                role=role,
                gold=obs.hero.gold,
                max_depth=max_depth_reached,
                steps=total_steps,
                attacks=total_attacks,
                descents=total_descents,
                searches=total_searches,
                eats=total_eats,
                prayers=total_prayers,
                death_category=death_cat,
                inventory_at_death=inventory_at_death_str,
                last_5_actions=" -> ".join(last_5_actions),
                turns_dl1=turns_dl1,
                turns_dl2=turns_dl2,
                turns_mines=turns_mines,
            )

        episode_summaries.append(
            {
                "episode": ep + 1,
                "depth": final_depth,
                "turns": final_turns,
                "hp": f"{final_hp}/{final_max_hp}",
                "reason": death_reason[:40],
                "wall_sec": ep_wall,
            }
        )

        print(
            f"Ep {ep + 1:2d}/{episodes:2d}: Depth {final_depth} | Turns {final_turns:4d} | "
            f"HP {final_hp:2d}/{final_max_hp:2d} | End: {death_reason[:35]} ({ep_wall:.2f}s)"
        )

    adapter.close()
    if logger is not None:
        logger.close()

    total_wall = time.perf_counter() - t_start

    # Consolidate into DuckDB and cleanup Parquet files
    consolidation_stats = None
    if record_telemetry:
        print("\n[Telemetry] Consolidating Parquet partitions into DuckDB...")
        consolidation_stats = consolidate_run(
            run_id=run_id,
            db_path=db_path,
            telemetry_dir=telemetry_dir,
            cleanup=True,
        )
        print(
            f"[Telemetry] Successfully consolidated {consolidation_stats['ticks_added']} ticks and {consolidation_stats['episodes_added']} episodes into {db_path}"
        )
        print(
            f"[Telemetry] Raw Parquet files safely cleaned up: {consolidation_stats['cleaned_up']}"
        )

        # Query DuckDB summary statistics
        con = safe_duckdb_connect(db_path, read_only=True)
        res = con.execute(
            "SELECT AVG(depth), MAX(depth), AVG(turns), COUNT(*) FROM episodes WHERE run_id = ?",
            [run_id],
        ).fetchone()
        con.close()
        if res and res[3] > 0:
            avg_depth = res[0] or 0.0
            max_depth = res[1] or 0
            avg_turns = res[2] or 0.0
            ep_count = res[3]
        else:
            avg_depth, max_depth, avg_turns, ep_count = 0.0, 0, 0.0, 0
        print("\n" + "=" * 65)
        print("DuckDB Evaluation Summary:")
        print(f"Total Episodes Analyzed: {ep_count}")
        print(f"Average Depth:           {avg_depth:.2f}")
        print(f"Max Depth:               {max_depth}")
        print(f"Average Turns:           {avg_turns:.1f}")
        print(f"Total Wall Time:         {total_wall:.2f}s")
        print("=" * 65)

    return {
        "run_id": run_id,
        "episodes": episode_summaries,
        "consolidation": consolidation_stats,
        "total_wall": total_wall,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max-turns", type=int, default=10000, help="Max turns per episode"
    )
    parser.add_argument(
        "--policy-path",
        default="data/latest_policy.py",
        help="Path to policy file to evaluate",
    )
    parser.add_argument("--role", default="valkyrie", help="NetHack hero role")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    parser.add_argument(
        "--db-path", default="data/lox.duckdb", help="Path to DuckDB database"
    )
    parser.add_argument(
        "--no-telemetry", action="store_true", help="Disable Parquet/DuckDB telemetry"
    )
    args = parser.parse_args()

    run_nethack_eval(
        episodes=args.episodes,
        max_turns=args.max_turns,
        policy_path=args.policy_path,
        role=args.role,
        run_id=args.run_id,
        db_path=args.db_path,
        record_telemetry=not args.no_telemetry,
    )
