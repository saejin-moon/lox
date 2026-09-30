#!/usr/bin/env python3
"""
LOX 2.0 NetHack Runner & Evaluator.
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
import nle.nethack as nh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.core.types import Action, HeroState, HungerState, Item, Observation, Status
from lox.core.spatial import SpatialEngine
from lox.core.tree import Blackboard
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter, DIR_CHARS
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.consolidator import consolidate_run
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType
from lox.telemetry.recorder import FlightRecorder


# Passable NetHack ASCII characters for spatial navigation
PASSABLE_CHARS = {
    ord("."),  # Room floor
    ord("#"),  # Corridor
    ord("<"),  # Stairs up
    ord(">"),  # Stairs down
    ord("'"),  # Open door
    ord("+"),  # Closed door (walkable to open)
    ord("$"),  # Gold
    ord("%"),  # Food / corpse
    ord("!"),  # Potion
    ord("?"),  # Scroll
    ord("/"),  # Wand
    ord("="),  # Ring
    ord("*"),  # Gem
    ord(")"),  # Weapon
    ord("["),  # Armor
    ord("("),  # Tool
    ord("0"),  # Iron ball / boulder
    ord("_"),  # Altar
    ord("\\"), # Throne / sink
    ord("^"),  # Trap
    ord("@"),  # Player
}


def build_walkable_mask(obs: Observation) -> np.ndarray:
    """Builds a 2D boolean mask of walkable tiles from chars and glyphs."""
    chars = obs.chars
    glyphs = obs.glyphs
    mask = np.zeros(chars.shape, dtype=bool)

    for c in PASSABLE_CHARS:
        mask |= (chars == c)

    # Exclude tiles with living monsters (path around them unless attacking)
    # Do NOT exclude the hero themselves
    hy, hx = obs.hero.y, obs.hero.x
    if glyphs is not None:
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                if (y, x) == (hy, hx):
                    continue
                g = int(glyphs[y, x])
                if nh.glyph_is_monster(g) and not nh.glyph_is_pet(g):
                    mask[y, x] = False

    # Hero position is unconditionally walkable
    mask[hy, hx] = True
    return mask


def run_nethack_eval(
    episodes: int = 10,
    max_turns: int = 1000,
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
    print(f"LOX 2.0 NetHack Autonomous Evaluation")
    print(f"Run ID:      {run_id}")
    print(f"Environment: {env_id} (Role: {role})")
    print(f"Episodes:    {episodes} | Max Turns: {max_turns}")
    print(f"Telemetry:   {'Parquet -> DuckDB' if record_telemetry else 'Disabled'}")
    print("=" * 65)

    adapter = NetHackAdapter(env_id=env_id, role=role)
    trigger_engine = DynamicTriggerEngine(stall_threshold=80, cluster_threshold=3)
    flight_recorder = FlightRecorder(capacity=100)

    logger = None
    if record_telemetry:
        logger = ParquetLogger(run_id=run_id, base_dir=telemetry_dir, flush_interval=100)

    # State tracking across actions
    current_depth = 1
    known_stairs_down: tuple[int, int] | None = None
    stuck_counter = 0
    last_position = (-1, -1)

    # --- Domain Action Handlers ---
    def handle_quaff_healing(bb: Blackboard, args) -> Action | Status:
        for it in bb.obs.inventory:
            if it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"]):
                return Action(name="quaff_healing", slot=it.slot)
        return Status.FAILURE

    def handle_emergency_pray(bb: Blackboard, args) -> Action | Status:
        if adapter.can_safely_pray(bb.obs.hero.turn):
            return Action(name="pray")
        return Status.FAILURE

    def handle_eat_food(bb: Blackboard, args) -> Action | Status:
        for it in bb.obs.inventory:
            if it.category == "food":
                return Action(name="eat_carried_food", slot=it.slot)
        return Status.FAILURE

    def handle_melee_attack(bb: Blackboard, args) -> Action | Status:
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        glyphs = bb.obs.glyphs
        if glyphs is None:
            return Status.FAILURE

        # Look in 8 adjacent directions for hostile monsters
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ty, tx = hy + dy, hx + dx
                if 0 <= ty < 21 and 0 <= tx < 79:
                    g = int(glyphs[ty, tx])
                    if nh.glyph_is_monster(g) and not nh.glyph_is_pet(g):
                        # Floating eye check: do NOT melee floating eye!
                        if not adapter.is_target_floating_eye(glyphs, ty, tx):
                            return Action(name="melee_attack_hostile", direction=(dy, dx))
        return Status.FAILURE

    def handle_step_to_stairs(bb: Blackboard, args) -> Action | Status:
        nonlocal known_stairs_down
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        hero_pos = (hy, hx)

        if known_stairs_down is not None and hero_pos == known_stairs_down:
            return Action(name="descend")

        if known_stairs_down is not None:
            walkable = build_walkable_mask(bb.obs)
            walkable[known_stairs_down[0], known_stairs_down[1]] = True
            path = SpatialEngine.find_path(hero_pos, known_stairs_down, walkable)
            if path:
                dy = path[0][0] - hy
                dx = path[0][1] - hx
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_step_to_frontier(bb: Blackboard, args) -> Action | Status:
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        hero_pos = (hy, hx)
        walkable = build_walkable_mask(bb.obs)

        frontier = SpatialEngine.find_nearest_frontier(hero_pos, walkable, adapter.visited)
        if frontier is not None:
            path = SpatialEngine.find_path(hero_pos, frontier, walkable)
            if path:
                dy = path[0][0] - hy
                dx = path[0][1] - hx
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_search(bb: Blackboard, args) -> Action | Status:
        return Action(name="search")

    # Policy AST Code using strict approved schema
    policy_code = """
def emergency_recovery():
    if hp_frac < 0.25:
        pray()
    if hp_frac < 0.50:
        quaff_healing()
    if hunger_state >= HUNGRY:
        eat_carried_food()

def combat():
    if adjacent_hostile:
        melee_attack_hostile()

def descend():
    if stairs_down_known:
        step_to_stairs_down()

def explore():
    if has_unvisited_frontier:
        step_to_frontier()
    else:
        search()

plan = [
    emergency_recovery,
    combat,
    descend,
    explore,
]
"""
    tree = compile_policy(
        policy_code,
        action_handlers={
            "quaff_healing": handle_quaff_healing,
            "pray": handle_emergency_pray,
            "eat_carried_food": handle_eat_food,
            "melee_attack_hostile": handle_melee_attack,
            "step_to_stairs_down": handle_step_to_stairs,
            "step_to_frontier": handle_step_to_frontier,
            "search": handle_search,
        },
    )

    episode_summaries = []
    t_start = time.perf_counter()

    for ep in range(episodes):
        ep_id = f"ep_{ep+1:03d}"
        ep_t0 = time.perf_counter()
        obs = adapter.reset(seed=ep + 2000)

        current_depth = obs.hero.depth
        known_stairs_down = None
        last_position = (-1, -1)
        stuck_counter = 0

        score = 0
        death_reason = "MaxTurnsReached"
        solved = False
        last_action_name = ""
        max_depth_reached = obs.hero.depth
        last_valid_turns = obs.hero.turn
        last_valid_hp = obs.hero.hp
        last_valid_max_hp = obs.hero.max_hp

        for turn_idx in range(max_turns):
            hero = obs.hero
            hy, hx = hero.y, hero.x
            max_depth_reached = max(max_depth_reached, hero.depth)
            last_valid_turns = hero.turn
            last_valid_hp = hero.hp
            last_valid_max_hp = hero.max_hp

            # Detect new floor / depth change
            if hero.depth != current_depth:
                current_depth = hero.depth
                known_stairs_down = None

            # Detect staircase on screen
            stairs_loc = np.argwhere(obs.chars == ord(">"))
            if len(stairs_loc) > 0:
                known_stairs_down = (int(stairs_loc[0, 0]), int(stairs_loc[0, 1]))

            # Check if movement is stuck (only when attempting to step)
            if last_action_name == "step" and (hy, hx) == last_position:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_position = (hy, hx)

            # Check for adjacent enemies
            has_adj_enemy = False
            if obs.glyphs is not None:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ty, tx = hy + dy, hx + dx
                        if 0 <= ty < 21 and 0 <= tx < 79:
                            g = int(obs.glyphs[ty, tx])
                            if nh.glyph_is_monster(g) and not nh.glyph_is_pet(g):
                                has_adj_enemy = True
                                break
                    if has_adj_enemy:
                        break

            has_frontier = bool(stuck_counter < 5)

            memory = {
                "adjacent_hostile": has_adj_enemy,
                "stairs_down_known": known_stairs_down is not None,
                "has_unvisited_frontier": has_frontier,
                "can_safely_pray": adapter.can_safely_pray(hero.turn),
            }

            # Execute behavior tree
            action = tree.execute(obs, memory=memory)
            if action is None:
                action = Action(name="search")
            last_action_name = action.name

            flight_recorder.record_turn(
                turn=hero.turn,
                depth=hero.depth,
                hp=hero.hp,
                max_hp=hero.max_hp,
                hunger=hero.hunger_state.name,
                pos=(hy, hx),
                action_name=action.name,
                message=obs.message,
            )

            # Dynamic trigger check
            trig, trig_reason = trigger_engine.check_turn(turns_on_level=hero.turns_on_level, depth=hero.depth)
            if trig == TriggerType.STALL and hero.turns_on_level == 80:
                print(f"  [Notice: Floor Stagnation Trigger] {trig_reason} at turn {hero.turn}")

            # Telemetry tick logging
            if logger is not None:
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
                )

            # Step environment
            obs, reward, term, trunc, info = adapter.step(action)

            if term or trunc:
                end_msg = obs.message.lower()
                if "died" in end_msg or "killed" in end_msg or "choked" in end_msg or "starved" in end_msg:
                    death_reason = obs.message
                elif hero.hp <= 0:
                    death_reason = "ZeroHP (Killed)"
                else:
                    death_reason = "Terminated"

                # Check cluster fatality trigger
                flight_recorder.record_death(death_reason)
                c_trig, c_reason = trigger_engine.check_death_cluster(flight_recorder)
                if c_trig == TriggerType.CLUSTER_DEATH:
                    print(f"  [Notice: Cluster Death Trigger] {c_reason}")
                break

        ep_wall = time.perf_counter() - ep_t0
        final_depth = obs.hero.depth if obs.hero.depth > 0 else max_depth_reached
        final_turns = obs.hero.turn if obs.hero.turn > 0 else last_valid_turns
        final_hp = obs.hero.hp if obs.hero.hp > 0 else 0
        final_max_hp = obs.hero.max_hp if obs.hero.max_hp > 0 else last_valid_max_hp
        score = obs.hero.gold + (final_depth * 100)

        if logger is not None:
            logger.log_episode(
                episode_id=ep_id,
                depth=final_depth,
                score=score,
                turns=final_turns,
                death_reason=death_reason,
                solved=solved,
                wall_sec=ep_wall,
            )

        episode_summaries.append({
            "episode": ep + 1,
            "depth": final_depth,
            "turns": final_turns,
            "hp": f"{final_hp}/{final_max_hp}",
            "reason": death_reason[:40],
            "wall_sec": ep_wall,
        })

        print(
            f"Ep {ep+1:2d}/{episodes:2d}: Depth {final_depth} | Turns {final_turns:4d} | "
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
        print(f"[Telemetry] Successfully consolidated {consolidation_stats['ticks_added']} ticks and {consolidation_stats['episodes_added']} episodes into {db_path}")
        print(f"[Telemetry] Raw Parquet files safely cleaned up: {consolidation_stats['cleaned_up']}")

        # Query DuckDB summary statistics
        con = duckdb.connect(db_path, read_only=True)
        res = con.execute("SELECT AVG(depth), MAX(depth), AVG(turns), COUNT(*) FROM episodes WHERE run_id = ?", [run_id]).fetchone()
        con.close()
        avg_depth, max_depth, avg_turns, ep_count = res if res else (0.0, 0, 0.0, 0)
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
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max-turns", type=int, default=500, help="Max turns per episode")
    parser.add_argument("--role", default="valkyrie", help="NetHack hero role")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    parser.add_argument("--db-path", default="data/lox.duckdb", help="Path to DuckDB database")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable Parquet/DuckDB telemetry")
    args = parser.parse_args()

    run_nethack_eval(
        episodes=args.episodes,
        max_turns=args.max_turns,
        role=args.role,
        run_id=args.run_id,
        db_path=args.db_path,
        record_telemetry=not args.no_telemetry,
    )
