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
from lox.core.tree import Blackboard, BehaviorTree
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter, DIR_CHARS
from lox.envs.minihack import MiniHackAdapter
from lox.author.agent import AuthorAgent
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType
from lox.telemetry.parquet import ParquetLogger
from lox.telemetry.consolidator import consolidate_run
from lox.telemetry.tokens import get_token_usage_summary


PASSABLE_CHARS = {
    ord("."), ord("#"), ord("<"), ord(">"), ord("'"), ord("+"),
    ord("$"), ord("%"), ord("!"), ord("?"), ord("/"), ord("="),
    ord("*"), ord(")"), ord("["), ord("("), ord("0"), ord("_"),
    ord("\\"), ord("^"), ord("@"),
}


def build_walkable_mask(obs: Observation) -> np.ndarray:
    """Builds a 2D boolean mask of walkable tiles from chars and glyphs."""
    chars = obs.chars
    glyphs = obs.glyphs
    mask = np.zeros(chars.shape, dtype=bool)

    for c in PASSABLE_CHARS:
        mask |= (chars == c)

    hy, hx = obs.hero.y, obs.hero.x
    if glyphs is not None:
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                if y == hy and x == hx:
                    continue
                g = int(glyphs[y, x])
                if nh.glyph_is_monster(g):
                    mask[y, x] = False
    return mask


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
):
    print("=" * 65)
    print("LOX 2.0 Embodied Batched Empirical Policy Synthesis Engine")
    print(f"Provider:    {provider} | Model: {model or 'default'}")
    print(f"Environment: {env_type.upper()} ({role if env_type == 'nethack' else task})")
    print(f"Database:    {db_path}")
    print(f"Config:      {max_generations} gens | {eval_episodes} eps/gen | {max_turns} max turns")
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

    # Shared action handlers
    known_stairs_down: tuple[int, int] | None = None
    known_stairs_up: tuple[int, int] | None = None

    def handle_quaff_healing(bb: Blackboard, args):
        for it in bb.obs.inventory:
            if it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"]):
                return Action(name="quaff_healing", slot=it.slot)
        return Status.FAILURE

    def handle_emergency_pray(bb: Blackboard, args):
        if hasattr(adapter, "can_safely_pray") and adapter.can_safely_pray(bb.obs.hero.turn):
            return Action(name="pray")
        return Status.FAILURE

    def handle_eat_food(bb: Blackboard, args):
        for it in bb.obs.inventory:
            if it.category == "food":
                return Action(name="eat_carried_food", slot=it.slot)
        return Status.FAILURE

    def handle_melee_attack(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        glyphs = bb.obs.glyphs
        if glyphs is None:
            return Status.FAILURE
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ty, tx = hy + dy, hx + dx
                if 0 <= ty < 21 and 0 <= tx < 79:
                    g = int(glyphs[ty, tx])
                    if nh.glyph_is_monster(g) and not nh.glyph_is_pet(g):
                        if not hasattr(adapter, "is_target_floating_eye") or not adapter.is_target_floating_eye(glyphs, ty, tx):
                            return Action(name="melee_attack_hostile", direction=(dy, dx))
        return Status.FAILURE

    def handle_step_to_stairs(bb: Blackboard, args):
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

    def handle_step_to_stairs_up(bb: Blackboard, args):
        nonlocal known_stairs_up
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        hero_pos = (hy, hx)
        if known_stairs_up is not None and hero_pos == known_stairs_up:
            return Action(name="ascend")
        if known_stairs_up is not None:
            walkable = build_walkable_mask(bb.obs)
            walkable[known_stairs_up[0], known_stairs_up[1]] = True
            path = SpatialEngine.find_path(hero_pos, known_stairs_up, walkable)
            if path:
                dy = path[0][0] - hy
                dx = path[0][1] - hx
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_step_to_frontier(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        hero_pos = (hy, hx)
        walkable = build_walkable_mask(bb.obs)
        frontier = SpatialEngine.find_nearest_frontier(hero_pos, walkable, getattr(adapter, "visited", set()))
        if frontier is not None:
            path = SpatialEngine.find_path(hero_pos, frontier, walkable)
            if path:
                dy = path[0][0] - hy
                dx = path[0][1] - hx
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_search(bb: Blackboard, args):
        return Action(name="search")

    def handle_descend(bb: Blackboard, args):
        return Action(name="descend")

    def handle_ascend(bb: Blackboard, args):
        return Action(name="ascend")

    def handle_wait(bb: Blackboard, args):
        return Action(name="wait")

    def handle_engrave_elbereth(bb: Blackboard, args):
        if hasattr(adapter, "has_elbereth_at") and not adapter.has_elbereth_at(bb.obs.hero.y, bb.obs.hero.x):
            return Action(name="engrave_elbereth")
        return Status.FAILURE

    def handle_open_door(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        chars = bb.obs.chars
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ty, tx = hy + dy, hx + dx
            if 0 <= ty < 21 and 0 <= tx < 79 and chars[ty, tx] == ord("+"):
                return Action(name="open_door", direction=(dy, dx))
        return Status.FAILURE

    def handle_kick_closed_door(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        chars = bb.obs.chars
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ty, tx = hy + dy, hx + dx
            if 0 <= ty < 21 and 0 <= tx < 79 and chars[ty, tx] == ord("+"):
                return Action(name="kick_closed_door", direction=(dy, dx))
        return Status.FAILURE

    def handle_eat_floor_corpse(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        chars = bb.obs.chars
        if chars[hy, hx] == ord("%"):
            return Action(name="eat_floor_corpse")
        return Status.FAILURE

    def handle_step_away_from_hostile(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        walkable = build_walkable_mask(bb.obs)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ty, tx = hy + dy, hx + dx
            if 0 <= ty < 21 and 0 <= tx < 79 and walkable[ty, tx]:
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_wield_weapon(bb: Blackboard, args):
        for it in bb.obs.inventory:
            if it.category == "weapon" and any(w in it.name.lower() for w in ["sword", "excalibur", "dagger", "spear"]):
                return Action(name="wield_weapon", slot=it.slot)
        return Status.FAILURE

    def handle_wear_armor(bb: Blackboard, args):
        for it in bb.obs.inventory:
            if it.category == "armor":
                return Action(name="wear_armor", slot=it.slot)
        return Status.FAILURE

    def handle_step_to_fountain(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        hero_pos = (hy, hx)
        fountain_locs = np.argwhere(bb.obs.chars == ord("{"))
        if len(fountain_locs) > 0:
            target = (int(fountain_locs[0, 0]), int(fountain_locs[0, 1]))
            walkable = build_walkable_mask(bb.obs)
            walkable[target[0], target[1]] = True
            path = SpatialEngine.find_path(hero_pos, target, walkable)
            if path:
                dy = path[0][0] - hy
                dx = path[0][1] - hx
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_dip_excalibur(bb: Blackboard, args):
        hy, hx = bb.obs.hero.y, bb.obs.hero.x
        chars = bb.obs.chars
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            ty, tx = hy + dy, hx + dx
            if 0 <= ty < 21 and 0 <= tx < 79 and chars[ty, tx] == ord("{"):
                return Action(name="dip_excalibur")
        return Status.FAILURE

    action_handlers = {
        "quaff_healing": handle_quaff_healing,
        "pray": handle_emergency_pray,
        "eat_carried_food": handle_eat_food,
        "melee_attack_hostile": handle_melee_attack,
        "step_to_stairs_down": handle_step_to_stairs,
        "step_to_stairs_up": handle_step_to_stairs_up,
        "step_to_stairs": handle_step_to_stairs,
        "step_to_frontier": handle_step_to_frontier,
        "search": handle_search,
        "descend": handle_descend,
        "ascend": handle_ascend,
        "wait": handle_wait,
        "engrave_elbereth": handle_engrave_elbereth,
        "open_door": handle_open_door,
        "kick_closed_door": handle_kick_closed_door,
        "eat_floor_corpse": handle_eat_floor_corpse,
        "step_away_from_hostile": handle_step_away_from_hostile,
        "retreat": handle_step_away_from_hostile,
        "step_to_chokepoint": handle_step_away_from_hostile,
        "step_to_dead_end": handle_search,
        "rest": handle_wait,
        "idle": handle_wait,
        "wield_weapon": handle_wield_weapon,
        "wear_armor": handle_wear_armor,
        "step_to_fountain": handle_step_to_fountain,
        "dip_excalibur": handle_dip_excalibur,
    }

    current_tree = compile_policy(current_policy, action_handlers=action_handlers)

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
            known_stairs_down = None
            known_stairs_up = None
            last_position = (-1, -1)
            stuck_counter = 0
            ep_turns = 0
            death_reason = "active"
            max_depth_reached = obs.hero.depth if obs.hero.depth > 0 else 1
            last_valid_gold = obs.hero.gold if hasattr(obs.hero, "gold") else 0
            last_5_actions: list[str] = []
            turns_dl1 = 0
            turns_dl2 = 0
            turns_mines = 0
            inventory_at_death_str = ""

            policy_runner = current_tree.create_runner(obs) if hasattr(current_tree, "create_runner") else None

            has_healing = any(
                it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"])
                for it in obs.inventory
            )
            has_food = any(it.category == "food" for it in obs.inventory)

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

                # Stairs detection
                stairs_down_loc = np.argwhere(obs.chars == ord(">"))
                if len(stairs_down_loc) > 0:
                    known_stairs_down = (int(stairs_down_loc[0, 0]), int(stairs_down_loc[0, 1]))
                stairs_up_loc = np.argwhere(obs.chars == ord("<"))
                if len(stairs_up_loc) > 0:
                    known_stairs_up = (int(stairs_up_loc[0, 0]), int(stairs_up_loc[0, 1]))

                # Stuck / frontier detection
                if (hy, hx) == last_position:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                last_position = (hy, hx)

                # Adjacent hostiles
                has_adj_hostile = False
                hostile_count_fov = 0
                if obs.glyphs is not None:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ty, tx = hy + dy, hx + dx
                            if 0 <= ty < 21 and 0 <= tx < 79:
                                g = int(obs.glyphs[ty, tx])
                                if nh.glyph_is_monster(g) and not nh.glyph_is_pet(g):
                                    has_adj_hostile = True
                                    hostile_count_fov += 1

                # Environmental detection
                adj_closed_door = False
                adj_fountain = False
                chars = obs.chars
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ty, tx = hy + dy, hx + dx
                    if 0 <= ty < 21 and 0 <= tx < 79:
                        if chars[ty, tx] == ord("+"):
                            adj_closed_door = True
                        elif chars[ty, tx] == ord("{"):
                            adj_fountain = True

                is_blind = False
                is_poisoned = False
                if obs.message:
                    msg_low = obs.message.lower()
                    if "blind" in msg_low or "can't see" in msg_low:
                        is_blind = True
                    if "poison" in msg_low or "faint" in msg_low:
                        is_poisoned = True

                has_frontier = bool(stuck_counter < 6)

                memory = {
                    "hp_frac": hero.hp / max(1, hero.max_hp),
                    "energy_frac": hero.energy / max(1, hero.max_energy) if hasattr(hero, "energy") else 1.0,
                    "depth": hero.depth,
                    "turn": hero.turn,
                    "turns_on_level": step,
                    "experience_level": hero.experience_level if hasattr(hero, "experience_level") else 1,
                    "gold": hero.gold if hasattr(hero, "gold") else 0,
                    "hunger_state": hero.hunger_state.value if hasattr(hero.hunger_state, "value") else 1,
                    "is_blind": is_blind,
                    "is_poisoned": is_poisoned,
                    "adjacent_hostile": has_adj_hostile,
                    "hostile_count_fov": hostile_count_fov,
                    "is_surrounded": hostile_count_fov >= 3,
                    "in_corridor": bool(obs.chars[hy, hx] == ord("#")),
                    "can_retreat": True,
                    "standing_on_elbereth": getattr(adapter, "has_elbereth_at", lambda y, x: False)(hy, hx),
                    "can_safely_pray": getattr(adapter, "can_safely_pray", lambda t: True)(hero.turn),
                    "has_healing": has_healing,
                    "adjacent_closed_door": adj_closed_door,
                    "adjacent_fountain": adj_fountain,
                    "stairs_down_known": known_stairs_down is not None,
                    "stairs_up_known": known_stairs_up is not None or bool(np.any(obs.chars == ord("<"))),
                    "standing_on_stairs_down": bool(obs.chars[hy, hx] == ord(">")),
                    "standing_on_stairs_up": bool(obs.chars[hy, hx] == ord("<")),
                    "floor_explored": not has_frontier,
                    "has_unvisited_frontier": has_frontier,
                    "has_unsearched_dead_end": bool(stuck_counter >= 3 and not has_frontier),
                    "has_carried_food": has_food,
                    "floor_corpse_adjacent": bool(obs.chars[hy, hx] == ord("%")),
                    "corpse_is_fresh": True,
                    "corpse_is_safe": True,
                    "has_poison_res": getattr(adapter, "has_poison_res", False),
                    "can_forge_excalibur": (hero.experience_level >= 5) if hasattr(hero, "experience_level") else False,
                }

                if policy_runner is not None:
                    action = policy_runner.send(obs)
                else:
                    action = current_tree.execute(obs, memory=memory)
                if action is None:
                    action = Action(name="search")

                last_5_actions.append(action.name)
                if len(last_5_actions) > 5:
                    last_5_actions.pop(0)

                tile_type = getattr(obs.dungeon, "tile_type", "room") if hasattr(obs, "dungeon") else "room"
                closest_name = getattr(obs.combat, "closest_hostile_name", "") if hasattr(obs, "combat") else ""
                closest_dist = getattr(obs.combat, "closest_hostile_dist", 99.0) if hasattr(obs, "combat") else 99.0
                hostiles_fov = getattr(obs.combat, "hostile_count_fov", hostile_count_fov) if hasattr(obs, "combat") else hostile_count_fov
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

                if done:
                    death_reason = info.get("death_reason", "died") if hasattr(info, "get") else "ended"
                    if "death" in death_reason.lower() or "killed" in death_reason.lower() or "starv" in death_reason.lower() or hero.hp <= 0:
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
            action_handlers=action_handlers,
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
            try:
                with open(policy_path, "w") as f:
                    f.write(new_code.strip() + "\n")
                ckpt_path = f"data/policies/gen_{gen:04d}.py"
                with open(ckpt_path, "w") as f:
                    f.write(new_code.strip() + "\n")

                con = duckdb.connect(db_path)
                con.execute(
                    "INSERT INTO evolved_policies VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [gen, run_id, float(avg_d), int(max_d), float(avg_t), new_code.strip(), datetime.datetime.now()],
                )
                con.close()
                print(f"[Checkpoint Saved] -> {policy_path} & {ckpt_path}")
            except Exception as e:
                print(f"[Warning] Failed to save policy checkpoint: {e}")

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
    )
