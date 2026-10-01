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
import numpy as np
import nle.nethack as nh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    eval_episodes: int = 2,
    max_turns: int = 300,
    stall_threshold: int = 80,
    cluster_threshold: int = 2,
    db_path: str = "data/lox.duckdb",
):
    print("=" * 65)
    print("LOX 2.0 Embodied Dynamic Policy Synthesis Engine")
    print(f"Provider:    {provider} | Model: {model or 'default'}")
    print(f"Environment: {env_type.upper()} ({role if env_type == 'nethack' else task})")
    print(f"Database:    {db_path}")
    print(f"Config:      {max_generations} gens | {eval_episodes} eps/gen | {max_turns} max turns")
    print("=" * 65)

    run_id = f"synth_{provider}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    author = AuthorAgent(provider=provider, model=model, api_key=api_key, db_path=db_path)
    trigger_engine = DynamicTriggerEngine(stall_threshold=stall_threshold, cluster_threshold=cluster_threshold)
    recorder = FlightRecorder(capacity=100)

    # Initialize Environment
    if env_type == "nethack":
        adapter = NetHackAdapter(env_id="NetHackChallenge-v0", role=role)
    else:
        adapter = MiniHackAdapter(task=task)

    # Initial minimal baseline policy
    current_policy = """
def emergency():
    if hp_frac < 0.25 and can_safely_pray:
        pray()
    elif hp_frac < 0.50 and has_healing:
        quaff_healing()
    elif hunger_state >= HUNGRY and has_carried_food:
        eat_carried_food()

def combat():
    if adjacent_hostile:
        melee_attack_hostile()

def explore():
    if stairs_down_known:
        step_to_stairs_down()
    elif has_unvisited_frontier:
        step_to_frontier()
    else:
        search()

plan = [
    emergency,
    combat,
    explore,
]
"""
    print("\n[Generation 0] Initial Seed Policy:")
    print(current_policy.strip())

    # Shared action handlers
    known_stairs_down: tuple[int, int] | None = None

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

    action_handlers = {
        "quaff_healing": handle_quaff_healing,
        "pray": handle_emergency_pray,
        "eat_carried_food": handle_eat_food,
        "melee_attack_hostile": handle_melee_attack,
        "step_to_stairs_down": handle_step_to_stairs,
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
        "step_to_chokepoint": handle_step_away_from_hostile,
        "step_to_dead_end": handle_search,
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
            ep_id = f"g{gen:03d}_e{ep_idx+1}"
            obs = adapter.reset(seed=(gen * 1000 + ep_idx))
            known_stairs_down = None
            last_position = (-1, -1)
            stuck_counter = 0
            ep_turns = 0
            death_reason = "active"

            has_healing = any(
                it.category == "potion" and any(k in it.name.lower() for k in ["heal", "extra heal"])
                for it in obs.inventory
            )
            has_food = any(it.category == "food" for it in obs.inventory)

            for step in range(max_turns):
                ep_turns += 1
                hero = obs.hero
                hy, hx = hero.y, hero.x

                # Stairs detection
                stairs_loc = np.argwhere(obs.chars == ord(">"))
                if len(stairs_loc) > 0:
                    known_stairs_down = (int(stairs_loc[0, 0]), int(stairs_loc[0, 1]))

                # Stuck / frontier detection
                if (hy, hx) == last_position:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                last_position = (hy, hx)

                # Adjacent hostiles
                has_adj_hostile = False
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
                                    break
                        if has_adj_hostile:
                            break

                has_frontier = bool(stuck_counter < 6)

                memory = {
                    "adjacent_hostile": has_adj_hostile,
                    "hostile_count_fov": 1 if has_adj_hostile else 0,
                    "is_surrounded": False,
                    "in_corridor": bool(obs.chars[hy, hx] == ord("#")),
                    "can_retreat": True,
                    "standing_on_elbereth": False,
                    "stairs_down_known": known_stairs_down is not None,
                    "stairs_up_known": bool(np.any(obs.chars == ord("<"))),
                    "standing_on_stairs_down": bool(obs.chars[hy, hx] == ord(">")),
                    "standing_on_stairs_up": bool(obs.chars[hy, hx] == ord("<")),
                    "floor_explored": not has_frontier,
                    "has_unvisited_frontier": has_frontier,
                    "has_unsearched_dead_end": False,
                    "has_carried_food": has_food,
                    "floor_corpse_adjacent": bool(obs.chars[hy, hx] == ord("%")),
                    "corpse_is_fresh": True,
                    "corpse_is_safe": True,
                    "can_safely_pray": getattr(adapter, "can_safely_pray", lambda t: True)(hero.turn),
                    "has_healing": has_healing,
                    "can_forge_excalibur": False,
                    "turns_on_level": step,
                }

                action = current_tree.execute(obs, memory=memory)
                if action is None:
                    action = Action(name="search")

                recorder.record_turn(
                    turn=hero.turn,
                    depth=hero.depth,
                    hp=hero.hp,
                    max_hp=hero.max_hp,
                    hunger=hero.hunger_state.name,
                    pos=(hy, hx),
                    action_name=action.name,
                    message=obs.message,
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
                )

                # Check turn trigger (real floor stall or starvation)
                trig, reason = trigger_engine.check_turn(
                    turns_on_level=step,
                    depth=hero.depth,
                    has_frontier=has_frontier,
                    hunger_state=hero.hunger_state.name,
                    food_count=1 if has_food else 0,
                )
                if trig in (TriggerType.STALL, TriggerType.STARVATION):
                    trigger_fired = True
                    trigger_reason = reason

                step_res = adapter.step(action)
                if len(step_res) == 5:
                    obs, reward, term, trunc, info = step_res
                    done = bool(term or trunc)
                else:
                    obs, reward, done, info = step_res

                if done:
                    death_reason = info.get("death_reason", "died") if hasattr(info, "get") else "ended"
                    if "death" in death_reason.lower() or "killed" in death_reason.lower() or "starv" in death_reason.lower():
                        recorder.record_death(death_reason)
                        trig_c, r_c = trigger_engine.check_death_cluster(recorder)
                        if trig_c == TriggerType.CLUSTER_DEATH:
                            trigger_fired = True
                            trigger_reason = r_c
                    break

            gen_depths.append(obs.hero.depth)
            gen_turns.append(ep_turns)

            logger.log_episode(
                episode_id=ep_id,
                depth=obs.hero.depth,
                score=obs.hero.score if hasattr(obs.hero, "score") else 0,
                turns=ep_turns,
                death_reason=death_reason,
                solved=False,
                wall_sec=time.perf_counter() - gen_start_t,
                role=role if env_type == "nethack" else "minihack",
                gold=obs.hero.gold if hasattr(obs.hero, "gold") else 0,
                max_depth=obs.hero.depth,
                steps=ep_turns,
            )

        # Consolidate raw telemetry into DuckDB so LLM tools query live empirical state
        logger.flush()
        consolidate_run(run_id=gen_dir_id, db_path=db_path, telemetry_dir="data/telemetry", cleanup=True)

        avg_d = np.mean(gen_depths)
        avg_t = np.mean(gen_turns)
        print(f"[Gen {gen} Summary] Avg Depth: {avg_d:.1f} | Avg Turns: {avg_t:.1f} | Trigger: {trigger_reason or 'None'}")

        if trigger_fired:
            print(f"[Trigger Fired] {trigger_reason}")
            print("\n[Author Agent] Initiating synthesis session (querying DuckDB & evaluating)...")
            status_rep = recorder.generate_compact_status_report(trigger_reason=trigger_reason)
            new_code, tree, error = author.synthesize_policy(
                current_policy=current_policy,
                trigger_reason=trigger_reason,
                status_report=status_rep,
                run_id=run_id,
                action_handlers=action_handlers,
            )

            if error:
                print(f"[Validation Failed] {error}")
            else:
                print(f"[Policy Verified & Compiled! Generation {gen} accepted]")
                current_policy = new_code
                current_tree = tree
                print("\nEvolved Policy Program:")
                print(new_code.strip())

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
    parser.add_argument("--eval-episodes", type=int, default=2, help="Real evaluation episodes per generation")
    parser.add_argument("--max-turns", type=int, default=300, help="Max turns per episode")
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
        stall_threshold=args.stall_threshold,
        cluster_threshold=args.cluster_threshold,
        db_path=args.db_path,
    )
