"""
Persistent Multi-Core NLE Worker Pool & Conditional Function-Trigger Evaluator.
- Each worker process initializes a `NetHackHarness` once and reuses it across episodes.
- Compiles `policy_code` + `macros_code` once per batch per worker.
- Enforces Conditional Episode Filtering: in `FUNC_MOD_ADD` mode, episodes that do not
  trigger any modified/added function (`triggered == False`) are immediately discarded
  and excluded from stats and database storage.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lox.core.contracts import (
    BatchEvaluation,
    EpisodeRun,
    split_full_and_shard_runs,
)
from lox.db.summarizer import summarize_episode_ticks
from lox.env import get_adapter
from lox.env.actions import Action
from lox.env.base import EnvHarness
from lox.lang.compiler import CompiledPolicy, compile_policy_package
from lox.lang.instrument import PolicyLoopError
from lox.ui.console import ui

# Process-local persistent state inside worker processes
_WORKER_HARNESS: EnvHarness | None = None
_WORKER_ROLE: str = "val-hum-law-fem"
_WORKER_MAX_STEPS: int = 5000
_WORKER_BENCHMARK: str = "nethack"
_WORKER_COMPILED_CACHE: tuple[tuple[int, int, int], CompiledPolicy] | None = None


def _init_worker(role: str, max_steps: int, benchmark: str = "nethack") -> None:
    # Ignore SIGINT in worker processes so Ctrl+C in the terminal is handled cleanly by the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_HARNESS, _WORKER_ROLE, _WORKER_MAX_STEPS, _WORKER_BENCHMARK
    _WORKER_ROLE = role
    _WORKER_MAX_STEPS = max_steps
    _WORKER_BENCHMARK = benchmark
    _WORKER_HARNESS = get_adapter(benchmark).create_harness(
        role=role, max_steps=max_steps
    )


def _get_worker_harness(
    role: str, max_steps: int, benchmark: str = "nethack"
) -> EnvHarness:
    global _WORKER_HARNESS, _WORKER_ROLE, _WORKER_MAX_STEPS, _WORKER_BENCHMARK
    if (
        _WORKER_HARNESS is None
        or _WORKER_ROLE != role
        or _WORKER_BENCHMARK != benchmark
    ):
        if _WORKER_HARNESS is not None:
            _WORKER_HARNESS.close()
        _WORKER_ROLE = role
        _WORKER_MAX_STEPS = max_steps
        _WORKER_BENCHMARK = benchmark
        _WORKER_HARNESS = get_adapter(benchmark).create_harness(
            role=role, max_steps=max_steps
        )
    else:
        _WORKER_MAX_STEPS = max_steps
        _WORKER_HARNESS.max_steps = max_steps
    return _WORKER_HARNESS


def _get_compiled_policy(
    policy_code: str, macros_code: str, skills: dict[str, str] | None = None
) -> CompiledPolicy:
    global _WORKER_COMPILED_CACHE
    h_pol = hash(policy_code)
    h_mac = hash(macros_code)
    h_skl = hash(tuple(sorted((skills or {}).items())))
    key = (h_pol, h_mac, h_skl)
    if _WORKER_COMPILED_CACHE is not None and _WORKER_COMPILED_CACHE[0] == key:
        return _WORKER_COMPILED_CACHE[1]
    compiled = compile_policy_package(policy_code, macros_code, skills=skills)
    _WORKER_COMPILED_CACHE = (key, compiled)
    return compiled


def run_single_episode(task: dict[str, Any]) -> dict[str, Any]:
    """
    Executes one episode on the persistent worker harness.
    If `target_functions` is non-empty and none of them were called during the episode,
    returns `{"triggered": False, "seed": seed, "steps": steps}` immediately without
    building tick logs or summaries (disappearing into the ether as required).
    """
    t0 = time.perf_counter()
    seed = int(task["seed"])
    run_id = str(task["run_id"])
    role = str(task.get("role", "val-hum-law-fem"))
    max_steps = int(task.get("max_steps", 25000))
    policy_code = str(task["policy_code"])
    macros_code = str(task.get("macros_code", ""))
    target_functions: list[str] = list(task.get("target_functions") or [])
    param_overrides: dict[str, float] | None = task.get("param_overrides")
    record_ticks: bool = bool(task.get("record_ticks", True))
    benchmark = str(task.get("benchmark", "nethack"))
    skills = task.get("skills")

    harness = _get_worker_harness(role, max_steps, benchmark=benchmark)
    compiled = _get_compiled_policy(policy_code, macros_code, skills=skills)
    runner = compiled.create_runner(param_overrides=param_overrides)

    replay_bytes = task.get("replay_bytes")
    post_replay_max_steps = task.get("post_replay_max_steps")
    is_shard = replay_bytes is not None

    if is_shard:
        state = harness.replay_bytes(seed=seed, recorded_bytes=replay_bytes)
        runner.tracer.reset_for_episode()
        max_steps = int(post_replay_max_steps or 200)
        harness.step_count = 0
        harness.max_steps = max_steps
        if harness.done:
            msg = getattr(state, "message", "")
            return {
                "triggered": True,
                "is_shard": True,
                "shard_id": task.get("shard_id"),
                "run_id": run_id,
                "seed": seed,
                "steps": 0,
                "turns": getattr(state, "turn", 0),
                "score": getattr(state, "score", 0),
                "gold": getattr(state, "gold", 0),
                "max_depth": getattr(state, "depth", 1),
                "hp": getattr(state, "hp", 0),
                "max_hp": getattr(state, "max_hp", 0),
                "ac": getattr(state, "ac", 10),
                "xp_level": getattr(state, "xp_level", 1),
                "end_reason": msg.split(".")[0][:60] if msg else "died",
                "death_message": msg[:160],
                "action_bytes": bytes(harness.action_bytes),
                "auto_esc_count": 0,
                "subgen_preemptions": 0,
                "policy_exceptions": 0,
                "unique_tiles": 1,
                "pct_wait": 0.0,
                "pct_search": 0.0,
                "starved_with_food": False,
                "died_unarmored": False,
                "ticks": [],
                "function_stats": {},
                "registered_params": {},
            }
        harness.done = False
        if hasattr(state, "mem") and hasattr(state.mem, "kv"):
            for k, v in state.mem.kv.items():
                if hasattr(runner.instance, k):
                    setattr(runner.instance, k, v)
            setattr(runner.instance, "kv", state.mem.kv)
    else:
        state = harness.reset(seed=seed)
        if hasattr(state, "mem") and hasattr(state.mem, "kv"):
            setattr(runner.instance, "kv", state.mem.kv)
    curr_d = getattr(state, "depth", len(getattr(state, "achievements", ())) + 1)
    max_depth = max(1, curr_d)
    last_score = getattr(state, "score", len(getattr(state, "achievements", ())) * 100)
    last_gold = getattr(state, "gold", 0)
    last_ac = getattr(state, "ac", 10)
    last_xl = getattr(state, "xp_level", 1)
    last_hp = getattr(state, "hp", 9)
    last_max_hp = getattr(state, "max_hp", 9)
    last_turn = getattr(state, "turn", getattr(state, "step_count", 0))
    last_active_fn = "root"
    end_reason = "ended"
    death_msg = ""
    visited_positions = {(state.y, state.x)}
    wait_count = 0
    search_count = 0
    stairs_down_found = bool(
        getattr(state, "stairs_down_pos", None) is not None
        or (hasattr(state, "is_stairs_down") and state.is_stairs_down(state.y, state.x))
        or curr_d > 1
    )

    ticks: list[dict[str, Any]] = []
    last_inv_items: list[Any] = []
    last_inv_summary: str = ""

    def _summarize_inv(st_obj: Any) -> tuple[list[Any], str]:
        if not hasattr(st_obj, "inventory"):
            return [], ""
        try:
            raw_inv = (
                st_obj.inventory() if callable(st_obj.inventory) else st_obj.inventory
            )
            if not raw_inv:
                return [], ""
            items_list = list(raw_inv)
            parts: list[str] = []
            for it in items_list[:12]:
                ltr = getattr(it, "letter", "?")
                nm = getattr(it, "name", "") or getattr(it, "char", "?")
                tags = []
                if getattr(it, "is_wielded", False):
                    tags.append("wielded")
                if getattr(it, "is_worn", False):
                    tags.append("worn")
                elif getattr(it, "is_armor", False):
                    tags.append("unworn")
                if getattr(it, "is_cursed", False):
                    tags.append("cursed")
                tag_s = f"({','.join(tags)})" if tags else ""
                parts.append(f"{ltr}:{nm}{tag_s}")
            return items_list, "; ".join(parts)
        except Exception:
            return [], ""

    last_inv_items, last_inv_summary = _summarize_inv(state)

    for step_idx in range(1, max_steps + 1):
        try:
            raw_act, active_fn = runner.next_action(state)
        except PolicyLoopError as loop_err:
            raw_act = Action.WAIT
            active_fn = runner.tracer.active_fn
            end_reason = f"loop_guard:{active_fn}"
            death_msg = str(loop_err)
            runner.policy_exceptions += 1
            break
        except Exception as exc:
            raw_act = Action.WAIT
            active_fn = runner.tracer.active_fn
            end_reason = f"policy_error:{type(exc).__name__}"
            death_msg = str(exc)[:120]
            runner.policy_exceptions += 1
            break

        is_wait = (
            raw_act == Action.WAIT
            or raw_act == "."
            or (hasattr(raw_act, "bytes_payload") and raw_act.bytes_payload == b".")
        )
        is_search = (
            raw_act == Action.SEARCH
            or raw_act == "s"
            or (hasattr(raw_act, "bytes_payload") and raw_act.bytes_payload == b"s")
        )
        if is_wait:
            wait_count += 1
        elif is_search:
            search_count += 1

        last_active_fn = active_fn
        prev_d = getattr(state, "depth", 1)
        next_state, done, info = harness.step(raw_act)

        visited_positions.add((next_state.y, next_state.x))
        next_d = getattr(
            next_state, "depth", len(getattr(next_state, "achievements", ())) + 1
        )
        if not stairs_down_found and (
            getattr(next_state, "stairs_down_pos", None) is not None
            or (
                hasattr(next_state, "is_stairs_down")
                and next_state.is_stairs_down(next_state.y, next_state.x)
            )
            or next_d > 1
        ):
            stairs_down_found = True

        if next_d > max_depth:
            max_depth = next_d
        next_sc = getattr(
            next_state, "score", len(getattr(next_state, "achievements", ())) * 100
        )
        if next_sc > last_score or (not done and next_sc >= 0):
            last_score = next_sc
        # On the terminal tombstone frame (done=True and next_d<=0), NLE zeroes out blstats (AC=0, gold=0, XL=0).
        # Preserve the hero's true pre-death AC, gold, XL, and max_hp from the last alive state!
        if not done and next_d > 0:
            last_gold = getattr(next_state, "gold", last_gold)
            last_ac = getattr(next_state, "ac", last_ac)
        next_xl = getattr(next_state, "xp_level", 1)
        if next_xl > 0:
            last_xl = next_xl
        last_hp = getattr(next_state, "hp", 0)
        next_max_hp = getattr(next_state, "max_hp", 0)
        if next_max_hp > 0:
            last_max_hp = next_max_hp
        next_turn = getattr(
            next_state, "turn", getattr(next_state, "step_count", step_idx)
        )
        if next_turn >= last_turn:
            last_turn = next_turn

        if not done and (step_idx == 1 or next_d != prev_d or step_idx % 20 == 0):
            cur_items, cur_sum = _summarize_inv(next_state)
            if cur_items:
                last_inv_items, last_inv_summary = cur_items, cur_sum
        elif done:
            cur_items, cur_sum = _summarize_inv(state)
            if cur_items:
                last_inv_items, last_inv_summary = cur_items, cur_sum

        if record_ticks:
            raw_key = int(harness.action_bytes[-1]) if harness.action_bytes else 0
            vis_mons = ""
            if (
                done or step_idx % 5 == 0 or last_hp < getattr(state, "hp", 0)
            ) and hasattr(next_state, "visible_monsters"):
                mons = next_state.visible_monsters()
                if not mons and done and hasattr(state, "visible_monsters"):
                    mons = state.visible_monsters()
                if mons:
                    mon_parts = []
                    for m in mons[:5]:
                        pet_tag = "pet," if getattr(m, "is_pet", False) else ""
                        m_info = getattr(m, "info", None)
                        diff = getattr(m_info, "difficulty", None) if m_info else None
                        d_tag = f"d{diff}" if diff is not None else ""
                        tag_str = f"[{pet_tag}{d_tag}]" if (pet_tag or d_tag) else ""
                        m_nm = getattr(m, "name", None) or (
                            m_info.name if m_info else "?"
                        )
                        mon_parts.append(f"{m.char}({m_nm}{tag_str}@{m.y},{m.x})")
                    vis_mons = ", ".join(mon_parts)
            tick_inv = (
                last_inv_summary
                if (done or step_idx == 1 or next_d != prev_d or step_idx % 20 == 0)
                else ""
            )
            ticks.append(
                {
                    "turn": last_turn,
                    "step_idx": step_idx,
                    "depth": next_d if next_d > 0 else prev_d,
                    "dungeon_num": getattr(next_state, "dungeon_num", 0),
                    "hp": last_hp,
                    "max_hp": last_max_hp,
                    "energy": getattr(next_state, "energy", 0),
                    "ac": last_ac,
                    "xp_level": last_xl,
                    "score": last_score,
                    "hunger_state": getattr(next_state, "hunger_state", 0),
                    "condition_mask": getattr(next_state, "condition_mask", 0),
                    "y": next_state.y,
                    "x": next_state.x,
                    "action_name": info.get("action_name", ""),
                    "raw_key": raw_key,
                    "active_function": active_fn,
                    "message": getattr(next_state, "message", ""),
                    "byte_offset": len(harness.action_bytes),
                    "visible_monsters": vis_mons,
                    "inventory_summary": tick_inv,
                    "achievements": len(getattr(next_state, "achievements", ())),
                }
            )

        state = next_state
        if done:
            msg = state.message
            death_msg = msg[:160]
            if info.get("is_ascended"):
                end_reason = "ascended"
            elif info.get("stalled_zero_turn"):
                end_reason = "stalled_zero_turn"
                if not death_msg:
                    death_msg = f"Zero-turn action stall ({getattr(harness, 'consecutive_zero_turn_steps', 100)} consecutive steps)"
            elif info.get("truncated"):
                end_reason = "max_steps"
            elif (
                last_hp <= 0
                or "die" in msg.lower()
                or "killed" in msg.lower()
                or "starv" in msg.lower()
            ):
                end_reason = msg.split(".")[0][:60] if msg else "died"
            else:
                end_reason = msg.split(".")[0][:60] if msg else "ended"
            break

    # Check Conditional Episode Filter: did the episode actually trigger the target function(s)?
    is_priority_seed = bool(task.get("is_priority_seed", False))
    enforce_trigger = bool(
        task.get("enforce_trigger", not is_priority_seed and not is_shard)
    )
    if (
        enforce_trigger
        and target_functions
        and not runner.tracer.triggered_any(target_functions)
    ):
        return {
            "triggered": False,
            "seed": seed,
            "steps": harness.step_count,
            "registered_params": {
                k: {
                    "name": p.name,
                    "default": p.default,
                    "min_val": p.min_val,
                    "max_val": p.max_val,
                    "is_int": p.is_int,
                }
                for k, p in runner.tracer.registered_params.items()
            },
        }

    wall_ms = (time.perf_counter() - t0) * 1000.0
    if benchmark != "nethack":
        adapter = get_adapter(benchmark)
        summary_digest, game_log, milestones = (
            adapter.extract_milestones_and_stalls(ticks)
            if record_ticks and ticks
            else (
                f"Overview: Progress={curr_d} (Max={max_depth}) | Steps={last_turn} | Score={last_score}",
                death_msg,
                {
                    "depths_visited": [max_depth],
                    "achievements": getattr(state, "achievements", set()),
                },
            )
        )
    else:
        summary_digest, game_log, milestones = (
            summarize_episode_ticks(ticks)
            if record_ticks and ticks
            else (
                f"Overview: Depth={curr_d} (Max={max_depth}) | Turns={last_turn} | Score={last_score}",
                death_msg,
                {"depths_visited": [max_depth], "best_ac": last_ac, "max_xl": last_xl},
            )
        )

    # Check tactical blunder flags at death using the last alive inventory snapshot
    death_low = f"{death_msg} {end_reason}".lower()
    is_starved = (
        "starv" in death_low or "lack of food" in death_low or "faint" in death_low
    )
    starved_with_food = False
    died_unarmored = False

    if not last_inv_items and hasattr(state, "inventory"):
        last_inv_items, last_inv_summary = _summarize_inv(state)

    if last_inv_items:
        try:
            has_food = any(
                getattr(it, "is_food", False)
                or getattr(it, "oclass", None) == 7
                or getattr(it, "char", "") == "%"
                for it in last_inv_items
            )
            starved_with_food = bool(is_starved and has_food)
            has_armor = any(
                getattr(it, "is_armor", False) or getattr(it, "oclass", None) == 3
                for it in last_inv_items
            )
            died_unarmored = bool(
                not info.get("is_ascended") and last_ac >= 10 and has_armor
            )
        except Exception:
            pass

    if isinstance(milestones, dict):
        milestones["final_inventory"] = last_inv_summary
        milestones["final_ac_true"] = last_ac

    total_steps = max(1, harness.step_count)
    pct_wait = round((wait_count / total_steps) * 100.0, 2)
    pct_search = round((search_count / total_steps) * 100.0, 2)

    # Preserve 100% full, lossless ticks for exact SQL window queries and action aggregations
    full_ticks = ticks

    return EpisodeRun(
        {
            "triggered": True,
            "is_shard": is_shard,
            "shard_id": str(task.get("shard_id", "")),
            "run_id": run_id,
            "seed": seed,
            "role": role,
            "depth": state.depth,
            "max_depth": max_depth,
            "score": last_score,
            "achievements": len(getattr(state, "achievements", ()))
            if hasattr(state, "achievements")
            else 0,
            "turns": last_turn,
            "steps": harness.step_count,
            "xp_level": last_xl,
            "final_xp_level": last_xl,
            "final_hp": last_hp,
            "max_hp": last_max_hp,
            "final_max_hp": last_max_hp,
            "final_ac": last_ac,
            "gold": last_gold,
            "end_reason": end_reason,
            "death_message": death_msg,
            "summary_digest": summary_digest,
            "game_log": game_log,
            "milestones": milestones,
            "auto_esc_count": runner.auto_esc_count
            + int(getattr(harness, "prompt_aborts", 0)),
            "subgen_preemptions": runner.subgen_preemptions,
            "policy_exceptions": runner.policy_exceptions
            + len(
                getattr(
                    getattr(getattr(runner, "instance", None), "blackboard", None),
                    "leaf_errors",
                    (),
                )
            ),
            "unique_tiles": len(visited_positions),
            "pct_wait": pct_wait,
            "pct_search": pct_search,
            "starved_with_food": starved_with_food,
            "died_unarmored": died_unarmored,
            "stairs_down_found": stairs_down_found,
            "triggered_functions": dict(runner.tracer.fn_calls),
            "functions_triggered": dict(runner.tracer.fn_calls),
            "fn_turns": dict(runner.tracer.fn_turns),
            "function_turns": dict(runner.tracer.fn_turns),
            "last_active_function": last_active_fn,
            "action_bytes": bytes(harness.action_bytes),
            "step_byte_offsets": list(getattr(harness, "step_byte_offsets", ())),
            "wall_ms": round(wall_ms, 2),
            "ticks": full_ticks,
            "registered_params": {
                k: {
                    "name": p.name,
                    "default": p.default,
                    "min_val": p.min_val,
                    "max_val": p.max_val,
                    "is_int": p.is_int,
                }
                for k, p in runner.tracer.registered_params.items()
            },
        }
    )


class PersistentEvalPool:
    """
    Manages a persistent pool of NLE worker processes across generations.
    Supports:
    - Fixed episode budgets (`FULL_WRITE`: 10,000 eps)
    - Conditional function-triggered budgets (`FUNC_MOD_ADD`: 1,000 triggered eps per modified function,
      discarding any episode that doesn't call the target function)
    - Sequential early stopping callbacks (`early_stop_fn`)
    """

    def __init__(
        self,
        workers: int = 8,
        role: str = "val-hum-law-fem",
        max_steps: int = 25000,
        benchmark: str = "nethack",
    ):
        self.workers = max(1, int(workers))
        self.role = role
        self.max_steps = max_steps
        self.benchmark = benchmark
        self._pool: mp.pool.Pool | None = None

    def _ensure_pool(self) -> mp.pool.Pool | None:
        if self.workers <= 1:
            return None
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(self.role, self.max_steps, self.benchmark),
            )
        return self._pool

    def evaluate_batch(
        self,
        batch_id: str,
        policy_code: str,
        macros_code: str = "",
        target_episodes: int = 30,
        target_functions: list[str] | None = None,
        priority_seeds: list[int] | None = None,
        unconditional_seeds: list[int] | set[int] | None = None,
        seed_start: int = 10000,
        max_raw_multiplier: int = 8,
        param_overrides: dict[str, float] | None = None,
        record_ticks: bool = True,
        show_progress: bool = True,
        early_stop_fn: Callable[[list[dict[str, Any]]], bool] | None = None,
        unreachable_abort_after: int = 500,
        shards: list[Any] | None = None,
        skills: dict[str, str] | None = None,
        benchmark: str = "nethack",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Runs episodes until collecting `target_episodes` episodes that trigger `target_functions`
        (or until `max_raw_episodes` is reached or `early_stop_fn` returns True).
        Non-triggering episodes are discarded immediately and never added to `runs`.
        """
        if target_episodes <= 0:
            return {
                "batch_id": batch_id,
                "runs": [],
                "episodes_recorded": 0,
                "episodes_discarded": 0,
                "avg_depth": 0.0,
                "p90_depth": 0.0,
                "max_depth": 0,
                "avg_score": 0.0,
                "max_score": 0,
                "avg_turns": 0.0,
                "steps_per_sec": 0.0,
                "wall_sec": 0.0,
                "early_stopped": False,
                "unreachable_abort": False,
                "registered_params": {},
            }

        effective_max_steps = int(
            max_steps if max_steps is not None else self.max_steps
        )
        target_fns = [f for f in (target_functions or []) if f and f != "__init__"]
        bench = benchmark or self.benchmark
        max_raw_episodes = (
            max(target_episodes * max_raw_multiplier, target_episodes)
            if target_fns
            else target_episodes
        )

        priority_seeds_set = {int(s) for s in (priority_seeds or [])}
        unconditional_seeds_set = (
            {int(s) for s in unconditional_seeds}
            if unconditional_seeds is not None
            else priority_seeds_set
        )
        shard_queue = list(shards) if shards else []

        def _shard_val(sh: Any, attr: str, default: Any) -> Any:
            if hasattr(sh, attr):
                return getattr(sh, attr)
            if isinstance(sh, dict):
                return sh.get(attr, default)
            return default

        shard_seeds = {int(_shard_val(x, "seed", 0)) for x in shard_queue}

        # Build seed queue: priority_seeds first (excluding seeds already executed as shards), then sequential fresh seeds
        used_seeds: set[int] = set(shard_seeds)
        seed_queue: list[int] = []
        if priority_seeds:
            for s in priority_seeds:
                si = int(s)
                if si not in used_seeds:
                    used_seeds.add(si)
                    seed_queue.append(si)

        next_seed_cursor = int(seed_start)
        triggered_runs: list[dict[str, Any]] = []
        discarded_count = 0
        raw_done = 0
        total_steps = 0
        target_fns_called_count = 0
        registered_params: dict[str, Any] = {}
        early_stopped = False
        unreachable_abort = False

        t0 = time.perf_counter()
        pool = self._ensure_pool()

        for attempt in range(2):
            try:
                while (
                    len(triggered_runs) < target_episodes
                    and raw_done < max_raw_episodes
                ):
                    needed = target_episodes - len(triggered_runs)
                    guaranteed_ready = len(shard_queue) + len(seed_queue)
                    if guaranteed_ready > 0 or not target_fns:
                        batch_capacity = min(
                            needed,
                            guaranteed_ready if guaranteed_ready > 0 else needed,
                            max_raw_episodes - raw_done,
                        )
                    else:
                        batch_capacity = min(
                            max(needed * 2, self.workers * 2),
                            max_raw_episodes - raw_done,
                        )
                    if batch_capacity <= 0:
                        break

                    tasks: list[dict[str, Any]] = []
                    for _ in range(batch_capacity):
                        raw_idx = raw_done + len(tasks) + 1
                        if shard_queue:
                            sh = shard_queue.pop(0)
                            s = int(_shard_val(sh, "seed", 0))
                            sh_id = str(_shard_val(sh, "shard_id", ""))
                            sh_bytes = bytes(_shard_val(sh, "action_bytes", b""))
                            post_steps = int(
                                _shard_val(sh, "post_replay_max_steps", 200)
                            )
                            tasks.append(
                                {
                                    "seed": s,
                                    "run_id": f"{batch_id}_s{raw_idx:05d}",
                                    "shard_id": sh_id,
                                    "role": self.role,
                                    "max_steps": post_steps,
                                    "policy_code": policy_code,
                                    "macros_code": macros_code,
                                    "target_functions": target_fns,
                                    "param_overrides": param_overrides,
                                    "record_ticks": record_ticks,
                                    "replay_bytes": sh_bytes,
                                    "post_replay_max_steps": post_steps,
                                    "is_priority_seed": True,
                                    "enforce_trigger": False,
                                    "skills": skills,
                                    "benchmark": bench,
                                }
                            )
                        else:
                            if seed_queue:
                                s = seed_queue.pop(0)
                            else:
                                while next_seed_cursor in used_seeds:
                                    next_seed_cursor += 1
                                s = next_seed_cursor
                                used_seeds.add(s)
                                next_seed_cursor += 1

                            is_pri = s in unconditional_seeds_set
                            tasks.append(
                                {
                                    "seed": s,
                                    "run_id": f"{batch_id}_r{raw_idx:05d}",
                                    "role": self.role,
                                    "max_steps": effective_max_steps,
                                    "policy_code": policy_code,
                                    "macros_code": macros_code,
                                    "target_functions": target_fns,
                                    "param_overrides": param_overrides,
                                    "record_ticks": record_ticks,
                                    "is_priority_seed": is_pri,
                                    "enforce_trigger": not is_pri and bool(target_fns),
                                    "skills": skills,
                                    "benchmark": bench,
                                }
                            )

                    # Use chunksize=1 so each episode is an independent work-stealing unit,
                    # eliminating tail-chunk serialization where 15 workers sit idle.
                    chunksize = 1
                    if pool is not None:
                        results_iter = pool.imap_unordered(
                            run_single_episode, tasks, chunksize=chunksize
                        )
                    else:
                        results_iter = (run_single_episode(tk) for tk in tasks)

                    last_prog_time = t0
                    for res in results_iter:
                        raw_done += 1
                        total_steps += int(res.get("steps", 0))
                        if res.get("registered_params") and not registered_params:
                            registered_params.update(res["registered_params"])

                        if target_fns:
                            fn_trig = (
                                res.get("triggered_functions")
                                or res.get("functions_triggered")
                                or {}
                            )
                            if isinstance(fn_trig, str):
                                try:
                                    fn_trig = json.loads(fn_trig)
                                except Exception:
                                    fn_trig = {}
                            if any(int(fn_trig.get(fn, 0)) > 0 for fn in target_fns):
                                target_fns_called_count += 1

                        if res.get("triggered", False):
                            if len(triggered_runs) < target_episodes:
                                triggered_runs.append(res)
                        else:
                            discarded_count += 1

                        now = time.perf_counter()
                        is_tail = (
                            target_episodes - len(triggered_runs)
                        ) <= self.workers
                        if show_progress and (
                            is_tail
                            or raw_done % max(1, min(self.workers, 4)) == 0
                            or len(triggered_runs) >= target_episodes
                            or (now - last_prog_time) >= 0.5
                        ):
                            last_prog_time = now
                            elapsed = max(1e-6, now - t0)
                            sps = total_steps / elapsed
                            curr_avg_d = (
                                float(np.mean([r["max_depth"] for r in triggered_runs]))
                                if triggered_runs
                                else 0.0
                            )
                            curr_max_d = (
                                int(max(r["max_depth"] for r in triggered_runs))
                                if triggered_runs
                                else 0
                            )
                            ui.progress(
                                triggered_done=len(triggered_runs),
                                triggered_target=target_episodes,
                                raw_done=raw_done,
                                steps_per_sec=sps,
                                avg_depth=curr_avg_d,
                                max_depth=curr_max_d,
                                outliers=0,
                            )

                        # Check termination conditions immediately as episodes finish
                        if len(triggered_runs) >= target_episodes:
                            break

                        # Unreachable function circuit breaker: if >= unreachable_abort_after raw episodes ran and 0 target calls
                        if (
                            target_fns
                            and raw_done
                            >= min(unreachable_abort_after, max_raw_episodes)
                            and target_fns_called_count == 0
                        ):
                            unreachable_abort = True
                            break

                        if early_stop_fn is not None and len(triggered_runs) >= 50:
                            if early_stop_fn(triggered_runs):
                                early_stopped = True
                                break

                    if (
                        len(triggered_runs) >= target_episodes
                        or unreachable_abort
                        or early_stopped
                    ):
                        if (
                            unreachable_abort or early_stopped or raw_done < len(tasks)
                        ) and self.workers > 1:
                            self.close()
                        break
                break
            except Exception as pool_exc:
                if attempt == 0 and self.workers > 1:
                    ui.warn(
                        f"Worker pool encountered transient error ({pool_exc}); "
                        "re-initializing pool and retrying batch..."
                    )
                    self.close()
                    self._pool = None
                    pool = self._ensure_pool()
                    triggered_runs.clear()
                    raw_done = 0
                    total_steps = 0
                    target_fns_called_count = 0
                    shard_queue = list(shards) if shards else []
                    seed_queue = []
                    if priority_seeds:
                        for s in priority_seeds:
                            si = int(s)
                            if si not in shard_seeds:
                                seed_queue.append(si)
                    used_seeds = set(shard_seeds) | set(seed_queue)
                    next_seed_cursor = int(seed_start)
                    continue
                raise

        wall_sec = time.perf_counter() - t0
        sps = total_steps / max(1e-6, wall_sec)
        full_runs, shard_runs = split_full_and_shard_runs(triggered_runs)
        stat_runs = full_runs if full_runs else triggered_runs
        depths = [r["max_depth"] for r in stat_runs]
        scores = [r["score"] for r in stat_runs]
        turns = [r["turns"] for r in stat_runs]

        return BatchEvaluation(
            {
                "batch_id": batch_id,
                "runs": triggered_runs,
                "full_runs": full_runs,
                "shard_runs": shard_runs,
                "episodes_recorded": len(triggered_runs),
                "episodes_discarded": discarded_count,
                "raw_episodes_run": raw_done,
                "avg_depth": float(np.mean(depths)) if depths else 0.0,
                "p90_depth": float(np.percentile(depths, 90)) if depths else 0.0,
                "max_depth": (
                    int(np.max([r["max_depth"] for r in triggered_runs]))
                    if triggered_runs
                    else 0
                ),
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "max_score": (
                    int(np.max([r["score"] for r in triggered_runs]))
                    if triggered_runs
                    else 0
                ),
                "avg_turns": float(np.mean(turns)) if turns else 0.0,
                "steps_per_sec": round(sps, 1),
                "wall_sec": round(wall_sec, 2),
                "early_stopped": early_stopped,
                "unreachable_abort": unreachable_abort,
                "registered_params": registered_params,
            }
        )

    def evaluate_tasks(
        self,
        tasks: list[dict[str, Any]],
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Executes an explicit list of tasks concurrently across persistent workers.
        Useful for paired SPSA evaluations (theta+ and theta-) to maximize worker saturation.
        """
        if not tasks:
            return []
        pool = self._ensure_pool()
        chunksize = 1
        if pool is not None:
            results_iter = pool.imap_unordered(
                run_single_episode, tasks, chunksize=chunksize
            )
        else:
            results_iter = (run_single_episode(tk) for tk in tasks)
        return list(results_iter)

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None
