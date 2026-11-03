"""
Persistent Multi-Core NLE Worker Pool & Conditional Function-Trigger Evaluator.
- Each worker process initializes a `NetHackHarness` once and reuses it across episodes.
- Compiles `policy_code` + `macros_code` once per batch per worker.
- Enforces Conditional Episode Filtering: in `FUNC_MOD_ADD` mode, episodes that do not
  trigger any modified/added function (`triggered == False`) are immediately discarded
  and excluded from stats and database storage.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from typing import Any

import numpy as np

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
        harness.done = False
    else:
        state = harness.reset(seed=seed)
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
    stairs_down_found = bool(
        getattr(state, "stairs_down_pos", None) is not None
        or (hasattr(state, "is_stairs_down") and state.is_stairs_down(state.y, state.x))
        or curr_d > 1
    )

    ticks: list[dict[str, Any]] = []

    for step_idx in range(1, max_steps + 1):
        try:
            raw_act, active_fn = runner.next_action(state)
        except PolicyLoopError as loop_err:
            raw_act = Action.WAIT
            active_fn = runner.tracer.active_fn
            end_reason = f"loop_guard:{active_fn}"
            death_msg = str(loop_err)
            break
        except Exception as exc:
            raw_act = Action.WAIT
            active_fn = runner.tracer.active_fn
            end_reason = f"policy_error:{type(exc).__name__}"
            death_msg = str(exc)[:120]
            break

        last_active_fn = active_fn
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
        if next_sc >= 0:
            last_score = next_sc
        last_gold = getattr(next_state, "gold", 0)
        last_ac = getattr(next_state, "ac", 10)
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

        if record_ticks:
            raw_key = int(harness.action_bytes[-1]) if harness.action_bytes else 0
            vis_mons = ""
            if (
                done or step_idx % 5 == 0 or last_hp < getattr(state, "hp", 0)
            ) and hasattr(next_state, "visible_monsters"):
                mons = next_state.visible_monsters()
                if mons:
                    vis_mons = ", ".join(
                        f"{m.char}({m.info.name if m.info else '?'}@{m.y},{m.x})"
                        for m in mons[:4]
                    )
            ticks.append(
                {
                    "turn": last_turn,
                    "step_idx": step_idx,
                    "depth": next_d,
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
                    "inventory_summary": "",
                    "achievements": len(getattr(next_state, "achievements", ())),
                }
            )

        state = next_state
        if done:
            msg = state.message
            death_msg = msg[:160]
            if info.get("is_ascended"):
                end_reason = "ascended"
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

    # Cap stored ticks per run at 500 (first 100 + every Nth + last 150) so memory never explodes on 25k-step runs
    if len(ticks) > 500:
        mid = ticks[100:-150]
        stride = max(1, len(mid) // 250)
        compact_ticks = ticks[:100] + mid[::stride][:250] + ticks[-150:]
    else:
        compact_ticks = ticks

    return {
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
        "final_hp": last_hp,
        "max_hp": last_max_hp,
        "final_ac": last_ac,
        "gold": last_gold,
        "end_reason": end_reason,
        "death_message": death_msg,
        "summary_digest": summary_digest,
        "game_log": game_log,
        "milestones": milestones,
        "unique_tiles": len(visited_positions),
        "stairs_down_found": stairs_down_found,
        "triggered_functions": dict(runner.tracer.fn_calls),
        "functions_triggered": dict(runner.tracer.fn_calls),
        "fn_turns": dict(runner.tracer.fn_turns),
        "function_turns": dict(runner.tracer.fn_turns),
        "last_active_function": last_active_fn,
        "action_bytes": bytes(harness.action_bytes),
        "wall_ms": round(wall_ms, 2),
        "ticks": compact_ticks,
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

        target_fns = [f for f in (target_functions or []) if f and f != "__init__"]
        bench = benchmark or self.benchmark
        max_raw_episodes = (
            max(target_episodes * max_raw_multiplier, 2000)
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
        registered_params: dict[str, Any] = {}
        early_stopped = False
        unreachable_abort = False

        t0 = time.perf_counter()
        pool = self._ensure_pool()

        while len(triggered_runs) < target_episodes and raw_done < max_raw_episodes:
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
                    post_steps = int(_shard_val(sh, "post_replay_max_steps", 200))
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
                            "max_steps": self.max_steps,
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

                if res.get("triggered", False):
                    if len(triggered_runs) < target_episodes:
                        triggered_runs.append(res)
                else:
                    discarded_count += 1

                now = time.perf_counter()
                is_tail = (target_episodes - len(triggered_runs)) <= self.workers
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

                # Unreachable function circuit breaker: if >= unreachable_abort_after raw episodes ran and 0 triggered
                if (
                    target_fns
                    and raw_done >= min(unreachable_abort_after, max_raw_episodes)
                    and len(triggered_runs) == 0
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
                break

        wall_sec = time.perf_counter() - t0
        sps = total_steps / max(1e-6, wall_sec)
        full_runs = [r for r in triggered_runs if not r.get("is_shard")]
        stat_runs = full_runs if full_runs else triggered_runs
        depths = [r["max_depth"] for r in stat_runs]
        scores = [r["score"] for r in stat_runs]
        turns = [r["turns"] for r in stat_runs]

        return {
            "batch_id": batch_id,
            "runs": triggered_runs,
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
