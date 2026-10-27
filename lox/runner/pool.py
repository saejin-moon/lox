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
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import CompiledPolicy, compile_policy_and_macros
from lox.lang.instrument import PolicyLoopError
from lox.ui.console import ui

# Process-local persistent state inside worker processes
_WORKER_HARNESS: NetHackHarness | None = None
_WORKER_ROLE: str = "val-hum-law-fem"
_WORKER_MAX_STEPS: int = 2500
_WORKER_COMPILED_CACHE: tuple[int, int, CompiledPolicy] | None = None


def _init_worker(role: str, max_steps: int) -> None:
    # Ignore SIGINT in worker processes so Ctrl+C in the terminal is handled cleanly by the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_HARNESS, _WORKER_ROLE, _WORKER_MAX_STEPS
    _WORKER_ROLE = role
    _WORKER_MAX_STEPS = max_steps
    _WORKER_HARNESS = NetHackHarness(role=role, max_steps=max_steps)


def _get_worker_harness(role: str, max_steps: int) -> NetHackHarness:
    global _WORKER_HARNESS, _WORKER_ROLE, _WORKER_MAX_STEPS
    if (
        _WORKER_HARNESS is None
        or _WORKER_ROLE != role
        or _WORKER_MAX_STEPS != max_steps
    ):
        if _WORKER_HARNESS is not None:
            _WORKER_HARNESS.close()
        _WORKER_ROLE = role
        _WORKER_MAX_STEPS = max_steps
        _WORKER_HARNESS = NetHackHarness(role=role, max_steps=max_steps)
    return _WORKER_HARNESS


def _get_compiled_policy(policy_code: str, macros_code: str) -> CompiledPolicy:
    global _WORKER_COMPILED_CACHE
    h_pol = hash(policy_code)
    h_mac = hash(macros_code)
    if (
        _WORKER_COMPILED_CACHE is not None
        and _WORKER_COMPILED_CACHE[0] == h_pol
        and _WORKER_COMPILED_CACHE[1] == h_mac
    ):
        return _WORKER_COMPILED_CACHE[2]
    compiled = compile_policy_and_macros(policy_code, macros_code)
    _WORKER_COMPILED_CACHE = (h_pol, h_mac, compiled)
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

    harness = _get_worker_harness(role, max_steps)
    compiled = _get_compiled_policy(policy_code, macros_code)
    runner = compiled.create_runner(param_overrides=param_overrides)

    state = harness.reset(seed=seed)
    max_depth = max(1, state.depth)
    last_score = state.score
    last_gold = state.gold
    last_ac = state.ac
    last_xl = state.xp_level
    last_hp = state.hp
    last_max_hp = state.max_hp
    last_turn = state.turn
    last_active_fn = "root"
    end_reason = "ended"
    death_msg = ""

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

        if next_state.depth > max_depth:
            max_depth = next_state.depth
        if next_state.score >= 0:
            last_score = next_state.score
        last_gold = next_state.gold
        last_ac = next_state.ac
        if next_state.xp_level > 0:
            last_xl = next_state.xp_level
        last_hp = next_state.hp
        if next_state.max_hp > 0:
            last_max_hp = next_state.max_hp
        if next_state.turn >= last_turn:
            last_turn = next_state.turn

        if record_ticks:
            raw_key = int(harness.action_bytes[-1]) if harness.action_bytes else 0
            vis_mons = ""
            if done or step_idx % 5 == 0 or last_hp < state.hp:
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
                    "depth": next_state.depth,
                    "dungeon_num": next_state.dungeon_num,
                    "hp": last_hp,
                    "max_hp": last_max_hp,
                    "energy": next_state.energy,
                    "ac": last_ac,
                    "xp_level": last_xl,
                    "score": last_score,
                    "hunger_state": next_state.hunger_state,
                    "condition_mask": next_state.condition_mask,
                    "y": next_state.y,
                    "x": next_state.x,
                    "action_name": info["action_name"],
                    "raw_key": raw_key,
                    "active_function": active_fn,
                    "message": next_state.message,
                    "visible_monsters": vis_mons,
                    "inventory_summary": "",
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
    if target_functions and not runner.tracer.triggered_any(target_functions):
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
    summary_digest, game_log, milestones = (
        summarize_episode_ticks(ticks)
        if record_ticks and ticks
        else (
            f"Overview: Depth={state.depth} (Max={max_depth}) | Turns={last_turn} | Score={last_score}",
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
        "run_id": run_id,
        "seed": seed,
        "role": role,
        "depth": state.depth,
        "max_depth": max_depth,
        "score": last_score,
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
    ):
        self.workers = max(1, int(workers))
        self.role = role
        self.max_steps = max_steps
        self._pool: mp.pool.Pool | None = None

    def _ensure_pool(self) -> mp.pool.Pool | None:
        if self.workers <= 1:
            return None
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(self.role, self.max_steps),
            )
        return self._pool

    def evaluate_batch(
        self,
        batch_id: str,
        policy_code: str,
        macros_code: str,
        target_episodes: int,
        target_functions: list[str] | None = None,
        priority_seeds: list[int] | None = None,
        seed_start: int = 10000,
        max_raw_multiplier: int = 8,
        param_overrides: dict[str, float] | None = None,
        record_ticks: bool = True,
        show_progress: bool = True,
        early_stop_fn: Callable[[list[dict[str, Any]]], bool] | None = None,
        unreachable_abort_after: int = 500,
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
        max_raw_episodes = (
            max(target_episodes * max_raw_multiplier, 2000)
            if target_fns
            else target_episodes
        )

        # Build seed queue: priority_seeds first, then sequential fresh seeds
        used_seeds: set[int] = set()
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
            batch_capacity = min(
                max(needed * 2 if target_fns else needed, self.workers * 8),
                max_raw_episodes - raw_done,
            )
            if batch_capacity <= 0:
                break

            tasks: list[dict[str, Any]] = []
            for _ in range(batch_capacity):
                if seed_queue:
                    s = seed_queue.pop(0)
                else:
                    while next_seed_cursor in used_seeds:
                        next_seed_cursor += 1
                    s = next_seed_cursor
                    used_seeds.add(s)
                    next_seed_cursor += 1

                raw_idx = raw_done + len(tasks) + 1
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
                    }
                )

            chunksize = max(1, min(8, len(tasks) // max(1, self.workers * 2)))
            if pool is not None:
                results_iter = pool.imap_unordered(
                    run_single_episode, tasks, chunksize=chunksize
                )
            else:
                results_iter = (run_single_episode(tk) for tk in tasks)

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

                if show_progress and (
                    raw_done % max(1, self.workers) == 0
                    or len(triggered_runs) >= target_episodes
                ):
                    elapsed = max(1e-6, time.perf_counter() - t0)
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
        depths = [r["max_depth"] for r in triggered_runs]
        scores = [r["score"] for r in triggered_runs]
        turns = [r["turns"] for r in triggered_runs]

        return {
            "batch_id": batch_id,
            "runs": triggered_runs,
            "episodes_recorded": len(triggered_runs),
            "episodes_discarded": discarded_count,
            "raw_episodes_run": raw_done,
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "p90_depth": float(np.percentile(depths, 90)) if depths else 0.0,
            "max_depth": int(np.max(depths)) if depths else 0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": int(np.max(scores)) if scores else 0,
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
        chunksize = max(1, min(8, len(tasks) // max(1, self.workers * 2)))
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
