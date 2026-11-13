"""
Shared Behavior Tree Subtree Evaluation & Condition-Matched Fixture Sampling.
Used by both AuthorToolRegistry (`dry_run_policy`) and SwarmWorker (`evaluate_candidate`)
to test standalone Behavior Tree subtrees against frontier checkpoints and DuckDB failure traces
without duplicating execution loops or falling back to dummy actions when an active scaffold exists.
"""

from __future__ import annotations

import collections
import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lox.db.schema import connect_db
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.bt import BehaviorTree, Blackboard
from lox.lang.compiler import compile_bt_subtree, compile_policy_package

if TYPE_CHECKING:
    from lox.engine.frontier import FrontierArchive, FrontierCheckpoint


@dataclass(slots=True)
class ObservationDigest:
    """Multi-dimensional empirical outcome from evaluating a Behavior Tree subtree."""

    survived: bool = False
    steps_taken: int = 0
    triggered_steps: int = 0
    total_eval_steps: int = 0
    end_reason: str = ""
    hp_start: int = 0
    hp_end: int = 0
    depth_start: int = 1
    depth_end: int = 1
    score_start: int = 0
    score_end: int = 0
    unique_tiles: int = 0
    actions_taken: dict[str, int] = field(default_factory=dict)
    rle_trace: list[str] = field(default_factory=list)
    dialog_resolved: bool = False
    context_label: str = "seed"

    @property
    def hp_delta(self) -> int:
        return self.hp_end - self.hp_start

    @property
    def depth_delta(self) -> int:
        return self.depth_end - self.depth_start

    @property
    def trigger_rate(self) -> float:
        denom = self.total_eval_steps or self.steps_taken
        return (self.triggered_steps / denom) if denom > 0 else 0.0

    def summary(self) -> str:
        status = (
            "SURVIVED"
            if self.survived
            else f"TERMINATED ({self.end_reason or 'death'})"
        )
        top_acts = ", ".join(
            f"{k}:{v}"
            for k, v in sorted(self.actions_taken.items(), key=lambda x: -x[1])[:4]
        )
        total = self.total_eval_steps or self.steps_taken
        base = (
            f"Status: {status} | Context: {self.context_label} | "
            f"Triggered: {self.triggered_steps}/{total} | Steps: {self.steps_taken} | "
            f"HP: {self.hp_end}/{self.hp_start} (Δ{self.hp_delta:+d}) | "
            f"Depth: Dlvl {self.depth_end} (Δ{self.depth_delta:+d}) | "
            f"Unique Tiles: {self.unique_tiles} | Actions: [{top_acts}]"
        )
        if self.rle_trace:
            tail = " -> ".join(self.rle_trace[-5:])
            return f"{base}\n  RLE Trace (tail): {tail}"
        return base


def sample_condition_fixtures(
    db_path: str,
    role: str = "",
    target_metrics: list[str] | None = None,
    generation: int = 1,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """
    Queries DuckDB for up to `limit` recorded failure/frontier trajectories with action bytes
    using domain-agnostic statistical stratification (recent non-trivial failure + high-depth frontier).
    Rewinds ~25 bytes before termination so candidate subtrees run on live mid/late-game states.
    """
    if not db_path or not Path(db_path).exists() or limit <= 0:
        return []

    fixtures: list[dict[str, Any]] = []
    con = None
    try:
        con = connect_db(db_path, read_only=True)
        seen_runs: set[str] = set()
        queries = [
            """
            SELECT run_id, seed, max_depth, end_reason, action_bytes
            FROM runs
            WHERE octet_length(action_bytes) > 10 AND end_reason != 'active'
            ORDER BY generation DESC, turns DESC
            LIMIT ?
            """,
            """
            SELECT run_id, seed, max_depth, end_reason, action_bytes
            FROM runs
            WHERE octet_length(action_bytes) > 10
            ORDER BY max_depth DESC, turns DESC
            LIMIT ?
            """,
        ]
        for sql in queries:
            if len(fixtures) >= limit:
                break
            rows = con.execute(sql, [max(1, limit)]).fetchall()
            for r_id, r_seed, r_depth, r_end, r_bytes in rows:
                r_id_s = str(r_id)
                if r_id_s in seen_runs:
                    continue
                seen_runs.add(r_id_s)
                buf = bytes(r_bytes or b"")
                if not buf:
                    continue
                stop_idx = (
                    max(1, len(buf) - 25) if len(buf) > 30 else max(1, len(buf) - 5)
                )
                fixtures.append(
                    {
                        "run_id": r_id_s,
                        "seed": int(r_seed),
                        "depth": int(r_depth or 1),
                        "end_reason": str(r_end or ""),
                        "action_bytes": buf,
                        "stop_at_byte_idx": stop_idx,
                    }
                )
                if len(fixtures) >= limit:
                    break
    except Exception:
        return fixtures
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    return fixtures


def _tick_node(node: Any, state: Any, bb: Blackboard) -> Any:
    if isinstance(node, BehaviorTree):
        return node.tick(state)
    if hasattr(node, "tick"):
        import inspect

        sig = inspect.signature(node.tick)
        if len(sig.parameters) >= 2:
            res = node.tick(state, bb)
        else:
            res = node.tick(state)
        return res[1] if isinstance(res, tuple) else res
    if hasattr(node, "step"):
        return node.step(state)
    return node(state)


def evaluate_bt_subtree(
    code_snippet: str,
    *,
    checkpoint: FrontierCheckpoint | None = None,
    frontier: FrontierArchive | None = None,
    replay_fixture: dict[str, Any] | None = None,
    seed: int | None = None,
    steps: int = 150,
    baseline_policy_code: str = "",
    skills: dict[str, str] | None = None,
) -> tuple[ObservationDigest, str]:
    """
    Validates, compiles, and empirically executes a Behavior Tree subtree on:
    - a `FrontierCheckpoint`,
    - a DuckDB condition-matched `replay_fixture`, or
    - a fresh NetHack seed.
    If `baseline_policy_code` is provided, falling through (`act is None`) delegates to
    the baseline runner so `triggered_steps` tracks real subtree activation cleanly.
    """
    try:
        node = compile_bt_subtree(code_snippet)
    except Exception as exc:
        tb = traceback.format_exc()
        return ObservationDigest(), f"{type(exc).__name__}: {exc}\n{tb}"

    baseline_runner = None
    if baseline_policy_code and baseline_policy_code.strip():
        try:
            compiled_base = compile_policy_package(
                baseline_policy_code, skills=skills or {}
            )
            baseline_runner = compiled_base.create_runner()
        except Exception:
            baseline_runner = None

    harness = NetHackHarness(max_steps=steps + 50)
    try:
        if replay_fixture is not None:
            buf = bytes(replay_fixture.get("action_bytes") or b"")
            stop_idx = int(replay_fixture.get("stop_at_byte_idx") or len(buf))
            r_seed = int(replay_fixture.get("seed", 42))
            st = harness.replay_bytes(
                seed=r_seed,
                recorded_bytes=buf,
                stop_at_byte_idx=stop_idx,
            )
            harness.step_count = 0
            ctx_label = f"replay:{str(replay_fixture.get('run_id', ''))[-8:]}"
        elif checkpoint is not None and frontier is not None:
            st = frontier.replay_into(harness, checkpoint)
            ctx_label = f"cp:dlvl{checkpoint.depth}_s{checkpoint.seed}"
        else:
            use_seed = int(seed if seed is not None else random.randint(1000, 9999))
            st = harness.reset(seed=use_seed)
            ctx_label = f"seed:{use_seed}"

        start_score = int(st.score)
        start_hp = int(st.hp)
        start_depth = int(st.depth)

        bb = Blackboard(strict_errors=True)
        action_counts: dict[str, int] = collections.defaultdict(int)
        visited_tiles: set[tuple[int, int]] = set()
        rle_runs: list[dict[str, Any]] = []
        triggered_steps = 0
        eval_steps = 0

        for step_idx in range(1, steps + 1):
            eval_steps += 1
            pre_pos = (int(st.y), int(st.x))
            pre_turn = int(st.turn)
            visited_tiles.add(pre_pos)
            cand_act = _tick_node(node, st, bb)
            if cand_act is not None:
                triggered_steps += 1
                chosen_act = cand_act
                act_source = "subtree"
            elif baseline_runner is not None:
                base_act, _ = baseline_runner.next_action(st)
                chosen_act = base_act if base_act is not None else Action.WAIT
                act_source = "baseline" if base_act is not None else "wait_fallback"
            else:
                chosen_act = Action.WAIT
                act_source = "wait_fallback"

            act_name = (
                chosen_act.name if hasattr(chosen_act, "name") else str(chosen_act)
            )
            action_counts[act_name] += 1
            st, done, _ = harness.step(chosen_act)
            post_pos = (int(st.y), int(st.x))
            turn_adv = int(st.turn) - pre_turn

            if (
                rle_runs
                and rle_runs[-1]["source"] == act_source
                and rle_runs[-1]["act"] == act_name
            ):
                seg = rle_runs[-1]
                seg["end_step"] = step_idx
                seg["count"] += 1
                seg["end_pos"] = post_pos
                seg["turn_adv"] += turn_adv
            else:
                rle_runs.append(
                    {
                        "start_step": step_idx,
                        "end_step": step_idx,
                        "count": 1,
                        "source": act_source,
                        "act": act_name,
                        "start_pos": pre_pos,
                        "end_pos": post_pos,
                        "turn_adv": turn_adv,
                    }
                )
            if done:
                break

        rle_trace = [
            f"[S{seg['start_step']}..{seg['end_step']} ({seg['count']}x) {seg['source']}:{seg['act']} "
            f"{seg['start_pos']}->{seg['end_pos']} dt={seg['turn_adv']}]"
            for seg in rle_runs
        ]
        visited_tiles.add((st.y, st.x))
        obs = ObservationDigest(
            survived=not harness.done,
            steps_taken=harness.step_count,
            triggered_steps=triggered_steps,
            total_eval_steps=eval_steps,
            end_reason=str(getattr(st, "end_reason", "")) if harness.done else "active",
            hp_start=start_hp,
            hp_end=int(st.hp),
            depth_start=start_depth,
            depth_end=int(st.depth),
            score_start=start_score,
            score_end=int(st.score),
            unique_tiles=len(visited_tiles),
            actions_taken=dict(action_counts),
            rle_trace=rle_trace,
            dialog_resolved=not getattr(st, "waiting_for_space", False)
            and not getattr(st, "in_text_prompt", False),
            context_label=ctx_label,
        )
        return obs, ""
    except Exception as exc:
        tb = traceback.format_exc()
        return ObservationDigest(), f"{type(exc).__name__}: {exc}\n{tb}"
    finally:
        harness.close()
