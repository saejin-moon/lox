"""
Prefix-Replay Evaluation Shards.
Extracts and manages checkpoint prefixes from recorded baseline runs
at pre-death, pre-stall, and floor-entry windows for fast, low-variance
targeted skill evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from lox.db.schema import connect_db


@dataclass(frozen=True)
class PrefixShard:
    """A checkpoint prefix of recorded action bytes for mid-game replay evaluation."""

    shard_id: str
    run_id: str
    seed: int
    cut_step_idx: int
    action_bytes: bytes
    depth_at_cut: int
    trigger_context: str
    post_replay_max_steps: int = 200


def _find_byte_offset(
    ticks: list[dict[str, Any]] | None,
    pred_step: int | None = None,
    pred_turn: int | None = None,
    pred_depth: int | None = None,
    default_offset: int = 0,
) -> int:
    """Finds exact byte_offset from tick list matching step/turn/depth boundaries."""
    if not ticks:
        return default_offset

    best_offset = None
    if pred_depth is not None:
        # First tick arriving on pred_depth
        for t in ticks:
            if t.get("depth", 1) >= pred_depth:
                if "byte_offset" in t:
                    return int(t["byte_offset"])

    if pred_step is not None:
        # Last tick with step_idx <= pred_step
        for t in ticks:
            if t.get("step_idx", 0) <= pred_step:
                if "byte_offset" in t:
                    best_offset = int(t["byte_offset"])
        if best_offset is not None:
            return best_offset

    if pred_turn is not None:
        # Last tick with turn <= pred_turn
        for t in ticks:
            if t.get("turn", 0) <= pred_turn:
                if "byte_offset" in t:
                    best_offset = int(t["byte_offset"])
        if best_offset is not None:
            return best_offset

    return default_offset


def extract_shards_from_runs(
    runs: list[dict[str, Any]],
    max_shards: int = 60,
    target_functions: list[str] | None = None,
) -> list[PrefixShard]:
    """
    Extracts checkpoint prefixes from baseline runs across three critical categories:
    1. Pre-Death Windows (30 steps before fatal encounter).
    2. Pre-Stall Windows (5 steps before detected oscillation/stagnation).
    3. Floor Entry Checkpoints (arrival step on depth >= 2).
    Uses exact byte offsets from recorded ticks when available.
    """
    shards: list[PrefixShard] = []
    seen_seeds: set[tuple[int, str]] = set()

    for r in runs:
        if len(shards) >= max_shards:
            break

        raw_bytes = r.get("action_bytes")
        if not raw_bytes or len(raw_bytes) < 10:
            continue

        b_bytes = bytes(raw_bytes)
        run_id = str(r.get("run_id", "run"))
        seed = int(r.get("seed", 0))
        steps = int(r.get("steps") or r.get("turns") or len(b_bytes))
        end_reason = str(r.get("end_reason", "ended"))
        summary = str(r.get("summary_digest", ""))
        depth = int(r.get("max_depth", r.get("depth", 1)))
        ticks = r.get("ticks") or []

        # 1. Pre-Death Windows (30 steps before fatal encounter)
        if end_reason not in ("max_steps", "ascended", "ended") and steps >= 15:
            cut_step = max(5, steps - 30 if steps >= 35 else steps - 5)
            ratio = cut_step / float(steps)
            default_cut = max(5, min(len(b_bytes) - 5, int(len(b_bytes) * ratio)))
            cut_idx = _find_byte_offset(
                ticks, pred_step=cut_step, default_offset=default_cut
            )
            cut_idx = max(1, min(len(b_bytes), cut_idx))
            ctx = f"pre_death:{end_reason[:30]}"
            key = (seed, ctx)
            if key not in seen_seeds:
                seen_seeds.add(key)
                shards.append(
                    PrefixShard(
                        shard_id=f"shard_d_{run_id}_{cut_step}",
                        run_id=run_id,
                        seed=seed,
                        cut_step_idx=cut_step,
                        action_bytes=b_bytes[:cut_idx],
                        depth_at_cut=depth,
                        trigger_context=ctx,
                        post_replay_max_steps=200,
                    )
                )

        # 2. Pre-Stall Windows (5 steps prior to stall onset)
        if "Stalls:" in summary and "Stalls: None" not in summary:
            m = re.search(r"\(T(\d+)\.\.\d+", summary)
            if m:
                stall_start_turn = int(m.group(1))
                cut_step = max(5, stall_start_turn - 5)
                ratio = cut_step / float(max(1, steps))
                default_cut = max(5, min(len(b_bytes) - 5, int(len(b_bytes) * ratio)))
                cut_idx = _find_byte_offset(
                    ticks, pred_turn=cut_step, default_offset=default_cut
                )
                cut_idx = max(1, min(len(b_bytes), cut_idx))
                ctx = "pre_stall"
                key = (seed, ctx)
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    shards.append(
                        PrefixShard(
                            shard_id=f"shard_s_{run_id}_{cut_step}",
                            run_id=run_id,
                            seed=seed,
                            cut_step_idx=cut_step,
                            action_bytes=b_bytes[:cut_idx],
                            depth_at_cut=depth,
                            trigger_context=ctx,
                            post_replay_max_steps=200,
                        )
                    )

        # 3. Floor Entry Checkpoints (arrival on Dlvl >= 2)
        milestones = r.get("milestones")
        if not milestones and r.get("milestones_json"):
            try:
                milestones = (
                    json.loads(r["milestones_json"])
                    if isinstance(r["milestones_json"], str)
                    else r["milestones_json"]
                )
            except Exception:
                milestones = {}
        if isinstance(milestones, str):
            try:
                milestones = json.loads(milestones)
            except Exception:
                milestones = {}
        if not isinstance(milestones, dict):
            milestones = {}
        first_turns = milestones.get("depth_first_turn") or {}
        for d_str, d_turn in first_turns.items():
            try:
                d_val = int(d_str)
                d_turn_val = int(d_turn)
            except Exception:
                continue
            if d_val >= 2 and d_turn_val > 0:
                cut_step = d_turn_val
                ratio = cut_step / float(max(1, steps))
                default_cut = max(5, min(len(b_bytes), int(len(b_bytes) * ratio)))
                cut_idx = _find_byte_offset(
                    ticks, pred_depth=d_val, default_offset=default_cut
                )
                cut_idx = max(1, min(len(b_bytes), cut_idx))
                ctx = f"floor_entry:{d_val}"
                key = (seed, ctx)
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    shards.append(
                        PrefixShard(
                            shard_id=f"shard_f_{run_id}_d{d_val}",
                            run_id=run_id,
                            seed=seed,
                            cut_step_idx=cut_step,
                            action_bytes=b_bytes[:cut_idx],
                            depth_at_cut=d_val,
                            trigger_context=ctx,
                            post_replay_max_steps=200,
                        )
                    )

    return shards[:max_shards]


def extract_shards_from_db(
    db_path: str,
    batch_id: str,
    max_shards: int = 60,
    target_functions: list[str] | None = None,
) -> list[PrefixShard]:
    """Extracts prefix shards directly from DuckDB runs for a given batch."""
    con = connect_db(db_path, read_only=True)
    try:
        cur = con.execute(
            """
            SELECT run_id, seed, depth, max_depth, end_reason, summary_digest, milestones_json, action_bytes, turns, steps
            FROM runs
            WHERE batch_id = ? AND action_bytes IS NOT NULL
            LIMIT 150
            """,
            (batch_id,),
        )
        cols = [d[0] for d in cur.description]
        runs: list[dict[str, Any]] = []
        for row in cur.fetchall():
            runs.append(dict(zip(cols, row)))
        return extract_shards_from_runs(
            runs, max_shards=max_shards, target_functions=target_functions
        )
    finally:
        con.close()
