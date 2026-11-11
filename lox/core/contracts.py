"""
Strict Typed Domain Contracts & Invariant Verifiers for LOX.
Eliminates silent dict-key mismatches, shard-vs-full-episode leakage,
and fragmented policy/skill state across the synthesis pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Canonical alias map: accessing or setting any alias automatically resolves to both
_KEY_ALIASES: dict[str, str] = {
    "final_xp_level": "xp_level",
    "xp_level": "final_xp_level",
    "final_max_hp": "max_hp",
    "max_hp": "final_max_hp",
    "functions_triggered": "triggered_functions",
    "triggered_functions": "functions_triggered",
    "function_turns": "fn_turns",
    "fn_turns": "function_turns",
}

KNOWN_EPISODE_KEYS: frozenset[str] = frozenset(
    {
        "triggered",
        "is_shard",
        "shard_id",
        "run_id",
        "batch_id",
        "generation",
        "seed",
        "role",
        "depth",
        "max_depth",
        "score",
        "achievements",
        "turns",
        "steps",
        "xp_level",
        "final_xp_level",
        "final_hp",
        "max_hp",
        "final_max_hp",
        "final_ac",
        "gold",
        "end_reason",
        "death_message",
        "is_outlier_good",
        "outlier_z_score",
        "batch_percentile",
        "summary_digest",
        "game_log",
        "milestones",
        "milestones_json",
        "auto_esc_count",
        "subgen_preemptions",
        "policy_exceptions",
        "unique_tiles",
        "pct_wait",
        "pct_search",
        "starved_with_food",
        "died_unarmored",
        "stairs_down_found",
        "triggered_functions",
        "functions_triggered",
        "fn_turns",
        "function_turns",
        "last_active_function",
        "action_bytes",
        "step_byte_offsets",
        "wall_ms",
        "ticks",
        "registered_params",
        # Summarizer / diagnostic enrichment keys
        "stagnation_turns",
        "stagnation_ratio",
        "livelock_detected",
        "recurrence_rate",
    }
)


class EpisodeRun(dict):
    """
    Schema-normalized episode record.
    Subclasses `dict` for zero-copy compatibility with multiprocessing `pickle` and `pyarrow`,
    while synchronizing key aliases (`xp_level` <-> `final_xp_level`, `max_hp` <-> `final_max_hp`,
    `triggered_functions` <-> `functions_triggered`) and exposing typed properties.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__()
        if data:
            self.update(data)
        if kwargs:
            self.update(kwargs)
        self._sync_aliases_and_defaults()

    def _sync_aliases_and_defaults(self) -> None:
        # Synchronize paired aliases so neither key is ever missing
        for k_a, k_b in (
            ("xp_level", "final_xp_level"),
            ("max_hp", "final_max_hp"),
            ("triggered_functions", "functions_triggered"),
            ("fn_turns", "function_turns"),
        ):
            v_a = super().get(k_a)
            v_b = super().get(k_b)
            if v_a is not None and v_b is None:
                super().__setitem__(k_b, v_a)
            elif v_b is not None and v_a is None:
                super().__setitem__(k_a, v_b)

        # Parse JSON strings for triggered_functions and milestones if loaded from DB
        trig = super().get("triggered_functions")
        if isinstance(trig, str):
            try:
                parsed_trig = json.loads(trig) if trig.strip() else {}
            except Exception:
                parsed_trig = {}
            super().__setitem__("triggered_functions", parsed_trig)
            super().__setitem__("functions_triggered", parsed_trig)

        ms = super().get("milestones")
        ms_json = super().get("milestones_json")
        if ms is None and isinstance(ms_json, str) and ms_json.strip():
            try:
                super().__setitem__("milestones", json.loads(ms_json))
            except Exception:
                super().__setitem__("milestones", {})

        # Ensure `is_shard` is accurately set if run_id starts with `shard_` or shard_id is non-empty
        run_id_str = str(super().get("run_id", "") or "")
        shard_id_str = str(super().get("shard_id", "") or "")
        if run_id_str.startswith("shard_") or shard_id_str:
            super().__setitem__("is_shard", True)
        elif "is_shard" not in self:
            super().__setitem__("is_shard", False)

        # Ensure `stairs_down_found` is consistent with `max_depth > 1` or `summary_digest`
        if "stairs_down_found" not in self or not super().get("stairs_down_found"):
            max_d = int(super().get("max_depth", super().get("depth", 1)) or 1)
            digest = str(super().get("summary_digest", "") or "")
            ms_dict = super().get("milestones")
            has_st = bool(
                max_d > 1
                or "'>'" in digest
                or (isinstance(ms_dict, dict) and ms_dict.get("stairs_down_found"))
            )
            if has_st or "stairs_down_found" not in self:
                super().__setitem__("stairs_down_found", has_st)

        # Coalesce NULL / None database values to typed defaults
        for k, default_val in (
            ("depth", 1),
            ("max_depth", 1),
            ("score", 0),
            ("turns", 0),
            ("steps", 0),
            ("xp_level", 1),
            ("final_xp_level", 1),
            ("final_hp", 0),
            ("max_hp", 0),
            ("final_max_hp", 0),
            ("final_ac", 10),
            ("gold", 0),
            ("outlier_z_score", 0.0),
            ("batch_percentile", 0.0),
            ("unique_tiles", 0),
            ("pct_wait", 0.0),
            ("pct_search", 0.0),
            ("auto_esc_count", 0),
            ("subgen_preemptions", 0),
            ("policy_exceptions", 0),
            ("starved_with_food", False),
            ("died_unarmored", False),
            ("is_outlier_good", False),
        ):
            if super().get(k) is None:
                super().__setitem__(k, default_val)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        alias = _KEY_ALIASES.get(key)
        if alias is not None:
            super().__setitem__(alias, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return super().get(key, default)
        alias = _KEY_ALIASES.get(key)
        if alias is not None and alias in self:
            return super().get(alias, default)
        return default

    @property
    def is_shard_run(self) -> bool:
        return bool(
            super().get("is_shard")
            or str(super().get("run_id", "")).startswith("shard_")
            or str(super().get("shard_id", ""))
        )


def normalize_episode_run(run: dict[str, Any] | EpisodeRun) -> EpisodeRun:
    """Converts a raw episode dict into a normalized `EpisodeRun` with synced aliases."""
    if isinstance(run, EpisodeRun):
        run._sync_aliases_and_defaults()
        return run
    return EpisodeRun(run)


def split_full_and_shard_runs(
    runs: list[dict[str, Any]] | list[EpisodeRun],
) -> tuple[list[EpisodeRun], list[EpisodeRun]]:
    """
    Partitions a batch of runs into `(full_runs, shard_runs)`, normalizing each item
    into an `EpisodeRun` so shards and full episodes are never accidentally conflated.
    """
    full_runs: list[EpisodeRun] = []
    shard_runs: list[EpisodeRun] = []
    for r in runs:
        er = normalize_episode_run(r)
        if er.is_shard_run:
            shard_runs.append(er)
        else:
            full_runs.append(er)
    return full_runs, shard_runs


class BatchEvaluation(dict):
    """
    Typed batch evaluation result returned by `PersistentEvalPool.evaluate_batch`.
    Explicitly separates `full_runs` from `shard_runs` while preserving dict access.
    """

    @property
    def full_runs(self) -> list[EpisodeRun]:
        if "full_runs" in self:
            return self["full_runs"]
        full, _ = split_full_and_shard_runs(self.get("runs", []))
        return full

    @property
    def shard_runs(self) -> list[EpisodeRun]:
        if "shard_runs" in self:
            return self["shard_runs"]
        _, shards = split_full_and_shard_runs(self.get("runs", []))
        return shards


@dataclass
class CandidatePackage:
    """
    Atomic unit representing a candidate policy + skill modification + pilot score.
    Replaces loose parallel attributes (`last_passed_code`, `last_passed_skill_name`,
    `last_passed_skill_code`, `best_pilot_code`, `best_pilot_skill_name`, `best_pilot_skill_code`)
    so `policy_code` and `skill_code` can NEVER become desynchronized.
    """

    code: str = ""
    skill_name: str = ""
    skill_code: str = ""
    pilot_delta: float = -1e9
    pilot_summary: str = ""
    improved: bool = False
    skills_snapshot: dict[str, str] = field(default_factory=dict)

    @property
    def is_populated(self) -> bool:
        return bool(self.code.strip() or self.skill_code.strip())


def assert_baseline_invariants(
    baseline_runs: list[dict[str, Any]],
) -> list[EpisodeRun]:
    """
    Verifies and normalizes `baseline_runs`:
    1. Strips out any accidental `is_shard` / `shard_*` runs so baseline is 100% full episodes.
    2. Deduplicates by `seed` (keeping the latest entry per seed).
    3. Normalizes every run into an `EpisodeRun` so `xp_level`/`final_xp_level` and `stairs_down_found` are synced.
    """
    by_seed: dict[int, EpisodeRun] = {}
    for r in baseline_runs:
        er = normalize_episode_run(r)
        if er.is_shard_run:
            continue
        if "seed" in er:
            by_seed[int(er["seed"])] = er
    return list(by_seed.values())
