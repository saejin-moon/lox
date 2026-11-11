"""
NetHack 3.6.6 Benchmark Adapter ("Hard Code Nothing").
Implements the BenchmarkAdapter protocol for NetHack/NLE.
"""

from __future__ import annotations

from typing import Any

from lox.datasets.wiki import WikiEngine
from lox.env.base import BenchmarkAdapter, EnvHarness
from lox.env.harness import NetHackHarness


class NetHackAdapter(BenchmarkAdapter):
    """Benchmark adapter for NetHack 3.6.6 via NLE."""

    def __init__(self, wiki_db_path: str = "data/wiki_index.db"):
        self.wiki_engine = WikiEngine(db_path=wiki_db_path)

    @property
    def name(self) -> str:
        return "nethack"

    def create_harness(
        self,
        role: str = "val-hum-law-fem",
        max_steps: int = 25000,
        lean_obs: bool = True,
        **kwargs: Any,
    ) -> EnvHarness:
        return NetHackHarness(role=role, max_steps=max_steps, lean_obs=lean_obs)

    def progression_metric(self, run: dict[str, Any]) -> float:
        import math

        d = float(run.get("max_depth", run.get("depth", 1)))
        raw_xl = run.get("xp_level")
        if raw_xl is None:
            raw_xl = run.get("final_xp_level", 1)
        xl = max(0.0, min(float(raw_xl or 1), 30.0) - 1.0) * 0.12
        ac_val = float(run.get("final_ac", 10))
        ac_bonus = max(0.0, 10.0 - ac_val) * 0.04
        sc = math.log1p(max(0.0, float(run.get("score", 0)))) * 0.06

        has_stairs = bool(
            run.get("stairs_down_found")
            or (
                isinstance(run.get("milestones"), dict)
                and run["milestones"].get("stairs_down_found")
            )
            or d > 1.0
            or "'>'" in str(run.get("summary_digest", ""))
        )
        stairs_bonus = 0.20 if has_stairs else 0.0
        tiles = float(
            run.get(
                "unique_tiles",
                run.get("milestones", {}).get("unique_tiles_d1", 0),
            )
        )
        tiles_bonus = 0.10 * min(1.0, tiles / 60.0)

        shard_survival_bonus = (
            0.08
            if (
                run.get("is_shard")
                and int(run.get("final_hp", 0)) > 0
                and str(run.get("end_reason", "")) in ("max_steps", "ascended")
            )
            else 0.0
        )
        return (
            d + xl + ac_bonus + sc + stairs_bonus + tiles_bonus + shard_survival_bonus
        )

    def extract_milestones_and_stalls(
        self, ticks: list[dict[str, Any]]
    ) -> tuple[str, str, dict[str, Any]]:
        from lox.db.summarizer import summarize_episode_ticks

        return summarize_episode_ticks(ticks)

    def get_api_docs(self) -> str:
        from lox.author.prompts import POLICY_LANG_DOCS

        return POLICY_LANG_DOCS

    def search_knowledge_base(self, query: str, top_k: int = 3) -> str:
        return self.wiki_engine.query(query, top_k=top_k)
