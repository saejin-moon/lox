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
        self._wiki_db_path = wiki_db_path
        self._wiki_engine: WikiEngine | None = None

    @property
    def wiki_engine(self) -> WikiEngine:
        if self._wiki_engine is None:
            self._wiki_engine = WikiEngine(db_path=self._wiki_db_path)
        return self._wiki_engine

    @property
    def name(self) -> str:
        return "nethack"

    def create_harness(
        self,
        role: str = "val-hum-law-fem",
        max_steps: int = 25000,
        lean_obs: bool = True,
        allow_menu_linger: bool = False,
        **kwargs: Any,
    ) -> EnvHarness:
        return NetHackHarness(
            role=role,
            max_steps=max_steps,
            lean_obs=lean_obs,
            allow_menu_linger=allow_menu_linger,
        )

    def progression_metric(self, run: dict[str, Any]) -> float:
        """
        Single-source-of-truth anti-Goodhart progression metric for NetHack.
        - Anchored on max_depth (d), XP level (xl), log1p(score), and dungeon branches.
        - Drops noisy oprop_bonus (r=0.036) and ac_bonus (r=0.087); scopes stairs/tiles
          bootstrap tiebreak strictly to Dlvl 1 so it guides Gen 0/1 without saturating on Dlvl 2+.
        - Anti-Goodhart cap: any non-shard max_steps or stalled* run at ANY depth zeroes out
          farmable secondary bonuses (xl, sc) and subtracts 0.35 so 10k-turn room-camping
          can NEVER outscore a genuine run of the same depth.
        """
        import math

        d = float(run.get("max_depth") or run.get("depth") or 1)
        raw_xl = run.get("xp_level")
        if raw_xl is None:
            raw_xl = run.get("final_xp_level")
        xl = (
            max(0.0, min(float(raw_xl if raw_xl is not None else 1), 30.0) - 1.0) * 0.18
        )
        branch_bonus = max(0, int(run.get("dungeons_visited") or 1) - 1) * 0.15
        sc = math.log1p(max(0.0, float(run.get("score") or 0))) * 0.12
        ms_dict = (
            run.get("milestones") if isinstance(run.get("milestones"), dict) else {}
        )

        # Dlvl-1 bootstrap exploration tiebreak only (avoids 98% saturation on Dlvl 2+)
        d1_bootstrap = 0.0
        if d <= 1.0:
            has_stairs = bool(
                run.get("stairs_down_found")
                or ms_dict.get("stairs_down_found")
                or "'>'" in str(run.get("summary_digest") or "")
            )
            raw_tiles = run.get("unique_tiles")
            tiles = float(
                raw_tiles
                if raw_tiles is not None
                else (ms_dict.get("unique_tiles_d1", 0) or 0)
            )
            d1_bootstrap = (0.10 if has_stairs else 0.0) + 0.10 * min(1.0, tiles / 60.0)

        shard_survival_bonus = (
            0.08
            if (
                run.get("is_shard")
                and int(run.get("final_hp") or 0) > 0
                and str(run.get("end_reason") or "") in ("max_steps", "ascended")
            )
            else 0.0
        )

        # Anti-Goodhart horizon-waste cap: zero out farmable xl/sc and penalize stalling at ALL depths
        if self.is_horizon_waste(run):
            return max(0.10, d + branch_bonus + d1_bootstrap - 0.35)

        return d + xl + branch_bonus + sc + d1_bootstrap + shard_survival_bonus

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
