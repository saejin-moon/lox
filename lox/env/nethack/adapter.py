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
        **kwargs: Any,
    ) -> EnvHarness:
        return NetHackHarness(role=role, max_steps=max_steps)

    def progression_metric(self, run: dict[str, Any]) -> float:
        from lox.engine.hypothesis import progression_metric

        return progression_metric(run)

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
