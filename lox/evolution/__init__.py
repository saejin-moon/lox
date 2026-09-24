"""LOX-ψ Evolution package (S2 scaffolding — population/selection land in S2 proper).

For now this exposes the run↔candidate lineage registry used for per-candidate
evidence tagging.
"""
from lox.evolution.lineage import RunLineage, DEFAULT_DB_PATH  # noqa: F401

__all__ = ["RunLineage", "DEFAULT_DB_PATH"]