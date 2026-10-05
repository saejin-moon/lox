"""
LOX Empirical Knowledge Base.
Structured, machine-readable repository of NetHack tactical invariants, architectural rules,
and empirical lessons synthesized across 48+ generations of empirical policy evolution.
"""

from lox.knowledge.invariants import REGISTRY, Invariant, InvariantRegistry

__all__ = ["Invariant", "InvariantRegistry", "REGISTRY"]
