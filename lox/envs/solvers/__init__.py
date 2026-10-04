"""
LOX Algorithmic Sub-Solvers Package:
Deterministic, stateful macro solvers for complex NetHack interactions
(Altar BUC identification, poison resistance harvesting, Sokoban).
"""

from __future__ import annotations

from lox.envs.solvers.altar_solver import AltarBUCSolver
from lox.envs.solvers.poison_solver import PoisonResHarvestSolver

__all__ = ["AltarBUCSolver", "PoisonResHarvestSolver"]
