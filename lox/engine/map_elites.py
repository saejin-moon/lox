"""
3D Quality-Diversity MAP-Elites Behavioral Archive.
Maintains diverse behavioral stepping stones across 3 phenotypic dimensions:
1. Depth: Max/mean depth reached (5 bins: D1, D2, D3, D4, D5+)
2. Caution: Search and wait proportion (4 bins: [0, 0.10), [0.10, 0.25), [0.25, 0.50), [0.50, 1.0])
3. Resource Mastery: Efficiency and score density (4 bins: [0, 0.2), [0.2, 0.5), [0.5, 1.0), [1.0, inf))

Total Niches: 5 * 4 * 4 = 80 cells.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Elite:
    cell_id: str
    depth_bin: int
    caution_bin: int
    resource_bin: int
    fitness: float
    policy_code: str
    batch_id: str
    generation: int
    metrics: dict[str, Any]
    updated_at: float = 0.0


class MAPElitesArchive:
    """
    Quality-Diversity MAP-Elites Archive storing the highest-fitness policy
    in each behavioral niche.
    """

    DEPTH_BINS = 5
    CAUTION_BINS = 4
    RESOURCE_BINS = 4
    DEFAULT_DEPTH_THRESHOLDS = (1.5, 2.5, 3.5, 4.5)
    DEFAULT_CAUTION_THRESHOLDS = (0.10, 0.25, 0.50)
    DEFAULT_DENSITY_THRESHOLDS = (0.20, 0.50, 1.00)
    DEFAULT_FITNESS_WEIGHTS = (1000.0, 250.0, 0.1, 0.05)

    def __init__(
        self,
        depth_thresholds: tuple[float, ...] = DEFAULT_DEPTH_THRESHOLDS,
        caution_thresholds: tuple[float, ...] = DEFAULT_CAUTION_THRESHOLDS,
        density_thresholds: tuple[float, ...] = DEFAULT_DENSITY_THRESHOLDS,
        fitness_weights: tuple[float, float, float, float] = DEFAULT_FITNESS_WEIGHTS,
    ) -> None:
        self.cells: dict[str, Elite] = {}
        self.depth_thresholds = tuple(depth_thresholds)
        self.caution_thresholds = tuple(caution_thresholds)
        self.density_thresholds = tuple(density_thresholds)
        self.fitness_weights = tuple(fitness_weights)

    @classmethod
    def compute_cell_id(cls, metrics: dict[str, Any]) -> tuple[str, int, int, int]:
        """
        Maps episode metrics to 3D discrete niche coordinates:
        (depth_bin, caution_bin, resource_bin).
        """
        # 1. Depth Dimension (0 to 4)
        avg_depth = float(metrics.get("avg_depth", metrics.get("mean_depth", 1.0)))
        max_depth = int(metrics.get("max_depth", 1))
        # Use max depth if progressed, otherwise average
        d_val = max(avg_depth, float(max_depth))
        if d_val < 1.5:
            d_bin = 0
        elif d_val < 2.5:
            d_bin = 1
        elif d_val < 3.5:
            d_bin = 2
        elif d_val < 4.5:
            d_bin = 3
        else:
            d_bin = 4

        # 2. Caution Dimension (0 to 3)
        pct_wait = float(metrics.get("pct_wait", 0.0))
        pct_search = float(metrics.get("pct_search", 0.0))
        caution_val = pct_wait + pct_search
        if caution_val < 0.10:
            c_bin = 0
        elif caution_val < 0.25:
            c_bin = 1
        elif caution_val < 0.50:
            c_bin = 2
        else:
            c_bin = 3

        # 3. Resource Mastery Dimension (0 to 3)
        score = float(metrics.get("avg_score", metrics.get("mean_score", 0.0)))
        turns = max(
            10.0, float(metrics.get("avg_turns", metrics.get("mean_turns", 100.0)))
        )
        density = score / turns
        if density < 0.20:
            r_bin = 0
        elif density < 0.50:
            r_bin = 1
        elif density < 1.00:
            r_bin = 2
        else:
            r_bin = 3

        cell_id = f"cell_d{d_bin}_c{c_bin}_r{r_bin}"
        return cell_id, d_bin, c_bin, r_bin

    @staticmethod
    def compute_fitness(metrics: dict[str, Any]) -> float:
        """
        Computes scalar fitness for comparing policies within the same niche.
        Prefers depth progression, survival duration, and score density.
        """
        avg_depth = float(metrics.get("avg_depth", metrics.get("mean_depth", 1.0)))
        max_depth = float(metrics.get("max_depth", 1.0))
        avg_score = float(metrics.get("avg_score", metrics.get("mean_score", 0.0)))
        avg_turns = float(metrics.get("avg_turns", metrics.get("mean_turns", 0.0)))
        # Depth dominates, followed by score and safe turns
        return (
            (avg_depth * 1000.0)
            + (max_depth * 250.0)
            + (avg_score * 0.1)
            + (avg_turns * 0.05)
        )

    def add_candidate(
        self,
        policy_code: str,
        metrics: dict[str, Any],
        batch_id: str,
        generation: int,
        con: Any = None,
    ) -> tuple[bool, str, Elite]:
        """
        Evaluates candidate against its niche cell.
        If cell is empty or candidate has higher fitness, adds/replaces the elite.
        Returns (improved, cell_id, elite).
        """
        cell_id, d_bin, c_bin, r_bin = self.compute_cell_id(metrics)
        fitness = self.compute_fitness(metrics)

        existing = self.cells.get(cell_id)
        if existing is not None and existing.fitness >= fitness:
            return False, cell_id, existing

        elite = Elite(
            cell_id=cell_id,
            depth_bin=d_bin,
            caution_bin=c_bin,
            resource_bin=r_bin,
            fitness=fitness,
            policy_code=policy_code,
            batch_id=batch_id,
            generation=generation,
            metrics=metrics,
            updated_at=time.time(),
        )
        self.cells[cell_id] = elite

        if con is not None:
            self._persist_elite(con, elite)

        return True, cell_id, elite

    def _persist_elite(self, con: Any, elite: Elite) -> None:
        try:
            m_json = json.dumps(elite.metrics)
            con.execute(
                """
                INSERT OR REPLACE INTO archive_policies (
                    cell_id, depth_bin, caution_bin, resource_bin, fitness,
                    policy_code, batch_id, generation, metrics_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    elite.cell_id,
                    elite.depth_bin,
                    elite.caution_bin,
                    elite.resource_bin,
                    elite.fitness,
                    elite.policy_code,
                    elite.batch_id,
                    elite.generation,
                    m_json,
                ],
            )
        except Exception:
            pass

    def sample_elite(self, strategy: str = "curiosity") -> Elite | None:
        """
        Samples an elite policy from the archive.
        - 'best': Highest fitness policy across all niches.
        - 'random': Uniform random selection across occupied niches.
        - 'curiosity': Prioritizes niches with few occupied neighbors (frontier of behavioral space).
        """
        if not self.cells:
            return None

        elites = list(self.cells.values())
        if strategy == "best":
            return max(elites, key=lambda e: e.fitness)
        if strategy == "random" or len(elites) == 1:
            return random.choice(elites)

        # Curiosity: count occupied neighbors in 3D grid
        weights = []
        for e in elites:
            neighbors = 0
            for dy in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if dy == 0 and dc == 0 and dr == 0:
                            continue
                        nid = f"cell_d{e.depth_bin + dy}_c{e.caution_bin + dc}_r{e.resource_bin + dr}"
                        if nid in self.cells:
                            neighbors += 1
            # Fewer neighbors = higher curiosity weight
            w = 1.0 / (1.0 + neighbors)
            weights.append(w)

        return random.choices(elites, weights=weights, k=1)[0]

    def coverage(self) -> float:
        """Returns the fraction of behavioral niches filled (0.0 to 1.0)."""
        total = self.DEPTH_BINS * self.CAUTION_BINS * self.RESOURCE_BINS
        return len(self.cells) / float(total)

    def load_from_db(self, con: Any) -> int:
        """Loads all elites from the archive_policies table in DuckDB."""
        try:
            rows = con.execute(
                """
                SELECT cell_id, depth_bin, caution_bin, resource_bin, fitness,
                       policy_code, batch_id, generation, metrics_json
                FROM archive_policies
                """
            ).fetchall()
            loaded = 0
            for r in rows:
                cid, db, cb, rb, fit, pcode, bid, gen, mjson = r
                m_dict = json.loads(mjson) if mjson else {}
                self.cells[cid] = Elite(
                    cell_id=cid,
                    depth_bin=int(db),
                    caution_bin=int(cb),
                    resource_bin=int(rb),
                    fitness=float(fit),
                    policy_code=str(pcode),
                    batch_id=str(bid),
                    generation=int(gen),
                    metrics=m_dict,
                )
                loaded += 1
            return loaded
        except Exception:
            return 0
