"""
3D Quality-Diversity MAP-Elites Behavioral Archive (v2 Vitals & Spatial Tempo Grid).
Maintains diverse behavioral stepping stones across 3 orthogonal phenotypic dimensions:
1. Depth (`avg_depth` non-aliasing): 5 bins via thresholds `(1.5, 2.5, 4.0, 6.5)`
2. Vitals Progression (`vital_bin`): 4 bins via `max(0.0, 10.0 - avg_ac) + 1.5*max(0.0, avg_xp_level - 1.0) + 2.0*max(0.0, avg_branches - 1.0)` at `(3.0, 7.0, 12.0)`
3. Spatial Tempo (`tempo_bin`): 4 bins via `avg_unique_tiles / max(10.0, avg_turns)` at `(0.3, 0.7, 1.2)`

Total Niches: 5 * 4 * 4 = 80 cells (`cell_v2_d{d}_v{v}_t{t}`).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
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
    skills: dict[str, str] = field(default_factory=dict)
    selections: int = 0
    promotions: int = 0


class MAPElitesArchive:
    """
    Quality-Diversity MAP-Elites Archive storing the highest-fitness policy
    in each non-aliasing behavioral niche, with UCB1 + neighbor-curiosity sampling.
    """

    DEPTH_BINS = 5
    CAUTION_BINS = 4
    RESOURCE_BINS = 4
    DEFAULT_DEPTH_THRESHOLDS = (1.25, 1.75, 2.5, 4.0)
    DEFAULT_CAUTION_THRESHOLDS = (3.0, 7.0, 12.0)  # Vitals progression axis
    DEFAULT_DENSITY_THRESHOLDS = (0.08, 0.16, 0.30)  # Spatial tempo axis
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

    @staticmethod
    def _bin_value(val: float, thresholds: tuple[float, ...]) -> int:
        for idx, cut in enumerate(thresholds):
            if val < cut:
                return idx
        return len(thresholds)

    @classmethod
    def compute_cell_id(
        cls,
        metrics: dict[str, Any],
        *,
        depth_thresholds: tuple[float, ...] | None = None,
        caution_thresholds: tuple[float, ...] | None = None,
        density_thresholds: tuple[float, ...] | None = None,
    ) -> tuple[str, int, int, int]:
        """
        Maps episode metrics to non-aliasing 3D niche coordinates (`cell_v2_d{d}_v{v}_t{t}`):
        (depth_bin, vital_bin, tempo_bin).
        """
        # 1. Avg Depth Dimension (0 to 4, non-aliasing: avg_depth rather than single outlier max_depth)
        avg_depth = float(metrics.get("avg_depth", metrics.get("mean_depth", 1.0)))
        d_bin = cls._bin_value(
            avg_depth, depth_thresholds or cls.DEFAULT_DEPTH_THRESHOLDS
        )

        # 2. Vitals Progression Dimension (0 to 3): rewards armor upgrades (low AC), XL, branches
        avg_ac = float(metrics.get("avg_ac", 10.0) or 10.0)
        avg_xl = float(metrics.get("avg_xp_level", metrics.get("avg_xl", 1.0)) or 1.0)
        avg_branches = float(metrics.get("avg_branches", 1.0) or 1.0)
        vitals_val = (
            max(0.0, 10.0 - avg_ac)
            + 1.5 * max(0.0, avg_xl - 1.0)
            + 2.0 * max(0.0, avg_branches - 1.0)
        )
        v_bin = cls._bin_value(
            vitals_val, caution_thresholds or cls.DEFAULT_CAUTION_THRESHOLDS
        )

        # 3. Spatial Tempo Dimension (0 to 3): unique_tiles explored per turn
        avg_turns = max(
            10.0,
            float(metrics.get("avg_turns", metrics.get("mean_turns", 100.0)) or 100.0),
        )
        avg_tiles = float(metrics.get("avg_unique_tiles", 0.0) or 0.0)
        if (
            avg_tiles <= 0.0
            and float(metrics.get("avg_score", metrics.get("mean_score", 0.0)) or 0.0)
            > 0.0
        ):
            # Fallback if spatial telemetry omitted in minimal synthetic tests
            tempo_val = (
                float(metrics.get("avg_score", metrics.get("mean_score", 0.0)) or 0.0)
                / avg_turns
            )
        else:
            tempo_val = avg_tiles / avg_turns
        t_bin = cls._bin_value(
            tempo_val, density_thresholds or cls.DEFAULT_DENSITY_THRESHOLDS
        )

        cell_id = f"cell_v2_d{d_bin}_v{v_bin}_t{t_bin}"
        return cell_id, d_bin, v_bin, t_bin

    @staticmethod
    def compute_fitness(
        metrics: dict[str, Any],
        *,
        fitness_weights: tuple[float, float, float, float] | None = None,
    ) -> float:
        """
        Computes scalar fitness for comparing policies within the same niche.
        Incorporates avg_depth, vitals (AC, XL, branches), max_depth, and score.
        """
        w_depth, w_max, w_score, w_turns = (
            fitness_weights
            if fitness_weights is not None
            else (1000.0, 150.0, 0.1, 0.05)
        )
        avg_depth = float(metrics.get("avg_depth", metrics.get("mean_depth", 1.0)))
        max_depth = float(metrics.get("max_depth", 1.0))
        avg_score = float(metrics.get("avg_score", metrics.get("mean_score", 0.0)))
        avg_turns = float(metrics.get("avg_turns", metrics.get("mean_turns", 0.0)))
        avg_ac = float(metrics.get("avg_ac", 10.0) or 10.0)
        avg_xl = float(metrics.get("avg_xp_level", metrics.get("avg_xl", 1.0)) or 1.0)
        avg_branches = float(metrics.get("avg_branches", 1.0) or 1.0)
        vitals_bonus = (
            max(0.0, 10.0 - avg_ac) * 120.0
            + max(0.0, avg_xl - 1.0) * 150.0
            + max(0.0, avg_branches - 1.0) * 200.0
        )
        return (
            (avg_depth * w_depth)
            + (max_depth * w_max)
            + vitals_bonus
            + (avg_score * w_score)
            + (avg_turns * w_turns)
        )

    def add_candidate(
        self,
        policy_code: str,
        metrics: dict[str, Any],
        batch_id: str,
        generation: int,
        con: Any = None,
        skills: dict[str, str] | None = None,
    ) -> tuple[bool, str, Elite]:
        """
        Evaluates candidate against its niche cell.
        If cell is empty or candidate has higher fitness, adds/replaces the elite.
        Returns (improved, cell_id, elite).
        """
        cell_id, d_bin, c_bin, r_bin = self.compute_cell_id(
            metrics,
            depth_thresholds=self.depth_thresholds,
            caution_thresholds=self.caution_thresholds,
            density_thresholds=self.density_thresholds,
        )
        fitness = self.compute_fitness(metrics, fitness_weights=self.fitness_weights)

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
            skills=dict(skills or {}),
            selections=existing.selections if existing is not None else 0,
            promotions=existing.promotions if existing is not None else 0,
        )
        self.cells[cell_id] = elite

        if con is not None:
            self._persist_elite(con, elite)

        return True, cell_id, elite

    def record_outcome(self, cell_id: str, promoted: bool, con: Any = None) -> None:
        """Records whether branching from `cell_id` yielded a tournament promotion for UCB1."""
        e = self.cells.get(cell_id)
        if e is None:
            return
        e.selections += 1
        if promoted:
            e.promotions += 1
        if con is not None:
            self._persist_elite(con, e)

    def _persist_elite(self, con: Any, elite: Elite) -> None:
        try:
            m_json = json.dumps(elite.metrics)
            s_json = json.dumps(elite.skills or {})
            con.execute(
                """
                INSERT OR REPLACE INTO archive_policies (
                    cell_id, depth_bin, caution_bin, resource_bin, fitness,
                    policy_code, skills_json, batch_id, generation, metrics_json,
                    selections, promotions, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    elite.cell_id,
                    elite.depth_bin,
                    elite.caution_bin,
                    elite.resource_bin,
                    elite.fitness,
                    elite.policy_code,
                    s_json,
                    elite.batch_id,
                    elite.generation,
                    m_json,
                    int(elite.selections),
                    int(elite.promotions),
                ],
            )
        except Exception:
            pass

    def sample_elite(self, strategy: str = "ucb_curiosity") -> Elite | None:
        """
        Samples an elite policy from the archive.
        - 'best': Highest fitness policy across all niches.
        - 'random': Uniform random selection across occupied niches.
        - 'curiosity': Prioritizes niches with few occupied neighbors.
        - 'ucb_curiosity': UCB1 selection combining promotion yield + exploration + neighbor scarcity.
        """
        if not self.cells:
            return None

        elites = list(self.cells.values())
        if strategy == "best":
            return max(elites, key=lambda e: e.fitness)
        if strategy == "caution":
            return max(elites, key=lambda e: (int(e.caution_bin), float(e.fitness)))
        if strategy == "random" or len(elites) == 1:
            return random.choice(elites)

        total_sel = sum(max(0, int(e.selections)) for e in elites) + len(elites)
        if strategy == "ucb_curiosity":
            scored: list[tuple[float, Elite]] = []
            max_fit = max(max(1.0, float(e.fitness)) for e in elites)
            for e in elites:
                sel = max(0, int(e.selections))
                prom = max(0, int(e.promotions))
                mean_r = prom / float(sel + 1) + 0.15 * (float(e.fitness) / max_fit)
                ucb_bonus = math.sqrt(
                    2.0 * math.log(max(2, total_sel)) / float(sel + 1)
                )
                neighbors = sum(
                    1
                    for dy in (-1, 0, 1)
                    for dv in (-1, 0, 1)
                    for dt in (-1, 0, 1)
                    if not (dy == 0 and dv == 0 and dt == 0)
                    and f"cell_v2_d{e.depth_bin + dy}_v{e.caution_bin + dv}_t{e.resource_bin + dt}"
                    in self.cells
                )
                scarcity = 0.25 / (1.0 + neighbors)
                scored.append((mean_r + ucb_bonus + scarcity, e))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            top_elites = [pair[1] for pair in scored[: max(1, min(3, len(scored)))]]
            weights = [max(0.05, s[0]) for s in scored[: len(top_elites)]]
            return random.choices(top_elites, weights=weights, k=1)[0]

        # Curiosity fallback
        weights_c = []
        for e in elites:
            neighbors = sum(
                1
                for dy in (-1, 0, 1)
                for dv in (-1, 0, 1)
                for dt in (-1, 0, 1)
                if not (dy == 0 and dv == 0 and dt == 0)
                and f"cell_v2_d{e.depth_bin + dy}_v{e.caution_bin + dv}_t{e.resource_bin + dt}"
                in self.cells
            )
            weights_c.append(1.0 / (1.0 + neighbors))
        return random.choices(elites, weights=weights_c, k=1)[0]

    def coverage(self) -> float:
        """Returns the fraction of behavioral niches filled (0.0 to 1.0)."""
        total = self.DEPTH_BINS * self.CAUTION_BINS * self.RESOURCE_BINS
        return len(self.cells) / float(total)

    def load_from_db(self, con: Any) -> int:
        """Loads v2 elites from `archive_policies`, purging legacy v1 `cell_d*` cells cleanly."""
        try:
            try:
                con.execute(
                    "DELETE FROM archive_policies WHERE cell_id NOT LIKE 'cell_v2_%'"
                )
            except Exception:
                pass
            rows = con.execute(
                """
                SELECT cell_id, depth_bin, caution_bin, resource_bin, fitness,
                       policy_code, COALESCE(skills_json, '{}'), batch_id, generation, metrics_json,
                       COALESCE(selections, 0), COALESCE(promotions, 0)
                FROM archive_policies
                WHERE cell_id LIKE 'cell_v2_%'
                """
            ).fetchall()
            loaded = 0
            for r in rows:
                cid, db, cb, rb, fit, pcode, sjson, bid, gen, mjson, sels, proms = r
                m_dict = json.loads(mjson) if mjson else {}
                s_dict = json.loads(sjson) if sjson else {}
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
                    skills=dict(s_dict) if isinstance(s_dict, dict) else {},
                    selections=int(sels or 0),
                    promotions=int(proms or 0),
                )
                loaded += 1
            return loaded
        except Exception:
            return 0
