"""
`AuthorToolRegistry`: Composes DB/Wiki tools, workspace code inspection tools,
and live NetHack `dry_run_policy` / Paired CRN Micro-Pilot verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lox.author.tools.code_tools import CodeToolsMixin
from lox.author.tools.db_tools import DBToolsMixin
from lox.author.tools.dry_run import execute_dry_run_policy, has_policy_class
from lox.author.tools.micro_pilot import (
    first_divergence_byte,
    run_peer_paired_micro_pilot,
)
from lox.core.contracts import CandidatePackage
from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine
from lox.db.schema import DEFAULT_DB_PATH
from lox.env.harness import NetHackHarness
from lox.lang.validator import normalize_mode


class AuthorToolRegistry(DBToolsMixin, CodeToolsMixin):
    """Executes tool calls from the Option-A Swarm Peer Author Agent."""

    _has_policy_class = staticmethod(has_policy_class)
    _first_divergence_byte = staticmethod(first_divergence_byte)

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        wiki_db_path: str = DEFAULT_WIKI_DB,
        workspace_dir: str | Path = "data/workspace",
        max_steps: int = 5000,
    ) -> None:
        self.db_path = db_path
        self.wiki = WikiEngine(db_path=wiki_db_path)
        self.workspace_dir = Path(workspace_dir)
        self.max_steps = max_steps
        self.current_policy_code: str = ""
        self.last_passed_pkg: CandidatePackage = CandidatePackage()
        self.pool: Any = None
        self.baseline_runs: list[dict[str, Any]] = []
        self.last_dry_run_mode: str = "none"
        self.last_dry_run_passed: bool = False
        self.skills: dict[str, str] = {}
        self.last_passed_skills: dict[str, str] = {}
        self.last_dry_run_diag: str = ""
        self.theories_store: Any = None
        self.generation: int = 0
        self.mode: str = "SWARM_PEER"
        self.swarm_checkpoint: Any = None
        self.swarm_frontier: Any = None
        self.swarm_scaffold_code: str = ""
        self._reusable_harness: NetHackHarness | None = None
        self.wiki_queries_run: int = 0
        self.theories_queries_run: int = 0
        self.db_queries_run: int = 0
        self.enforce_peer_research_gate: bool = False
        self.claimed_deficit: Any = None
        self.scenario_gate: str = "off"
        self.peer_guard_seeds: list[int] = []
        self.peer_target_seeds: list[int] = []
        self.peer_probe_seeds: list[int] = []
        self.peer_micro_pilots_used: int = 0
        self.max_peer_micro_pilots: int = 4
        self.peer_id: str = ""
        self.baseline_policy_code: str = ""
        self.baseline_skills: dict[str, str] | None = None
        self._dead_patch_bonus_granted: bool = False
        self.best_peer_micro_pilot: dict[str, Any] | None = None
        self.last_peer_micro_pilot: dict[str, Any] | None = None

    def _get_harness(self, max_steps: int) -> NetHackHarness:
        steps = max(10, int(max_steps))
        if self._reusable_harness is None:
            self._reusable_harness = NetHackHarness(max_steps=steps)
        else:
            self._reusable_harness.max_steps = steps
        return self._reusable_harness

    def close(self) -> None:
        """Closes any cached C NetHack harness owned by this registry."""
        if self._reusable_harness is not None:
            try:
                self._reusable_harness.close()
            except Exception:
                pass
            self._reusable_harness = None

    @property
    def last_passed_code(self) -> str:
        return self.last_passed_pkg.code

    @last_passed_code.setter
    def last_passed_code(self, val: str) -> None:
        self.last_passed_pkg.code = str(val or "")

    @property
    def last_passed_skill_name(self) -> str | None:
        return self.last_passed_pkg.skill_name or None

    @last_passed_skill_name.setter
    def last_passed_skill_name(self, val: str | None) -> None:
        self.last_passed_pkg.skill_name = str(val or "")

    @property
    def last_passed_skill_code(self) -> str | None:
        return self.last_passed_pkg.skill_code or None

    @last_passed_skill_code.setter
    def last_passed_skill_code(self, val: str | None) -> None:
        self.last_passed_pkg.skill_code = str(val or "")

    def configure_generation(
        self,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        skills: dict[str, str] | None = None,
        generation: int = 0,
        mode: str = "SWARM_PEER",
        swarm_checkpoint: Any = None,
        swarm_frontier: Any = None,
        swarm_scaffold_code: str = "",
        claimed_deficit: Any = None,
        scenario_gate: str = "off",
        peer_guard_seeds: list[int] | None = None,
        peer_target_seeds: list[int] | None = None,
        peer_probe_seeds: list[int] | None = None,
        max_peer_micro_pilots: int = 4,
        peer_id: str = "",
    ) -> None:
        """Resets per-generation tool state and configures paired CRN micro-pilot context."""
        self.pool = pool
        self.baseline_runs = list(baseline_runs or [])
        self.last_dry_run_mode = "none"
        self.last_dry_run_passed = False
        self.last_dry_run_diag = ""
        self.last_passed_pkg = CandidatePackage()
        self.last_passed_skills = {}
        self.generation = int(generation)
        self.mode = normalize_mode(mode)
        self.swarm_checkpoint = swarm_checkpoint
        self.swarm_frontier = swarm_frontier
        self.swarm_scaffold_code = str(swarm_scaffold_code or "")
        self.claimed_deficit = claimed_deficit
        self.scenario_gate = str(scenario_gate or "off").strip().lower()
        self.wiki_queries_run = 0
        self.theories_queries_run = 0
        self.db_queries_run = 0
        self.enforce_peer_research_gate = False
        self.peer_guard_seeds = [int(s) for s in (peer_guard_seeds or [])]
        self.peer_target_seeds = [int(s) for s in (peer_target_seeds or [])]
        self.peer_probe_seeds = [int(s) for s in (peer_probe_seeds or [])]
        self.peer_micro_pilots_used = 0
        self.max_peer_micro_pilots = max(0, int(max_peer_micro_pilots))
        self.peer_id = str(peer_id or self.peer_id or "")
        self._dead_patch_bonus_granted = False
        self.best_peer_micro_pilot = None
        self.last_peer_micro_pilot = None
        if skills is not None:
            self.skills = dict(skills)

    def _run_peer_paired_micro_pilot(
        self,
        policy_code: str,
        cand_skills: dict[str, str],
        macros_code: str = "",
    ) -> str:
        _ = macros_code
        return run_peer_paired_micro_pilot(
            registry=self,
            policy_code=policy_code,
            cand_skills=cand_skills,
        )

    def dry_run_policy(
        self,
        code: str = "",
        seed: int = 42,
        max_turns: int = 150,
        replay_run_id: str = "",
        replay_steps: int = 0,
        smoke_only: bool = False,
        skill_name: str | None = None,
        skill_code: str | None = None,
        skills: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Compiles and tests candidate `code` in a live NetHack sandbox."""
        return execute_dry_run_policy(
            registry=self,
            code=code,
            seed=seed,
            max_turns=max_turns,
            replay_run_id=replay_run_id,
            replay_steps=replay_steps,
            smoke_only=smoke_only,
            skill_name=skill_name,
            skill_code=skill_code,
            skills=skills,
        )
