"""
Swarm Worker Agent for category-specialized policy synthesis.
Specializes in one of 4 functional domains:
- 'dialog': interactive text prompts, --More-- pagination, y/n traps
- 'threat': tactical combat, kiting, retreat thresholds, melee
- 'survival': hunger thresholds, corpse safety (query_monster), prayer, armor
- 'navigation': BFS pathfinding, room exploration, secret/locked doors, stairs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.bt_eval import (
    ObservationDigest,
    evaluate_bt_subtree,
    sample_condition_fixtures,
)
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive, FrontierCheckpoint
from lox.ui.console import ui


class SwarmWorker:
    """Category-specialized worker agent that runs iterative experiments on tier checkpoints."""

    def __init__(
        self,
        role: str,
        description: str = "",
        target_metrics: list[str] | None = None,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        db_path: str = "data/lox.duckdb",
        theories_path: str = "data/workspace/theories.lance",
        frontier_dir: str = "data/frontier",
        max_turns: int = 4,
        max_repair_turns: int = 3,
    ) -> None:
        self.role = role.lower()
        self.description = description
        self.target_metrics = target_metrics or []
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = TheoryDocumentStore(db_path=theories_path)
        self.frontier = FrontierArchive(frontier_dir=frontier_dir)
        self.max_turns = max_turns
        self.max_repair_turns = max_repair_turns
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role=self.role,
        )
        self.author.tools.theories_store = self.theories
        self.best_candidate_code: str = ""
        self.best_score: float = -1e9
        self.history: list[dict[str, Any]] = []

    def get_role_prompt(self) -> str:
        desc = self.description or (
            f"Category specialist for '{self.role}'. Synthesizes modular Behavior Tree subtrees "
            f"grounded in empirical DuckDB telemetry and game reference queries."
        )
        metrics_hint = (
            f"\n### Target Telemetry Metrics: {self.target_metrics}"
            if self.target_metrics
            else ""
        )
        return f"""You are the **{self.role.upper()} Specialist** in the LOX Multi-Agent Swarm.
{desc}{metrics_hint}

### Strict Swarm & 'Hard Code Nothing' Invariants:
1. **Behavior Tree Subtree Output**:
   Your output MUST be a clean Behavior Tree component (`ActionLeaf`, `ConditionLeaf`, `Sequence`, or `PrioritySelector`) or specialist method that plugs into the shared `Blackboard`.
2. **Zero Arbitrary Attributes**:
   Do NOT mutate or expect raw `self.*` attributes on Policy. Inter-node state MUST read and write through `blackboard.get_var(key)` / `blackboard.set_var(key, val)`.
3. **Empirical Verification**:
   Query `query_monster` for exact monster stats and corpse poison/nutrition, `query_item` for items, and `query_db` for telemetry.
"""

    def _build_worker_brief(
        self,
        generation: int,
        checkpoint: FrontierCheckpoint | None,
        replay_fixtures: list[dict[str, Any]] | None = None,
    ) -> str:
        """Builds dynamic category brief with checkpoint, failure fixtures, and prior LanceDB memory."""
        parts = [self.get_role_prompt()]
        if checkpoint is not None:
            raw_log = getattr(checkpoint, "game_log", "") or ""
            log_tail = (
                " | ".join(raw_log.strip().splitlines()[-3:])
                if raw_log.strip()
                else "none"
            )
            parts.append(
                f"### Frontier Checkpoint Context (Gen {generation}, {self.max_turns} Competing Hypotheses):\n"
                f"- Checkpoint: `{checkpoint.checkpoint_id}` | Seed: {checkpoint.seed} | Depth: Dlvl {checkpoint.depth} ({checkpoint.milestone})\n"
                f"- Turns Reached: {checkpoint.turns} | Prefix Bytes: {len(checkpoint.action_bytes)}\n"
                f"- Recent Log: {log_tail[:300]}"
            )
        else:
            parts.append(
                f"### Fresh Seed Exploration (Gen {generation}, {self.max_turns} Competing Hypotheses)"
            )

        if replay_fixtures:
            fix_lines = "\n".join(
                f"- run_id=`{f['run_id']}` (seed={f['seed']}, dlvl={f['depth']}, end=`{f['end_reason']}`, rewind_step={f['stop_at_byte_idx']})"
                for f in replay_fixtures
            )
            parts.append(
                f"### Condition-Matched DuckDB Failure Fixtures (auto-tested in `dry_run_policy`):\n{fix_lines}"
            )

        try:
            q_str = f"{self.role} {' '.join(self.target_metrics)}".strip()
            prior_docs = self.theories.search_theories(q_str, limit=3)
            if prior_docs:
                prior_lines = "\n".join(
                    f"- [{d.get('doc_type', 'DOC')}] {d.get('title', '')}: {str(d.get('content', '')).strip()[:280]}"
                    for d in prior_docs
                )
                parts.append(
                    f"### Prior Theories & Ruled-Out Findings from LanceDB:\n{prior_lines}"
                )
        except Exception:
            pass

        return "\n\n".join(parts)

    def evaluate_candidate(
        self,
        code_snippet: str,
        checkpoint: FrontierCheckpoint | None = None,
        steps: int = 25,
        replay_fixture: dict[str, Any] | None = None,
        baseline_policy_code: str = "",
        skills: dict[str, str] | None = None,
    ) -> tuple[ObservationDigest, str]:
        """
        Empirically evaluates candidate code directly on a frontier checkpoint, failure replay, or fresh seed.
        Delegates to shared `evaluate_bt_subtree`. Zero arbitrary scalar score formulas.
        """
        return evaluate_bt_subtree(
            code_snippet,
            checkpoint=checkpoint,
            frontier=self.frontier,
            replay_fixture=replay_fixture,
            steps=steps,
            baseline_policy_code=baseline_policy_code,
            skills=skills,
        )

    def _load_baseline_scaffold(self, explicit_scaffold: str = "") -> str:
        if explicit_scaffold and explicit_scaffold.strip():
            return explicit_scaffold
        try:
            cand = Path(self.db_path).parent / "workspace" / "policy.py"
            if cand.exists():
                return cand.read_text(encoding="utf-8")
        except Exception:
            pass
        return ""

    def _load_skills(
        self, explicit_skills: dict[str, str] | None = None
    ) -> dict[str, str]:
        if explicit_skills is not None:
            return dict(explicit_skills)
        skills_dir = Path(self.db_path).parent / "workspace" / "skills"
        if skills_dir.exists():
            return {
                p.stem: p.read_text(encoding="utf-8")
                for p in sorted(skills_dir.glob("*.py"))
                if p.stem != "__init__"
            }
        return {}

    def run_cycle(
        self,
        generation: int,
        checkpoint: FrontierCheckpoint | None = None,
        scaffold_code: str = "",
        skills: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Runs a single warm multi-hypothesis exploration cycle for this category:
        1. Audits DuckDB / Wiki / LanceDB / Skills
        2. Brainstorms `self.max_turns` competing hypotheses (`H1..HK`)
        3. Translates, dry-runs (with baseline scaffold + skills + failure fixtures), and repairs each sequentially
        4. Persists per-hypothesis THEORY docs and a comparative REFLECTION_REPORT to LanceDB.
        """
        ui.info(
            f"Swarm Worker [{self.role}]: Starting Gen {generation} warm multi-hypothesis session ({self.max_turns} hypotheses)..."
        )
        if checkpoint is None:
            checkpoint = self.frontier.sample_checkpoint()

        active_scaffold = self._load_baseline_scaffold(scaffold_code)
        active_skills = self._load_skills(skills)
        replay_fixtures = sample_condition_fixtures(
            self.db_path,
            self.role,
            self.target_metrics,
            generation=generation,
            limit=2,
        )
        brief = self._build_worker_brief(
            generation=generation,
            checkpoint=checkpoint,
            replay_fixtures=replay_fixtures,
        )

        session = self.author.run_swarm_worker_session(
            generation=generation,
            brief=brief,
            current_policy=active_scaffold,
            num_hypotheses=self.max_turns,
            checkpoint=checkpoint,
            frontier=self.frontier,
            replay_fixtures=replay_fixtures,
            scaffold_code=active_scaffold,
            skills=active_skills,
        )

        raw_exps = list(session.get("experiments") or [])
        hyp_list = list(session.get("hypotheses") or [])
        reflection_text = str(session.get("reflection") or "").strip()

        # Deduplicate repaired runs by hypothesis label, keeping the latest attempt per label
        by_hyp: dict[str, dict[str, Any]] = {}
        for idx, exp in enumerate(raw_exps):
            h_key = str(exp.get("hypothesis") or f"H{idx + 1}").strip()
            by_hyp[h_key] = exp
        experiments = list(by_hyp.values())

        # If LLM emitted code in text without any dry_run_policy call recorded, evaluate it now
        if not experiments:
            fallback_code = str(session.get("final_code") or "").strip()
            obs, err_diag = self.evaluate_candidate(
                fallback_code,
                checkpoint=checkpoint,
                steps=25,
                baseline_policy_code=active_scaffold,
                skills=active_skills,
            )
            experiments.append(
                {
                    "hypothesis": "H1: emitted subtree",
                    "code": fallback_code,
                    "observations": [obs] if not err_diag else [],
                    "triggered_steps": obs.triggered_steps if not err_diag else 0,
                    "total_steps": obs.total_eval_steps if not err_diag else 0,
                    "error": err_diag,
                    "passed": not bool(err_diag),
                }
            )

        # Persist per-hypothesis THEORY docs and score for best selection
        evaluated_records: list[dict[str, Any]] = []
        for idx, exp in enumerate(experiments, start=1):
            h_label = str(exp.get("hypothesis") or f"H{idx}")
            h_code = str(exp.get("code") or "")
            h_err = str(exp.get("error") or "")
            obs_list: list[ObservationDigest] = list(exp.get("observations") or [])
            primary_obs = obs_list[0] if obs_list else ObservationDigest()
            trig_sum = int(exp.get("triggered_steps", primary_obs.triggered_steps))
            tot_sum = int(
                exp.get(
                    "total_steps",
                    primary_obs.total_eval_steps or primary_obs.steps_taken,
                )
            )
            status_label = "OK" if not h_err else "ERROR"
            obs_block = (
                "\n".join(f"- {o.summary()}" for o in obs_list)
                if obs_list
                else f"- Execution Error: {h_err}"
            )
            self.theories.save_document(
                generation=generation,
                agent_role=self.role,
                doc_type="THEORY",
                title=f"{self.role.capitalize()} Hypothesis G{generation}#{idx} ({h_label[:48]}) [{status_label}]",
                content=(
                    f"### Hypothesis: {h_label}\n\n"
                    f"```python\n{h_code}\n```\n\n"
                    f"### Empirical Outcomes (Triggered: {trig_sum}/{tot_sum}):\n"
                    f"{obs_block}\n"
                    f"- Status: {'Clean' if not h_err else h_err}"
                ),
                tags=[self.role, f"gen_{generation}", "theory", "hypothesis"],
            )
            evaluated_records.append(
                {
                    "idx": idx,
                    "hypothesis": h_label,
                    "code": h_code,
                    "obs": primary_obs,
                    "obs_list": obs_list,
                    "triggered_steps": trig_sum,
                    "total_steps": tot_sum,
                    "error": h_err,
                }
            )

        # Pick winning verified subtree: clean > triggered > survived > depth > unique_tiles
        clean_recs = [
            r for r in evaluated_records if not r["error"] and r["code"].strip()
        ]
        if clean_recs:
            selected = max(
                clean_recs,
                key=lambda r: (
                    1 if r["triggered_steps"] > 0 else 0,
                    1 if r["obs"].survived else 0,
                    max(
                        (o.depth_end for o in r["obs_list"]), default=r["obs"].depth_end
                    ),
                    max(
                        (o.unique_tiles for o in r["obs_list"]),
                        default=r["obs"].unique_tiles,
                    ),
                    r["triggered_steps"],
                ),
            )
        else:
            selected = evaluated_records[-1]

        best_code = selected["code"]
        best_obs = selected["obs"]

        hyps_bullet = "\n".join(
            f"- {h.get('id', '?')}: {h.get('mechanism', str(h))}" for h in hyp_list
        )
        findings_bullets = "\n".join(
            f"- {r['hypothesis']} (Trig {r['triggered_steps']}/{r['total_steps']}): "
            + (r["obs"].summary() if not r["error"] else f"ERROR: {r['error']}")
            for r in evaluated_records
        )
        reflection_sec = (
            f"\n## Worker Comparative Reflection:\n{reflection_text}\n"
            if reflection_text
            else ""
        )
        planned_sec = (
            f"\n## Planned Competing Hypotheses:\n{hyps_bullet}\n"
            if hyps_bullet
            else ""
        )
        report_content = f"""# {self.role.capitalize()} Comparative Reflection Report (Generation {generation})
## Chosen Winning Subtree ({selected["hypothesis"]}):
```python
{best_code}
```
## Empirical Outcome of Chosen Subtree:
- Triggered {selected["triggered_steps"]}/{selected["total_steps"]} steps across {max(1, len(selected["obs_list"]))} context(s)
- {best_obs.summary()}
{planned_sec}
## Tested Hypotheses Comparison:
{findings_bullets}
{reflection_sec}"""
        report_id = self.theories.save_document(
            generation=generation,
            agent_role=self.role,
            doc_type="REFLECTION_REPORT",
            title=f"{self.role.capitalize()} Reflection Report G{generation}",
            content=report_content,
            tags=[self.role, f"gen_{generation}", "reflection"],
        )
        ui.info(
            f"Swarm Worker [{self.role}]: Tested {len(evaluated_records)} hypothesis/hypotheses -> Winner [{selected['hypothesis']}] "
            f"(Trig {selected['triggered_steps']}/{selected['total_steps']} | {best_obs.summary()}). Saved report: {report_id}"
        )

        return {
            "role": self.role,
            "best_code": best_code,
            "observation": best_obs,
            "report_id": report_id,
            "report_content": report_content,
            "experiments": evaluated_records,
            "reflection": reflection_text,
        }
