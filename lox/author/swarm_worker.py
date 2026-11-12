"""
Swarm Worker Agent for category-specialized policy synthesis.
Specializes in one of 4 functional domains:
- 'dialog': interactive text prompts, --More-- pagination, y/n traps
- 'threat': tactical combat, kiting, retreat thresholds, melee
- 'survival': hunger thresholds, corpse safety (query_monster), prayer, armor
- 'navigation': BFS pathfinding, room exploration, secret/locked doors, stairs
"""

from __future__ import annotations

import inspect
import json
import random
from typing import Any

from lox.author.agent import AuthorAgent
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive, FrontierCheckpoint
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.ui.console import ui

WORKER_ROLE_DESCRIPTIONS: dict[str, str] = {
    "dialog": (
        "Specialist in interactive game dialogs, --More-- pagination, and text/naming prompts. "
        "Produces reactive Behavior Tree ActionLeaf or condition checks to prevent episode stall traps."
    ),
    "threat": (
        "Specialist in tactical combat and threat response. "
        "Produces reactive Behavior Tree combat subtrees (kiting, corridor bottlenecks, retreat thresholds, melee)."
    ),
    "survival": (
        "Specialist in hero survival, nutrition, and equipment. "
        "Produces reactive Behavior Tree survival subtrees (food consumption, checking corpses for poison via query_monster, prayer, armor equipping)."
    ),
    "navigation": (
        "Specialist in spatial exploration and dungeon navigation. "
        "Produces reactive Behavior Tree navigation subtrees (BFS pathfinding, room exploration, locked door kicking, stairs descent)."
    ),
}


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
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role=self.role,
        )
        self.best_candidate_code: str = ""
        self.best_score: float = -1e9
        self.history: list[dict[str, Any]] = []

    def get_role_prompt(self) -> str:
        desc = self.description or WORKER_ROLE_DESCRIPTIONS.get(
            self.role, "General Policy Specialist"
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

    def evaluate_candidate(
        self,
        code_snippet: str,
        checkpoint: FrontierCheckpoint | None = None,
        steps: int = 25,
    ) -> float:
        """
        Empirically evaluates candidate code directly on a frontier checkpoint or fresh seed.
        Returns the true empirical progression score delta.
        """
        harness = NetHackHarness(max_steps=steps + 50)
        try:
            if checkpoint is not None:
                st = self.frontier.replay_into(harness, checkpoint)
            else:
                st = harness.reset(seed=random.randint(1000, 9999))

            start_score = st.score
            start_hp = st.hp
            start_depth = st.depth

            local_scope: dict[str, Any] = {}
            exec(code_snippet, globals(), local_scope)

            node: Any = None
            for candidate in local_scope.values():
                if hasattr(candidate, "tick"):
                    node = candidate
                    break
                elif inspect.isclass(candidate) and hasattr(candidate, "step"):
                    node = candidate()
                    break

            if node is None:
                return 0.0

            for _ in range(steps):
                if hasattr(node, "tick"):
                    act = node.tick(st)
                elif hasattr(node, "step"):
                    act = node.step(st)
                else:
                    act = node(st)
                if act is None:
                    act = Action.SEARCH
                st, done, _ = harness.step(act)
                if done:
                    break

            score_gain = float(st.score - start_score)
            hp_gain = float((st.hp - start_hp) * 5)
            depth_gain = float((st.depth - start_depth) * 200)
            survival_bonus = 20.0 if not harness.done else -50.0
            return float(score_gain + hp_gain + depth_gain + survival_bonus)
        except Exception:
            return -10.0
        finally:
            harness.close()

    def run_cycle(
        self,
        generation: int,
        checkpoint: FrontierCheckpoint | None = None,
    ) -> dict[str, Any]:
        """
        Runs an N-turn exploration cycle for this category.
        If checkpoint is provided, evaluates candidate code directly on that mid-game state in ~15ms.
        At the end of the cycle, persists a structured reflection report into LanceDB.
        """
        ui.info(
            f"Swarm Worker [{self.role}]: Starting Generation {generation} cycle ({self.max_turns} turns)..."
        )
        best_code = ""
        best_delta = -999.0

        if checkpoint is None:
            checkpoint = self.frontier.sample_checkpoint()

        for turn in range(1, self.max_turns + 1):
            if self.provider == "mock":
                code_snippet = (
                    f"# Subtree for {self.role} Gen {generation} Turn {turn}\n"
                )
                if "dialog" in self.role:
                    code_snippet += "from lox.lang.bt import ActionLeaf\nfrom lox.env.actions import Action\ndialog_node = ActionLeaf(lambda st: Action.SPACE if getattr(st, 'waiting_for_space', False) else None)\n"
                elif "threat" in self.role or "combat" in self.role:
                    code_snippet += "from lox.lang.bt import ActionLeaf\nfrom lox.env.actions import Action\nthreat_node = ActionLeaf(lambda st, bb: Action.WAIT if bb and bb.get_var('in_danger') else None)\n"
                elif (
                    "survival" in self.role
                    or "health" in self.role
                    or "resource" in self.role
                ):
                    code_snippet += "from lox.lang.bt import ActionLeaf\nfrom lox.env.actions import Action\nsurvival_node = ActionLeaf(lambda st, bb: None)\n"
                else:
                    code_snippet += "from lox.lang.bt import ActionLeaf\nfrom lox.env.actions import Action\nnavigation_node = ActionLeaf(lambda st: Action.SEARCH)\n"
                hyp = {
                    "role": self.role,
                    "mechanism": f"Implements reactive {self.role} logic",
                    "observation_from_db": "Empirical inspection from DuckDB",
                }
            else:
                proposal = self.author.synthesize(
                    mode="WRITE",
                    generation=generation,
                    current_policy=best_code,
                    max_tool_turns=4,
                )
                code_snippet = proposal.raw_code_patch
                hyp = proposal.hypothesis

            # Empirically evaluate candidate code on the frontier checkpoint!
            delta = self.evaluate_candidate(
                code_snippet, checkpoint=checkpoint, steps=25
            )
            if delta > best_delta or not best_code:
                best_delta = delta
                best_code = code_snippet

            # Persist intermediate theory to LanceDB with real empirical delta
            self.theories.save_document(
                generation=generation,
                agent_role=self.role,
                doc_type="HYPOTHESIS",
                title=f"{self.role.capitalize()} Hypothesis G{generation}T{turn} (Delta: {delta:+.1f})",
                content=f"### Hypothesis\n{json.dumps(hyp, indent=2)}\n\n```python\n{code_snippet}\n```\n\n**Empirical Checkpoint Delta**: {delta:+.2f}",
                tags=[self.role, f"gen_{generation}"],
                verified_c_rule=str(hyp.get("verified_c_rule", "")),
            )

        # Compile final reflection report with real empirical delta
        report_content = f"""# {self.role.capitalize()} Reflection Report (Generation {generation})
## Best Verified Subtree:
```python
{best_code}
```
## Summary of Findings:
- Domain: {self.role}
- Completed {self.max_turns} exploration turns.
- Best verified score delta: {best_delta:+.2f}
"""
        report_id = self.theories.save_document(
            generation=generation,
            agent_role=self.role,
            doc_type="REFLECTION_REPORT",
            title=f"{self.role.capitalize()} Reflection Report G{generation}",
            content=report_content,
            tags=[self.role, f"gen_{generation}", "reflection"],
        )
        ui.info(
            f"Swarm Worker [{self.role}]: Completed cycle (Best Delta: {best_delta:+.2f}). Saved report: {report_id}"
        )

        return {
            "role": self.role,
            "best_code": best_code,
            "best_delta": best_delta,
            "report_id": report_id,
            "report_content": report_content,
        }
