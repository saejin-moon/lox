"""
Swarm Worker Agent for category-specialized policy synthesis.
Specializes in one of 4 functional domains:
- 'dialog': interactive text prompts, --More-- pagination, y/n traps
- 'threat': tactical combat, kiting, retreat thresholds, melee
- 'survival': hunger thresholds, corpse safety (query_monster), prayer, armor
- 'navigation': BFS pathfinding, room exploration, secret/locked doors, stairs
"""

from __future__ import annotations

import collections
import inspect
import random
import traceback
from dataclasses import dataclass, field
from typing import Any

from lox.author.agent import AuthorAgent
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive, FrontierCheckpoint
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.bt import (
    ActionLeaf,
    BehaviorNode,
    BehaviorTree,
    Blackboard,
    ConditionLeaf,
)
from lox.ui.console import ui


@dataclass(slots=True)
class ObservationDigest:
    """Rich multi-dimensional empirical outcome from evaluating a candidate Behavior Tree node."""

    survived: bool = False
    steps_taken: int = 0
    end_reason: str = ""
    hp_start: int = 0
    hp_end: int = 0
    depth_start: int = 1
    depth_end: int = 1
    score_start: int = 0
    score_end: int = 0
    unique_tiles: int = 0
    actions_taken: dict[str, int] = field(default_factory=dict)
    dialog_resolved: bool = False

    @property
    def hp_delta(self) -> int:
        return self.hp_end - self.hp_start

    @property
    def depth_delta(self) -> int:
        return self.depth_end - self.depth_start

    def summary(self) -> str:
        status = (
            "SURVIVED"
            if self.survived
            else f"TERMINATED ({self.end_reason or 'death'})"
        )
        hp_str = f"{self.hp_end}/{self.hp_start} (Δ{self.hp_delta:+d})"
        depth_str = f"Dlvl {self.depth_end} (Δ{self.depth_delta:+d})"
        actions_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(self.actions_taken.items()))
            or "none"
        )
        return (
            f"Status: {status} | Steps: {self.steps_taken} | HP: {hp_str} | "
            f"Depth: {depth_str} | Unique Tiles: {self.unique_tiles} | Actions: [{actions_str}]"
        )


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
    ) -> tuple[ObservationDigest, str]:
        """
        Empirically evaluates candidate code directly on a frontier checkpoint or fresh seed.
        Returns (ObservationDigest, error_diagnostic). Zero arbitrary scalar score formulas.
        """
        harness = NetHackHarness(max_steps=steps + 50)
        try:
            if checkpoint is not None:
                st = self.frontier.replay_into(harness, checkpoint)
            else:
                st = harness.reset(seed=random.randint(1000, 9999))

            start_score = int(st.score)
            start_hp = int(st.hp)
            start_depth = int(st.depth)

            local_scope: dict[str, Any] = {}
            exec(code_snippet, globals(), local_scope)

            node: Any = None
            # 1. Prefer instantiated BehaviorTree / BehaviorNode objects
            for candidate in local_scope.values():
                if not inspect.isclass(candidate) and (
                    hasattr(candidate, "tick")
                    or hasattr(candidate, "step")
                    or callable(candidate)
                ):
                    node = candidate
                    break

            # 2. If no instance found, check for Policy or Node classes to instantiate
            if node is None:
                builtin_bt_types = (
                    BehaviorTree,
                    ActionLeaf,
                    ConditionLeaf,
                    Blackboard,
                    BehaviorNode,
                )
                for candidate in local_scope.values():
                    if (
                        inspect.isclass(candidate)
                        and candidate not in builtin_bt_types
                        and (hasattr(candidate, "step") or hasattr(candidate, "tick"))
                    ):
                        try:
                            node = candidate()
                            break
                        except Exception:
                            pass

            if node is None:
                return (
                    ObservationDigest(hp_start=start_hp, depth_start=start_depth),
                    "No BehaviorTree node (.tick(st)) or Policy class (.step(st)) found in code snippet.",
                )

            bb = Blackboard()
            action_counts: dict[str, int] = collections.defaultdict(int)
            visited_tiles: set[tuple[int, int]] = set()

            for _ in range(steps):
                visited_tiles.add((st.y, st.x))
                if isinstance(node, BehaviorTree):
                    act = node.tick(st)
                elif hasattr(node, "tick"):
                    try:
                        res = node.tick(st, bb)
                    except TypeError:
                        res = node.tick(st)
                    act = res[1] if isinstance(res, tuple) else res
                elif hasattr(node, "step"):
                    act = node.step(st)
                else:
                    act = node(st)
                if act is None:
                    act = Action.SEARCH

                act_name = act.name if hasattr(act, "name") else str(act)
                action_counts[act_name] += 1
                st, done, _ = harness.step(act)
                if done:
                    break

            visited_tiles.add((st.y, st.x))
            obs = ObservationDigest(
                survived=not harness.done,
                steps_taken=harness.step_count,
                end_reason=str(getattr(st, "end_reason", ""))
                if harness.done
                else "active",
                hp_start=start_hp,
                hp_end=int(st.hp),
                depth_start=start_depth,
                depth_end=int(st.depth),
                score_start=start_score,
                score_end=int(st.score),
                unique_tiles=len(visited_tiles),
                actions_taken=dict(action_counts),
                dialog_resolved=not getattr(st, "waiting_for_space", False)
                and not getattr(st, "in_text_prompt", False),
            )
            return obs, ""
        except Exception as e:
            tb = traceback.format_exc()
            return ObservationDigest(), f"{type(e).__name__}: {e}\n{tb}"
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
        if checkpoint is None:
            checkpoint = self.frontier.sample_checkpoint()

        turn_results: list[dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            theory_desc = ""
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
                theory_desc = f"Implements reactive {self.role} Behavior Tree logic for checkpoint depth {checkpoint.depth if checkpoint else 1}."
            else:
                last_code = turn_results[-1]["code"] if turn_results else ""
                proposal = self.author.synthesize(
                    mode="WRITE",
                    generation=generation,
                    current_policy=last_code,
                    max_tool_turns=4,
                )
                code_snippet = proposal.raw_code_patch
                theory_desc = str(
                    proposal.hypothesis.get("mechanism")
                    or proposal.hypothesis.get("observation_from_db")
                    or f"Evolving {self.role} subtree"
                )

            # Empirically evaluate candidate code on the frontier checkpoint!
            obs, err_diag = self.evaluate_candidate(
                code_snippet, checkpoint=checkpoint, steps=25
            )

            # In-turn self-repair loop: if checkpoint evaluation raised a syntax or runtime error
            repair_turns = 0
            while (
                err_diag
                and repair_turns < self.max_repair_turns
                and self.provider != "mock"
            ):
                repair_turns += 1
                ui.warn(
                    f"Swarm Worker [{self.role}]: Checkpoint error caught; executing self-repair turn ({repair_turns}/{self.max_repair_turns})..."
                )
                repair_proposal = self.author.synthesize(
                    mode="WRITE",
                    generation=generation,
                    current_policy=code_snippet,
                    max_tool_turns=2,
                )
                code_snippet = repair_proposal.raw_code_patch
                obs, err_diag = self.evaluate_candidate(
                    code_snippet, checkpoint=checkpoint, steps=25
                )

            turn_results.append(
                {
                    "turn": turn,
                    "code": code_snippet,
                    "obs": obs,
                    "theory": theory_desc,
                    "error": err_diag,
                }
            )

            # Persist intermediate theory to LanceDB with real empirical observation facts
            status_label = "OK" if not err_diag else "ERROR"
            self.theories.save_document(
                generation=generation,
                agent_role=self.role,
                doc_type="THEORY",
                title=f"{self.role.capitalize()} Theory G{generation}T{turn} ({status_label})",
                content=(
                    f"### Theory: {self.role.capitalize()} Exploration (Turn {turn})\n"
                    f"{theory_desc}\n\n"
                    f"```python\n{code_snippet}\n```\n\n"
                    f"### Empirical Outcome:\n"
                    f"- {obs.summary()}\n"
                    f"- Execution: {'Clean' if not err_diag else err_diag}"
                ),
                tags=[self.role, f"gen_{generation}", "theory"],
            )

        # Select chosen subtree: prefer clean runs, prioritized by survival and progression
        clean_runs = [r for r in turn_results if not r["error"]]
        if clean_runs:
            selected = max(
                clean_runs,
                key=lambda r: (
                    1 if r["obs"].survived else 0,
                    r["obs"].depth_end,
                    r["obs"].unique_tiles,
                ),
            )
        else:
            selected = turn_results[-1]

        best_code = selected["code"]
        best_obs = selected["obs"]

        # Compile final reflection report summarizing all exploration theories
        findings_bullets = "\n".join(
            f"- Turn {r['turn']}: {r['theory']} -> {r['obs'].summary()}"
            for r in turn_results
        )
        report_content = f"""# {self.role.capitalize()} Reflection Report (Generation {generation})
## Chosen Verified Subtree:
```python
{best_code}
```
## Empirical Outcome of Chosen Subtree:
- {best_obs.summary()}

## Exploration Turn Digest:
{findings_bullets}
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
            f"Swarm Worker [{self.role}]: Completed cycle ({best_obs.summary()}). Saved report: {report_id}"
        )

        return {
            "role": self.role,
            "best_code": best_code,
            "observation": best_obs,
            "report_id": report_id,
            "report_content": report_content,
        }
