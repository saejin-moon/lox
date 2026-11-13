"""
Dedicated Read-Only Critic & Diagnostician Agent for LOX Swarm Synthesis.

Strictly separated from implementation (ZERO code-writing or `dry_run_policy` tools):
1. Post-Eval Gate (`evaluate_and_gate`): Compares Gen N against the true champion
   and issues a binding PROMOTE or ROLLBACK verdict.
2. Pre-Gen Diagnostician (`diagnose_for_generation`): Investigates DuckDB, policy
   diffs, and NetHack Wiki mechanics to produce a grounded `DiagnosticReport`
   (root causes, protected invariants, and next-gen focus scopes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.prompts import build_system_prompt
from lox.db.schema import connect_db
from lox.db.theories import TheoryDocumentStore
from lox.ui.console import ui

READ_ONLY_CRITIC_TOOLS: set[str] = {
    "query_db",
    "read_policy",
    "diff_policies",
    "read_skill",
    "query_policy_docs",
    "query_wiki",
    "query_monster",
    "query_item",
    "query_theories",
}


@dataclass
class GateVerdict:
    decision: str  # "PROMOTE" | "ROLLBACK"
    target_policy_id: str
    reason: str
    regressions: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    generation: int
    champion_policy_id: str
    root_causes: list[dict[str, Any]] = field(default_factory=list)
    protected_invariants: list[str] = field(default_factory=list)
    next_gen_focus: list[str] = field(default_factory=list)
    raw_summary: str = ""

    def to_prompt_block(self) -> str:
        if (
            not self.root_causes
            and not self.protected_invariants
            and not self.next_gen_focus
        ):
            return self.raw_summary
        parts = [
            f"### Critic Diagnostic Report (Baseline Champion: `{self.champion_policy_id}`)"
        ]
        if self.root_causes:
            parts.append("**Empirical Root Causes:**")
            for rc in self.root_causes:
                sym = rc.get("symptom", "")
                fn = rc.get("offending_function_or_skill", "")
                rids = rc.get("evidence_run_ids", [])
                mech = rc.get("mechanism", "")
                parts.append(f"- [{fn}] {sym} (runs={rids}): {mech}")
        if self.protected_invariants:
            parts.append("**Protected Invariants (DO NOT remove or preempt):**")
            for inv in self.protected_invariants:
                parts.append(f"- {inv}")
        if self.next_gen_focus:
            parts.append("**Recommended Focus Scopes:**")
            for foc in self.next_gen_focus:
                parts.append(f"- {foc}")
        return "\n".join(parts)


class SwarmCriticAgent:
    """Read-only empirical critic & diagnostician (no code-writing or dry-run tools)."""

    def __init__(
        self,
        provider: str = "openrouter",
        model: str | None = None,
        db_path: str = "data/lox.duckdb",
        theories_store: TheoryDocumentStore | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = theories_store or TheoryDocumentStore()
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role="critic",
        )
        self.author.tools.theories_store = self.theories

    @staticmethod
    def _parse_json_object(raw_txt: str) -> dict[str, Any]:
        txt = str(raw_txt or "").strip()
        if "```" in txt:
            m = re.search(r"```(?:json)?\s*\n(.*?)```", txt, flags=re.DOTALL)
            if m:
                txt = m.group(1).strip()
        try:
            parsed = json.loads(txt)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            m_obj = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            if m_obj:
                try:
                    parsed = json.loads(m_obj.group(0))
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    pass
        return {}

    def _fetch_batch_pair_metrics(
        self, candidate_policy_id: str, champion_policy_id: str
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                cand_row = con.execute(
                    """
                    SELECT avg_depth, max_depth, avg_score, max_score, avg_turns
                    FROM batches
                    WHERE candidate_policy_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    [candidate_policy_id],
                ).fetchone()
                champ_row = con.execute(
                    """
                    SELECT avg_depth, max_depth, avg_score, max_score, avg_turns
                    FROM batches
                    WHERE candidate_policy_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    [champion_policy_id],
                ).fetchone()

                def _to_dict(r: Any) -> dict[str, float] | None:
                    if not r:
                        return None
                    return {
                        "avg_depth": float(r[0] or 0.0),
                        "max_depth": float(r[1] or 0.0),
                        "avg_score": float(r[2] or 0.0),
                        "max_score": float(r[3] or 0.0),
                        "avg_turns": float(r[4] or 0.0),
                    }

                return _to_dict(cand_row), _to_dict(champ_row)
            finally:
                con.close()
        except Exception:
            return None, None

    def evaluate_and_gate(
        self,
        generation: int,
        batch_id: str,
        candidate_policy_id: str,
        champion_policy_id: str | None,
    ) -> GateVerdict:
        """
        Post-Eval Gate: compares `candidate_policy_id` (`batch_id`) against
        `champion_policy_id` and decides PROMOTE vs ROLLBACK.
        """
        if (
            generation <= 1
            or not champion_policy_id
            or champion_policy_id == candidate_policy_id
        ):
            return GateVerdict(
                decision="PROMOTE",
                target_policy_id=candidate_policy_id,
                reason="Initial Generation 1 baseline promoted.",
            )

        cand_m, champ_m = self._fetch_batch_pair_metrics(
            candidate_policy_id, champion_policy_id
        )
        hard_regressions: list[str] = []
        if cand_m and champ_m:
            if cand_m["avg_depth"] < champ_m["avg_depth"] - 0.15:
                hard_regressions.append(
                    f"avg_depth regressed ({champ_m['avg_depth']:.2f} -> {cand_m['avg_depth']:.2f})"
                )
            if (
                champ_m["avg_score"] > 0
                and cand_m["avg_score"] < champ_m["avg_score"] * 0.80
            ):
                hard_regressions.append(
                    f"avg_score regressed ({champ_m['avg_score']:.1f} -> {cand_m['avg_score']:.1f})"
                )

        if self.provider == "mock":
            if hard_regressions:
                return GateVerdict(
                    decision="ROLLBACK",
                    target_policy_id=champion_policy_id,
                    reason="; ".join(hard_regressions),
                    regressions=hard_regressions,
                )
            return GateVerdict(
                decision="PROMOTE",
                target_policy_id=candidate_policy_id,
                reason="Mock gate non-inferior to champion.",
            )

        ui.info(
            f"Critic Gate: Comparing `{candidate_policy_id}` (batch `{batch_id}`) vs champion `{champion_policy_id}`..."
        )
        self.author.tools.generation = int(generation)
        prompt = f"""You are the Read-Only Critic Gate for Generation {generation}.
Compare candidate `{candidate_policy_id}` (batch `{batch_id}`) against the current champion `{champion_policy_id}`.

Use your read-only tools (`query_db`, `diff_policies`, `read_policy`, `read_skill`, `query_wiki`) to inspect:
1. Paired metrics in `batches` and `runs` (depth, score, `end_reason` shifts, `starved_with_food`, `policy_exceptions`, `stalled_zero_turn`).
2. `function_stats` (`stagnation_share`, `livelock_score`, `liblit_failure_score`) for `{batch_id}` vs champion.
3. `diff_policies('{champion_policy_id}', '{candidate_policy_id}')` to see what changed in `policy.py` and `skills/*.py`.

Return ONLY a valid JSON object:
{{
  "decision": "PROMOTE" or "ROLLBACK",
  "reason": "Concise empirical justification citing observed numbers and run_ids",
  "regressions": ["list of any observed regressions or empty list"]
}}
"""
        decision = "ROLLBACK" if hard_regressions else "PROMOTE"
        reason = (
            "; ".join(hard_regressions)
            if hard_regressions
            else "Non-inferior to champion."
        )
        regressions = list(hard_regressions)

        try:
            raw = self.author.run_tool_session(
                system_prompt=build_system_prompt(self.db_path),
                user_prompt=prompt,
                max_tool_turns=4,
                allowed_tools=READ_ONLY_CRITIC_TOOLS,
                generation=generation,
                session_mode="CRITIC_GATE",
                agent_role_override="critic",
            )
            parsed = self._parse_json_object(raw)
            llm_dec = str(parsed.get("decision", "")).strip().upper()
            if llm_dec in ("PROMOTE", "ROLLBACK"):
                # Deterministic safety backstop: never promote a hard depth/score collapse
                decision = "ROLLBACK" if hard_regressions else llm_dec
                reason = str(parsed.get("reason") or reason)
                if isinstance(parsed.get("regressions"), list):
                    regressions = [
                        str(x) for x in parsed["regressions"] if str(x).strip()
                    ]
                    for hr in hard_regressions:
                        if hr not in regressions:
                            regressions.append(hr)
        except Exception as exc:
            ui.warn(f"Critic Gate fallback ({exc})")

        target_id = (
            champion_policy_id if decision == "ROLLBACK" else candidate_policy_id
        )
        return GateVerdict(
            decision=decision,
            target_policy_id=target_id,
            reason=reason,
            regressions=regressions,
        )

    def diagnose_for_generation(
        self,
        generation: int,
        champion_policy_id: str | None,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
    ) -> DiagnosticReport:
        """
        Pre-Gen Diagnostician: investigates DuckDB, policy diffs, and NetHack Wiki
        to identify root causes, protected invariants, and next-gen focus scopes.
        """
        champ_id = champion_policy_id or "none"
        if self.provider == "mock":
            return DiagnosticReport(
                generation=generation,
                champion_policy_id=champ_id,
                root_causes=[
                    {
                        "symptom": "Early exploration/survival baseline",
                        "offending_function_or_skill": "none",
                        "evidence_run_ids": [],
                        "mechanism": "Mock diagnostic baseline.",
                    }
                ],
                protected_invariants=[
                    "Preserve working navigation and dialog handling"
                ],
                next_gen_focus=["exploration", "survival"],
                raw_summary="Mock DiagnosticReport",
            )

        ui.info(
            f"Critic Diagnostician: Analyzing empirical state for Generation {generation} (champion=`{champ_id}`)..."
        )
        self.author.tools.generation = int(generation)
        self.author.tools.current_policy_code = str(current_policy or "")
        if skills is not None:
            self.author.tools.skills = dict(skills)

        gen1_note = (
            "In Generation 1, campaign tables (`runs`, `batches`, `ticks`, `function_stats`, `policies`) "
            "are empty and DISALLOWED. Call BOTH `query_wiki` and `query_db` on `expert_runs` / `expert_ticks`."
            if generation == 1
            else (
                f"Inspect `batches`, `runs` (`champion_runs` / recent rolled-back runs), `ticks`, `function_stats`, "
                f"`policies` (`read_policy('{champ_id}')` / `diff_policies`), `expert_runs`, and `query_wiki`."
            )
        )
        prompt = f"""You are the Read-Only Critic & Diagnostician for Generation {generation}.
Your sole job is empirical diagnosis — you DO NOT write code.

{gen1_note}

Identify:
1. Top empirical root causes (cite concrete `run_id`s, `function_stats`, `end_reason`s, and C mechanics from `query_wiki`).
2. `protected_invariants`: existing branches/priorities in the champion that are working and MUST NOT be removed or preempted.
3. `next_gen_focus`: 2–4 non-overlapping problem scopes for the next generation's workers.

Return ONLY a valid JSON object:
{{
  "root_causes": [
    {{
      "symptom": "...",
      "offending_function_or_skill": "...",
      "evidence_run_ids": ["..."],
      "mechanism": "..."
    }}
  ],
  "protected_invariants": ["..."],
  "next_gen_focus": ["..."]
}}
"""
        report = DiagnosticReport(
            generation=generation,
            champion_policy_id=champ_id,
        )
        try:
            raw = self.author.run_tool_session(
                system_prompt=build_system_prompt(self.db_path),
                user_prompt=prompt,
                max_tool_turns=5,
                required_tools=(
                    ["query_wiki", "query_db:expert_runs"] if generation == 1 else None
                ),
                allowed_tools=READ_ONLY_CRITIC_TOOLS,
                generation=generation,
                session_mode="CRITIC_DIAGNOSIS",
                agent_role_override="critic",
            )
            report.raw_summary = raw
            parsed = self._parse_json_object(raw)
            if isinstance(parsed.get("root_causes"), list):
                report.root_causes = [
                    rc for rc in parsed["root_causes"] if isinstance(rc, dict)
                ]
            if isinstance(parsed.get("protected_invariants"), list):
                report.protected_invariants = [
                    str(x) for x in parsed["protected_invariants"] if str(x).strip()
                ]
            if isinstance(parsed.get("next_gen_focus"), list):
                report.next_gen_focus = [
                    str(x) for x in parsed["next_gen_focus"] if str(x).strip()
                ]
        except Exception as exc:
            ui.warn(f"Critic Diagnostician fallback ({exc})")

        try:
            self.theories.save_document(
                generation=generation,
                agent_role="critic",
                doc_type="CRITIC_DIAGNOSIS",
                title=f"Critic Diagnosis G{generation} (Champion: {champ_id})",
                content=report.to_prompt_block(),
                tags=["critic", f"gen_{generation}", "diagnosis"],
            )
        except Exception:
            pass
        return report
