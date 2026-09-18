"""
CORP-Ω Policy Layer: the reviser — author prompt builder + diff proposal (R3, §4.5).

System prompt = role definition + output contract (ONE diff, closed vocabulary, grounded
reasons). User prompt = vocabulary manifest + run-report bundle + last 5 ledger entries
+ optional RAG slices. The provider emits raw text; `extract_diff_text` isolates the
(revision ...) form; a parse failure triggers exactly ONE repair retry (MACRO.md §2.4).
"""
from __future__ import annotations

import json
import time

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.policy.dsl import DiffParseError, extract_diff_text
from corp.policy.ledger import RevisionLedger
from corp.policy.manifest import VocabularyManifest
from corp.policy.program import PolicyProgram
from corp.policy.report import build_report_bundle

SYSTEM_PROMPT = """You are the strategy-tuning author of a symbolic NetHack agent (CORP).
Your ONLY output is ONE policy diff — an S-expression document whose first form is
(revision ...). You may ONLY use the vocabulary manifest provided in the user message:
predicates, verbs, certified goals, and policy_params.* leaves with their bounds. Every
change needs a reason grounded in the run report (deaths, stalls, goal stats). You cannot
write code, invent predicates, or exceed declared bounds. Emit the diff and nothing else.

Diff forms you may use (one per line, omit any part you do not need — never write
square brackets, they are only used below to mark optional parts):
  (set policy_params.<dotted.leaf> <value>)
  (rule add tactic_rules (when <expr>) (do <verb>) (unless <expr>)? (note "<str>")?)
  (rule remove tactic_rules <index>)
  (goal prioritize <goal> (when <expr>)? (until <expr>)?)
  (goal deprioritize <goal> (after <expr>)?)
  (goal set_threshold <goal> <owned-param-path> <value>)
  (nogood add (when <expr>) (cause "<str>") (forbid <task>)?)
  (defmacro <name> <expr>)   ; parameterless sugar over manifest predicates — no recursion
  (note "<str>")

Rules:
- header: (revision <parent+1> (parent <parent>) (author "<model>") (domain <domain>) (reason "<grounded rationale>"))
- conditions MUST be parenthesized predicate calls, e.g. (when (is_fighting)) or
  (hp_frac <= 0.40); combinators and/or/not take call expressions only.
- condition depth <= 3.
- keep diffs surgical: tune params, add/retire tactic rules, reprioritize goals.
- do NOT restate unchanged parts of the program; do NOT remove certified goals.
- emit the diff forms flat, one per line, no leading indentation needed.
"""


def build_author_prompt(
    program: PolicyProgram,
    manifest: VocabularyManifest,
    bundle: dict | None = None,
    ledger_tail: list[dict] | None = None,
    rag_slices: list[str] | None = None,
) -> tuple[str, str]:
    """Renders (system, user) prompts. The manifest IS the vocabulary prompt (§4.2 rule 5)."""
    bundle = bundle or {}
    ledger_tail = ledger_tail or []
    rag_slices = rag_slices or []

    user_parts = [
        "# VOCABULARY MANIFEST (the only symbols you may reference)",
        manifest.render(),
        "",
        "# CURRENT PROGRAM (parent version %d)" % program.version,
        "strategy_plan (in priority order — first active goal wins):",
    ]
    for g in program.strategy_plan:
        user_parts.append(f"  - {g.goal}: when={g.when} until={g.until}"
                          + (f" directive={g.directive}" if g.directive else "")
                          + (f" max_steps={g.max_steps}" if g.max_steps else ""))
    if program.tactic_rules:
        user_parts.append("tactic_rules (first match wins):")
        for i, r in enumerate(program.tactic_rules):
            user_parts.append(f"  [{i}] when={r['when']} do={r['do']}"
                              + (f" unless={r['unless']}" if r.get("unless") else ""))
    else:
        user_parts.append("tactic_rules: (none)")
    if program.macros:
        user_parts.append("live macros:")
        for m in program.macros:
            user_parts.append(f"  {m['name']}: {m['body']}")
    if program.params:
        user_parts.append(f"tuned params: {json.dumps(program.params)}")

    user_parts += [
        "",
        "# RUN REPORT (last batch)",
        "```json",
        json.dumps(bundle, indent=2)[:6000],
        "```",
        "",
        "# LAST LEDGER ENTRIES (recent revisions and their fates)",
    ]
    for e in ledger_tail[-5:]:
        keep = {k: e.get(k) for k in ("revision", "accepted", "reject_code", "reject_detail", "reason")}
        user_parts.append(json.dumps(keep))
    if rag_slices:
        user_parts += ["", "# DOMAIN KNOWLEDGE (RAG slices)"]
        user_parts += [s[:1500] for s in rag_slices[:4]]
    user_parts += [
        "",
        f"# TASK: propose ONE policy diff for version {program.version + 1} "
        f"(parent {program.version}). Respond with the diff only.",
    ]
    return SYSTEM_PROMPT, "\n".join(user_parts)


class Reviser:
    """Calls the author provider, isolates the diff, performs one repair retry."""

    def __init__(self, provider: LLMProvider, ledger: RevisionLedger | None = None,
                 rag_slices: list[str] | None = None):
        self.provider = provider
        self.ledger = ledger or RevisionLedger()
        self.rag_slices = rag_slices or []
        self.last_meta: dict = {}

    async def propose_diff(self, program: PolicyProgram, manifest: VocabularyManifest,
                           bundle: dict | None = None) -> str:
        system, user = build_author_prompt(
            program, manifest, bundle,
            ledger_tail=self.ledger.tail(5),
            rag_slices=self.rag_slices,
        )
        context = {"version": program.version}
        start = time.perf_counter()
        resp = await self.provider.generate_text(system, user, context)
        self.last_meta = {
            "tokens_in": getattr(resp, "tokens_in", 0) or resp.tokens_consumed,
            "tokens_out": getattr(resp, "tokens_out", 0),
            "tokens_thought": getattr(resp, "tokens_thought", 0),
            "latency_ms": resp.latency_ms,
            "wall_sec": time.perf_counter() - start,
            "model": getattr(self.provider, "model", "mock"),
            "thinking": (resp.thinking_content or "")[:8000],
            "response_text": (resp.raw_text or "")[:8000],
        }
        try:
            return extract_diff_text(resp.raw_text)
        except DiffParseError as e:
            # ONE repair retry with the error fed back (MACRO.md §2.4)
            repair_user = (
                user + "\n\n# REPAIR REQUIRED\nYour previous output failed to parse:\n"
                f"{e}\nEmit ONE valid diff document. The FIRST form must be (revision ...)."
            )
            resp2 = await self.provider.generate_text(system, repair_user, context)
            for k in ("tokens_in", "tokens_out", "tokens_thought"):
                self.last_meta[k] += getattr(resp2, k, 0) or 0
            self.last_meta["wall_sec"] = time.perf_counter() - start
            self.last_meta["repaired"] = True
            self.last_meta["thinking"] += "\n--- repair ---\n" + (resp2.thinking_content or "")[:4000]
            self.last_meta["response_text"] += "\n--- repair ---\n" + (resp2.raw_text or "")[:4000]
            return extract_diff_text(resp2.raw_text)  # may raise → caller rejects ERR_PARSE