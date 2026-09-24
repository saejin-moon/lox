"""
LOX-ψ Policy Layer (LOX-ψ): the agentic author (S0, AGENT_PLAN §1.3/§2-S0).

Unlike the bundle-only Reviser (one prompt, one diff), the AgenticAuthor runs a
bounded tool-calling loop: the LLM gathers evidence (query_duckdb, wiki_search,
read_trajectory, read_env_schema, read_manifest, read_program_tree) and emits
exactly ONE final policy diff, which flows through the same validator gates.

Provider-agnostic by design: the loop rides the existing `generate_text`
surface. The protocol is textual S-expressions — each non-final turn contains
only `(tool <name> <args...>)` forms; a turn containing `(revision ...)` ends
the session as the final diff. One repair retry on unextractable output
(MACRO.md §2.4), mirroring the Reviser.

`author_session` orchestrates one gated session end-to-end (manifest + bundle →
author → validator gates → ledger entry → optional commit) and is shared by
scripts/run_author_session.py, scripts/run_prompt_ab.py, and the revision loop.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from lox.deliberative.providers.base import LLMProvider
from lox.policy.author_tools import (
    TOOL_SPECS, execute_tool_call, parse_tool_calls, tool_docs,
)
from lox.policy.dsl import DiffParseError, extract_diff_text
from lox.policy.failure_counters import counters_ctx, load_failure_counters
from lox.policy.ledger import RevisionLedger, DEFAULT_LEDGER_PATH
from lox.policy.manifest import VocabularyManifest, build_manifest
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH
from lox.policy.prompts import load_prompt_text, render_system_prompt
from lox.policy.report import build_report_bundle, DEFAULT_DB_PATH
from lox.policy.validator import validate_diff, ValidatorHooks

MAX_TOOL_OUTPUT_CHARS = 3000   # per-tool-result budget inside the transcript
MAX_TRANSCRIPT_CHARS = 24000   # total conversation budget (tokens are the cost driver)


def _trim_transcript(transcript: str) -> str:
    """Bounds the tool-loop conversation. The transcript is re-sent every turn, so an
    unbounded one is a quadratic token cost. On overflow, keep the opening brief and
    the most recent turns, eliding the middle with an explicit marker."""
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    head = transcript[:MAX_TRANSCRIPT_CHARS // 3]
    tail = transcript[-(2 * MAX_TRANSCRIPT_CHARS // 3):]
    return head + "\n\n[…] earlier tool results elided to bound session tokens (re-query if needed) […]\n\n" + tail


def format_bundle_summary(bundle: dict) -> str:
    """Concise markdown summary of the run report (~400-800 chars) instead of dumping
    14,000 chars of raw JSON. Detailed telemetry is accessible via query_duckdb / read_trajectory."""
    lines = ["# RUN REPORT (scoped summary — use query_duckdb / read_trajectory for details)"]
    scope = bundle.get("scope", {})
    batch = bundle.get("batch", {})
    domain = bundle.get("domain", "unknown")
    rid = scope.get("run_id") or "none"
    eps = batch.get("episodes", scope.get("episode_count", 0))
    lines.append(f"- Domain: {domain} | Run ID: {rid} | Episodes: {eps} ({scope.get('source', 'none')})")

    metrics = []
    if "success_rate" in batch and batch.get("episodes", 0) > 0:
        metrics.append(f"success_rate={batch['success_rate']*100:.1f}%")
    if "median_steps_on_success" in batch and batch["median_steps_on_success"] is not None:
        metrics.append(f"median_steps_success={batch['median_steps_on_success']}")
    if "mean_score" in batch and batch.get("episodes", 0) > 0:
        metrics.append(f"mean_score={batch['mean_score']:.1f}")
    if "median_depth" in batch and batch.get("episodes", 0) > 0:
        metrics.append(f"median_depth={batch['median_depth']:.1f}")
    if "survival_rate" in batch and batch.get("episodes", 0) > 0:
        metrics.append(f"survival_rate={batch['survival_rate']*100:.1f}%")
    if metrics:
        lines.append(f"- Batch Metrics: {', '.join(metrics)}")

    deaths = bundle.get("death_taxonomy", [])
    if deaths:
        top_deaths = [f"{d.get('cause')}: {d.get('count')} (d{d.get('median_depth_at_death')})" for d in deaths[:4]]
        lines.append(f"- Top Deaths: {', '.join(top_deaths)}")

    fc = bundle.get("failure_counters", [])
    if fc:
        top_fc = [f"{c.get('goal')}: {c.get('failures')}" for c in fc[:4]]
        lines.append(f"- Goal Failures: {', '.join(top_fc)}")

    digests = bundle.get("episode_digests", [])
    if digests:
        lines.append("- Representative Episodes:")
        for d in digests[:2]:
            first_line = d.strip().split("\n")[0]
            lines.append(f"    * {first_line}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgenticAuthor: bounded tool loop → one validated diff
# ---------------------------------------------------------------------------

class AgenticAuthor:
    """Evidence-conditioned tool-calling author. Mirrors `Reviser.propose_diff`
    (async, returns diff text, records `last_meta`) so the revision loop and the
    prompt A/B harness can run either author over the same contract."""

    def __init__(
        self,
        provider: LLMProvider,
        ledger: RevisionLedger | None = None,
        *,
        prompt_version: str | None = None,
        max_turns: int = 6,
        db_path: str = DEFAULT_DB_PATH,
        wiki_db_path: str = "data/wiki_index.db",
        include_bundle: bool = True,
        include_program_tree: bool = False,   # program is fetched via tool instead
        failure_counters: dict[str, int] | None = None,
        max_tokens: int = 60000,
    ):
        self.provider = provider
        self.ledger = ledger or RevisionLedger()
        self.prompt_version = prompt_version
        self.max_turns = max_turns
        self.db_path = db_path
        self.wiki_db_path = wiki_db_path
        self.include_bundle = include_bundle
        self.include_program_tree = include_program_tree
        self.failure_counters = failure_counters
        self.max_tokens = max_tokens
        self.last_meta: dict = {}

    # ------------------------------------------------------------------
    async def propose_diff(self, program: PolicyProgram,
                           manifest: VocabularyManifest,
                           bundle: dict | None = None) -> str:
        prompt_text = load_prompt_text(self.prompt_version)
        system = render_system_prompt(prompt_text, tool_docs(domain=manifest.domain))

        user_parts = [
            "# VOCABULARY MANIFEST (the only symbols your diff may reference; "
            "call read_manifest for full descriptions)",
            manifest.render(compact=True),
        ]
        if self.include_program_tree:
            from lox.policy.author_tools import read_program_tree
            user_parts += ["", read_program_tree(program)]
        if self.include_bundle and bundle is not None:
            user_parts += [
                "",
                format_bundle_summary(bundle),
                "",
                "# LAST LEDGER ENTRIES (recent revisions and their fates)",
            ]
            for e in self.ledger.tail(5):
                keep = {k: e.get(k) for k in
                        ("revision", "accepted", "reject_code", "reject_detail", "reason")}
                user_parts.append(json.dumps(keep))
        user_parts += [
            "",
            f"# TASK: propose ONE policy diff for version {program.version + 1} "
            f"(parent {program.version}). Gather evidence with tool calls first; "
            "when you have enough, emit the final diff (first form (revision ...)).",
        ]
        user = "\n".join(user_parts)

        context = {"version": program.version, "domain": manifest.domain}
        tool_log: list[dict] = []
        start = time.perf_counter()
        meta = {"tokens_in": 0, "tokens_out": 0, "tokens_thought": 0}
        self._budget_warned = False

        diff_text = await self._loop(system, user, program, manifest, context,
                                     tool_log, meta, repair=False)
        if diff_text is None:
            diff_text = await self._loop(
                system, user + "\n\n# REPAIR REQUIRED\nYour previous session produced "
                "no extractable diff. Re-emit ONE valid diff document whose FIRST form "
                "is (revision ...). You may still use tools first.",
                program, manifest, context, tool_log, meta, repair=True)
        if diff_text is None:
            raise DiffParseError("ERR_PARSE", "author produced no (revision ...) form "
                                             "within the tool-turn budget")

        self.last_meta = {
            **meta,
            "wall_sec": time.perf_counter() - start,
            "model": getattr(self.provider, "model", "mock"),
            "turns": len(tool_log) and max(t["turn"] for t in tool_log) or 0,
            "tool_log": tool_log,
            "prompt_version": self.prompt_version or "default",
            "token_budget_exceeded": bool(getattr(self, "_budget_warned", False)),
        }
        return diff_text

    # ------------------------------------------------------------------
    async def _loop(self, system: str, user: str, program: PolicyProgram,
                    manifest: VocabularyManifest, context: dict,
                    tool_log: list[dict], meta: dict, repair: bool) -> str | None:
        """Runs the bounded tool loop. Returns the final diff text or None."""
        transcript = user if not repair else user  # repair rewrites the task line
        for turn in range(self.max_turns):
            resp = await self.provider.generate_text(
                system, transcript,
                {**context, "mode": "agentic", "session_turn": turn})
            for k in ("tokens_in", "tokens_out", "tokens_thought"):
                meta[k] += getattr(resp, k, 0) or 0
            raw = resp.raw_text or ""

            # A turn containing (revision ...) is the FINAL answer.
            if "(revision" in raw:
                try:
                    return extract_diff_text(raw)
                except DiffParseError:
                    if turn == self.max_turns - 1:
                        return None
                    transcript += (f"\n\n# AUTHOR TURN {turn + 1} (REJECTED — unextractable)\n"
                                   f"{raw[:1500]}\n\n# SYSTEM\nYour revision form was "
                                   "unbalanced or malformed. Emit tool calls or ONE "
                                   "well-formed diff.")
                    transcript = _trim_transcript(transcript)
                    continue

            calls = parse_tool_calls(raw)
            if not calls:
                transcript += (f"\n\n# AUTHOR TURN {turn + 1}\n{raw[:800]}\n\n# SYSTEM\n"
                               "No tool call and no diff detected. Emit (tool <name> ...) "
                               "forms, or the final (revision ...) diff.")
                transcript = _trim_transcript(transcript)
                continue

            results = []
            for i, (name, args) in enumerate(calls):
                result = execute_tool_call(
                    name, args, manifest=manifest, program=program,
                    db_path=self.db_path, wiki_db_path=self.wiki_db_path)
                tool_log.append({
                    "turn": turn + 1, "call": i, "tool": name,
                    "args": [str(a)[:120] for a in args],
                    "result_preview": result[:200],
                })
                results.append(f"### {name}({', '.join(str(a)[:120] for a in args)})\n{result}")
            transcript += (f"\n\n# AUTHOR TURN {turn + 1} (tool calls)\n{raw[:800]}\n\n"
                           f"# TOOL RESULTS (turn {turn + 1}, truncated per tool)\n"
                           + ("\n\n".join(results))[:MAX_TOOL_OUTPUT_CHARS * 4])
            transcript = _trim_transcript(transcript)
            # Cost guard: the transcript is re-sent every turn. When the session's token
            # budget is spent, nudge the author to emit its final diff now.
            spent = meta["tokens_in"] + meta["tokens_out"] + meta["tokens_thought"]
            if spent > self.max_tokens and not getattr(self, "_budget_warned", False):
                self._budget_warned = True
                transcript += ("\n\n# SYSTEM: TOKEN BUDGET REACHED — emit the final "
                               "(revision ...) diff NOW; no further tool calls.")
        return None


# ---------------------------------------------------------------------------
# One gated author session (shared orchestration)
# ---------------------------------------------------------------------------

@dataclass
class SessionOutcome:
    accepted: bool
    program: PolicyProgram
    candidate: PolicyProgram | None = None
    diff_text: str = ""
    error_code: str | None = None
    gate: str | None = None
    detail: str = ""
    meta: dict = field(default_factory=dict)
    wall_sec: float = 0.0


async def author_session(
    provider: LLMProvider,
    program: PolicyProgram,
    program_path: str,
    *,
    ledger: RevisionLedger | None = None,
    domain: str = "nethack",
    author_mode: str = "agentic",       # "agentic" | "bundle"
    prompt_version: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    ledger_path: str = DEFAULT_LEDGER_PATH,
    max_turns: int = 6,
    hooks: ValidatorHooks | None = None,
    manifest: VocabularyManifest | None = None,
    bundle: dict | None = None,
    commit: bool = True,
    author_label: str | None = None,
) -> SessionOutcome:
    """ONE author session: evidence → diff → validator gates → ledger (+ commit).

    The ONLY program writer is the validator: `commit=True` saves the validated
    candidate; a rejected diff never touches the program file.
    """
    ledger = ledger or RevisionLedger(ledger_path)
    t0 = time.perf_counter()

    if manifest is None:
        manifest = build_manifest(program.version, program.domain,
                                  live_macros={m["name"]: m["body"] for m in program.macros})
    if bundle is None:
        bundle = build_report_bundle(db_path, ledger_path, program.domain)

    model_name = author_label or getattr(provider, "model", None) or "mock"

    if author_mode == "agentic":
        author: AgenticAuthor = AgenticAuthor(
            provider, ledger, prompt_version=prompt_version, max_turns=max_turns,
            db_path=db_path,
            failure_counters=load_failure_counters(db_path),
        )
    else:
        from lox.policy.reviser import Reviser
        author = Reviser(provider, ledger,
                         system_prompt=load_prompt_text(prompt_version))

    try:
        diff_text = await author.propose_diff(program, manifest, bundle)
    except Exception as e:  # noqa: BLE001 — author path failure = parse-level rejection
        entry = ledger.revision_entry(
            revision=program.version + 1, parent_version=program.version,
            author_model=model_name, provider=type(provider).__name__, accepted=False,
            reject_code="ERR_PARSE", reject_detail=f"author output unusable: {e}",
            gate="parse", wall_sec=time.perf_counter() - t0)
        ledger.append(entry)
        return SessionOutcome(accepted=False, program=program, error_code="ERR_PARSE",
                              gate="parse", detail=str(e)[:300],
                              meta=author.last_meta, wall_sec=time.perf_counter() - t0)
    meta = dict(author.last_meta)

    result = validate_diff(diff_text, program, manifest, hooks=hooks)
    wall = time.perf_counter() - t0
    if result.ok:
        candidate = result.candidate
        if commit:
            # S1: the acceptance path writes the per-file authoring tree and the
            # canonical compiled artifact (byte-equivalence enforced fail-loud).
            from lox.policy import compiler  # noqa: PLC0415
            compiler.compile_and_commit(candidate, compiler.tree_dir_for(candidate.domain),
                                        program_path)
        ledger.append(ledger.revision_entry(
            revision=candidate.version, parent_version=program.version,
            author_model=model_name, provider=type(provider).__name__, accepted=True,
            reason=result.diff.header.reason, diff_text=diff_text,
            tokens_in=meta.get("tokens_in", 0), tokens_out=meta.get("tokens_out", 0),
            tokens_thought=meta.get("tokens_thought", 0),
            latency_ms=0.0, wall_sec=wall, delta={"tool_calls": len(meta.get("tool_log", []))},
            prompt_id=f"prompt:{meta.get('prompt_version', 'default')}"))
        return SessionOutcome(accepted=True, program=program, candidate=candidate,
                              diff_text=diff_text, meta=meta, wall_sec=wall)
    ledger.append(ledger.revision_entry(
        revision=result.diff.header.revision if result.diff else program.version + 1,
        parent_version=program.version,
        author_model=model_name, provider=type(provider).__name__, accepted=False,
        reject_code=result.error_code, reject_detail=result.detail[:300],
        gate=result.gate, reason=result.diff.header.reason if result.diff else "",
        diff_text=diff_text[:2000],
        tokens_in=meta.get("tokens_in", 0), tokens_out=meta.get("tokens_out", 0),
        tokens_thought=meta.get("tokens_thought", 0),
        wall_sec=wall,
        prompt_id=f"prompt:{meta.get('prompt_version', 'default')}"))
    return SessionOutcome(accepted=False, program=program, diff_text=diff_text,
                          error_code=result.error_code, gate=result.gate,
                          detail=result.detail[:300], meta=meta, wall_sec=wall)
