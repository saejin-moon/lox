"""
LOX-ψ Policy Layer (LOX-ψ): the agentic author (S0, AGENT_PLAN §1.3/§2-S0).

Unlike the bundle-only Reviser (one prompt, one diff), the AgenticAuthor runs a
bounded tool-calling loop: the LLM gathers evidence (query_duckdb, wiki_search,
read_trajectory, read_env_schema, read_manifest, read_program_tree) and emits
exactly ONE final policy diff, which flows through the same validator gates.

Provider-agnostic by design: the loop rides the existing `generate_text`
surface. The protocol is Pythonic Infix AST — tool calls use standard functional call syntax;
a turn containing a revision header (e.g. `revision: ...`) ends the session as the final diff.
One repair retry on unextractable output (MACRO.md §2.4), mirroring the Reviser.

`author_session` orchestrates one gated session end-to-end (manifest + bundle →
author → validator gates → ledger entry → optional commit) and is shared by
scripts/run_author_session.py, scripts/run_prompt_ab.py, and the revision loop.
"""
from __future__ import annotations

import json
import os
import re
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

MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("LOX_MAX_TOOL_OUTPUT_CHARS", "32000"))   # per-tool-result budget
MAX_TRANSCRIPT_CHARS = int(os.environ.get("LOX_MAX_TRANSCRIPT_CHARS", "450000"))   # total conversation budget (~128k tokens)


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


def repair_diff_text(
    diff_text: str,
    program: PolicyProgram,
    manifest: VocabularyManifest | None = None,
) -> str:
    """Deterministic programmatic repairs for common LLM syntax slips (Tier 1).

    1. Strip markdown fences (```lisp, ```python, etc.)
    2. Normalize revision and parent header to (program.version + 1, program.version)
    3. Ensure `set` tunables have `policy_params.` prefix and correct path
    4. Normalize defmacro formatting (strip stray 'return')
    5. Strip trailing punctuation/commas
    """
    if not diff_text or not diff_text.strip():
        return diff_text

    cleaned = diff_text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    lines = cleaned.splitlines()
    repaired_lines = []

    leaf_map: dict[str, str] = {}
    if manifest and hasattr(manifest, "tunables"):
        for t_path in manifest.tunables:
            leaf = t_path.split(".")[-1]
            leaf_map[leaf] = t_path

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Strip trailing commas or semicolons
        stripped = re.sub(r"[,;]+$", "", stripped).strip()

        # Header normalization
        m_rev = re.match(r"^revision[:\s=]+(\d+)", stripped, re.I)
        if m_rev:
            m_auth = re.search(r'author[:\s=]+["\']?([^,"\']+)["\']?', stripped, re.I)
            m_reas = re.search(r'reason[:\s=]+(?:"([^"\n]*)"|\'([^\'\n]*)\'|([^\n]+))', stripped, re.I)
            auth_str = f', author: "{m_auth.group(1).strip()}"' if m_auth else ""
            reason_val = (m_reas.group(1) or m_reas.group(2) or m_reas.group(3) or "").strip() if m_reas else ""
            reas_str = f', reason: "{reason_val}"' if reason_val else ""
            repaired_lines.append(f"revision: {program.version + 1}, parent: {program.version}{auth_str}{reas_str}")
            continue

        # Callable comparisons: hp_frac('<', 0.6) or count_hostiles(==, 0) -> infix
        op_map = {'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '==', 'ne': '!='}
        cmp_preds = ('hp_frac', 'count_hostiles', 'stairs_dist', 'closest_speed', 'closest_threat',
                     'carried_food_count', 'turns_since_pray', 'current_ac', 'weapon_enchantment', 'unvisited_count',
                     'turns_on_current_level', 'closest_monster_dist')
        def _repl_call_cmp(m):
            pred = m.group(1)
            raw_op = m.group(2).strip('"\'')
            op = op_map.get(raw_op, raw_op)
            val = m.group(3)
            return f"{pred} {op} {val}"
        stripped = re.sub(
            rf'\b({"|".join(cmp_preds)})\s*\(\s*[\'\"]?([<>=!]+|[a-zA-Z]+)[\'\"]?\s*,\s*([a-zA-Z0-9_.-]+)\s*\)',
            _repl_call_cmp,
            stripped
        )

        # Set operations: prefix with policy_params.
        m_set = re.match(r"^(?:set\s+)?([a-zA-Z0-9_.]+)\s*[:= ]\s*(.+)$", stripped, re.I)
        if m_set and not stripped.lower().startswith(("rule", "goal", "macro", "defmacro", "nogood", "note", "reason")):
            var_name = m_set.group(1).strip()
            val_part = m_set.group(2).strip()
            if not var_name.startswith("policy_params."):
                if leaf_map and var_name in leaf_map:
                    canonical_path = leaf_map[var_name]
                    repaired_lines.append(f"set {canonical_path} = {val_part}")
                    continue
                elif "." not in var_name:
                    repaired_lines.append(f"set policy_params.{var_name} = {val_part}")
                    continue

        # Defmacro: `defmacro name(args): return expr` -> `defmacro name(args) = expr`
        m_macro_ret = re.match(r"^(defmacro|macro)\s+([a-zA-Z0-9_()]+)\s*[:=]\s*return\s+(.+)$", stripped, re.I)
        if m_macro_ret:
            repaired_lines.append(f"{m_macro_ret.group(1)} {m_macro_ret.group(2)} = {m_macro_ret.group(3)}")
            continue

        repaired_lines.append(stripped)

    return "\n".join(repaired_lines)


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
            f"when you have enough, emit the final diff in pure Pythonic format:\n"
            f"revision: {program.version + 1}, parent: {program.version}, author: <model>\n"
            "reason: <explanation>\n"
            "<changes>",
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
                "no extractable diff. Re-emit ONE valid diff document whose header is:\n"
                f"revision: {program.version + 1}, parent: {program.version}, author: <model>\n"
                "reason: <explanation>\n"
                "followed by your changes. Emit the diff directly NOW without calling tools.",
                program, manifest, context, tool_log, meta, repair=True)
        if diff_text is None:
            raise DiffParseError("ERR_PARSE", "author produced no valid revision diff "
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

            # A turn containing revision header is the candidate answer.
            if "(revision" in raw or "revision:" in raw or re.search(r"^\s*revision\s+[\d:]", raw, re.MULTILINE | re.IGNORECASE):
                try:
                    cand_diff = extract_diff_text(raw)
                    cand_diff = repair_diff_text(cand_diff, program, manifest)
                    val = validate_diff(cand_diff, program, manifest)
                    if val.ok or turn == self.max_turns - 1:
                        return cand_diff
                    # Compilation error in turn before final turn -> feed back error to LLM!
                    transcript += (
                        f"\n\n# AUTHOR TURN {turn + 1} (DIFF COMPILATION ERROR: Gate '{val.gate}' / '{val.error_code}')\n"
                        f"{raw[:1500]}\n\n"
                        f"# COMPILER DIAGNOSTIC:\n"
                        f"{val.detail}\n\n"
                        f"# SYSTEM\n"
                        f"Your proposed diff failed validator gate '{val.gate}' ({val.error_code}): {val.detail}. "
                        f"Please diagnose and emit the corrected policy diff."
                    )
                    transcript = _trim_transcript(transcript)
                    continue
                except DiffParseError as e:
                    if turn == self.max_turns - 1:
                        return None
                    transcript += (f"\n\n# AUTHOR TURN {turn + 1} (REJECTED — unextractable)\n"
                                   f"{raw[:1500]}\n\n# SYSTEM\nYour revision diff was "
                                   f"unbalanced or malformed: {e}. Emit tool calls or ONE "
                                   "well-formed diff.")
                    transcript = _trim_transcript(transcript)
                    continue

            calls = parse_tool_calls(raw)
            if not calls:
                transcript += (f"\n\n# AUTHOR TURN {turn + 1}\n{raw[:800]}\n\n# SYSTEM\n"
                               "No tool call and no diff detected. Emit tool calls like "
                               "`query_duckdb(\"SELECT ...\")`, or the final policy diff.")
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
            # Cost and turn guard: nudge author to emit diff as turn budget approaches
            spent = meta["tokens_in"] + meta["tokens_out"] + meta["tokens_thought"]
            if spent > self.max_tokens and not getattr(self, "_budget_warned", False):
                self._budget_warned = True
                transcript += ("\n\n# SYSTEM: TOKEN BUDGET REACHED — emit the final "
                               "policy diff NOW; no further tool calls.")
            elif turn == self.max_turns - 2:
                transcript += (f"\n\n# SYSTEM: Turn {turn + 1}/{self.max_turns} complete. "
                               f"You have 1 turn remaining. In the next turn, you MUST emit your "
                               f"final policy diff starting with `revision: {program.version + 1}, "
                               f"parent: {program.version}`.")
            elif turn >= self.max_turns - 1:
                transcript += ("\n\n# SYSTEM: FINAL TURN — emit the final policy diff NOW; "
                               "no further tool calls.")
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

    # Tier 1: Programmatic auto-repair
    diff_text = repair_diff_text(diff_text, program, manifest)
    result = validate_diff(diff_text, program, manifest, hooks=hooks)

    # Tier 2: Interactive LLM error recovery if Tier 1 could not resolve the error
    if not result.ok:
        repair_system = (
            "You are an expert autonomous policy engineer. "
            "Your previous policy diff failed compiler validation. "
            "You must diagnose the error, fix the policy diff, and output ONE valid corrected diff."
        )
        repair_user = (
            f"# TASK: FIX COMPILATION FAILURE FOR REVISION {program.version + 1}\n\n"
            f"## Compiler Diagnostic:\n"
            f"- Gate: {result.gate}\n"
            f"- Error Code: {result.error_code}\n"
            f"- Detail: {result.detail}\n\n"
            f"## Your Previous Diff That Failed:\n"
            f"{diff_text}\n\n"
            f"## Closed Vocabulary Manifest:\n"
            f"{manifest.render(compact=True)}\n\n"
            f"## Repair Instructions:\n"
            f"1. Fix the compilation error above (ensure valid predicate signatures, certified goals, and closed verbs).\n"
            f"2. Emit ONLY the corrected policy diff starting with:\n"
            f"revision: {program.version + 1}, parent: {program.version}, author: {model_name}\n"
            f"reason: <explanation of fix>\n"
            f"followed by your corrected changes. Emit the diff directly now."
        )
        try:
            repair_resp = await provider.generate_text(
                repair_system, repair_user,
                {"version": program.version, "domain": manifest.domain, "mode": "repair"}
            )
            for k in ("tokens_in", "tokens_out", "tokens_thought"):
                meta[k] += getattr(repair_resp, k, 0) or 0
            repair_raw = repair_resp.raw_text or ""
            if "revision:" in repair_raw or "(revision" in repair_raw or re.search(r"^\s*revision\s+[\d:]", repair_raw, re.M | re.I):
                cand_repaired = extract_diff_text(repair_raw)
                cand_repaired = repair_diff_text(cand_repaired, program, manifest)
                tier2_val = validate_diff(cand_repaired, program, manifest, hooks=hooks)
                if tier2_val.ok:
                    result = tier2_val
                    diff_text = cand_repaired
        except Exception:
            pass  # Fall through to logging the original rejection

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
