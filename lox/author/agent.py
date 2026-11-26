"""
Stateless Per-Generation Author Agent with Compiler-in-the-Loop Self-Correction
for the Option-A Peer Swarm.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv

from lox.author.llm_client import (
    call_llm_api,
    compact_tool_messages,
    extract_usage_cost,
    extract_usage_tokens,
    resolve_api_key,
)
from lox.author.parsing import (
    extract_all_skill_codes,
    extract_skill_code,
    extract_yaml_and_python,
    has_executable_skill_input,
    has_valid_json,
    inspect_python_fragment,
    parse_tool_args,
    safe_as_int,
    safe_normalize_skill_key,
    sanitize_skill_code,
)
from lox.author.prompts import build_system_prompt, build_user_prompt
from lox.author.tools import (
    OPENAI_TOOL_SPECS,
    AuthorToolRegistry,
    get_tool_specs,
    merge_skill_module_update,
)
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH
from lox.lang.validator import (
    is_ast_equivalent,
    normalize_mode,
    skill_key_to_relpath,
    splice_functions_into_policy,
    validate_python_code,
)
from lox.ui.console import ui

_LOG = logging.getLogger(__name__)

load_dotenv()


@dataclass(slots=True)
class SynthesisProposal:
    """Validated output from an Author Agent synthesis session."""

    mode: str
    hypothesis: dict[str, Any]
    raw_code_patch: str
    full_policy_code: str
    full_macros_code: str
    target_functions: list[str]
    tool_calls_made: int = 0
    skill_code: str = ""
    skill_modules: dict[str, str] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""
    pilot_best_delta: float = 0.0


class AuthorAgent:
    """
    Stateless Option-A Swarm Peer Author Agent.
    Starts a fresh conversation each generation with a static cached System Prompt,
    executes tool calls (`query_db`, `query_wiki`, `read_skill`, `dry_run_policy`, etc.),
    and runs a compiler-in-the-loop gate to ensure emitted code compiles and runs cleanly.
    """

    _sanitize_skill_code = staticmethod(sanitize_skill_code)
    _parse_tool_args = staticmethod(parse_tool_args)
    extract_yaml_and_python = staticmethod(extract_yaml_and_python)
    extract_skill_code = staticmethod(extract_skill_code)
    extract_all_skill_codes = staticmethod(extract_all_skill_codes)
    _inspect_python_fragment = staticmethod(inspect_python_fragment)
    _has_executable_skill_input = staticmethod(has_executable_skill_input)
    _compact_tool_messages = staticmethod(compact_tool_messages)
    _has_valid_json = staticmethod(has_valid_json)

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        api_key: str | None = None,
        base_url: str | None = None,
        db_path: str = DEFAULT_DB_PATH,
        max_tokens: int = 65536,
        reasoning_effort: str = "medium",
        routing: str = "nitro",
        max_repair_turns: int = 5,
        agent_role: str = "author",
    ) -> None:
        self.provider = str(provider or "openrouter").lower()
        self.model = model
        self.agent_role = str(agent_role)
        self.api_key = resolve_api_key(self.provider, explicit_key=api_key)
        self.base_url = base_url
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.routing = routing
        self.max_repair_turns = max_repair_turns
        self.latest_attempted_hyp: dict[str, Any] = {}
        self.tools = AuthorToolRegistry(db_path=db_path)
        self.db_logger = BatchDatabaseLogger(db_path=db_path)

    def build_repair_prompt(
        self,
        status: str,
        diag: str,
        mode: str,
        repair_idx: int,
        max_repair_turns: int,
        candidate_code: str = "",
    ) -> str:
        """Builds a precise, actionable repair prompt for the Author Agent."""
        _ = candidate_code
        canon_mode = normalize_mode(mode)
        hdr = (
            f"Your candidate code failed the compiler/sandbox verification gate "
            f"(Attempt {repair_idx}/{max_repair_turns}):"
        )
        if canon_mode == "SWARM_PEER":
            return (
                f"{hdr}\n\n"
                f"```\n{diag.strip()}\n```\n\n"
                "PROBLEM: SWARM PEER PATCH FAILED VERIFICATION\n"
                "REQUIRED FIX:\n"
                "1. Emit surgical updates ONLY for touched functions/skills using "
                '`<skill_code module="domain/submodule">...</skill_code>` and/or `<policy_code>...</policy_code>` '
                "(do NOT paste unchanged functions).\n"
                "2. Check undefined symbols, imports (`from lox.env.actions import ...`, `from skills.<m> import ...`), "
                "and call arities reported above.\n"
                "3. Keep `Policy.step(state)` thin using `self.ws` and `self.goals.tick(state, self.ws)` (Option-A)."
            )
        if "COMPILE" in status or "SyntaxError" in diag or "IndentationError" in diag:
            guidance = (
                "PROBLEM: PYTHON SYNTAX OR INDENTATION ERROR\n"
                "Fix all syntax/indentation errors using only permitted modules and builtins, "
                "then output the corrected `<skill_code>` / `<policy_code>` blocks."
            )
        elif "MISSING_CODE" in status:
            guidance = (
                "PROBLEM: NO PYTHON CODE BLOCK FOUND\n"
                'Emit your YAML hypothesis and `<skill_code module="...">` / `<policy_code>` Python implementation.'
            )
        else:
            guidance = (
                "PROBLEM: VERIFICATION FAILURE\n"
                "Review the error diagnostic above, fix the issue, and emit the corrected `<skill_code>` / `<policy_code>` blocks."
            )
        return f"{hdr}\n\n```\n{diag.strip()}\n```\n\n{guidance}"

    @staticmethod
    def _build_post_tool_budget_notice(
        *,
        turn_idx: int,
        effective_max_turns: int,
        turns_since_last_dry_run: int,
        last_dry_run_passed: bool,
        last_peer_micro_pilot: dict[str, Any] | None,
    ) -> str | None:
        """
        Detects both pre-pilot research stalls AND post-pilot SQL death-balls when
        smoke checks passed (`last_dry_run_passed=True`) yet the micro-pilot remained unresolved.
        """
        mp = last_peer_micro_pilot if isinstance(last_peer_micro_pilot, dict) else None
        if mp is None and not last_dry_run_passed:
            if turn_idx >= max(
                4, min(effective_max_turns - 3, effective_max_turns // 2)
            ):
                return (
                    f"⚠️ Budget Notice: You are on Turn {turn_idx}/{effective_max_turns}. "
                    "You have completed empirical research. Please synthesize your candidate code now and call `dry_run_policy(code=...)` "
                    "to verify your hypothesis before tool turns run out."
                )
            return None

        if mp is not None:
            mp_div = int(mp.get("diverged", 0) or 0)
            mp_exc = int(mp.get("policy_exceptions", 0) or 0)
            mp_greg = int(mp.get("guard_regressions", 0) or 0)
            mp_w = int(mp.get("wins", 0) or 0)
            mp_l = int(mp.get("losses", 0) or 0)
            mp_dd = float(mp.get("avg_depth_delta", 0.0) or 0.0)
            has_issue = (
                mp_div == 0
                or mp_exc > 0
                or mp_greg > 0
                or (mp_dd <= -0.25 and mp_w <= mp_l)
                or (mp_l > mp_w and mp_dd < 0.0)
            )
            if has_issue:
                if turns_since_last_dry_run >= 3 or (
                    turn_idx >= effective_max_turns - 3
                    and turns_since_last_dry_run >= 1
                ):
                    reasons: list[str] = []
                    if mp_div == 0:
                        reasons.append("0-byte divergence")
                    if mp_exc > 0:
                        reasons.append(f"exc={mp_exc}")
                    if mp_greg > 0:
                        reasons.append(f"guard_reg={mp_greg}")
                    if mp_dd < 0.0 or mp_l > mp_w:
                        reasons.append(f"W/L={mp_w}/{mp_l}, avg_Δdepth={mp_dd:+.2f}")
                    issue_str = ", ".join(reasons) or f"W/L={mp_w}/{mp_l}"
                    return (
                        f"⚠️ Re-Pilot Notice (Turn {turn_idx}/{effective_max_turns}): Your last `dry_run_policy` micro-pilot is still "
                        f"unresolved ({issue_str}). Do NOT spend remaining turns paginating sequential `OFFSET` tick diffs — "
                        "query backward from outcomes (`ORDER BY step_idx DESC LIMIT 20`), fix your draft, and re-run `dry_run_policy(code=...)` "
                        "before turns run out."
                    )
                return None

        if turns_since_last_dry_run >= 2 and (
            turn_idx >= effective_max_turns - 4 or turns_since_last_dry_run >= 3
        ):
            return (
                f"ℹ️ Budget Notice (Turn {turn_idx}/{effective_max_turns}): Your latest `dry_run_policy` already passed cleanly. "
                "Emit your final YAML hypothesis and `<skill_code>`/`<policy_code>` now unless you are testing a concrete edit."
            )
        return None

    def _call_llm_api(
        self,
        messages: list[dict[str, Any]],
        allow_tools: bool = True,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Dispatches a single chat completion request via `llm_client.call_llm_api`."""
        active_tools = OPENAI_TOOL_SPECS if tool_specs is None else tool_specs
        return call_llm_api(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            routing=self.routing,
            agent_role=self.agent_role,
            messages=messages,
            allow_tools=allow_tools,
            tool_specs=active_tools,
        )

    def _invoke_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Signature-aware wrapper around `self._call_llm_api` (supports monkeypatched test doubles)."""
        try:
            params = inspect.signature(self._call_llm_api).parameters
        except (TypeError, ValueError):
            params = {}
        if tool_specs is not None and "tool_specs" in params:
            return self._call_llm_api(
                messages, allow_tools=allow_tools, tool_specs=tool_specs
            )
        return self._call_llm_api(messages, allow_tools=allow_tools)

    def _execute_tool_call(
        self,
        turn_idx: int,
        name: str,
        args: dict[str, Any],
        assistant_text: str = "",
        latest_hyp: dict[str, Any] | None = None,
    ) -> str:
        t0 = time.perf_counter()
        if name == "query_db":
            sql = str(args.get("sql", ""))
            out, summary = self.tools.query_db(sql)
            arg_disp = sql
        elif name == "query_wiki":
            q = str(args.get("query", ""))
            tk = safe_as_int(args.get("top_k"), 3)
            out, summary = self.tools.query_wiki(q, top_k=tk)
            arg_disp = q
        elif name == "query_monster":
            m_name = str(args.get("name", ""))
            out, summary = self.tools.query_monster(m_name)
            arg_disp = m_name
        elif name == "query_item":
            i_name = str(args.get("name", ""))
            out, summary = self.tools.query_item(i_name)
            arg_disp = i_name
        elif name == "query_policy_docs":
            topic = str(args.get("topic", "all"))
            out, summary = self.tools.query_policy_docs(topic)
            arg_disp = topic
        elif name == "read_skill":
            skill_name = str(args.get("skill_name", ""))
            policy_id = str(args.get("policy_id", ""))
            start_line = safe_as_int(args.get("start_line"), 1)
            raw_end = args.get("end_line")
            end_line = safe_as_int(raw_end, 0) if raw_end is not None else None
            if end_line == 0 and raw_end is not None:
                end_line = None
            out, summary = self.tools.read_skill(
                skill_name,
                policy_id=policy_id,
                start_line=start_line,
                end_line=end_line,
            )
            arg_disp = f"{skill_name}@{policy_id}" if policy_id else skill_name
        elif name == "read_symbol":
            q_name = str(args.get("qualified_name", ""))
            out, summary = self.tools.read_symbol(q_name)
            arg_disp = q_name[:40]
        elif name == "read_policy":
            policy_id = str(args.get("policy_id", ""))
            out, summary = self.tools.read_policy(policy_id)
            arg_disp = policy_id
        elif name == "diff_policies":
            id_a = str(args.get("policy_id_a", ""))
            id_b = str(args.get("policy_id_b", ""))
            out, summary = self.tools.diff_policies(id_a, id_b)
            arg_disp = f"{id_a}..{id_b}"
        elif name == "query_theories":
            q = str(args.get("query", ""))
            dt = str(args.get("doc_type", ""))
            lim = safe_as_int(args.get("limit"), 6)
            out, summary = self.tools.query_theories(query=q, doc_type=dt, limit=lim)
            arg_disp = q[:40]
        elif name == "read_wiki_article":
            title = str(args.get("title", ""))
            sec = str(args.get("section", ""))
            off = safe_as_int(args.get("offset"), 0)
            mc = safe_as_int(args.get("max_chars"), 0)
            out, summary = self.tools.read_wiki_article(
                title=title, section=sec, offset=off, max_chars=mc
            )
            arg_disp = f"{title[:30]}@{sec}" if sec else title[:35]
        elif name == "find_references":
            sym = str(args.get("symbol", ""))
            out, summary = self.tools.find_references(sym)
            arg_disp = sym[:35]
        elif name == "grep_skills":
            pat = str(args.get("pattern", ""))
            out, summary = self.tools.grep_skills(pat)
            arg_disp = pat[:35]
        elif name == "architecture_map":
            out, summary = self.tools.architecture_map()
            arg_disp = "architecture"
        elif name == "benchmark_expert_parity":
            tn = safe_as_int(args.get("top_n"), 10)
            out, summary = self.tools.benchmark_expert_parity(top_n=tn)
            arg_disp = f"top_n={tn}"
        elif name == "dry_run_policy":
            code = str(args.get("code", "")).strip()
            s_name = args.get("skill_name")
            s_code = args.get("skill_code")
            merged_s_map: dict[str, str] = {}
            if assistant_text:
                for k, v in (
                    self.extract_all_skill_codes(assistant_text, latest_hyp) or {}
                ).items():
                    clean_k = safe_normalize_skill_key(k)
                    if clean_k and str(v or "").strip():
                        merged_s_map[clean_k] = str(v)
            if isinstance(args.get("skills"), dict):
                for k, v in args["skills"].items():
                    clean_k = safe_normalize_skill_key(k)
                    if clean_k and str(v or "").strip():
                        merged_s_map[clean_k] = str(v)
            s_map = merged_s_map or None
            if (
                not code
                and not (s_name and s_code)
                and not self._has_executable_skill_input(s_map)
            ):
                out = "Error: `dry_run_policy` requires non-empty `code`, `(skill_name, skill_code)`, or `skills`."
                summary = "MISSING_CODE"
                arg_disp = "empty code"
            else:
                seed = safe_as_int(args.get("seed"), 42)
                mt = safe_as_int(args.get("max_turns"), 150)
                replay_id = str(args.get("replay_run_id", ""))
                replay_steps = safe_as_int(args.get("replay_steps"), 0)
                smoke_only = bool(args.get("smoke_only", False))
                out, summary = self.tools.dry_run_policy(
                    code=code,
                    seed=seed,
                    max_turns=mt,
                    replay_run_id=replay_id,
                    replay_steps=replay_steps,
                    smoke_only=smoke_only,
                    skill_name=str(s_name) if s_name else None,
                    skill_code=str(s_code) if s_code else None,
                    skills=s_map,
                )
                arg_disp = (
                    f"{len(code.splitlines())} lines, seed={seed}"
                    if code
                    else f"patch/skills-only, seed={seed}"
                )
                if smoke_only:
                    arg_disp += ", smoke_only"
                if s_name:
                    arg_disp += f", skill={s_name}"
                elif s_map:
                    arg_disp += f", skills={len(s_map)}"
                if replay_id:
                    arg_disp += f", replay={replay_id[:10]}..."
        else:
            out, summary = f"Unknown tool '{name}'.", "unknown"
            arg_disp = str(args)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ui.tool_call(
            turn_idx,
            name,
            arg_disp,
            summary,
            elapsed_ms,
            agent_role=self.agent_role,
        )
        return out

    def _synthesize_mock(
        self,
        mode: str,
        current_policy: str,
        skills: dict[str, str] | None = None,
    ) -> SynthesisProposal:
        """Deterministic mock synthesizer for automated tests (`provider='mock'`)."""
        self.tools.current_policy_code = current_policy
        self._execute_tool_call(
            1, "query_db", {"sql": "SELECT COUNT(*) AS n FROM expert_runs"}
        )
        self._execute_tool_call(1, "query_wiki", {"query": "Excalibur", "top_k": 1})

        if not current_policy.strip():
            from lox.lang.starter import build_clean_option_a_starter

            code, starter_skills = build_clean_option_a_starter()
            hyp = {
                "mode": mode,
                "target_functions": ["step"],
                "observation_from_db": "Initial baseline synthesis from expert_runs and wiki.",
                "mechanism": "Builds Option-A modular Policy with GoalRunner and WorldState.",
            }
            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_mock_{int(time.time() * 1000)}",
                    generation=0,
                    batch_id="batch_mock",
                    mode=mode,
                    turn_index=1,
                    turn_type="MOCK_SYNTHESIS",
                    repair_idx=0,
                    model="mock",
                    thinking_content="Mock reasoning step for testing telemetry persistence.",
                    message_content="Mock proposal content",
                    tool_calls_json="[]",
                    tool_results_json="[]",
                    candidate_code=code,
                    skill_name="",
                    skill_code="",
                    hypothesis_yaml=yaml.dump(hyp),
                    status="PASSED",
                    error_diagnostic="",
                    prompt_sent="Mock prompt",
                    prompt_tokens=100,
                    completion_tokens=200,
                    reasoning_tokens=50,
                    agent_role=self.agent_role,
                    wall_ms=1.5,
                )
            except Exception as log_exc:
                _LOG.warning("save_synthesis_log failed: %s", log_exc)
            return SynthesisProposal(
                mode=mode,
                hypothesis=hyp,
                raw_code_patch=code,
                full_policy_code=code,
                full_macros_code="",
                target_functions=["step"],
                tool_calls_made=2,
                skill_modules=starter_skills,
            )

        hyp = {
            "mode": mode,
            "target_functions": ["step"],
            "observation_from_db": "Mock incremental synthesis.",
            "mechanism": "Preserves current Option-A Policy cleanly.",
        }
        return SynthesisProposal(
            mode=mode,
            hypothesis=hyp,
            raw_code_patch=current_policy,
            full_policy_code=current_policy,
            full_macros_code="",
            target_functions=["step"],
            tool_calls_made=2,
            skill_modules=dict(skills or {}),
        )

    def synthesize(
        self,
        mode: str = "SWARM_PEER",
        generation: int = 0,
        current_policy: str = "",
        batch_digest: str = "",
        recent_experiments: str = "",
        max_tool_turns: int | None = None,
        pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        skills: dict[str, str] | None = None,
        recent_discoveries: list[dict[str, Any]] | None = None,
        recent_failures: list[dict[str, Any]] | None = None,
        max_steps: int = 5000,
        stagnation_function: str = "",
    ) -> SynthesisProposal:
        """
        Runs a fresh, stateless multi-turn tool-calling session for `generation` and `mode`
        with compiler-in-the-loop verification and sequential self-repair.
        """
        if self.provider == "mock":
            return self._synthesize_mock(
                mode=mode, current_policy=current_policy, skills=skills
            )

        canon_mode = normalize_mode(mode)
        base_skills_snapshot = dict(skills or {})
        self.tools.current_policy_code = current_policy
        self.tools.last_passed_code = ""
        self.tools.skills = dict(base_skills_snapshot)
        self.tools.max_steps = max_steps

        # Preserve peer context already configured on `self.tools` by `SwarmPeerAgent.run_cycle`
        prev_swarm_frontier = getattr(self.tools, "swarm_frontier", None)
        prev_swarm_checkpoint = getattr(self.tools, "swarm_checkpoint", None)
        prev_claimed_deficit = getattr(self.tools, "claimed_deficit", None)
        prev_scenario_gate = getattr(self.tools, "scenario_gate", "off")
        prev_peer_research_gate = getattr(
            self.tools, "enforce_peer_research_gate", False
        )
        prev_peer_pool = getattr(self.tools, "pool", None)
        prev_peer_baselines = list(getattr(self.tools, "baseline_runs", []) or [])
        prev_peer_guard_seeds = list(getattr(self.tools, "peer_guard_seeds", []) or [])
        prev_peer_target_seeds = list(
            getattr(self.tools, "peer_target_seeds", []) or []
        )
        prev_peer_probe_seeds = list(getattr(self.tools, "peer_probe_seeds", []) or [])
        prev_peer_id = str(getattr(self.tools, "peer_id", "") or "")
        prev_base_pol = str(getattr(self.tools, "baseline_policy_code", "") or "")
        raw_base_skills = getattr(self.tools, "baseline_skills", None)
        prev_base_skills: dict[str, str] | None = (
            dict(raw_base_skills) if isinstance(raw_base_skills, dict) else None
        )
        prev_max_peer_micro_pilots = int(
            getattr(self.tools, "max_peer_micro_pilots", 4)
        )

        self.tools.configure_generation(
            pool=pool if pool is not None else prev_peer_pool,
            baseline_runs=(
                baseline_runs if baseline_runs is not None else prev_peer_baselines
            ),
            skills=base_skills_snapshot,
            generation=generation,
            mode=canon_mode,
            swarm_frontier=prev_swarm_frontier,
            swarm_checkpoint=prev_swarm_checkpoint,
            claimed_deficit=prev_claimed_deficit,
            scenario_gate=prev_scenario_gate,
            peer_guard_seeds=prev_peer_guard_seeds,
            peer_target_seeds=prev_peer_target_seeds,
            peer_probe_seeds=prev_peer_probe_seeds,
            max_peer_micro_pilots=prev_max_peer_micro_pilots,
            peer_id=prev_peer_id,
        )
        self.tools.enforce_peer_research_gate = prev_peer_research_gate
        if prev_base_pol:
            self.tools.baseline_policy_code = prev_base_pol
        if prev_base_skills is not None:
            self.tools.baseline_skills = prev_base_skills

        if max_tool_turns is None:
            max_tool_turns = 16

        sys_prompt = build_system_prompt(self.db_path, is_swarm=True, is_peer=True)
        user_prompt = build_user_prompt(
            mode=mode,
            generation=generation,
            current_policy=current_policy,
            batch_digest=batch_digest,
            recent_experiments=recent_experiments,
            skills=base_skills_snapshot,
            recent_discoveries=recent_discoveries,
            recent_failures=recent_failures,
            stagnation_function=stagnation_function,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_tool_calls = 0
        research_tools_called: set[str] = set()
        called_expert_query = False
        gen1_research_nudges = 0
        bonus_research_turns = 0
        turns_since_last_dry_run = 0
        turn_idx = 0
        final_content = ""
        latest_hyp: dict[str, Any] = {}
        batch_id = f"gen_{generation:03d}_{mode.lower()}"
        active_tool_specs = get_tool_specs(canon_mode)

        while turn_idx < max_tool_turns + bonus_research_turns + 1:
            turn_idx += 1
            effective_max_turns = max_tool_turns + bonus_research_turns
            allow_tools = turn_idx <= effective_max_turns
            if not allow_tools and messages and messages[-1].get("role") == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool turn budget reached. Do NOT call any more tools or emit `<tool_call>` tags—"
                            "output your final YAML hypothesis and `<skill_code>` / `<policy_code>` patches now."
                        ),
                    }
                )
            if self.agent_role in ("", "author"):
                ui.status(
                    f"Author Agent querying {self.model} ({'initial reasoning & tool generation' if turn_idx == 1 else f'tool synthesis turn {turn_idx}'})..."
                )
            t0_turn = time.perf_counter()
            llm_messages = self._compact_tool_messages(messages, keep_recent_turns=2)
            msg = self._invoke_llm(
                llm_messages, allow_tools=allow_tools, tool_specs=active_tool_specs
            )
            turn_ms = (time.perf_counter() - t0_turn) * 1000.0
            msg_text = msg.get("content") or ""
            reasoning_text = msg.get("_reasoning") or ""
            prompt_toks, comp_toks, reasoning_toks = extract_usage_tokens(
                msg.get("_usage")
            )
            turn_cost_usd = extract_usage_cost(msg.get("_usage"))

            clean_text = re.sub(
                r"<think>.*?</think>", "", msg_text, flags=re.DOTALL
            ).strip()
            clean_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
            if msg.get("tool_calls"):
                clean_msg["tool_calls"] = msg["tool_calls"]
            messages.append(clean_msg)
            if msg_text:
                inter_hyp, _ = self.extract_yaml_and_python(msg_text)
                if inter_hyp:
                    latest_hyp = inter_hyp
                    self.latest_attempted_hyp = latest_hyp

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                _, found_code = self.extract_yaml_and_python(msg_text)
                found_skills = self.extract_all_skill_codes(
                    clean_text, latest_hyp or {}
                )
                if (
                    found_code
                    and canon_mode == "SWARM_PEER"
                    and self.tools.last_passed_code
                ):
                    has_stmts, has_cls = self._inspect_python_fragment(found_code)
                    has_policy_tag = bool(
                        re.search(
                            r"<(?:policy_code|policy)>",
                            msg_text or "",
                            flags=re.IGNORECASE,
                        )
                    )
                    if not has_stmts or (not has_cls and not has_policy_tag):
                        found_code = ""

                if (
                    generation == 1
                    and self.tools._campaign_tables_empty()
                    and turn_idx <= effective_max_turns
                    and not str(batch_digest).startswith("Self-Repair Required:")
                    and (
                        "query_wiki" not in research_tools_called
                        or not called_expert_query
                    )
                    and gen1_research_nudges < 2
                ):
                    gen1_research_nudges += 1
                    missing_g1 = []
                    if "query_wiki" not in research_tools_called:
                        missing_g1.append("`query_wiki`")
                    if not called_expert_query:
                        missing_g1.append(
                            "`query_db` on `expert_runs` / `expert_ticks`"
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"🛑 GENERATION 1 MANDATORY RESEARCH: You have not yet called {' and '.join(missing_g1)}.\n"
                                "In Generation 1, campaign tables (`runs`, `batches`, etc.) are disallowed; "
                                "you MUST call both `query_wiki` and `query_db` (querying `expert_runs` / `expert_ticks`) "
                                "before outputting final policy code."
                            ),
                        }
                    )
                    continue

                # Auto-dry-run if candidate code/skills were emitted in text without calling dry_run_policy
                has_new_payload = bool(found_code or found_skills)
                is_already_verified = bool(
                    self.tools.last_dry_run_passed
                    and self.tools.last_passed_code
                    and (
                        not found_code
                        or is_ast_equivalent(found_code, self.tools.last_passed_code)
                    )
                    and (
                        not found_skills
                        or all(
                            k in self.tools.last_passed_skills
                            and is_ast_equivalent(v, self.tools.last_passed_skills[k])
                            for k, v in found_skills.items()
                        )
                    )
                )
                if (
                    has_new_payload
                    and not is_already_verified
                    and turn_idx < effective_max_turns
                ):
                    total_tool_calls += 1
                    t0 = time.perf_counter()
                    auto_s_name = str(
                        (latest_hyp or {}).get("skill_module", "")
                    ) or getattr(self.tools, "last_passed_skill_name", None)
                    auto_s_code = self.extract_skill_code(
                        clean_text,
                        latest_hyp or {},
                        fallback_skill_code=(
                            getattr(self.tools, "last_passed_skill_code", None) or ""
                        ),
                        fallback_skill_name=auto_s_name,
                    )
                    auto_skills_map = {
                        **(
                            getattr(self.tools, "last_passed_skills", {})
                            or base_skills_snapshot
                        ),
                        **(found_skills or {}),
                    }
                    turns_since_last_dry_run = 0
                    dry_res, dry_summary = self.tools.dry_run_policy(
                        code=found_code,
                        skill_name=auto_s_name if auto_s_name else None,
                        skill_code=auto_s_code if auto_s_code else None,
                        skills=auto_skills_map or None,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    arg_disp = (
                        f"auto-dry-run ({len(found_code.splitlines())} lines)"
                        if found_code
                        else f"auto-dry-run ({len(found_skills)} skill(s))"
                    )
                    ui.tool_call(
                        turn_idx,
                        "dry_run_policy",
                        arg_disp,
                        dry_summary,
                        elapsed_ms,
                        agent_role=self.agent_role,
                    )
                    auto_dry_passed = bool(
                        getattr(self.tools, "last_dry_run_passed", False)
                    )
                    try:
                        auto_status = (
                            "DRY_RUN_PASSED"
                            if auto_dry_passed
                            else (
                                "COMPILE_ERROR"
                                if "COMPILE_ERROR" in dry_summary
                                else (
                                    "RUNTIME_ERROR"
                                    if "RUNTIME_ERROR" in dry_summary
                                    else "DRY_RUN_FAILED"
                                )
                            )
                        )
                        self.db_logger.save_synthesis_log(
                            log_id=f"syn_{self.agent_role}_{generation}_{turn_idx}_autodry_{uuid.uuid4().hex[:6]}",
                            generation=generation,
                            batch_id=batch_id,
                            mode=mode,
                            turn_index=turn_idx,
                            turn_type="DRY_RUN_CHECK",
                            repair_idx=0,
                            model=self.model,
                            thinking_content=reasoning_text,
                            message_content=clean_text,
                            tool_calls_json="[]",
                            tool_results_json=json.dumps(
                                [{"dry_run_summary": dry_summary}]
                            ),
                            candidate_code=found_code,
                            skill_name=str(auto_s_name or ""),
                            skill_code=str(auto_s_code or ""),
                            hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                            status=auto_status,
                            error_diagnostic=dry_res[:4000],
                            prompt_sent="",
                            prompt_tokens=prompt_toks,
                            completion_tokens=comp_toks,
                            reasoning_tokens=reasoning_toks,
                            agent_role=self.agent_role,
                            cost_usd=turn_cost_usd,
                            wall_ms=turn_ms,
                        )
                    except Exception as log_exc:
                        _LOG.warning("save_synthesis_log failed: %s", log_exc)

                    # Fix C-6: Only nudge when dry-run actually failed; break cleanly when "check" mode passed!
                    if not auto_dry_passed:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your candidate code failed automatic `dry_run_policy` ({dry_summary}):\n"
                                    f"{dry_res}\n"
                                    "Please fix the issue and verify with `dry_run_policy`."
                                ),
                            }
                        )
                        continue
                    final_content = msg_text
                    break

                if (
                    not has_new_payload
                    and not self.tools.last_passed_code
                    and turn_idx < effective_max_turns
                ):
                    if total_tool_calls == 0 and turn_idx == 1:
                        nudge_msg = (
                            "Please begin your empirical investigation by calling `query_db`, `read_skill`/`read_symbol`, "
                            "or `query_wiki`, then test your candidate patch with `dry_run_policy(code=...)`."
                        )
                    else:
                        nudge_msg = (
                            "Please test your candidate `<skill_code>` / `<policy_code>` with `dry_run_policy(code=...)` "
                            "before finalizing."
                        )
                    messages.append({"role": "user", "content": nudge_msg})
                    continue

                try:
                    _, final_turn_code = self.extract_yaml_and_python(msg_text)
                    self.db_logger.save_synthesis_log(
                        log_id=f"syn_{self.agent_role}_{generation}_{turn_idx}_final_{uuid.uuid4().hex[:6]}",
                        generation=generation,
                        batch_id=batch_id,
                        mode=mode,
                        turn_index=turn_idx,
                        turn_type="FINAL_SYNTHESIS",
                        repair_idx=0,
                        model=self.model,
                        thinking_content=reasoning_text,
                        message_content=clean_text,
                        tool_calls_json="[]",
                        tool_results_json="[]",
                        candidate_code=final_turn_code,
                        skill_name=str((latest_hyp or {}).get("skill_module", "")),
                        skill_code=self.extract_skill_code(clean_text, latest_hyp),
                        hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                        status="FINAL_SUBMITTED",
                        error_diagnostic="",
                        prompt_sent="",
                        prompt_tokens=prompt_toks,
                        completion_tokens=comp_toks,
                        reasoning_tokens=reasoning_toks,
                        agent_role=self.agent_role,
                        cost_usd=turn_cost_usd,
                        wall_ms=turn_ms,
                    )
                except Exception as log_exc:
                    _LOG.warning("save_synthesis_log failed: %s", log_exc)

                final_content = msg_text
                break

            tool_results_list: list[dict[str, Any]] = []
            for tc in tool_calls:
                total_tool_calls += 1
                fn_spec = tc.get("function") or {}
                t_name = str(fn_spec.get("name") or "")
                if t_name in (
                    "query_db",
                    "query_wiki",
                    "read_wiki_article",
                    "query_monster",
                    "query_item",
                    "benchmark_expert_parity",
                    "find_references",
                    "grep_skills",
                    "architecture_map",
                    "read_symbol",
                ):
                    research_tools_called.add(t_name)
                    if t_name == "read_wiki_article":
                        research_tools_called.add("query_wiki")
                    if t_name == "benchmark_expert_parity":
                        called_expert_query = True
                raw_args = fn_spec.get("arguments", "{}")
                t_args, arg_err = self._parse_tool_args(raw_args)
                if arg_err:
                    tool_out = arg_err
                else:
                    tool_out = self._execute_tool_call(
                        turn_idx,
                        t_name,
                        t_args,
                        assistant_text=msg_text,
                        latest_hyp=latest_hyp,
                    )
                if t_name == "query_db":
                    sql_low = str((t_args or {}).get("sql", "")).lower()
                    if "expert_runs" in sql_low or "expert_ticks" in sql_low:
                        called_expert_query = True
                if (
                    t_name == "dry_run_policy"
                    and isinstance(tool_out, str)
                    and tool_out.startswith("RESEARCH_REQUIRED:")
                    and bonus_research_turns < 3
                ):
                    bonus_research_turns += 1
                tool_results_list.append(
                    {"name": t_name, "args": t_args, "output": str(tool_out)[:2000]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                        "name": t_name,
                        "content": tool_out,
                    }
                )

            has_dry = any(
                (tc.get("function") or {}).get("name") == "dry_run_policy"
                for tc in tool_calls
            )
            if has_dry:
                turns_since_last_dry_run = 0
            else:
                turns_since_last_dry_run += 1

            if not has_dry and turn_idx < effective_max_turns:
                budget_notice = self._build_post_tool_budget_notice(
                    turn_idx=turn_idx,
                    effective_max_turns=effective_max_turns,
                    turns_since_last_dry_run=turns_since_last_dry_run,
                    last_dry_run_passed=bool(
                        getattr(self.tools, "last_dry_run_passed", False)
                    ),
                    last_peer_micro_pilot=getattr(
                        self.tools, "last_peer_micro_pilot", None
                    ),
                )
                if budget_notice:
                    messages.append({"role": "user", "content": budget_notice})

            try:
                if has_dry:
                    t_type = "DRY_RUN_CHECK"
                    t_stat = (
                        "DRY_RUN_PASSED"
                        if getattr(self.tools, "last_dry_run_passed", False)
                        else "DRY_RUN_FAILED"
                    )
                else:
                    t_type = "RESEARCH_TOOL"
                    t_stat = "TOOL_CALLS"
                _, cand_code_turn = self.extract_yaml_and_python(msg_text)
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_{self.agent_role}_{generation}_{turn_idx}_0_{uuid.uuid4().hex[:6]}",
                    generation=generation,
                    batch_id=batch_id,
                    mode=mode,
                    turn_index=turn_idx,
                    turn_type=t_type,
                    repair_idx=0,
                    model=self.model,
                    thinking_content=reasoning_text,
                    message_content=clean_text,
                    tool_calls_json=json.dumps(tool_calls),
                    tool_results_json=json.dumps(tool_results_list),
                    candidate_code=cand_code_turn,
                    skill_name=str((latest_hyp or {}).get("skill_module", "")),
                    skill_code=self.extract_skill_code(clean_text, latest_hyp),
                    hypothesis_yaml=yaml.dump(latest_hyp) if latest_hyp else "",
                    status=t_stat,
                    error_diagnostic=getattr(self.tools, "last_dry_run_diag", ""),
                    prompt_sent=str(messages[-len(tool_calls) - 2].get("content", ""))
                    if len(messages) >= len(tool_calls) + 2
                    else "",
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    reasoning_tokens=reasoning_toks,
                    agent_role=self.agent_role,
                    cost_usd=turn_cost_usd,
                    wall_ms=turn_ms,
                )
            except Exception as log_exc:
                _LOG.warning("save_synthesis_log failed: %s", log_exc)

        # Compiler-in-the-loop verification & sequential self-repair
        diag = ""
        merged_skills_map: dict[str, str] = {}
        for repair_idx in range(self.max_repair_turns + 1):
            s_name: str | None = None
            s_code = ""
            hyp, code_patch = self.extract_yaml_and_python(final_content)
            if hyp:
                latest_hyp = hyp
            elif latest_hyp:
                hyp = latest_hyp

            extracted_skills_map = self.extract_all_skill_codes(
                final_content, hyp or latest_hyp
            )
            base_skills = dict(getattr(self.tools, "skills", {}) or {})
            # Fix H-6: In SWARM_PEER, if new <skill_code> blocks were emitted in final_content,
            # seed from clean base_skills so stale edits from an abandoned earlier dry_run don't leak in.
            if extracted_skills_map and canon_mode == "SWARM_PEER":
                merged_skills_map = dict(base_skills)
            else:
                merged_skills_map = {
                    **(getattr(self.tools, "last_passed_skills", {}) or base_skills),
                }
            for s_mod, s_src in (extracted_skills_map or {}).items():
                clean_mod = safe_normalize_skill_key(s_mod)
                if clean_mod and str(s_src or "").strip():
                    existing_mod_src = str(
                        merged_skills_map.get(clean_mod, base_skills.get(clean_mod, ""))
                        or ""
                    )
                    merged_skills_map[clean_mod] = merge_skill_module_update(
                        existing_mod_src, str(s_src)
                    )

            if canon_mode == "SWARM_PEER" and self.tools.last_passed_code:
                has_stmts, has_cls = self._inspect_python_fragment(code_patch)
                has_policy_tag = bool(
                    re.search(
                        r"<(?:policy_code|policy)>",
                        final_content or "",
                        flags=re.IGNORECASE,
                    )
                )
                if (
                    not code_patch
                    or not has_stmts
                    or (not has_cls and not has_policy_tag)
                ):
                    code_patch = self.tools.last_passed_code
            elif not code_patch and self.tools.last_passed_code:
                code_patch = self.tools.last_passed_code

            if not code_patch and not merged_skills_map:
                diag = "No <policy_code>, <skill_code>, or ```python ... ``` code block found in your response. Please emit your YAML hypothesis and Python code block."
                status = "MISSING_CODE"
            else:
                s_name = (hyp or latest_hyp).get("skill_module") or getattr(
                    self.tools, "last_passed_skill_name", None
                )
                s_code = self.extract_skill_code(
                    final_content,
                    hyp or latest_hyp,
                    fallback_skill_code=(
                        getattr(self.tools, "last_passed_skill_code", None) or ""
                    ),
                    fallback_skill_name=s_name,
                )
                norm_s_name = safe_normalize_skill_key(s_name) if s_name else ""
                already_merged = bool(norm_s_name and norm_s_name in merged_skills_map)
                diag, status = self.tools.dry_run_policy(
                    code=code_patch,
                    seed=42,
                    max_turns=100,
                    smoke_only=True,
                    skill_name=None if already_merged else s_name,
                    skill_code="" if already_merged else s_code,
                    skills=merged_skills_map or None,
                )

            try:
                self.db_logger.save_synthesis_log(
                    log_id=f"syn_{self.agent_role}_{generation}_{max_tool_turns + repair_idx}_{repair_idx}_{uuid.uuid4().hex[:6]}",
                    generation=generation,
                    batch_id=batch_id,
                    mode=mode,
                    turn_index=max_tool_turns + repair_idx,
                    turn_type="SMOKE_CHECK" if repair_idx == 0 else "REPAIR_VERIFY",
                    repair_idx=repair_idx,
                    model=self.model,
                    thinking_content="",
                    message_content=final_content[:2000],
                    tool_calls_json="[]",
                    tool_results_json="[]",
                    candidate_code=code_patch,
                    skill_name=str(s_name or ""),
                    skill_code=str(s_code or ""),
                    hypothesis_yaml=yaml.dump(hyp or latest_hyp)
                    if (hyp or latest_hyp)
                    else "",
                    status=status,
                    error_diagnostic=diag,
                    prompt_sent="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    agent_role=self.agent_role,
                    wall_ms=0.0,
                )
            except Exception as log_exc:
                _LOG.warning("save_synthesis_log failed: %s", log_exc)

            if status.startswith("PASSED") or status.startswith("VALID"):
                skill_code = self.extract_skill_code(
                    final_content,
                    hyp,
                    fallback_skill_code=(
                        getattr(self.tools, "last_passed_skill_code", None) or ""
                    ),
                    fallback_skill_name=getattr(
                        self.tools, "last_passed_skill_name", None
                    ),
                )
                # Fix H-2: Never `continue` skipping the repair LLM call on SKILL_SYNTAX_ERROR!
                skill_valid = True
                if skill_code and skill_code.strip():
                    try:
                        raw_skill_k = safe_normalize_skill_key(
                            str(hyp.get("skill_module", "skill") or "skill")
                        )
                        validate_python_code(
                            skill_code,
                            filename=skill_key_to_relpath(raw_skill_k or "skill"),
                        )
                    except Exception as val_err:
                        diag = f"SyntaxError in skill code: {val_err}"
                        status = "SYNTAX_ERROR"
                        skill_valid = False
                        if repair_idx >= self.max_repair_turns:
                            return SynthesisProposal(
                                mode=mode,
                                hypothesis=hyp,
                                raw_code_patch=code_patch,
                                full_policy_code=current_policy,
                                full_macros_code="",
                                target_functions=hyp.get("target_functions")
                                or ["step"],
                                tool_calls_made=total_tool_calls,
                                skill_code="",
                                aborted=True,
                                abort_reason=f"SKILL_SYNTAX_ERROR: {val_err}",
                            )
                if skill_valid:
                    _, patch_has_cls = self._inspect_python_fragment(code_patch)
                    if (
                        canon_mode == "SWARM_PEER"
                        and current_policy.strip()
                        and not patch_has_cls
                    ):
                        if code_patch.strip():
                            spliced_policy, changed_fns = splice_functions_into_policy(
                                current_policy, code_patch
                            )
                        else:
                            spliced_policy = current_policy
                            changed_fns = hyp.get("target_functions") or ["step"]
                    else:
                        spliced_policy = code_patch or current_policy
                        changed_fns = hyp.get("target_functions") or ["step"]
                    if isinstance(changed_fns, str):
                        changed_fns = [changed_fns]
                    return SynthesisProposal(
                        mode=mode,
                        hypothesis=hyp,
                        raw_code_patch=code_patch,
                        full_policy_code=spliced_policy,
                        full_macros_code="",
                        target_functions=list(changed_fns),
                        tool_calls_made=total_tool_calls,
                        skill_code=skill_code,
                        skill_modules=merged_skills_map,
                    )

            if repair_idx < self.max_repair_turns:
                ui.warn(
                    f"Sandbox compiler gate caught {status}; requesting in-turn repair ({repair_idx + 1}/{self.max_repair_turns})..."
                )
                repair_prompt = self.build_repair_prompt(
                    status=status,
                    diag=diag,
                    mode=mode,
                    repair_idx=repair_idx + 1,
                    max_repair_turns=self.max_repair_turns,
                    candidate_code=code_patch,
                )

                if not messages or messages[-1].get("role") != "assistant":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": final_content
                            or (
                                f"```python\n{code_patch}\n```"
                                if code_patch
                                else "I will implement the policy changes."
                            ),
                        }
                    )
                messages.append({"role": "user", "content": repair_prompt})

                t0_rep = time.perf_counter()
                rep_llm_messages = self._compact_tool_messages(
                    messages, keep_recent_turns=2
                )
                repair_msg = self._invoke_llm(
                    rep_llm_messages, allow_tools=True, tool_specs=active_tool_specs
                )
                rep_p, rep_c, rep_r = extract_usage_tokens(repair_msg.get("_usage"))
                rep_cost_usd = extract_usage_cost(repair_msg.get("_usage"))
                if repair_msg.get("tool_calls"):
                    tcs = repair_msg.get("tool_calls") or []
                    messages.append(
                        {
                            "role": "assistant",
                            "content": repair_msg.get("content") or "",
                            "tool_calls": tcs,
                        }
                    )
                    for idx_tc, tc in enumerate(tcs, start=1):
                        tc_fn_name = str((tc.get("function") or {}).get("name") or "")
                        res_out = self._dispatch_raw_tool_call(
                            max_tool_turns + repair_idx + 1, tc
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", f"rep_tc_{idx_tc}"),
                                "name": tc_fn_name,
                                "content": res_out,
                            }
                        )
                    rep_llm_messages2 = self._compact_tool_messages(
                        messages, keep_recent_turns=2
                    )
                    repair_msg = self._invoke_llm(
                        rep_llm_messages2,
                        allow_tools=False,
                        tool_specs=active_tool_specs,
                    )
                    rep_p2, rep_c2, rep_r2 = extract_usage_tokens(
                        repair_msg.get("_usage")
                    )
                    rep_cost2 = extract_usage_cost(repair_msg.get("_usage"))
                    rep_p += rep_p2
                    rep_c += rep_c2
                    rep_r += rep_r2
                    if rep_cost2 is not None:
                        rep_cost_usd = (rep_cost_usd or 0.0) + rep_cost2
                rep_ms = (time.perf_counter() - t0_rep) * 1000.0

                rep_text = repair_msg.get("content") or ""
                rep_reasoning = repair_msg.get("_reasoning") or ""
                clean_rep = re.sub(
                    r"<think>.*?</think>", "", rep_text, flags=re.DOTALL
                ).strip()
                messages.append({"role": "assistant", "content": clean_rep})

                r_hyp, r_code = self.extract_yaml_and_python(clean_rep)
                if r_hyp:
                    latest_hyp = r_hyp
                if r_code:
                    code_patch = r_code
                    final_content = clean_rep
                elif clean_rep:
                    final_content = clean_rep

                try:
                    self.db_logger.save_synthesis_log(
                        log_id=f"syn_{self.agent_role}_{generation}_{max_tool_turns + repair_idx + 1}_rep_{repair_idx + 1}_{uuid.uuid4().hex[:6]}",
                        generation=generation,
                        batch_id=batch_id,
                        mode=mode,
                        turn_index=max_tool_turns + repair_idx + 1,
                        turn_type="REPAIR_TURN",
                        repair_idx=repair_idx + 1,
                        model=self.model,
                        thinking_content=rep_reasoning,
                        message_content=clean_rep,
                        tool_calls_json="[]",
                        tool_results_json="[]",
                        candidate_code=r_code or code_patch,
                        skill_name=str((r_hyp or latest_hyp).get("skill_module", "")),
                        skill_code=self.extract_skill_code(
                            clean_rep, r_hyp or latest_hyp
                        ),
                        hypothesis_yaml=yaml.dump(r_hyp or latest_hyp)
                        if (r_hyp or latest_hyp)
                        else "",
                        status="REPAIR_SUBMITTED",
                        error_diagnostic="",
                        prompt_sent=repair_prompt,
                        prompt_tokens=rep_p,
                        completion_tokens=rep_c,
                        reasoning_tokens=rep_r,
                        agent_role=self.agent_role,
                        cost_usd=rep_cost_usd,
                        wall_ms=rep_ms,
                    )
                except Exception as log_exc:
                    _LOG.warning("save_synthesis_log failed: %s", log_exc)

        # Fix H-6: Pair fallback `last_passed_code` strictly with its matching `skills_snapshot`!
        best_code = self.tools.last_passed_code
        if best_code:
            ui.info(
                "Using verified code + matching skills snapshot from earlier passing dry_run_policy call."
            )
            matched_skills = dict(
                self.tools.last_passed_pkg.skills_snapshot
                or self.tools.last_passed_skills
                or merged_skills_map
            )
            _, patch_has_cls = self._inspect_python_fragment(best_code)
            if current_policy.strip() and not patch_has_cls:
                spliced_policy, changed_fns = splice_functions_into_policy(
                    current_policy, best_code
                )
            else:
                spliced_policy = best_code
                changed_fns = latest_hyp.get("target_functions") or ["step"]
            if isinstance(changed_fns, str):
                changed_fns = [changed_fns]
            return SynthesisProposal(
                mode=mode,
                hypothesis=latest_hyp,
                raw_code_patch=best_code,
                full_policy_code=spliced_policy,
                full_macros_code="",
                target_functions=list(changed_fns),
                tool_calls_made=total_tool_calls,
                skill_code=str(self.tools.last_passed_pkg.skill_code or ""),
                skill_modules=matched_skills,
            )

        raise RuntimeError(
            f"Author Agent failed to produce valid compiling code after {self.max_repair_turns} repair attempts. Last error: {diag}"
        )

    def _dispatch_raw_tool_call(self, turn_idx: int, tc: dict[str, Any]) -> str:
        """Parses an OpenAI-format tool call dict and executes it via `_execute_tool_call`."""
        fn_spec = tc.get("function") or {}
        t_name = str(fn_spec.get("name") or "")
        t_args, arg_err = self._parse_tool_args(fn_spec.get("arguments", "{}"))
        if arg_err:
            return arg_err
        return self._execute_tool_call(turn_idx, t_name, t_args)

    def run_tool_session(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tool_turns: int = 5,
        required_tools: list[str] | None = None,
        allowed_tools: set[str] | None = None,
        generation: int = 0,
        session_mode: str = "TOOL_SESSION",
        agent_role_override: str = "",
        require_json: bool = False,
        expected_json_type: str = "object",
    ) -> str:
        """Runs a lightweight tool-enabled (or 0-turn JSON) conversation and returns the final assistant text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_turns_int = int(max_tool_turns)
        if max_turns_int <= 0:
            tool_specs: list[dict[str, Any]] | None = []
            num_iters = 1
        elif allowed_tools is not None:
            tool_specs = [
                spec
                for spec in OPENAI_TOOL_SPECS
                if (spec.get("function") or {}).get("name") in allowed_tools
            ]
            num_iters = max_turns_int + 1
        else:
            tool_specs = None
            num_iters = max_turns_int + 1

        prev_role = self.agent_role
        if agent_role_override:
            self.agent_role = agent_role_override
        last_content = ""
        called_tools: set[str] = set()
        called_expert_query = False
        json_nudges = 0
        req = [str(t) for t in (required_tools or []) if str(t).strip()]
        try:
            for turn_idx in range(1, num_iters + 1):
                allow_tools = max_turns_int > 0 and turn_idx <= max_turns_int
                # Fix M-3: Only append budget-reached message after actual tool outputs
                if (
                    not allow_tools
                    and max_turns_int > 0
                    and messages
                    and messages[-1].get("role") == "tool"
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool turn budget reached. Do NOT call any more tools or emit `<tool_call>` tags—"
                                "output your final required JSON response now based on the evidence gathered above."
                            ),
                        }
                    )
                t0_turn = time.perf_counter()
                msg = self._invoke_llm(
                    messages, allow_tools=allow_tools, tool_specs=tool_specs
                )
                turn_ms = (time.perf_counter() - t0_turn) * 1000.0
                raw_content = str(msg.get("content") or "")
                content = re.sub(
                    r"<think>.*?</think>", "", raw_content, flags=re.DOTALL
                )
                content = re.sub(
                    r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL
                ).strip()
                reasoning_text = str(msg.get("_reasoning") or "")
                p_toks, c_toks, r_toks = extract_usage_tokens(msg.get("_usage"))
                ts_cost_usd = extract_usage_cost(msg.get("_usage"))
                if content:
                    last_content = content
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    missing: list[str] = []
                    for t in req:
                        if t == "query_db:expert_runs":
                            if not called_expert_query:
                                missing.append("query_db (on expert_runs/expert_ticks)")
                        elif t not in called_tools:
                            missing.append(t)
                    if missing and turn_idx <= max_turns_int:
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Required tool(s) not yet called: {', '.join(missing)}. "
                                    "You MUST call both `query_wiki` and `query_db` on `expert_runs` / `expert_ticks` "
                                    "before returning your final JSON response."
                                ),
                            }
                        )
                        continue
                    if (
                        require_json
                        and not self._has_valid_json(content, expected_json_type)
                        and json_nudges < 2
                        and turn_idx <= max_turns_int
                    ):
                        json_nudges += 1
                        shape_desc = (
                            "a valid JSON array `[...]`"
                            if expected_json_type == "array"
                            else "a valid JSON object `{...}`"
                        )
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your previous response did not contain {shape_desc} and made no tool calls. "
                                    f"Either invoke your tools to investigate first, or return ONLY {shape_desc} matching the required schema."
                                ),
                            }
                        )
                        continue
                    try:
                        self.db_logger.save_synthesis_log(
                            log_id=f"syn_ts_{generation}_{self.agent_role}_{turn_idx}_{int(time.time() * 1000)}",
                            generation=generation,
                            batch_id=f"swarm_g{generation}_{self.agent_role}",
                            mode=session_mode,
                            turn_index=turn_idx,
                            turn_type=session_mode,
                            repair_idx=0,
                            model=self.model,
                            thinking_content=reasoning_text,
                            message_content=content,
                            tool_calls_json="[]",
                            tool_results_json="[]",
                            status="FINAL_RESPONSE",
                            prompt_tokens=p_toks,
                            completion_tokens=c_toks,
                            reasoning_tokens=r_toks,
                            agent_role=self.agent_role,
                            cost_usd=ts_cost_usd,
                            wall_ms=turn_ms,
                        )
                    except Exception as log_exc:
                        _LOG.warning("save_synthesis_log failed: %s", log_exc)
                    return last_content
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": tcs,
                    }
                )
                tool_results_list: list[dict[str, str]] = []
                for idx_tc, tc in enumerate(tcs, start=1):
                    fn_spec = tc.get("function") or {}
                    fn_name = str(fn_spec.get("name") or "")
                    if fn_name:
                        called_tools.add(fn_name)
                        if fn_name == "query_db":
                            parsed_args, _ = self._parse_tool_args(
                                fn_spec.get("arguments", "{}")
                            )
                            sql_low = str((parsed_args or {}).get("sql", "")).lower()
                            if "expert_runs" in sql_low or "expert_ticks" in sql_low:
                                called_expert_query = True
                    if allowed_tools is not None and fn_name not in allowed_tools:
                        res_out = f"Error: Tool '{fn_name}' is not permitted in read-only mode."
                    else:
                        res_out = self._dispatch_raw_tool_call(turn_idx, tc)
                    tool_results_list.append(
                        {"name": fn_name, "result": str(res_out)[:2000]}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"tc_{turn_idx}_{idx_tc}"),
                            "name": fn_name,
                            "content": res_out,
                        }
                    )
                try:
                    self.db_logger.save_synthesis_log(
                        log_id=f"syn_ts_{generation}_{self.agent_role}_{turn_idx}_{uuid.uuid4().hex[:6]}",
                        generation=generation,
                        batch_id=f"swarm_g{generation}_{self.agent_role}",
                        mode=session_mode,
                        turn_index=turn_idx,
                        turn_type=session_mode,
                        repair_idx=0,
                        model=self.model,
                        thinking_content=reasoning_text,
                        message_content=content,
                        tool_calls_json=json.dumps(tcs),
                        tool_results_json=json.dumps(tool_results_list),
                        status="TOOL_CALLS",
                        prompt_tokens=p_toks,
                        completion_tokens=c_toks,
                        reasoning_tokens=r_toks,
                        agent_role=self.agent_role,
                        cost_usd=ts_cost_usd,
                        wall_ms=turn_ms,
                    )
                except Exception as log_exc:
                    _LOG.warning("save_synthesis_log failed: %s", log_exc)
            return last_content
        finally:
            self.agent_role = prev_role
