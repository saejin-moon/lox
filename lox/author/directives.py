"""
Swarm Directive Planner & Skill Tree Utilities for Layout A Peer Swarm.

Replaces legacy `SwarmMetaAgent` (BehaviorTree assembly / critic revision) with:
- `prune_unused_skills(...)`: transitive import reachability pruner for `skills/**/*.py`
- `extract_policy_branches(...)`: Option-A AST extractor for `register_goal(...)` names and functions
- `SwarmDirectivePlanner`: lightweight 1-shot orthogonal directive partitioner across peers
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from lox.lang.bundle import (
    extract_json_payload,
    extract_skill_imports,
    write_skills_dir,
)
from lox.lang.validator import PolicyValidationError, normalize_skill_key
from lox.ui.console import ui


def _safe_normalize_skill_key(raw: str) -> str:
    try:
        return normalize_skill_key(str(raw or ""))
    except (PolicyValidationError, ValueError, TypeError):
        return ""


def prune_unused_skills(
    policy_code: str,
    skills: dict[str, str],
    skills_dir: str | Path | None = None,
) -> dict[str, str]:
    """
    Finds the exact transitive set of `skills.<name>` / `skills.<domain>.<name>` modules
    imported by `policy_code` and removes orphaned `.py` files from `skills_dir` and `skills`.
    """
    norm_skills: dict[str, str] = {}
    for k, v in (skills or {}).items():
        nk = _safe_normalize_skill_key(k)
        if nk:
            norm_skills[nk] = v
    known_keys = set(norm_skills.keys())
    pkg_names = {k.split("/", 1)[0] for k in known_keys if "/" in k}

    if not norm_skills:
        if skills_dir is not None:
            s_dir = Path(skills_dir)
            if s_dir.exists():
                write_skills_dir(s_dir, {}, sync=True)
        return {}

    def _extract_imports(src: str) -> tuple[set[str], bool]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return set(), False
        info = extract_skill_imports(tree, known_keys, pkg_names=pkg_names)
        return info.referenced_keys, True

    keep: set[str] = set()
    seed_refs, pol_ok = _extract_imports(policy_code)
    if not pol_ok:
        return norm_skills
    queue = list(seed_refs)
    while queue:
        s_name = queue.pop(0)
        if s_name in keep or s_name not in norm_skills:
            continue
        keep.add(s_name)
        sub_refs, _ = _extract_imports(norm_skills[s_name])
        queue.extend(sub_refs)

    pruned = {k: v for k, v in norm_skills.items() if k in keep}
    removed = sorted(set(norm_skills.keys()) - keep)
    if skills_dir is not None:
        write_skills_dir(Path(skills_dir), pruned, sync=True)
    if removed and skills_dir is not None:
        ui.info(
            f"Swarm: Pruned {len(removed)} unused skill module(s) from workspace "
            f"(kept {len(pruned)} active): {removed[:8]}"
            + ("..." if len(removed) > 8 else "")
        )
    return pruned


def extract_policy_branches(
    policy_code: str,
    skills: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    Extracts Option-A registered goal names (`register_goal(...)`) and non-dunder
    callable function/method names across `policy_code` and `skills`.
    Returns a deduplicated list of `{"leaf_name": ..., "fn_name": ...}` dicts.
    """
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(leaf_name: str, fn_name: str) -> None:
        ln = str(leaf_name or "").strip()
        fn = str(fn_name or "").strip()
        if not ln and not fn:
            return
        key = (ln, fn)
        if key not in seen:
            seen.add(key)
            found.append({"leaf_name": ln or fn, "fn_name": fn or ln})

    sources: list[str] = []
    if policy_code and policy_code.strip():
        sources.append(policy_code)
    for sk_src in (skills or {}).values():
        if sk_src and sk_src.strip():
            sources.append(sk_src)

    for src in sources:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    )
                )
                if callee == "register_goal":
                    g_name = ""
                    handler_name = ""
                    if node.args and isinstance(node.args[0], ast.Constant):
                        g_name = str(node.args[0].value or "").strip()
                    for kw in node.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            g_name = str(kw.value.value or "").strip()
                        elif kw.arg in ("fn", "step", "handler", "tick"):
                            if isinstance(kw.value, ast.Name):
                                handler_name = kw.value.id
                            elif isinstance(kw.value, ast.Attribute):
                                handler_name = kw.value.attr
                    if g_name:
                        _add(g_name, handler_name or g_name)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):
                    _add(node.name, node.name)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not sub.name.startswith("__"):
                            _add(sub.name, sub.name)
    return found


def _parse_directives_json(raw_txt: str) -> list[dict[str, Any]]:
    parsed = extract_json_payload(raw_txt or "")
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for k in ("directives", "peers", "items", "categories"):
            val = parsed.get(k)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


class SwarmDirectivePlanner:
    """
    Lightweight 1-shot Directive Planner running on `self.model` (`--model`).
    Partitions parallel Layout A peers onto mutually exclusive investigative questions
    and subsystem scopes without prescribing hardcoded game strategies.
    """

    prune_unused_skills = staticmethod(prune_unused_skills)

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        db_path: str = "data/lox.duckdb",
        theories_store: Any = None,
        reasoning_effort: str = "medium",
        routing: str = "nitro",
    ) -> None:
        from lox.author.agent import AuthorAgent

        self.provider = str(provider or "openrouter").lower()
        self.model = str(model)
        self.db_path = db_path
        self.theories = theories_store
        self.author = AuthorAgent(
            provider=self.provider,
            model=self.model,
            db_path=db_path,
            agent_role="meta_directive_planner",
            reasoning_effort=reasoning_effort,
            routing=routing,
        )
        if theories_store is not None:
            self.author.tools.theories_store = theories_store

    def plan_peer_directives(
        self,
        generation: int,
        fresh_peer_ids: list[str],
        deficit_report: Any,
        architecture_map: str = "",
        active_stepping_stone: dict[str, Any] | None = None,
        peer_roles: dict[str, str] | None = None,
        active_proactive_branch: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Assigns mutually exclusive investigative questions & subsystem scopes across
        Layout A peers (`reactive_1`, `reactive_2`, `proactive_1`, `proactive_2`).
        Strictly honors 'Hard Code Nothing': neutral investigative scopes only.
        """
        peer_ids = [str(p) for p in fresh_peer_ids if str(p).strip()]
        if not peer_ids:
            return []

        roles_map: dict[str, str] = {}
        for pid in peer_ids:
            if peer_roles and pid in peer_roles:
                roles_map[pid] = str(peer_roles[pid])
            elif pid.startswith("proactive_2"):
                roles_map[pid] = "proactive_branch"
            elif pid.startswith("proactive"):
                roles_map[pid] = "proactive_capability"
            else:
                roles_map[pid] = "reactive"

        clusters = list(
            getattr(deficit_report, "fail_clusters", None)
            or getattr(deficit_report, "clusters", None)
            or []
        )
        cluster_ids = [str(getattr(c, "cluster_id", "") or "") for c in clusters]
        cooled_down = dict(getattr(deficit_report, "cooled_down_deficits", {}) or {})
        expert_gaps_txt = str(
            getattr(deficit_report, "expert_parity_gaps", "") or ""
        ).strip()

        def _fallback_directives() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            stone_sub = ""
            if active_stepping_stone:
                tfs = active_stepping_stone.get("target_functions") or []
                stone_sub = ", ".join(str(x) for x in tfs[:3])
            branch_goal = (
                str(active_proactive_branch.get("goal_summary", "") or "")[:120]
                if isinstance(active_proactive_branch, dict)
                else ""
            )
            ordered_clusters = sorted(
                clusters,
                key=lambda c: (
                    1 if str(getattr(c, "cluster_id", "")) in cooled_down else 0,
                    -int(getattr(c, "run_count", 0) or 0),
                ),
            )
            r_idx = 0
            for pid in peer_ids:
                role = roles_map.get(pid, "reactive")
                if role == "proactive_capability":
                    out.append(
                        {
                            "peer_id": pid,
                            "track": "proactive_capability",
                            "target_cluster_id": "proactive_expert_capability",
                            "focus_subsystem": "skills/goals (additive capability)",
                            "investigative_question": (
                                "Inspect `champion_runs`/`champion_ticks`, `expert_runs` progression milestones "
                                "(AC/XL/depth), and NetHack Wiki rules to design one clean, self-contained additive capability."
                            ),
                            "exclude_overlap": (
                                "Do NOT patch reactive death traces covered by reactive_1/reactive_2"
                                + (
                                    f" or duplicate proactive_2's branch ({branch_goal})"
                                    if branch_goal
                                    else ""
                                )
                                + "."
                            ),
                        }
                    )
                elif role == "proactive_branch":
                    step_n = (
                        int((active_proactive_branch or {}).get("steps_taken", 0) or 0)
                        + 1
                    )
                    max_s = int(
                        (active_proactive_branch or {}).get("max_steps", 4) or 4
                    )
                    out.append(
                        {
                            "peer_id": pid,
                            "track": "proactive_branch",
                            "target_cluster_id": "proactive_sheltered_branch",
                            "focus_subsystem": "sheltered proactive architecture branch",
                            "investigative_question": (
                                f"Advance multi-step proactive branch (step {step_n}/{max_s}): "
                                + (
                                    f"build upon '{branch_goal}' toward champion graduation."
                                    if branch_goal
                                    else "inspect `champion_runs`/`champion_ticks` + Wiki rules + `expert_runs` milestones and start a clean multi-step capability upgrade."
                                )
                            ),
                            "exclude_overlap": "Do NOT collide with reactive_1/reactive_2 failure patches or proactive_1's single-gen capability.",
                        }
                    )
                else:
                    cl = (
                        ordered_clusters[r_idx % len(ordered_clusters)]
                        if ordered_clusters
                        else None
                    )
                    r_idx += 1
                    cid = str(getattr(cl, "cluster_id", "general")) if cl else "general"
                    lbl = (
                        str(getattr(cl, "label", "systemic progression"))
                        if cl
                        else "systemic progression"
                    )
                    excl = (
                        f"Do NOT duplicate active stepping-stone work on ({stone_sub}); "
                        if stone_sub and role != "revision"
                        else ""
                    ) + "focus on a distinct subsystem from other peers."
                    out.append(
                        {
                            "peer_id": pid,
                            "track": role,
                            "target_cluster_id": cid,
                            "focus_subsystem": cid,
                            "investigative_question": f"Investigate root causes of `{cid}` ({lbl}) directly in `champion_ticks` and `skills/`.",
                            "exclude_overlap": excl,
                        }
                    )
            return out

        if self.provider == "mock":
            return _fallback_directives()

        brief_txt = (
            deficit_report.format_brief()
            if hasattr(deficit_report, "format_brief")
            else str(deficit_report or "")
        )[:3200]
        arch_txt = str(architecture_map or "")[:2000]
        stone_txt = "None (no active reactive stepping stone this generation)."
        if active_stepping_stone:
            stone_txt = (
                f"Active reactive stepping stone (handled by `reactive_1`) is refining Gen "
                f"{active_stepping_stone.get('generation')} `{active_stepping_stone.get('source_peer_id')}` "
                f"(targets={active_stepping_stone.get('target_functions')}, "
                f"W/L={active_stepping_stone.get('wins')}/{active_stepping_stone.get('losses')}, "
                f"reject_reason={active_stepping_stone.get('reject_reason')}, "
                f"mechanism={str(active_stepping_stone.get('mechanism', ''))[:200]}). "
                "Keep all other peers on ORTHOGONAL subsystems/questions."
            )
        branch_txt = "None (starting a fresh 4-generation sheltered proactive branch)."
        if isinstance(active_proactive_branch, dict) and active_proactive_branch:
            branch_txt = (
                f"Active `proactive_2` branch `{active_proactive_branch.get('branch_id')}` "
                f"(step {int(active_proactive_branch.get('steps_taken', 0)) + 1}/"
                f"{int(active_proactive_branch.get('max_steps', 4))}, "
                f"last_avg_depth={float(active_proactive_branch.get('last_avg_depth', 0.0)):.2f}, "
                f"last_winsor_delta={float(active_proactive_branch.get('last_winsor_delta', 0.0)):+.3f}): "
                f"goal=`{str(active_proactive_branch.get('goal_summary', ''))[:200]}`, "
                f"capabilities_added={active_proactive_branch.get('capabilities_added', [])}."
            )
        roles_desc = "\n".join(
            f"- `{pid}`: track=`{roles_map.get(pid, 'reactive')}`" for pid in peer_ids
        )
        parity_section = (
            f"\n### Champion-vs-Expert State Milestones & Unused Verbs:\n{expert_gaps_txt[:1500]}\n"
            if expert_gaps_txt
            else ""
        )
        cooldown_section = (
            "\n### Recently Cooled-Down Clusters (ROTATE AWAY Unless Zero Untouched Alternatives):\n"
            + ", ".join(
                f"`{cid}` (failed {cnt}x in last 4 gens)"
                for cid, cnt in sorted(
                    cooled_down.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            + "\n"
            if cooled_down
            else ""
        )

        prompt = f"""Assign {len(peer_ids)} ORTHOGONAL investigative directives for Generation {generation} peers:
{roles_desc}

### Active Reactive Stepping Stone (`reactive_1` when present):
{stone_txt}

### Active Sheltered Proactive Branch (`proactive_2`):
{branch_txt}
{parity_section}{cooldown_section}
### Empirical Deficit Report:
{brief_txt}

### Current Codebase Architecture Map:
{arch_txt}

### Strict Rules ("Hard Code Nothing"):
1. Assign each peer in {peer_ids} a **distinct `focus_subsystem` and neutral `investigative_question`** so NO two peers investigate or patch the same mechanism/subsystem (and none duplicate recently rejected mechanisms in Rejection Memory).
2. **Stagnation Rotation**: Prefer fresh, non-cooled-down, high-signal failure clusters (`slice_stalled`, `slice_recurring_outcome_*`, `slice_max_steps_confinement`, `slice_frontier_failures`, etc.) over clusters listed in `Recently Cooled-Down Clusters`. Do NOT keep re-assigning the same subsystem after 2+ rejections.
3. Respect each peer's track:
   - `reactive` / `revision` peers target concrete empirical failure clusters from `champion_runs` / `champion_ticks` (prefer valid `target_cluster_id` from {cluster_ids or ["general"]}).
   - `proactive_capability` (`proactive_1`) targets a clean, self-contained missing capability or state-progression gap grounded in `champion_runs`/`champion_ticks` + Wiki rules + `expert_runs` milestones across Dlvl 1..p90+2 (use `target_cluster_id: "proactive_expert_capability"`). Never tell peers to chase synthetic `expert_ticks` action-share percentages.
   - `proactive_branch` (`proactive_2`) advances the multi-step sheltered proactive architecture branch over its 4-generation horizon (use `target_cluster_id: "proactive_sheltered_branch"`).
4. Do NOT prescribe code changes, thresholds, or hardcoded game tactics — only specify WHICH empirical failure mode / capability gap / subsystem to investigate and WHAT other peers are covering (`exclude_overlap`).

Return ONLY a valid JSON array of {len(peer_ids)} objects, each with keys:
- "peer_id": one of {peer_ids}
- "target_cluster_id": cluster_id string
- "focus_subsystem": short module/subsystem scope (e.g. "skills/goals/explore.py" or "skills/survival/...")
- "investigative_question": 1-2 sentence neutral empirical question for the peer to answer via `query_db` / `query_wiki`
- "exclude_overlap": explicit 1-sentence boundary naming what other peers are covering so this peer does not collide with them
"""
        try:
            raw_txt = self.author.run_tool_session(
                system_prompt=(
                    "You are the LOX Swarm Directive Planner. Your ONLY job is to partition "
                    "parallel reactive and proactive peer agents onto non-overlapping empirical "
                    "questions and subsystems without prescribing hardcoded strategies. Output strict JSON only."
                ),
                user_prompt=prompt,
                max_tool_turns=0,
                generation=generation,
                session_mode="META_DIRECTIVES",
                agent_role_override="meta_directive_planner",
                require_json=True,
                expected_json_type="array",
            )
            parsed = _parse_directives_json(raw_txt)
            if not parsed:
                return _fallback_directives()
            by_pid: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(parsed):
                pid = str(item.get("peer_id") or "").strip()
                if pid not in peer_ids and idx < len(peer_ids):
                    pid = peer_ids[idx]
                if pid not in peer_ids or pid in by_pid:
                    continue
                role = roles_map.get(pid, "reactive")
                cid = str(item.get("target_cluster_id") or "").strip()
                if role == "proactive_capability":
                    cid = "proactive_expert_capability"
                elif role == "proactive_branch":
                    cid = "proactive_sheltered_branch"
                elif cluster_ids and cid not in cluster_ids:
                    cid = cluster_ids[idx % len(cluster_ids)]
                by_pid[pid] = {
                    "peer_id": pid,
                    "track": role,
                    "target_cluster_id": cid or "general",
                    "focus_subsystem": str(
                        item.get("focus_subsystem") or cid or "general"
                    )[:120],
                    "investigative_question": str(
                        item.get("investigative_question") or ""
                    )[:400],
                    "exclude_overlap": str(item.get("exclude_overlap") or "")[:300],
                }
            fallback = {d["peer_id"]: d for d in _fallback_directives()}
            return [by_pid.get(pid, fallback[pid]) for pid in peer_ids]
        except Exception as exc:
            ui.warn(f"Swarm Directive Planner fallback ({exc})")
            return _fallback_directives()
