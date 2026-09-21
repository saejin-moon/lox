"""
CORP-Ω Policy Layer: validator gate pipeline (R3, MACRO.md §6).

Order is NORMATIVE:
  1 PARSE → 2 HEADER → 3 VOCAB → 4 BOUNDS → 5 BUDGETS → 6 MACRO CHECK → 7 EXPANSION
  → 8 PROGRAM MOUNT → 9 INVARIANTS → 10 SHADOW TEST → 11 CERTIFICATION* → 12 QUICK BATCH*
  (*11/12 are injectable callables — the real gates run via scripts/run_revision_loop.py.)

Every failure returns a stable error code (MACRO.md §6 taxonomy) recorded in the ledger
and fed back to the author LLM for one repair retry. Rejection is first-class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corp.policy import dsl
from corp.policy.predicates import evaluate, nethack_bindings, default_ctx
from corp.policy.dsl import (
    PolicyDiff, SetOp, RuleOp, GoalOp, NogoodOp, DefmacroOp, render_node,
)
from corp.policy import macros as macro_mod
from corp.policy.manifest import VocabularyManifest, build_manifest
from corp.policy.manifest import VERBS as COMBAT_RESPONSE_VERBS
from corp.policy.program import PolicyProgram, GoalSpec

# MACRO.md §2.3 size budgets (per accepted revision)
BUDGET_DIFF_FORMS = 40
BUDGET_SET_OPS = 10
BUDGET_RULE_OPS = 5
BUDGET_DEFMACRO_OPS = 3
BUDGET_GOAL_OPS = 2
BUDGET_NOGOOD_OPS = 10
MAX_LIVE_MACROS = 64

# §6 step-9 invariants
DESCEND_CLASS_GOALS = {"descend", "explore_floor"}   # ≥1 must remain present


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    ok: bool
    error_code: str | None = None
    detail: str = ""
    gate: str | None = None
    candidate: PolicyProgram | None = None
    diff: PolicyDiff | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def reject(cls, code: str, gate: str, detail: str, diff: PolicyDiff | None = None) -> "GateResult":
        return cls(ok=False, error_code=code, gate=gate, detail=detail, diff=diff)


# ---------------------------------------------------------------------------
# Signature + vocabulary checking (manifest-driven, MACRO.md §4.2 rule 1)
# ---------------------------------------------------------------------------

def check_signature(name: str, args: list, manifest: VocabularyManifest) -> str:
    """Validates a predicate call's arity and argument types. Returns '' when OK,
    otherwise an ERR_SIGNATURE detail string."""
    sig = manifest.predicates.get(name)
    if sig is None:
        return f"symbol {name!r} not in vocabulary"
    specs = sig["args"]
    if len(args) != len(specs):
        return f"predicate {name!r} expects {len(specs)} args, got {len(args)}"
    for a, spec in zip(args, specs):
        kind = spec[0]
        if kind == "op":
            if not (isinstance(a, str) and str(a) in ("<=", ">=", "<", ">", "==")):
                return f"predicate {name!r}: bad comparison operator {a!r}"
        elif kind == "float":
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                return f"predicate {name!r}: expected float, got {a!r}"
            if len(spec) == 3 and not (spec[1] <= float(a) <= spec[2]):
                return f"predicate {name!r}: value {a} outside [{spec[1]}, {spec[2]}]"
        elif kind == "int":
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                return f"predicate {name!r}: expected int, got {a!r}"
        elif kind == "str":
            if not isinstance(a, str):
                return f"predicate {name!r}: expected string, got {a!r}"
        elif kind == "level":
            from corp.policy.predicates import _HUNGER_LEVELS
            if not (isinstance(a, str) and str(a) in _HUNGER_LEVELS):
                return f"predicate {name!r}: bad hunger level {a!r}"
    return ""


def _walk_expr(node, manifest: VocabularyManifest, macros_live: set[str]) -> str:
    """Vocabulary walk over an expr node. Returns '' or an error detail string."""
    if isinstance(node, str):
        if node in manifest.predicates or node in macros_live:
            return ""
        return f"bare symbol {node!r} not in vocabulary"
    if not macro_mod.is_expr_node(node):
        return f"bad expression node {node!r}"
    _, name, args = node
    if name in ("and", "or", "not"):
        if name == "not" and len(args) != 1:
            return "not/ expects exactly 1 argument"
        errs = [_walk_atom(a, manifest, macros_live) for a in args]
        return next((e for e in errs if e), "")
    if name in macros_live:
        if args:
            return f"macro {name!r} is parameterless (v1); called with {len(args)} args"
        return ""
    if name not in manifest.predicates:
        return f"symbol {name!r} not in vocabulary"
    err = check_signature(name, args, manifest)
    if err:
        return err
    # Predicate arguments are values (numbers/op strings) or nested exprs — only
    # nested nodes recurse; bare atoms here are data, not symbols.
    for a in args:
        if macro_mod.is_expr_node(a):
            err = _walk_expr(a, manifest, macros_live)
            if err:
                return err
    return ""


def _walk_atom(a, manifest: VocabularyManifest, macros_live: set[str]) -> str:
    if macro_mod.is_expr_node(a):
        return _walk_expr(a, manifest, macros_live)
    if isinstance(a, str):
        if a in manifest.predicates or a in macros_live:
            return ""
        return f"bare symbol {a!r} not in vocabulary"
    return f"bad combinator argument {a!r}"


# ---------------------------------------------------------------------------
# Injectable heavy gates + shadow fixtures
# ---------------------------------------------------------------------------

@dataclass
class ValidatorHooks:
    fixture_states: list[dict] = field(default_factory=list)   # ctx dicts for shadow tests
    certification: object | None = None    # fn(candidate) -> None (raise on fail)
    quick_batch: object | None = None      # fn(candidate) -> {"mean_score": float}
    baseline_score: float | None = None    # current program's rolling quick-batch mean
    baseline_transfer: dict = field(default_factory=dict)  # R5: success_rate + median_steps_on_success


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def validate_diff(
    diff_text: str,
    program: PolicyProgram,
    manifest: VocabularyManifest | None = None,
    hooks: ValidatorHooks | None = None,
) -> GateResult:
    hooks = hooks or ValidatorHooks()
    if manifest is None:
        manifest = build_manifest(program.version, program.domain,
                                  live_macros={m["name"]: m["body"] for m in program.macros})

    # ---- 1 PARSE ----------------------------------------------------------
    try:
        diff = dsl.parse_diff(diff_text)
    except dsl.DiffParseError as e:
        return GateResult.reject("ERR_PARSE", "parse", e.detail)
    except Exception as e:  # noqa: BLE001 — any parser crash is a parse rejection
        return GateResult.reject("ERR_PARSE", "parse", str(e))

    # ---- 2 HEADER ---------------------------------------------------------
    if diff.header.parent != program.version:
        return GateResult.reject("ERR_HEADER", "header",
                                 f"parent {diff.header.parent} != current program version {program.version}", diff)
    if diff.header.revision != program.version + 1:
        return GateResult.reject("ERR_HEADER", "header",
                                 f"revision {diff.header.revision} != parent+1 ({program.version + 1})", diff)
    if diff.header.domain != program.domain:
        return GateResult.reject("ERR_HEADER", "header",
                                 f"domain {diff.header.domain!r} != program domain {program.domain!r}", diff)

    # ---- 5 BUDGETS (cheap, fail loud before semantic gates) ----------------
    n_forms = len(diff.sets) + len(diff.rules) + len(diff.goals) + len(diff.nogoods) + len(diff.defmacros) + len(diff.notes)
    if n_forms > BUDGET_DIFF_FORMS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{n_forms} forms > {BUDGET_DIFF_FORMS}", diff)
    if len(diff.sets) > BUDGET_SET_OPS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{len(diff.sets)} set ops > {BUDGET_SET_OPS}", diff)
    if len(diff.rules) > BUDGET_RULE_OPS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{len(diff.rules)} rule ops > {BUDGET_RULE_OPS}", diff)
    if len(diff.defmacros) > BUDGET_DEFMACRO_OPS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{len(diff.defmacros)} defmacro ops > {BUDGET_DEFMACRO_OPS}", diff)
    if len(diff.goals) > BUDGET_GOAL_OPS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{len(diff.goals)} goal ops > {BUDGET_GOAL_OPS}", diff)
    if len(diff.nogoods) > BUDGET_NOGOOD_OPS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets", f"{len(diff.nogoods)} nogood ops > {BUDGET_NOGOOD_OPS}", diff)

    # ---- 3 VOCAB (new macros known by name; full resolution post-expansion) --
    n_live_after = len(program.macros) + len(diff.defmacros)
    if n_live_after > MAX_LIVE_MACROS:
        return GateResult.reject("ERR_FORM_BUDGET", "budgets",
                                 f"live macro count {n_live_after} exceeds {MAX_LIVE_MACROS}", diff)
    macros_live = {m["name"] for m in program.macros} | {d.name for d in diff.defmacros}
    for d in diff.defmacros:
        if d.name in manifest.predicates or d.name in manifest.verbs or d.name in manifest.goals:
            return GateResult.reject("ERR_MACRO_SHADOW", "macros",
                                     f"macro name {d.name!r} collides with a primitive", diff)
    for expr in diff.all_exprs():
        err = _walk_expr(expr, manifest, macros_live)
        if err:
            return GateResult.reject("ERR_UNKNOWN_SYMBOL", "vocab", err, diff)
    for op in diff.rules:
        if op.rule is not None and op.rule["do"] not in manifest.verbs:
            return GateResult.reject("ERR_UNKNOWN_SYMBOL", "vocab",
                                     f"verb {op.rule['do']!r} not in closed verb set", diff)
    for op in diff.goals:
        if op.kind != "set_threshold" and not manifest.is_goal(op.goal):
            return GateResult.reject("ERR_UNKNOWN_SYMBOL", "vocab",
                                     f"goal {op.goal!r} is not a certified goal handler", diff)
        if op.kind == "set_threshold" and op.path not in manifest.goal_owned_params(op.goal):
            return GateResult.reject("ERR_UNKNOWN_SYMBOL", "vocab",
                                     f"param {op.path!r} is not owned by goal {op.goal!r}", diff)

    # ---- 4 BOUNDS ---------------------------------------------------------
    for op in diff.sets:
        leaf = manifest.leaf(_leaf_path(op.path))
        if leaf is None:
            return GateResult.reject("ERR_BOUNDS", "bounds",
                                     f"param path {op.path!r} is not a tunable leaf", diff)
        coerced = _coerce_leaf(leaf, op.value)
        if coerced is None:
            return GateResult.reject("ERR_BOUNDS", "bounds",
                                     f"param {op.path!r}: value {op.value!r} mismatches type {leaf['type']}", diff)
        if not (leaf["min"] <= coerced <= leaf["max"]):
            return GateResult.reject("ERR_BOUNDS", "bounds",
                                     f"param {op.path!r} = {coerced} outside [{leaf['min']}, {leaf['max']}]", diff)
        op.value = coerced  # normalized
    for op in diff.goals:
        if op.kind == "set_threshold":
            leaf = manifest.leaf(_leaf_path(op.path))
            if leaf is None:
                return GateResult.reject("ERR_BOUNDS", "bounds",
                                         f"param {op.path!r} is not a tunable leaf", diff)
            coerced = _coerce_leaf(leaf, op.value)
            if coerced is None or not (leaf["min"] <= coerced <= leaf["max"]):
                return GateResult.reject("ERR_BOUNDS", "bounds",
                                         f"goal threshold {op.path!r} = {op.value!r} out of bounds", diff)
            op.value = coerced

    # ---- 6 MACRO CHECK + 7 EXPANSION --------------------------------------
    env = macro_mod.MacroEnv()
    for m in program.macros:
        try:
            env.add(m["name"], dsl.parse_diff_forms(m["body"])[0])
        except Exception as e:  # noqa: BLE001
            return GateResult.reject("ERR_PARSE", "macros",
                                     f"live macro {m['name']!r} body unparseable: {e}", diff)
    for d in diff.defmacros:
        try:
            macro_mod.check_closure(d.body, set(manifest.predicates), env.names())
        except macro_mod.MacroError as e:
            return GateResult.reject(e.code, "macros", f"defmacro {d.name!r}: {e.detail}", diff)
        env.add(d.name, d.body)

    expanded: dict[int, tuple] = {}   # id(expr) -> expanded node
    try:
        for expr in diff.all_exprs():
            expanded[id(expr)] = macro_mod.expand_all(expr, env, set(manifest.predicates))
    except macro_mod.MacroError as e:
        return GateResult.reject(e.code, "expansion", e.detail, diff)

    # ---- 8 PROGRAM MOUNT --------------------------------------------------
    result = _mount(diff, program, expanded, manifest, env)
    if isinstance(result, GateResult):
        return result
    candidate = result

    # ---- 9 INVARIANTS -----------------------------------------------------
    parent_goals = {g.goal for g in program.strategy_plan}
    cand_goals = {g.goal for g in candidate.strategy_plan}
    missing = parent_goals - cand_goals
    if missing:
        return GateResult.reject("ERR_INVARIANT", "invariants",
                                 f"diff removes certified goals: {sorted(missing)}", diff)
    if not (cand_goals & DESCEND_CLASS_GOALS):
        return GateResult.reject("ERR_INVARIANT", "invariants",
                                 "no descent-class goal present (must keep descend/explore_floor)", diff)
    if len(candidate.nogoods) < len(program.nogoods):
        return GateResult.reject("ERR_INVARIANT", "invariants",
                                 "nogood count decreased (nogood removal is not expressible in v1)", diff)

    # ---- 10 SHADOW TEST ---------------------------------------------------
    if hooks.fixture_states:
        try:
            _shadow_test(candidate, hooks.fixture_states)
        except Exception as e:  # noqa: BLE001
            return GateResult.reject("ERR_SHADOW_FAIL", "shadow", str(e), diff)

    # ---- 11 CERTIFICATION -------------------------------------------------
    if hooks.certification is not None:
        try:
            hooks.certification(candidate)
        except Exception as e:  # noqa: BLE001
            return GateResult.reject("ERR_CERT_FAIL", "certification", str(e), diff)

    # ---- 12 QUICK BATCH ---------------------------------------------------
    if hooks.quick_batch is not None:
        try:
            stats = hooks.quick_batch(candidate)
        except Exception as e:  # noqa: BLE001
            return GateResult.reject("ERR_BATCH_REGRESS", "quick-batch", str(e), diff)
        if hooks.baseline_score is not None and stats.get("mean_score") is not None:
            if stats["mean_score"] < hooks.baseline_score * 0.5:
                return GateResult.reject(
                    "ERR_BATCH_REGRESS", "quick-batch",
                    f"mean score {stats['mean_score']} < 50% of baseline {hooks.baseline_score}", diff)

    return GateResult(ok=True, candidate=candidate, diff=diff)


# ---------------------------------------------------------------------------
# Mount + helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf_path(path: str) -> str:
    """Normalizes a diff param path to the manifest's dotted-leaf form (no prefix)."""
    return path[len("policy_params."):] if path.startswith("policy_params.") else path


def _coerce_leaf(leaf: dict, value):
    """Type-coerce value for a param leaf; returns None on mismatch."""
    try:
        if leaf["type"] == "bool":
            return value if isinstance(value, bool) else None
        if leaf["type"] == "int":
            return int(round(float(value)))
        if leaf["type"] == "float":
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _render_rule(rule: dict, expanded: dict) -> dict:
    """Renders a rule dict (expr nodes) into program-storage strings."""
    when_node = expanded.get(id(rule["when"]), rule["when"])
    unless_node = rule.get("unless")
    if unless_node is not None:
        unless_node = expanded.get(id(unless_node), unless_node)
    return {
        "when": render_node(when_node),
        "do": rule["do"],
        "unless": render_node(unless_node) if unless_node is not None else None,
        "note": rule.get("note"),
    }


def _mount(diff: PolicyDiff, program: PolicyProgram, expanded: dict,
           manifest: VocabularyManifest, env: "macro_mod.MacroEnv") -> PolicyProgram | GateResult:
    """Applies the diff to a candidate program copy (§6 step 8)."""
    cand = PolicyProgram.from_dict(program.to_dict())

    # sets (stored WITHOUT the policy_params. prefix — PolicyProgram.apply_overlay contract)
    for op in diff.sets:
        cand.params[_leaf_path(op.path)] = op.value

    # rule ops (tactic_rules stored as rendered dicts — evaluator lands R4)
    for op in diff.rules:
        if op.action == "remove":
            if op.index is not None:
                if not (0 <= op.index < len(cand.tactic_rules)):
                    return GateResult.reject("ERR_BOUNDS", "mount",
                                             f"rule remove index {op.index} out of range", diff)
                if cand.tactic_rules[op.index].get("interlock"):
                    return GateResult.reject("ERR_INVARIANT", "invariants",
                                             f"rule {op.index} is a safety interlock and "
                                             "cannot be removed by a diff (#9b)", diff)
                cand.tactic_rules.pop(op.index)
            else:
                rendered = _render_rule(op.rule, expanded)
                for i, r in enumerate(cand.tactic_rules):
                    if r.get("when") == rendered["when"] and r.get("do") == rendered["do"]:
                        if r.get("interlock"):
                            return GateResult.reject("ERR_INVARIANT", "invariants",
                                                     "matched rule is a safety interlock and "
                                                     "cannot be removed by a diff (#9b)", diff)
                        cand.tactic_rules.pop(i)
                        break
                else:
                    return GateResult.reject("ERR_UNKNOWN_SYMBOL", "mount",
                                             "rule remove: matching rule not found", diff)
        else:
            rendered = _render_rule(op.rule, expanded)
            # R6 regression lesson (2026-09-19, ledger v6): the bare catch-all
            # `(when (adjacent_hostiles)) -> ranged_then_kill` mis-fired across the
            # whole early game and cost 45% mean score. Combat-verb rules MUST be
            # target-conditional — the author prompt states this; enforce here.
            if (rendered.get("do") in COMBAT_RESPONSE_VERBS
                    and "(monster " not in rendered.get("when", "")
                    and "(item " not in rendered.get("when", "")):
                return GateResult.reject(
                    "ERR_INVARIANT", "invariants",
                    "combat-verb rules must be target-conditional: the 'when' expression "
                    "must contain a (monster \"...\") or (item \"...\") match — global "
                    "catch-alls are rejected (v6 A/B: catch-all cost 45% mean score)", diff)
            cand.tactic_rules.append(rendered)

    # goal ops
    plan = list(cand.strategy_plan)
    for op in diff.goals:
        idx = next((i for i, g in enumerate(plan) if g.goal == op.goal), None)
        if op.kind == "prioritize":
            if idx is not None:
                g = plan.pop(idx)
            else:
                g = _goal_spec_from(op)
            if op.when is not None:
                g.raw["when"] = g.when = render_node(expanded[id(op.when)])
            if op.until is not None:
                g.raw["until"] = g.until = render_node(expanded[id(op.until)])
            plan.insert(0, g)
        elif op.kind == "deprioritize":
            if idx is None:
                return GateResult.reject("ERR_UNKNOWN_SYMBOL", "mount",
                                         f"deprioritize: goal {op.goal!r} not in plan", diff)
            g = plan.pop(idx)
            if op.after is not None:
                g.raw["after"] = render_node(expanded[id(op.after)])
            plan.append(g)
        else:  # set_threshold
            if idx is None:
                return GateResult.reject("ERR_UNKNOWN_SYMBOL", "mount",
                                         f"set_threshold: goal {op.goal!r} not in plan", diff)
            cand.params[_leaf_path(op.path)] = op.value
    cand.strategy_plan = plan

    # defmacros: store as {"name", "body"} with the ORIGINAL body text (expansion is
    # performed at validation/compile time, never stored expanded — §5.1 definitional).
    for d in diff.defmacros:
        cand.macros = [m for m in cand.macros if m["name"] != d.name]
        cand.macros.append({"name": d.name, "body": render_node(d.body)})

    # nogoods: append declarative entries (mask compilation lands R4)
    for op in diff.nogoods:
        cand.nogoods.append({
            "when": render_node(expanded[id(op.when)]),
            "cause": op.cause,
            "forbid": op.forbid,
        })

    cand.version = diff.header.revision
    cand.provenance = {
        **program.provenance,
        "author": diff.header.author,
        "parent_version": program.version,
        "reason": diff.header.reason,
    }
    return cand


def _goal_spec_from(op: GoalOp) -> GoalSpec:
    raw = {"goal": op.goal, "when": "(true)", "until": "(false)"}
    if op.when is not None:
        raw["when"] = render_node(op.when)
    if op.until is not None:
        raw["until"] = render_node(op.until)
    return GoalSpec(goal=op.goal, when=raw["when"], until=raw["until"], raw=raw)


def _shadow_test(candidate: PolicyProgram, fixture_states: list[dict]) -> None:
    """§5.5/§6 step 10: every expanded rule/goal/nogood condition must evaluate to a
    deterministic bool without raising on every fixture state."""
    live_macros = {m["name"]: m["body"] for m in candidate.macros}
    manifest = build_manifest(candidate.version, candidate.domain, live_macros=live_macros)
    env = macro_mod.MacroEnv()
    for m in candidate.macros:
        env.add(m["name"], dsl.parse_diff_forms(m["body"])[0])
    for ctx in fixture_states:
        bindings = nethack_bindings(default_ctx(**ctx))
        for g in candidate.strategy_plan:
            for cond in (g.when, g.until):
                node = dsl.parse_diff_forms(cond)[0]
                evaluate(macro_mod.expand_all(node, env, set(manifest.predicates)), bindings)
        for r in candidate.tactic_rules:
            evaluate(macro_mod.expand_all(dsl.parse_diff_forms(r["when"])[0], env,
                                          set(manifest.predicates)), bindings)
        for ng in candidate.nogoods:
            if "when" in ng:
                evaluate(macro_mod.expand_all(dsl.parse_diff_forms(ng["when"])[0], env,
                                              set(manifest.predicates)), bindings)