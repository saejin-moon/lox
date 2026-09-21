"""
CORP-Ω Policy Layer: named predicate registry + S-expression condition evaluator (R2).

Conditions in the policy program are S-expression strings over a **closed vocabulary** of
named predicates bound per-episode by the GoalInterpreter, e.g.:

    "(or (xl_ge 4) (depth_ge 4))"
    "(and (lawful) (xl_ge 5) (not has_excalibur))"

This is the R2 subset of the full diff DSL (MACRO.md): conditions only — `set`/`rule`/`goal`
operations arrive in R3 and reuse this parser. Depth limit 3 enforced here as well.
"""
from __future__ import annotations

from corp.policy.config import default_config


# ---------------------------------------------------------------------------
# Tokenizer + parser
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "()":
            out.append(c)
            i += 1
        elif c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise ValueError("unterminated string")
            out.append(text[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            out.append(text[i:j])
            i = j
    return out


def parse(text: str):
    """Parse one S-expression. Returns nested tuples: ('call', name, [args...]).
    Args are str | int | float | nested node."""
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("empty condition")
    pos = 0

    def parse_one(depth: int):
        nonlocal pos
        if depth > 3:
            raise ValueError("ERR_DEPTH_BUDGET: condition nesting exceeds 3")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            if tokens[pos] == "(":
                raise ValueError("expected predicate name after '('")
            name = tokens[pos]
            pos += 1
            args: list = []
            while pos < len(tokens) and tokens[pos] != ")":
                if tokens[pos] == "(":
                    args.append(parse_one(depth + 1))
                else:
                    args.append(_atom(tokens[pos]))
                    pos += 1
            if pos >= len(tokens):
                raise ValueError("unbalanced parens")
            pos += 1  # consume ')'
            return ("call", name, args)
        raise ValueError(f"expected '(' got {tok!r}")

    node = parse_one(0)
    if pos != len(tokens):
        raise ValueError("trailing tokens after expression")
    return node


def _atom(tok: str):
    if tok.startswith('"'):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


# ---------------------------------------------------------------------------
# Built-in predicate bindings (NetHack). Bindings are closures over a per-turn
# context snapshot assembled by the GoalInterpreter.
# ---------------------------------------------------------------------------

def _c(ctx: dict, key: str, default):
    """Robust context read: fixture states and partial contexts must never KeyError."""
    v = ctx.get(key, default)
    return default if v is None else v


DEFAULT_FIXTURE_CTX: dict = {
    "xl": 1, "depth": 1, "dnum": 0, "lawful": False,
    "has_poison_res": False, "has_reflection": False, "has_excalibur": False,
    "sokoban_completed": False, "minetown_visited": False, "donations_count": 0,
    "ac": 9, "steps_in_goal": 0, "turn": 1,
    "hp": 20, "max_hp": 20, "hunger_state": 1, "adjacent_hostiles": 0,
    "has_healing": True, "is_fighting": False, "encumbrance": 0,
    "stairs_known": False, "failures": {}, "failures_total": 0,
}


def default_ctx(**overrides) -> dict:
    """A complete context for shadow/fixture evaluation, with safe overrides."""
    ctx = dict(DEFAULT_FIXTURE_CTX)
    ctx.update(overrides)
    return ctx


def _cmp(a, op: str, b) -> bool:
    op = str(op)
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "==":
        return a == b
    raise ValueError(f"ERR_SIGNATURE: bad comparison operator {op!r}")


_HUNGER_LEVELS = {"satiated": 0, "normal": 1, "hungry": 2, "weak": 3, "fainting": 4}


def _hunger_level(level) -> int:
    s = str(level)
    if s in _HUNGER_LEVELS:
        return _HUNGER_LEVELS[s]
    if s.isdigit():
        return int(s)
    raise ValueError(f"ERR_SIGNATURE: bad hunger level {level!r}")


def nethack_bindings(ctx: dict) -> dict:
    """ctx keys: xl, depth, dnum, lawful, has_poison_res, has_reflection, has_excalibur,
    minetown_visited, donations_count, ac, steps_in_goal, turn — plus R3 additions:
    hp, max_hp, hunger_state, adjacent_hostiles, has_healing, is_fighting,
    encumbrance, stairs_known, failures (dict goal→count). Missing keys default safely
    (R3 manifest predicates must be evaluable on partial/fixture contexts)."""
    hp_frac = (_c(ctx, "hp", 0) / max(1, _c(ctx, "max_hp", 1)))
    hunger_raw = _c(ctx, "hunger_state", 1)
    failures = ctx.get("failures") or {}
    return {
        "true": lambda: True,
        "false": lambda: False,
        # R3 comparison-sugar predicates: (hp_frac <= 0.40)
        "hp_frac": lambda op, v: _cmp(hp_frac, op, float(v)),
        "hunger_ge": lambda level: hunger_raw >= _hunger_level(level),
        "adjacent_hostiles": lambda: _c(ctx, "adjacent_hostiles", 0) > 0,
        "has_healing": lambda: bool(_c(ctx, "has_healing", False)),
        "is_fighting": lambda: bool(_c(ctx, "is_fighting", False)),
        "encumbrance_le": lambda n: _c(ctx, "encumbrance", 0) <= int(n),
        "donations_lt": lambda n: _c(ctx, "donations_count", 0) < int(n),
        "minetown_known": lambda: bool(_c(ctx, "minetown_visited", False)),
        "stairs_known": lambda: bool(_c(ctx, "stairs_known", False)),
        "failures_in_10_episodes_ge": lambda n, goal="": failures.get(goal, _c(ctx, "failures_total", 0)) >= int(n) if failures else _c(ctx, "failures_total", 0) >= int(n),
        # R4 tactic-rule match predicates (combat context: monster_name, item_name)
        "monster": lambda name: str(name).lower() in str(_c(ctx, "monster_name", "")).lower(),
        "item": lambda name: str(name).lower() in str(_c(ctx, "item_name", "")).lower(),
        "xl_ge": lambda n: ctx["xl"] >= int(n),
        "xl_le": lambda n: ctx["xl"] <= int(n),
        "depth_ge": lambda n: ctx["depth"] >= int(n),
        "depth_le": lambda n: ctx["depth"] <= int(n),
        "depth_between": lambda lo, hi: int(lo) <= ctx["depth"] <= int(hi),
        "in_dungeons": lambda: ctx["dnum"] == 0,
        "in_mines": lambda: ctx["dnum"] == 2,
        "in_sokoban": lambda: ctx["dnum"] == 3,
        "lawful": lambda: bool(ctx["lawful"]),
        "has_poison_res": lambda: bool(ctx["has_poison_res"]),
        "has_reflection": lambda: bool(ctx["has_reflection"]),
        "has_excalibur": lambda: bool(ctx["has_excalibur"]),
        # R6 milestone predicates (inventory/message-derived, see GoalInterpreter.update_state)
        "has_mr": lambda: bool(_c(ctx, "has_mr", False)),
        "has_light": lambda: bool(_c(ctx, "has_light", False)),
        "has_wishing_wand": lambda: bool(_c(ctx, "has_wishing_wand", False)),
        "has_amulet": lambda: bool(_c(ctx, "has_amulet", False)),
        "has_candelabrum": lambda: bool(_c(ctx, "has_candelabrum", False)),
        "castle_done": lambda: bool(_c(ctx, "castle_wishing_done", False)),
        "vlad_done": lambda: bool(_c(ctx, "vlad_defeated", False)),
        "invocation_done": lambda: bool(_c(ctx, "invocation_done", False)),
        "sokoban_completed": lambda: bool(ctx["sokoban_completed"]),
        "minetown_visited": lambda: bool(ctx["minetown_visited"]),
        "donations_ge": lambda n: ctx["donations_count"] >= int(n),
        "ac_le": lambda n: ctx["ac"] <= int(n),
        "steps_in_goal_ge": lambda n: ctx["steps_in_goal"] >= int(n),
        "turn_ge": lambda t: ctx["turn"] >= int(t),
    }


def _eval_arg(a, bindings: dict) -> bool:
    """Combinator args may be nested nodes OR bare zero-arg predicate symbols (sugar:
    `(not has_excalibur)` == `(not (has_excalibur))`)."""
    if isinstance(a, tuple):
        return evaluate(a, bindings)
    if isinstance(a, str):
        if a not in bindings:
            raise ValueError(f"ERR_UNKNOWN_SYMBOL: predicate {a!r} not in vocabulary")
        return bool(bindings[a]())
    raise ValueError(f"ERR_SIGNATURE: bad combinator argument {a!r}")


def evaluate(node, bindings: dict) -> bool:
    """Evaluate a parsed node against a binding table. and/or/not are built-in combinators."""
    kind, name, args = node
    if name in ("and", "or", "not"):
        sub = [_eval_arg(a, bindings) for a in args]
        if name == "and":
            return all(sub)
        if name == "or":
            return any(sub)
        return not sub[0]
    if name not in bindings:
        raise ValueError(f"ERR_UNKNOWN_SYMBOL: predicate {name!r} not in vocabulary")
    return bool(bindings[name](*args))


def eval_condition(text: str, bindings: dict) -> bool:
    return evaluate(parse(text), bindings)
