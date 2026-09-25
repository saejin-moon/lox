"""
LOX-ψ Policy Layer (LOX-ψ): named predicate registry + Pythonic Infix AST condition evaluator (R2).

Conditions in the policy program are Pythonic Infix AST expressions over a **closed vocabulary** of
named predicates bound per-episode by the GoalInterpreter, e.g.:

    "xl_ge(4) or depth_ge(4)"
    "lawful and xl_ge(5) and not has_excalibur"
    "monster == 'jackal' and hp_frac <= 0.40"

Pure Pythonic Infix AST is the primary condition syntax (zero Lisp parentheses), with legacy
forms supported for backward compatibility. Depth limit 3 enforced across all forms.
"""
from __future__ import annotations

from lox.policy.config import default_config


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
    """Parse a condition expression into canonical nested tuples: ('call', name, [args...]).
    Supports Pure Pythonic Infix AST (zero parentheses) and legacy parenthesized forms."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty condition")

    if not stripped.startswith("("):
        from lox.policy.infix import parse_infix_to_canonical  # noqa: PLC0415
        return parse_infix_to_canonical(text)

    try:
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
                if pos >= len(tokens):
                    raise ValueError("unbalanced parens")
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
    except ValueError as sexpr_err:
        try:
            from lox.policy.infix import parse_infix_to_canonical  # noqa: PLC0415
            return parse_infix_to_canonical(text)
        except Exception:
            raise sexpr_err


def _atom(tok: str):
    if tok.startswith('"'):
        # Quoted string literal: preserve quotedness through parsing so the S1
        # authoring-tree compiler can round-trip stored programs byte-equivalently
        # (a bare `has_excalibur` sugar symbol must stay bare; a literal "scorpion"
        # must stay quoted). QuotedStr == plain str by value, so the diff/validator
        # comparisons and str-typed signatures are unaffected.
        return QuotedStr(tok[1:-1])
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


class QuotedStr(str):
    """A quoted string literal parsed from S-expression text (see `_atom`)."""

    __slots__ = ()


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
    "stairs_dist": 10, "unvisited_count": 20, "is_corridor": False, "is_doorway": False,
    "count_hostiles": 0, "closest_threat": 0.0, "closest_speed": 12,
    "has_ranged_target": False, "is_monster_fleeing": False,
    "carried_food_count": 1, "turns_since_pray": 500, "can_safely_pray": True,
    "intrinsics": set(), "weapon_enchantment": 0, "inventory_items": set(),
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
        "failures_in_10_episodes_ge": lambda n, goal="": (failures.get(str(goal), 0) if goal else _c(ctx, "failures_total", 0)) >= int(n),
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
        # Gen-3 Sensory & Epistemic Predicates
        "stairs_dist": lambda op, v: _cmp(_c(ctx, "stairs_dist", 10), op, int(v)),
        "unvisited_count": lambda op, v: _cmp(_c(ctx, "unvisited_count", 20), op, int(v)),
        "is_corridor": lambda: bool(_c(ctx, "is_corridor", False)),
        "is_doorway": lambda: bool(_c(ctx, "is_doorway", False)),
        "count_hostiles": lambda op, v: _cmp(_c(ctx, "count_hostiles", 0), op, int(v)),
        "closest_threat": lambda op, v: _cmp(_c(ctx, "closest_threat", 0.0), op, float(v)),
        "closest_speed": lambda op, v: _cmp(_c(ctx, "closest_speed", 12), op, int(v)),
        "has_ranged_target": lambda: bool(_c(ctx, "has_ranged_target", False)),
        "is_monster_fleeing": lambda: bool(_c(ctx, "is_monster_fleeing", False)),
        "carried_food_count": lambda op, v: _cmp(_c(ctx, "carried_food_count", 1), op, int(v)),
        "turns_since_pray": lambda op, v: _cmp(_c(ctx, "turns_since_pray", 500), op, int(v)),
        "can_safely_pray": lambda: bool(_c(ctx, "can_safely_pray", True)),
        "has_intrinsic": lambda name: (str(name) in _c(ctx, "intrinsics", set())) or bool(ctx.get(f"has_{name}", False)),
        "current_ac": lambda op, v: _cmp(_c(ctx, "ac", 9), op, int(v)),
        "weapon_enchantment": lambda op, v: _cmp(_c(ctx, "weapon_enchantment", 0), op, int(v)),
        "has_item": lambda cat, name: any(str(cat).lower() in item.lower() and str(name).lower() in item.lower() for item in _c(ctx, "inventory_items", set())),
        "hunger_level": lambda op, v: _cmp(hunger_raw, op, _hunger_level(v)),
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
