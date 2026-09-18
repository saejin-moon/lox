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

def nethack_bindings(ctx: dict) -> dict:
    """ctx keys: xl, depth, dnum, lawful, has_poison_res, has_reflection, has_excalibur,
    minetown_visited, donations_count, ac, steps_in_goal, turn."""
    return {
        "true": lambda: True,
        "false": lambda: False,
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
