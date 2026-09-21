"""
LOX-ψ Policy Layer (CORP): defmacro definitional expansion + closure checker (R3, MACRO.md §5).

A macro is a named expression abbreviation over already-live primitives and already-live
macros. Expansion is purely definitional (§5.1): every (m ...) call is replaced by its
body, the result is re-validated against the same closed vocabulary, and the expanded
depth counts against the caller's budget. No recursion, no shadowing, no parameter
capture, no I/O — deterministic and pure (§5.3).

v1 macros are parameterless (grammar `(defmacro <name> <expr>)`); a macro call with
arguments is ERR_SIGNATURE.
"""
from __future__ import annotations

MAX_EXPANSION_DEPTH = 4       # §5.3: no recursion; max expansion depth
MAX_CONDITION_DEPTH = 3       # §2.2: condition node depth (post-expansion)


class MacroError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def is_expr_node(node) -> bool:
    return isinstance(node, tuple) and len(node) == 3 and node[0] == "call"


def node_depth(node) -> int:
    """Condition-node depth per corp.policy.predicates.parse convention: the root call
    is depth 0, each nested call +1. Atoms (str/int/float/bool) contribute 0."""
    if not is_expr_node(node):
        return 0
    _, _, args = node
    if not args:
        return 0
    return 1 + max(node_depth(a) for a in args)


def free_symbols(node) -> set[str]:
    """Bare (non-call) string symbols referenced by the expression."""
    if is_expr_node(node):
        _, _, args = node
        out: set[str] = set()
        for a in args:
            out |= free_symbols(a)
        return out
    return {node} if isinstance(node, str) else set()


def expr_names(node) -> set[str]:
    """All called predicate/macro/combinator names in the expression."""
    if not is_expr_node(node):
        return set()  # atoms contribute no names
    _, name, args = node
    out = {name}
    for a in args:
        out |= expr_names(a)
    return out


class MacroEnv:
    """Live macro table: name → body expr node. Immutable after construction except
    through the validator's diff-application path."""

    def __init__(self, macros: dict[str, tuple] | None = None):
        self.macros: dict[str, tuple] = dict(macros or {})

    def __contains__(self, name: str) -> bool:
        return name in self.macros

    def names(self) -> set[str]:
        return set(self.macros)

    def add(self, name: str, body: tuple) -> None:
        self.macros[name] = body


def expand(node, env: MacroEnv, _stack: tuple = ()):
    """Recursively substitutes macro calls (§5.2). Raises MacroError on cycles,
    arity violations, unknown macros, or budget exhaustion. Deterministic and pure."""
    if isinstance(node, str):
        # Bare-symbol sugar: a live macro name used as a zero-arg predicate.
        if node in env.names():
            return expand(env.macros[node], env, _stack)
        return node
    if is_expr_node(node):
        pass  # handled below
    elif isinstance(node, (int, float, bool)):
        return node  # numeric/bool literal (e.g. a comparison value)
    else:
        raise MacroError("ERR_SIGNATURE", f"bad expr node {node!r}")
    _, name, args = node

    if name in ("and", "or", "not"):
        return ("call", name, [expand(a, env, _stack) for a in args])

    if name in env.names():
        if name in _stack:
            chain = " -> ".join(_stack + (name,))
            raise MacroError("ERR_MACRO_CYCLE", f"macro cycle detected: {chain}")
        if args:
            raise MacroError("ERR_SIGNATURE",
                             f"macro {name!r} is parameterless (v1); called with {len(args)} args")
        if len(_stack) >= MAX_EXPANSION_DEPTH:
            raise MacroError("ERR_MACRO_CYCLE",
                             f"expansion depth exceeds {MAX_EXPANSION_DEPTH} at {name!r}")
        body = env.macros[name]
        return expand(body, env, _stack + (name,))

    # Ordinary predicate call: expand children (macro names may appear as bare args
    # inside combinators; a macro as a direct predicate-call head with args is an error
    # handled above — here the name must be a manifest predicate, checked by the caller).
    return ("call", name, [expand(a, env, _stack) for a in args])


def check_closure(body, known_predicates: set[str], known_macros: set[str],
                  where: str = "macro body") -> None:
    """§5.3 closure check: every referenced symbol is a manifest predicate, a live macro,
    or a built-in combinator. Any other symbol → ERR_MACRO_CLOSURE."""
    names = expr_names(body)
    for n in names:
        if n in ("and", "or", "not"):
            continue
        if n in known_predicates or n in known_macros:
            continue
        raise MacroError("ERR_MACRO_CLOSURE",
                         f"{where}: symbol {n!r} is not in the closed vocabulary")
    # Free (bare) symbols must also resolve — they are zero-arg predicates, macros,
    # or comparison-op strings (values, not symbols).
    for s in free_symbols(body):
        if s in known_predicates or s in known_macros or s in ("<=", ">=", "<", ">", "=="):
            continue
        raise MacroError("ERR_MACRO_CLOSURE",
                         f"{where}: bare symbol {s!r} is not in the closed vocabulary")


def expand_all(node, env: MacroEnv, known_predicates: set[str]) -> tuple:
    """Expand then depth-check. Returns the fully expanded node."""
    out = expand(node, env)
    if node_depth(out) > MAX_CONDITION_DEPTH:
        raise MacroError("ERR_DEPTH_BUDGET",
                         f"expanded expression depth {node_depth(out)} exceeds {MAX_CONDITION_DEPTH}")
    # Post-expansion closure: every leaf must now be a manifest predicate.
    for n in expr_names(out):
        if n in ("and", "or", "not"):
            continue
        if n not in known_predicates:
            raise MacroError("ERR_MACRO_CLOSURE",
                             f"post-expansion symbol {n!r} is not a manifest predicate")
    return out