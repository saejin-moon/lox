"""
LOX-ψ Policy Layer (LOX-ψ): defmacro definitional expansion + closure checker (R3, MACRO.md §5).

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
    """Condition-node depth per lox.policy.predicates.parse convention: the root call
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
    from lox.policy.predicates import QuotedStr
    return {node} if isinstance(node, str) and not isinstance(node, QuotedStr) else set()


def expr_names(node) -> set[str]:
    """All called predicate/macro/combinator names in the expression."""
    if not is_expr_node(node):
        return set()  # atoms contribute no names
    _, name, args = node
    out = {name}
    for a in args:
        out |= expr_names(a)
    return out


class MacroDef:
    """Represents a defined macro with optional parameters."""

    def __init__(self, name: str, body: tuple, params: tuple[str, ...] | list[str] = ()):
        self.name = name
        self.body = body
        self.params = tuple(params)

    def __eq__(self, other):
        if isinstance(other, MacroDef):
            return self.name == other.name and self.body == other.body and self.params == other.params
        return self.body == other

    def __repr__(self):
        if self.params:
            return f"MacroDef({self.name}({', '.join(self.params)}) = {self.body})"
        return f"MacroDef({self.name} = {self.body})"


class MacroEnv:
    """Live macro table: name → MacroDef. Immutable after construction except
    through the validator's diff-application path."""

    def __init__(self, macros: dict[str, Any] | None = None):
        self.macros: dict[str, MacroDef] = {}
        if macros:
            for k, v in macros.items():
                if isinstance(v, MacroDef):
                    self.macros[k] = v
                elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], (list, tuple)) and not (len(v[0]) > 0 and v[0][0] == "call"):
                    # (params, body)
                    self.macros[k] = MacroDef(k, v[1], tuple(v[0]))
                else:
                    self.macros[k] = MacroDef(k, v, ())

    def __contains__(self, name: str) -> bool:
        return name in self.macros

    def names(self) -> set[str]:
        return set(self.macros)

    def add(self, name: str, body: tuple, params: tuple[str, ...] | list[str] = ()) -> None:
        self.macros[name] = MacroDef(name, body, tuple(params))

    def get(self, name: str) -> MacroDef | None:
        return self.macros.get(name)


def _substitute(node, param_map: dict[str, Any]):
    """Substitutes parameter tokens with caller argument values."""
    if not param_map:
        return node
    if isinstance(node, str):
        return param_map.get(node, node)
    if is_expr_node(node):
        tag, name, args = node
        new_args = [_substitute(a, param_map) for a in args]
        if name in param_map:
            replacement = param_map[name]
            if is_expr_node(replacement):
                if not new_args:
                    return replacement
                return replacement
            elif isinstance(replacement, str):
                return (tag, replacement, new_args)
            else:
                return replacement
        return (tag, name, new_args)
    if isinstance(node, tuple):
        return tuple(_substitute(x, param_map) for x in node)
    if isinstance(node, list):
        return [_substitute(x, param_map) for x in node]
    return node


def expand(node, env: MacroEnv, _stack: tuple = ()):
    """Recursively substitutes macro calls (§5.2). Raises MacroError on cycles,
    arity violations, unknown macros, or budget exhaustion. Deterministic and pure."""
    if isinstance(node, str):
        # Bare-symbol sugar: a live macro name used as a zero-arg predicate.
        if node in env.names():
            macro_def = env.macros[node]
            body = macro_def.body if hasattr(macro_def, "body") else macro_def
            params = macro_def.params if hasattr(macro_def, "params") else ()
            if params:
                raise MacroError("ERR_SIGNATURE",
                                 f"parameterized macro {node!r} cannot be used as bare symbol without arguments")
            return expand(body, env, _stack)
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
        macro_def = env.macros[name]
        body = macro_def.body if hasattr(macro_def, "body") else macro_def
        params = macro_def.params if hasattr(macro_def, "params") else ()

        if len(args) != len(params):
            if not params:
                raise MacroError("ERR_SIGNATURE",
                                 f"macro {name!r} is parameterless (v1); called with {len(args)} args")
            raise MacroError("ERR_SIGNATURE",
                             f"macro {name!r} expects {len(params)} args ({', '.join(params)}); called with {len(args)}")

        if len(_stack) >= MAX_EXPANSION_DEPTH:
            raise MacroError("ERR_MACRO_CYCLE",
                             f"expansion depth exceeds {MAX_EXPANSION_DEPTH} at {name!r}")

        # Substitute arguments into parameters if parameterized
        if params:
            param_map = dict(zip(params, args))
            body = _substitute(body, param_map)

        return expand(body, env, _stack + (name,))

    # Ordinary predicate call: expand children
    return ("call", name, [expand(a, env, _stack) for a in args])


def check_closure(body, known_predicates: set[str], known_macros: set[str],
                  where: str = "macro body",
                  params: set[str] | None = None) -> None:
    """§5.3 closure check: every referenced symbol is a manifest predicate, a live macro,
    a declared parameter, or a built-in combinator. Any other symbol → ERR_MACRO_CLOSURE."""
    declared = params or set()
    names = expr_names(body)
    for n in names:
        if n in ("and", "or", "not"):
            continue
        if n in known_predicates or n in known_macros or n in declared:
            continue
        raise MacroError("ERR_MACRO_CLOSURE",
                         f"{where}: symbol {n!r} is not in the closed vocabulary")
    # Free (bare) symbols must also resolve — they are zero-arg predicates, macros,
    # declared parameters, or comparison-op strings (values, not symbols).
    for s in free_symbols(body):
        if s in known_predicates or s in known_macros or s in declared or s in ("<=", ">=", "<", ">", "==", "!="):
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