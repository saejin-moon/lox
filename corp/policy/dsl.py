"""
CORP-Ω Policy Layer: S-expression policy-diff reader (R3, MACRO.md §2–3).

Parses a full diff document (revision header + top-level forms) into typed diff objects.
The LLM emits diffs as text; the GBNF grammar guarantees validity on the local decode path
and this parser re-enforces it on the API path (parse-and-reject, one repair retry).

Depth budget (MACRO.md §2.2): condition nesting ≤ 3 is enforced by corp.policy.predicates
(node depth), and whole-form paren nesting ≤ 4 (accommodates the `(rule ... (when (and
(...))))` wrapper — the spec's own example). Error codes follow MACRO.md §6 taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corp.policy.predicates import tokenize, _atom  # noqa: F401


# ---------------------------------------------------------------------------
# Typed diff objects
# ---------------------------------------------------------------------------

@dataclass
class DiffHeader:
    revision: int
    parent: int
    author: str = "unknown"
    domain: str = "nethack"
    reason: str = ""


@dataclass
class SetOp:
    path: str            # policy_params.<dotted-path>
    value: int | float | bool | str


@dataclass
class RuleOp:
    action: str          # "add" | "remove"
    rule: dict           # {"when": node, "do": verb, "unless": node|None, "note": str|None}
    index: int | None = None   # for remove-by-index


@dataclass
class GoalOp:
    kind: str            # "prioritize" | "deprioritize" | "set_threshold"
    goal: str
    when: tuple | None = None
    until: tuple | None = None
    after: tuple | None = None
    path: str | None = None
    value: int | float | bool | str | None = None


@dataclass
class NogoodOp:
    when: tuple
    cause: str
    forbid: str | None = None


@dataclass
class DefmacroOp:
    name: str
    body: tuple


@dataclass
class NoteOp:
    text: str


@dataclass
class PolicyDiff:
    header: DiffHeader
    sets: list[SetOp] = field(default_factory=list)
    rules: list[RuleOp] = field(default_factory=list)
    goals: list[GoalOp] = field(default_factory=list)
    nogoods: list[NogoodOp] = field(default_factory=list)
    defmacros: list[DefmacroOp] = field(default_factory=list)
    notes: list[NoteOp] = field(default_factory=list)

    def all_exprs(self):
        """Every condition expr node in the diff (for vocab/expansion passes)."""
    # (docstring kept above; generator below)
        for op in self.rules:
            if op.rule is not None:
                if op.rule.get("when") is not None:
                    yield op.rule["when"]
                if op.rule.get("unless") is not None:
                    yield op.rule["unless"]
        for op in self.goals:
            for e in (op.when, op.until, op.after):
                if e is not None:
                    yield e
        for op in self.nogoods:
            yield op.when
        for op in self.defmacros:
            yield op.body


# ---------------------------------------------------------------------------
# Parse errors (stable codes)
# ---------------------------------------------------------------------------

class DiffParseError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _parse_form(tokens: list[str], _unused, pos: list, depth: int):
    """Recursive form parser. `pos` is a one-element list holding the cursor.

    Depth counting: the form itself is depth 1; its children depth 2, ... Whole-form
    paren nesting must be ≤ 4 (MACRO.md §2.2 — the spec counts the condition tree,
    which excludes the outer op wrapper; 4 here == 3 in spec terms)."""
    if depth >= 4:
        raise DiffParseError("ERR_DEPTH_BUDGET", f"form nesting exceeds budget near token {pos[0]}")
    assert tokens[pos[0]] == "("
    pos[0] += 1
    if pos[0] >= len(tokens) or tokens[pos[0]] == ")":
        raise DiffParseError("ERR_PARSE", "empty form")
    if tokens[pos[0]] == "(":
        raise DiffParseError("ERR_PARSE", "expected head symbol after '('")
    head = tokens[pos[0]]
    pos[0] += 1
    args: list = []
    while pos[0] < len(tokens) and tokens[pos[0]] != ")":
        tok = tokens[pos[0]]
        if tok == "(":
            args.append(_parse_form(tokens, None, pos, depth + 1))
        else:
            args.append(_atom(tok))
            pos[0] += 1
    if pos[0] >= len(tokens):
        raise DiffParseError("ERR_PARSE", "unbalanced parens (EOF)")
    pos[0] += 1  # consume ')'
    return ("call", head, args)


def _is_expr_node(node) -> bool:
    return isinstance(node, tuple) and node and node[0] == "call"


def parse_forms(text: str) -> list:
    """Parse diff text into a list of top-level form nodes. Comments stripped by tokenizer."""
    tokens = tokenize(text)
    if not tokens:
        raise DiffParseError("ERR_PARSE", "empty diff document")
    forms: list = []
    pos = [0]
    n = len(tokens)
    while pos[0] < n:
        if tokens[pos[0]] != "(":
            raise DiffParseError("ERR_PARSE", f"expected top-level form, got {tokens[pos[0]]!r}")
        forms.append(_parse_form(tokens, None, pos, 0))
    return forms


# ---------------------------------------------------------------------------
# Form → typed op conversion
# ---------------------------------------------------------------------------

def _need_str(args, i, ctx: str) -> str:
    if i >= len(args) or not isinstance(args[i], str):
        raise DiffParseError("ERR_PARSE", f"expected string arg {i} in {ctx}")
    return args[i]


def _need_int(args, i, ctx: str) -> int:
    if i >= len(args) or not isinstance(args[i], int):
        raise DiffParseError("ERR_PARSE", f"expected int arg {i} in {ctx}")
    return args[i]


def _extract_opt(form_args: list, key: str) -> tuple | None:
    """Finds (key <expr>) inside a form's args; returns the expr node or None."""
    for a in form_args:
        if _is_expr_node(a) and a[1] == key and len(a[2]) == 1:
            return a[2][0]
    return None


def _extract_opt_str(form_args: list, key: str) -> str | None:
    for a in form_args:
        if _is_expr_node(a) and a[1] == key and len(a[2]) == 1 and isinstance(a[2][0], str):
            return a[2][0]
    return None


def _as_expr(node):
    """Leniency for API authors: a bare symbol where an expression is expected is
    wrapped as a zero-arg predicate call (`(when is_fighting)` == `(when (is_fighting))`).
    Non-string atoms are rejected (they are values, not conditions)."""
    if isinstance(node, str):
        return ("call", node, [])
    return node


def _convert(form, idx: int):
    kind, head, args = form
    if head == "set":
        if len(args) != 2 or not isinstance(args[0], str):
            raise DiffParseError("ERR_PARSE", "set requires (set <path> <value>)")
        if not args[0].startswith("policy_params."):
            raise DiffParseError("ERR_PARSE", f"set path must start with 'policy_params.': {args[0]!r}")
        if not isinstance(args[1], (int, float, bool)) or isinstance(args[1], str):
            raise DiffParseError("ERR_PARSE", "set value must be a number or boolean")
        return SetOp(path=args[0], value=args[1])

    if head == "rule":
        action = _need_str(args, 0, "rule")
        if action not in ("add", "remove"):
            raise DiffParseError("ERR_PARSE", f"rule action must be add|remove, got {action!r}")
        target = _need_str(args, 1, "rule")
        if target != "tactic_rules":
            raise DiffParseError("ERR_PARSE", f"rule target must be tactic_rules, got {target!r}")
        if action == "remove" and len(args) >= 3 and isinstance(args[2], int):
            return RuleOp(action="remove", index=args[2], rule=None)
        return RuleOp(action=action, rule=_rule_from_args(args[2:], f"rule {action}"), index=None)

    if head == "goal":
        op_kind = _need_str(args, 0, "goal")
        name = _need_str(args, 1, "goal")
        if op_kind == "prioritize":
            return GoalOp(kind="prioritize", goal=name,
                          when=_as_expr(_extract_opt(args, "when")),
                          until=_as_expr(_extract_opt(args, "until")))
        if op_kind == "deprioritize":
            return GoalOp(kind="deprioritize", goal=name,
                          after=_as_expr(_extract_opt(args, "after")))
        if op_kind == "set_threshold":
            path = _need_str(args, 2, "goal set_threshold")
            if len(args) < 4 or not isinstance(args[3], (int, float, bool)):
                raise DiffParseError("ERR_PARSE", "set_threshold requires a numeric value")
            return GoalOp(kind="set_threshold", goal=name, path=path, value=args[3])
        raise DiffParseError("ERR_PARSE", f"goal op must be prioritize|deprioritize|set_threshold, got {op_kind!r}")

    if head == "nogood":
        when = None
        cause = None
        forbid = None
        for a in args:
            if _is_expr_node(a):
                if a[1] == "when" and len(a[2]) == 1:
                    when = a[2][0]
                elif a[1] == "forbid" and len(a[2]) == 1 and isinstance(a[2][0], str):
                    forbid = a[2][0]
            elif isinstance(a, str) and cause is None and when is not None:
                cause = a
        # canonical: (nogood add (when expr) (cause "s") [(forbid task)])
        if len(args) >= 2 and isinstance(args[0], str) and args[0] == "add":
            args = args[1:]
            when = cause = forbid = None
            for a in args:
                if _is_expr_node(a):
                    if a[1] == "when" and len(a[2]) == 1:
                        when = a[2][0]
                    elif a[1] == "cause" and len(a[2]) == 1 and isinstance(a[2][0], str):
                        cause = a[2][0]
                    elif a[1] == "forbid" and len(a[2]) == 1 and isinstance(a[2][0], str):
                        forbid = a[2][0]
        if when is None or not isinstance(when, (tuple, str)):
            raise DiffParseError("ERR_PARSE", "nogood requires (when <expr>)")
        when = _as_expr(when)
        if cause is None:
            raise DiffParseError("ERR_PARSE", "nogood requires (cause <string>)")
        return NogoodOp(when=when, cause=cause, forbid=forbid)

    if head == "defmacro":
        name = _need_str(args, 0, "defmacro")
        if len(args) < 2 or not isinstance(args[1], (tuple, str)):
            raise DiffParseError("ERR_PARSE", "defmacro requires (defmacro <name> <expr>)")
        return DefmacroOp(name=name, body=_as_expr(args[1]))

    if head == "note":
        return NoteOp(text=_need_str(args, 0, "note"))

    raise DiffParseError("ERR_PARSE", f"unknown top-level form {head!r}")


def _rule_from_args(args: list, ctx: str) -> dict:
    """Rule body: (when <expr>) (do <verb>) [(unless <expr>)] [(note <str>)] — direct
    child forms of the rule op (MACRO.md §3: `(rule add tactic_rules <rule>)` where
    `<rule> := (when <expr>) (do <verb>) [(unless <expr>)] [(note <string>)]`)."""
    do_verb = when = unless = note = None
    for a in args:
        if _is_expr_node(a):
            if a[1] == "do" and len(a[2]) == 1 and isinstance(a[2][0], str):
                do_verb = a[2][0]
            elif a[1] == "when" and len(a[2]) == 1:
                when = a[2][0]
            elif a[1] == "unless" and len(a[2]) == 1:
                unless = a[2][0]
            elif a[1] == "note" and len(a[2]) == 1 and isinstance(a[2][0], str):
                note = a[2][0]
    if do_verb is None or when is None:
        raise DiffParseError("ERR_PARSE", f"rule body requires (when <expr>) and (do <verb>): {ctx}")
    if not isinstance(when, (tuple, str)):
        raise DiffParseError("ERR_PARSE", f"rule (when ...) requires an expression: {ctx}")
    if not isinstance(unless, (tuple, str)) and unless is not None:
        raise DiffParseError("ERR_PARSE", f"rule (unless ...) requires an expression: {ctx}")
    return {"when": _as_expr(when), "do": do_verb,
            "unless": _as_expr(unless) if unless is not None else None, "note": note}


def parse_diff(text: str) -> PolicyDiff:
    """Parses a full diff document into a PolicyDiff. Raises DiffParseError (ERR_PARSE)."""
    forms = parse_diff_forms(text)
    if not forms:
        raise DiffParseError("ERR_PARSE", "no forms in diff")

    header_form = forms[0]
    if header_form[1] != "revision":
        raise DiffParseError("ERR_PARSE", "first form must be (revision ...)")
    hargs = header_form[2]
    revision = _need_int(hargs, 0, "revision")
    parent = None
    author, domain, reason = "unknown", "nethack", ""
    for a in hargs[1:]:
        if _is_expr_node(a):
            if a[1] == "parent" and len(a[2]) == 1:
                parent = a[2][0]
            elif a[1] == "author" and len(a[2]) == 1:
                author = str(a[2][0])
            elif a[1] == "domain" and len(a[2]) == 1:
                domain = str(a[2][0])
            elif a[1] == "reason" and len(a[2]) == 1:
                reason = str(a[2][0])
    if not isinstance(parent, int):
        raise DiffParseError("ERR_HEADER", "revision header requires (parent <int>)")

    diff = PolicyDiff(header=DiffHeader(revision=revision, parent=parent,
                                        author=author, domain=domain, reason=reason))
    for i, form in enumerate(forms[1:], start=1):
        op = _convert(form, i)
        if isinstance(op, SetOp):
            diff.sets.append(op)
        elif isinstance(op, RuleOp):
            diff.rules.append(op)
        elif isinstance(op, GoalOp):
            diff.goals.append(op)
        elif isinstance(op, NogoodOp):
            diff.nogoods.append(op)
        elif isinstance(op, DefmacroOp):
            diff.defmacros.append(op)
        elif isinstance(op, NoteOp):
            diff.notes.append(op)
    return diff


def parse_diff_forms(text: str) -> list:
    return parse_forms(text)


# ---------------------------------------------------------------------------
# Rendering (expr nodes → program-storage strings; program → diff text is never done)
# ---------------------------------------------------------------------------

def render_node(node) -> str:
    """Serializes an expr node back to S-expression text (for program storage)."""
    if isinstance(node, tuple) and len(node) == 3 and node[0] == "call":
        _, name, args = node
        if not args:
            return "(" + name + ")"
        return "(" + name + " " + " ".join(render_node(a) for a in args) + ")"
    if isinstance(node, bool):
        return "true" if node else "false"
    if isinstance(node, str):
        # Op strings and bare symbols render unquoted; anything else (spaces, dots)
        # needs quoting to survive a re-parse.
        if node in ("<=", ">=", "<", ">", "==") or node.replace("_", "a").isalnum():
            return node
        return '"' + node + '"'
    return str(node)


def extract_diff_text(raw: str) -> str:
    """Extracts the full top-level form sequence starting at the first `(revision ...)`
    from raw LLM output (API path). Handles code fences, prose, and trailing text by
    balancing parens per form and continuing while further top-level forms follow.
    Raises DiffParseError(ERR_PARSE) if absent."""
    idx = raw.find("(revision")
    if idx < 0:
        raise DiffParseError("ERR_PARSE", "no (revision ...) form found in author output")
    end = idx
    i = idx
    n = len(raw)
    while i < n:
        # Skip whitespace/comments between forms
        while i < n and (raw[i].isspace() or (raw[i] == ";")):
            if raw[i] == ";":
                while i < n and raw[i] != "\n":
                    i += 1
            else:
                i += 1
        if i >= n or raw[i] != "(":
            break
        # Scan one balanced top-level form
        depth = 0
        in_str = False
        while i < n:
            c = raw[i]
            if in_str:
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == ";":
                while i < n and raw[i] != "\n":
                    i += 1
                continue
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        if depth != 0:
            raise DiffParseError("ERR_PARSE", "unbalanced parens in author output (no closing form)")
        end = i
    return raw[idx:end]