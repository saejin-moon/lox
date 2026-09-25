"""
LOX-ψ Policy Layer: Policy-diff reader & AST parser (Pure Pythonic Infix AST + backward-compatible forms).

Parses a full diff document into typed diff objects.
The primary diff language is Pure Pythonic Infix AST (zero parentheses in authoring),
parsed safely via `lox.policy.infix`. Legacy parenthesized diff forms remain fully
supported for backward compatibility.

Depth budget: condition nesting ≤ 3 is enforced by AST depth bounding.
Error codes follow MACRO.md §6 taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lox.policy.predicates import tokenize, _atom, QuotedStr  # noqa: F401


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
    params: list[str] = field(default_factory=list)


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

def _parse_form(tokens: list[str], _unused, pos: list, depth: int, max_depth: int = 4):
    """Recursive form parser. `pos` is a one-element list holding the cursor.

    Depth counting: the form itself is depth 1; its children depth 2, ... Whole-form
    paren nesting must be ≤ 4 for DIFFS (MACRO.md §2.2 — the spec counts the condition
    tree, which excludes the outer op wrapper; 4 here == 3 in spec terms). The
    authoring-tree compiler (S1) passes a larger budget: stored programs carry
    expanded macro conditions that legitimately nest deeper than author input."""
    if depth >= max_depth:
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
            args.append(_parse_form(tokens, None, pos, depth + 1, max_depth))
        else:
            args.append(_atom(tok))
            pos[0] += 1
    if pos[0] >= len(tokens):
        raise DiffParseError("ERR_PARSE", "unbalanced parens (EOF)")
    pos[0] += 1  # consume ')'
    return ("call", head, args)


def _is_expr_node(node) -> bool:
    return isinstance(node, tuple) and node and node[0] == "call"


def parse_forms(text: str, max_depth: int = 4) -> list:
    """Parse diff text into a list of top-level form nodes. Comments stripped by tokenizer.
    `max_depth` defaults to the diff budget (4); the tree compiler passes more."""
    tokens = tokenize(text)
    if not tokens:
        raise DiffParseError("ERR_PARSE", "empty diff document")
    forms: list = []
    pos = [0]
    n = len(tokens)
    while pos[0] < n:
        if tokens[pos[0]] != "(":
            raise DiffParseError("ERR_PARSE", f"expected top-level form, got {tokens[pos[0]]!r}")
        forms.append(_parse_form(tokens, None, pos, 0, max_depth))
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


def _extract_opt(form_args: list, key: str):
    """Finds (key <expr>) inside a form's args; returns the expr node, token list, or None."""
    for a in form_args:
        if _is_expr_node(a) and a[1] == key:
            if len(a[2]) == 1:
                return a[2][0]
            elif len(a[2]) > 1:
                return a[2]
    return None


def _extract_opt_str(form_args: list, key: str) -> str | None:
    for a in form_args:
        if _is_expr_node(a) and a[1] == key and len(a[2]) == 1 and isinstance(a[2][0], str):
            return a[2][0]
    return None


def _as_expr(node):
    """Leniency and Pythonic Infix AST parsing:
    1. If node is a list of tokens (e.g. from `(when hp_frac <= 0.40 and not is_fighting)`),
       parse it as a Pythonic infix AST condition.
    2. If node is a string containing infix operators ('and', 'or', 'not', '<=', '>=', '<', '>', '==', '!='),
       parse it as a Pythonic infix AST condition.
    3. If node is a bare string identifier (e.g. 'is_fighting'), wrap as ('call', node, []).
    4. If node is a ('call', name, args) tuple where args contain unparsed infix operators,
       parse via infix.
    5. Otherwise return the canonical ('call', name, args) tuple.
    """
    if node is None:
        return None

    def _fmt(x):
        if isinstance(x, QuotedStr):
            return f'"{x}"'
        if isinstance(x, str):
            if x.isidentifier() or x in ("and", "or", "not", "<=", ">=", "<", ">", "==", "!="):
                return x
            if x.startswith('"') and x.endswith('"'):
                return x
            return f'"{x}"'
        return str(x)

    if isinstance(node, list):
        tokens_str = " ".join(_fmt(x) for x in node)
        try:
            from lox.policy.infix import parse_infix_to_canonical  # noqa: PLC0415
            return parse_infix_to_canonical(tokens_str)
        except Exception:
            pass
    if isinstance(node, str):
        if any(tok in node for tok in (" and ", " or ", "not ", "<=", ">=", "<", ">", "==", "!=")):
            try:
                from lox.policy.infix import parse_infix_to_canonical  # noqa: PLC0415
                return parse_infix_to_canonical(node)
            except Exception:
                pass
        return ("call", node, [])
    if isinstance(node, tuple) and node and node[0] == "call":
        name, args = node[1], node[2]
        if args and any(isinstance(a, str) and a in ("and", "or", "not", "<=", ">=", "<", ">", "==", "!=") for a in args):
            tokens_str = f"{name} {' '.join(_fmt(a) for a in args)}"
            try:
                from lox.policy.infix import parse_infix_to_canonical  # noqa: PLC0415
                return parse_infix_to_canonical(tokens_str)
            except Exception:
                pass
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
        params = []
        if len(args) >= 3 and isinstance(args[1], (tuple, list)):
            params = [x for x in args[1] if isinstance(x, str)]
            body_arg = args[2]
        elif len(args) >= 2 and isinstance(args[1], (tuple, str)):
            body_arg = args[1]
        else:
            raise DiffParseError("ERR_PARSE", "defmacro requires (defmacro <name> [<params>] <expr>)")
        return DefmacroOp(name=name, body=_as_expr(body_arg), params=params)

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
            elif a[1] == "when":
                when = a[2][0] if len(a[2]) == 1 else a[2]
            elif a[1] == "unless":
                unless = a[2][0] if len(a[2]) == 1 else a[2]
            elif a[1] == "note" and len(a[2]) == 1 and isinstance(a[2][0], str):
                note = a[2][0]
    if do_verb is None or when is None:
        raise DiffParseError("ERR_PARSE", f"rule body requires (when <expr>) and (do <verb>): {ctx}")
    if not isinstance(when, (tuple, str, list)):
        raise DiffParseError("ERR_PARSE", f"rule (when ...) requires an expression: {ctx}")
    if not isinstance(unless, (tuple, str, list)) and unless is not None:
        raise DiffParseError("ERR_PARSE", f"rule (unless ...) requires an expression: {ctx}")
    return {"when": _as_expr(when), "do": do_verb,
            "unless": _as_expr(unless) if unless is not None else None, "note": note}


def parse_diff(text: str) -> PolicyDiff:
    """Parses a full diff document into a PolicyDiff. Raises DiffParseError (ERR_PARSE).
    Supports both pure Pythonic Infix AST diffs (zero parentheses) and legacy S-expressions."""
    stripped = text.strip()
    if not stripped.startswith("("):
        from lox.policy.infix import parse_pythonic_diff  # noqa: PLC0415
        return parse_pythonic_diff(text)

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
        # Quoted literals always render quoted. Bare atoms render bare when they are
        # operator tokens or symbol-shaped (predicate-name sugar like `has_excalibur`
        # inside `(not has_excalibur)`); anything else (spaces, punctuation) is quoted
        # so it survives a re-parse as one string atom.
        if isinstance(node, QuotedStr):
            return '"' + node + '"'
        if node in ("<=", ">=", "<", ">", "==") or node.replace("_", "a").isalnum():
            return node
        return '"' + node + '"'
    return str(node)


def render_infix(node, top: bool = True) -> str:
    """Serializes a canonical ('call', name, args) expr node to Pythonic Infix AST text."""
    if not isinstance(node, tuple) or len(node) != 3 or node[0] != "call":
        if isinstance(node, bool):
            return "true" if node else "false"
        if isinstance(node, str):
            clean = node.strip("\"'")
            return f'"{clean}"'
        return str(node)
    _, name, args = node
    if name == "and":
        s = " and ".join(render_infix(a, False) for a in args)
        return s if top else f"({s})"
    if name == "or":
        s = " or ".join(render_infix(a, False) for a in args)
        return s if top else f"({s})"
    if name == "not":
        return f"not {render_infix(args[0], False)}"
    if name in ("true", "false") and not args:
        return name
    if not args:
        return name
    if len(args) == 1 and name in ("monster", "item"):
        clean = str(args[0]).strip("\"'")
        return f'{name} == "{clean}"'
    if len(args) == 2 and args[0] in ("<=", ">=", "<", ">", "==", "!="):
        return f"{name} {args[0]} {args[1]}"
    rendered_args = ", ".join(render_infix(a, True) for a in args)
    return f"{name}({rendered_args})"


def extract_diff_text(raw: str) -> str:
    """Extracts the diff from raw LLM output.
    Supports both:
    1. Pythonic Infix AST diffs (natural lines starting with 'revision:', or inside code fences)
    2. S-expression diffs (starting with '(revision ...)')
    Raises DiffParseError(ERR_PARSE) if absent."""
    import re
    # Check for markdown code fences first
    fence_match = re.search(r"```(?:diff|python|lox)?\s*\n(.*?revision.*?)\n```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cand = fence_match.group(1).strip()
        if "(revision" in cand:
            return extract_diff_text(cand)
        if re.search(r"\brevision\b", cand, re.IGNORECASE):
            return cand

    idx = raw.find("(revision")
    if idx < 0:
        # Check for Pythonic diff
        m_rev = re.search(r"^\s*(revision\s*[:\s=].*)$", raw, re.MULTILINE | re.IGNORECASE)
        if m_rev:
            start_pos = m_rev.start()
            lines = []
            for line in raw[start_pos:].splitlines():
                sline = line.strip()
                if sline.startswith("# ") or sline.startswith("## ") or sline.startswith("### "):
                    break
                lines.append(line)
            diff_cand = "\n".join(lines).strip()
            if diff_cand:
                return diff_cand
        raise DiffParseError("ERR_PARSE", "no revision form found in author output")
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