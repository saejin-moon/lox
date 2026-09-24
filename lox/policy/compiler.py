"""
LOX-ψ Policy Layer (LOX-ψ): the authoring-tree compiler (S1, AGENT_PLAN §1.1).

The LLM authors into INDIVIDUAL files under `data/program/<domain>/`:

    program.json      # metadata: version, domain, provenance, role/domain profiles
    params.json       # {dotted-path: value} typed overlay (same shape as program.params)
    macros/*.sexpr    # (defmacro <name> <expr>) — parameterless sugar, one per file
    rules/*.sexpr     # (rule add tactic_rules (when ...) (do ...) [(unless ...)]
                      #  [(note "...")] [(interlock true)]) — one per file
    goals/*.sexpr     # (goal <name> (when ...) (until ...) [(directive "...")]
                      #  [(max_steps n)]) — one per file, NN_ prefixes fix plan order
    nogoods.jsonl     # one {"when","cause","forbid"} JSON object per line
    handlers/*.sexpr  # RESERVED for S3 (handler ladder) — must be empty in S1

`compile_authoring_tree` assembles + compiles the tree into the EXACT dict shape
`PolicyProgram.to_dict()` produces — the executor contract is unchanged
(PolicyProgram.load consumes data/compiled/<domain>.json with zero rewrites).
`write_authoring_tree` serializes a program back into the tree. Round-trips are
byte-equivalent: write → compile → to_dict == original program.to_dict
(enforced fail-loud in `compile_and_commit`).

Write authority is unchanged: only the validator's acceptance path
(revision loop / author session) may write trees + compiled artifacts; every
compiled artifact is archived under data/programs/archive/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager

from lox.policy import dsl
from lox.policy.program import PolicyProgram

PROGRAM_DIR = "data/program"
COMPILED_DIR = "data/compiled"
ARCHIVE_DIR = "data/programs/archive"

TREE_METADATA_FILE = "program.json"
TREE_PARAMS_FILE = "params.json"
TREE_NOGOODS_FILE = "nogoods.jsonl"
TREE_SUBDIRS = ("macros", "rules", "goals", "handlers")

_NUM_PREFIX = re.compile(r"^(\d+)_")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def tree_dir_for(domain: str) -> str:
    return os.path.join(PROGRAM_DIR, domain)


@contextmanager
def output_dirs(program_dir: str | None = None, compiled_dir: str | None = None,
                archive_dir: str | None = None):
    """Temporarily redirects the compiler's output dirs (experiment harnesses must
    never overwrite the live authoring tree / compiled artifact / archive)."""
    global PROGRAM_DIR, COMPILED_DIR, ARCHIVE_DIR
    saved = (PROGRAM_DIR, COMPILED_DIR, ARCHIVE_DIR)
    if program_dir is not None:
        PROGRAM_DIR = program_dir
    if compiled_dir is not None:
        COMPILED_DIR = compiled_dir
    if archive_dir is not None:
        ARCHIVE_DIR = archive_dir
    try:
        yield
    finally:
        PROGRAM_DIR, COMPILED_DIR, ARCHIVE_DIR = saved


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", str(name).strip().lower()) or "item"


def _sexpr_str(s: str, ctx: str) -> str:
    """A string literal inside a tree .sexpr file. The tokenizer does NOT process
    escape sequences, so quotes/newlines inside notes/directives are unrepresentable
    — fail loud instead of silently corrupting the tree."""
    if '"' in s or "\n" in s or "\\" in s:
        raise ValueError(f"ERR_UNREPRESENTABLE: {ctx}: string {s!r} contains quote, "
                         "newline, or backslash — not representable in tree .sexpr")
    return '"' + s + '"'


# ---------------------------------------------------------------------------
# S-expression form parsing (dedicated tree grammar — NOT the diff reader:
# tree forms carry full records, diff forms carry deltas)
# ---------------------------------------------------------------------------

def _parse_forms(text: str, ctx: str) -> list:
    """Tree forms are full records (expanded macro conditions included), so they get
    a deeper parser budget than diff input's MACRO.md cap of 4."""
    try:
        return dsl.parse_forms(text, max_depth=64)
    except dsl.DiffParseError as e:
        raise ValueError(f"ERR_PARSE: {ctx}: {e}") from e


def _child(node, key: str):
    """Finds (key <args...>) among a form's args; returns the arg list or None."""
    for a in node[2]:
        if dsl._is_expr_node(a) and a[1] == key:
            return a[2]
    return None


def _need_child(node, key: str, ctx: str) -> list:
    child = _child(node, key)
    if child is None:
        raise ValueError(f"ERR_PARSE: {ctx}: missing ({key} ...) clause")
    return child


def _parse_rule_file(path: str, text: str) -> dict:
    """(rule add tactic_rules (when ...) (do ...) [(unless ...)] [(note "…")]
    [(interlock true)]) → tactic-rule dict (key order = clause order)."""
    ctx = os.path.basename(path)
    forms = _parse_forms(text, ctx)
    if len(forms) != 1:
        raise ValueError(f"ERR_PARSE: {ctx}: exactly one (rule ...) form per file")
    node = forms[0]
    if node[1] != "rule" or not node[2] or node[2][0] != "add":
        raise ValueError(f"ERR_PARSE: {ctx}: expected (rule add tactic_rules ...)")
    target = node[2][1] if len(node[2]) > 1 else None
    if target != "tactic_rules":
        raise ValueError(f"ERR_PARSE: {ctx}: rule target must be tactic_rules")
    rule: dict = {}
    for a in node[2][2:]:
        if not dsl._is_expr_node(a):
            raise ValueError(f"ERR_PARSE: {ctx}: unexpected atom {a!r}")
        head, hargs = a[1], a[2]
        if head == "when" and len(hargs) == 1:
            rule["when"] = dsl.render_node(hargs[0])
        elif head == "do" and len(hargs) == 1 and isinstance(hargs[0], str):
            rule["do"] = hargs[0]
        elif head == "unless" and len(hargs) == 1:
            # `(unless null)` preserves the explicit "unless": None key some
            # program records carry; anything else is a condition expression.
            rule["unless"] = None if hargs[0] == "null" else dsl.render_node(hargs[0])
        elif head == "note" and len(hargs) == 1:
            # `(note null)` preserves an explicit "note": None key (the validator's
            # _render_rule always emits the key; byte-equivalence requires it back).
            if hargs[0] == "null":
                rule["note"] = None
            elif isinstance(hargs[0], str):
                rule["note"] = hargs[0]
            else:
                raise ValueError(f"ERR_PARSE: {ctx}: note must be a string or null")
        elif head == "interlock" and len(hargs) == 1 and hargs[0] in (True, "true"):
            rule["interlock"] = True
        else:
            raise ValueError(f"ERR_PARSE: {ctx}: unexpected clause ({head} ...)")
    if "when" not in rule or "do" not in rule:
        raise ValueError(f"ERR_PARSE: {ctx}: rule requires (when ...) and (do ...)")
    return rule


def _parse_goal_file(path: str, text: str) -> dict:
    """(goal <name> (when ...) (until ...) [(directive "…")] [(max_steps n)])
    → strategy-plan raw dict (key order = clause order)."""
    ctx = os.path.basename(path)
    forms = _parse_forms(text, ctx)
    if len(forms) != 1:
        raise ValueError(f"ERR_PARSE: {ctx}: exactly one (goal ...) form per file")
    node = forms[0]
    if node[1] != "goal" or not node[2] or not isinstance(node[2][0], str):
        raise ValueError(f"ERR_PARSE: {ctx}: expected (goal <name> ...)")
    raw: dict = {"goal": node[2][0]}
    for a in node[2][1:]:
        if not dsl._is_expr_node(a):
            raise ValueError(f"ERR_PARSE: {ctx}: unexpected atom {a!r}")
        head, hargs = a[1], a[2]
        if head == "when" and len(hargs) == 1:
            raw["when"] = dsl.render_node(hargs[0])
        elif head == "until" and len(hargs) == 1:
            raw["until"] = dsl.render_node(hargs[0])
        elif head == "directive" and len(hargs) == 1 and isinstance(hargs[0], str):
            raw["directive"] = hargs[0]
        elif head == "max_steps" and len(hargs) == 1 and isinstance(hargs[0], int):
            raw["max_steps"] = hargs[0]
        else:
            raise ValueError(f"ERR_PARSE: {ctx}: unexpected clause ({head} ...)")
    return raw


def _parse_macro_file(path: str, text: str) -> dict:
    """(defmacro <name> <expr>) → {"name", "body"} (body = ORIGINAL text)."""
    ctx = os.path.basename(path)
    forms = _parse_forms(text, ctx)
    if len(forms) != 1:
        raise ValueError(f"ERR_PARSE: {ctx}: exactly one (defmacro ...) form per file")
    node = forms[0]
    if node[1] != "defmacro" or len(node[2]) < 2:
        raise ValueError(f"ERR_PARSE: {ctx}: expected (defmacro <name> <expr>)")
    name = node[2][0]
    if not isinstance(name, str):
        raise ValueError(f"ERR_PARSE: {ctx}: macro name must be a symbol")
    return {"name": name, "body": dsl.render_node(node[2][1])}


def _ordered_tree_files(subdir: str) -> list[str]:
    """Files in a tree subdir, ordered by their NN_ numeric prefix (fallback: name)."""
    def key(fn: str):
        m = _NUM_PREFIX.match(fn)
        return (int(m.group(1)) if m else 10_000, fn)
    return sorted((f for f in os.listdir(subdir) if f.endswith(".sexpr")), key=key)


# ---------------------------------------------------------------------------
# Tree → PolicyProgram (compile)
# ---------------------------------------------------------------------------

def parse_authoring_tree(tree_dir: str) -> PolicyProgram:
    """Reads the authoring tree and compiles it into a PolicyProgram. Fail-loud:
    any malformed/missing piece raises ValueError with a stable ERR_ code."""
    if not os.path.isdir(tree_dir):
        raise ValueError(f"ERR_NO_TREE: authoring tree not found at {tree_dir}")
    meta_path = os.path.join(tree_dir, TREE_METADATA_FILE)
    params_path = os.path.join(tree_dir, TREE_PARAMS_FILE)
    if not os.path.exists(meta_path):
        raise ValueError(f"ERR_NO_TREE: missing {TREE_METADATA_FILE} in {tree_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    domain = meta.get("domain") or os.path.basename(os.path.normpath(tree_dir))
    params: dict = {}
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        params = loaded.get("params", loaded) if isinstance(loaded, dict) else {}

    macros = []
    macros_dir = os.path.join(tree_dir, "macros")
    if os.path.isdir(macros_dir):
        for fn in _ordered_tree_files(macros_dir):
            with open(os.path.join(macros_dir, fn), "r", encoding="utf-8") as f:
                macros.append(_parse_macro_file(fn, f.read()))

    rules = []
    rules_dir = os.path.join(tree_dir, "rules")
    if os.path.isdir(rules_dir):
        for fn in _ordered_tree_files(rules_dir):
            with open(os.path.join(rules_dir, fn), "r", encoding="utf-8") as f:
                rules.append(_parse_rule_file(fn, f.read()))

    goals = []
    goals_dir = os.path.join(tree_dir, "goals")
    if os.path.isdir(goals_dir):
        for fn in _ordered_tree_files(goals_dir):
            with open(os.path.join(goals_dir, fn), "r", encoding="utf-8") as f:
                goals.append(_parse_goal_file(fn, f.read()))

    nogoods = []
    nogoods_path = os.path.join(tree_dir, TREE_NOGOODS_FILE)
    if os.path.exists(nogoods_path):
        with open(nogoods_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    nogoods.append(json.loads(line))

    # S3 reservation: handler files must not exist yet — fail loud, never silently
    # accept an unverified handler into production (AGENT_PLAN §2-S3).
    handlers_dir = os.path.join(tree_dir, "handlers")
    if os.path.isdir(handlers_dir):
        stray = [f for f in os.listdir(handlers_dir) if f.endswith(".sexpr")]
        if stray:
            raise ValueError(
                f"ERR_HANDLERS_UNVERIFIED: handler files not yet in the vocabulary (S3): {stray}")

    return PolicyProgram.from_dict({
        "version": int(meta.get("version", 1)),
        "domain": domain,
        "params": params,
        "macros": macros,
        "tactic_rules": rules,
        "nogoods": nogoods,
        "role_profiles": meta.get("role_profiles", {}),
        "domain_profiles": meta.get("domain_profiles", {}),
        "strategy_plan": goals,
        "provenance": meta.get("provenance", {}),
    })


def compile_authoring_tree(tree_dir: str) -> PolicyProgram:
    """Alias with the compile name (assembly is validating by construction here:
    PolicyProgram.from_dict re-parses goal/when strings lazily at evaluation)."""
    return parse_authoring_tree(tree_dir)


# ---------------------------------------------------------------------------
# PolicyProgram → tree (write)
# ---------------------------------------------------------------------------

def _write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    return path


def write_authoring_tree(program: PolicyProgram, tree_dir: str) -> list[str]:
    """Serializes the program into per-file tree form. Deterministic: stable NN_
    prefixes fix multi-file ordering (rule match order / goal plan order).
    Returns the list of files written."""
    written: list[str] = []
    os.makedirs(tree_dir, exist_ok=True)
    for sub in TREE_SUBDIRS:
        os.makedirs(os.path.join(tree_dir, sub), exist_ok=True)

    meta = {
        "version": program.version,
        "domain": program.domain,
        "provenance": program.provenance,
        "role_profiles": program.role_profiles,
        "domain_profiles": program.domain_profiles,
    }
    written.append(_write(os.path.join(tree_dir, TREE_METADATA_FILE),
                          json.dumps(meta, indent=2)))
    written.append(_write(os.path.join(tree_dir, TREE_PARAMS_FILE),
                          json.dumps({"params": program.params}, indent=2)))

    # clean the subdirs (tree = exact mirror of the program)
    for sub in TREE_SUBDIRS:
        d = os.path.join(tree_dir, sub)
        for fn in os.listdir(d):
            if fn.endswith(".sexpr"):
                os.unlink(os.path.join(d, fn))

    for i, m in enumerate(program.macros):
        text = f"(defmacro {m['name']} {m['body']})"
        written.append(_write(os.path.join(tree_dir, "macros", f"{i:02d}_{_slug(m['name'])}.sexpr"), text))

    for i, r in enumerate(program.tactic_rules):
        clauses = [f"(when {r['when']})", f"(do {r['do']})"]
        if "unless" in r:
            u = r.get("unless")
            clauses.append("(unless null)" if u is None else f"(unless {u})")
        if "note" in r:
            n = r.get("note")
            clauses.append("(note null)" if n is None
                           else f"(note {_sexpr_str(n, f'rule {i} note')})")
        if r.get("interlock"):
            clauses.append("(interlock true)")
        text = "(rule add tactic_rules " + " ".join(clauses) + ")"
        written.append(_write(os.path.join(
            tree_dir, "rules", f"{i:02d}_{_slug(r['do'])}_{i}.sexpr"), text))

    for i, g in enumerate(program.strategy_plan):
        clauses = [f"(when {g.when})", f"(until {g.until})"]
        if g.directive:
            clauses.append(f"(directive {_sexpr_str(g.directive, f'goal {g.goal} directive')})")
        if g.max_steps is not None:
            clauses.append(f"(max_steps {int(g.max_steps)})")
        text = f"(goal {g.goal} " + " ".join(clauses) + ")"
        written.append(_write(os.path.join(
            tree_dir, "goals", f"{i:02d}_{_slug(g.goal)}.sexpr"), text))

    nogoods_path = os.path.join(tree_dir, TREE_NOGOODS_FILE)
    if program.nogoods:
        with open(nogoods_path, "w", encoding="utf-8") as f:
            for ng in program.nogoods:
                f.write(json.dumps(ng, ensure_ascii=False) + "\n")
        written.append(nogoods_path)
    elif os.path.exists(nogoods_path):
        os.unlink(nogoods_path)
    return written


# ---------------------------------------------------------------------------
# Commit path: tree → compiled artifact (+ archive). Byte-equivalence enforced.
# ---------------------------------------------------------------------------

def compile_and_commit(program: PolicyProgram, tree_dir: str,
                       live_path: str | None = None, *, archive: bool = True) -> str:
    """The acceptance path's single writer: writes the tree, compiles it back,
    FAILS LOUD if the round-trip is not byte-equivalent to the candidate, then
    saves the compiled artifact to the live program path + data/compiled/ and
    archives it. Returns the compiled artifact path."""
    write_authoring_tree(program, tree_dir)
    compiled = compile_authoring_tree(tree_dir)

    if compiled.to_dict() != program.to_dict():
        raise ValueError(
            "ERR_ROUNDTRIP: authoring-tree compile is not byte-equivalent to the "
            "accepted candidate — refusing to commit (validator write authority)")

    os.makedirs(os.path.dirname(live_path) or ".", exist_ok=True)
    if live_path:
        compiled.save(live_path)
    out = os.path.join(COMPILED_DIR, f"{compiled.domain}.json")
    os.makedirs(COMPILED_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(compiled.to_dict(), f, indent=2)
    if archive:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stem = (f"policy_program_v{compiled.version}" if compiled.domain == "nethack"
                else f"policy_program_{compiled.domain}_v{compiled.version}")
        shutil.copy2(out, os.path.join(ARCHIVE_DIR, f"{stem}.json"))
    return out
