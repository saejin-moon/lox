"""
LOX-ψ Policy Layer (LOX-ψ): author evidence tools (S0, AGENT_PLAN §1.3).

The agentic author sees EVERYTHING through this closed tool set:
  query_duckdb(sql)          — read-only SQL over 6.6M+ ticks / episodes / goal_events
  wiki_search(q)             — the full NetHack 3.6.6 wiki FTS5 dump (offline)
  read_trajectory(ep, range) — flight-recorder autopsies: per-tick state + goal events
  read_env_schema()          — every action, its key, and every observation channel
  read_manifest()            — the machine-generated vocabulary (predicates/verbs/goals/params)
  read_program_tree()        — the current policy program source

Access is READ-ONLY: query_duckdb opens the DuckDB read-only AND rejects any
non-SELECT statement textually (defense in depth — the diff channel is the only
write path). Tools return markdown strings, sized for prompt injection.
"""
from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass

from lox.policy.manifest import VocabularyManifest
from lox.policy.program import PolicyProgram

DEFAULT_DB_PATH = "data/lox_telemetry.duckdb"
DEFAULT_WIKI_DB_PATH = "data/wiki_index.db"

# Output budgets (author prompt hygiene — expanded for up to 128k context window)
MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("LOX_MAX_TOOL_OUTPUT_CHARS", "32000"))
MAX_QUERY_ROWS = int(os.environ.get("LOX_MAX_QUERY_ROWS", "500"))
MAX_CELL_CHARS = int(os.environ.get("LOX_MAX_CELL_CHARS", "1000"))
MAX_TRAJECTORY_TICKS = int(os.environ.get("LOX_MAX_TRAJECTORY_TICKS", "2000"))


# ---------------------------------------------------------------------------
# Tool specs (the author-facing contract: name → positional args + doc)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    args: tuple[str, ...]          # positional arg names, all required
    desc: str


TOOL_SPECS: dict[str, ToolSpec] = {
    "query_duckdb": ToolSpec(
        "query_duckdb", ("sql",),
        "Read-only SQL over run telemetry. Tables: episodes (episode_id, run_id, seed, "
        "role, total_turns, max_depth, final_score, final_hp, is_ascended, "
        "death_message, death_category, timestamp), ticks (episode_id, step, turn, "
        "depth, dungeon_number, hp, max_hp, ac, xp_level, hunger_state, "
        "condition_bits, action_name, x, y), goal_events (episode_id, turn, depth, "
        "goal, event[activated|completed|skipped_budget|skipped_when], "
        "steps_in_goal). Only SELECT/WITH/DESCRIBE/SHOW statements."),
    "wiki_search": ToolSpec(
        "wiki_search", ("query",),
        "Full-text search of the offline NetHack 3.6.6 wiki. Returns title, BM25 "
        "snippet, and preview per hit."),
    "read_trajectory": ToolSpec(
        "read_trajectory", ("episode_id", "turn_from", "turn_to"),
        "Flight-recorder slice for one episode: per-tick hp/depth/action + goal "
        "events in the turn window. Use query_duckdb to find episode_id values "
        "(episodes.episode_id, joinable into ticks and goal_events)."),
    "read_env_schema": ToolSpec(
        "read_env_schema", (),
        "Every NetHack action (name, key, description) and every observation "
        "channel (blstats fields, condition flags)."),
    "read_manifest": ToolSpec(
        "read_manifest", (),
        "The closed policy vocabulary: predicates with signatures, tactic verbs, "
        "certified goals + owned params, tunable param leaves with bounds, live "
        "macros. The diff DSL may reference ONLY these symbols."),
    "read_program_tree": ToolSpec(
        "read_program_tree", (),
        "The current policy program: ordered strategy plan, tactic rules, macros, "
        "nogoods, tuned params, provenance."),
}


def tool_docs(domain: str = "nethack") -> str:
    """Rendered tool documentation for the author system prompt."""
    lines = []
    for spec in TOOL_SPECS.values():
        if domain != "nethack" and spec.name == "wiki_search":
            continue
        args = ", ".join(f'"{a}"' for a in spec.args)
        call = f'{spec.name}({args})' if args else f'{spec.name}()'
        lines.append(f"  {call}\n      # {spec.desc}")
    return "\n".join(lines)


def parse_tool_calls(text: str) -> list[tuple[str, list]]:
    """Parses tool calls out of an author turn. Supports Pythonic syntax
    `read_env_schema()` / `query_duckdb("SELECT ...")` and legacy `(tool ...)` forms.
    Returns (name, [str|num args]) pairs."""
    import ast
    calls: list[tuple[str, list]] = []
    tool_names = set(TOOL_SPECS.keys())

    # 1. Pythonic function call syntax: name(...)
    for name in tool_names:
        pattern = rf"(?:call:\s*|tool:\s*)?\b({name})\s*\((.*?)\)"
        for match in re.finditer(pattern, text, re.DOTALL):
            fn_name = match.group(1)
            raw_args = match.group(2).strip()
            if not raw_args:
                calls.append((fn_name, []))
                continue
            try:
                parsed = ast.parse(f"{fn_name}({raw_args})")
                call_node = parsed.body[0].value
                args = []
                for a in call_node.args:
                    if isinstance(a, ast.Constant):
                        args.append(a.value)
                    elif isinstance(a, ast.UnaryOp) and isinstance(a.operand, ast.Constant):
                        val = a.operand.value
                        args.append(-val if isinstance(a.op, ast.USub) else val)
                    else:
                        args.append(ast.unparse(a))
                for kw in call_node.keywords:
                    if isinstance(kw.value, ast.Constant):
                        args.append(kw.value.value)
                    else:
                        args.append(ast.unparse(kw.value))
                calls.append((fn_name, args))
            except Exception:
                if (raw_args.startswith('"') and raw_args.endswith('"')) or (raw_args.startswith("'") and raw_args.endswith("'")):
                    calls.append((fn_name, [raw_args[1:-1]]))

    if calls:
        return calls

    # 2. Legacy S-expression fallback: (tool <name> ...)
    from lox.policy.predicates import tokenize, _atom
    try:
        tokens = tokenize(text)
    except ValueError:
        return calls
    if "(tool" not in text and "(tool\n" not in text and "tool " not in text:
        return calls
    pos = 0
    n = len(tokens)

    def parse_one():
        nonlocal pos
        if tokens[pos] != "(":
            raise ValueError("expected '('")
        pos += 1
        head = tokens[pos]
        pos += 1
        args: list = []
        while pos < n and tokens[pos] != ")":
            if tokens[pos] == "(":
                args.append(parse_one())
            else:
                args.append(_atom(tokens[pos]))
                pos += 1
        pos += 1
        return head, args

    try:
        while pos < n:
            if tokens[pos] == "(":
                head, args = parse_one()
                if head == "tool" and args and isinstance(args[0], str):
                    calls.append((args[0], args[1:]))
            else:
                pos += 1
    except (ValueError, IndexError):
        pass
    return calls


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text)} → {limit} chars]"


# ---------------------------------------------------------------------------
# query_duckdb — read-only telemetry SQL
# ---------------------------------------------------------------------------

# Statement-level guard: whole-word forbidden keywords anywhere in the SQL.
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|COPY|ATTACH|DETACH|EXPORT|"
    r"IMPORT|INSTALL|LOAD|CALL|SET|PRAGMA|VACUUM|CHECKPOINT|BEGIN|COMMIT|"
    r"ROLLBACK|GRANT|REVOKE|MERGE|TRUNCATE)\b", re.IGNORECASE)
_ALLOWED_PREFIX = re.compile(r"^\s*(SELECT|WITH|DESCRIBE|SHOW|EXPLAIN|VALUES)\b",
                             re.IGNORECASE)


def check_sql_readonly(sql: str) -> str:
    """Returns '' when the statement is a read-only query, else an error string."""
    if not sql or not sql.strip():
        return "empty SQL statement"
    if not _ALLOWED_PREFIX.match(sql):
        return "only SELECT / WITH / DESCRIBE / SHOW / EXPLAIN statements are allowed"
    m = _FORBIDDEN_SQL.search(sql)
    if m:
        return f"write keyword {m.group(1).upper()} is forbidden (read-only tool)"
    return ""


def query_duckdb(sql: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """Executes a read-only SQL query over the telemetry DuckDB. Returns a
    markdown table (row-limited, cell-truncated) or an ERROR line."""
    err = check_sql_readonly(sql)
    if err:
        return f"ERROR: {err}"
    if not os.path.exists(db_path):
        return f"ERROR: telemetry database not found at {db_path}"
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        try:
            rows = con.execute(sql).fetchmany(MAX_QUERY_ROWS + 1)
            cols = [d for d in (con.description or [])]
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 — author SQL errors are results, not crashes
        return f"ERROR: {type(e).__name__}: {e}"
    if not rows:
        return "OK: 0 rows"
    header = [c[0] if c else f"col{i}" for i, c in enumerate(cols)]
    truncated = len(rows) > MAX_QUERY_ROWS
    rows = rows[:MAX_QUERY_ROWS]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        cells = [str(v)[:MAX_CELL_CHARS] for v in r]
        out.append("| " + " | ".join(cells) + " |")
    table = "\n".join(out)
    if truncated:
        table += f"\n[{MAX_QUERY_ROWS} row limit hit — narrow the query]"
    return _truncate(table)


# ---------------------------------------------------------------------------
# wiki_search — offline NetHack 3.6.6 wiki (FTS5)
# ---------------------------------------------------------------------------

def _load_wiki_searcher():
    """Loads scripts/wiki_search.py (argparse-free on import) via importlib."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "scripts", "wiki_search.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("lox_wiki_search", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wiki_search(query: str, top_k: int = 3, db_path: str = DEFAULT_WIKI_DB_PATH) -> str:
    """BM25 search over the offline NetHack 3.6.6 wiki dump."""
    if not query or not str(query).strip():
        return "ERROR: empty query"
    if not os.path.exists(db_path):
        return f"ERROR: wiki index not found at {db_path}"
    mod = _load_wiki_searcher()
    if mod is None:
        return "ERROR: wiki_search module unavailable"
    try:
        hits = mod.search_wiki(str(query), top_k=int(top_k), db_path=db_path)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    if not hits:
        return f"OK: no wiki hits for {query!r}"
    out = []
    for h in hits:
        tag = "EXACT" if h.get("is_exact") else f"bm25 {h.get('score', 0):.2f}"
        body = h.get("text") or h.get("preview") or ""
        out.append(f"### {h['title']} [{tag}]\n{body}")
    return _truncate("\n\n".join(out))


# ---------------------------------------------------------------------------
# read_trajectory — flight-recorder autopsies from telemetry
# ---------------------------------------------------------------------------

def read_trajectory(episode_id: str, turn_from: int = 0, turn_to: int = 0,
                    db_path: str = DEFAULT_DB_PATH) -> str:
    """Per-tick state slice + goal events for one episode's turn window.

    Ticks are RUN-LENGTH encoded (consecutive identical depth/hp/action collapse to
    one `t<a>-<b> … ×n` line) — never stride-sampled: stride can skip the few turns
    that contain the actual kill/stall transition. For whole-episode shape use the
    bundle's `episode_digests`. NOTE: per-turn MESSAGES are not persisted in
    telemetry (only episodes.death_message), so this shows state, not the prose why."""
    if not episode_id:
        return "ERROR: episode_id required"
    if not os.path.exists(db_path):
        return f"ERROR: telemetry database not found at {db_path}"
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        try:
            try:
                turn_from = int(turn_from)
                turn_to = int(turn_to)
            except (TypeError, ValueError):
                return "ERROR: turn_from/turn_to must be integers"
            if turn_to <= 0:
                turn_to = 1 << 30
            rows = con.execute(
                """SELECT turn, depth, hp, max_hp, ac, hunger_state,
                          action_name, condition_bits
                   FROM ticks WHERE episode_id = ? AND turn >= ? AND turn <= ?
                   ORDER BY step""",
                (episode_id, turn_from, turn_to),
            ).fetchall()
            goals = con.execute(
                """SELECT turn, depth, goal, event, steps_in_goal
                   FROM goal_events WHERE episode_id = ? AND turn >= ? AND turn <= ?
                   ORDER BY turn""",
                (episode_id, turn_from, turn_to),
            ).fetchall()
            try:
                events = con.execute(
                    """SELECT turn, kind, code, detail FROM events
                       WHERE episode_id = ? AND turn >= ? AND turn <= ?
                       ORDER BY turn LIMIT 80""",
                    (episode_id, turn_from, turn_to),
                ).fetchall()
            except Exception:  # noqa: BLE001 — events table may not exist yet
                events = []
            meta = con.execute(
                """SELECT death_message, max_depth, final_score, total_turns
                   FROM episodes WHERE episode_id = ?""",
                (episode_id,),
            ).fetchall()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    if not rows and not goals:
        return f"OK: no ticks or goal_events for episode {episode_id!r} in [{turn_from}, {turn_to}]"

    out = [f"## TRAJECTORY {episode_id} (turns {turn_from}–{turn_to})"]
    if meta:
        d = meta[0]
        out.append(f"outcome: death={d[0]!r} max_depth={d[1]} final_score={d[2]} "
                   f"total_turns={d[3]}")
    if rows:
        # run-length encode on material state (depth, hp, max_hp, action)
        lines: list[str] = []
        cur = (rows[0][1], rows[0][2], rows[0][3], rows[0][6])
        start_t = last_t = rows[0][0]
        count = 1
        hunger = rows[0][5]
        ac = rows[0][4]
        for r in rows[1:]:
            k = (r[1], r[2], r[3], r[6])
            if k == cur:
                count += 1
                last_t = r[0]
                hunger = r[5]
                ac = r[4]
            else:
                lines.append(f"t{start_t}-{last_t} d{cur[0]} hp{cur[1]}/{cur[2]} "
                             f"ac{ac} hunger{hunger} {cur[3]} ×{count}")
                cur = k
                start_t = last_t = r[0]
                hunger, ac = r[5], r[4]
                count = 1
        lines.append(f"t{start_t}-{last_t} d{cur[0]} hp{cur[1]}/{cur[2]} "
                     f"ac{ac} hunger{hunger} {cur[3]} ×{count}")
        if len(lines) > MAX_TRAJECTORY_TICKS:
            head = lines[:5]
            tail = lines[-(MAX_TRAJECTORY_TICKS - 5):]
            lines = head + [f"… [{len(lines) - MAX_TRAJECTORY_TICKS} RLE segments elided] …"] + tail
        out += ["RLE ticks (material-state changes only):", *lines]
    if goals:
        out.append("goal events in window:")
        for turn, depth, goal, event, steps in goals:
            out.append(f"  t={turn} d={depth}: {goal} → {event} (steps_in_goal={steps})")
    if events:
        out.append("message/anomaly events in window:")
        for turn, kind, code, detail in events:
            out.append(f"  t={turn} [{kind}] {code}: {detail}")
    return _truncate("\n".join(out))


# ---------------------------------------------------------------------------
# read_env_schema — actions + observation channels
# ---------------------------------------------------------------------------

# Curated action surface (name → (key, description)). Keys per the dispatcher's
# delta/char tables (lox/workers/dispatcher.py); descriptions are mechanism-level
# facts, not strategy (strategy is the LLM's job, mechanisms are human-owned).
NETHACK_ACTIONS: dict[str, tuple[str, str]] = {
    "STEP": ("<dir>", "move one tile (8-way: k/l/j/h/u/n/b/y); the navigation manager picks the step"),
    "WAIT": (".", "pass a turn (recover, let a timed effect expire)"),
    "SEARCH": ("s", "search adjacent walls/trap/secret doors"),
    "OPEN": ("o<dir>", "open an adjacent door"),
    "CLOSE": ("c<dir>", "close an adjacent door"),
    "KICK": ("^D<dir>", "kick an adjacent tile (locked doors, monster)"),
    "PICKUP": (",", "pick up items underfoot"),
    "DROP": ("d", "drop an inventory item"),
    "EAT": ("e", "eat (food, corpse — rot/freshness rules apply)"),
    "QUAFF": ("q", "quaff a potion (slot-qualified)"),
    "READ": ("r", "read a scroll/spellbook (slot-qualified)"),
    "ZAP": ("z", "zap a wand (slot-qualified, direction)"),
    "ZAP_WAND": ("z<dir>", "wand zap at an adjacent/directional target"),
    "FIRE": ("f", "fire the quivered missile at a target"),
    "THROW": ("t", "throw an inventory item (slot-qualified)"),
    "WIELD": ("w", "wield a weapon (slot-qualified)"),
    "WEAR": ("W", "wear armor"),
    "TAKEOFF": ("T", "take off armor"),
    "PUTON": ("P", "put on accessory (ring/amulet)"),
    "APPLY": ("a", "apply a tool (lockpick, key, skeleton key — unlock doors)"),
    "DIP": ("#dip", "dip an item (fountain Excalibur forge)"),
    "ENHANCE": ("#enhance", "train weapon/armor skills (ESC-terminated fiber)"),
    "ENGRAVE": ("E", "engrave (Elbereth, dust, identifying wands)"),
    "PRAY": ("#pray", "pray (safe-pray gate only; resets nutrition at WEAK+)"),
    "OFFER": ("#offer", "sacrifice at an altar"),
    "PAY": ("p", "pay a shopkeeper (unpaid-item debt relief)"),
    "DESCEND": (">", "take stairs down"),
    "ASCEND": ("<", "take stairs up"),
    "FIGHT": ("F<dir>", "deliberate melee attack (combat manager issues it)"),
    "TWOWEAPON": ("#twoweapon", "toggle two-weapon combat"),
    "LOOK_HERE": (":", "inspect the current tile"),
    "MORE": ("SPACE/ESC", "dismiss (X of Y)/(end) menus — AutoMoreWrapper guards this"),
}

_CONDITION_CHANNELS = (
    "STONE(1 petrify) SLIME(2) STRANGLE(4) FOOD_POISON(8) TERM_ILL(16) "
    "BLIND(32) DEAF(64) STUN(128) CONFUSED(256) HALLU(512) LEVITATING(1024) "
    "FLYING(2048) RIDING(4096)"
)


def read_env_schema(domain: str = "nethack") -> str:
    """Every action + every observation channel, as markdown."""
    lines = [f"# ENV SCHEMA: {domain}"]
    if domain == "minihack":
        lines.append("obs: Gymnasium dict — glyphs, chars (21×79), blstats, message; step cap 4000; seeded")
        lines += ["", "## ACTIONS (8-way movement + standard navigation)",
                  "  compass movement: k/l/j/h/u/n/b/y (north/east/south/west/ne/se/sw/nw)",
                  "  wait: '.'",
                  "  search: 's'",
                  "  open: 'o<dir>'",
                  "  kick: '^D<dir>'"]
        lines += ["", "## REWARD & TERMINATION",
                  "  Reward is granted on reaching the staircase ('>').",
                  "  ExploreMaze family episodes terminate on stairs or step limit."]
        return _truncate("\n".join(lines), limit=4000)
    if domain == "craftax":
        lines.append("obs: 8268-float flat observation vector")
        lines += ["", "## ACTIONS",
                  "  noop, left, right, up, down, do, sleep, ..."]
        return _truncate("\n".join(lines), limit=4000)
    lines.append("obs: NLE dict — glyphs/chars grid (21×79), message line, "
                 "blstats(26-vector), inventory parses, AutoMoreWrapper "
                 "instrumentation; episode step cap 20000; unseeded "
                 "(NetHackChallenge-v0 forbids seeding)")
    lines += ["", "## ACTIONS (name → key: mechanism description)"]
    for name, (key, desc) in NETHACK_ACTIONS.items():
        lines.append(f"  {name}: {key!r} — {desc}")
    lines += ["", "## OBSERVATION CHANNELS (blstats vector)"]
    from lox.env.blstats import BottomLineStats  # noqa: PLC0415
    for f in BottomLineStats.__dataclass_fields__.values():
        lines.append(f"  blstats.{f.name}")
    lines.append(f"  condition_bits mask: {_CONDITION_CHANNELS}")
    # S1: the behavior-primitive manifest (plan vocabulary, S3 handler ladder)
    from lox.policy.primitives import render_primitives  # noqa: PLC0415
    lines += ["", render_primitives()]
    # the schema is a complete reference (actions + obs + primitives), so it gets a
    # larger budget than generic tool results
    return _truncate("\n".join(lines), limit=16000)


# ---------------------------------------------------------------------------
# read_manifest / read_program_tree — the vocabulary and the program source
# ---------------------------------------------------------------------------

def read_manifest(manifest: VocabularyManifest) -> str:
    return _truncate(manifest.render())


def read_program_tree(program: PolicyProgram) -> str:
    lines = [f"# PROGRAM v{program.version} ({program.domain})"]
    p = program.provenance or {}
    lines.append(f"provenance: author={p.get('author', '?')} parent=v{p.get('parent_version', '?')} "
                 f"reason={p.get('reason', '')!r}")
    lines += ["", "## strategy_plan (ordered; first active goal wins)"]
    for i, g in enumerate(program.strategy_plan):
        line = f"  [{i}] {g.goal}: when={g.when} until={g.until}"
        if g.directive:
            line += f" directive={g.directive}"
        if g.max_steps is not None:
            line += f" max_steps={g.max_steps}"
        lines.append(line)
    lines += ["", "## tactic_rules (first match wins; engine-evaluated)"]
    if program.tactic_rules:
        for i, r in enumerate(program.tactic_rules):
            inter = " [INTERLOCK]" if r.get("interlock") else ""
            lines.append(f"  [{i}]{inter} when={r['when']} do={r['do']}"
                         + (f" unless={r['unless']}" if r.get("unless") else "")
                         + (f" note={r['note']!r}" if r.get("note") else ""))
    else:
        lines.append("  (none)")
    lines += ["", "## macros (parameterless sugar over predicates)"]
    if program.macros:
        for m in program.macros:
            lines.append(f"  {m['name']}: {m['body']}")
    else:
        lines.append("  (none)")
    lines += ["", "## nogoods (cross-episode CDCL negatives)"]
    if program.nogoods:
        for ng in program.nogoods:
            lines.append(f"  when={ng.get('when')} cause={ng.get('cause')!r}"
                         + (f" forbid={ng.get('forbid')!r}" if ng.get("forbid") else ""))
    else:
        lines.append("  (none)")
    lines += ["", "## tuned params (dotted-path overlay)"]
    if program.params:
        for k in sorted(program.params):
            lines.append(f"  {k} = {program.params[k]}")
    else:
        lines.append("  (all defaults)")
    return _truncate("\n".join(lines))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_tool_call(name: str, args: list, *, manifest: VocabularyManifest | None = None,
                      program: PolicyProgram | None = None,
                      db_path: str = DEFAULT_DB_PATH,
                      wiki_db_path: str = DEFAULT_WIKI_DB_PATH) -> str:
    """Executes one parsed tool call; returns the result string (never raises).
    Unknown tools / wrong arity are reported as ERROR results."""
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return f"ERROR: unknown tool {name!r}; available: {', '.join(TOOL_SPECS)}"
    if len(args) != len(spec.args):
        return (f"ERROR: tool {name} takes {len(spec.args)} arg(s) ({', '.join(spec.args)}), "
                f"got {len(args)}")
    try:
        domain = manifest.domain if manifest else (program.domain if program else "nethack")
        if name == "query_duckdb":
            return query_duckdb(str(args[0]), db_path=db_path)
        if name == "wiki_search":
            if domain != "nethack":
                return f"ERROR: wiki_search is only available for NetHack (current domain: {domain})"
            return wiki_search(str(args[0]), db_path=wiki_db_path)
        if name == "read_trajectory":
            return read_trajectory(str(args[0]), args[1], args[2], db_path=db_path)
        if name == "read_env_schema":
            return read_env_schema(domain=domain)
        if name == "read_manifest":
            if manifest is None:
                return "ERROR: manifest not loaded"
            return read_manifest(manifest)
        if name == "read_program_tree":
            if program is None:
                return "ERROR: program not loaded"
            return read_program_tree(program)
    except Exception as e:  # noqa: BLE001 — tool crashes become results
        return f"ERROR: {type(e).__name__}: {e}"
    return f"ERROR: tool {name!r} not dispatched"
