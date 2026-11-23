"""
Multi-Table DuckDB Schema for LOX ("Hard Code Nothing").
Consolidates:
- `batches`: Generation batch metadata & statistical verdicts
- `runs`: Every recorded episode with summary digest, game log, overview stats, `is_outlier_good` marker, and 4KB replay bytes
- `ticks`: Granular tick-by-tick records linked to `run_id`
- `function_stats`: Per-function call/turn telemetry & Ochiai/Liblit correlation with good/bad runs
- `policies`: Versioned `policy.py` and `macros.py` checkpoints
- `experiments`: Lab notebook of hypotheses and statistical verdicts
- `expert_runs` & `expert_ticks`: Consolidated human/bot successful NetHack runs using the exact same summary/log/stats structure as `runs` and `ticks`
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time

import duckdb

_LOG = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/lox.duckdb"
_DB_CONN_LOCK = threading.RLock()
_DB_INIT_LOCK = threading.Lock()
_INITIALIZED_DBS: set[str] = set()

# Canonical SQL predicate filtering out micro-shard & non-champion peer/pilot runs from champion aggregations
NON_SHARD_RUNS_SQL = (
    "COALESCE(is_shard, FALSE) = FALSE "
    "AND COALESCE(run_id, '') NOT LIKE 'shard_%' "
    "AND COALESCE(run_kind, 'champion') = 'champion'"
)


def connect_db(
    db_path: str = DEFAULT_DB_PATH,
    read_only: bool = False,
    retries: int = 8,
) -> duckdb.DuckDBPyConnection:
    """
    Connects to DuckDB with exponential backoff while honoring `read_only=True`.
    When this process already initialized `db_path` in read-write mode (i.e. inside
    an active campaign writer process with concurrent peer query/write threads),
    opens a read-write MVCC connection so concurrent `query_db` reads and
    `save_runs_and_ticks` writes never collide on DuckDB's per-process config rule.
    """
    abs_path = os.path.abspath(db_path)
    if not read_only:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    elif not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB database does not exist: {db_path}")

    effective_ro = bool(read_only) and (abs_path not in _INITIALIZED_DBS)

    delay = 0.05
    last_err: Exception | None = None
    for attempt in range(retries):
        with _DB_CONN_LOCK:
            try:
                if effective_ro:
                    try:
                        return duckdb.connect(db_path, read_only=True)
                    except Exception as ro_exc:
                        if "different configuration" in str(ro_exc).lower():
                            return duckdb.connect(db_path, read_only=False)
                        raise
                return duckdb.connect(db_path, read_only=False)
            except Exception as exc:
                last_err = exc
        if attempt < retries - 1:
            time.sleep(delay)
            delay = min(1.0, delay * 2.0)
    raise RuntimeError(f"Failed to connect to DuckDB at {db_path}: {last_err}")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    mode VARCHAR,
    parent_policy_id VARCHAR,
    candidate_policy_id VARCHAR,
    target_functions VARCHAR,
    episodes_recorded INTEGER,
    episodes_discarded INTEGER,
    avg_depth DOUBLE,
    p90_depth DOUBLE,
    max_depth INTEGER,
    avg_score DOUBLE,
    max_score INTEGER,
    avg_turns DOUBLE,
    outlier_count INTEGER,
    hypothesis_passed BOOLEAN,
    wilcoxon_p DOUBLE,
    ci_lower DOUBLE,
    ci_upper DOUBLE,
    seeds_evaluated INTEGER DEFAULT 0,
    is_early_stopped BOOLEAN DEFAULT FALSE,
    pilot_best_delta DOUBLE DEFAULT 0.0,
    total_cost_usd DOUBLE DEFAULT 0.0,
    total_llm_tokens BIGINT DEFAULT 0,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    batch_id VARCHAR,
    generation INTEGER,
    seed BIGINT,
    role VARCHAR,
    depth INTEGER,
    max_depth INTEGER,
    score INTEGER,
    turns INTEGER,
    steps INTEGER,
    xp_level INTEGER,
    final_hp INTEGER,
    max_hp INTEGER,
    final_ac INTEGER,
    gold INTEGER,
    end_reason VARCHAR,
    death_message VARCHAR,
    is_outlier_good BOOLEAN,
    outlier_z_score DOUBLE,
    batch_percentile DOUBLE,
    summary_digest TEXT,
    game_log TEXT,
    milestones_json VARCHAR,
    triggered_functions VARCHAR,
    last_active_function VARCHAR,
    action_bytes BLOB,
    wall_ms DOUBLE,
    auto_esc_count INTEGER DEFAULT 0,
    subgen_preemptions INTEGER DEFAULT 0,
    policy_exceptions INTEGER DEFAULT 0,
    unique_tiles INTEGER DEFAULT 0,
    pct_wait DOUBLE DEFAULT 0.0,
    pct_search DOUBLE DEFAULT 0.0,
    starved_with_food BOOLEAN DEFAULT FALSE,
    died_unarmored BOOLEAN DEFAULT FALSE,
    worn_oprop_count INTEGER DEFAULT 0,
    dungeons_visited INTEGER DEFAULT 1,
    achievements INTEGER DEFAULT 0,
    is_shard BOOLEAN DEFAULT FALSE,
    shard_id VARCHAR DEFAULT '',
    run_kind VARCHAR DEFAULT 'champion',
    peer_id VARCHAR DEFAULT '',
    attempt_num INTEGER DEFAULT 0,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticks (
    run_id VARCHAR,
    turn INTEGER,
    step_idx INTEGER,
    depth INTEGER,
    dungeon_num INTEGER,
    hp INTEGER,
    max_hp INTEGER,
    energy INTEGER,
    ac INTEGER,
    xp_level INTEGER,
    hunger_state INTEGER,
    condition_mask INTEGER,
    y INTEGER,
    x INTEGER,
    action_name VARCHAR,
    raw_key INTEGER,
    active_function VARCHAR,
    byte_offset INTEGER DEFAULT 0,
    message VARCHAR,
    visible_monsters VARCHAR,
    inventory_summary VARCHAR
);

CREATE TABLE IF NOT EXISTS function_stats (
    batch_id VARCHAR,
    generation INTEGER,
    function_name VARCHAR,
    episodes_triggered INTEGER,
    trigger_rate DOUBLE,
    total_calls BIGINT,
    total_turns_active BIGINT,
    avg_depth_when_triggered DOUBLE,
    death_rate_while_active DOUBLE,
    ochiai_good_run_score DOUBLE,
    liblit_failure_score DOUBLE,
    stagnation_turns BIGINT,
    stagnation_share DOUBLE,
    livelock_score DOUBLE
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    mode VARCHAR,
    policy_code TEXT,
    macros_code TEXT,
    skills_json TEXT DEFAULT '{}',
    params_json VARCHAR,
    avg_depth DOUBLE,
    max_depth INTEGER,
    is_promoted BOOLEAN,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    batch_id VARCHAR,
    mode VARCHAR,
    target_functions VARCHAR,
    hypothesis_yaml TEXT,
    observation_from_db TEXT,
    mechanism TEXT,
    passed BOOLEAN,
    mean_delta DOUBLE,
    ci_lower DOUBLE,
    ci_upper DOUBLE,
    wilcoxon_p DOUBLE,
    wins INTEGER,
    losses INTEGER,
    ties INTEGER,
    seeds_evaluated INTEGER DEFAULT 0,
    is_early_stopped BOOLEAN DEFAULT FALSE,
    pilot_best_delta DOUBLE DEFAULT 0.0,
    total_cost_usd DOUBLE DEFAULT 0.0,
    verdict_reason VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS discoveries (
    discovery_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    batch_id VARCHAR,
    mode VARCHAR,
    target_functions VARCHAR,
    binding_constraint VARCHAR,
    verified_c_rule VARCHAR,
    mechanism VARCHAR,
    mean_delta DOUBLE,
    avg_depth DOUBLE,
    p90_depth DOUBLE,
    max_depth INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS expert_runs (
    run_id VARCHAR PRIMARY KEY,
    source VARCHAR,
    player VARCHAR,
    role VARCHAR,
    depth INTEGER,
    max_depth INTEGER,
    score BIGINT,
    turns INTEGER,
    xp_level INTEGER,
    final_hp INTEGER,
    max_hp INTEGER,
    final_ac INTEGER,
    gold INTEGER,
    end_reason VARCHAR,
    death_message VARCHAR,
    summary_digest TEXT,
    game_log TEXT,
    milestones_json VARCHAR,
    dumplog_text TEXT
);

CREATE TABLE IF NOT EXISTS expert_ticks (
    run_id VARCHAR,
    turn INTEGER,
    step_idx INTEGER,
    depth INTEGER,
    dungeon_num INTEGER,
    hp INTEGER,
    max_hp INTEGER,
    ac INTEGER,
    xp_level INTEGER,
    hunger_state INTEGER,
    y INTEGER,
    x INTEGER,
    action_name VARCHAR,
    raw_key INTEGER,
    message VARCHAR,
    screen_ascii TEXT
);

CREATE TABLE IF NOT EXISTS synthesis_logs (
    log_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    batch_id VARCHAR,
    mode VARCHAR,
    turn_index INTEGER,
    turn_type VARCHAR,
    repair_idx INTEGER,
    model VARCHAR,
    thinking_content TEXT,
    message_content TEXT,
    tool_calls_json TEXT,
    tool_results_json TEXT,
    candidate_code TEXT,
    skill_name VARCHAR,
    skill_code TEXT,
    hypothesis_yaml TEXT,
    status VARCHAR,
    error_diagnostic TEXT,
    prompt_sent TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    agent_role VARCHAR DEFAULT 'author',
    cost_usd DOUBLE DEFAULT 0.0,
    wall_ms DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_cost_rates (
    model VARCHAR PRIMARY KEY,
    prompt_usd_per_m DOUBLE,
    completion_usd_per_m DOUBLE,
    reasoning_usd_per_m DOUBLE
);

CREATE TABLE IF NOT EXISTS archive_policies (
    cell_id VARCHAR PRIMARY KEY,
    depth_bin INTEGER,
    caution_bin INTEGER,
    resource_bin INTEGER,
    fitness DOUBLE,
    policy_code TEXT,
    skills_json TEXT DEFAULT '{}',
    batch_id VARCHAR,
    generation INTEGER,
    metrics_json VARCHAR,
    selections INTEGER DEFAULT 0,
    promotions INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS theories (
    doc_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    agent_role VARCHAR,
    doc_type VARCHAR,
    title VARCHAR,
    content TEXT,
    tags VARCHAR,
    verified_c_rule VARCHAR,
    metrics_json TEXT,
    created_at DOUBLE
);

CREATE TABLE IF NOT EXISTS tournament_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    peer_id VARCHAR,
    deficit_id VARCHAR,
    candidate_fp VARCHAR,
    promoted BOOLEAN,
    sandbox_passed BOOLEAN,
    sandbox_error TEXT,
    avg_depth DOUBLE,
    max_depth INTEGER,
    avg_score DOUBLE,
    avg_ac DOUBLE,
    avg_xp_level DOUBLE,
    policy_exceptions INTEGER,
    seeds_gained_json TEXT,
    seeds_lost_json TEXT,
    top_tracebacks_json TEXT,
    holdout_avg_depth DOUBLE,
    holdout_iqm_delta DOUBLE,
    holdout_rejected BOOLEAN DEFAULT FALSE,
    iqm_delta DOUBLE,
    wilcoxon_p DOUBLE,
    wins INTEGER,
    losses INTEGER,
    reject_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stepping_stones (
    stone_id VARCHAR PRIMARY KEY,
    generation INTEGER,
    parent_policy_id VARCHAR,
    source_peer_id VARCHAR,
    source_run_kind VARCHAR,
    source_batch_id VARCHAR,
    revision_count INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'active',
    policy_code TEXT,
    skills_json TEXT DEFAULT '{}',
    mechanism TEXT,
    target_functions VARCHAR,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    iqm_delta DOUBLE DEFAULT 0.0,
    winsor_delta DOUBLE DEFAULT 0.0,
    wilcoxon_p DOUBLE DEFAULT 1.0,
    sign_p DOUBLE DEFAULT 1.0,
    reject_reason TEXT DEFAULT '',
    seeds_gained_json TEXT DEFAULT '[]',
    seeds_lost_json TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE VIEW champion_runs AS
SELECT * FROM runs
WHERE batch_id = (
    SELECT batch_id FROM batches
    WHERE hypothesis_passed = TRUE
      AND mode NOT IN (
          'PILOT',
          'BASELINE_BACKFILL',
          'MACRO_WRITE',
          'MICRO_PILOT',
          'MICRO_PILOT_BASELINE',
          'INCUBATE',
          'INCUBATE_BASELINE',
          'SELECT',
          'SELECT_INCUMBENT',
          'CONFIRM_CHALLENGER'
      )
    ORDER BY generation DESC, created_at DESC
    LIMIT 1
)
AND COALESCE(run_kind, 'champion') = 'champion'
AND COALESCE(is_shard, FALSE) = FALSE
AND run_id NOT LIKE 'shard_%';

CREATE OR REPLACE VIEW champion_function_stats AS
SELECT * FROM function_stats
WHERE batch_id = (
    SELECT batch_id FROM batches
    WHERE hypothesis_passed = TRUE
      AND mode NOT IN (
          'PILOT',
          'BASELINE_BACKFILL',
          'MACRO_WRITE',
          'MICRO_PILOT',
          'MICRO_PILOT_BASELINE',
          'INCUBATE',
          'INCUBATE_BASELINE',
          'SELECT',
          'SELECT_INCUMBENT',
          'CONFIRM_CHALLENGER'
      )
    ORDER BY generation DESC, created_at DESC
    LIMIT 1
);

CREATE OR REPLACE VIEW champion_ticks AS
SELECT * FROM ticks
WHERE run_id IN (SELECT run_id FROM champion_runs);

CREATE OR REPLACE VIEW latest_pilot_runs AS
SELECT * FROM runs
WHERE batch_id = (
    SELECT batch_id FROM batches
    WHERE mode = 'PILOT'
    ORDER BY created_at DESC
    LIMIT 1
)
AND COALESCE(is_shard, FALSE) = FALSE
AND COALESCE(run_id, '') NOT LIKE 'shard_%';

CREATE OR REPLACE VIEW latest_pilot_ticks AS
SELECT * FROM ticks
WHERE run_id IN (SELECT run_id FROM latest_pilot_runs);

CREATE OR REPLACE VIEW peer_runs AS
SELECT * FROM runs
WHERE COALESCE(run_kind, 'champion') != 'champion';

CREATE OR REPLACE VIEW peer_ticks AS
SELECT
    t.*,
    r.generation,
    r.peer_id,
    r.run_kind,
    r.attempt_num,
    r.seed,
    r.batch_id
FROM ticks t
JOIN runs r ON t.run_id = r.run_id
WHERE COALESCE(r.run_kind, 'champion') != 'champion';

CREATE OR REPLACE VIEW all_ticks AS
SELECT
    t.*,
    r.generation,
    r.peer_id,
    r.run_kind,
    r.attempt_num,
    r.seed,
    r.batch_id
FROM ticks t
JOIN runs r ON t.run_id = r.run_id;
"""


DEFAULT_LLM_COST_RATES: dict[str, dict[str, float]] = {
    "z-ai/glm-5.3-flash:nitro": {"prompt": 0.06, "completion": 0.40, "reasoning": 0.40},
    "z-ai/glm-5.3-flash": {"prompt": 0.06, "completion": 0.40, "reasoning": 0.40},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60, "reasoning": 0.60},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00, "reasoning": 10.00},
    "google/gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40, "reasoning": 0.40},
    "anthropic/claude-3.5-haiku": {
        "prompt": 0.80,
        "completion": 4.00,
        "reasoning": 4.00,
    },
    "anthropic/claude-3.5-sonnet": {
        "prompt": 3.00,
        "completion": 15.00,
        "reasoning": 15.00,
    },
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28, "reasoning": 0.28},
}


def compute_llm_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    """Computes exact USD cost of an LLM call from per-million token pricing rates."""
    m_low = (model or "").lower().strip()
    rates = DEFAULT_LLM_COST_RATES.get(m_low)
    if not rates:
        for k, v in DEFAULT_LLM_COST_RATES.items():
            if k in m_low or m_low in k:
                rates = v
                break
    if not rates:
        rates = {"prompt": 0.15, "completion": 0.60, "reasoning": 0.60}
    p_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
    c_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
    r_cost = (reasoning_tokens / 1_000_000.0) * rates["reasoning"]
    return float(p_cost + c_cost + r_cost)


def init_db(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Initializes all DuckDB tables for LOX and automatically migrates outdated schemas."""
    abs_path = os.path.abspath(db_path)
    with _DB_INIT_LOCK:
        if not os.path.exists(abs_path):
            _INITIALIZED_DBS.discard(abs_path)
        con = connect_db(db_path, read_only=False)
        if abs_path in _INITIALIZED_DBS:
            return con
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "ticks" in tables:
                cols = [
                    r[1] for r in con.execute("PRAGMA table_info('ticks')").fetchall()
                ]
                if "run_id" not in cols or len(cols) < 20:
                    con.execute("DROP TABLE ticks")
                elif "byte_offset" not in cols:
                    try:
                        con.execute(
                            "ALTER TABLE ticks ADD COLUMN byte_offset INTEGER DEFAULT 0"
                        )
                    except Exception:
                        pass
            if "runs" in tables:
                cols_runs = [
                    r[1] for r in con.execute("PRAGMA table_info('runs')").fetchall()
                ]
                for c_name, c_typ in [
                    ("is_shard", "BOOLEAN DEFAULT FALSE"),
                    ("shard_id", "VARCHAR DEFAULT ''"),
                    ("auto_esc_count", "INTEGER DEFAULT 0"),
                    ("subgen_preemptions", "INTEGER DEFAULT 0"),
                    ("policy_exceptions", "INTEGER DEFAULT 0"),
                    ("unique_tiles", "INTEGER DEFAULT 0"),
                    ("pct_wait", "DOUBLE DEFAULT 0.0"),
                    ("pct_search", "DOUBLE DEFAULT 0.0"),
                    ("starved_with_food", "BOOLEAN DEFAULT FALSE"),
                    ("died_unarmored", "BOOLEAN DEFAULT FALSE"),
                    ("worn_oprop_count", "INTEGER DEFAULT 0"),
                    ("dungeons_visited", "INTEGER DEFAULT 1"),
                    ("achievements", "INTEGER DEFAULT 0"),
                    ("run_kind", "VARCHAR DEFAULT 'champion'"),
                    ("peer_id", "VARCHAR DEFAULT ''"),
                    ("attempt_num", "INTEGER DEFAULT 0"),
                    ("created_at", "TIMESTAMP"),
                ]:
                    if c_name not in cols_runs:
                        try:
                            con.execute(f"ALTER TABLE runs ADD COLUMN {c_name} {c_typ}")
                        except Exception:
                            pass
            if "policies" in tables:
                cols_pol = [
                    r[1]
                    for r in con.execute("PRAGMA table_info('policies')").fetchall()
                ]
                if "skills_json" not in cols_pol:
                    try:
                        con.execute(
                            "ALTER TABLE policies ADD COLUMN skills_json TEXT DEFAULT '{}'"
                        )
                    except Exception:
                        pass
            if "batches" in tables:
                cols_b = [
                    r[1] for r in con.execute("PRAGMA table_info('batches')").fetchall()
                ]
                for c_name, c_typ in [
                    ("seeds_evaluated", "INTEGER DEFAULT 0"),
                    ("is_early_stopped", "BOOLEAN DEFAULT FALSE"),
                    ("pilot_best_delta", "DOUBLE DEFAULT 0.0"),
                    ("total_cost_usd", "DOUBLE DEFAULT 0.0"),
                    ("total_llm_tokens", "BIGINT DEFAULT 0"),
                ]:
                    if c_name not in cols_b:
                        try:
                            con.execute(
                                f"ALTER TABLE batches ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
            if "experiments" in tables:
                cols_exp = [
                    r[1]
                    for r in con.execute("PRAGMA table_info('experiments')").fetchall()
                ]
                for c_name, c_typ in [
                    ("seeds_evaluated", "INTEGER DEFAULT 0"),
                    ("is_early_stopped", "BOOLEAN DEFAULT FALSE"),
                    ("pilot_best_delta", "DOUBLE DEFAULT 0.0"),
                    ("total_cost_usd", "DOUBLE DEFAULT 0.0"),
                ]:
                    if c_name not in cols_exp:
                        try:
                            con.execute(
                                f"ALTER TABLE experiments ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
            if "synthesis_logs" in tables:
                cols_syn = [
                    r[1]
                    for r in con.execute(
                        "PRAGMA table_info('synthesis_logs')"
                    ).fetchall()
                ]
                for c_name, c_typ in [
                    ("agent_role", "VARCHAR DEFAULT 'author'"),
                    ("cost_usd", "DOUBLE DEFAULT 0.0"),
                ]:
                    if c_name not in cols_syn:
                        try:
                            con.execute(
                                f"ALTER TABLE synthesis_logs ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
            if "function_stats" in tables:
                cols_fs = [
                    r[1]
                    for r in con.execute(
                        "PRAGMA table_info('function_stats')"
                    ).fetchall()
                ]
                for c_name, c_typ in [
                    ("stagnation_turns", "BIGINT"),
                    ("stagnation_share", "DOUBLE"),
                    ("livelock_score", "DOUBLE"),
                ]:
                    if c_name not in cols_fs:
                        try:
                            con.execute(
                                f"ALTER TABLE function_stats ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
            if "archive_policies" in tables:
                cols_ap = [
                    r[1]
                    for r in con.execute(
                        "PRAGMA table_info('archive_policies')"
                    ).fetchall()
                ]
                for c_name, c_typ in [
                    ("skills_json", "TEXT DEFAULT '{}'"),
                    ("selections", "INTEGER DEFAULT 0"),
                    ("promotions", "INTEGER DEFAULT 0"),
                ]:
                    if c_name not in cols_ap:
                        try:
                            con.execute(
                                f"ALTER TABLE archive_policies ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
            if "tournament_attempts" in tables:
                cols_ta = [
                    r[1]
                    for r in con.execute(
                        "PRAGMA table_info('tournament_attempts')"
                    ).fetchall()
                ]
                for c_name, c_typ in [
                    ("holdout_avg_depth", "DOUBLE"),
                    ("holdout_iqm_delta", "DOUBLE"),
                    ("holdout_rejected", "BOOLEAN DEFAULT FALSE"),
                    ("iqm_delta", "DOUBLE"),
                    ("wilcoxon_p", "DOUBLE"),
                    ("wins", "INTEGER"),
                    ("losses", "INTEGER"),
                    ("reject_reason", "TEXT"),
                ]:
                    if c_name not in cols_ta:
                        try:
                            con.execute(
                                f"ALTER TABLE tournament_attempts ADD COLUMN {c_name} {c_typ}"
                            )
                        except Exception:
                            pass
        except Exception as mig_exc:
            _LOG.warning("DuckDB schema migration notice for %s: %s", db_path, mig_exc)
        con.execute(SCHEMA_SQL)
        try:
            for m_name, rates in DEFAULT_LLM_COST_RATES.items():
                con.execute(
                    """
                    INSERT OR IGNORE INTO llm_cost_rates (model, prompt_usd_per_m, completion_usd_per_m, reasoning_usd_per_m)
                    VALUES (?, ?, ?, ?)
                    """,
                    [m_name, rates["prompt"], rates["completion"], rates["reasoning"]],
                )
        except Exception as rate_exc:
            _LOG.warning("Failed seeding llm_cost_rates: %s", rate_exc)
        _INITIALIZED_DBS.add(abs_path)
        return con


def _parse_schema_tables(schema_sql: str) -> dict[str, list[tuple[str, str]]]:
    """Programmatically extracts `{table_name: [(col, type), ...]}` directly from `SCHEMA_SQL`."""
    tables: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(
        r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\);",
        schema_sql,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        t_name = m.group(1)
        cols: list[tuple[str, str]] = []
        for raw_ln in m.group(2).splitlines():
            ln = raw_ln.strip().rstrip(",")
            if not ln or ln.upper().startswith(
                ("PRIMARY KEY", "UNIQUE", "FOREIGN KEY", "CHECK")
            ):
                continue
            parts = ln.split()
            if len(parts) >= 2:
                cols.append((parts[0], parts[1].upper()))
        tables[t_name] = cols
    return tables


def get_schema_markdown(
    db_path: str = DEFAULT_DB_PATH,
    is_swarm: bool = False,
) -> str:
    """Programmatically generates a DuckDB schema reference from `SCHEMA_SQL` (and live columns if present)."""
    parsed_tables = _parse_schema_tables(SCHEMA_SQL)
    if db_path and os.path.exists(db_path):
        try:
            con = connect_db(db_path, read_only=True)
            try:
                for t_row in con.execute("SHOW TABLES").fetchall():
                    t_nm = str(t_row[0])
                    if t_nm in parsed_tables:
                        live_cols = [
                            (str(r[1]), str(r[2]))
                            for r in con.execute(
                                f"PRAGMA table_info('{t_nm}')"
                            ).fetchall()
                        ]
                        if live_cols:
                            parsed_tables[t_nm] = live_cols
            finally:
                con.close()
        except Exception:
            pass

    def _fmt_cols(t_name: str, drop: set[str] | None = None) -> str:
        cols = parsed_tables.get(t_name, [])
        skip = drop or set()
        return ", ".join(f"`{c}`" for c, _ in cols if c not in skip)

    def _fmt_typed(t_name: str) -> str:
        cols = parsed_tables.get(t_name, [])
        return ", ".join(f"`{c}` ({tp})" for c, tp in cols)

    if is_swarm:
        return "\n".join(
            [
                "### DuckDB Schema (`query_db(sql)`)",
                "- Views: `champion_runs` / `champion_function_stats` / `champion_ticks` (latest promoted batch), `peer_runs` / `peer_ticks` (all non-champion peer micro-pilot/incubate/select/confirm_challenger runs+ticks with `generation`, `peer_id`, `run_kind`, `attempt_num`, `seed`), `all_ticks` (unified ticks across ALL runs joined with run metadata).",
                f"- `runs`: {_fmt_cols('runs', {'action_bytes'})}. *(Notes: `run_kind` distinguishes `'champion'`, `'micro_pilot'`, `'micro_pilot_baseline'`, `'incubate'`, `'select'`, `'confirm_challenger'`; `hunger_state` on `ticks`; `seed` on `runs`; `pct_wait`/`pct_search` in 0..100; filter shards via `NOT is_shard`.)*",
                f"- `ticks`: {_fmt_cols('ticks')}. *(Unified lossless ticks across 100% of runs, joined via `run_id`.)*",
                f"- `batches`: {_fmt_cols('batches')}.",
                f"- `policies`: {_fmt_cols('policies')}.",
                f"- `function_stats`: {_fmt_cols('function_stats')}.",
                f"- `theories`: {_fmt_cols('theories')}.",
                f"- `tournament_attempts`: {_fmt_cols('tournament_attempts')}.",
                f"- `stepping_stones`: {_fmt_cols('stepping_stones', {'policy_code', 'skills_json'})}.",
                f"- `expert_runs`: {_fmt_cols('expert_runs')}.",
                f"- `expert_ticks`: {_fmt_cols('expert_ticks')}.",
            ]
        )

    lines = [
        "### DuckDB Database Schema (`query_db(sql)`):",
        "- **Canonical Champion & Pilot Views**: `champion_runs`, `champion_function_stats`, `champion_ticks`, `latest_pilot_runs`, `latest_pilot_ticks`.",
    ]
    for tbl in parsed_tables:
        if tbl == "runs":
            lines.append(
                f"- **`runs`**: {_fmt_typed('runs')}. *(Notes: `hunger_state` is on `ticks`; `stall_count` & `unworn_armor_in_pack` inside `milestones_json`; `pct_wait`/`pct_search` in 0..100.)*"
            )
        else:
            lines.append(f"- **`{tbl}`**: {_fmt_typed(tbl)}")
    return "\n".join(lines)
