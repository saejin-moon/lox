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

import os
import time

import duckdb

DEFAULT_DB_PATH = "data/lox.duckdb"


def connect_db(
    db_path: str = DEFAULT_DB_PATH,
    read_only: bool = False,
    retries: int = 5,
) -> duckdb.DuckDBPyConnection:
    """Connects to DuckDB with exponential backoff for transient lock contention."""
    if not read_only:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    delay = 0.1
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(db_path, read_only=read_only)
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2.0
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
    is_shard BOOLEAN DEFAULT FALSE,
    shard_id VARCHAR DEFAULT ''
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
    batch_id VARCHAR,
    generation INTEGER,
    metrics_json VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE VIEW champion_runs AS
SELECT * FROM runs
WHERE batch_id = (
    SELECT batch_id FROM batches
    WHERE hypothesis_passed = TRUE
      AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
    ORDER BY generation DESC, created_at DESC
    LIMIT 1
)
AND run_id NOT LIKE 'shard_%';

CREATE OR REPLACE VIEW champion_function_stats AS
SELECT * FROM function_stats
WHERE batch_id = (
    SELECT batch_id FROM batches
    WHERE hypothesis_passed = TRUE
      AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
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
);

CREATE OR REPLACE VIEW latest_pilot_ticks AS
SELECT * FROM ticks
WHERE run_id IN (SELECT run_id FROM latest_pilot_runs);
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
    con = connect_db(db_path, read_only=False)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "ticks" in tables:
            cols = [r[1] for r in con.execute("PRAGMA table_info('ticks')").fetchall()]
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
            ]:
                if c_name not in cols_runs:
                    try:
                        con.execute(f"ALTER TABLE runs ADD COLUMN {c_name} {c_typ}")
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
                        con.execute(f"ALTER TABLE batches ADD COLUMN {c_name} {c_typ}")
                    except Exception:
                        pass
        if "experiments" in tables:
            cols_exp = [
                r[1] for r in con.execute("PRAGMA table_info('experiments')").fetchall()
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
                for r in con.execute("PRAGMA table_info('synthesis_logs')").fetchall()
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
                for r in con.execute("PRAGMA table_info('function_stats')").fetchall()
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
    except Exception:
        pass
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
    except Exception:
        pass
    return con


def get_schema_markdown(db_path: str = DEFAULT_DB_PATH) -> str:
    """Returns a concise, exact DuckDB schema reference for the Author Agent's system prompt."""
    return """### DuckDB Database Schema (`query_db(sql)`):
- **Canonical Champion & Pilot Views**:
  - `champion_runs`: All episodes from the latest promoted policy batch (`hypothesis_passed = TRUE`).
  - `champion_function_stats`: Function stats from the latest promoted policy batch.
  - `champion_ticks`: Tick-by-tick logs linked to `champion_runs`.
  - `latest_pilot_runs`: All episodes from the most recent pilot batch.
- **`runs`** (Every evaluated episode):
  `run_id` (VARCHAR), `batch_id` (VARCHAR), `generation` (INT), `seed` (BIGINT), `role` (VARCHAR), `depth` (INT), `max_depth` (INT), `score` (INT), `turns` (INT), `steps` (INT), `xp_level` (INT), `final_hp` (INT), `max_hp` (INT), `final_ac` (INT), `gold` (INT), `end_reason` (VARCHAR), `death_message` (VARCHAR), **`is_outlier_good` (BOOLEAN: True for statistical outlier good runs in that batch)**, `outlier_z_score` (DOUBLE), `batch_percentile` (DOUBLE), **`summary_digest` (TEXT: RLE timeline + inflection windows + stalls)**, **`game_log` (TEXT: chronological game messages)**, `milestones_json` (VARCHAR), `triggered_functions` (VARCHAR: JSON dict of function call counts), `last_active_function` (VARCHAR), **`auto_esc_count` (INT: times harness forced auto-ESC for unhandled prompt)**, **`subgen_preemptions` (INT: generator subroutines aborted by emergency threat)**, **`policy_exceptions` (INT: non-fatal policy exceptions caught)**, **`unique_tiles` (INT: distinct floor tiles visited)**, **`pct_wait` (DOUBLE: % of steps spent waiting/resting)**, **`pct_search` (DOUBLE: % of steps spent searching 's')**, **`starved_with_food` (BOOLEAN: True if hero died of starvation while carrying food in pack)**, **`died_unarmored` (BOOLEAN: True if hero died with starting AC >= 10 despite carrying armor in pack)**.
- **`ticks`** (Granular tick-by-tick log linked via `run_id` for outlier good runs, exemplars, and sampled runs):
  `run_id` (VARCHAR), `turn` (INT), `step_idx` (INT), `depth` (INT), `dungeon_num` (INT), `hp` (INT), `max_hp` (INT), `energy` (INT), `ac` (INT), `xp_level` (INT), `hunger_state` (INT), `condition_mask` (INT), `y` (INT), `x` (INT), `action_name` (VARCHAR), `raw_key` (INT), `active_function` (VARCHAR), `message` (VARCHAR), `visible_monsters` (VARCHAR), `inventory_summary` (VARCHAR).
- **`batches`** (Per-generation batch overview):
  `batch_id`, `generation`, `mode`, `target_functions`, `episodes_recorded`, `episodes_discarded`, `avg_depth`, `p90_depth`, `max_depth`, `avg_score`, `max_score`, `avg_turns`, `outlier_count`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`, `seeds_evaluated` (INT), `is_early_stopped` (BOOLEAN), `pilot_best_delta` (DOUBLE).
- **`archive_policies`** (Quality-Diversity MAP-Elites Behavioral Archive - 80 niches):
  `cell_id` (VARCHAR: e.g. 'cell_d1_c2_r0'), `depth_bin` (INT: 0-4), `caution_bin` (INT: 0-3), `resource_bin` (INT: 0-3), `fitness` (DOUBLE), `policy_code` (TEXT), `batch_id` (VARCHAR), `generation` (INT), `metrics_json` (VARCHAR), `updated_at` (TIMESTAMP).
- **`function_stats`** (Per-function execution & statistical debugging metrics per batch):
  `batch_id`, `generation`, `function_name`, `episodes_triggered`, `trigger_rate`, `total_calls`, `total_turns_active`, `avg_depth_when_triggered`, `death_rate_while_active`, **`ochiai_good_run_score` (DOUBLE: correlation with `is_outlier_good=TRUE`)**, **`liblit_failure_score` (DOUBLE: correlation with failure)**, **`stagnation_turns` (BIGINT: turns spent stalled after last progress)**, **`stagnation_share` (DOUBLE: fraction of all stagnant turns in batch consumed by this function)**, **`livelock_score` (DOUBLE: livelock sink severity)**.
- **`experiments`** (Historical hypothesis outcomes):
  `experiment_id`, `generation`, `mode`, `target_functions`, `observation_from_db`, `mechanism`, `passed`, `mean_delta`, `ci_lower`, `ci_upper`, `wilcoxon_p`, `wins`, `losses`, `ties`, `seeds_evaluated` (INT), `is_early_stopped` (BOOLEAN), `pilot_best_delta` (DOUBLE), `verdict_reason`.
- **`expert_runs`** (Consolidated successful human & championship-bot NetHack runs using the **same summary/log/stats columns as `runs`**):
  `run_id`, `source` ('nao_human' | 'autoascend'), `player`, `role`, `depth`, `max_depth`, `score`, `turns`, `xp_level`, `final_hp`, `max_hp`, `final_ac`, `gold`, `end_reason` ('ascended' | 'deep_run'), **`summary_digest` (TEXT)**, **`game_log` (TEXT)**, `milestones_json` (VARCHAR), `dumplog_text` (TEXT).
- **`expert_ticks`** (Tick-by-tick trajectories of expert runs linked via `run_id`):
  `run_id`, `turn`, `step_idx`, `depth`, `dungeon_num`, `hp`, `max_hp`, `ac`, `xp_level`, `hunger_state`, `y`, `x`, `action_name`, `raw_key`, `message`, `screen_ascii`.
- **`synthesis_logs`** (Turn-by-turn LLM reasoning tokens, candidate code, compiler/sandbox errors, and tool telemetry):
  `log_id`, `generation`, `batch_id`, `mode`, `turn_index`, `turn_type`, `repair_idx`, `model`, `thinking_content`, `message_content`, `tool_calls_json`, `tool_results_json`, `candidate_code`, `skill_name`, `skill_code`, `hypothesis_yaml`, `status`, `error_diagnostic`, `prompt_sent`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `wall_ms`, `created_at`.
"""
