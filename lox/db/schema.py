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
    wall_ms DOUBLE
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
    liblit_failure_score DOUBLE
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
    verdict_reason VARCHAR,
    created_at TIMESTAMP
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
"""


def init_db(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Initializes all DuckDB tables for LOX."""
    con = connect_db(db_path, read_only=False)
    con.execute(SCHEMA_SQL)
    return con


def get_schema_markdown(db_path: str = DEFAULT_DB_PATH) -> str:
    """Returns a concise, exact DuckDB schema reference for the Author Agent's system prompt."""
    return """### DuckDB Database Schema (`query_db(sql)`):
- **`runs`** (Every evaluated episode):
  `run_id` (VARCHAR), `batch_id` (VARCHAR), `generation` (INT), `seed` (BIGINT), `role` (VARCHAR), `depth` (INT), `max_depth` (INT), `score` (INT), `turns` (INT), `steps` (INT), `xp_level` (INT), `final_hp` (INT), `max_hp` (INT), `final_ac` (INT), `gold` (INT), `end_reason` (VARCHAR), `death_message` (VARCHAR), **`is_outlier_good` (BOOLEAN: True for statistical outlier good runs in that batch)**, `outlier_z_score` (DOUBLE), `batch_percentile` (DOUBLE), **`summary_digest` (TEXT: RLE timeline + inflection windows + stalls)**, **`game_log` (TEXT: chronological game messages)**, `milestones_json` (VARCHAR), `triggered_functions` (VARCHAR: JSON dict of function call counts), `last_active_function` (VARCHAR).
- **`ticks`** (Granular tick-by-tick log linked via `run_id` for outlier good runs, exemplars, and sampled runs):
  `run_id` (VARCHAR), `turn` (INT), `step_idx` (INT), `depth` (INT), `dungeon_num` (INT), `hp` (INT), `max_hp` (INT), `energy` (INT), `ac` (INT), `xp_level` (INT), `hunger_state` (INT), `condition_mask` (INT), `y` (INT), `x` (INT), `action_name` (VARCHAR), `raw_key` (INT), `active_function` (VARCHAR), `message` (VARCHAR), `visible_monsters` (VARCHAR), `inventory_summary` (VARCHAR).
- **`batches`** (Per-generation batch overview):
  `batch_id`, `generation`, `mode`, `target_functions`, `episodes_recorded`, `episodes_discarded`, `avg_depth`, `p90_depth`, `max_depth`, `avg_score`, `max_score`, `avg_turns`, `outlier_count`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`.
- **`function_stats`** (Per-function execution & statistical debugging metrics per batch):
  `batch_id`, `generation`, `function_name`, `episodes_triggered`, `trigger_rate`, `total_calls`, `total_turns_active`, `avg_depth_when_triggered`, `death_rate_while_active`, **`ochiai_good_run_score` (DOUBLE: correlation with `is_outlier_good=TRUE`)**, **`liblit_failure_score` (DOUBLE: correlation with dying while function is active)**.
- **`experiments`** (Historical hypothesis outcomes):
  `experiment_id`, `generation`, `mode`, `target_functions`, `observation_from_db`, `mechanism`, `passed`, `mean_delta`, `ci_lower`, `ci_upper`, `wilcoxon_p`, `wins`, `losses`, `ties`, `verdict_reason`.
- **`expert_runs`** (Consolidated successful human & championship-bot NetHack runs using the **same summary/log/stats columns as `runs`**):
  `run_id`, `source` ('nao_human' | 'autoascend'), `player`, `role`, `depth`, `max_depth`, `score`, `turns`, `xp_level`, `final_hp`, `max_hp`, `final_ac`, `gold`, `end_reason` ('ascended' | 'deep_run'), **`summary_digest` (TEXT)**, **`game_log` (TEXT)**, `milestones_json` (VARCHAR), `dumplog_text` (TEXT).
- **`expert_ticks`** (Tick-by-tick trajectories of expert runs linked via `run_id`):
  `run_id`, `turn`, `step_idx`, `depth`, `dungeon_num`, `hp`, `max_hp`, `ac`, `xp_level`, `hunger_state`, `y`, `x`, `action_name`, `raw_key`, `message`, `screen_ascii`.
"""
