"""
LOX-ψ Policy Layer (LOX-ψ): cross-episode goal-failure counters (S0, AGENT_PLAN §1.3).

Wires DuckDB goal_events (skipped_budget / skipped_when) into the condition
vocabulary so the `failures_in_10_episodes_ge <n> <goal>` predicate evaluates
against REAL telemetry — the loop can deprioritize what keeps failing (S0 item 2).

Counters are computed over the LAST N recorded episodes and cached briefly
(one query per refresh window, shared across episode construction).
"""
from __future__ import annotations

import os
import time

DEFAULT_DB_PATH = "data/lox_telemetry.duckdb"
DEFAULT_WINDOW_EPISODES = 10
_CACHE_TTL_SEC = 60.0

_CACHE: dict[str, tuple[float, dict[str, int]]] = {}


def load_failure_counters(
    db_path: str = DEFAULT_DB_PATH,
    last_n_episodes: int = DEFAULT_WINDOW_EPISODES,
    *, refresh: bool = False,
) -> dict[str, int]:
    """Per-goal failure counts over the last N episodes: dict goal → count of
    skipped_budget + skipped_when goal_events. Degrades to {} (never raises)."""
    key = f"{os.path.abspath(db_path)}:{int(last_n_episodes)}"
    now = time.monotonic()
    if not refresh:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SEC:
            return dict(hit[1])
    counters: dict[str, int] = {}
    try:
        if os.path.exists(db_path):
            import duckdb
            con = duckdb.connect(db_path, read_only=True)
            try:
                rows = con.execute(
                    """SELECT goal, COUNT(*) FROM (
                           SELECT * FROM goal_events
                           WHERE event IN ('skipped_budget', 'skipped_when')
                           ORDER BY ts DESC
                       )
                       WHERE episode_id IN (
                           SELECT DISTINCT episode_id FROM (
                               SELECT episode_id FROM goal_events
                               ORDER BY ts DESC LIMIT ?
                           )
                       )
                       GROUP BY goal ORDER BY goal""",
                    (int(last_n_episodes),),
                ).fetchall()
            finally:
                con.close()
            counters = {str(r[0]): int(r[1]) for r in rows}
    except Exception:  # noqa: BLE001 — telemetry is advisory; an empty dict is safe
        counters = {}
    _CACHE[key] = (now, dict(counters))
    return counters


def counters_ctx(counters: dict[str, int]) -> dict:
    """The ctx keys consumed by lox.policy.predicates.nethack_bindings."""
    return {"failures": dict(counters), "failures_total": sum(counters.values())}
