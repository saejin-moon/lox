"""
CORP-Ω Policy Layer: run-report bundle builder (R3, AGENT_PLAN §4.5).

Builds the reviser's input from DuckDB: batch summary, death taxonomy, stall census,
goal stats — plus prev_revisions from the ledger. Missing tables/DBs degrade to an
empty-but-shaped bundle so the loop never crashes on a fresh install.

Exact queries per the R3 addendum:
  death taxonomy  ← episodes.death_message LIKE-grouping
  goal stats      ← goal_events (completions / skips_budget / skips_when / median steps)
  hp-at-death     ← ticks last-rows per episode
"""
from __future__ import annotations

import json
import math
import statistics

from corp.policy.ledger import RevisionLedger, DEFAULT_LEDGER_PATH

DEFAULT_DB_PATH = "data/corp_telemetry.duckdb"
VARIANCE_NOTE = "NetHackChallenge unseedable; deltas < 1 batch-sigma are noise"


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0


SURVIVED_MARKERS = ("", "survived", "alive", "ascended")


def _is_survived(death_message: str) -> bool:
    return (death_message or "").strip().lower() in SURVIVED_MARKERS


def build_report_bundle(
    db_path: str = DEFAULT_DB_PATH,
    ledger_path: str = DEFAULT_LEDGER_PATH,
    domain: str = "nethack",
    last_n_episodes: int = 100,
) -> dict:
    bundle: dict = {
        "domain": domain,
        "batch": {"episodes": 0, "median_depth": 0.0, "mean_score": 0.0,
                  "best_score": 0, "survival_rate": 0.0, "mean_turns": 0.0},
        "death_taxonomy": [],
        "stall_census": [],
        "goal_stats": [],
        "prev_revisions": [],
        "variance_note": VARIANCE_NOTE,
    }

    # ---- DuckDB: batch summary + death taxonomy + stalls + goal stats --------
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        try:
            row = con.execute(
                f"""SELECT COUNT(*), MEDIAN(max_depth), AVG(final_score), MAX(final_score),
                           AVG(CASE WHEN LOWER(COALESCE(death_message, '')) IN ('', 'survived', 'alive', 'ascended') THEN 1.0 ELSE 0.0 END),
                           AVG(total_turns)
                    FROM (SELECT * FROM episodes ORDER BY rowid DESC LIMIT {last_n_episodes})"""
            ).fetchone()
            if row and row[0]:
                bundle["batch"] = {
                    "episodes": int(row[0]), "median_depth": float(row[1] or 0),
                    "mean_score": float(row[2] or 0), "best_score": int(row[3] or 0),
                    "survival_rate": float(row[4] or 0), "mean_turns": float(row[5] or 0),
                }

            # Death taxonomy: LIKE-grouped death messages with depth/hp-at-death medians
            # ('Survived' episodes are NOT deaths — excluded from the taxonomy)
            rows = con.execute(
                f"""SELECT death_message, COUNT(*) AS n, MEDIAN(max_depth)
                    FROM (SELECT * FROM episodes
                          WHERE death_message IS NOT NULL AND death_message != ''
                            AND LOWER(death_message) NOT IN ('survived', 'alive', 'ascended')
                          ORDER BY rowid DESC LIMIT {last_n_episodes})
                    GROUP BY death_message ORDER BY n DESC LIMIT 12"""
            ).fetchall()
            taxonomy = [
                {"cause": r[0], "count": int(r[1]), "median_depth_at_death": int(r[2] or 0),
                 "median_hp_frac_at_death": 0.0}
                for r in rows
            ]
            # hp-at-death from the last tick of each recent episode (per addendum)
            try:
                hp_rows = con.execute(
                    f"""WITH last_ticks AS (
                            SELECT episode_id, arg_max(hp, step) AS hp, arg_max(max_hp, step) AS max_hp
                            FROM (SELECT * FROM ticks ORDER BY rowid DESC LIMIT 50000)
                            GROUP BY episode_id)
                        SELECT episode_id, hp, max_hp FROM last_ticks LIMIT {last_n_episodes}"""
                ).fetchall()
                hp_by_episode = {r[0]: (r[1], r[2]) for r in hp_rows}
                ep_rows = con.execute(
                    f"""SELECT episode_id, death_message FROM
                        (SELECT episode_id, death_message FROM episodes
                         WHERE death_message IS NOT NULL AND death_message != ''
                         ORDER BY rowid DESC LIMIT {last_n_episodes})"""
                ).fetchall()
                frac_by_cause: dict[str, list[float]] = {}
                for ep_id, cause in ep_rows:
                    if ep_id in hp_by_episode and hp_by_episode[ep_id][1]:
                        frac = hp_by_episode[ep_id][0] / max(1, hp_by_episode[ep_id][1])
                        frac_by_cause.setdefault(cause, []).append(frac)
                for entry in taxonomy:
                    fr = frac_by_cause.get(entry["cause"], [])
                    entry["median_hp_frac_at_death"] = round(statistics.median(fr), 3) if fr else 0.0
            except Exception:  # noqa: BLE001 — hp enrichment is best-effort
                pass
            bundle["death_taxonomy"] = taxonomy

            # Goal stats from goal_events
            try:
                rows = con.execute(
                    """SELECT goal,
                              SUM(CASE WHEN event = 'completed' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN event = 'skipped_budget' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN event = 'skipped_when' THEN 1 ELSE 0 END),
                              MEDIAN(CASE WHEN event = 'completed' THEN steps_in_goal END)
                       FROM goal_events GROUP BY goal ORDER BY goal"""
                ).fetchall()
                bundle["goal_stats"] = [
                    {"goal": r[0], "completions": int(r[1] or 0), "skips_budget": int(r[2] or 0),
                     "skips_when": int(r[3] or 0), "median_steps": int(r[4] or 0)}
                    for r in rows
                ]
                # Stall census: goals that burn steps without completing
                stalls = [
                    {"goal": g["goal"],
                     "stalls": g["skips_budget"] + g["skips_when"],
                     "median_stall_turns": 0}
                    for g in bundle["goal_stats"]
                    if (g["skips_budget"] + g["skips_when"]) > 0
                ]
                bundle["stall_census"] = sorted(stalls, key=lambda s: -s["stalls"])[:8]
            except Exception:  # noqa: BLE001 — goal_events table may not exist yet
                pass
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — no DB yet: empty bundle
        pass

    # ---- Ledger: prev_revisions ----------------------------------------------
    ledger = RevisionLedger(ledger_path)
    prev = []
    for e in ledger.all()[-5:]:
        if e.get("type") == "policy_revision":
            prev.append({
                "version": e.get("revision"),
                "accepted": bool(e.get("accepted")),
                "delta_score": (e.get("delta") or {}).get("after"),
                "reason": e.get("reason", ""),
            })
    bundle["prev_revisions"] = prev
    return bundle


if __name__ == "__main__":  # pragma: no cover — debugging helper
    print(json.dumps(build_report_bundle(), indent=2)[:4000])