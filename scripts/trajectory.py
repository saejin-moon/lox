"""R0: Reconstruct the LOX-ψ capability trajectory from DuckDB telemetry.

Produces data/trajectory.csv (per-run aggregates ordered by first-timestamp) and
data/trajectory.png (Figure 1: mean score + max depth vs calendar time).

Usage: uv run python scripts/trajectory.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import duckdb

DB = "data/lox_telemetry.duckdb"
OUT_CSV = "data/trajectory.csv"
OUT_PNG = "data/trajectory.png"


def main():
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute(
        """
        WITH per_run AS (
            SELECT
                run_id,
                MIN(timestamp)                            AS first_ts,
                MAX(timestamp)                            AS last_ts,
                COUNT(*)                                  AS episodes,
                AVG(total_turns)                          AS mean_turns,
                AVG(final_score)                          AS mean_score,
                MAX(final_score)                          AS best_score,
                MEDIAN(max_depth)                         AS median_depth,
                MAX(max_depth)                            AS best_depth,
                SUM(CASE WHEN is_ascended THEN 1 ELSE 0 END) AS ascensions,
                MIN(role)                                 AS role
            FROM episodes
            GROUP BY run_id
        )
        SELECT * FROM per_run ORDER BY first_ts
        """
    ).fetchall()
    con.close()

    cols = [
        "run_id", "first_ts", "last_ts", "episodes", "mean_turns", "mean_score",
        "best_score", "median_depth", "best_depth", "ascensions", "role",
    ]
    import csv
    import datetime as dt

    def to_iso(v):
        if isinstance(v, str):
            return v
        return dt.datetime.fromtimestamp(float(v)).isoformat(timespec="seconds")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            d = dict(zip(cols, r))
            d["first_ts"] = to_iso(d["first_ts"])
            d["last_ts"] = to_iso(d["last_ts"])
            w.writerow([d[c] for c in cols])

    print(f"Wrote {OUT_CSV} ({len(rows)} runs)")
    print(f"{'first_ts':<20} {'run':<8} {'eps':>4} {'mean_score':>10} {'best':>6} {'med_d':>5} {'best_d':>6}")
    for r in rows:
        d = dict(zip(cols, r))
        print(
            f"{d['first_ts'][:16]:<20} "
            f"{d['run_id']:<8} {d['episodes']:>4} {d['mean_score']:>10.1f} {d['best_score']:>6.0f} "
            f"{d['median_depth']:>5.1f} {d['best_depth']:>6.0f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def to_epoch(v):
            if isinstance(v, str):
                return dt.datetime.fromisoformat(v).timestamp()
            return float(v)

        t0 = to_epoch(rows[0][1])
        xs = [(to_epoch(r[1]) - t0) / 3600.0 for r in rows]  # hours since first run
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        sizes = [max(10, min(120, r[3] * 1.2)) for r in rows]
        ax1.scatter(xs, [r[6] for r in rows], s=sizes, alpha=0.6, label="mean score", color="tab:blue")
        ax1.plot(xs, [r[6] for r in rows], alpha=0.25, color="tab:blue")
        ax1.scatter(xs, [r[5] for r in rows], s=sizes, alpha=0.6, label="best score", color="tab:orange", marker="^")
        ax1.set_ylabel("score")
        ax1.legend()
        ax1.set_title("LOX-ψ capability trajectory (each point = one benchmark run; size = episode count)")
        ax2.scatter(xs, [r[7] for r in rows], s=sizes, alpha=0.6, label="median depth", color="tab:green")
        ax2.plot(xs, [r[7] for r in rows], alpha=0.25, color="tab:green")
        ax2.scatter(xs, [r[8] for r in rows], s=12, alpha=0.6, label="best depth", color="tab:red", marker="x")
        ax2.set_ylabel("dungeon depth")
        ax2.set_xlabel("hours since first recorded run")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        print(f"Wrote {OUT_PNG}")
    except ImportError:
        print("matplotlib unavailable — CSV only")


if __name__ == "__main__":
    main()
