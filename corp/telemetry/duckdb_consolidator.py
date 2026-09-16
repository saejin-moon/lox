"""
Telemetry Subsystem: DuckDB Telemetry Consolidator.
Consolidates distributed Parquet log partitions into a persistent DuckDB database
and constructs pre-indexed analytical views for instant zero-copy SQL queries.
"""

import os
from typing import Any
import duckdb


class DuckDBConsolidator:
    """
    Consolidates streaming Parquet files into a unified DuckDB database
    and builds canonical analytics views.
    """

    def __init__(
        self,
        db_path: str = "data/corp_telemetry.duckdb",
        parquet_dir: str = "logs/parquet",
    ):
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.ticks_glob = os.path.join(parquet_dir, "ticks", "*.parquet")
        self.episodes_glob = os.path.join(parquet_dir, "episodes", "*.parquet")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def consolidate(self) -> dict[str, int]:
        """
        Ingests all Parquet files into the persistent DuckDB database
        and rebuilds analytical views.
        """
        conn = duckdb.connect(self.db_path)
        episodes_count = 0
        ticks_count = 0

        # 1. Ingest Episodes
        has_episodes = any(
            f.endswith(".parquet")
            for f in os.listdir(os.path.join(self.parquet_dir, "episodes"))
        ) if os.path.exists(os.path.join(self.parquet_dir, "episodes")) else False

        if has_episodes:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS episodes AS 
                SELECT * FROM read_parquet('{self.episodes_glob}') WHERE 1=0;
            """)
            # Insert records not already present
            conn.execute(f"""
                INSERT INTO episodes
                SELECT p.* FROM read_parquet('{self.episodes_glob}') p
                WHERE p.episode_id NOT IN (SELECT episode_id FROM episodes);
            """)
            episodes_count = conn.execute("SELECT count(*) FROM episodes").fetchone()[0]

        # 2. Ingest Ticks
        has_ticks = any(
            f.endswith(".parquet")
            for f in os.listdir(os.path.join(self.parquet_dir, "ticks"))
        ) if os.path.exists(os.path.join(self.parquet_dir, "ticks")) else False

        if has_ticks:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS ticks AS 
                SELECT * FROM read_parquet('{self.ticks_glob}') WHERE 1=0;
            """)
            # Direct insert from files
            conn.execute(f"""
                INSERT INTO ticks
                SELECT p.* FROM read_parquet('{self.ticks_glob}') p
                WHERE NOT EXISTS (
                    SELECT 1 FROM ticks t 
                    WHERE t.episode_id = p.episode_id AND t.step = p.step
                );
            """)
            ticks_count = conn.execute("SELECT count(*) FROM ticks").fetchone()[0]

        # 3. Build Canonical Analytical Views
        if has_episodes:
            conn.execute("""
                CREATE OR REPLACE VIEW v_eval_summary AS
                SELECT 
                    run_id,
                    eval_type,
                    mode,
                    count(*) as episodes_count,
                    median(max_depth) as median_depth,
                    round(avg(total_turns), 1) as avg_turns,
                    round(avg(final_score), 1) as avg_score,
                    round(sum(CASE WHEN is_ascended THEN 1 ELSE 0 END) * 100.0 / count(*), 2) as ascension_pct,
                    round(avg(mean_sps), 1) as avg_sps,
                    max(nogoods_active) as active_nogoods
                FROM episodes
                GROUP BY run_id, eval_type, mode
                ORDER BY max(timestamp) DESC;
            """)

            conn.execute("""
                CREATE OR REPLACE VIEW v_persona_performance AS
                SELECT 
                    role,
                    mode,
                    count(*) as total_runs,
                    median(max_depth) as median_depth,
                    round(avg(total_turns), 1) as avg_turns,
                    round(avg(final_score), 1) as avg_score,
                    round(avg(persona_resilience), 3) as avg_resilience,
                    round(avg(persona_mana), 3) as avg_mana
                FROM episodes
                GROUP BY role, mode
                ORDER BY median_depth DESC, avg_turns DESC;
            """)

            conn.execute("""
                CREATE OR REPLACE VIEW v_lethal_taxonomy AS
                SELECT 
                    death_category,
                    death_message,
                    count(*) as occurrences,
                    round(avg(total_turns), 1) as avg_survival_turns,
                    round(avg(max_depth), 1) as avg_depth,
                    sum(nogoods_synthesized) as total_nogoods_derived
                FROM episodes
                WHERE death_message != '' AND death_message != 'Survived'
                GROUP BY death_category, death_message
                ORDER BY occurrences DESC;
            """)

        if has_ticks:
            conn.execute("""
                CREATE OR REPLACE VIEW v_latency_stats AS
                SELECT 
                    action_name,
                    count(*) as action_count,
                    round(quantile_cont(decision_latency_us, 0.5), 1) as p50_latency_us,
                    round(quantile_cont(decision_latency_us, 0.90), 1) as p90_latency_us,
                    round(quantile_cont(decision_latency_us, 0.99), 1) as p99_latency_us,
                    round(avg(decision_latency_us), 1) as mean_latency_us
                FROM ticks
                GROUP BY action_name
                ORDER BY action_count DESC;
            """)

        conn.close()
        return {
            "episodes": episodes_count,
            "ticks": ticks_count,
        }

    def query(self, sql: str) -> list[tuple]:
        """Runs an arbitrary SQL query against the consolidated DuckDB database."""
        conn = duckdb.connect(self.db_path)
        result = conn.execute(sql).fetchall()
        conn.close()
        return result

    def query_formatted(self, sql: str) -> str:
        """Runs a query and returns a clean ASCII formatted table."""
        conn = duckdb.connect(self.db_path)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "(0 rows returned)"

        # Calculate column widths
        widths = [len(c) for c in cols]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        # Format header and rows
        header = " | ".join(f"{col:<{widths[i]}}" for i, col in enumerate(cols))
        sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
        body = "\n".join(
            " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row))
            for row in rows
        )
        return f"{header}\n{sep}\n{body}"
