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
        self.llm_queries_glob = os.path.join(parquet_dir, "llm_queries", "*.parquet")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def consolidate(self, clean_parquet: bool = False) -> dict[str, Any]:
        """
        Ingests all Parquet files into the persistent DuckDB database,
        rebuilds analytical views, and optionally deletes raw Parquet files post-verification.
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

        # 3. Ingest LLM Queries
        has_llm_queries = any(
            f.endswith(".parquet")
            for f in os.listdir(os.path.join(self.parquet_dir, "llm_queries"))
        ) if os.path.exists(os.path.join(self.parquet_dir, "llm_queries")) else False

        llm_queries_count = 0
        if has_llm_queries:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS llm_queries AS 
                SELECT * FROM read_parquet('{self.llm_queries_glob}') WHERE 1=0;
            """)
            conn.execute(f"""
                INSERT INTO llm_queries
                SELECT p.* FROM read_parquet('{self.llm_queries_glob}') p
                WHERE NOT EXISTS (
                    SELECT 1 FROM llm_queries q 
                    WHERE q.query_id = p.query_id
                );
            """)
            llm_queries_count = conn.execute("SELECT count(*) FROM llm_queries").fetchone()[0]

            # LLM query frequency view
            conn.execute("""
                CREATE OR REPLACE VIEW v_llm_query_frequency AS
                SELECT 
                    run_id,
                    trigger_type,
                    provider,
                    count(*) as total_queries,
                    round(avg(latency_ms), 1) as avg_latency_ms,
                    round(avg(turns_since_last_query), 1) as avg_turn_gap,
                    min(turns_since_last_query) as min_turn_gap,
                    round(avg(wall_seconds_since_last_query), 2) as avg_sec_gap,
                    round(case when avg(wall_seconds_since_last_query) > 0 then 60.0 / avg(wall_seconds_since_last_query) else 0.0 end, 1) as effective_rpm
                FROM llm_queries
                GROUP BY run_id, trigger_type, provider
                ORDER BY run_id, count(*) DESC;
            """)

            # LLM run summary view
            conn.execute("""
                CREATE OR REPLACE VIEW v_llm_run_summary AS
                SELECT 
                    run_id,
                    count(*) as total_llm_queries,
                    count(distinct episode_id) as episodes_with_queries,
                    round(count(*) * 1.0 / count(distinct episode_id), 2) as queries_per_episode,
                    round(avg(latency_ms), 1) as avg_latency_ms,
                    round(avg(turns_since_last_query), 1) as avg_turn_interval,
                    min(turns_since_last_query) as min_turn_interval,
                    round(avg(wall_seconds_since_last_query), 2) as avg_sec_interval,
                    round(case when avg(wall_seconds_since_last_query) > 0 then 60.0 / avg(wall_seconds_since_last_query) else 0.0 end, 1) as effective_rpm
                FROM llm_queries
                GROUP BY run_id
                ORDER BY run_id DESC;
            """)

        # 4. Build Canonical Analytical Views
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
        cleanup_stats = {}
        if clean_parquet:
            cleanup_stats = self.clean_parquet_files()

        return {
            "episodes": episodes_count,
            "ticks": ticks_count,
            "llm_queries": llm_queries_count,
            **cleanup_stats,
        }

    def clean_parquet_files(self) -> dict[str, Any]:
        """
        Safely deletes all ingested Parquet files from logs/parquet/
        once verified inside DuckDB.
        """
        deleted_count = 0
        deleted_bytes = 0
        for sub in ("episodes", "ticks", "llm_queries"):
            sub_dir = os.path.join(self.parquet_dir, sub)
            if not os.path.exists(sub_dir):
                continue
            for fname in os.listdir(sub_dir):
                if fname.endswith(".parquet"):
                    fpath = os.path.join(sub_dir, fname)
                    try:
                        deleted_bytes += os.path.getsize(fpath)
                        os.unlink(fpath)
                        deleted_count += 1
                    except OSError:
                        pass
        return {
            "deleted_files": deleted_count,
            "freed_mb": round(deleted_bytes / (1024 * 1024), 2),
        }

    def vacuum(self) -> None:
        """Runs VACUUM on DuckDB to reclaim free space."""
        conn = duckdb.connect(self.db_path)
        conn.execute("VACUUM;")
        conn.close()

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
