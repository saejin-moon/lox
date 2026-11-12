"""
Bootstraps Go-Explore Frontier Checkpoints from recorded DuckDB runs.
Extracts verified multi-depth trajectories (Dlvl 2, 3, 4, 5) with exact action bytes
to seed the frontier archive for deep-tier sub-agent testing.
"""

from __future__ import annotations

from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.engine.frontier import DEFAULT_FRONTIER_DIR, FrontierArchive


def bootstrap_frontier_from_duckdb(
    db_path: str = DEFAULT_DB_PATH,
    frontier_dir: str = DEFAULT_FRONTIER_DIR,
    max_checkpoints_per_depth: int = 5,
) -> int:
    """
    Extracts high-progress trajectories from DuckDB runs with recorded action bytes
    and populates initial frontier checkpoints for Dlvl 2..5+.
    """
    archive = FrontierArchive(frontier_dir=frontier_dir)
    con = connect_db(db_path, read_only=True)
    added = 0

    try:
        for depth in (2, 3, 4, 5):
            query = """
                SELECT run_id, seed, max_depth, turns, action_bytes, game_log
                FROM runs
                WHERE octet_length(action_bytes) > 0 AND max_depth = ?
                ORDER BY score DESC, turns DESC
                LIMIT ?
            """
            rows = con.execute(query, [depth, max_checkpoints_per_depth]).fetchall()
            for row in rows:
                run_id, seed, max_d, turns, act_bytes, game_log = row
                cid = f"fc_dlvl_{max_d}_seed{seed}_{str(run_id)[-8:]}"
                if cid not in archive.checkpoints:
                    archive.add_checkpoint(
                        seed=int(seed),
                        depth=int(max_d),
                        milestone=f"DLVL_{max_d}",
                        turns=int(turns),
                        action_bytes=bytes(act_bytes),
                        game_log=str(game_log or ""),
                        checkpoint_id=cid,
                    )
                    added += 1
    finally:
        con.close()

    return added


if __name__ == "__main__":
    count = bootstrap_frontier_from_duckdb()
    print(f"Bootstrapped {count} frontier checkpoints into {DEFAULT_FRONTIER_DIR}")
