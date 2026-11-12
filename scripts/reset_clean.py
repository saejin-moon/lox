"""
Utility to reset the LOX campaign to Generation 0 clean state
while preserving the offline Wiki and expert ascensions corpus.
"""

import os

import duckdb

DB_PATH = "data/lox.duckdb"
POLICY_PATH = "data/workspace/policy.py"
MACROS_PATH = "data/workspace/macros.py"


def reset_clean(
    db_path: str = DB_PATH,
    policy_path: str = POLICY_PATH,
    macros_path: str = MACROS_PATH,
    theories_path: str | None = None,
):
    import shutil
    from pathlib import Path

    workspace_dir = Path(policy_path).parent

    if os.path.exists(policy_path):
        os.remove(policy_path)
        print(f"Removed {policy_path}")
    if os.path.exists(macros_path):
        os.remove(macros_path)
        print(f"Removed {macros_path}")

    status_md = workspace_dir / "STATUS.md"
    if status_md.exists():
        os.remove(status_md)
        print(f"Removed {status_md}")

    skills_dir = workspace_dir / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
        skills_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {skills_dir}")

    t_path = (
        Path(theories_path) if theories_path else (workspace_dir / "theories.lance")
    )
    if t_path.exists():
        shutil.rmtree(t_path)
        print(f"Cleared {t_path}")

    if os.path.exists(db_path):
        con = duckdb.connect(db_path)
        try:
            for table in [
                "batches",
                "runs",
                "ticks",
                "function_stats",
                "policies",
                "experiments",
                "discoveries",
                "synthesis_logs",
            ]:
                try:
                    con.execute(f"DELETE FROM {table}")
                    print(f"Cleared table `{table}`")
                except Exception:
                    pass
            try:
                exp_count = con.execute("SELECT COUNT(*) FROM expert_runs").fetchone()[
                    0
                ]
                print(f"Preserved `expert_runs` ({exp_count} expert runs ready)")
            except Exception:
                pass
        finally:
            con.close()

        from lox.db.schema import init_db

        con_init = init_db(db_path)
        con_init.close()
        print("Verified DuckDB table schemas and views.")
    print("LOX workspace and database successfully reset to clean state.")


if __name__ == "__main__":
    reset_clean()
