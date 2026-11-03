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
):
    import shutil
    from pathlib import Path

    if os.path.exists(policy_path):
        os.remove(policy_path)
        print(f"Removed {policy_path}")
    if os.path.exists(macros_path):
        os.remove(macros_path)
        print(f"Removed {macros_path}")
    skills_dir = Path(policy_path).parent / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
        skills_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {skills_dir}")

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
            ]:
                con.execute(f"DELETE FROM {table}")
                print(f"Cleared table `{table}`")
            exp_count = con.execute("SELECT COUNT(*) FROM expert_runs").fetchone()[0]
            print(f"Preserved `expert_runs` ({exp_count} expert runs ready)")
        finally:
            con.close()
    print("LOX workspace and database successfully reset to clean state.")


if __name__ == "__main__":
    reset_clean()
