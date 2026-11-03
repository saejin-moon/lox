"""
Utility to reset the LOX campaign to Generation 0 clean state
while preserving the offline Wiki and expert ascensions corpus.
"""

import os

import duckdb

DB_PATH = "data/lox.duckdb"
POLICY_PATH = "data/workspace/policy.py"
MACROS_PATH = "data/workspace/macros.py"


def reset_clean():
    if os.path.exists(POLICY_PATH):
        os.remove(POLICY_PATH)
        print(f"Removed {POLICY_PATH}")
    if os.path.exists(MACROS_PATH):
        os.remove(MACROS_PATH)
        print(f"Removed {MACROS_PATH}")
    skills_dir = "data/workspace/skills"
    if os.path.exists(skills_dir):
        import shutil

        shutil.rmtree(skills_dir)
        os.makedirs(skills_dir, exist_ok=True)
        print(f"Cleared {skills_dir}")

    if os.path.exists(DB_PATH):
        con = duckdb.connect(DB_PATH)
        try:
            for table in [
                "batches",
                "runs",
                "ticks",
                "function_stats",
                "policies",
                "experiments",
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
