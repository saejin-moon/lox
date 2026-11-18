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
    frontier_dir: str | None = None,
    seed_starter: bool = False,
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

    for extra_name in (
        "checkpoints",
        "frontier",
        "nethack_scenarios",
        "campaign_errors.log",
    ):
        extra_p = workspace_dir / extra_name
        if extra_p.is_dir():
            shutil.rmtree(extra_p)
            print(f"Cleared {extra_p}")
        elif extra_p.exists():
            os.remove(extra_p)
            print(f"Removed {extra_p}")

    skills_dir = workspace_dir / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
        skills_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {skills_dir}")

    if seed_starter:
        from lox.lang.bundle import write_skills_dir
        from lox.lang.starter import build_clean_option_a_starter

        starter_pol, starter_skills = build_clean_option_a_starter()
        Path(policy_path).parent.mkdir(parents=True, exist_ok=True)
        Path(policy_path).write_text(starter_pol, encoding="utf-8")
        write_skills_dir(
            skills_dir, starter_skills, sync=True, ensure_trailing_newline=False
        )
        print(f"Seeded bare Option-A Gen 0 starter ({len(starter_skills)} modules)")

    for theory_p in (
        Path(theories_path) if theories_path else (workspace_dir / "theories.lance"),
        workspace_dir / "theories.duckdb",
    ):
        if theory_p.is_dir():
            shutil.rmtree(theory_p)
            print(f"Cleared {theory_p}")
        elif theory_p.exists():
            os.remove(theory_p)
            print(f"Removed {theory_p}")

    if frontier_dir:
        f_path = Path(frontier_dir)
        if f_path.exists():
            shutil.rmtree(f_path)
            f_path.mkdir(parents=True, exist_ok=True)
            print(f"Cleared {f_path}")

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
                "theories",
                "tournament_attempts",
                "archive_policies",
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
