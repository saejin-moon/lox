"""Debug Ep2 stall: run seed 44 and print decision state every N steps."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corp.env.nle_wrapper import make_env
from corp.agent.corp_agent import CORPAgent
from corp.planner.nogood import NogoodStore
from corp.deliberative.providers.mock_provider import MockProvider


def main():
    env = make_env(character="val-hum-law-fem")
    agent = CORPAgent(
        env=env,
        nogood_store=NogoodStore(),
        llm_provider=MockProvider(),
        eval_type="debug",
        mode="random",
        seed=46,
        role="valkyrie",
    )

    # Monkey-patch select_action to log state
    orig_select = agent.select_action
    counters = {"n": 0}

    def logged_select():
        task = orig_select()
        counters["n"] += 1
        if counters["n"] % 100 == 0:
            bl = agent.current_blstats
            lvl = agent.nav_mgr.levels.get((bl.dungeon_number, bl.level_number))
            info = ""
            if lvl is not None:
                info = (
                    f"sd={lvl.stairs_down} su={lvl.stairs_up} "
                    f"blk={len(lvl.blocked_tiles)} unvis={int((lvl.walkable & (lvl.visited == 0)).sum())} "
                    f"turns={lvl.turns_spent}"
                )
            directive = agent.macro_director.get_navigation_directive(bl, agent.dungeon_graph)
            try:
                dg = agent.nav_mgr._compute_bfs_distances(bl.y, bl.x, lvl)
                unvis_mask = (lvl.walkable & (lvl.visited == 0)) & (dg >= 0)
                unvis_reach = int(unvis_mask.sum())
            except Exception:
                unvis_reach = -1
            mons = agent.combat_mgr.scan_monsters(agent.current_glyphs, bl)
            mon_str = ";".join(
                f"{m.name}@d{m.distance}{'A' if m.is_adjacent else ''}sp{m.speed}th{m.threat_score:.0f}"
                for m in mons[:4]
            )
            print(
                f"[step {counters['n']:5d}] depth={bl.depth} dnum={bl.dungeon_number} dlvl={bl.level_number} "
                f"pos=({bl.y},{bl.x}) hp={bl.hp}/{bl.max_hp} task={task.name:12s} dir={directive} msg='{agent.current_message[:60]}' mons=[{mon_str}] ureach={unvis_reach} {info}",
                flush=True,
            )
        return task

    agent.select_action = logged_select
    result = agent.run_episode(max_steps=6000)
    print(f"END: turns={result.turns} depth={result.max_depth} score={result.final_score} death={result.death_message}")


if __name__ == "__main__":
    main()
