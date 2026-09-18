import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from corp.env.nle_wrapper import make_env
from corp.agent.corp_agent import CORPAgent
from corp.deliberative.providers.mock_provider import MockProvider

def run_diagnostic():
    env = make_env()
    agent = CORPAgent(env=env, llm_provider=MockProvider())
    obs, info = agent.reset(seed=42)
    
    for step in range(1, 550):
        task = agent.select_action()
        obs, reward, terminated, truncated, info = agent.dispatcher.dispatch(
            task,
            step_callback=agent._micro_step_callback,
        )
        agent.step_counter += 1
        new_blstats = info["blstats"]
        if new_blstats.turn > 0:
            agent.current_blstats = new_blstats
        agent.current_chars = obs["chars"]
        agent.current_glyphs = obs["glyphs"]
        agent.current_message = info.get("full_message", "")
        
        if 500 <= step <= 520:
            lvl = agent.nav_mgr.get_or_create_level(agent.current_blstats)
            unv = agent.nav_mgr.find_nearest_unvisited(agent.current_blstats.y, agent.current_blstats.x, lvl)
            print(f"Step {step:3d}: pos=({agent.current_blstats.y}, {agent.current_blstats.x}), task={task.name} {task.args}, unv_target={unv}, msg='{agent.current_message}'")

    lvl = agent.nav_mgr.get_or_create_level(agent.current_blstats)
    unvisited_pts = np.argwhere(lvl.walkable & (lvl.visited == 0))
    py, px = agent.current_blstats.y, agent.current_blstats.x
    print(f"\nAt step 550:")
    print(f"Hero pos: ({py}, {px})")
    print(f"Total unvisited walkable tiles: {len(unvisited_pts)}")
    for pt in unvisited_pts:
        ch = chr(agent.current_chars[pt[0], pt[1]])
        path = agent.nav_mgr.astar.find_path((py, px), (int(pt[0]), int(pt[1])), lvl.walkable)
        print(f"  Tile ({pt[0]}, {pt[1]}): char '{ch}', reachable={path is not None}")

if __name__ == "__main__":
    run_diagnostic()
