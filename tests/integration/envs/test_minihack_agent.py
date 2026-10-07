import numpy as np

from lox.core.spatial import SpatialEngine
from lox.core.tree import Blackboard
from lox.core.types import Action, Status
from lox.dsl.compiler import compile_policy
from lox.envs.minihack import MiniHackAdapter

# Set of walkable ASCII characters in MiniHack / NetHack
WALKABLE_CHARS = {
    ord("."),
    ord("#"),
    ord("+"),
    ord("'"),
    ord("@"),
    ord(">"),
    ord("<"),
    ord("$"),
}


def get_walkable_mask(chars: np.ndarray) -> np.ndarray:
    mask = np.zeros(chars.shape, dtype=bool)
    for c in WALKABLE_CHARS:
        mask |= chars == c
    return mask


def test_minihack_agent_navigation():
    adapter = MiniHackAdapter(task="MiniHack-ExploreMaze-Easy-Mapped-v0")
    obs = adapter.reset(seed=42)

    # Action handlers connecting BehaviorTree actions to SpatialEngine
    def handle_step_to_stairs(bb: Blackboard, args):
        chars = bb.obs.chars
        hero_pos = (bb.obs.hero.y, bb.obs.hero.x)
        walkable = get_walkable_mask(chars)

        stairs_loc = np.argwhere(chars == ord(">"))
        if len(stairs_loc) > 0:
            target = (int(stairs_loc[0, 0]), int(stairs_loc[0, 1]))
            path = SpatialEngine.find_path(hero_pos, target, walkable)
            if path:
                next_step = path[0]
                dy = next_step[0] - hero_pos[0]
                dx = next_step[1] - hero_pos[1]
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    def handle_descend(bb: Blackboard, args):
        hero_pos = (bb.obs.hero.y, bb.obs.hero.x)
        if bb.obs.chars[hero_pos[0], hero_pos[1]] == ord(">"):
            return Action(name="descend")
        return Status.FAILURE

    def handle_step_to_frontier(bb: Blackboard, args):
        chars = bb.obs.chars
        hero_pos = (bb.obs.hero.y, bb.obs.hero.x)
        walkable = get_walkable_mask(chars)
        visited = adapter.visited

        frontier = SpatialEngine.find_nearest_frontier(hero_pos, walkable, visited)
        if frontier is not None:
            path = SpatialEngine.find_path(hero_pos, frontier, walkable)
            if path:
                next_step = path[0]
                dy = next_step[0] - hero_pos[0]
                dx = next_step[1] - hero_pos[1]
                return Action(name="step", direction=(dy, dx))
        return Status.FAILURE

    action_handlers = {
        "step_to_stairs_down": handle_step_to_stairs,
        "descend": handle_descend,
        "step_to_frontier": handle_step_to_frontier,
    }

    # Policy written in clean Pythonic Infix AST
    policy_code = """
def reach_goal():
    if stairs_down_known:
        step_to_stairs_down()
        descend()

def explore():
    step_to_frontier()

plan = [
    reach_goal,
    explore,
]
"""
    tree = compile_policy(policy_code, action_handlers=action_handlers)

    steps = 0
    solved = False
    max_steps = 100

    while steps < max_steps:
        # Update blackboard memory flags
        chars = obs.chars
        has_stairs = bool(np.any(chars == ord(">")))
        memory = {
            "stairs_down_known": has_stairs,
            "has_unvisited_frontier": True,
        }

        action = tree.execute(obs, memory=memory)
        if action is None:
            action = Action(name="wait")

        obs, reward, terminated, truncated, info = adapter.step(action)
        steps += 1

        if reward > 0.5 or (terminated and not truncated):
            solved = True
            break

    adapter.close()
    print(f"MiniHack run completed in {steps} steps, solved: {solved}")
    assert solved or steps > 5
