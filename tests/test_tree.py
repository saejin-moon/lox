import numpy as np
from lox.core.types import Status, Action, Observation, HeroState, HungerState
from lox.core.tree import BehaviorTree, Selector, Sequence, Condition, ActionNode


def test_behavior_tree_execution():
    obs = Observation(
        chars=np.zeros((21, 79), dtype=np.uint8),
        glyphs=None,
        hero=HeroState(hp=20, max_hp=100, hunger_state=HungerState.WEAK),
    )

    action_taken = None

    def heal(bb):
        return Action(name="HEAL")

    def eat(bb):
        return Action(name="EAT")

    def explore(bb):
        return Action(name="EXPLORE")

    # Priority tree:
    # 1. If HP <= 25% -> HEAL
    # 2. If Hunger >= WEAK -> EAT
    # 3. Default -> EXPLORE
    tree = BehaviorTree(
        root=Selector([
            Sequence([
                Condition(lambda bb: bb.obs.hero.hp_frac <= 0.25),
                ActionNode(heal),
            ]),
            Sequence([
                Condition(lambda bb: bb.obs.hero.hunger_state == HungerState.WEAK),
                ActionNode(eat),
            ]),
            ActionNode(explore),
        ])
    )

    # HP is 20/100 (<= 0.25) -> HEAL should trigger first
    act = tree.execute(obs)
    assert act is not None
    assert act.name == "HEAL"

    # Now HP is 50/100 -> EAT should trigger next
    obs.hero.hp = 50
    act = tree.execute(obs)
    assert act is not None
    assert act.name == "EAT"

    # Now hunger is NORMAL -> EXPLORE should trigger
    obs.hero.hunger_state = HungerState.NORMAL
    act = tree.execute(obs)
    assert act is not None
    assert act.name == "EXPLORE"
