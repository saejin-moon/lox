from pathlib import Path
import numpy as np

from lox.author.agent import AuthorAgent
from lox.core.types import Action, HeroState, Observation, SpatialView
from lox.dsl.compiler import compile_policy
from lox.envs.nethack import NetHackAdapter


def test_extract_hypothesis():
    agent = AuthorAgent(provider="mock")
    text = """
Here is my hypothesis:
```yaml
hypothesis:
  causal_finding: "EQUIPMENT_NEGLECT with 80% naked body armor"
  targeted_skill: "skill_scavenge_armor"
  mechanism: "Prioritize scavenge_loot before stairs down"
  predicted_outcome:
    target_metric: "avg_depth"
    expected_direction: "increase"
    min_improvement: 1.5
```

```python
def skill_scavenge_armor(self, obs):
    if obs.inventory.has_unworn_armor:
        obs = (yield wear_armor())
        return obs
    obs = (yield wait())
    return obs
```
"""
    hyp = agent.extract_hypothesis(text)
    assert hyp is not None
    assert hyp["targeted_skill"] == "skill_scavenge_armor"
    assert hyp["causal_finding"] == "EQUIPMENT_NEGLECT with 80% naked body armor"
    assert hyp["predicted_outcome"]["expected_direction"] == "increase"


def test_modular_starter_policy_compiles_and_runs():
    starter_path = Path("data/modular_starter_policy.py")
    assert starter_path.exists(), "Modular starter policy must exist"
    code = starter_path.read_text()
    executor = compile_policy(code)
    assert executor is not None

    # Test step execution on basic observation
    obs = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int16),
        hero=HeroState(y=10, x=10, hp=16, max_hp=16, depth=1, turn=10),
        spatial=SpatialView(has_unvisited_frontier=True),
    )
    runner = executor.create_runner(obs)
    action = runner.send(obs)
    assert isinstance(action, Action)
    assert action.name in ("step_to_frontier", "step_to_dead_end", "wait", "step_to_loot", "descend")


def test_macro_primitives_dispatch():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=101)
    assert adapter.current_strategic_goal == "explore"
    assert len(adapter.global_fountains) == 0

    # Test set_strategic_goal
    act_goal = Action(name="set_strategic_goal", goal_name="scavenge_armor")
    obs, _, _, _, info = adapter.step(act_goal)
    assert adapter.current_strategic_goal == "scavenge_armor"
    assert info.get("strategic_goal") == "scavenge_armor"

    # Test scavenge_loot fallback
    act_scavenge = Action(name="scavenge_loot")
    obs, _, _, _, _ = adapter.step(act_scavenge)
    assert obs is not None

    # Test backtrack_to_depth
    adapter.known_stairs_up = (obs.hero.y, obs.hero.x)
    act_backtrack = Action(name="backtrack_to_depth", target_depth=1)
    obs, _, _, _, _ = adapter.step(act_backtrack)
    assert obs is not None


def test_counterfactual_seed_determinism_and_paired_metrics():
    # 1. Deterministic seed replay in NetHackAdapter
    a1 = NetHackAdapter()
    obs1 = a1.reset(seed=9876)
    chars1 = a1._last_raw_obs["chars"].copy()
    a1.close()

    a2 = NetHackAdapter()
    obs2 = a2.reset(seed=9876)
    chars2 = a2._last_raw_obs["chars"].copy()
    a2.close()

    assert np.array_equal(chars1, chars2), "Seeded NetHack layout must be 100% deterministic"
    assert obs1.hero.y == obs2.hero.y and obs1.hero.x == obs2.hero.x

    # 2. Paired counterfactual metrics calculation
    seeds = [101, 102, 103, 104, 105]
    batch_g = {
        101: {"final_depth": 2, "root_cause": "COMBAT_FAST_PREDATOR"},
        102: {"final_depth": 3, "root_cause": "COMBAT_FAST_PREDATOR"},
        103: {"final_depth": 1, "root_cause": "STARVATION_FAINTING"},
        104: {"final_depth": 4, "root_cause": "ARMOR_DEFICIT"},
        105: {"final_depth": 2, "root_cause": "COMBAT_FAST_PREDATOR"},
    }
    batch_g1 = {
        101: {"final_depth": 5, "root_cause": "COMBAT_GENERAL"},
        102: {"final_depth": 6, "root_cause": "ARMOR_DEFICIT"},
        103: {"final_depth": 1, "root_cause": "STARVATION_FAINTING"},
        104: {"final_depth": 3, "root_cause": "COMBAT_GENERAL"},
        105: {"final_depth": 4, "root_cause": "ARMOR_DEFICIT"},
    }

    paired_deltas = [batch_g1[s]["final_depth"] - batch_g[s]["final_depth"] for s in seeds]
    # Deltas: [3, 3, 0, -1, 2] -> sum = 7, mean = 1.4
    assert np.isclose(np.mean(paired_deltas), 1.4)
    seeds_improved = sum(1 for d in paired_deltas if d > 0)
    seeds_regressed = sum(1 for d in paired_deltas if d < 0)
    seeds_unchanged = sum(1 for d in paired_deltas if d == 0)
    assert seeds_improved == 3
    assert seeds_regressed == 1
    assert seeds_unchanged == 1

    # Check incident resolution rate on COMBAT_FAST_PREDATOR
    failed_seeds = [s for s, r in batch_g.items() if r["root_cause"] == "COMBAT_FAST_PREDATOR"]
    # 101 (2->5), 102 (3->6), 105 (2->4): all 3 improved!
    resolved = sum(1 for s in failed_seeds if batch_g1[s]["final_depth"] > batch_g[s]["final_depth"])
    assert resolved == 3
    assert resolved / len(failed_seeds) == 1.0

