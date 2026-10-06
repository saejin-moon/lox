class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore_and_dive"

    def determine_goal(self, obs) -> str:
        """
        Determines the active strategic goal:
        - 'combat': adjacent hostile
        - 'explore_and_dive': default exploration
        """
        if obs.combat.adjacent_hostile:
            return "combat"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Universal Reflex: Immediate Descent on stairs down
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                continue

            goal = self.determine_goal(obs)
            self.current_goal = goal

            if goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == "scavenge_armor":
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == "forge_excalibur":
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        """
        Primitive Combat: Attack adjacent hostiles in melee, otherwise wait.
        """
        if obs.combat.adjacent_hostile:
            obs = (yield melee_attack_hostile())
            return obs
        obs = (yield wait())
        return obs

    def skill_scavenge_armor(self, obs):
        """
        Primitive Scavenge: Fall back to exploration.
        """
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_forge_excalibur(self, obs):
        """
        Primitive Forge Excalibur: Fall back to exploration.
        """
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_explore_and_dive(self, obs):
        """
        Primitive Exploration: Move toward known stairs down, frontiers, or dead ends.
        """
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs

        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs

        obs = (yield step_to_dead_end())
        return obs
