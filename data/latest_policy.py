class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore_and_dive"

    def determine_goal(self, obs) -> str:
        # Phase 1: Endgame Ascension run
        if obs.inventory.has_amulet_of_yendor or obs.dungeon.phase == "ascension_run":
            return "ascension_run"
        # Phase 2: Invocations & Sanctum
        if obs.dungeon.phase == "invocation" or (obs.hero.depth >= 45 and obs.hero.turn > 15000):
            return "perform_invocation"
        # Phase 3: Castle Drawbridge Breach
        if obs.dungeon.phase == "castle_breach" or (obs.hero.depth in (25, 26, 27) and obs.dungeon.is_castle_level):
            return "breach_castle"
        # Phase 4: Sokoban Solver
        if obs.dungeon.phase == "sokoban" or obs.dungeon.is_sokoban_level:
            return "solve_sokoban"
        if obs.combat.adjacent_hostile:
            return "combat"
        if obs.inventory.has_unworn_armor:
            return "scavenge_armor"
        if obs.dungeon.can_forge_excalibur and obs.dungeon.has_known_fountain:
            return "forge_excalibur"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Universal Reflex: Immediate Descent on stairs down
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                continue

            goal = self.determine_goal(obs)
            self.current_goal = goal

            if goal == "ascension_run":
                obs = (yield from self.skill_ascension_run(obs))
                continue
            elif goal == "perform_invocation":
                obs = (yield from self.skill_perform_invocation(obs))
                continue
            elif goal == "breach_castle":
                obs = (yield from self.skill_breach_castle(obs))
                continue
            elif goal == "solve_sokoban":
                obs = (yield from self.skill_solve_sokoban(obs))
                continue
            elif goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == "scavenge_armor":
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == "forge_excalibur":
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            elif goal == "explore_and_dive":
                obs = (yield from self.skill_explore_and_dive(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        """Primitive Combat: Attack adjacent hostiles in melee, otherwise wait."""
        if obs.combat.adjacent_hostile:
            obs = (yield melee_attack_hostile())
            return obs
        obs = (yield wait())
        return obs

    def skill_scavenge_armor(self, obs):
        """Primitive Scavenge: Wear unworn armor if available, otherwise explore."""
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_forge_excalibur(self, obs):
        """Primitive Forge Excalibur: Dip long sword in fountain if standing on it, otherwise explore."""
        if obs.dungeon.standing_on_fountain:
            obs = (yield dip_excalibur())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_explore_and_dive(self, obs):
        """Primitive Exploration: Move toward known stairs down, frontiers, or dead ends."""
        if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_solve_sokoban(self, obs):
        if obs.dungeon.can_solve_sokoban or obs.dungeon.is_sokoban:
            obs = (yield solve_sokoban())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_breach_castle(self, obs):
        if obs.dungeon.can_breach_drawbridge or obs.dungeon.drawbridge_in_fov:
            obs = (yield breach_drawbridge())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_perform_invocation(self, obs):
        if obs.dungeon.standing_on_vibrating_square or obs.dungeon.can_perform_invocation:
            obs = (yield perform_invocation_step())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_ascension_run(self, obs):
        if obs.dungeon.standing_on_high_altar:
            obs = (yield offer_amulet_on_altar())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs
