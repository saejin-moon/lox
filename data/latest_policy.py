class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = 'explore_and_dive'

    def determine_goal(self, obs) -> str:
        if obs.spatial.stairs_down_known and (obs.hero.turns_on_level >= 100 or obs.hero.hunger_state >= 2 or (not obs.spatial.has_unvisited_frontier)):
            return 'rapid_descent'
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            return 'nutrition_management'
        if obs.combat.adjacent_hostile:
            return 'combat_decisive'
        if obs.hero.depth <= 5 and obs.hero.ac >= 6 and (obs.inventory.has_unworn_armor or obs.spatial.has_nearby_loot):
            return 'scavenge_armor'
        if obs.dungeon.can_forge_excalibur and (obs.dungeon.standing_on_fountain or obs.dungeon.has_known_fountain):
            return 'forge_excalibur'
        return 'explore_and_dive'

    def run(self, obs):
        while True:
            if obs.hero.dungeon_branch == 'mines':
                if obs.spatial.standing_on_stairs_up:
                    obs = (yield ascend())
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = (yield step_to_stairs_up())
                    continue
                else:
                    obs = (yield step_to_frontier())
                    continue
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and (not obs.inventory.has_food))) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue
            goal = self.determine_goal(obs)
            self.current_goal = goal
            if goal == 'rapid_descent':
                obs = (yield from self.skill_rapid_descent(obs))
                continue
            elif goal == 'nutrition_management':
                obs = (yield from self.skill_nutrition_management(obs))
                continue
            elif goal == 'combat_decisive':
                obs = (yield from self.skill_combat_decisive(obs))
                continue
            elif goal == 'scavenge_armor':
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == 'forge_excalibur':
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            elif goal == 'explore_and_dive':
                obs = (yield from self.skill_explore_and_dive(obs))
                continue
            skill_fn = getattr(self, 'skill_' + goal, None)
            if skill_fn is not None:
                obs = (yield from skill_fn(obs))
                continue
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
        """Prioritize body armor acquisition to achieve AC <= 2 before descending."""
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_forge_excalibur(self, obs):
        """Forge Excalibur at DL 5+ when fountain available."""
        if not obs.dungeon.standing_on_fountain:
            if obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                obs = (yield step_to_fountain())
                return obs
            obs = (yield step_to_frontier())
            return obs
        if obs.hero.hp_frac >= 0.85 and obs.dungeon.can_forge_excalibur:
            obs = (yield dip_excalibur())
            return obs
        obs = (yield step_to_stairs_down())
        return obs

    def skill_explore_and_dive(self, obs):
        """Restructured exploration: descend immediately when stairs known and pacing conditions met, eliminating wall-searching stalls."""
        should_descend = obs.spatial.stairs_down_known and (obs.hero.turns_on_level >= 100 or not obs.spatial.has_unvisited_frontier or obs.hero.hunger_state >= 2 or (obs.hero.depth >= 3))
        if should_descend:
            obs = (yield step_to_stairs_down())
            return obs
        if not obs.spatial.stairs_down_known:
            if obs.spatial.has_unvisited_frontier:
                obs = (yield step_to_frontier())
                return obs
            if obs.spatial.has_nearby_loot:
                obs = (yield step_to_loot())
                return obs
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
                    obs = (yield kick_closed_door())
                else:
                    obs = (yield open_door())
                return obs
            if obs.dungeon.has_closed_door:
                obs = (yield step_to_closed_door())
                return obs
            if obs.spatial.has_unsearched_dead_end:
                obs = (yield step_to_dead_end())
                return obs
            obs = (yield search())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        if obs.spatial.has_unsearched_dead_end:
            obs = (yield step_to_dead_end())
            return obs
        obs = (yield search())
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

    def skill_rapid_descent(self, obs):
        """Aggressive floor-to-floor progression: descend immediately when stairs known and pacing conditions met."""
        if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
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

    def skill_nutrition_management(self, obs):
        """Proactive nutrition: eat when hunger_state >= 2, not when fainting."""
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs
        if obs.hero.hunger_state >= 4 and obs.hero.can_pray:
            obs = (yield pray())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_combat_decisive(self, obs):
        """Decisive combat: melee adjacent hostiles at hp_frac > 0.35, never retreat from adjacent in open rooms. Handle floating eyes with ranged attacks."""
        if obs.combat.adjacent_hostile:
            if obs.combat.adjacent_floating_eye:
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
                obs = (yield step_away_from_hostile())
                return obs
            if obs.combat.adjacent_gas_spore:
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
                obs = (yield step_away_from_hostile())
                return obs
            if obs.hero.hp_frac > 0.35 or not obs.combat.can_retreat:
                obs = (yield melee_attack_hostile())
                return obs
            if not obs.combat.standing_on_elbereth and (not obs.combat.hostile_ignores_elbereth):
                obs = (yield engrave_dust_elbereth())
                return obs
            obs = (yield melee_attack_hostile())
            return obs
        if obs.combat.hostile_count_fov > 0 and obs.combat.is_fast_dangerous:
            if obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2:
                if not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    return obs
            if obs.combat.closest_hostile_dist >= 2 and obs.combat.can_retreat:
                obs = (yield step_to_chokepoint())
                return obs
        obs = (yield wait())
        return obs
