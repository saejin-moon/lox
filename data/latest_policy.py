class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = 'explore_and_dive'
        self.poison_res_acquired = False

    def determine_goal(self, obs) -> str:
        if obs.hero.depth <= 2 and obs.spatial.stairs_down_known:
            return 'early_rush'
        if obs.hero.depth <= 5 and (obs.inventory.has_unworn_armor or obs.dungeon.can_forge_excalibur):
            return 'early_scaling'
        if obs.hero.depth <= 10 and obs.dungeon.has_sokoban_entrance:
            return 'mid_branches'
        if obs.hero.depth >= 11:
            return 'deep_dungeon'
        if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
            return 'combat'
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            return 'handle_nutrition'
        if obs.inventory.has_unworn_armor:
            return 'scavenge_armor'
        if obs.dungeon.can_forge_excalibur and (obs.dungeon.standing_on_fountain or obs.dungeon.has_known_fountain):
            return 'forge_excalibur'
        if obs.dungeon.standing_on_altar or obs.dungeon.adjacent_altar:
            return 'altar_sacrifice'
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
            if goal == 'early_rush':
                obs = (yield from self.skill_early_rush(obs))
                continue
            elif goal == 'early_scaling':
                obs = (yield from self.skill_early_scaling(obs))
                continue
            elif goal == 'mid_branches':
                obs = (yield from self.skill_mid_branches(obs))
                continue
            elif goal == 'deep_dungeon':
                obs = (yield from self.skill_deep_dungeon(obs))
                continue
            elif goal == 'handle_nutrition':
                obs = (yield from self.skill_handle_nutrition(obs))
                continue
            elif goal == 'combat':
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == 'scavenge_armor':
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == 'forge_excalibur':
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            elif goal == 'altar_sacrifice':
                obs = (yield from self.skill_altar_sacrifice(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        if obs.combat.adjacent_floating_eye:
            if obs.inventory.has_daggers:
                obs = (yield throw_dagger())
                return obs
            if obs.inventory.has_offensive_wand:
                obs = (yield zap_offensive_wand())
                return obs
            obs = (yield step_away_from_hostile())
            return obs
        if obs.combat.adjacent_hostile:
            if obs.hero.hp_frac > 0.35 or not obs.combat.can_retreat:
                obs = (yield melee_attack_hostile())
                return obs
            if not obs.combat.standing_on_elbereth:
                obs = (yield engrave_dust_elbereth())
                return obs
            obs = (yield melee_attack_hostile())
            return obs
        if obs.combat.is_fast_dangerous:
            if obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2:
                if not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    return obs
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.4:
                    obs = (yield melee_attack_hostile())
                    return obs
            if obs.combat.can_retreat:
                obs = (yield step_to_chokepoint())
                return obs
        if obs.combat.hostile_count_fov > 0:
            if obs.combat.adjacent_hostile:
                obs = (yield melee_attack_hostile())
                return obs
            if obs.inventory.has_daggers:
                obs = (yield throw_dagger())
                return obs
            if obs.inventory.has_offensive_wand:
                obs = (yield zap_offensive_wand())
                return obs
            obs = (yield step_away_from_hostile())
            return obs
        obs = (yield wait())
        return obs

    def skill_scavenge_armor(self, obs):
        if obs.inventory.has_unworn_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_forge_excalibur(self, obs):
        if obs.dungeon.standing_on_fountain:
            if obs.hero.hp_frac >= 0.85 and obs.dungeon.can_forge_excalibur:
                obs = (yield dip_excalibur())
                return obs
        if obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
            obs = (yield step_to_fountain())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_explore_and_dive(self, obs):
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

    def skill_early_rush(self, obs):
        if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known:
            if obs.spatial.has_nearby_loot:
                obs = (yield step_to_loot())
                return obs
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield search())
        return obs

    def skill_early_scaling(self, obs):
        if obs.inventory.has_unworn_armor:
            obs = (yield wear_armor())
            return obs
        if obs.dungeon.can_forge_excalibur and obs.dungeon.standing_on_fountain:
            if obs.hero.hp_frac >= 0.85:
                obs = (yield dip_excalibur())
                return obs
        if not self.poison_res_acquired and obs.corpses:
            for corpse in obs.corpses:
                if corpse.is_poisonous:
                    obs = (yield eat_floor_corpse())
                    self.poison_res_acquired = True
                    return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_mid_branches(self, obs):
        if obs.dungeon.has_sokoban_entrance:
            obs = (yield step_to_sokoban_entrance())
            return obs
        if obs.hero.dungeon_branch == 'mines':
            if obs.spatial.stairs_up_known:
                obs = (yield step_to_stairs_up())
                return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_deep_dungeon(self, obs):
        if obs.combat.is_fast_dangerous:
            if obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2:
                if not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    return obs
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.4:
                    obs = (yield melee_attack_hostile())
                    return obs
            if obs.combat.can_retreat:
                obs = (yield step_to_chokepoint())
                return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_handle_nutrition(self, obs):
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs
        if obs.hero.hunger_state >= 4 and (not obs.inventory.has_food):
            if obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850:
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_altar_sacrifice(self, obs):
        if obs.dungeon.standing_on_altar or obs.dungeon.adjacent_altar:
            if obs.corpses:
                for corpse in obs.corpses:
                    if corpse.is_fresh:
                        obs = (yield sacrifice_on_altar())
                        return obs
            if obs.inventory.has_corpse:
                obs = (yield sacrifice_on_altar())
                return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs
