class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = 'explore_and_dive'

    def determine_goal(self, obs) -> str:
        if obs.inventory.has_amulet_of_yendor or obs.dungeon.phase == 'ascension_run':
            return 'ascension_run'
        if obs.dungeon.phase == 'invocation' or (obs.hero.depth >= 45 and obs.hero.turn > 15000):
            return 'perform_invocation'
        if obs.dungeon.phase == 'castle_breach' or (obs.hero.depth in (25, 26, 27) and obs.dungeon.is_castle_level):
            return 'breach_castle'
        if obs.dungeon.phase == 'sokoban' or obs.dungeon.is_sokoban_level:
            return 'solve_sokoban'
        if (obs.hero.hunger_state >= 2 and obs.inventory.has_food) or (
            obs.hero.hunger_state >= 4 and obs.hero.turn - self.last_prayer_turn >= 850
        ) or (
            obs.hero.hunger_state >= 2 and obs.dungeon.floor_corpse_adjacent and obs.dungeon.corpse_is_safe
        ):
            return 'handle_nutrition'
        if obs.combat.adjacent_hostile:
            return 'combat'
        if obs.inventory.has_unworn_armor:
            return 'scavenge_armor'
        if obs.dungeon.can_forge_excalibur and obs.dungeon.has_known_fountain:
            return 'forge_excalibur'
        return 'explore_and_dive'

    def run(self, obs):
        while True:
            if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
                obs = (yield descend())
                continue
            goal = self.determine_goal(obs)
            self.current_goal = goal
            if goal == 'handle_nutrition':
                obs = (yield from self.skill_handle_nutrition(obs))
                continue
            elif goal == 'ascension_run':
                obs = (yield from self.skill_ascension_run(obs))
                continue
            elif goal == 'perform_invocation':
                obs = (yield from self.skill_perform_invocation(obs))
                continue
            elif goal == 'breach_castle':
                obs = (yield from self.skill_breach_castle(obs))
                continue
            elif goal == 'solve_sokoban':
                obs = (yield from self.skill_solve_sokoban(obs))
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
        """Decisive Melee Engagement: Strike adjacent hostiles, but avoid passive hazards like floating eyes."""
        if obs.combat.adjacent_hostile:
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
                obs = (yield step_away_from_hostile())
                return obs
            if obs.hero.hp_frac > 0.35:
                obs = (yield melee_attack_hostile())
                return obs
            else:
                if not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    return obs
                obs = (yield melee_attack_hostile())
                return obs
        if obs.combat.hostile_count_fov > 0:
            obs = (yield wait())
            return obs
        obs = (yield wait())
        return obs

    def skill_scavenge_armor(self, obs):
        """Proactive Armor Acquisition: Wear armor immediately and loot nearby items."""
        if obs.inventory.has_unworn_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
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
        """Paced Exploration: Prioritize rapid descent and door navigation over wall searching."""
        if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        if obs.spatial.has_nearby_loot and obs.hero.depth >= 3:
            obs = (yield step_to_loot())
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

    def skill_handle_nutrition(self, obs):
        """Proactively consume food when hungry to prevent fainting."""
        if obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs
        if obs.dungeon.floor_corpse_adjacent and obs.dungeon.corpse_is_safe:
            obs = (yield eat_floor_corpse())
            return obs
        if obs.hero.hunger_state >= 4 and obs.hero.turn - self.last_prayer_turn >= 850:
            self.last_prayer_turn = obs.hero.turn
            obs = (yield pray())
            return obs
        obs = (yield wait())
        return obs
