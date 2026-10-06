class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000

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
            if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
                if not (obs.hero.dungeon_branch == 'sokoban' or obs.dungeon.is_sokoban):
                    sweeping_sokoban = 6 <= obs.hero.depth <= 10 and obs.hero.dungeon_branch == 'dungeon' and (not obs.dungeon.has_sokoban_entrance) and (not (obs.hero.has_reflection or obs.inventory.has_bag_of_holding)) and (obs.hero.hp_frac >= 0.5) and (obs.hero.hunger_state <= 2)
                    if not sweeping_sokoban:
                        obs = (yield descend())
                        continue
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
                obs = (yield from self.handle_combat(obs))
                continue
            if obs.hero.hunger_state >= 2:
                if any((c.is_safe for c in obs.corpses)):
                    obs = (yield from self.handle_corpse_consumption(obs))
                    continue
                elif obs.inventory.has_food:
                    obs = (yield eat_carried_food())
                    continue
            if obs.hero.hp_frac < 0.4 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue
            if obs.hero.dungeon_branch == 'sokoban' or obs.dungeon.is_sokoban:
                obs = (yield from self.phase_sokoban(obs))
                continue
            phase = getattr(obs, 'dungeon_phase', 'early_rush')
            if phase == 'early_rush' or obs.hero.depth <= 2:
                obs = (yield from self.phase_early_rush(obs))
            elif phase == 'early_scaling' or obs.hero.depth <= 5:
                obs = (yield from self.phase_early_scaling(obs))
            elif phase == 'mid_branches' or obs.hero.depth <= 10:
                obs = (yield from self.phase_mid_branches(obs))
            else:
                obs = (yield from self.phase_deep_dungeon(obs))

    def phase_sokoban(self, obs):
        """
        Special Branch: Sokoban (dnum == 4).
        Priority: Solve boulder puzzles, ascend levels 1-4,
        collect Reflection / Bag of Holding + rations, then descend back to Dungeons of Doom.
        """
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_body_armor and (not obs.inventory.has_worn_body_armor):
            obs = (yield wear_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.dungeon.can_solve_sokoban:
            obs = (yield step_solve_sokoban())
            return obs
        has_prize = obs.hero.has_reflection or obs.inventory.has_bag_of_holding
        if not has_prize and obs.hero.depth < 4:
            if obs.spatial.standing_on_stairs_up:
                obs = (yield ascend())
                return obs
            elif obs.spatial.stairs_up_known:
                obs = (yield step_to_stairs_up())
                return obs
            elif obs.spatial.has_unvisited_frontier:
                obs = (yield step_to_frontier())
                return obs
        if obs.spatial.standing_on_stairs_down:
            obs = (yield descend())
            return obs
        elif obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        elif obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_early_rush(self, obs):
        """
        Phase 0: Depths 1-2.
        Priority: Fast exploration & descent. Strict zero-loot staircase priority.
        """
        if obs.hero.hunger_state >= 1 and (not obs.inventory.has_food) and obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_body_armor and (not obs.inventory.has_worn_body_armor):
            obs = (yield wear_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
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
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_early_scaling(self, obs):
        """
        Phase 1: Depths 3-5.
        Priority: Character power-spiking. Forge Excalibur, upgrade starting armor to mithril/iron,
        harvest safe corpses for poison resistance, and test BUC at altars.
        """
        if obs.hero.hunger_state >= 1 and (not obs.inventory.has_food) and obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_body_armor and (not obs.inventory.has_worn_body_armor):
            obs = (yield wear_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
            if obs.dungeon.standing_on_fountain:
                obs = (yield dip_excalibur())
                return obs
            elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                obs = (yield step_to_fountain())
                return obs
        if obs.dungeon.standing_on_altar:
            if obs.dungeon.can_sacrifice:
                obs = (yield sacrifice_on_altar())
                return obs
            elif obs.epistemic.has_untested_items and (not self.altar_tested):
                obs = (yield test_altar_buc())
                self.altar_tested = True
                return obs
        elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items and (not self.altar_tested):
            obs = (yield step_to_altar())
            return obs
        if obs.dungeon.can_harvest_poison and (not obs.hero.has_poison_res) and (obs.hero.hp_frac > 0.85):
            obs = (yield harvest_poison_res())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
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
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_mid_branches(self, obs):
        """
        Phase 2: Depths 6-10.
        Priority: Branch discrimination & ascension tools.
        Thoroughly sweep floor to discover Sokoban (<) before descending (>).
        """
        if obs.dungeon.has_sokoban_entrance and obs.hero.hp_frac >= 0.6:
            if obs.spatial.standing_on_stairs_up and obs.hero.dungeon_branch != 'mines':
                obs = (yield ascend())
                return obs
            obs = (yield step_to_sokoban_entrance())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        has_prize = obs.hero.has_reflection or obs.inventory.has_bag_of_holding
        if not obs.dungeon.has_sokoban_entrance and (not has_prize) and (obs.hero.hp_frac >= 0.5) and (obs.hero.hunger_state <= 2):
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
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
            obs = (yield from self.handle_dead_end(obs))
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
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
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_deep_dungeon(self, obs):
        """
        Phase 3: Depths 11-19.
        Priority: High caution, corridor chokepoints, Quest readiness, and descent toward the Castle.
        """
        if obs.hero.dungeon_branch == 'mines':
            if obs.spatial.standing_on_stairs_up:
                obs = (yield ascend())
                return obs
            obs = (yield step_to_stairs_up())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and (not obs.dungeon.in_shop):
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
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            if obs.spatial.standing_on_stairs_down and (not obs.status.is_levitating):
                obs = (yield descend())
                break
            if not obs.combat.adjacent_hostile and (not obs.combat.has_active_hostile):
                if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
                    break
            if (obs.hero.hp_frac < 0.2 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = (yield read_scroll_teleport())
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = (yield zap_wand_teleport())
                    continue
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth or obs.hero.hunger_state >= 3:
                    obs = (yield eat_carried_food())
                    continue
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and (not obs.inventory.has_food))) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue
            if obs.hero.hp_frac < 0.4 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue
            if obs.hero.hp_frac < 0.2 and (not obs.inventory.has_healing) and (not obs.combat.has_panic_escape):
                if obs.inventory.has_unidentified_potion:
                    obs = (yield quaff_emergency_potion())
                    continue
                elif obs.inventory.has_unidentified_scroll:
                    obs = (yield read_emergency_scroll())
                    continue
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = (yield melee_attack_hostile())
                    continue
                if obs.combat.adjacent_floating_eye:
                    if obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                    elif obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                obs = (yield step_away_from_hostile())
                continue
            closest_name = obs.combat.closest_hostile_name.lower()
            if any((p in closest_name for p in ('shopkeeper', 'watchman', 'watch captain', 'guard', 'priest', 'priestess', 'oracle'))):
                if obs.combat.adjacent_hostile and obs.combat.can_retreat:
                    obs = (yield retreat())
                    continue
                elif obs.combat.adjacent_hostile:
                    obs = (yield step_away_from_hostile())
                    continue
                else:
                    break
            if obs.combat.is_corrosive_target:
                if obs.combat.closest_hostile_dist >= 1:
                    if obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                    elif obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.4:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))
                        continue
            if obs.combat.is_pack_threat and (not obs.combat.in_corridor):
                if not obs.combat.standing_on_elbereth and (not obs.combat.hostile_ignores_elbereth):
                    obs = (yield engrave_dust_elbereth())
                    continue
                elif obs.combat.closest_hostile_dist >= 2 and obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint())
                    continue
                elif obs.combat.adjacent_hostile:
                    obs = (yield melee_attack_hostile())
                    continue
            if obs.combat.is_heavy_weapon_threat:
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                    elif obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                if obs.combat.adjacent_hostile and obs.hero.hp_frac < 0.5 and obs.combat.can_retreat:
                    obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))
                    continue
            if (obs.hero.hp_frac < 0.35 or (obs.hero.hp_frac < 0.6 and (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2))) and (not obs.combat.standing_on_elbereth):
                obs = (yield engrave_dust_elbereth())
                continue
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.4 or not obs.combat.can_retreat:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))
                        continue
                if obs.hero.hp_frac < 0.3 and obs.hero.turn - self.last_prayer_turn >= 850:
                    self.last_prayer_turn = obs.hero.turn
                    obs = (yield pray())
                    continue
                elif obs.hero.hp_frac < 0.6 and obs.inventory.has_healing:
                    obs = (yield quaff_healing())
                    continue
                elif obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                    obs = (yield zap_offensive_wand())
                    continue
                elif obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = (yield throw_dagger())
                    continue
                elif not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = (yield step_to_stairs_down())
                    elif obs.spatial.has_unvisited_frontier:
                        obs = (yield step_to_frontier())
                    else:
                        obs = (yield step_away_from_hostile())
                    continue
                else:
                    obs = (yield melee_attack_hostile())
                    continue
            if obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                obs = (yield zap_offensive_wand())
                continue
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = (yield throw_dagger())
                continue
            if obs.combat.is_fast_dangerous:
                if (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2) and (not obs.combat.standing_on_elbereth):
                    obs = (yield engrave_dust_elbereth())
                    continue
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.4 or not obs.combat.can_retreat:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        if not obs.combat.standing_on_elbereth:
                            obs = (yield engrave_dust_elbereth())
                        else:
                            obs = (yield melee_attack_hostile())
                        continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint())
                    continue
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.35 or not obs.combat.can_retreat:
                    obs = (yield melee_attack_hostile())
                elif not obs.combat.standing_on_elbereth and (not obs.combat.hostile_ignores_elbereth):
                    obs = (yield engrave_dust_elbereth())
                else:
                    obs = (yield melee_attack_hostile())
                continue
            elif obs.hero.hp_frac < 0.4 and (not obs.combat.in_corridor) and obs.combat.can_retreat:
                obs = (yield step_to_chokepoint())
            else:
                obs = (yield melee_attack_hostile())
            if obs.combat.hostile_count_fov == 0 and (not obs.combat.adjacent_hostile):
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = (yield wait())
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = (yield eat_floor_corpse())
                    return obs
                obs = (yield step_to(corpse.y, corpse.x))
                return obs
        obs = (yield (step_to_frontier() if obs.spatial.has_unvisited_frontier else step_to_dead_end()))
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.standing_on_dead_end:
            prev_hp = obs.hero.hp
            search_limit = 20 if obs.hero.depth <= 2 else 12
            for _ in range(search_limit):
                if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile or obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier or (obs.hero.hp < prev_hp):
                    return obs
                obs = (yield search())
            if obs.spatial.stairs_down_known:
                obs = (yield step_to_stairs_down())
            elif obs.spatial.has_unvisited_frontier:
                obs = (yield step_to_frontier())
            elif obs.dungeon.has_closed_door:
                obs = (yield step_to_closed_door())
            else:
                obs = (yield step_to_dead_end())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs
