class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Highest Priority)
            if obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 3 and not obs.inventory.has_food):
                if obs.hero.turn - self.last_prayer_turn >= 850:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Combat Logic: Immediate reaction to any hostile presence
            # We check for adjacent hostiles, active hostiles, or any hostile in FOV if we have ranged options.
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile or (obs.combat.hostile_count_fov > 0 and (obs.inventory.has_offensive_wand or obs.inventory.has_daggers)):
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Hunger Prevention (Safe zone only)
            if obs.hero.hunger_state >= 1:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif any(c.is_safe for c in obs.corpses):
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 4. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 5. Immediate Stair Descent
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                continue
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue

            # 6. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 7. Loot Scooping (Only if absolutely no hostiles in FOV)
            if obs.spatial.has_nearby_loot and obs.combat.hostile_count_fov == 0:
                obs = yield step_to_loot()
                continue

            # 8. Altar BUC Testing
            if obs.dungeon.standing_on_altar and obs.epistemic.has_untested_items:
                obs = yield test_altar_buc()
                continue
            elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
                obs = yield step_to_altar()
                continue

            # 9. Poison Resistance
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.9:
                obs = yield harvest_poison_res()
                continue

            # 10. Mines Avoidance
            if obs.hero.dungeon_branch == 'mines':
                if obs.spatial.standing_on_stairs_up:
                    obs = yield ascend()
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = yield step_to_stairs_up()
                    continue
                else:
                    obs = yield step_to_stairs_up()
                    continue

            # 11. Navigation & Exploration
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue
            elif obs.spatial.has_unvisited_frontier:
                self.search_count = 0
                obs = yield step_to_frontier()
                continue
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
                continue
            else:
                # Fallback to dead-end searching to find secret doors/stairs
                obs = yield from self.handle_dead_end(obs)
                continue

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # Exit condition: No immediate threats and no ranged options
            if not obs.combat.adjacent_hostile and not obs.combat.has_active_hostile:
                if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
                    break

            # 0. Panic Escape
            if (obs.hero.hp_frac < 0.2 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 1. Emergency Healing
            if obs.hero.hp_frac < 0.5 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 2. Divine Intervention
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state >= 3) and obs.hero.turn - self.last_prayer_turn >= 850:
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 3. Passive Hazards (Floating Eyes/Gas Spores)
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = yield melee_attack_hostile()
                    continue
                if obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue
                elif obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                else:
                    if not obs.combat.standing_on_elbereth:
                        obs = yield engrave_dust_elbereth()
                        continue
                    obs = yield step_away_from_hostile()
                    continue

            # 4. Peaceful NPCs
            if obs.combat.closest_hostile_name in ('shopkeeper', 'watchman', 'watch captain', 'guard', 'priest', 'priestess', 'oracle') or obs.dungeon.in_shop:
                obs = yield (retreat() if obs.combat.can_retreat else step_away_from_hostile())
                continue

            # 5. Elbereth Sanctuary Logic
            if not obs.combat.standing_on_elbereth:
                if obs.hero.hp_frac < 0.4 or (obs.hero.hp_frac < 0.6 and obs.combat.hostile_count_fov >= 2):
                    if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth):
                        obs = yield engrave_dust_elbereth()
                        continue

            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.4 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
                        continue
                if not obs.combat.adjacent_hostile:
                    if obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                        obs = yield zap_offensive_wand()
                        continue
                    if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                        obs = yield throw_dagger()
                        continue
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                        continue
                    obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 6. Ranged Harassment
            if obs.combat.closest_hostile_dist >= 2:
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue

            # 7. Fast Dangerous Attackers
            if obs.combat.is_fast_dangerous:
                if (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2) and not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.45 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
                        continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # 8. Standard Melee
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.45 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
            elif obs.hero.hp_frac > 0.4:
                obs = yield melee_attack_hostile()
            else:
                obs = yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())

            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = yield wait()
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = yield eat_floor_corpse()
                    return obs
                else:
                    obs = yield step_to(corpse.y, corpse.x)
                    return obs
        if obs.spatial.has_unvisited_frontier:
            obs = yield step_to_frontier()
        else:
            obs = yield step_to_dead_end()
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.standing_on_dead_end:
            for _ in range(12):
                if obs.combat.hostile_count_fov > 0:
                    return obs
                if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    return obs
                obs = yield search()
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            else:
                obs = yield step_to_dead_end()
            return obs
        else:
            obs = yield step_to_dead_end()
            return obs
