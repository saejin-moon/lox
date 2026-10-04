class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.last_searched_pos = None
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble: Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Gnomish Mines Immediate Evacuation
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up:
                    obs = yield ascend()
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = yield step_to_stairs_up()
                    continue

            # 3. Combat Logic (Highest Priority)
            if obs.combat.adjacent_hostile or obs.combat.hostile_count_fov > 0 or obs.combat.has_active_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Proactive Hunger Prevention (Prevent Fainting)
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 5. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 6. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 7. Nearby Floor Loot Scooping
            if obs.spatial.has_nearby_loot:
                if not obs.status.is_encumbered or obs.status.encumbrance_level < 3:
                    obs = yield step_to_loot()
                    continue

            # 8. Altar BUC Testing
            if obs.dungeon.standing_on_altar and obs.epistemic.has_untested_items:
                obs = yield test_altar_buc()
                continue
            elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
                obs = yield step_to_altar()
                continue

            # 9. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 10. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                continue
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                    continue
                else:
                    obs = yield open_door()
                    continue
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
                continue
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
                continue
            else:
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # 0. Emergency Panic Escape (Teleport)
            if (obs.hero.hp_frac < 0.20 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 0.1 In-Combat Hunger Emergency (Prevent Fainting)
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth:
                    obs = yield eat_carried_food()
                    continue

            # 0.2 Major Trouble Divine Intervention
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state >= 3) and (obs.hero.turn - self.last_prayer_turn >= 150):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 1. Tactical Healing (Aggressive attrition prevention)
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 2. Non-Aggression
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 3. Passive Hazards (Floating Eye) - Ranged is safe at distance 1
            if obs.combat.adjacent_floating_eye:
                if obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue
                elif obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.combat.has_safe_melee_target:
                    obs = yield melee_attack_hostile()
                    continue
                elif obs.combat.can_retreat:
                    obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 3.1 Exploding Hazards (Gas Spore)
            if obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = yield melee_attack_hostile()
                    continue
                elif obs.combat.can_retreat:
                    obs = yield step_away_from_hostile()
                    continue
                elif not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue
                else:
                    obs = yield wait()
                    continue

            # 4. Panic Sanctuary (Elbereth)
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth) or obs.hero.hp_frac < 0.20:
                    obs = yield engrave_dust_elbereth()
                    continue

            # 5. Sanctuary Logic
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
                        continue

                if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                elif obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = yield zap_offensive_wand()
                        continue
                    elif obs.inventory.has_daggers:
                        obs = yield throw_dagger()
                        continue
                elif not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                    else:
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
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.50 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        if not obs.combat.standing_on_elbereth:
                            obs = yield engrave_dust_elbereth()
                            continue
                        obs = yield melee_attack_hostile()
                        continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # 8. Tactical Melee / Retreat
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.50 or self.retreat_streak >= 2 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                    continue
                
                if obs.combat.in_corridor:
                    obs = yield step_to_chokepoint()
                    self.retreat_streak += 1
                else:
                    if obs.hero.hp_frac < 0.30 and not obs.combat.standing_on_elbereth:
                        obs = yield engrave_dust_elbereth()
                    else:
                        obs = yield step_away_from_hostile()
                    self.retreat_streak += 1
            else:
                if obs.hero.hp_frac > 0.40:
                    obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                else:
                    obs = yield step_away_from_hostile()
            
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
                else:
                    obs = yield step_to(corpse.y, corpse.x)
                return obs
        obs = yield wait()
        return obs

    def handle_dead_end(self, obs):
        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end:
            if current_pos != getattr(self, "last_searched_pos", None):
                self.last_searched_pos = current_pos
                for _ in range(8):
                    if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                        return obs
                    if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                        break
                    obs = yield search()
                return obs
            else:
                if obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()
                elif obs.spatial.has_unvisited_frontier:
                    obs = yield step_to_frontier()
                else:
                    obs = yield step_away_from_hostile()
                return obs
        else:
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                return obs
            obs = yield step_to_dead_end()
        return obs
