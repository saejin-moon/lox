def emergency():
    if hunger_state >= WEAK and has_carried_food:
        eat_carried_food()
    elif hp_frac < 0.10 and can_safely_pray:
        pray()
    elif hp_frac < 0.25 and has_healing:
        quaff_healing()
    elif hunger_state >= HUNGRY and has_carried_food:
        eat_carried_food()

def recovery():
    if is_surrounded:
        step_away_from_hostile()
    elif adjacent_hostile and hp_frac < 0.40:
        step_away_from_hostile()
    elif standing_on_elbereth and hostile_count_fov > 0:
        wait()

def combat():
    if adjacent_hostile:
        if hp_frac < 0.30:
            step_away_from_hostile()
        else:
            melee_attack_hostile()
    elif hostile_count_fov > 0:
        if in_corridor:
            step_to_chokepoint()
        else:
            step_to_frontier()

def navigation():
    if standing_on_stairs_down:
        descend()
    elif stairs_down_known:
        step_to_stairs_down()
    elif depth > 1 and hp_frac < 0.20 and standing_on_stairs_up:
        ascend()
    elif depth > 1 and hp_frac < 0.20 and stairs_up_known:
        step_to_stairs_up()

def maintenance():
    if adjacent_closed_door:
        open_door()
    elif floor_corpse_adjacent and corpse_is_safe:
        eat_floor_corpse()

def explore():
    if has_unvisited_frontier:
        step_to_frontier()
    elif has_unsearched_dead_end and hostile_count_fov == 0:
        search()
    else:
        wait()

plan = [
    emergency,
    recovery,
    combat,
    navigation,
    maintenance,
    explore,
]
