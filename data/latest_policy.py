def emergency():
    if hp_frac < 0.25 and can_safely_pray:
        pray()
    elif hp_frac < 0.50 and has_healing:
        quaff_healing()
    elif hunger_state >= HUNGRY and has_carried_food:
        eat_carried_food()

def combat():
    if adjacent_hostile:
        melee_attack_hostile()
    elif hostile_count_fov > 0:
        step_to_chokepoint()

def navigation():
    if standing_on_stairs_down:
        descend()
    elif stairs_down_known:
        step_to_stairs_down()
    elif standing_on_stairs_up:
        ascend()
    elif stairs_up_known:
        step_to_stairs_up()

def maintenance():
    if adjacent_closed_door:
        open_door()
    elif floor_corpse_adjacent and corpse_is_safe:
        eat_floor_corpse()

def explore():
    if has_unvisited_frontier:
        step_to_frontier()
    elif has_unsearched_dead_end:
        search()
    elif stairs_down_known:
        step_to_stairs_down()
    else:
        wait()

def recovery():
    if adjacent_hostile and can_retreat:
        step_away_from_hostile()
    else:
        wait()

plan = [
    emergency,
    combat,
    navigation,
    maintenance,
    explore,
    recovery,
]
