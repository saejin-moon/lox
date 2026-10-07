import time

import numpy as np

from lox.core.spatial import SpatialEngine


def test_distance_grid():
    walkable = np.ones((10, 10), dtype=bool)
    walkable[5, 1:9] = False  # Wall across middle with gaps at ends

    dist = SpatialEngine.distance_grid((0, 0), walkable)
    assert dist[0, 0] == 0
    assert dist[0, 1] == 1
    assert dist[5, 5] == -1  # Inside wall
    assert dist[9, 9] > 0  # Reachable around wall


def test_astar_pathfinding():
    walkable = np.ones((21, 79), dtype=bool)
    # Add a barrier
    walkable[10, 20:60] = False

    path = SpatialEngine.find_path((5, 30), (15, 30), walkable)
    assert len(path) > 0
    assert path[-1] == (15, 30)

    # Performance benchmark: Warmup + 100 queries
    t0 = time.perf_counter()
    for _ in range(100):
        SpatialEngine.find_path((5, 30), (15, 30), walkable)
    dt = (time.perf_counter() - t0) / 100.0

    print(f"Average pathfinding latency: {dt * 1e6:.1f} µs")
    assert dt < 0.001  # Less than 1 millisecond (typically 15-40 µs with JIT)


def test_frontier_discovery():
    walkable = np.zeros((21, 79), dtype=bool)
    visited = np.zeros((21, 79), dtype=bool)

    # 5x5 room
    walkable[2:7, 2:7] = True
    visited[2:7, 2:7] = True

    # Corridor exiting east
    walkable[4, 7:15] = True  # unvisited

    frontier = SpatialEngine.find_nearest_frontier((4, 4), walkable, visited)
    assert frontier == (4, 7)


def test_doorway_cardinal_movement_constraint():
    # Setup open space with a doorway at (5, 5)
    walkable = np.ones((10, 10), dtype=bool)
    is_door = np.zeros((10, 10), dtype=bool)
    is_door[5, 5] = True

    # Start diagonal to the door at (4, 4)
    # Goal is inside or past the door at (6, 5)
    path = SpatialEngine.find_path((4, 4), (6, 5), walkable, is_door=is_door)
    assert len(path) > 0
    # Every step into or out of (5, 5) must be cardinal (abs(dy) + abs(dx) == 1)
    curr = (4, 4)
    for step in path:
        if curr == (5, 5) or step == (5, 5):
            dy = abs(step[0] - curr[0])
            dx = abs(step[1] - curr[1])
            assert dy + dx == 1, (
                f"Movement between {curr} and {step} involving doorway (5, 5) must be cardinal!"
            )
        curr = step


def test_start_tile_departure_when_unwalkable():
    # Setup walkable grid where start tile (2, 2) is accidentally False
    walkable = np.ones((5, 5), dtype=bool)
    walkable[2, 2] = False
    target_mask = np.zeros((5, 5), dtype=bool)
    target_mask[2, 4] = True

    # Target search must still find reachable target from start (2, 2)
    target = SpatialEngine.find_nearest_target(
        (2, 2), walkable, target_mask=target_mask
    )
    assert target == (2, 4)

    # Distance grid must still compute distance from start (2, 2)
    dist = SpatialEngine.distance_grid((2, 2), walkable)
    assert dist[2, 2] == 0
    assert dist[2, 3] == 1
    assert dist[2, 4] == 2
