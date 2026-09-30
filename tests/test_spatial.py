import time
import numpy as np
import pytest
from lox.core.spatial import SpatialEngine


def test_distance_grid():
    walkable = np.ones((10, 10), dtype=bool)
    walkable[5, 1:9] = False  # Wall across middle with gaps at ends

    dist = SpatialEngine.distance_grid((0, 0), walkable)
    assert dist[0, 0] == 0
    assert dist[0, 1] == 1
    assert dist[5, 5] == -1  # Inside wall
    assert dist[9, 9] > 0    # Reachable around wall


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
