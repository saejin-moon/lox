"""
Frontier Exploration: Identifies unexplored boundaries and secret door candidates on 80x21 grid.
"""

from collections import deque
from typing import Optional, List, Tuple
import numpy as np
import numpy.typing as npt


class FrontierExplorer:
    """
    Identifies unexplored boundary tiles and schedules BFS exploration goals.
    """

    ROWS = 21
    COLS = 79

    CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ALL_NEIGHBORS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    @classmethod
    def get_frontier_mask(
        cls,
        walkable_mask: npt.NDArray[np.bool_],
        unmapped_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.bool_]:
        """
        Returns a boolean mask where True indicates a walkable tile adjacent to unmapped space.
        """
        frontier = np.zeros((cls.ROWS, cls.COLS), dtype=bool)

        for r in range(cls.ROWS):
            for c in range(cls.COLS):
                if not walkable_mask[r, c]:
                    continue

                # Check if any neighbor is unmapped
                has_unmapped_neighbor = False
                for dr, dc in cls.ALL_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cls.ROWS and 0 <= nc < cls.COLS:
                        if unmapped_mask[nr, nc]:
                            has_unmapped_neighbor = True
                            break

                if has_unmapped_neighbor:
                    frontier[r, c] = True

        return frontier

    @classmethod
    def find_nearest_frontier(
        cls,
        start: tuple[int, int],
        walkable_mask: npt.NDArray[np.bool_],
        unmapped_mask: npt.NDArray[np.bool_],
    ) -> Optional[Tuple[int, int]]:
        """
        Runs BFS from player start position to find the closest reachable frontier tile.
        """
        frontier_mask = cls.get_frontier_mask(walkable_mask, unmapped_mask)

        start_r, start_c = start
        if frontier_mask[start_r, start_c]:
            return start

        queue = deque([start])
        visited = {start}

        while queue:
            curr_r, curr_c = queue.popleft()

            if frontier_mask[curr_r, curr_c]:
                return (curr_r, curr_c)

            for dr, dc in cls.ALL_NEIGHBORS:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < cls.ROWS and 0 <= nc < cls.COLS:
                    if walkable_mask[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return None

    @classmethod
    def is_corridor_dead_end(
        cls,
        pos: tuple[int, int],
        walkable_mask: npt.NDArray[np.bool_],
    ) -> bool:
        """
        Returns True if pos is a walkable tile with exactly 1 walkable neighbor.
        Dead-end corridors are prime candidates for secret doors in NetHack.
        """
        r, c = pos
        if not walkable_mask[r, c]:
            return False

        walkable_neighbors = 0
        for dr, dc in cls.CARDINALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cls.ROWS and 0 <= nc < cls.COLS:
                if walkable_mask[nr, nc]:
                    walkable_neighbors += 1

        return walkable_neighbors <= 1
