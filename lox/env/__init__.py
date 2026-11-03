from lox.env.actions import (
    ACTION_BY_CHAR,
    ACTION_BY_NAME,
    DIR_DELTAS,
    Action,
    ActionSequence,
    RawAction,
    key,
    normalize_action_to_bytes,
    seq,
)
from lox.env.base import BenchmarkAdapter, EnvHarness
from lox.env.craftax import (
    CraftaxAction,
    CraftaxAdapter,
    CraftaxGameState,
    CraftaxHarness,
)
from lox.env.grid_utils import astar_next_step, bfs_distance_grid
from lox.env.harness import NetHackHarness
from lox.env.nethack import NetHackAdapter
from lox.env.state import (
    FloorMemory,
    GameState,
    InventoryItem,
    MonsterInfo,
    ObjectInfo,
    VisibleMonster,
    VisibleObject,
    WorldMemory,
)

_ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    "nethack": NetHackAdapter,
    "craftax": CraftaxAdapter,
}


def get_adapter(benchmark_name: str = "nethack", **kwargs: object) -> BenchmarkAdapter:
    """Instantiates a BenchmarkAdapter for the given environment name."""
    cls = _ADAPTERS.get(benchmark_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. Registered: {list(_ADAPTERS.keys())}"
        )
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ACTION_BY_CHAR",
    "ACTION_BY_NAME",
    "DIR_DELTAS",
    "Action",
    "ActionSequence",
    "BenchmarkAdapter",
    "CraftaxAction",
    "CraftaxAdapter",
    "CraftaxGameState",
    "CraftaxHarness",
    "EnvHarness",
    "FloorMemory",
    "GameState",
    "InventoryItem",
    "MonsterInfo",
    "NetHackAdapter",
    "NetHackHarness",
    "ObjectInfo",
    "RawAction",
    "VisibleMonster",
    "VisibleObject",
    "WorldMemory",
    "astar_next_step",
    "bfs_distance_grid",
    "get_adapter",
    "key",
    "normalize_action_to_bytes",
    "seq",
]
