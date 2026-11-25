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
from lox.env.benchmarks.nethack_scenarios import NetHackScenariosAdapter
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
    "nethack_scenarios": NetHackScenariosAdapter,
}


_ADAPTER_CACHE: dict[str, BenchmarkAdapter] = {}


def get_adapter(benchmark_name: str = "nethack", **kwargs: object) -> BenchmarkAdapter:
    """Instantiates (or returns cached default) BenchmarkAdapter for the given environment name."""
    key = benchmark_name.lower()
    if not kwargs and key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    cls = _ADAPTERS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. Registered: {list(_ADAPTERS.keys())}"
        )
    inst = cls(**kwargs)  # type: ignore[arg-type]
    if not kwargs:
        _ADAPTER_CACHE[key] = inst
    return inst


__all__ = [
    "ACTION_BY_CHAR",
    "ACTION_BY_NAME",
    "DIR_DELTAS",
    "Action",
    "ActionSequence",
    "BenchmarkAdapter",
    "EnvHarness",
    "FloorMemory",
    "GameState",
    "InventoryItem",
    "MonsterInfo",
    "NetHackAdapter",
    "NetHackHarness",
    "NetHackScenariosAdapter",
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
