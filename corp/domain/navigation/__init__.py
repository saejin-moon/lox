"""Navigation subsystem: level maps, stepping/door primitives, exploration mixins.

`NavigationManager` lives in `corp/domain/navigation_manager.py` (the canonical
import path) and composes these mixins — behavior-preserving split.
"""

from corp.domain.navigation.level_map import LevelMap, LevelStore
from corp.domain.navigation.stepping import SteppingMixin
from corp.domain.navigation.exploration import ExplorationMixin


def __getattr__(name):
    # Convenience re-export of the composed manager without a circular import
    # (navigation_manager imports this package's mixins at module load).
    if name == "NavigationManager":
        from corp.domain.navigation_manager import NavigationManager
        return NavigationManager
    raise AttributeError(name)

__all__ = ["LevelMap", "LevelStore", "SteppingMixin", "ExplorationMixin", "NavigationManager"]
