"""Combat subsystem: tactical responses and threat scanning mixins.

`TacticalCombatManager` itself lives in `lox/domain/combat_manager.py` (the
canonical import path) and composes these mixins — behavior-preserving split.
"""

from lox.domain.combat.threat_scan import MonsterTrack, ThreatScanMixin
from lox.domain.combat.responses import CombatResponsesMixin


def __getattr__(name):
    if name == "TacticalCombatManager":
        from lox.domain.combat_manager import TacticalCombatManager
        return TacticalCombatManager
    raise AttributeError(name)

__all__ = ["MonsterTrack", "ThreatScanMixin", "CombatResponsesMixin", "TacticalCombatManager"]
