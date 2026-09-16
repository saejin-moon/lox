"""
Deliberative LLM Rate Limiting, Budgeting, and Gating Engine.
Protects against API rate-limit exhaustion (HTTP 429) and game-loop latency stalls
via multi-tiered composite throttling.
"""

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass(slots=True)
class ThrottlerConfig:
    """Configuration parameters for deliberative LLM rate limiting."""
    max_rpm: float = 10.0                 # Max allowable requests per minute
    min_wall_seconds: float = 6.0          # Minimum wall-clock gap between queries
    min_turn_gap: int = 100                # Minimum turns between in-game queries
    max_in_game_per_episode: int = 3       # Hard quota on in-game deliberative interventions
    max_autopsies_per_episode: int = 1     # Hard quota on post-mortem autopsies
    escalation_threshold: int = 2          # Symbolic escape attempts required before LLM escalation
    cache_capacity: int = 128              # Maximum cached patch signatures


class LLMRateThrottler:
    """
    Multi-tiered gating system controlling access to deliberative LLM backends.
    
    Tiers enforced:
    1. Hierarchical Escalation: Requires N symbolic fallback attempts before LLM escalation.
    2. Per-Episode Quota: Hard caps on in-game queries and dedicated autopsy reservations.
    3. Dual Cooldown: Enforces both wall-clock minimum seconds and game-turn spacing.
    4. State Fingerprint Cache: Deduplicates identical spatial/tactical deadlocks.
    """

    def __init__(self, config: ThrottlerConfig | None = None):
        self.config = config or ThrottlerConfig()
        self.in_game_count = 0
        self.autopsy_count = 0
        self.last_query_wall = 0.0
        self.last_query_turn = -1
        self.stall_counts: dict[str, int] = {}
        self.cache: dict[int, Any] = {}
        self._cache_keys: list[int] = []

    def reset_episode(self):
        """Resets episode-specific counters while preserving inter-generational caches."""
        self.in_game_count = 0
        self.autopsy_count = 0
        self.stall_counts.clear()

    def record_stall(self, task_name: str) -> int:
        """Records a cyclic stall for a specific task and returns current consecutive count."""
        count = self.stall_counts.get(task_name, 0) + 1
        self.stall_counts[task_name] = count
        return count

    def clear_stall(self, task_name: str):
        """Clears the consecutive stall count when progress is observed."""
        self.stall_counts.pop(task_name, None)

    def get_cached_patch(self, state_hash: int | None) -> Any | None:
        """Retrieves a cached patch if available for this state signature."""
        if state_hash is None:
            return None
        return self.cache.get(state_hash)

    def store_cached_patch(self, state_hash: int | None, patch: Any):
        """Stores a validated patch in the LRU state cache."""
        if state_hash is None or patch is None:
            return
        if state_hash not in self.cache:
            if len(self._cache_keys) >= self.config.cache_capacity:
                oldest = self._cache_keys.pop(0)
                self.cache.pop(oldest, None)
            self._cache_keys.append(state_hash)
        self.cache[state_hash] = patch

    def should_allow_query(
        self,
        trigger_type: str,
        current_turn: int,
        task_name: str = "",
        state_hash: int | None = None,
    ) -> tuple[bool, str]:
        """
        Evaluates all throttling tiers.
        Returns: (allowed: bool, reason: str)
        """
        now = time.time()

        # 1. State Cache Check
        if state_hash is not None and state_hash in self.cache:
            return False, "cached_patch_available"

        # 2. Autopsy Lane (Special reserved lane)
        if trigger_type == "autopsy":
            if self.autopsy_count >= self.config.max_autopsies_per_episode:
                return False, f"autopsy_quota_exceeded ({self.autopsy_count}/{self.config.max_autopsies_per_episode})"
            # Enforce wall-clock spacing to prevent back-to-back episode bursts
            wall_gap = now - self.last_query_wall
            if self.last_query_wall > 0 and wall_gap < self.config.min_wall_seconds:
                return False, f"wall_cooldown_active ({wall_gap:.2f}s < {self.config.min_wall_seconds}s)"
            return True, "autopsy_granted"

        # 3. In-Game Deliberative Gating
        # 3a. Episode Quota Check
        if self.in_game_count >= self.config.max_in_game_per_episode:
            return False, f"in_game_quota_exceeded ({self.in_game_count}/{self.config.max_in_game_per_episode})"

        # 3b. Hierarchical Escalation Check
        stalls = self.stall_counts.get(task_name, 0)
        if stalls < self.config.escalation_threshold:
            return False, f"hierarchical_escalation_required ({stalls}/{self.config.escalation_threshold})"

        # 3c. Game-Turn Interval Spacing
        if self.last_query_turn >= 0:
            turn_gap = current_turn - self.last_query_turn
            if turn_gap < self.config.min_turn_gap:
                return False, f"turn_cooldown_active ({turn_gap} turns < {self.config.min_turn_gap})"

        # 3d. Wall-Clock Rate Limiting
        wall_gap = now - self.last_query_wall
        if self.last_query_wall > 0 and wall_gap < self.config.min_wall_seconds:
            return False, f"wall_cooldown_active ({wall_gap:.2f}s < {self.config.min_wall_seconds}s)"

        return True, "in_game_deliberation_granted"

    def record_query_dispatched(self, trigger_type: str, current_turn: int):
        """Updates internal rate tracking timestamps and counters upon query dispatch."""
        self.last_query_wall = time.time()
        self.last_query_turn = current_turn
        if trigger_type == "autopsy":
            self.autopsy_count += 1
        else:
            self.in_game_count += 1
