"""
Environment factory for NetHack Learning Environment (NLE).
"""

from typing import Any
import gymnasium as gym
import nle  # Register NLE environments
from lox.env.auto_more import AutoMoreWrapper


def make_env(
    env_id: str = "NetHackChallenge-v0",
    flight_recorder_capacity: int = 100,
    **kwargs: Any,
) -> AutoMoreWrapper:
    """
    Creates and returns a NetHack environment fully instrumented with
    AutoMoreWrapper, AnomalySentry, InventoryNormalizer, and FlightRecorder.
    """
    base_env = gym.make(env_id, **kwargs)
    wrapped_env = AutoMoreWrapper(
        base_env,
        flight_recorder_capacity=flight_recorder_capacity,
    )
    return wrapped_env
