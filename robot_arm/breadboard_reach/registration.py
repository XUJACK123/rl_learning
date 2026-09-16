"""Gymnasium registration for the breadboard task."""

from __future__ import annotations


ENV_ID = "BreadboardReach-v0"


def register_env() -> str:
    """Register the task lazily so geometry utilities need no simulator import."""
    import gymnasium as gym

    if ENV_ID not in gym.envs.registry:
        gym.register(id=ENV_ID, entry_point="breadboard_reach.env:BreadboardReachEnv")
    return ENV_ID
