"""Episode-local future goal relabeling before RLinf transition replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .geometry import pose_error


Observation = dict[str, np.ndarray]


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: np.ndarray
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    collision: bool
    episode_id: int
    step_index: int
    shielded: bool = False
    future_index: int | None = None


def _copy_with_goal(observation: Observation, goal: np.ndarray) -> Observation:
    copied = {key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()}
    copied["desired_goal"] = np.asarray(goal, dtype=np.float32).copy()
    return copied


def relabel_episode(
    episode: list[Transition],
    *,
    future_k: int,
    rng: np.random.Generator,
    reward_fn: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
) -> list[Transition]:
    """Keep originals and add k future-goal copies of every transition.

    A future state is selected from the same completed episode. The physical
    collision and timeout flags do not depend on which goal was requested.
    """
    if future_k < 0:
        raise ValueError("future_k must be nonnegative")
    if not episode:
        return []
    episode_id = episode[0].episode_id
    if any(t.episode_id != episode_id or t.step_index != index for index, t in enumerate(episode)):
        raise ValueError("HER input must be one ordered, complete episode")
    if not (episode[-1].terminated or episode[-1].truncated):
        raise ValueError("HER input episode must end in termination or truncation")
    result = list(episode)
    for index, transition in enumerate(episode):
        current_goal = transition.next_observation["achieved_goal"]
        eligible = []
        for future_index in range(index + 1, len(episode)):
            future_goal = episode[future_index].next_observation["achieved_goal"]
            distance, angle = pose_error(current_goal, future_goal)
            if float(distance) > 0.005 or float(angle) > np.deg2rad(5):
                eligible.append(future_index)
        if not eligible:
            continue
        for _ in range(future_k):
            future_index = eligible[int(rng.integers(len(eligible)))]
            future_goal = episode[future_index].next_observation["achieved_goal"]
            next_obs = _copy_with_goal(transition.next_observation, future_goal)
            reward = float(
                np.asarray(
                    reward_fn(
                        next_obs["achieved_goal"],
                        future_goal,
                        {"collision": transition.collision, "shielded": transition.shielded},
                    )
                )
            )
            result.append(
                replace(
                    transition,
                    observation=_copy_with_goal(transition.observation, future_goal),
                    next_observation=next_obs,
                    reward=reward,
                    future_index=future_index,
                )
            )
    return result
