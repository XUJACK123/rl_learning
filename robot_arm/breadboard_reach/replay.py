"""Bridge completed HER episodes into RLinf's TrajectoryReplayBuffer."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .her import Transition, relabel_episode


def _rlinf_types() -> tuple[type, type]:
    try:
        from rlinf.data.schema.embodied_types import Trajectory
        from rlinf.data.storage.replay import TrajectoryReplayBuffer
    except ImportError as exc:
        raise RuntimeError(
            "RLinf is required for training. Install this project with the 'rlinf' extra."
        ) from exc
    return Trajectory, TrajectoryReplayBuffer


def to_rlinf_trajectory(transitions: list[Transition], max_episode_steps: int) -> Any:
    """Pack relabeled transition samples into one RLinf [T, B=1, ...] trajectory."""
    if not transitions:
        raise ValueError("Cannot pack an empty transition list")
    trajectory_type, _ = _rlinf_types()

    def stack_obs(field: str) -> dict[str, torch.Tensor]:
        keys = transitions[0].observation.keys()
        return {
            key: torch.from_numpy(
                np.stack([getattr(t, field)[key] for t in transitions], axis=0)
            ).unsqueeze(1).contiguous()
            for key in keys
        }

    def column(values: list[float | bool], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype).reshape(-1, 1, 1)

    return trajectory_type(
        max_episode_length=max_episode_steps,
        model_weights_id="breadboard_ddpg",
        actions=torch.from_numpy(np.stack([t.action for t in transitions])).unsqueeze(1).unsqueeze(1),
        rewards=column([t.reward for t in transitions], torch.float32),
        terminations=column([t.terminated for t in transitions], torch.bool),
        truncations=column([t.truncated for t in transitions], torch.bool),
        dones=column([t.terminated or t.truncated for t in transitions], torch.bool),
        curr_obs=stack_obs("observation"),
        next_obs=stack_obs("next_observation"),
    )


class RlinfHERReplay:
    """RLinf-managed persistence and transition sampling after episode HER."""

    def __init__(
        self,
        path: Path,
        *,
        seed: int,
        window_episodes: int,
        cache_episodes: int,
        max_episode_steps: int,
    ) -> None:
        _, buffer_type = _rlinf_types()
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_episode_steps = max_episode_steps
        self.buffer = buffer_type(
            seed=seed,
            enable_cache=True,
            cache_size=cache_episodes,
            sample_window_size=window_episodes,
            auto_save=False,
            auto_save_path=str(self.path),
            trajectory_format="pt",
        )

    @property
    def total_samples(self) -> int:
        return int(self.buffer.total_samples)

    def add_episode(
        self,
        episode: list[Transition],
        *,
        future_k: int,
        rng: np.random.Generator,
        reward_fn: Any,
    ) -> int:
        samples = relabel_episode(episode, future_k=future_k, rng=rng, reward_fn=reward_fn)
        self.buffer.add_trajectories([to_rlinf_trajectory(samples, self.max_episode_steps)])
        return len(samples)

    def sample(self, batch_size: int) -> dict[str, Any]:
        return self.buffer.sample(num_chunks=batch_size)

    def save_checkpoint(self, path: Path) -> None:
        # RLinf persists incoming trajectories asynchronously. Drain its
        # writer before taking a consistent snapshot of the replay window.
        executor = self.buffer._save_executor
        executor.shutdown(wait=True)
        self.buffer._save_executor = ThreadPoolExecutor(max_workers=20)
        if self.total_samples:
            self.buffer.sample(num_chunks=1)
        self.buffer.save_checkpoint(str(path))

    def load_checkpoint(self, path: Path) -> None:
        self.buffer.load_checkpoint(
            load_path=str(path), is_distributed=False, local_rank=0, world_size=1
        )

    def close(self) -> None:
        self.buffer.close()
