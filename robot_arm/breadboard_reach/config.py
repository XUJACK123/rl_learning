"""Validated task and training defaults, kept with saved checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskConfig:
    board_center: tuple[float, float, float] = (0.45, 0.0, 0.255)
    board_half_size: tuple[float, float, float] = (0.08, 0.04, 0.005)
    hole_pitch: float = 0.00254
    hole_stride: int = 8
    target_height: float = 0.03
    position_tolerance: float = 0.005
    angular_tolerance_deg: float = 10.0
    hold_steps: int = 5
    max_episode_steps: int = 100
    physics_dt: float = 0.002
    control_dt: float = 0.05
    max_joint_delta: float = 0.02
    collision_penalty: float = 5.0
    reset_joint_noise: float = 0.025
    split_seed: int = 2026

    def __post_init__(self) -> None:
        if self.hole_pitch <= 0 or self.hole_stride < 1:
            raise ValueError("Hole pitch and stride must be positive")
        if self.position_tolerance <= 0 or self.angular_tolerance_deg <= 0:
            raise ValueError("Goal tolerances must be positive")
        if self.hold_steps < 1 or self.max_episode_steps < self.hold_steps:
            raise ValueError("Invalid episode or hold length")
        substeps = self.control_dt / self.physics_dt
        if abs(substeps - round(substeps)) > 1e-8:
            raise ValueError("control_dt must be a multiple of physics_dt")
        if self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 1_000_000
    warmup_steps: int = 10_000
    batch_size: int = 256
    replay_window_episodes: int = 2_000
    replay_cache_episodes: int = 8
    her_future_k: int = 4
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    exploration_std: float = 0.1
    updates_per_step: int = 1
    eval_interval: int = 10_000
    eval_episodes: int = 20
    checkpoint_interval: int = 50_000

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.warmup_steps < 0 or self.batch_size < 1:
            raise ValueError("Invalid training length or batch size")
        if self.her_future_k < 0 or self.replay_window_episodes < 1:
            raise ValueError("Invalid HER or replay configuration")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")

    def asdict(self) -> dict:
        return asdict(self)
