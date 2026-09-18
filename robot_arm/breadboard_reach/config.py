"""Validated task and training defaults, kept with saved checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


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
    collision_penalty: float = 10.0
    shield_penalty: float = 1.0
    success_bonus: float = 5.0
    position_reward_scale: float = 0.1
    angular_reward_scale_deg: float = 45.0
    action_shield: bool = True
    max_action_halvings: int = 4
    reset_joint_noise: float = 0.025
    home_joint_positions: tuple[float, ...] | None = None
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
        if self.position_reward_scale <= 0 or self.angular_reward_scale_deg <= 0:
            raise ValueError("Reward scales must be positive")
        if self.max_action_halvings < 0:
            raise ValueError("max_action_halvings must be nonnegative")
        if self.home_joint_positions is not None and len(self.home_joint_positions) != 7:
            raise ValueError("home_joint_positions must contain seven joints")

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 300_000
    warmup_steps: int = 0
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
    checkpoint_interval: int = 10_000
    demo_episodes_per_hole: int = 5
    bc_updates: int = 5_000
    dagger_rounds: int = 3
    dagger_bc_updates: int = 1_000
    critic_pretrain_updates: int = 5_000
    bc_weight: float = 5.0
    profile: str = "full"

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.warmup_steps < 0 or self.batch_size < 1:
            raise ValueError("Invalid training length or batch size")
        if self.her_future_k < 0 or self.replay_window_episodes < 1:
            raise ValueError("Invalid HER or replay configuration")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if min(self.demo_episodes_per_hole, self.bc_updates, self.dagger_rounds, self.dagger_bc_updates, self.critic_pretrain_updates) < 0:
            raise ValueError("Demonstration and pretraining counts must be nonnegative")
        if self.bc_weight < 0:
            raise ValueError("bc_weight must be nonnegative")
        if self.profile not in ("full", "smoke"):
            raise ValueError("Unknown training profile")

    @classmethod
    def for_profile(cls, profile: str) -> "TrainConfig":
        if profile == "full":
            return cls()
        if profile == "smoke":
            return replace(
                cls(),
                total_steps=2_000,
                batch_size=64,
                replay_window_episodes=200,
                demo_episodes_per_hole=1,
                bc_updates=200,
                dagger_rounds=1,
                dagger_bc_updates=100,
                critic_pretrain_updates=200,
                eval_interval=1_000,
                eval_episodes=5,
                checkpoint_interval=1_000,
                profile="smoke",
            )
        raise ValueError(f"Unknown training profile: {profile}")

    def asdict(self) -> dict:
        return asdict(self)
