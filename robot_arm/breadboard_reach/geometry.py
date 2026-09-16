"""Board coordinates and pose error shared by the environment and HER."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TaskConfig


DOWN_QUATERNION = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class Hole:
    hole_id: str
    row: int
    column: int
    local_xy: tuple[float, float]
    split: str


def board_to_world(local_xyz: np.ndarray, center: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Transform board-relative points; points may have leading batch dimensions."""
    xyz = np.asarray(local_xyz, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return xyz @ rotation.T + center


def make_holes(cfg: TaskConfig) -> list[Hole]:
    """Use an eight-pitch grid with a physical margin from the board edge."""
    pitch = cfg.hole_pitch * cfg.hole_stride
    x_extent = cfg.board_half_size[0] - 0.012
    y_extent = cfg.board_half_size[1] - 0.012
    max_col = int(np.floor(x_extent / pitch))
    max_row = int(np.floor(y_extent / pitch))
    candidates = [
        (row, col, (col * pitch, row * pitch))
        for row in range(-max_row, max_row + 1)
        for col in range(-max_col, max_col + 1)
    ]
    if len(candidates) < 5:
        raise ValueError("Board is too small for a train/eval target split")
    rng = np.random.default_rng(cfg.split_seed)
    order = rng.permutation(len(candidates))
    n_eval = max(1, round(len(candidates) * 0.2))
    eval_ids = set(order[:n_eval].tolist())
    return [
        Hole(f"R{row:+d}C{col:+d}", row, col, xy, "eval" if index in eval_ids else "train")
        for index, (row, col, xy) in enumerate(candidates)
    ]


def target_pose(cfg: TaskConfig, hole: Hole) -> np.ndarray:
    local = np.asarray(
        [hole.local_xy[0], hole.local_xy[1], cfg.board_half_size[2] + cfg.target_height],
        dtype=np.float64,
    )
    position = board_to_world(local, np.asarray(cfg.board_center))
    return np.concatenate((position, DOWN_QUATERNION)).astype(np.float32)


def canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return np.where(q[..., :1] < 0, -q, q)


def pose_error(achieved: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return metric distance and shortest quaternion angle in radians."""
    achieved = np.asarray(achieved, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    distance = np.linalg.norm(achieved[..., :3] - desired[..., :3], axis=-1)
    a = canonical_quaternion(achieved[..., 3:])
    d = canonical_quaternion(desired[..., 3:])
    cosine = np.abs(np.sum(a * d, axis=-1))
    angle = 2 * np.arccos(np.clip(cosine, 0.0, 1.0))
    return distance, angle


def goal_reached(achieved: np.ndarray, desired: np.ndarray, cfg: TaskConfig) -> np.ndarray:
    position_error, angular_error = pose_error(achieved, desired)
    return (position_error <= cfg.position_tolerance) & (
        angular_error <= np.deg2rad(cfg.angular_tolerance_deg)
    )


def compute_reward(
    achieved: np.ndarray, desired: np.ndarray, info: dict | None, cfg: TaskConfig
) -> np.ndarray:
    """Sparse goal reward plus a goal-independent contact penalty, vectorized."""
    reached = goal_reached(achieved, desired, cfg)
    collision = 0 if info is None else info.get("collision", 0)
    return np.asarray(reached, dtype=np.float32) - 1.0 - cfg.collision_penalty * np.asarray(
        collision, dtype=np.float32
    )
