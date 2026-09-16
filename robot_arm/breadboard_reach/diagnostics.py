"""Reachability checks using the exact Panda MJCF Jacobian and joint limits."""

from __future__ import annotations

import mujoco
import numpy as np

from .env import BreadboardReachEnv
from .geometry import goal_reached, pose_error, target_pose


def _rotation_vector(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Shortest rotation taking the current wxyz quaternion to goal."""
    gw, gx, gy, gz = goal
    cw, cx, cy, cz = current
    qw = gw * cw + gx * cx + gy * cy + gz * cz
    qx = -gw * cx + gx * cw - gy * cz + gz * cy
    qy = -gw * cy + gx * cz + gy * cw - gz * cx
    qz = -gw * cz - gx * cy + gy * cx + gz * cw
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0:
        q = -q
    norm = np.linalg.norm(q[1:])
    if norm < 1e-10:
        return np.zeros(3)
    return q[1:] * (2 * np.arctan2(norm, q[0]) / norm)


def solve_hole_ik(
    env: BreadboardReachEnv,
    hole_id: str,
    *,
    starts: int = 5,
    iterations: int = 300,
    seed: int = 123,
) -> dict:
    """Try damped least-squares IK without changing the learned controller."""
    hole = env.holes_by_id[hole_id]
    goal = target_pose(env.cfg, hole)
    home_key = env._id(mujoco.mjtObj.mjOBJ_KEY, "home")
    rng = np.random.default_rng(seed)
    jacobian_position = np.zeros((3, env.model.nv))
    jacobian_rotation = np.zeros((3, env.model.nv))
    best = (float("inf"), float("inf"))
    for attempt in range(starts):
        mujoco.mj_resetDataKeyframe(env.model, env.data, home_key)
        if attempt:
            env.data.qpos[env._qpos_addr] = np.clip(
                env.data.qpos[env._qpos_addr] + rng.normal(0, 0.18, size=7),
                env._joint_low + 0.02,
                env._joint_high - 0.02,
            )
        env.data.qvel[:] = 0
        for _ in range(iterations):
            mujoco.mj_forward(env.model, env.data)
            achieved = env._achieved_goal()
            distance, angle = pose_error(achieved, goal)
            if (float(distance), float(angle)) < best:
                best = (float(distance), float(angle))
            if bool(goal_reached(achieved, goal, env.cfg)) and not env._has_collision():
                return {
                    "hole_id": hole_id,
                    "reachable": True,
                    "position_error_m": float(distance),
                    "angular_error_deg": float(np.rad2deg(angle)),
                    "joint_positions": env.data.qpos[env._qpos_addr].tolist(),
                }
            mujoco.mj_jacSite(
                env.model, env.data, jacobian_position, jacobian_rotation, env._tcp_id
            )
            position_error = goal[:3] - achieved[:3]
            rotation_error = _rotation_vector(goal[3:], achieved[3:])
            jacobian = np.vstack(
                (jacobian_position[:, env._qvel_addr], 0.12 * jacobian_rotation[:, env._qvel_addr])
            )
            residual = np.concatenate((position_error, 0.12 * rotation_error))
            damping = 0.03
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), residual
            )
            env.data.qpos[env._qpos_addr] = np.clip(
                env.data.qpos[env._qpos_addr] + np.clip(delta, -0.08, 0.08),
                env._joint_low + 0.01,
                env._joint_high - 0.01,
            )
    return {
        "hole_id": hole_id,
        "reachable": False,
        "position_error_m": best[0],
        "angular_error_deg": float(np.rad2deg(best[1])),
    }
