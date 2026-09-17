"""Reachability checks using the exact Panda MJCF Jacobian and joint limits."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from .env import BreadboardReachEnv
from .geometry import goal_reached, pose_error, target_pose
from .her import Transition


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


def diagnose_control(env: BreadboardReachEnv, *, seeds: int = 20) -> dict:
    """Measure the actual neutral servo response, including first contact."""
    records = []
    hole_id = env.available_holes[0].hole_id
    for seed in range(seeds):
        initial, _ = env.reset(seed=seed, options={"hole_id": hole_id})
        start_q = env.data.qpos[env._qpos_addr].copy()
        start_tcp = initial["achieved_goal"][:3].copy()
        first_collision = None
        max_drift = 0.0
        max_joint_drift = 0.0
        max_control_error = 0.0
        for step in range(env.cfg.max_episode_steps):
            observation, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            max_drift = max(max_drift, float(np.linalg.norm(observation["achieved_goal"][:3] - start_tcp)))
            max_joint_drift = max(max_joint_drift, float(np.max(np.abs(env.data.qpos[env._qpos_addr] - start_q))))
            max_control_error = max(max_control_error, float(np.max(np.abs(env.data.ctrl[env._actuator_ids] - env.data.qpos[env._qpos_addr]))))
            if info["collision"]:
                first_collision = {"step": step + 1, "geoms": info["collision_pair"]}
            if terminated or truncated:
                break
        records.append({
            "seed": seed,
            "start_joint_positions": start_q.tolist(),
            "max_tcp_drift_m": max_drift,
            "max_joint_drift_rad": max_joint_drift,
            "max_control_error_rad": max_control_error,
            "first_collision": first_collision,
        })
    return {
        "stable": all(r["first_collision"] is None and r["max_tcp_drift_m"] <= 0.01 for r in records),
        "records": records,
    }


def find_safe_home(env: BreadboardReachEnv) -> tuple[float, ...]:
    """Search near the Menagerie keyframe for a stable shared reset pose."""
    original_cfg = env.cfg
    env.reset(seed=0, options={"joint_noise": 0})
    starting = env.data.qpos[env._qpos_addr].copy()
    rng = np.random.default_rng(2026)
    candidates = [starting]
    for scale in (0.08, 0.16, 0.3, 0.45):
        candidates.extend(starting + rng.normal(0, scale, 7) for _ in range(30))
    try:
        for candidate in candidates:
            candidate = np.clip(candidate, env._joint_low + 0.05, env._joint_high - 0.05)
            env.cfg = replace(original_cfg, home_joint_positions=tuple(float(x) for x in candidate))
            if diagnose_control(env, seeds=3)["stable"]:
                return env.cfg.home_joint_positions
    finally:
        env.cfg = original_cfg
    raise RuntimeError("No collision-free stable home found near the Panda keyframe")


def _jacobian_action(env: BreadboardReachEnv, desired: np.ndarray) -> np.ndarray:
    achieved = env._achieved_goal()
    position_error = desired[:3] - achieved[:3]
    rotation_error = _rotation_vector(desired[3:], achieved[3:])
    jacobian_position = np.zeros((3, env.model.nv))
    jacobian_rotation = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacobian_position, jacobian_rotation, env._tcp_id)
    jacobian = np.vstack((jacobian_position[:, env._qvel_addr], 0.12 * jacobian_rotation[:, env._qvel_addr]))
    residual = np.concatenate((position_error, 0.12 * rotation_error))
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03**2 * np.eye(6), residual)
    return np.clip(0.6 * delta / env.cfg.max_joint_delta, -1.0, 1.0).astype(np.float32)


def generate_demonstration(
    env: BreadboardReachEnv, hole_id: str, *, seed: int, waypoint: bool = False
) -> tuple[list[Transition], dict]:
    """Roll out a collision-aware Jacobian controller in the real environment."""
    observation, _ = env.reset(seed=seed, options={"hole_id": hole_id})
    desired = observation["desired_goal"].copy()
    transit = desired.copy()
    transit[2] = max(float(desired[2]) + 0.12, 0.42)
    moving_to_goal = not waypoint
    episode: list[Transition] = []
    shield_count = 0
    last_info = None
    for step in range(env.cfg.max_episode_steps):
        if not moving_to_goal:
            distance, angle = pose_error(observation["achieved_goal"], transit)
            if float(distance) < 0.025 and float(angle) < np.deg2rad(20):
                moving_to_goal = True
        controller_goal = desired if moving_to_goal else transit
        if bool(goal_reached(observation["achieved_goal"], desired, env.cfg)):
            action = np.zeros(7, dtype=np.float32)
        else:
            action = _jacobian_action(env, controller_goal)
        next_observation, reward, terminated, truncated, info = env.step(action)
        shield_count += int(info["shielded"])
        episode.append(Transition(
            observation=observation,
            action=info["executed_action"],
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            collision=info["collision"],
            episode_id=0,
            step_index=step,
            shielded=info["shielded"],
        ))
        observation = next_observation
        last_info = info
        if info["success"] or terminated or truncated:
            break
    assert last_info is not None
    if last_info["success"] and not episode[-1].terminated and not episode[-1].truncated:
        from dataclasses import replace

        episode[-1] = replace(episode[-1], truncated=True)
    return episode, {
        "hole_id": hole_id,
        "success": bool(last_info["success"]),
        "collision": bool(last_info["collision"]),
        "steps": len(episode),
        "shielded_steps": shield_count,
        "position_error_m": last_info["position_error_m"],
        "angular_error_deg": last_info["angular_error_deg"],
    }
