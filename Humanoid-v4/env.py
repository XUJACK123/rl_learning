"""Humanoid stand environment for the Language-to-Rewards mini reproduction.

Loads the MJPC humanoid model and implements a reward of the form
    R(x, u) = -sum_i w_i * ||r_i(x, u)||^2
that mirrors the residual set of the MJPC ``humanoid/stand`` task
(see mjpc/tasks/humanoid/stand/stand.cc, Apache-2.0), plus three terms this
project adds for language steering: torso pitch, torso roll and heading.

Model credits: ``assets/humanoid_modified.xml`` is dm_control's humanoid
(Apache-2.0) modified by the MJPC humanoid task patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import mujoco
import numpy as np


ASSETS = Path(__file__).resolve().parent / "assets"


@dataclass
class SteppingParams:
    """单个脚的动态步态参数（论文 set_feet_stepping_parameters）。"""
    frequency: float = 0.0        # 步态频率 Hz
    air_ratio: float = 0.0        # 脚在空中时间占周期比例 [0,1]
    phase_offset: float = 0.0     # 相位偏移 [0,1]，两脚差 0.5 = 交替
    swing_up_down: float = 0.0    # 垂直摆动幅度 (m)
    swing_forward_back: float = 0.0  # 前后摆动幅度 (m)，正=向前
    active: bool = False          # 是否启用步态


@dataclass
class StandParams:
    """Reward parameters set by the LLM-generated reward script."""

    target_height: float = 1.28
    target_pitch: float = 0.0
    target_roll: float = 0.0
    target_heading: float = 0.0
    # 目标速度（局部坐标系）：x 前进，y 侧向（正=左），单位 m/s
    target_velocity_xy: tuple[float, float] = (0.0, 0.0)
    # 目标转向速度，rad/s
    target_turning_speed: float = 0.0
    feet_lift: dict[str, float] = field(default_factory=dict)  # name -> lift height
    # 动态步态：name -> SteppingParams
    feet_stepping: dict[str, SteppingParams] = field(default_factory=dict)

    # residual weights (diagonal of W in r' W r)
    w_height: float = 100.0
    w_balance: float = 0.0
    w_com_vel: float = 10.0
    w_joint_vel: float = 0.01
    w_ctrl: float = 0.025
    w_pitch: float = 10.0
    w_roll: float = 10.0
    w_heading: float = 20.0
    w_foot: float = 20.0
    # 速度追踪权重（走路时要把这个调大）
    w_velocity: float = 100.0
    w_turning: float = 20.0
    # 步态追踪权重（要比速度权重大，才能逼脚真的抬起来）
    w_stepping: float = 500.0


FEET = {"left_foot": ("foot_left", ("sp0", "sp1")),
        "right_foot": ("foot_right", ("sp2", "sp3"))}


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """Body ZYX Euler angles (roll, pitch, yaw) from quaternion (w, x, y, z)."""
    w, x, y, z = q
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(r21, r22)
    pitch = math.atan2(-r20, math.sqrt(r00 * r00 + r10 * r10))
    yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


class HumanoidEnv:
    """MuJoCo humanoid with a stand residual reward and iLQG-friendly helpers."""

    def __init__(self, model_path: str | Path | None = None):
        model_path = Path(model_path) if model_path else ASSETS / "task.xml"
        self.m = mujoco.MjModel.from_xml_path(str(model_path))
        self.d = mujoco.MjData(self.m)

        def body(name: str) -> int:
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

        def site(name: str) -> int:
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)

        self.body_torso = body("torso")
        self.site_feet = {
            "left_foot": (site("sp0"), site("sp1")),
            "right_foot": (site("sp2"), site("sp3")),
        }
        self.nq, self.nv, self.nu = self.m.nq, self.m.nv, self.m.nu
        self.dt = self.m.opt.timestep
        self.total_mass = float(self.m.body_mass.sum())
        self.rest_foot_z: dict[str, float] = {}
        self.time = 0.0   # 仿真时间（秒），reset 归零，step 累加

    # ------------------------------------------------------------------ state
    def get_state(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = d or self.d
        return np.concatenate([d.qpos, d.qvel]).copy()

    def set_state(self, x: np.ndarray, d: mujoco.MjData | None = None) -> None:
        d = d or self.d
        d.qpos[:] = x[: self.nq]
        # keep the free-joint quaternion unit length (finite differences perturb
        # individual components, so renormalise before any forward call)
        q = d.qpos[3:7]
        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            d.qpos[3:7] = q / norm
        d.qvel[:] = x[self.nq : self.nq + self.nv]
        mujoco.mj_forward(self.m, d)

    def reset(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = d or self.d
        mujoco.mj_resetData(self.m, d)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, d)
        # 给自由关节一个向前的初始速度微扰，打破“完美静止”冷启动：
        # iLQG 从零速度出发会陷入零动作局部最优（机器人不动），
        # 一个小的初速度能给它一个“顺势迈步”的初始梯度。
        d.qvel[0] = 0.3   # 向前线速度 0.3 m/s
        mujoco.mj_forward(self.m, d)
        self.time = 0.0
        self.rest_foot_z = {
            name: float(
                np.mean([d.site_xpos[i][2] for i in self.site_feet[name]])
            )
            for name in FEET
        }
        return self.get_state(d)

    def step(self, u: np.ndarray, d: mujoco.MjData | None = None) -> None:
        d = d or self.d
        d.ctrl[:] = np.clip(u, self.m.actuator_ctrlrange[:, 0],
                            self.m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.m, d)
        if d is self.d:
            self.time += self.dt

    # -------------------------------------------------------------- kinematics
    def com(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = d or self.d
        return d.subtree_com[self.body_torso].copy()

    def com_vel(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = d or self.d
        return d.subtree_linvel[self.body_torso].copy()

    def torso_orientation(self, d: mujoco.MjData | None = None) -> tuple:
        d = d or self.d
        return quat_to_euler_zyx(d.xquat[self.body_torso])

    def feet_center(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = d or self.d
        pts = [d.site_xpos[i] for name in FEET for i in self.site_feet[name]]
        return np.mean(pts, axis=0)

    # ------------------------------------------------------------------ reward
    def residual(self, p: StandParams, d: mujoco.MjData | None = None,
                 time: float | None = None) -> np.ndarray:
        d = d or self.d
        if time is None:
            time = self.time
        r: list[float] = []

        # torso 朝向（速度需要转到局部系，heading 残差也要用）
        roll, pitch, yaw = self.torso_orientation(d)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # (0) height: torso z vs target
        r.append(d.xpos[self.body_torso][2] - p.target_height)

        # (1) balance: capture point xy vs feet centre xy (MJPC stand residual)
        com, com_vel = self.com(d), self.com_vel(d)
        capture = com[:2] + 0.2 * com_vel[:2]
        r.append(float(np.linalg.norm(capture - self.feet_center(d)[:2])))

        # (2-3) com 速度（局部坐标系）-> 目标速度
        # 世界系速度转到 body 局部系：R(-yaw) = R(yaw)^T
        local_vx = cy * com_vel[0] + sy * com_vel[1]
        local_vy = -sy * com_vel[0] + cy * com_vel[1]
        r.append(local_vx - p.target_velocity_xy[0])
        r.append(local_vy - p.target_velocity_xy[1])

        # (4-24) joint velocity -> 0
        r.extend(d.qvel[6:].tolist())

        # (25-45) control -> 0
        r.extend(d.ctrl.tolist())

        # torso orientation vs targets
        r.append(wrap_angle(roll - p.target_roll))
        r.append(wrap_angle(pitch - p.target_pitch))
        r.append(wrap_angle(yaw - p.target_heading))

        # 转向速度残差：自由关节绕 z 轴角速度 (qvel[5]) -> 目标转向速度
        r.append(d.qvel[5] - p.target_turning_speed)

        # feet: target height above the rest height recorded at reset
        for name in FEET:
            if name in p.feet_lift:
                z = float(np.mean([d.site_xpos[i][2] for i in self.site_feet[name]]))
                rest = self.rest_foot_z.get(name, 0.0)
                r.append(z - (rest + p.feet_lift[name]))

        # 动态步态残差：脚跟随一个周期性的抬脚/前伸目标
        # 相位 phase = (t*freq + phase_offset) mod 1，摆动相在 [0, air_ratio)
        com_xy = self.com(d)[:2]
        for name in FEET:
            sp = p.feet_stepping.get(name)
            if sp is None or not sp.active:
                continue
            foot_pos = np.mean([d.site_xpos[i] for i in self.site_feet[name]], axis=0)
            z = float(foot_pos[2])
            rest = self.rest_foot_z.get(name, 0.0)
            phase = (time * sp.frequency + sp.phase_offset) % 1.0
            if phase < sp.air_ratio and sp.air_ratio > 0.0:
                # 摆动相：用半正弦波形（0 -> 峰值 -> 0）
                s = math.sin(math.pi * phase / sp.air_ratio)
                lift = sp.swing_up_down * s
                fwd = sp.swing_forward_back * s
            else:
                # 支撑相：脚贴地、不前伸
                lift = 0.0
                fwd = 0.0
            # 抬脚高度残差
            r.append(z - (rest + lift))
            # 前后摆动残差：脚相对质心的前后距离（沿 body 朝向 x 轴）
            dx = foot_pos[0] - com_xy[0]
            dy = foot_pos[1] - com_xy[1]
            foot_rel = cy * dx + sy * dy
            r.append(foot_rel - fwd)

        return np.asarray(r)

    def weights(self, p: StandParams) -> np.ndarray:
        w = [
            p.w_height,
            p.w_balance,
            p.w_velocity,
            p.w_velocity,
        ]
        w += [p.w_joint_vel] * (self.nv - 6)
        w += [p.w_ctrl] * self.nu
        w += [p.w_roll, p.w_pitch, p.w_heading]
        w += [p.w_turning]
        w += [p.w_foot for _ in p.feet_lift]
        # 每个激活步态的脚贡献 2 条残差（抬脚 + 前后摆动）
        n_step = sum(1 for sp in p.feet_stepping.values() if sp.active)
        w += [p.w_stepping] * (2 * n_step)
        return np.asarray(w)

    def cost(self, p: StandParams, d: mujoco.MjData | None = None,
             time: float | None = None) -> float:
        r = self.residual(p, d, time)
        w = self.weights(p)
        assert r.shape == w.shape, (r.shape, w.shape)
        return float(w @ (r * r))

    def residual_jacobian(
        self,
        p: StandParams,
        x: np.ndarray,
        u: np.ndarray,
        d: mujoco.MjData | None = None,
        eps_x: float = 1e-6,
        eps_u: float = 1e-5,
        time: float | None = None,
    ) -> np.ndarray:
        """Jacobian dr/d[x; u] by finite differences around (x, u)."""
        d = d or self.d
        self.set_state(x, d)
        d.ctrl[:] = u
        mujoco.mj_forward(self.m, d)
        base = self.residual(p, d, time)
        nr, nx, nu = base.size, x.size, u.size
        J = np.zeros((nr, nx + nu))

        for j in range(nx):
            xp = x.copy()
            xp[j] += eps_x
            self.set_state(xp, d)
            J[:, j] = (self.residual(p, d, time) - base) / eps_x

        for j in range(nu):
            up = u.copy()
            up[j] += eps_u
            d.ctrl[:] = up
            mujoco.mj_forward(self.m, d)
            J[:, nx + j] = (self.residual(p, d, time) - base) / eps_u

        self.set_state(x, d)
        d.ctrl[:] = u
        return J

    def cost_quadratics(
        self, p: StandParams, x: np.ndarray, u: np.ndarray, d: mujoco.MjData,
        time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gauss-Newton cost derivatives: l_x, l_u, l_xx, l_uu, l_ux."""
        J = self.residual_jacobian(p, x, u, d, time=time)
        r = self.residual(p, d, time)
        w = self.weights(p)
        Wr = w * r
        nx, nu = x.size, u.size
        Jx, Ju = J[:, :nx], J[:, nx:]
        l_x = Jx.T @ Wr
        l_u = Ju.T @ Wr
        l_xx = (Jx * w[:, None]).T @ Jx
        l_uu = (Ju * w[:, None]).T @ Ju
        l_ux = (Ju * w[:, None]).T @ Jx
        return l_x, l_u, l_xx, l_uu, l_ux
