"""
小车追踪目标 - ZED 2i 摄像头版 Gymnasium 环境

模拟你的实际系统：
  📷 ZED 2i → 🔍 目标检测模型 → 检测结果 → 🧠 RL策略 → (vx, vy, ω)

ZED 2i 参数：
  分辨率: 1280×720 @60fps
  FOV:    110°(H) × 70°(V)
  深度:   0.2m ~ 20m

观测空间 (3维)：模拟目标检测模型 + 深度估计输出的信息
  [bbox_cx, bbox_cy, z]
    ↑ 目标在图像中的位置   ↑ 目标到摄像头的深度距离
    (来自检测模型)          (ZED 2i 深度图, 归一化到 [0, 1])

动作空间 (3维)：直接速度控制
  [vx, vy, omega]
    ↑ 前进速度  ↑ 横向速度  ↑ 旋转角速度
    (m/s)       (m/s)       (rad/s)

小车动力学：差速驱动 / 全向轮
  x'     = vx*cos(θ) - vy*sin(θ)
  y'     = vx*sin(θ) + vy*cos(θ)
  θ'     = ω
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class CarTrackingEnv(gym.Env):
    """
    摄像头 + 检测模型 + PPO 控制 —— 完整管线仿真

    关键设计：
    - 观测是"检测模型输出"，不是全局坐标
    - 动作是"速度指令"，不是加速度+转向角
    - 加入检测噪声模拟真实视觉系统的不确定性
    """

    # ========== 物理参数 ==========
    VX_MAX = 1.0      # 最大前进速度 (m/s)
    VY_MAX = 1.0      # 最大横向速度 (m/s)，差速车为0
    OMEGA_MAX = 1.5   # 最大旋转角速度 (rad/s)
    DT = 0.1          # 仿真步长 (s)，100ms = 10Hz 控制频率

    # ========== 场景参数 ==========
    ARENA_SIZE = 20.0
    TARGET_SPEED = 1.0
    MAX_STEPS = 300
    CATCH_RADIUS = 1.0

    # ========== 摄像头参数 —— ZED 2i ==========
    # 来源: https://www.stereolabs.com/zed-2i/
    CAMERA_FOV_H = 110.0               # 水平视场角 (度)
    CAMERA_FOV_V = 70.0                # 垂直视场角 (度)
    CAMERA_FOV_H_RAD = np.deg2rad(110.0)
    CAMERA_FOV_V_RAD = np.deg2rad(70.0)
    IMAGE_W = 1280                      # 图像宽度 (像素) — 720p @60fps
    IMAGE_H = 720                       # 图像高度 (像素)
    CAMERA_MAX_RANGE = 20.0             # ZED 2i 深度范围 0.2m~20m
    CAMERA_HEIGHT = 1.0                 # 摄像头离地高度 (米)，取决于你安装位置
    DETECTION_NOISE = 3.0               # 检测框中心噪声标准差 (像素)
    DEPTH_NOISE = 0.02                  # 深度噪声标准差 (归一化)，模拟 ZED 深度误差

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # ======== 观测空间：3维 ========
        # [bbox_cx, bbox_cy, z]
        # bbox_cx/cy: 检测框中心在图像中的归一化坐标 [-1, 1]
        # z: 目标到摄像头的深度距离，归一化 [0, 1]（0.2m≈0，20m=1）
        obs_low = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ======== 动作空间：3维速度指令 ========
        # [vx, vy, omega] 归一化到 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.render_mode = render_mode

        # ---- 状态变量 ----
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_theta = 0.0       # 朝向角 (rad)
        self.car_vx = 0.0
        self.car_vy = 0.0
        self.car_omega = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        self.steps = 0

    # ========== 核心：模拟摄像头 + 目标检测 ==========

    def _world_to_camera(self, tx: float, ty: float):
        """
        将目标的世界坐标转换到小车坐标系，再投影到图像平面

        ZED 2i: 水平 FOV 110°, 垂直 FOV 70°

        返回:
            (img_cx, img_cy, dist, in_view)   # dist 为距离 (米)
        """
        # 1. 转换到小车局部坐标系
        dx = tx - self.car_x
        dy = ty - self.car_y
        dist = np.sqrt(dx**2 + dy**2)

        # 目标相对小车的角度
        angle_to_target = np.arctan2(dy, dx)
        angle_in_car_frame = angle_to_target - self.car_theta
        angle_in_car_frame = (angle_in_car_frame + np.pi) % (2 * np.pi) - np.pi

        # 2. 检查是否在 ZED 2i 视野内 (H:110°, V:70°)
        half_fov_h = self.CAMERA_FOV_H_RAD / 2
        in_view = (
            abs(angle_in_car_frame) < half_fov_h and
            dist < self.CAMERA_MAX_RANGE and
            dist > 0.2  # ZED 2i 最小深度
        )

        if not in_view:
            # 视野外：中心归零，深度取最大值（z=1），表示"看不到/太远"
            return 0.0, 0.0, self.CAMERA_MAX_RANGE, False

        # 3. 投影到图像平面
        img_cx = angle_in_car_frame / half_fov_h  # 水平偏移 [-1,1]
        img_cx = np.clip(img_cx, -1, 1)

        # 垂直: 利用 ZED 2i 垂直 FOV 70°，目标在地面上
        pitch = np.arctan2(-self.CAMERA_HEIGHT, dist)
        half_fov_v = self.CAMERA_FOV_V_RAD / 2
        img_cy = pitch / half_fov_v
        img_cy = np.clip(img_cy, -1, 1)

        return img_cx, img_cy, dist, True

    def _get_detection_obs(self):
        """
        模拟目标检测模型 + ZED 深度估计的输出

        返回 3维观测向量:
          [bbox_cx, bbox_cy, z]
          bbox_cx/cy: 检测框中心归一化坐标 [-1, 1]
          z: 深度距离归一化 [0, 1]（0.2m≈0，20m=1）
        """
        cx, cy, dist, in_view = self._world_to_camera(self.target_x, self.target_y)

        # 加入检测/深度噪声（模拟真实传感器的不确定性）
        rng = np.random.default_rng()
        cx += rng.normal(0, self.DETECTION_NOISE / self.IMAGE_W)
        cy += rng.normal(0, self.DETECTION_NOISE / self.IMAGE_H)
        z = dist / self.CAMERA_MAX_RANGE
        z += rng.normal(0, self.DEPTH_NOISE)
        z = np.clip(z, 0.0, 1.0)

        return np.array([cx, cy, z], dtype=np.float32)

    # ========== 小车动力学：直接速度控制 ==========

    def _car_dynamics(self, vx: float, vy: float, omega: float, dt: float):
        """
        速度控制模式：直接给定 (vx, vy, ω)

        世界坐标更新：
          x'     = vx*cos(θ) - vy*sin(θ)
          y'     = vx*sin(θ) + vy*cos(θ)
          θ'     = ω
        """
        # 更新朝向
        self.car_theta += omega * dt
        self.car_theta = (self.car_theta + np.pi) % (2 * np.pi) - np.pi

        # 更新全局位置
        self.car_x += (vx * np.cos(self.car_theta) - vy * np.sin(self.car_theta)) * dt
        self.car_y += (vx * np.sin(self.car_theta) + vy * np.cos(self.car_theta)) * dt

        # 边界
        self.car_x = np.clip(self.car_x, 0, self.ARENA_SIZE)
        self.car_y = np.clip(self.car_y, 0, self.ARENA_SIZE)

        # 记录速度（用于 reward）
        self.car_vx = vx
        self.car_vy = vy
        self.car_omega = omega

    # ========== 目标移动 ==========

    def _target_dynamics(self, dt: float):
        """目标随机游走（可替换为任何轨迹）"""
        if self.steps % 30 == 0:
            self.target_theta += np.random.uniform(-np.pi / 3, np.pi / 3)

        self.target_x += self.TARGET_SPEED * np.cos(self.target_theta) * dt
        self.target_y += self.TARGET_SPEED * np.sin(self.target_theta) * dt

        # 边界反弹
        if self.target_x <= 0 or self.target_x >= self.ARENA_SIZE:
            self.target_theta = np.pi - self.target_theta
        if self.target_y <= 0 or self.target_y >= self.ARENA_SIZE:
            self.target_theta = -self.target_theta
        self.target_x = np.clip(self.target_x, 0, self.ARENA_SIZE)
        self.target_y = np.clip(self.target_y, 0, self.ARENA_SIZE)

    # ========== 奖励函数 ==========

    def _compute_reward(self) -> float:
        """奖励：鼓励保持在目标附近，面朝目标"""
        dx = self.car_x - self.target_x
        dy = self.car_y - self.target_y
        distance = np.sqrt(dx**2 + dy**2)

        reward = 0.0

        # (a) 距离惩罚
        reward -= distance * 0.2

        # (b) 追上了
        if distance < self.CATCH_RADIUS:
            reward += 15.0

        # (c) 鼓励面朝目标（align heading to target）
        target_angle = np.arctan2(dy, dx)
        heading_error = abs((target_angle - self.car_theta + np.pi) % (2 * np.pi) - np.pi)
        reward += (np.pi - heading_error) * 0.5 / np.pi  # 0~0.5，正对时最高

        # (d) 存活
        reward += 0.05

        return reward

    # ========== Gymnasium 接口 ==========

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car_x = np.random.uniform(1, 4)
        self.car_y = np.random.uniform(1, 4)
        self.car_theta = np.random.uniform(-np.pi, np.pi)
        self.car_vx = 0.0
        self.car_vy = 0.0
        self.car_omega = 0.0

        self.target_x = np.random.uniform(12, 19)
        self.target_y = np.random.uniform(12, 19)
        self.target_theta = np.random.uniform(-np.pi, np.pi)
        self.steps = 0
        return self._get_detection_obs(), {}

    def step(self, action: np.ndarray):
        """
        action: [vx_norm, vy_norm, omega_norm] 归一化到 [-1, 1]
        """
        # 映射到物理量
        vx = float(action[0]) * self.VX_MAX
        vy = float(action[1]) * self.VY_MAX
        omega = float(action[2]) * self.OMEGA_MAX

        # 更新
        self._car_dynamics(vx, vy, omega, self.DT)
        self._target_dynamics(self.DT)
        reward = self._compute_reward()
        self.steps += 1

        # 终止
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        terminated = dist < self.CATCH_RADIUS
        truncated = self.steps >= self.MAX_STEPS

        obs = self._get_detection_obs()
        info = {"distance": dist, "car_speed": np.sqrt(vx**2 + vy**2)}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        in_view = abs(
            (np.arctan2(self.target_y - self.car_y, self.target_x - self.car_x)
             - self.car_theta + np.pi) % (2 * np.pi) - np.pi
        ) < self.CAMERA_FOV_H_RAD / 2 and dist < self.CAMERA_MAX_RANGE
        view_str = "👁 ZED2i" if in_view else "🙈 视野外"
        print(
            f"Step {self.steps:3d} | "
            f"Car: ({self.car_x:5.1f}, {self.car_y:5.1f}) θ={np.degrees(self.car_theta):6.1f}° | "
            f"Target: ({self.target_x:5.1f}, {self.target_y:5.1f}) | "
            f"Dist: {dist:.2f}m | {view_str} | "
            f"Act: (vx={self.car_vx:.2f}, vy={self.car_vy:.2f}, ω={self.car_omega:.2f})"
        )

    def close(self):
        pass
