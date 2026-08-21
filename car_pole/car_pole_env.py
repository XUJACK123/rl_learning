import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomCartPoleEnv(gym.Env):
    # 标准元数据：声明支持的渲染模式和渲染帧率
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        # ---------- 1. 物理参数（与 CartPole-v1 相同） ----------
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5            # 摆杆质心到转轴距离 (m)，摆杆全长 1.0 m
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0        # 作用在小车上的力 (N)
        self.tau = 0.02              # 仿真时间步长 (s)

        # 终止阈值
        self.x_threshold = 2.4
        self.theta_threshold_radians = 30 * 2 * math.pi / 360

        # ---------- 2. 动作空间与观察空间 ----------
        # 动作：0 = 向左推，1 = 向右推
        self.action_space = spaces.Discrete(2)

        # 观察：[小车位置, 小车速度, 摆杆角度, 摆杆角速度]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ---------- 3. 渲染参数 ----------
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.screen_width = 600
        self.screen_height = 400
        world_width = self.x_threshold * 2
        self.scale = self.screen_width / world_width  # 物理米 -> 像素

        self.cartwidth = 50.0
        self.cartheight = 30.0
        self.polewidth = 10.0
        self.polelen = self.scale * (2 * self.length)

        # 内部状态
        self.state = None
        self.steps_beyond_terminated = None

    def reset(self, *, seed=None, options=None):
        """重置环境。标准返回 (observation, info)。"""
        # 必须调用父类 reset，它会根据 seed 初始化 self.np_random，保证可复现
        super().reset(seed=seed)
        # 初始状态在 0 附近小幅随机扰动
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        """执行一步仿真。标准返回 (obs, reward, terminated, truncated, info) 五元组。"""
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) 不是合法动作"

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # 牛顿-欧拉动力学（与 CartPole-v1 完全一致）
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 欧拉积分更新状态
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # 500 步上限由 gym.make 自动包上的 TimeLimit 包装器负责，这里置 False
        truncated = False

        # CartPole 的奖励约定：终止那一步也算 +1
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        """按 render_mode 渲染；rgb_array 模式返回 HWC 的 uint8 数组。"""
        if self.render_mode is None:
            raise RuntimeError("创建环境时必须指定 render_mode（如 render_mode='human'）")
        if self.state is None:
            return None

        import pygame

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        x, _, theta, _ = self.state
        self.screen.fill((255, 255, 255))

        # 轨道基准线
        track_y = self.screen_height * 0.7
        pygame.draw.line(self.screen, (0, 0, 0), (0, track_y), (self.screen_width, track_y), 2)

        # 小车
        cart_x = x * self.scale + self.screen_width / 2.0
        cart_rect = pygame.Rect(
            cart_x - self.cartwidth / 2.0,
            track_y - self.cartheight / 2.0,
            self.cartwidth,
            self.cartheight,
        )
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)

        # 摆杆与转轴
        pole_bottom = (cart_x, track_y)
        pole_top = (
            cart_x + self.polelen * math.sin(theta),
            track_y - self.polelen * math.cos(theta),
        )
        pygame.draw.line(self.screen, (204, 153, 102), pole_bottom, pole_top, int(self.polewidth))
        pygame.draw.circle(self.screen, (127, 127, 204), (int(cart_x), int(track_y)), 6)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """释放 pygame 资源。"""
        if self.screen is not None:
            import pygame

            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self.screen = None

gym.register(
    id="CustomCartPole-v0",
    entry_point="car_pole.car_pole_env:CustomCartPoleEnv",
    max_episode_steps=700,
    reward_threshold=475.0,
)
