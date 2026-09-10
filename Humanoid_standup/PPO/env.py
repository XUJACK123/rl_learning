import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

class CustomHumanoidWalkingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path("humanoid.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.viewer = None
        # qpos0为XML中定义的初始姿态
        self.init_qpos_ = self.model.qpos0.copy()
        self.init_qvel_ = self.model.qvel0.copy()
        self.step_count_ = 0
        self.max_steps_ = 1000
        self.repeated_steps = 5

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.model.actuator_forcerange[:,0], high=self.model.actuator_forcerange[:,1], shape=(17,), dtype=np.float32)

    def get_obs_(self):
        quaternion = self.data.qpos[3:7]
        linear_speed = self.data.qvel[0:3]
        angular_speed = self.data.qvel[3:6]
        robot_angle = self.data.qpos[7:]
        robot_angular_speed = self.data.qvel[6:]
        height = self.data.qpos[2]
        obs = np.concatenate([quaternion, linear_speed, angular_speed, robot_angle, robot_angular_speed, [height]]).astype(np.float32)
        return obs

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="track")
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.init_qpos_ + self.np_random.uniform(-0.001, 0.001, size=self.model.nq)
        self.data.qvel[:] = self.init_qvel_ + self.np_random.uniform(-0.001, 0.001, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.step_count_ = 0
        return self.get_obs_(), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward_fall = 0

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for i in range(self.repeated_steps):
            mujoco.mj_step(self.model, self.data)
        # 前进奖励
        reward_forward = 1.0*self.data.qvel[0]
        # 存活奖励
        reward_alive = 0.5
        # 控制代价
        reward_ctrl = -0.1*np.sum(np.square(action))
        # 摔倒惩罚
        if(self.data.qpos[2] <= 0.8):
            reward_fall = -5
            terminated = True
        if self.step_count_ >= self.max_steps_:
            truncated = True
        self.step_count_ += 1
        reward = reward_forward + reward_alive + reward_ctrl + reward_fall
        info = {"reward_forward": reward_forward, "reward_alive": reward_alive, "reward_ctrl": reward_ctrl, "x_velocity": self.data.qvel[0], "height": self.data.qpos[2]}
        return self.get_obs_(), reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None
