# first time:
# 前进奖励（世界系 x 方向速度，乘以站立因子）
        reward_forward = 1.0 * self.data.qvel[0] * standing
        # 存活奖励
        reward_alive = 0.5
        # 控制代价
        reward_ctrl = -0.1 * np.sum(np.square(action))
        # 摔倒惩罚
        if height <= 0.8:
            reward_fall = -5
            terminated = True
        if self.step_count_ >= self.max_steps_:
            truncated = True
        self.step_count_ += 1
        reward = reward_forward + reward_alive + reward_ctrl + reward_fall