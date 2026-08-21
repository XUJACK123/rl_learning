import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(done).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        target_update_freq=10,
        buffer_capacity=50000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        # 策略网络（持续更新）与目标网络（周期性同步）
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity)

    @torch.no_grad()
    def select_action(self, state, action_space=None, epsilon=None):
        """
        epsilon-greedy 选动作：
        以 epsilon 概率随机探索，否则取 Q 值最大的动作。
        评估时传 epsilon=0.0 即为纯贪心（利用）。
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        if random.random() < epsilon:
            if action_space is not None:
                return action_space.sample()
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(device)
        return self.policy_net(state_tensor).argmax().item()

    def update(self, batch_size):
        """从回放缓冲采样一批样本，做一次 Double DQN 梯度更新，返回 loss。"""
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # 当前状态下所选动作的 Q 值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN：用 policy 网络选动作，用 target 网络评价值
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q_values)  # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        """每回合结束后衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """把 policy 网络参数同步到 target 网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
