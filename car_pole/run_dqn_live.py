"""
运行训练好的 DQN 模型 —— 可视化演示

特性:
  - 持续运行，pole 倒了自动重开新一局
  - 按 'q' 或 'Q' 退出
  - 按 'r' 手动重开一局

用法:
    python3 run_dqn_live.py
"""
import os
import gymnasium as gym
import torch
import torch.nn as nn
import pygame
from car_pole_env import CustomCartPoleEnv  # 触发注册 CustomCartPole-v0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN_training(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


def check_keys():
    """返回要执行的动作: 'quit' / 'reset' / None"""
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q,):
                return "quit"
            if event.key == pygame.K_r:
                return "reset"
    return None


def main():
    model_path = "dqn_cartpole.pth"
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "dqn_cartpole.pth")
    env = gym.make("CustomCartPole-v0", render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN_training(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path} on {device}")
    print("按键: [q] 退出  [r] 重开一局")
    print("=" * 50)

    episode = 1
    state, _ = env.reset()
    steps = 0
    total_reward = 0

    while True:
        env.render()

        # 处理按键
        key = check_keys()
        if key == "quit":
            print("\n退出")
            break
        elif key == "reset":
            print(f"↻ 手动重开 Episode {episode}")
            state, _ = env.reset()
            steps = 0
            total_reward = 0
            continue

        # 选动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        # pole 倒了 -> 结束程序
        if terminated or truncated:
            status = "满分!" if steps >= 500 else "倒了"
            print(f"Episode {episode}: {status} | 存活 {steps} 步 | 奖励 {total_reward:.0f}")
            print("程序结束")
            break

    env.close()


if __name__ == "__main__":
    main()
