"""训练后可视化：加载 checkpoint，单环境渲染机器人走路。

用法:
    python3 visualize.py                          # 加载 final 模型
    python3 visualize.py --checkpoint checkpoints/ppo_humanoid_iter_20.pth
"""
import argparse
import os

import numpy as np
import torch

from env import CustomHumanoidWalkingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与 PPO.py 中 ActorCritic 结构一致
from PPO import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint is None:
        ckpt_path = os.path.join(save_dir, "checkpoints", "ppo_humanoid_final.pth")
    else:
        ckpt_path = args.checkpoint

    agent = ActorCritic().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"加载模型: {ckpt_path} (iteration={ckpt.get('iteration', '?')})")

    env = CustomHumanoidWalkingEnv(render_mode="human")
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    done = False
    try:
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist, _ = agent.forward(obs_t)
                action = dist.mean[0].cpu().numpy()   # 确定性动作（用均值，不加噪声）
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            done = terminated or truncated
            if done:
                status = "摔倒" if terminated else "超时"
                print(f"结束: {status} | 存活 {steps} 步 | 总奖励 {total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
