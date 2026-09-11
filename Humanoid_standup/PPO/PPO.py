import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

from stable_baselines3.common.vec_env import SubprocVecEnv

from env import CustomHumanoidWalkingEnv

# 设备：优先 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env():
    """每个并行子进程用它新建一个环境（训练环境不渲染）。"""
    return CustomHumanoidWalkingEnv(render_mode=None)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=45, action_dim=17, hidden=256):
        super().__init__()
        self.shared_ = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_ = nn.Linear(hidden, action_dim)
        self.critic_ = nn.Linear(hidden, 1)
        # 初始探索幅度：std 从约 0.6 开始（exp(-0.5)），而不是 1.0，
        # 避免动作一开始就被 clip 到 ±0.4 饱和导致剧烈抖动摔倒。
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs):
        features = self.shared_(obs)
        mean = self.actor_(features)
        std = torch.exp(self.log_std)
        value = self.critic_(features).squeeze(-1)
        dist = Normal(mean, std)
        return dist, value

    def get_action_and_value(self, obs, action=None):
        distribution, value = self.forward(obs)
        if action is None:
            action = distribution.sample()
        log_probs = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return action, log_probs, entropy, value


# 并行采样：16 个环境同时跑，每个环境 num_steps 步，收集一整批数据
def rollout(agent, env, num_steps):
    n_envs = env.num_envs
    obs = env.reset()                      # (n_envs, 45)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    obs_list, action_list, log_prob_list, value_list = [], [], [], []
    reward_list, done_list = [], []
    episode_returns, episode_lengths = [], []
    current_return = np.zeros(n_envs)
    current_length = np.zeros(n_envs)

    for _ in range(num_steps):
        with torch.no_grad():
            action, log_prob, _, value = agent.get_action_and_value(obs)
        next_obs, reward, dones, infos = env.step(action.cpu().numpy())

        obs_list.append(obs)
        action_list.append(action)
        log_prob_list.append(log_prob)
        value_list.append(value)
        reward_list.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.as_tensor(dones, dtype=torch.float32, device=device))

        current_return += reward
        current_length += 1
        for e in range(n_envs):
            if dones[e]:
                episode_returns.append(current_return[e])
                episode_lengths.append(current_length[e])
                current_return[e] = 0.0
                current_length[e] = 0.0

        obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, next_value = agent.forward(obs)

    obs_t = torch.stack(obs_list)          # (num_steps, n_envs, 45)
    act_t = torch.stack(action_list)       # (num_steps, n_envs, 17)
    lp_t = torch.stack(log_prob_list)      # (num_steps, n_envs)
    val_t = torch.stack(value_list)        # (num_steps, n_envs)
    rew_t = torch.stack(reward_list)       # (num_steps, n_envs)
    don_t = torch.stack(done_list)         # (num_steps, n_envs)
    return obs_t, act_t, lp_t, val_t, rew_t, don_t, next_value, episode_returns, episode_lengths


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """计算 GAE 优势与回报。

    rewards/values/dones: (num_steps, n_envs)，next_value: (n_envs,)。
    每个环境独立沿时间反向递推，环境之间不混淆。
    """
    num_steps, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(n_envs, device=rewards.device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam * (1 - dones[t])
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def update_ppo(agent, optimizer, obs, actions, old_log_probs, advantages, returns, epsilon, c1, c2):
    _, new_log_probs, entropy, values = agent.get_action_and_value(obs, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    L_entropy = c1 * entropy.mean()
    policy_loss = torch.minimum(surr1, surr2).mean()
    value_loss = ((values - returns) ** 2).mean()
    loss = -policy_loss - L_entropy + c2 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item(), entropy.mean().item()


def main():
    num_envs = 16
    num_steps = 2048
    num_epochs = 10
    # 总步数 ≈ num_iterations × num_steps × num_envs = 122 × 2048 × 16 ≈ 400 万步
    num_iterations = 122
    minibatch_size = 64
    gamma = 0.99
    lam = 0.95
    epsilon = 0.2
    c1 = 0.01
    c2 = 0.5
    lr_rate = 3e-4

    agent = ActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr_rate)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_interval = 5

    print(f"设备: {device} | 并行环境: {num_envs} | 每轮总步数: {num_steps * num_envs}")

    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    try:
        pbar = tqdm(range(1, num_iterations + 1), desc="PPO 训练", unit="iter")
        for iteration in pbar:
            obs, actions, old_log_probs, value, reward, done, next_value, ep_returns, ep_lengths = \
                rollout(agent, env, num_steps)

            # 展平 (num_steps, n_envs, ...) -> (total, ...) 供小批量更新使用
            total = num_steps * num_envs
            b_obs = obs.reshape(total, -1)
            b_actions = actions.reshape(total, -1)
            b_log_probs = old_log_probs.reshape(-1)
            b_value = value.reshape(-1)
            b_reward = reward.reshape(-1)
            b_done = done.reshape(-1)

            # GAE 按 (num_steps, n_envs) 计算，保持环境边界正确
            advantages, returns = compute_gae(reward, value, done, next_value, gamma, lam)
            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            p_losses, v_losses, entropies = [], [], []
            for _ in range(num_epochs):
                perm = torch.randperm(total, device=device)
                for start in range(0, total, minibatch_size):
                    mb = perm[start:start + minibatch_size]
                    p_loss, v_loss, ent = update_ppo(
                        agent, optimizer,
                        obs=b_obs[mb], actions=b_actions[mb],
                        old_log_probs=b_log_probs[mb],
                        advantages=advantages[mb], returns=returns[mb],
                        epsilon=epsilon, c1=c1, c2=c2,
                    )
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)
                    entropies.append(ent)

            mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0
            mean_ep_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            pbar.set_postfix(
                ep_ret=f"{mean_ep_return:.1f}",
                ep_len=f"{mean_ep_length:.0f}",
                p_loss=f"{np.mean(p_losses):.3f}",
                v_loss=f"{np.mean(v_losses):.3f}",
                ent=f"{np.mean(entropies):.2f}",
            )

            if iteration % save_interval == 0 or iteration == num_iterations:
                checkpoint = {
                    "iteration": iteration,
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_dir, f"ppo_humanoid_iter_{iteration}.pth"))

        torch.save(
            {"iteration": num_iterations, "agent_state_dict": agent.state_dict()},
            os.path.join(save_dir, "ppo_humanoid_final.pth"),
        )
        print(f"\n训练完成，模型已保存到 {save_dir}/ppo_humanoid_final.pth")
    finally:
        env.close()


if __name__ == "__main__":
    main()
