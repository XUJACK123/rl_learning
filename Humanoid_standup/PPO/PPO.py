import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from PPO.env import CustomHumanoidWalkingEnv

RENDER_MODE = "human"
env = CustomHumanoidWalkingEnv(render_mode=RENDER_MODE)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim = 45, action_dim = 17, hidden = 256):
        super().__init__()
        self.shared_ = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_ = nn.Linear(hidden, action_dim)
        self.critic_ = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        # 提取特征
        features = self.shared_(obs)
        # 输出动作均值
        mean = self.actor_(features)
        # 构造随机性
        std = torch.exp(self.log_std)
        value = self.critic_(features).squeeze(-1)
        dist = Normal(mean, std)
        return dist, value

    def get_action_and_value(self, obs, action=None):
        # 输出策略分布和critic的评价
        distribution, value = self.forward(obs)
        if action is None:
            action = distribution.sample()
        # 计算前策略分布下的对数概率
        log_probs = distribution.log_prob(action).sum(-1)
        # 计算策略不确定性的度量
        entropy = distribution.entropy().sum(-1)
        return action, log_probs, entropy, value

# 在参数更新之前采集数据，先用当前的策略和环境连续跑num_steps步，每步记一笔账(PPO更新前会先拿这些反复用几轮)
def rollout(agent, env, num_steps):
    obs, _ = env.reset()
    next_value = torch.tensor(0.0)
    obs_list = []
    action_list = []
    log_prob_list = []
    value_list = []
    reward_list = []
    done_list = []
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0
    for i in range(num_steps):
        action, log_prob, _, value = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32))
        next_obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
        obs_list.append(torch.tensor(obs, dtype=torch.float32))
        action_list.append(action.detach())
        log_prob_list.append(log_prob.detach())
        value_list.append(value.detach())
        reward_list.append(torch.tensor(reward, dtype=torch.float32))
        done = terminated or truncated
        done_list.append(torch.tensor(float(done), dtype=torch.float32))
        current_return += reward
        current_length += 1
        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            current_return = 0
            current_length = 0
            obs, _ = env.reset()
            next_value = torch.tensor(0.0)
        else:
            obs = next_obs
    if not done:
        _, next_value = agent.forward(torch.tensor(obs, dtype=torch.float32))
    return torch.stack(obs_list), torch.stack(action_list), torch.stack(log_prob_list), torch.stack(value_list), torch.stack(reward_list), torch.stack(done_list), next_value.detach(), episode_returns, episode_lengths

# 计算整条轨迹（Trajectory）中“每一个时间步 $t$”的优势值（Advantage）和目标回报（Return），并打包成两个张量（Tensor）统一返回
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = []
    returns = []
    lastgaelam = 0
    for t in reversed(range(len(values))):
        # 计算出实际表现比网络估计好/差多少
        if t == len(values) - 1:
            next_val = next_value
        else:
            next_val = values[t+1]
        delta = rewards[t] + gamma*next_val - values[t]
        # 计算出这个动作比平均预期好多少，未来的好 = 当前动作的好(delta)+对于现在而言未来的好
        lastgaelam = delta + gamma*lam*lastgaelam*(1-dones[t])
        returns.append(lastgaelam + values[t])
        advantages.append(lastgaelam)
    returns.reverse()
    advantages.reverse()
    return torch.stack(advantages).float(), torch.stack(returns).float()

def update_ppo(agent, optimizer, obs, actions, old_log_probs, advantages, returns, epsilon, c1, c2):
    _, new_log_probs, entropy, values = agent.get_action_and_value(obs, actions)
    # 计算重要性采样比值
    ratio = torch.exp(new_log_probs - old_log_probs)
    # 未裁剪目标
    surr1 = ratio*advantages
    # 裁剪后的目标
    surr2 = torch.clip(ratio, 1-epsilon, 1+epsilon)*advantages
    # 熵损失,c1为熵损失系数
    L_entropy = c1*(entropy.mean())
    policy_loss = torch.minimum(surr1, surr2).mean()
    value_loss = ((values - returns) ** 2).mean()
    # c2为价值损失系数
    loss = - policy_loss - L_entropy + c2*value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item(), entropy.mean().item()

def main():
    num_steps = 2048
    num_epochs = 10
    num_iterations = 20
    minibatch_size = 64
    gamma = 0.99
    lam = 0.95
    epsilon = 0.2
    c1 = 0.01
    c2 = 0.5
    lr_rate = 3e-4
    agent = ActorCritic()
    optimizer = optim.Adam(agent.parameters(), lr=lr_rate)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_interval = 5
    for iteration in range(1, num_iterations+1):
        obs, actions, old_log_probs, value, reward, done, next_value, ep_returns, ep_lengths = rollout(agent, env, num_steps)
        advantages, returns = compute_gae(reward, value, done, next_value, gamma, lam)
        advantages = advantages.detach()
        returns = returns.detach()
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        batch_size = num_steps
        b_obs = obs.reshape(-1, *obs.shape[1:])
        b_actions = actions.reshape(-1, *actions.shape[1:])
        b_log_probs = old_log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        p_losses, v_losses, entropies = [], [], []
        for epoch in range(num_epochs):
            perm_indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = perm_indices[start:end]
                p_loss, v_loss, ent = update_ppo(agent=agent,
                    optimizer=optimizer,
                    obs=b_obs[mb_inds],
                    actions=b_actions[mb_inds],
                    old_log_probs=b_log_probs[mb_inds],
                    advantages=b_advantages[mb_inds],
                    returns=b_returns[mb_inds],
                    epsilon=epsilon,
                    c1=c1,
                    c2=c2
                )
                p_losses.append(p_loss)
                v_losses.append(v_loss)
                entropies.append(ent)
        mean_step_reward = reward.mean().item()
        mean_ep_return = np.mean(ep_returns) if len(ep_returns) > 0 else 0.0
        mean_ep_length = np.mean(ep_lengths) if len(ep_lengths) > 0 else 0.0
        print(
            f"Iter [{iteration}/{num_iterations}] | "
            f"Step Reward: {mean_step_reward:.3f} | "
            f"Ep Return: {mean_ep_return:.2f} ({len(ep_returns)} eps) | "
            f"Ep Len: {mean_ep_length:.1f} | "
            f"Policy Loss: {np.mean(p_losses):.4f} | "
            f"Value Loss: {np.mean(v_losses):.4f} | "
            f"Entropy: {np.mean(entropies):.4f}"
        )
        env.render()
        if iteration % save_interval == 0 or iteration == num_iterations:
            checkpoint = {
                "iteration": iteration,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(save_dir, f"ppo_humanoid_iter_{iteration}.pth")
            torch.save(checkpoint, checkpoint_path)
    final_checkpoint = {
        "iteration": num_iterations,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(save_dir, "ppo_humanoid_final.pth"))
    env.close()

if __name__ == "__main__":
    main()
