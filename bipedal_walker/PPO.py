import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        # Actor 
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
    def get_value(self, x):
        return self.critic(x)
    # get an action randomly and adjust the percentage of the action
    def get_action_and_value(self, x, action = None):
        mean = self.action_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, entropy, value

def compute_gae(rewards, dones, values, next_value, next_done, gamma = 0.99, gae_lambda = 0.95):
    steps = len(rewards)
    advantages = torch.zeros(steps)
    lastgaelam = 0
    for t in reversed(range(steps)):
        if t == steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        # A_t = δ_t + (γ * λ) * A_{t+1}
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns