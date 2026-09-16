"""A compact goal-conditioned DDPG learner for continuous joint increments."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import TrainConfig


OBS_SCALE = torch.tensor(
    [3.0] * 7 + [3.0] * 7 + [0.8] * 3 + [1.0] * 4 + [1.0] * 7 + [0.8] * 3 + [1.0] * 4,
    dtype=torch.float32,
)


def _mlp(input_dim: int, output_dim: int, output_tanh: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim),
    ]
    if output_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _encode(observation: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    state = observation["observation"].to(device=device, dtype=torch.float32)
    goal = observation["desired_goal"].to(device=device, dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
        goal = goal.unsqueeze(0)
    return torch.cat((state, goal), dim=-1) / OBS_SCALE.to(device)


class DDPGAgent:
    def __init__(self, cfg: TrainConfig, *, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = _mlp(35, 7, output_tanh=True).to(self.device)
        self.critic = _mlp(42, 1).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.critic_target = copy.deepcopy(self.critic).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.updates = 0

    def act(
        self,
        observation: dict[str, np.ndarray],
        *,
        rng: np.random.Generator | None = None,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        tensors = {key: torch.as_tensor(value) for key, value in observation.items()}
        with torch.no_grad():
            action = self.actor(_encode(tensors, self.device))[0].cpu().numpy()
        if noise_std and rng is not None:
            action = action + rng.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        state = _encode(batch["curr_obs"], self.device)
        next_state = _encode(batch["next_obs"], self.device)
        action = batch["actions"].to(self.device, dtype=torch.float32).reshape(-1, 7)
        reward = batch["rewards"].to(self.device, dtype=torch.float32).reshape(-1, 1)
        terminated = batch["terminations"].to(self.device, dtype=torch.float32).reshape(-1, 1)

        # Time-limit truncation does not kill the bootstrap. A physical
        # collision does, and is stored as terminated by the environment.
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + self.cfg.gamma * (1.0 - terminated) * self.critic_target(
                torch.cat((next_state, next_action), dim=-1)
            )
        q = self.critic(torch.cat((state, action), dim=-1))
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        actor_action = self.actor(state)
        actor_loss = -self.critic(torch.cat((state, actor_action), dim=-1)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        for source, target in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.lerp_(source_param.data, self.cfg.tau)
        self.updates += 1
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "mean_q": float(q.detach().mean().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict[str, Any], *, policy_only: bool = False) -> None:
        self.actor.load_state_dict(state["actor"])
        if policy_only:
            return
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates = int(state["updates"])
        torch.set_rng_state(state["torch_rng"])
