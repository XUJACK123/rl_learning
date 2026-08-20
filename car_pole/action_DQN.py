import random
import gymnasium as gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v1")
"""
four action space
- Cart Position
- Cart Velocity
- Pole Angle
- Pole Velocity at Tip
two discrete actions: left or right
"""
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
"""
using DQN(Deep Q-Network) to solve the problem
"""
class DQN_training(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer, for storing the experience, to break the correlation between the samples
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
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)

# training progress
def train_dqn():
    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    learning_rate = 0.001
    target_update_freq = 10

    # for eval network, the parameters and target will change continuously
    policy_net = DQN_training(state_dim, action_dim)
    # for target network, the parameters and theQ will not change, for calculating the target Q value
    target_net = DQN_training(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(capacity=10000)
    for episode in range(300):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # exploration, using different action to explore
            if random.random() < epsilon:
                action = env.action_space.sample()
            # exploitation, using the policy network to choose the best action
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # save the experience to the replay buffer
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
            print(f"\n demostration of the vision of the agent")
            eval_env = gym.make("CartPole-v1", render_mode="human")
            eval_state, _ = eval_env.reset()
            eval_done = False
            eval_steps = 0
            while not eval_done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(eval_state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()
                eval_state, _, terminated, truncated, _ = eval_env.step(action)
                eval_done = terminated or truncated
                eval_steps += 1
            eval_env.close()

    # save the trained model
    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
    print("Model saved to dqn_cartpole.pth")

    test_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = test_env.reset()
    done = False
    steps = 0
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        steps += 1
    print(f"Test completed after {steps} steps.")

if __name__ == "__main__":
    train_dqn()
