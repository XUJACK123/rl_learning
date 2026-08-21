import gymnasium as gym
from car_pole import car_pole_env  # 导入即触发 gym.register("CustomCartPole-v0")
from car_pole.DQN import DQNAgent, device

def run_episode(agent, env, train=True, batch_size=256):
    """跑一个回合,train=True 时收集经验并做梯度更新，返回总奖励。"""
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        # 训练时 epsilon-greedy 探索；评估时 epsilon=0 纯贪心
        epsilon = None if train else 0.0
        action = agent.select_action(state, env.action_space, epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.memory.push(state, action, reward, next_state, done)
            agent.update(batch_size)
        state = next_state
        total_reward += reward
    return total_reward


def evaluate(agent, render=True):
    """用渲染环境评估一个回合，返回存活步数。"""
    eval_env = gym.make("CustomCartPole-v0", render_mode="human" if render else None)
    try:
        return run_episode(agent, eval_env, train=False)
    finally:
        eval_env.close()


def train_dqn(num_episodes=400, batch_size=256, eval_every=50):
    # 训练环境不渲染，速度比 render_mode="human" 快很多
    env = gym.make("CustomCartPole-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Using device: {device} | state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgent(state_dim, action_dim)

    for episode in range(num_episodes):
        total_reward = run_episode(agent, env, train=True, batch_size=batch_size)
        agent.decay_epsilon()

        # 周期性同步 target 网络（与训练频率解耦，按回合计）
        if episode % 10 == 0:
            agent.update_target()

        if episode % eval_every == 0:
            print(
                f"Episode {episode}: Total Reward = {total_reward:.0f}, "
                f"Epsilon = {agent.epsilon:.4f}"
            )
            print("demonstration of the vision of the agent")
            eval_steps = evaluate(agent)
            print(f"Eval steps: {eval_steps}")

    agent.save("dqn_cartpole.pth")
    print("Model saved to dqn_cartpole.pth")
    env.close()

    # 最终测试
    test_steps = evaluate(agent)
    print(f"Test completed after {test_steps} steps.")


if __name__ == "__main__":
    train_dqn()