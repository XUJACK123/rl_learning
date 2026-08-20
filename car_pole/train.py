"""Train PPO on CartPole with CPU and save a curve, GIF, and model.

Run in a ModelScope Notebook or locally:
    pip install "gymnasium[classic-control]" stable-baselines3 matplotlib imageio
    python train.py --timesteps 30000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 42


def evaluate(model: PPO, episodes: int = 5) -> tuple[float, float]:
    """Return the mean and standard deviation of deterministic episode rewards."""
    env = gym.make("CartPole-v1")
    rewards, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def save_curve(steps: list[int], rewards: list[float], output_dir: Path) -> Path:
    """Save the evaluation reward curve."""
    path = output_dir / "reward-curve.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, marker="o", linewidth=2, color="#4f46e5")
    ax.axhline(475, linestyle="--", linewidth=1.2, color="#16a34a", label="Solved: 475")
    ax.set(xlabel="Training steps", ylabel="Mean reward", ylim=(0, 510))
    ax.set_title("PPO on CartPole-v1")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_gif(model: PPO, output_dir: Path) -> tuple[Path, float]:
    """Record one deterministic episode as a GIF."""
    path = output_dir / "cartpole-trained-policy.gif"
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=SEED + 1)
    frames = []
    score = 0.0

    for _ in range(500):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(path, frames, duration=1 / 30, loop=0)
    return path, score


def train(total_timesteps: int, output_dir: Path) -> None:
    """Train PPO in chunks and report independent evaluation rewards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        device="cpu",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
    )

    chunk_size = 2_000
    steps = [0]
    initial_mean, initial_std = evaluate(model)
    rewards = [initial_mean]
    print(f"Initial policy: {initial_mean:.1f} +/- {initial_std:.1f}")

    started_at = time.perf_counter()
    trained = 0
    while trained < total_timesteps:
        chunk = min(chunk_size, total_timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        trained += chunk
        mean_reward, std_reward = evaluate(model)
        steps.append(trained)
        rewards.append(mean_reward)
        print(
            f"{trained:>6,}/{total_timesteps:,} steps | "
            f"reward {mean_reward:>6.1f} +/- {std_reward:>5.1f}"
        )

    model_path = output_dir / "ppo-cartpole"
    model.save(model_path)
    curve_path = save_curve(steps, rewards, output_dir)
    gif_path, demo_score = save_gif(model, output_dir)
    final_mean, final_std = evaluate(model, episodes=10)
    elapsed = time.perf_counter() - started_at
    env.close()

    print(f"\nFinished in {elapsed:.1f} seconds")
    print(f"10-episode reward: {final_mean:.1f} +/- {final_std:.1f}")
    print(f"Recorded episode: {demo_score:.0f}/500")
    print(f"Curve: {curve_path}")
    print(f"GIF:   {gif_path}")
    print(f"Model: {model_path}.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.timesteps, args.output_dir)
