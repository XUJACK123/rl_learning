"""Single-node RLinf replay runner for DDPG + episode-local HER."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .agent import DDPGAgent
from .config import TaskConfig, TrainConfig
from .env import BreadboardReachEnv
from .her import Transition
from .replay import RlinfHERReplay


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _log(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=_json_default) + "\n")


def load_agent(checkpoint: Path, device: str = "cpu") -> tuple[DDPGAgent, TaskConfig, TrainConfig]:
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    task_cfg = TaskConfig(**state["task_config"])
    train_cfg = TrainConfig(**state["train_config"])
    agent = DDPGAgent(train_cfg, device=device)
    weights = torch.load(checkpoint / "agent.pt", map_location=device, weights_only=False)
    agent.load_state_dict(weights, policy_only=True)
    agent.actor.eval()
    return agent, task_cfg, train_cfg


def evaluate_policy(
    agent: DDPGAgent,
    env: BreadboardReachEnv,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    hole_ids = [hole.hole_id for hole in env.available_holes]
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        hole_id = hole_ids[index % len(hole_ids)]
        observation, _ = env.reset(seed=seed + index, options={"hole_id": hole_id})
        last_info: dict[str, Any] = {}
        first_success_step: int | None = None
        for _ in range(env.cfg.max_episode_steps):
            action = agent.act(observation)
            observation, _, terminated, truncated, last_info = env.step(action)
            if last_info["success"] and first_success_step is None:
                first_success_step = int(last_info["step"])
            if terminated or truncated:
                break
        records.append(
            {
                "hole_id": hole_id,
                "success": bool(last_info["success"]),
                "collision": bool(last_info["collision"]),
                "position_error_m": float(last_info["position_error_m"]),
                "angular_error_deg": float(last_info["angular_error_deg"]),
                "first_success_step": first_success_step,
            }
        )
    successful_steps = [r["first_success_step"] for r in records if r["first_success_step"] is not None]
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "position_error_mean_m": float(np.mean([r["position_error_m"] for r in records])),
        "angular_error_mean_deg": float(np.mean([r["angular_error_deg"] for r in records])),
        "arrival_step_mean": float(np.mean(successful_steps)) if successful_steps else None,
        "records": records,
    }


def _save_checkpoint(
    run_dir: Path,
    seed: int,
    step: int,
    episode_id: int,
    agent: DDPGAgent,
    replay: RlinfHERReplay,
    rng: np.random.Generator,
    task_cfg: TaskConfig,
    train_cfg: TrainConfig,
) -> Path:
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:09d}"
    if final.exists():
        return final
    temporary = Path(tempfile.mkdtemp(prefix="checkpoint_", dir=root))
    replay.save_checkpoint(temporary / "replay")
    agent_state = agent.state_dict()
    agent_state["replay_rng"] = replay.buffer.random_generator.get_state()
    torch.save(agent_state, temporary / "agent.pt")
    _write_json(
        temporary / "state.json",
        {
            "seed": seed,
            "step": step,
            "episode_id": episode_id,
            "rng_state": rng.bit_generator.state,
            "task_config": task_cfg.asdict(),
            "train_config": train_cfg.asdict(),
            "rlinf_replay": True,
        },
    )
    temporary.rename(final)
    (run_dir / "latest_checkpoint.txt").write_text(str(final), encoding="utf-8")
    return final


def train(
    run_dir: Path,
    *,
    seed: int,
    task_cfg: TaskConfig | None = None,
    train_cfg: TrainConfig | None = None,
    device: str = "cpu",
    resume: Path | None = None,
) -> Path:
    """Train one seed; restartable checkpoints are written at episode boundaries."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = task_cfg or TaskConfig()
    train_cfg = train_cfg or TrainConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = DDPGAgent(train_cfg, device=device)
    train_env = BreadboardReachEnv(task_cfg, split="train")
    eval_env = BreadboardReachEnv(task_cfg, split="eval")
    replay = RlinfHERReplay(
        run_dir / "replay",
        seed=seed,
        window_episodes=train_cfg.replay_window_episodes,
        cache_episodes=train_cfg.replay_cache_episodes,
        max_episode_steps=task_cfg.max_episode_steps,
    )
    step = episode_id = 0
    if resume is not None:
        resume = Path(resume)
        saved = json.loads((resume / "state.json").read_text(encoding="utf-8"))
        if int(saved["seed"]) != seed:
            raise ValueError("Resume seed differs from checkpoint seed")
        normalized_task = json.loads(json.dumps(task_cfg.asdict()))
        normalized_train = json.loads(json.dumps(train_cfg.asdict()))
        prior_train = dict(saved["train_config"])
        prior_train["total_steps"] = normalized_train["total_steps"]
        if saved["task_config"] != normalized_task or prior_train != normalized_train:
            raise ValueError("Resume config differs from checkpoint config")
        if train_cfg.total_steps <= int(saved["step"]):
            raise ValueError("Resume total_steps must exceed the checkpoint step")
        agent_state = torch.load(resume / "agent.pt", map_location=device, weights_only=False)
        agent.load_state_dict(agent_state)
        replay.load_checkpoint(resume / "replay")
        replay.buffer.random_generator.set_state(agent_state["replay_rng"])
        rng.bit_generator.state = saved["rng_state"]
        step, episode_id = int(saved["step"]), int(saved["episode_id"])
    _write_json(
        run_dir / "config.json",
        {"seed": seed, "task_config": task_cfg.asdict(), "train_config": train_cfg.asdict()},
    )
    metrics_path = run_dir / "metrics.jsonl"
    observation, _ = train_env.reset(seed=seed + episode_id)
    episode: list[Transition] = []
    episode_return = 0.0
    last_checkpoint_step = step
    last_eval_step = step
    try:
        progress = tqdm(
            total=train_cfg.total_steps,
            initial=step,
            desc=f"seed {seed}",
            unit="step",
            dynamic_ncols=True,
        )
        while step < train_cfg.total_steps:
            step += 1
            progress.update(1)
            if step <= train_cfg.warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
            else:
                action = agent.act(observation, rng=rng, noise_std=train_cfg.exploration_std)
            next_observation, reward, terminated, truncated, info = train_env.step(action)
            episode.append(
                Transition(
                    observation=observation,
                    action=np.asarray(action, dtype=np.float32),
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    collision=bool(info["collision"]),
                    episode_id=episode_id,
                    step_index=len(episode),
                )
            )
            episode_return += reward
            observation = next_observation
            if terminated or truncated:
                samples_added = replay.add_episode(
                    episode,
                    future_k=train_cfg.her_future_k,
                    rng=rng,
                    reward_fn=train_env.compute_reward,
                )
                _log(
                    metrics_path,
                    {
                        "type": "episode",
                        "step": step,
                        "episode_id": episode_id,
                        "hole_id": info["hole_id"],
                        "return": episode_return,
                        "length": len(episode),
                        "success": info["success"],
                        "collision": info["collision"],
                        "position_error_m": info["position_error_m"],
                        "angular_error_deg": info["angular_error_deg"],
                        "samples_added": samples_added,
                        "replay_samples": replay.total_samples,
                    },
                )
                # 更新进度条上的实时指标
                progress.set_postfix(
                    ep=episode_id,
                    ret=f"{episode_return:.0f}",
                    succ=f"{info['success']}",
                    err=f"{1000 * info['position_error_m']:.0f}mm",
                    col=f"{info['collision']}",
                )
                episode_id += 1
                episode = []
                episode_return = 0.0
                observation, _ = train_env.reset(seed=seed + episode_id)
                if step - last_eval_step >= train_cfg.eval_interval:
                    result = evaluate_policy(
                        agent, eval_env, episodes=train_cfg.eval_episodes, seed=seed + 100_000
                    )
                    _log(metrics_path, {"type": "eval", "step": step, **{k: v for k, v in result.items() if k != "records"}})
                    progress.write(f"eval @ {step}: success={result['success_rate']:.3f} collision={result['collision_rate']:.3f}")
                    last_eval_step = step
                if step - last_checkpoint_step >= train_cfg.checkpoint_interval:
                    _save_checkpoint(
                        run_dir, seed, step, episode_id, agent, replay, rng, task_cfg, train_cfg
                    )
                    progress.write(f"checkpoint @ {step}")
                    last_checkpoint_step = step
            if step > train_cfg.warmup_steps and replay.total_samples >= train_cfg.batch_size:
                for _ in range(train_cfg.updates_per_step):
                    losses = agent.update(replay.sample(train_cfg.batch_size))
                if step % 1000 == 0:
                    _log(metrics_path, {"type": "update", "step": step, **losses})
        progress.close()
        # Finish a partial episode so all collected transitions have valid
        # boundaries and can be relabeled before the final checkpoint.
        if episode:
            last = episode[-1]
            episode[-1] = replace(last, truncated=True)
            replay.add_episode(
                episode,
                future_k=train_cfg.her_future_k,
                rng=rng,
                reward_fn=train_env.compute_reward,
            )
            episode_id += 1
        return _save_checkpoint(run_dir, seed, step, episode_id, agent, replay, rng, task_cfg, train_cfg)
    finally:
        replay.close()
        train_env.close()
        eval_env.close()
