"""Single-node RLinf replay runner for DDPG + episode-local HER."""

from __future__ import annotations

import json
import platform
import shutil
import tempfile
from importlib import metadata
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


def _runtime_metadata() -> dict:
    packages = ("torch", "mujoco", "mujoco-menagerie", "rlinf", "gymnasium", "numpy")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


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
    action_sum = np.zeros(7, dtype=np.float64)
    saturated = shielded = blocked = total_actions = 0
    for index in range(episodes):
        hole_id = hole_ids[index % len(hole_ids)]
        observation, _ = env.reset(seed=seed + index, options={"hole_id": hole_id})
        last_info: dict[str, Any] = {}
        first_success_step: int | None = None
        episode_shielded = 0
        episode_blocked = 0
        for _ in range(env.cfg.max_episode_steps):
            action = agent.act(observation)
            observation, _, terminated, truncated, last_info = env.step(action)
            action_sum += action
            saturated += int(np.sum(np.abs(action) >= 0.95))
            shielded += int(last_info["shielded"])
            blocked += int(last_info["blocked"])
            episode_shielded += int(last_info["shielded"])
            episode_blocked += int(last_info["blocked"])
            total_actions += 1
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
                "shielded_steps": episode_shielded,
                "blocked_steps": episode_blocked,
                "collision_pair": last_info["collision_pair"],
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
        "shield_rate": shielded / max(1, total_actions),
        "blocked_rate": blocked / max(1, total_actions),
        "action_mean": (action_sum / max(1, total_actions)).tolist(),
        "action_saturation_rate": saturated / max(1, 7 * total_actions),
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
        (run_dir / "latest_checkpoint.txt").write_text(str(final.relative_to(run_dir)), encoding="utf-8")
        return final
    temporary = Path(tempfile.mkdtemp(prefix="checkpoint_", dir=root))
    if (run_dir / "demonstrations.pt").exists():
        shutil.copy2(run_dir / "demonstrations.pt", temporary / "demonstrations.pt")
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
    (run_dir / "latest_checkpoint.txt").write_text(str(final.relative_to(run_dir)), encoding="utf-8")
    return final


def _seed_demonstrations(
    train_env: BreadboardReachEnv,
    eval_env: BreadboardReachEnv,
    replay: RlinfHERReplay,
    agent: DDPGAgent,
    rng: np.random.Generator,
    cfg: TrainConfig,
    metrics_path: Path,
) -> dict[str, Any]:
    from .diagnostics import _jacobian_action, generate_demonstration
    from .geometry import goal_reached

    demonstrations: list[Transition] = []
    episode_id = -1
    for hole in train_env.available_holes:
        completed = 0
        for attempt in range(max(4, cfg.demo_episodes_per_hole * 4)):
            episode, result = generate_demonstration(
                train_env, hole.hole_id, seed=10_000 + attempt, waypoint=bool(attempt % 2)
            )
            if not result["success"]:
                continue
            episode = [replace(transition, episode_id=episode_id) for transition in episode]
            replay.add_episode(
                episode,
                future_k=cfg.her_future_k,
                rng=rng,
                reward_fn=train_env.compute_reward,
            )
            demonstrations.extend(episode)
            episode_id -= 1
            completed += 1
            if completed == cfg.demo_episodes_per_hole:
                break
        _log(metrics_path, {"type": "demonstration", "hole_id": hole.hole_id, "successful": completed})
        if completed != cfg.demo_episodes_per_hole:
            raise RuntimeError(f"IK controller could only generate {completed} successful demonstrations for {hole.hole_id}")
    for hole in eval_env.available_holes:
        attempts = [
            generate_demonstration(eval_env, hole.hole_id, seed=20_000 + attempt, waypoint=bool(attempt % 2))[1]
            for attempt in range(4)
        ]
        if not any(result["success"] for result in attempts):
            raise RuntimeError(f"Held-out hole {hole.hole_id} failed dynamic reachability: {attempts}")
    observations = {
        key: torch.from_numpy(np.stack([transition.observation[key] for transition in demonstrations]))
        for key in ("observation", "desired_goal")
    }
    actions = torch.from_numpy(np.stack([transition.action for transition in demonstrations]))
    for index in range(cfg.bc_updates):
        batch_indices = rng.integers(len(demonstrations), size=cfg.batch_size)
        loss = agent.behavior_clone(
            {key: value[batch_indices] for key, value in observations.items()},
            actions[batch_indices],
        )
        if (index + 1) % 100 == 0:
            _log(metrics_path, {"type": "bc", "update": index + 1, "loss": loss})
    for round_index in range(cfg.dagger_rounds):
        collected: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        for hole_index, hole in enumerate(train_env.available_holes):
            observation, _ = train_env.reset(
                seed=40_000 + round_index * 1_000 + hole_index,
                options={"hole_id": hole.hole_id},
            )
            for _ in range(train_env.cfg.max_episode_steps):
                teacher = (
                    np.zeros(7, dtype=np.float32)
                    if bool(goal_reached(observation["achieved_goal"], observation["desired_goal"], train_env.cfg))
                    else _jacobian_action(train_env, observation["desired_goal"])
                )
                safe_teacher, _, blocked = train_env._safe_action(teacher)
                if not blocked:
                    collected.append((observation, safe_teacher.copy()))
                policy_action = agent.act(observation)
                observation, _, terminated, truncated, _ = train_env.step(policy_action)
                if terminated or truncated:
                    break
        if collected:
            observations = {
                key: torch.cat((value, torch.from_numpy(np.stack([item[0][key] for item in collected]))))
                for key, value in observations.items()
            }
            actions = torch.cat((actions, torch.from_numpy(np.stack([item[1] for item in collected]))))
        for update_index in range(cfg.dagger_bc_updates):
            batch_indices = rng.integers(len(actions), size=cfg.batch_size)
            loss = agent.behavior_clone(
                {key: value[batch_indices] for key, value in observations.items()},
                actions[batch_indices],
            )
        _log(metrics_path, {
            "type": "dagger",
            "round": round_index + 1,
            "samples_added": len(collected),
            "samples_total": len(actions),
            "loss": loss if cfg.dagger_bc_updates else None,
        })
    dataset = {"observations": observations, "actions": actions}
    agent.sync_actor_target()
    for index in range(cfg.critic_pretrain_updates):
        result = agent.update(replay.sample(cfg.batch_size), actor_update=False)
        if (index + 1) % 100 == 0:
            _log(metrics_path, {"type": "critic_pretrain", "update": index + 1, "loss": result["critic_loss"]})
    return dataset


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
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}; use --resume or a new output root")
    run_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = task_cfg or TaskConfig()
    train_cfg = train_cfg or TrainConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = DDPGAgent(train_cfg, device=device)
    train_env = BreadboardReachEnv(task_cfg, split="train")
    eval_env = BreadboardReachEnv(task_cfg, split="eval")
    if resume is None:
        from .diagnostics import diagnose_control, find_safe_home

        control_check = diagnose_control(train_env, seeds=5)
        if not control_check["stable"]:
            safe_home = find_safe_home(train_env)
            train_env.close()
            eval_env.close()
            task_cfg = replace(task_cfg, home_joint_positions=safe_home)
            train_env = BreadboardReachEnv(task_cfg, split="train")
            eval_env = BreadboardReachEnv(task_cfg, split="eval")
            control_check = diagnose_control(train_env, seeds=20)
            if not control_check["stable"]:
                raise RuntimeError("Reset is still unstable after safe-home selection")
        _write_json(run_dir / "control_diagnostic.json", control_check)
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
        replay.buffer.random_generator.set_state(agent_state["replay_rng"].cpu())
        rng.bit_generator.state = saved["rng_state"]
        step, episode_id = int(saved["step"]), int(saved["episode_id"])
    _write_json(run_dir / "config.json", {
        "seed": seed,
        "task_config": task_cfg.asdict(),
        "train_config": train_cfg.asdict(),
        "runtime": _runtime_metadata(),
    })
    metrics_path = run_dir / "metrics.jsonl"
    if resume is None:
        demonstrations = _seed_demonstrations(train_env, eval_env, replay, agent, rng, train_cfg, metrics_path)
        torch.save(demonstrations, run_dir / "demonstrations.pt")
    else:
        demo_path = run_dir / "demonstrations.pt"
        if not demo_path.exists():
            shutil.copy2(resume / "demonstrations.pt", demo_path)
        demonstrations = torch.load(demo_path, map_location="cpu", weights_only=True)
    observation, _ = train_env.reset(seed=seed + episode_id)
    episode: list[Transition] = []
    episode_return = 0.0
    episode_shielded = 0
    episode_blocked = 0
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
            episode_shielded += int(info["shielded"])
            episode_blocked += int(info["blocked"])
            episode.append(
                Transition(
                    observation=observation,
                    action=np.asarray(info["executed_action"], dtype=np.float32),
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    collision=bool(info["collision"]),
                    episode_id=episode_id,
                    step_index=len(episode),
                    shielded=bool(info["shielded"]),
                )
            )
            episode_return += reward
            observation = next_observation
            completed_episode = terminated or truncated
            if completed_episode:
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
                        "collision_pair": info["collision_pair"],
                        "shielded_steps": episode_shielded,
                        "blocked_steps": episode_blocked,
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
                    sh=f"{episode_shielded}",
                )
                episode_id += 1
                episode = []
                episode_return = 0.0
                episode_shielded = 0
                episode_blocked = 0
                observation, _ = train_env.reset(seed=seed + episode_id)
            if step > train_cfg.warmup_steps and replay.total_samples >= train_cfg.batch_size:
                for _ in range(train_cfg.updates_per_step):
                    indices = rng.integers(len(demonstrations["actions"]), size=train_cfg.batch_size)
                    demo_batch = (
                        {key: value[indices] for key, value in demonstrations["observations"].items()},
                        demonstrations["actions"][indices],
                    )
                    losses = agent.update(replay.sample(train_cfg.batch_size), demonstration=demo_batch)
                if step % 1000 == 0:
                    _log(metrics_path, {"type": "update", "step": step, **losses})
            if completed_episode:
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
