"""Commands for training, validation, deterministic evaluation and replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import TaskConfig, TrainConfig


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="RLinf DDPG+HER breadboard reach")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one or more independent seeds")
    train_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    train_parser.add_argument("--steps", type=int, default=1_000_000)
    train_parser.add_argument("--warmup", type=int, default=10_000)
    train_parser.add_argument("--eval-interval", type=int, default=10_000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    train_parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "outputs" / "runs")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--resume", type=Path)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate held-out holes without rendering")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--episodes", type=int, default=200)
    eval_parser.add_argument("--seed", type=int, default=42_000)
    eval_parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs" / "visualization" / "evaluation.json")
    eval_parser.add_argument("--device", default="auto")

    report_parser = subparsers.add_parser("report", help="Evaluate and summarize three seed checkpoints")
    report_parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    report_parser.add_argument("--episodes", type=int, default=200)
    report_parser.add_argument("--seed", type=int, default=42_000)
    report_parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs" / "visualization" / "report.json")
    report_parser.add_argument("--device", default="auto")

    view_parser = subparsers.add_parser("visualize", help="Play or record annotated MuJoCo episodes")
    view_parser.add_argument("--checkpoint", type=Path, required=True)
    view_parser.add_argument("--mode", choices=("record", "view"), default="record")
    view_parser.add_argument("--hole-id")
    view_parser.add_argument("--episodes", type=int, default=10)
    view_parser.add_argument("--seed", type=int, default=42_000)
    view_parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "visualization")
    view_parser.add_argument("--device", default="auto")

    check_parser = subparsers.add_parser("check-env", help="Check goal sampling, spaces and actuation")
    check_parser.add_argument("--seed", type=int, default=123)
    check_parser.add_argument("--steps", type=int, default=10)
    check_parser.add_argument("--skip-ik", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        from .runner import train

        if args.resume is not None and len(args.seeds) != 1:
            parser.error("--resume requires exactly one --seeds value")
        cfg = replace(
            TrainConfig(),
            total_steps=args.steps,
            warmup_steps=args.warmup,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
        )
        for seed in args.seeds:
            run_dir = (
                args.resume.parent.parent if args.resume is not None else args.output_root / f"seed_{seed}"
            )
            checkpoint = train(
                run_dir,
                seed=seed,
                task_cfg=TaskConfig(),
                train_cfg=cfg,
                device=_device(args.device),
                resume=args.resume,
            )
            print(f"seed {seed}: {checkpoint}")
    elif args.command == "evaluate":
        from .env import BreadboardReachEnv
        from .runner import evaluate_policy, load_agent
        from .visualize import _plot_holes, _plot_training

        agent, task_cfg, _ = load_agent(args.checkpoint, device=_device(args.device))
        env = BreadboardReachEnv(task_cfg, split="eval")
        try:
            summary = evaluate_policy(agent, env, episodes=args.episodes, seed=args.seed)
        finally:
            env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _plot_holes(summary["records"], args.output.parent / "per_hole.png")
        _plot_training(args.checkpoint.parent.parent / "metrics.jsonl", args.output.parent / "training.png")
        print(f"success={summary['success_rate']:.3f} collision={summary['collision_rate']:.3f}")
    elif args.command == "report":
        from .env import BreadboardReachEnv
        from .runner import evaluate_policy, load_agent
        from .visualize import _plot_holes

        if len(args.checkpoints) != 3:
            parser.error("--checkpoints requires exactly three independently trained seeds")
        checkpoint_seeds = [
            json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))["seed"]
            for checkpoint in args.checkpoints
        ]
        if len(set(checkpoint_seeds)) != 3:
            parser.error("--checkpoints must come from three different seeds")
        per_seed = []
        all_records = []
        for index, checkpoint in enumerate(args.checkpoints):
            agent, task_cfg, _ = load_agent(checkpoint, device=_device(args.device))
            env = BreadboardReachEnv(task_cfg, split="eval")
            try:
                result = evaluate_policy(agent, env, episodes=args.episodes, seed=args.seed + index * 10_000)
            finally:
                env.close()
            all_records.extend(result["records"])
            per_seed.append({"seed": checkpoint_seeds[index], "checkpoint": str(checkpoint), **result})
        fields = ("success_rate", "collision_rate", "position_error_mean_m", "angular_error_mean_deg", "arrival_step_mean")
        aggregate = {}
        for field in fields:
            values = [entry[field] for entry in per_seed if entry[field] is not None]
            aggregate[field] = float(np.mean(values)) if values else None
            aggregate[field + "_std"] = float(np.std(values)) if values else None
        report = {"episodes_per_seed": args.episodes, "seeds": per_seed, "aggregate": aggregate}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _plot_holes(all_records, args.output.parent / "per_hole.png")
        print(f"success={aggregate['success_rate']:.3f} collision={aggregate['collision_rate']:.3f} output={args.output}")
    elif args.command == "visualize":
        from .visualize import visualize_checkpoint

        summary = visualize_checkpoint(
            args.checkpoint,
            output_dir=args.output_dir,
            mode=args.mode,
            hole_id=args.hole_id,
            episodes=args.episodes,
            seed=args.seed,
            device=_device(args.device),
        )
        print(f"success={summary['success_rate']:.3f} output={args.output_dir}")
    else:
        from .env import BreadboardReachEnv
        from .registration import register_env

        register_env()
        env = BreadboardReachEnv(TaskConfig(), split="train")
        rng = np.random.default_rng(args.seed)
        try:
            sampled = set()
            for index in range(max(len(env.available_holes) * 2, 10)):
                observation, info = env.reset(seed=args.seed + index)
                assert env.observation_space.contains(observation)
                sampled.add(info["hole_id"])
                for _ in range(args.steps):
                    action = rng.uniform(-1, 1, size=7).astype(np.float32)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    assert env.observation_space.contains(observation)
                    assert np.isfinite(reward)
                    if terminated or truncated:
                        break
            print(f"checked {len(sampled)} sampled holes of {len(env.available_holes)} training holes")
            if not args.skip_ik:
                from .diagnostics import solve_hole_ik

                results = [
                    solve_hole_ik(env, hole.hole_id, seed=args.seed)
                    for hole in env.holes
                ]
                failed = [result for result in results if not result["reachable"]]
                print(f"IK reachable: {len(results) - len(failed)}/{len(results)}")
                for result in failed:
                    print(f"  unreachable: {result['hole_id']} ({result['position_error_m']:.3f} m)")
                if failed:
                    raise SystemExit(1)
        finally:
            env.close()


if __name__ == "__main__":
    main()
