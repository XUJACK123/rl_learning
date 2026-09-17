"""Checkpoint evaluation, annotated MuJoCo video and training plots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .env import BreadboardReachEnv
from .geometry import pose_error
from .runner import load_agent


def annotate_frame(frame: np.ndarray, *, hole_id: str, step: int, info: dict) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 64), fill=(15, 20, 28))
    line1 = f"Target {hole_id}  Step {step:03d}  {'SUCCESS' if info['success'] else 'tracking'}"
    line2 = (
        f"Position {1000 * info['position_error_m']:.1f} mm  "
        f"Angle {info['angular_error_deg']:.1f} deg  "
        f"Hold {info['hold_count']}  Collision {int(info['collision'])}"
    )
    draw.text((12, 9), line1, fill=(255, 255, 255))
    draw.text((12, 34), line2, fill=(225, 235, 245))
    return np.asarray(image)


def _check_record_consistency(
    env: BreadboardReachEnv, observation: dict[str, np.ndarray], info: dict
) -> None:
    """Fail recording if visible markers or HUD numbers diverge from CSV data."""
    achieved = observation["achieved_goal"]
    desired = observation["desired_goal"]
    trace_site = env._trace_site_ids[env._trace_count - 1]
    position_error, angular_error = pose_error(achieved, desired)
    if not np.allclose(env.model.site_pos[env._goal_site_id], desired[:3], atol=1e-6):
        raise RuntimeError("Goal marker disagrees with exported target pose")
    if not np.allclose(env.data.site_xpos[trace_site], achieved[:3], atol=1e-6):
        raise RuntimeError("Trajectory marker disagrees with exported TCP pose")
    if not np.isclose(position_error, info["position_error_m"], atol=1e-7):
        raise RuntimeError("Position error disagrees with exported pose")
    if not np.isclose(np.rad2deg(angular_error), info["angular_error_deg"], atol=1e-5):
        raise RuntimeError("Angular error disagrees with exported pose")


def _plot_training(metrics_path: Path, output: Path) -> Path | None:
    if not metrics_path.exists():
        return None
    import matplotlib.pyplot as plt

    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    episodes = [event for event in events if event.get("type") == "episode"]
    evaluations = [event for event in events if event.get("type") == "eval"]
    if not episodes and not evaluations:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if episodes:
        window = min(50, len(episodes))
        values = np.asarray([float(e["success"]) for e in episodes])
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0].plot([e["step"] for e in episodes[window - 1 :]], smooth, label="train success")
        axes[1].plot(
            [e["step"] for e in episodes],
            [1000 * e["position_error_m"] for e in episodes],
            alpha=0.35,
            label="train endpoint error",
        )
    if evaluations:
        axes[0].plot(
            [e["step"] for e in evaluations],
            [e["success_rate"] for e in evaluations],
            marker="o",
            label="held-out success",
        )
        axes[1].plot(
            [e["step"] for e in evaluations],
            [1000 * e["position_error_mean_m"] for e in evaluations],
            marker="o",
            label="held-out endpoint error",
        )
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Position error (mm)")
    axes[1].set_xlabel("Environment steps")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _plot_holes(records: list[dict], output: Path) -> Path:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["hole_id"]].append(record)
    labels = sorted(grouped)
    rates = [np.mean([r["success"] for r in grouped[label]]) for label in labels]
    errors = [1000 * np.mean([r["position_error_m"] for r in grouped[label]]) for label in labels]
    figure, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.65), 7), sharex=True)
    axes[0].bar(labels, rates, color="#2a9d8f")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Success rate")
    axes[1].bar(labels, errors, color="#457b9d")
    axes[1].set_ylabel("Endpoint error (mm)")
    axes[1].tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def visualize_checkpoint(
    checkpoint: Path,
    *,
    output_dir: Path,
    mode: str = "record",
    hole_id: str | None = None,
    episodes: int = 10,
    seed: int = 42_000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Play or record deterministic episodes from the same RLinf checkpoint."""
    if mode not in ("record", "view"):
        raise ValueError("mode must be 'record' or 'view'")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent, task_cfg, _ = load_agent(checkpoint, device=device)
    env = BreadboardReachEnv(task_cfg, split="eval", render_mode="rgb_array")
    if hole_id is not None and hole_id not in env.holes_by_id:
        raise ValueError(f"Unknown hole {hole_id}")
    selected_holes = [hole_id] if hole_id else [h.hole_id for h in env.available_holes]
    plot_window = None
    image_artist = None
    if mode == "view":
        import matplotlib.pyplot as plt

        plt.ion()
        plot_window, axis = plt.subplots(figsize=(10, 7.5))
        axis.set_axis_off()
        image_artist = axis.imshow(np.zeros((env.render_height, env.render_width, 3), dtype=np.uint8))
        plot_window.tight_layout(pad=0)
        plt.show(block=False)

    records: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            selected = selected_holes[episode % len(selected_holes)]
            observation, _ = env.reset(seed=seed + episode, options={"hole_id": selected})
            filename = f"episode_{episode:03d}_{selected}"
            video = (
                imageio.get_writer(output_dir / f"{filename}.mp4", fps=20, codec="libx264")
                if mode == "record"
                else None
            )
            rows: list[dict[str, Any]] = []
            last_info: dict[str, Any] = {}
            first_success_step: int | None = None
            try:
                for _ in range(task_cfg.max_episode_steps):
                    action = agent.act(observation)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    _check_record_consistency(env, next_observation, info)
                    last_info = info
                    if info["success"] and first_success_step is None:
                        first_success_step = int(info["step"])
                    row = {
                        "step": info["step"],
                        "hole_id": selected,
                        "reward": reward,
                        "position_error_m": info["position_error_m"],
                        "angular_error_deg": info["angular_error_deg"],
                        "hold_count": info["hold_count"],
                        "success": info["success"],
                        "collision": info["collision"],
                        "shielded": info["shielded"],
                        "blocked": info["blocked"],
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    row.update({f"action_{i+1}": float(action[i]) for i in range(7)})
                    row.update({f"executed_action_{i+1}": float(info["executed_action"][i]) for i in range(7)})
                    row.update(
                        {f"tcp_{axis}": float(next_observation["achieved_goal"][i]) for i, axis in enumerate("xyz")}
                    )
                    row.update(
                        {f"target_{axis}": float(next_observation["desired_goal"][i]) for i, axis in enumerate("xyz")}
                    )
                    row.update(
                        {f"tcp_q{i}": float(next_observation["achieved_goal"][i + 3]) for i in range(4)}
                    )
                    row.update(
                        {f"target_q{i}": float(next_observation["desired_goal"][i + 3]) for i in range(4)}
                    )
                    rows.append(row)
                    frame = annotate_frame(env.render(), hole_id=selected, step=info["step"], info=info)
                    if video is not None:
                        video.append_data(frame)
                    else:
                        assert image_artist is not None and plot_window is not None
                        image_artist.set_data(frame)
                        plot_window.canvas.draw_idle()
                        plot_window.canvas.flush_events()
                    observation = next_observation
                    if terminated or truncated:
                        break
            finally:
                if video is not None:
                    video.close()
            with (output_dir / f"{filename}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records.append(
                {
                    "hole_id": selected,
                    "success": bool(last_info["success"]),
                    "collision": bool(last_info["collision"]),
                    "position_error_m": float(last_info["position_error_m"]),
                    "angular_error_deg": float(last_info["angular_error_deg"]),
                    "shielded_steps": sum(int(row["shielded"]) for row in rows),
                    "blocked_steps": sum(int(row["blocked"]) for row in rows),
                    "first_success_step": first_success_step,
                    "episode_csv": f"{filename}.csv",
                    "episode_video": f"{filename}.mp4" if mode == "record" else None,
                }
            )
    finally:
        env.close()
        if plot_window is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.close(plot_window)

    summary = {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_holes(records, output_dir / "per_hole.png")
    _plot_training(checkpoint.parent.parent / "metrics.jsonl", output_dir / "training.png")
    return summary
