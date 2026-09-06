"""Run the Language-to-Rewards loop: script -> reward params -> iLQG -> viewer.

Usage:
    python main.py                      # run reward_script.py with live viewer
    python main.py --no-view            # headless, print final state
    python main.py --save-video out.mp4 # also export the rollout as video
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import mujoco.viewer  # 显式加载 viewer 子模块（mujoco 2.x 需要）
import numpy as np

from action_iLQG import iLQG
from env import HumanoidEnv, StandParams, SteppingParams


def _make_namespace() -> tuple[dict, StandParams, list[float]]:
    """Build the restricted namespace in which reward_script.py executes."""
    params = StandParams()
    duration = [3.0]

    def reset_reward() -> None:
        params.__init__()

    def set_torso_targets(
        target_torso_height: float | None = None,
        target_torso_pitch: float | None = None,
        target_torso_roll: float | None = None,
        target_torso_heading: float | None = None,
        target_torso_velocity_xy: tuple[float, float] | None = None,
        target_turning_speed: float | None = None,
    ) -> None:
        if target_torso_height is not None:
            params.target_height = float(target_torso_height)
        if target_torso_pitch is not None:
            params.target_pitch = float(target_torso_pitch)
        if target_torso_roll is not None:
            params.target_roll = float(target_torso_roll)
        if target_torso_heading is not None:
            params.target_heading = float(target_torso_heading)
        if target_torso_velocity_xy is not None:
            params.target_velocity_xy = (
                float(target_torso_velocity_xy[0]),
                float(target_torso_velocity_xy[1]),
            )
        if target_turning_speed is not None:
            params.target_turning_speed = float(target_turning_speed)

    def set_feet_pos_parameters(
        feet_name: str, lift_height: float | None = None
    ) -> None:
        if lift_height is None:
            params.feet_lift.pop(feet_name, None)
        else:
            params.feet_lift[feet_name] = float(lift_height)

    def set_feet_stepping_parameters(
        feet_name: str,
        stepping_frequency: float = 0.0,
        air_ratio: float = 0.0,
        phase_offset: float = 0.0,
        swing_up_down: float = 0.0,
        swing_forward_back: float = 0.0,
        should_activate: bool = True,
    ) -> None:
        params.feet_stepping[feet_name] = SteppingParams(
            frequency=float(stepping_frequency),
            air_ratio=float(air_ratio),
            phase_offset=float(phase_offset),
            swing_up_down=float(swing_up_down),
            swing_forward_back=float(swing_forward_back),
            active=bool(should_activate),
        )

    def execute_plan(plan_duration: float = 3.0) -> None:
        duration[0] = float(plan_duration)

    return {
        "reset_reward": reset_reward,
        "set_torso_targets": set_torso_targets,
        "set_feet_pos_parameters": set_feet_pos_parameters,
        "set_feet_stepping_parameters": set_feet_stepping_parameters,
        "execute_plan": execute_plan,
        "np": np,
        "numpy": np,
    }, params, duration


def run_reward_script(script: Path) -> tuple[StandParams, float]:
    namespace, params, duration = _make_namespace()
    code = script.read_text(encoding="utf-8")
    exec(compile(code, str(script), "exec"), namespace, namespace)
    return params, duration[0]


def _report(env: HumanoidEnv) -> None:
    roll, pitch, yaw = env.torso_orientation()
    torso_z = env.d.xpos[env.body_torso][2]
    print(
        f"[final] torso_z={torso_z:.3f}m "
        f"heading={math.degrees(yaw) % 360:.1f}deg "
        f"pitch={math.degrees(pitch):.1f}deg roll={math.degrees(roll):.1f}deg"
    )


def main() -> None:
    # 默认 reward_script.py 的绝对路径（相对 main.py 所在目录），
    # 这样无论从哪个目录运行 main.py 都能找到脚本。
    _script_default = Path(__file__).resolve().parent / "reward_script.py"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", type=Path, default=_script_default)
    ap.add_argument("--horizon", type=int, default=20, help="iLQG horizon (steps)")
    ap.add_argument("--iters", type=int, default=4, help="iLQG iterations per plan")
    ap.add_argument("--replan", type=int, default=10,
                    help="replan every N sim steps")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-view", action="store_true", help="run headless")
    ap.add_argument("--save-video", type=Path, default=None,
                    help="save rollout as mp4 (needs ffmpeg) or PNG frames")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    params, duration = run_reward_script(args.script)
    print(
        f"[reward] height={params.target_height:.3f} "
        f"pitch={math.degrees(params.target_pitch):.1f}deg "
        f"roll={math.degrees(params.target_roll):.1f}deg "
        f"heading={math.degrees(params.target_heading) % 360:.1f}deg "
        f"vel_xy={params.target_velocity_xy} "
        f"turning={params.target_turning_speed:.3f}rad/s "
        f"feet={params.feet_lift} plan={duration}s"
    )

    env = HumanoidEnv()
    x = env.reset()
    planner = iLQG(env, params, horizon=args.horizon, max_iters=args.iters)

    viewer = None
    if not args.no_view:
        viewer = mujoco.viewer.launch_passive(env.m, env.d)

    renderer = None
    frames_dir = None
    if args.save_video:
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("[video] Pillow is required for video export (pip install pillow)")
            args.save_video = None
        if args.save_video is not None:
            try:
                renderer = mujoco.Renderer(env.m, args.width, args.height)
                frames_dir = Path(tempfile.mkdtemp(prefix="l2r_frames_"))
            except Exception as exc:  # no GL context available
                print(f"[video] offscreen rendering unavailable: {exc}")
                renderer = None

    U = None
    u_idx = 0
    frame_no = 0
    sim_t = 0.0
    step = 0
    max_steps = min(args.max_steps, int(duration / env.dt))

    try:
        while step < max_steps:
            if step % args.replan == 0:
                U = planner.plan(x, U, shift=u_idx)
                u_idx = 0
                print(f"[t={sim_t:.2f}s] cost={planner.last_cost:.3f}")

            env.step(U[u_idx])
            u_idx += 1
            x = env.get_state()
            sim_t += env.dt
            step += 1

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break

            if renderer is not None and step % 5 == 0:
                renderer.update_scene(env.d, camera="side")
                pix = renderer.render()
                (frames_dir / f"frame_{frame_no:06d}.png").write_bytes(
                    _encode_png(pix)
                )
                frame_no += 1

    finally:
        if viewer is not None:
            viewer.close()

    _report(env)

    if renderer is not None and frames_dir is not None:
        _finish_video(frames_dir, args.save_video, fps=40)


def _encode_png(pix: np.ndarray) -> bytes:
    """Encode an RGB (H, W, 3) uint8 array as PNG using PIL if available."""
    from PIL import Image

    import io

    buf = io.BytesIO()
    Image.fromarray(pix).save(buf, format="PNG")
    return buf.getvalue()


def _finish_video(frames_dir: Path, out: Path, fps: int) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=False,
        )
        print(f"[video] saved {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for frame in frames_dir.glob("frame_*.png"):
            frame.rename(out / frame.name)
        print(f"[video] ffmpeg not found, PNG frames saved to {out}/")
    shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
