"""离屏录制：加载训练好的模型，把机器人行为渲染成 mp4 视频。

用法:
    python3 record_video.py                        # 默认输出 demo.mp4
    python3 record_video.py --out demo.mp4 --steps 300
"""
import argparse
import subprocess
import tempfile
import os

import numpy as np
import torch
import mujoco

from env import CustomHumanoidWalkingEnv
from PPO import ActorCritic, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="demo.mp4")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.checkpoint or os.path.join(save_dir, "checkpoints", "ppo_humanoid_final.pth")

    agent = ActorCritic().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"加载模型: {ckpt_path}")

    env = CustomHumanoidWalkingEnv(render_mode=None)
    renderer = mujoco.Renderer(env.model, height=480, width=640)

    frames_dir = tempfile.mkdtemp(prefix="humanoid_frames_")
    obs, _ = env.reset()
    done = False
    steps = 0

    while steps < args.steps and not done:
        with torch.no_grad():
            ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist, _ = agent.forward(ot)
            action = dist.mean[0].cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

        renderer.update_scene(env.data, camera="track")
        pix = renderer.render()
        frame_path = os.path.join(frames_dir, f"frame_{steps:05d}.png")
        _write_png(pix, frame_path)

    env.close()
    print(f"渲染了 {steps} 帧（{('摔倒' if done and not truncated else '正常') if done else '未结束'}）")

    out_abs = os.path.join(save_dir, args.out)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "30",
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_abs,
    ], check=True)
    print(f"视频已保存: {out_abs}")


def _write_png(rgb, path):
    from PIL import Image
    Image.fromarray(rgb).save(path, format="PNG")


if __name__ == "__main__":
    main()
