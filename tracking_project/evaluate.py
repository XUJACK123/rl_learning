"""
评估脚本：加载训练好的 PPO 模型，可视化小车追踪效果

用法:
    python evaluate.py                      # 控制台文字
    python evaluate.py --render matplotlib  # Matplotlib 动画
    python evaluate.py --episodes 10        # 跑 10 回合
"""
import argparse
import numpy as np
import time
from stable_baselines3 import PPO
from car_tracking_env import CarTrackingEnv


def run_console(env, model, episodes=3):
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}")
        print(f"{'='*60}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
            time.sleep(0.02)

        dist = np.sqrt((env.car_x - env.target_x)**2 + (env.car_y - env.target_y)**2)
        status = "🎯 追上了!" if dist < env.CATCH_RADIUS else "⏰ 超时"
        print(f"结果: {status} | 奖励: {total_reward:.1f} | 最终距离: {dist:.2f}m")


def run_matplotlib(env, model, episodes=3):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        traj_car, traj_target = [], []

        while not done:
            ax.clear()
            ax.set_xlim(0, env.ARENA_SIZE)
            ax.set_ylim(0, env.ARENA_SIZE)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # 画目标
            ax.add_patch(plt.Circle((env.target_x, env.target_y), 0.3, color="red", alpha=0.8))
            ax.text(env.target_x + 0.5, env.target_y + 0.5, "TARGET", color="red", fontsize=8)

            # 画小车（三角形 = 朝向）
            car_tri = patches.RegularPolygon(
                (env.car_x, env.car_y), numVertices=3, radius=0.4,
                orientation=env.car_theta, color="blue", alpha=0.8)
            ax.add_patch(car_tri)

            # 画小车视野锥
            half_fov = env.CAMERA_FOV_H_RAD / 2
            fov_left = patches.Wedge(
                (env.car_x, env.car_y), env.CAMERA_MAX_RANGE,
                np.degrees(env.car_theta - half_fov),
                np.degrees(env.car_theta + half_fov),
                alpha=0.08, color="cyan")
            ax.add_patch(fov_left)

            # 追踪圆圈
            ax.add_patch(plt.Circle(
                (env.target_x, env.target_y), env.CATCH_RADIUS,
                fill=False, edgecolor="green", linestyle="--", alpha=0.4))

            # 轨迹
            traj_car.append((env.car_x, env.car_y))
            traj_target.append((env.target_x, env.target_y))
            if len(traj_car) > 1:
                xs, ys = zip(*traj_car)
                ax.plot(xs, ys, "b-", alpha=0.3, linewidth=0.5)
                xs, ys = zip(*traj_target)
                ax.plot(xs, ys, "r-", alpha=0.3, linewidth=0.5)

            ax.set_title(f"Episode {ep+1} | Dist={np.sqrt((env.car_x-env.target_x)**2+(env.car_y-env.target_y)**2):.2f}m")

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            plt.pause(0.01)

        dist = np.sqrt((env.car_x - env.target_x)**2 + (env.car_y - env.target_y)**2)
        status = "CAUGHT!" if dist < env.CATCH_RADIUS else "TIMEOUT"
        print(f"Episode {ep+1}: {status} | Reward={total_reward:.1f} | Final Dist={dist:.2f}m")
        time.sleep(0.5)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_car_tracking_policy")
    parser.add_argument("--render", choices=["console", "matplotlib"], default="console")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading model: {args.model}.zip ...")
    model = PPO.load(args.model)
    env = CarTrackingEnv(render_mode="human")

    if args.render == "matplotlib":
        run_matplotlib(env, model, episodes=args.episodes)
    else:
        run_console(env, model, episodes=args.episodes)

    env.close()


if __name__ == "__main__":
    main()
