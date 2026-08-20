"""
训练脚本 - PPO 小车追踪（摄像头版）

管线：
  摄像头 → 目标检测模型 → 检测结果(3维) → PPO策略 → (vx, vy, ω)

观测维度: 3  (bbox中心xy + 深度z)
动作维度: 3  (前进速度, 横向速度, 旋转速度)
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from car_tracking_env import CarTrackingEnv


def main():
    log_dir = "./tensorboard_logs/"
    checkpoint_dir = "./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练环境（不渲染）
    env = CarTrackingEnv(render_mode=None)

    # 评估环境（单独实例，避免干扰训练统计）
    eval_env = CarTrackingEnv(render_mode=None)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,            # 熵系数：鼓励探索
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 定期保存
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=checkpoint_dir,
        name_prefix="ppo_car_tracking",
    )

    # 定期评估（每 10000 步跑 5 个回合看成功率）
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print("=" * 60)
    print("开始训练 PPO 小车追踪策略")
    print(f"  观测空间: {env.observation_space.shape}  (bbox中心xy + 深度z)")
    print(f"  动作空间: {env.action_space.shape}  (vx, vy, ω)")
    print(f"  总步数:   500,000")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 60)

    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save("final_car_tracking_policy")
    print("\n训练完成！模型已保存至 final_car_tracking_policy.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
