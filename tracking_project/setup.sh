#!/bin/bash
# ============================================
# 小车追踪 PPO 训练 —— 一键安装 & 快速测试
# ============================================
set -e

echo "============================================"
echo "  小车追踪 PPO 训练环境安装"
echo "============================================"
echo ""
echo "管线: 检测结果(3维) → PPO策略 → 速度指令(3维)"
echo ""

echo "=== 安装 Python 依赖 ==="
pip install gymnasium numpy stable-baselines3 torch matplotlib

echo ""
echo "=== 验证依赖是否装好 ==="
python3 -c "
import gymnasium; print(f'gymnasium:          {gymnasium.__version__}')
import numpy as np; print(f'numpy:              {np.__version__}')
import torch;        print(f'torch:              {torch.__version__}')
from stable_baselines3 import PPO; print('stable-baselines3:   OK')
print('✅ 所有依赖就绪')
"

echo ""
echo "=== 快速测试：随机动作跑 100 步 ==="
python3 -c "
import numpy as np
from car_tracking_env import CarTrackingEnv

env = CarTrackingEnv(render_mode='human')
obs, _ = env.reset()

print(f'观测(3维): {obs}  ← 这就是检测模型会输出的格式')
print(f'动作(3维): (vx, vy, ω) 归一化到 [-1,1]')
print()

total_reward = 0
for step in range(100):
    action = np.random.uniform(-1, 1, size=3)  # 3维：vx, vy, ω
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()

    if terminated:
        print(f'>>> Step {step}: 随机居然追上了!')
        break
    if truncated:
        print(f'>>> Step {step}: 超时')
        break

print(f'总奖励: {total_reward:.1f}')
env.close()
print('✅ 环境测试通过!')
"

echo ""
echo "============================================"
echo "  环境就绪，开始训练:"
echo ""
echo "    python train.py"
echo ""
echo "  训练时另开终端看 TensorBoard:"
echo "    tensorboard --logdir tensorboard_logs/"
echo ""
echo "  训练完评估:"
echo "    python evaluate.py --render matplotlib"
echo ""
echo "  导出 ONNX 部署到小车:"
echo "    python export_onnx.py"
echo "============================================"
