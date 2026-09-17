# 面包板随机孔位到点：MuJoCo + DDPG/HER + RLinf 回放

Franka Panda 在 MuJoCo 中从固定初始姿态出发，随机到达面包板孔位上方 30 mm，保持工具轴向下。训练循环使用本项目的 DDPG 与 HER；RLinf 提供 `Trajectory` 和 `TrajectoryReplayBuffer`。采样、评估和调度仍由本项目的 `runner.py` 执行。

## 任务与控制

- 21 个候选孔位按固定种子划分为 17 个训练孔位、4 个保留孔位。每个回合最多 100 个控制步，每步 50 ms。
- 动作是 7 维 `[-1, 1]` 关节目标增量，每维每步最多 0.02 rad。控制器会**保持上一次关节目标**；零动作不会让目标随着重力下垂的实际关节位置移动。
- 末端位置误差不超过 5 mm、朝向误差不超过 10°，连续 5 步且未碰撞即成功。碰撞立即终止。
- 每步先在独立的 MuJoCo 数据状态中预测桌面或面包板碰撞。若候选动作不安全，依次缩小动作并尝试反向退让；仍无安全动作时拒绝本步并记录 `blocked`。回放、观测和日志均记录实际执行的动作；安全干预会产生奖励惩罚。
- 奖励为 `-位置误差/0.1 m - 0.5×朝向误差/45° + 5×到位 - 10×碰撞 - 1×安全干预`。HER 仅从同回合**后续且位姿有变化**的状态重标目标，并使用同一奖励函数重算。

旧检查点的网络输入与奖励定义不同，不能用于这版续训。

## 安装与诊断

在本目录使用 Python 3.12：

```bash
uv sync --locked --extra rlinf --extra test
.venv/bin/mujoco-menagerie prefetch franka_emika_panda
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")'
.venv/bin/python -m pytest tests -q
.venv/bin/breadboard-reach diagnose-control --seeds 20
```

`diagnose-control` 输出 `outputs/diagnostics/control.json`，包含零动作下的关节与末端位移、控制误差、首次碰撞部件，以及每个孔位的 IK 与动态轨迹结果。零动作不稳定时，训练会自动在 Menagerie 初始姿态附近搜索无碰撞、可保持的重置姿态；找不到时停止并报告。孔位动态轨迹不可达时也停止训练。

无显示器录制视频时，按机器支持情况设置 `MUJOCO_GL=egl` 或 `MUJOCO_GL=osmesa`。WSL 中若当前命令沙盒无法访问 GPU，需要在允许 GPU 访问的终端运行训练命令。

## WSL 短训练

```bash
.venv/bin/breadboard-reach train --profile smoke --seeds 0 --device cuda
.venv/bin/breadboard-reach train --profile smoke --seeds 0 --steps 2500 --device cuda --resume latest
```

`smoke` 默认 2000 个环境步、每个训练孔位 1 条示范、200 次 actor 示范学习、1 轮策略状态下的 IK 纠偏学习、200 次 critic 预热、批量 64；每 1000 步评估并保存。产物在 `outputs/smoke/seed_0/`。其目的是检查示范、更新、评估、保存与恢复链路，不能据此判断收敛。若 PyTorch 无法识别 CUDA，可用 `--device cpu --steps 500` 做更短的流程验证。

## Linux GPU 主训练

在另一台机器使用相同的 `uv.lock`、Python 3.12 和上述诊断命令，然后运行：

```bash
.venv/bin/breadboard-reach train --profile full --seeds 0 1 2 --steps 400000 --device cuda
.venv/bin/breadboard-reach report --checkpoints \
  outputs/runs/seed_0/checkpoints/step_000400000 \
  outputs/runs/seed_1/checkpoints/step_000400000 \
  outputs/runs/seed_2/checkpoints/step_000400000 --episodes 200
```

`full` 对每个训练孔位收集 5 条成功轨迹，进行 5000 次 actor 示范学习、3 轮策略状态下的 IK 纠偏学习和 5000 次 critic 预热，之后执行带示范约束的 DDPG + HER。每 10000 步评估与保存。若 400000 步评估未达到目标，分别以 `--seeds N --steps 1000000 --resume latest` 续训对应种子；除总步数外配置必须与检查点一致。

三个种子分别在保留孔位评估 200 回合。调参目标为每个种子成功率至少 80%、碰撞率低于 1%。报告同时给出误差、到达步数、安全干预率、拒绝执行比例和各孔位结果。示范及纠偏样本只来自训练孔位，保留孔位只做可达性检查和评估。

`outputs/runs/seed_N/` 包含配置及包版本、`metrics.jsonl`、RLinf 回放数据、周期检查点和 `latest_checkpoint.txt`。运行目录已有内容时，新的训练会拒绝覆盖；继续训练须显式传 `--resume latest`。

## 评估与可视化

```bash
.venv/bin/breadboard-reach evaluate --checkpoint outputs/runs/seed_0/checkpoints/step_000400000 --episodes 200
.venv/bin/breadboard-reach visualize --checkpoint outputs/runs/seed_0/checkpoints/step_000400000 --mode record --episodes 10
.venv/bin/breadboard-reach visualize --checkpoint outputs/runs/seed_0/checkpoints/step_000400000 --mode view --hole-id R+0C+0 --episodes 1
```

录制输出每回合 MP4 与 CSV，并生成成功率和误差图。当前只模拟关节状态与几何板；未接相机、插入、电路接触或真实机械臂。
