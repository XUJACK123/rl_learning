# 面包板随机孔位到点：MuJoCo + RLinf + DDPG/HER

这个目录包含一个可以独立启动的 RLinf 扩展实验。机器人使用 MuJoCo Menagerie 的 Franka Panda；训练循环由 `breadboard_reach.runner` 实现，完整回合先经过 HER，再以 RLinf 的 `Trajectory` 格式写入 `TrajectoryReplayBuffer`。评估和视频从同一训练检查点读取 actor。

## 任务定义

| 项目 | 第一版设置 |
| --- | --- |
| 机器人 | Panda 7 个关节，夹爪保持张开 |
| 面包板 | 固定，长 160 mm、宽 80 mm、厚 10 mm 的几何原型 |
| 候选孔位 | 基础孔距 2.54 mm，每 8 个孔取一个目标，约 20.32 mm 间隔 |
| 目标 | 所选孔中心上方 30 mm，工具轴竖直向下 |
| 动作 | 7 维 `[-1, 1]` 关节增量，每维最多 0.02 rad，控制周期 50 ms |
| 成功 | 位置误差 ≤5 mm、朝向误差 ≤10°，连续 5 个控制步，无桌面或面包板碰撞 |
| 回合 | 最长 100 个控制步；到达后继续采样，碰撞立即终止 |
| 奖励 | 未达到 −1，达到 0；碰撞额外 −5 |

每次 `reset` 从训练孔位随机抽样一个目标。20% 的孔位由固定种子预留用于评估。`reset(options={"hole_id": "R+0C+0"})` 可以指定孔位。观测是 `observation`（28 维机器人状态）、`achieved_goal`（末端位置与四元数）及 `desired_goal`（目标位置与四元数）。`compute_reward` 支持批量重算，使 HER 不需要重新运行物理仿真。

可调用 `breadboard_reach.registration.register_env()` 注册 Gymnasium ID `BreadboardReach-v0`，之后用 `gymnasium.make("BreadboardReach-v0")` 创建环境。当前 RLinf 集成使用其 `Trajectory` 和 `TrajectoryReplayBuffer`；单机采样、评估与训练调度在本项目的 `runner.py` 中，尚未接入 RLinf 的分布式 Env Worker / Cluster 流水线。

面包板的几何尺寸与位置在 `TaskConfig` 中定义。当前只模拟到板上方，没有插入、抓取、电路接触或相机识别。真实机械臂型号确定后，应替换模型、关节控制映射与板坐标标定。

## 安装

建议 Python 3.12。在本目录运行：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[rlinf,test]'
.venv/bin/mujoco-menagerie prefetch franka_emika_panda
```

`rlinf` 可选依赖固定在提交 `e0531f40d1d07a70bc17709b1f497767d2d1aa8d`；Panda 模型包固定为 `mujoco-menagerie==2026.9.0`。Menagerie 第一次取模型资产需要网络，取到之后使用本地缓存。[RLinf 源码](https://github.com/RLinf/RLinf)、[Menagerie 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)。

若 RLinf 已按其官方说明安装在当前 Python 环境，也可以只安装本项目的基础依赖：`uv pip install --python .venv/bin/python -e '.[test]'`。训练时仍会检查 RLinf 的数据结构与回放缓冲区是否可导入。

无显示器录制视频时，根据可用的 OpenGL 后端设置 `MUJOCO_GL=egl` 或 `MUJOCO_GL=osmesa`。交互窗口需要图形桌面与 Matplotlib 图形后端。

## 先检查环境，再训练

```bash
.venv/bin/breadboard-reach check-env
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/breadboard-reach train --seeds 0 1 2 --steps 1000000
```

`check-env` 检查观测、动作、目标抽样，并用 MuJoCo Jacobian 尝试对全部孔位求逆运动学。若某些孔位不可达，命令会列出孔位并以非零状态退出。小规模控制链路冒烟测试可运行：

```bash
.venv/bin/breadboard-reach train --seeds 0 --steps 500 --warmup 100 --eval-interval 500 --checkpoint-interval 500
```

冒烟测试的步数不足以证明策略收敛。正式训练默认使用两层 256 单元 MLP、`γ=0.98`、`τ=0.005`、批量 256、预热 1 万步、每步一次更新及 4 个未来目标。结果位于 `outputs/runs/seed_N/`，包括 `metrics.jsonl`、RLinf 回放数据、周期检查点和 `latest_checkpoint.txt`。检查点包含 actor、critic、目标网络、优化器、随机状态、配置及 RLinf 回放快照。可以使用单个种子和对应检查点继续运行：

```bash
.venv/bin/breadboard-reach train --seeds 0 --steps 1500000 --resume outputs/runs/seed_0/checkpoints/step_001000000
```

继续运行时必须使用与检查点相同的训练配置，只有总步数可以增大；若调整其他参数，应作为新实验运行。

## 评估和可视化

假设训练检查点为 `outputs/runs/seed_0/checkpoints/step_001000000`：

```bash
.venv/bin/breadboard-reach evaluate --checkpoint outputs/runs/seed_0/checkpoints/step_001000000 --episodes 200
.venv/bin/breadboard-reach report --checkpoints outputs/runs/seed_0/checkpoints/step_001000000 outputs/runs/seed_1/checkpoints/step_001000000 outputs/runs/seed_2/checkpoints/step_001000000 --episodes 200
.venv/bin/breadboard-reach visualize --checkpoint outputs/runs/seed_0/checkpoints/step_001000000 --mode record --episodes 10
.venv/bin/breadboard-reach visualize --checkpoint outputs/runs/seed_0/checkpoints/step_001000000 --mode view --hole-id R+0C+0 --episodes 1
```

录制命令默认输出到 `outputs/visualization/`：每回合一个 MP4 和 CSV，同时生成 `summary.json`、`per_hole.png` 与 `training.png`。视频画面包含目标标记、末端轨迹、目标孔位、位置与朝向误差、连续到位次数、碰撞及成功状态。录制时会检查标记、轨迹和数值与导出位姿一致。`evaluate` 不渲染视频，`report` 将三个独立检查点各评估 200 回合，汇总成功率、末端误差、碰撞率和成功回合的平均到达步数。

## 数据流与关键约束

```text
随机孔位 → MuJoCo 观测 → DDPG actor → 关节目标 → 物理步进
           │                                  │
           └──── 完整回合 ←───────────────────┘
                      ↓
          同回合未来末端位姿重标目标
                      ↓
          RLinf TrajectoryReplayBuffer
                      ↓
                DDPG 更新与检查点
```

HER 仅改变目标与目标相关奖励；原动作、状态、碰撞和超时不变。价值目标在碰撞终止时停止自举，在 100 步超时截断时使用最后一个真实观测继续自举。RLinf 原生回放缓冲区按转移采样，因此本项目先收集完整回合再做未来目标重标。

第一阶段目标是三个随机种子在保留孔位上各评估 200 回合，报告成功率、位置与朝向误差、到达步数及碰撞率；成功率 80%、碰撞率低于 1% 是调参目标，不是尚未运行训练时的既成结果。

# 目标：
# RLinf 面包板随机目标到点仿真

## 目标

在目前为空的 `robot_arm` 目录中建立 MuJoCo 仿真，并正式接入 RLinf。每回合随机指定固定面包板上的一个孔位，机械臂用 **DDPG + HER** 学习到达孔位上方并保持朝向。真实机械臂型号未定，第一版采用 [MuJoCo Menagerie 的 Franka Panda 模型](https://github.com/google-deepmind/mujoco_menagerie)；最终交付包含可观看的运行过程和训练曲线。

## 仿真与任务接口

- 建立 Panda、桌面和固定面包板场景。先从间隔约 20 mm 的孔位集合抽样；目标为孔位上方 30 mm、末端竖直向下的位姿。
- 策略每 50 ms 输出 7 维关节增量，由限位内的位置伺服执行。观测包含关节状态、末端位姿、上一步动作和目标位姿。
- 环境提供 `observation`、`achieved_goal`、`desired_goal` 及可批量重算的 `compute_reward`，遵循[目标条件环境接口](https://robotics.farama.org/v1.2.2/content/multi-goal_api/)。
- 初始成功条件为位置误差 ≤5 mm、朝向误差 ≤10°，连续保持 5 个控制步且没有碰撞。回合最多 100 步；到达后继续采样至回合结束，以保留 HER 所需的未来状态。

## RLinf 训练流程

- 固定 RLinf 源码版本，增加环境注册、动作适配、DDPG 学习器和训练配置，由 RLinf 管理采样、评估与检查点。[环境扩展入口见官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/extending/new_env.html)。
- 按环境编号收集完整回合；HER 对每个转移从同一回合未来状态抽取 4 个已达到位姿，替换目标并重算奖励，然后将原始及重标转移写入 RLinf 回放缓冲区。完整回合收集必须放在缓冲区之前，因为[该缓冲区按转移采样，不负责按 episode 切分](https://rlinf.readthedocs.io/en/latest/rst_source/reference/api/replay_buffer.html)。
- DDPG 使用目标条件 actor、critic 和目标网络。起始配置：两层 256 单元 MLP、`γ=0.98`、`τ=0.005`、批量 256、预热 1 万步、每采样一步更新一次。保存网络、优化器、回放数据、随机状态及配置。

## 可视化与验证

- 提供一条加载 RLinf 检查点的评估命令，可指定孔位，也可按固定种子依次展示保留孔位。MuJoCo 画面显示机械臂、面包板、**目标标记、末端轨迹**；画面文字显示当前孔位、位置误差、朝向误差和成功状态。
- 同一命令支持交互式观看与无界面录制；无界面模式将每回合导出为 MP4，并保存逐步位姿、动作、误差和碰撞记录。另生成训练成功率与误差曲线，以及各目标孔位的评估结果图。默认输出放在 `robot_arm/outputs/visualization/`。
- 验证目标变换、可达性、关节限位、碰撞和 HER 的同回合未来目标抽样；检查录制画面中的目标、轨迹、误差与导出的数据一致。用 3 个种子训练，每个种子在保留孔位评估 200 回合，报告成功率、误差、碰撞率和到达步数；第一阶段调参目标为保留孔位成功率 ≥80%、碰撞率 <1%。

**默认边界：**第一版使用仿真状态，不接相机或真实机械臂。确定真实型号后，再替换机器人模型、控制映射和面包板位姿标定；仿真可视化也可作为后续真机结果的对照。
