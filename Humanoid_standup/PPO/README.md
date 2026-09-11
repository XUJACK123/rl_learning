# PPO 并行训练 —— 使用说明

用强化学习（PPO）训练 MuJoCo Humanoid 学会向前走路。

## 一、文件结构

```
Humanoid_standup/
├── humanoid.xml              # MuJoCo 模型（gymnasium 官方同款）
├── PROGRESS.md               # 项目整体进度
└── PPO/
    ├── env.py                # 自定义环境（观测/动作/奖励）
    ├── PPO.py                # 训练主程序（并行 + GPU）
    ├── visualize.py          # 训练后可视化（单环境渲染）
    ├── PPO.md                # PPO 算法笔记
    └── checkpoints/          # 模型保存目录（自动创建）
```

## 二、怎么运行

```bash
cd ~/Desktop/rl_learning/Humanoid_standup/PPO
python3 PPO.py
```

训练完成后：

```bash
python3 visualize.py          # 加载 final 模型，弹窗口看机器人走路
```

## 三、并行是怎么实现的（重点）

### 3.1 核心思路

RL 训练最慢的地方是 **采样（rollout）**——让机器人一步步跑、收集经验。单个环境一次只能跑一条轨迹，CPU 大部分时间在空等。

并行化的思路：**同时开 16 个一模一样的环境，让 16 个机器人同时跑**，一次 `step` 就能拿到 16 份经验，采样速度接近 16 倍。

### 3.2 用到的工具：`SubprocVecEnv`

代码在 `PPO.py` 顶部：

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    """每个并行子进程用它新建一个环境（训练环境不渲染）。"""
    return CustomHumanoidWalkingEnv(render_mode=None)

env = SubprocVecEnv([make_env for _ in range(num_envs)])
```

`SubprocVecEnv` 是 stable-baselines3 提供的"多进程向量化环境"，它做的事：

1. **开 16 个子进程**（用 Python 的 `multiprocessing`）
2. 每个子进程里调用 `make_env()` 新建一个独立的环境实例
3. 主进程调 `env.step(actions)` 时，它把 16 个动作**分发**给 16 个子进程
4. 每个子进程各自跑一步物理仿真
5. 把 16 份结果**收集回来**，拼成一个大的批量返回

### 3.3 为什么用"子进程"而不是"线程"

因为 MuJoCo 的物理仿真是纯 CPU 计算，而 Python 的**线程**受 GIL（全局解释器锁）限制，无法真正并行跑 CPU 计算。

**子进程**每个都有独立的 Python 解释器，能真正利用你机器的 **24 个 CPU 核心**，物理仿真才能并行加速。

### 3.4 关键：环境工厂函数 `make_env`

```python
def make_env():
    return CustomHumanoidWalkingEnv(render_mode=None)
```

注意传给 `SubprocVecEnv` 的是**函数**（`make_env`），不是环境实例。因为：

- 环境里有 MuJoCo 的 `MjModel` / `MjData`，这些 C++ 对象**不能跨进程传输**
- 所以在**子进程内部**重新调用 `make_env()` 新建，而不是把主进程的环境传过去

### 3.5 训练环境不能渲染

渲染窗口（viewer）不能在子进程里打开，所以训练环境统一 `render_mode=None`。要看效果，用 `visualize.py` 在主进程单独开一个渲染环境。

## 四、GPU 是怎么用的

代码在 `PPO.py`：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ActorCritic().to(device)
```

- **采样**（rollout）：16 个子进程在 CPU 上并行跑物理仿真，产生的观测批量传给 GPU
- **网络前向**（选动作）：在 GPU 上做
- **更新**（PPO 的 10 个 epoch 梯度下降）：完全在 GPU 上

分工：**CPU 多核负责"跑物理"，GPU 负责"算网络"**，各司其职。

## 五、每一步的完整数据流

```
rollout 阶段（每个 iteration 开始时）：
  主进程
    │  观测 obs (16, 45) 放 GPU
    ▼
  网络 forward（GPU）→ 采样 16 个动作 (16, 17)
    │  动作转回 CPU numpy
    ▼
  SubprocVecEnv.step(actions)  ← 把 16 个动作分发出去
    ├─ 子进程 0：env.step(action[0])  → (obs, reward, done)
    ├─ 子进程 1：env.step(action[1])  → (obs, reward, done)
    ├─ ...（共 16 个）
    └─ 子进程 15：env.step(action[15])
    │  收集 16 份结果，拼成 (16, ...) 批量
    ▼
  存进经验列表，重复 2048 步
    ▼
  得到一整批数据 (2048, 16, ...) = 32768 个样本

更新阶段（rollout 之后）：
  把 (2048, 16, ...) 展平成 (32768, ...)
    │
  计算 GAE 优势（按每个环境独立沿时间反向递推）
    │
  重复 10 个 epoch：
    随机打乱 → 切 minibatch(64) → GPU 上做梯度更新
```

## 六、关键超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_envs` | 16 | 并行环境数 |
| `num_steps` | 2048 | 每个环境每轮跑的步数 |
| `num_iterations` | 122 | 迭代轮数 |
| `num_epochs` | 10 | 每批数据重复更新的次数 |
| `minibatch_size` | 64 | 每次梯度更新的小批量 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 的 λ |
| `epsilon` | 0.2 | PPO clip 范围 |
| `lr_rate` | 3e-4 | 学习率 |

**总训练步数** = `num_iterations × num_steps × num_envs` = `122 × 2048 × 16` ≈ **400 万步**。

## 七、训练时间估算

实测每轮（iteration）约 16 秒：

```
122 轮 × 16 秒 ≈ 33 分钟
```

（实际会随机器负载略有浮动。）

## 八、注意事项

1. **为什么脚会摔倒**：Humanoid 学会走路是 RL 的 hard 任务，需要百万级步数。400 万步只是起点，训练过程中 `ep_len`（每集存活步数）会逐步上升但可能反复震荡，属正常现象。
2. **模型保存**：每 5 轮存一次 `checkpoints/ppo_humanoid_iter_X.pth`，最后存 `ppo_humanoid_final.pth`。
3. **中断恢复**：目前代码不支持断点续训，中断后需从头开始（后续可加）。
4. **观测未归一化**：当前观测各维度量纲不同（四元数~1、高度~1.4），后续可考虑加归一化提升稳定性。
