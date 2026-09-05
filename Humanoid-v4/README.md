# Language-to-Rewards 最小复现（MJPC Humanoid）

复现论文 *Language to Rewards for Robotic Skill Synthesis* 的核心闭环：
自然语言指令 → 奖励函数代码 → MJPC 实时优化 → 人形机器人动作。

第一版只做两个技能：**站立（stand）** 和 **向左朝向（turn left）**。

## 系统构成与分工

1. **MJPC Humanoid 任务**（本仓库代码负责）：使用 MJPC humanoid 任务的
   dm_control 修改版模型（`assets/humanoid_modified.xml`），保留 stand 任务的
   balance/高度/速度残差，新增 pitch、roll、heading（朝向）残差。
2. **奖励翻译层**（AI agent 负责）：论文里的 Motion Descriptor + Reward Coder
   两步 LLM，这里由你交给一个 AI agent（如 OpenClaw + DeepSeek）完成。
   本 README 的 API 部分就等价于论文附录 A.5 的 Reward Coder prompt。
3. **人在回路**：你在 viewer 里看渲染结果，不满意就用语言让 agent 改代码。

LLM 不需要以任何方式接入控制代码。论文里 LLM 每个回合只在开头被调用一次，
生成 reward 代码后 MJPC 以实时频率独立规划，控制回路里没有 LLM。

## 交互工作流（每回合）

1. 对 agent 说指令，例如「让机器人站立并转向左边，面朝北」。
2. agent 按照下方 API 生成一个 reward 脚本（只允许调用列出的函数）。
3. 把脚本内容放进 `reward_script.py`（或直接让 agent 编辑该文件）。
4. 运行 `python main.py`，在 viewer 里观察效果。
5. 不满意就带着上下文让 agent 改 `reward_script.py`，重复步骤 4。

## Reward API（AI agent 唯一允许调用的接口）

```python
def reset_reward()
    清空所有奖励项。开始一个新任务时调用。

def set_torso_targets(target_torso_height,
                      target_torso_pitch,
                      target_torso_roll,
                      target_torso_heading)
    target_torso_height: 躯干中心目标高度（米）。初始站立位姿下躯干中心约
                         1.28 米，脚面约 0.027 米。
    target_torso_pitch:  目标俯仰角（弧度）。0 表示躯干竖直。
    target_torso_roll:   目标横滚角（弧度）。0 表示不向左右倾斜。
    target_torso_heading: 目标朝向（弧度），范围 [0, 2*pi)。
                          0 表示东，pi/2 表示北，pi 表示西，3*pi/2 表示南。

def set_feet_pos_parameters(feet_name, lift_height)
    feet_name: "left_foot" 或 "right_foot"。
    lift_height: 目标抬脚高度（米）。0 表示脚贴地、维持接触。

def execute_plan(plan_duration=3)
    把参数交给 MJPC 并执行 plan_duration 秒，默认 3 秒。
```

## 规则

1. 只输出一个 Python 代码块；块外可以有简短说明，但代码必须可直接执行。
2. 只能调用上面列出的四个函数；不发明函数或类；不写文件；不修改主程序。
3. 只允许 import numpy；用到了 `np` 就必须写 `import numpy as np`。
4. heading 用弧度，牢记 0 东 / pi/2 北 / pi 西 / 3*pi/2 南。
5. 不确定的数值用最佳估计，不要使用 None。
6. 必须调用 `execute_plan`。
7. 站立时双脚 `lift_height` 为 0（贴地），不要抬脚。

## 示例

站立：

```python
import numpy as np
reset_reward()
set_torso_targets(1.28, 0.0, 0.0, 0.0)
set_feet_pos_parameters("left_foot", 0.0)
set_feet_pos_parameters("right_foot", 0.0)
execute_plan(5)
```

站立并面朝北（向左转 90 度）：

```python
import numpy as np
reset_reward()
set_torso_targets(1.28, 0.0, 0.0, np.deg2rad(90))
set_feet_pos_parameters("left_foot", 0.0)
set_feet_pos_parameters("right_foot", 0.0)
execute_plan(5)
```

## 默认约定（可改）

- 初始位姿：站立、面朝东（heading = 0）。
- 朝向采用**绝对方位**（左转 = heading 加 90 度）。如需「相对任意初始朝向
  左转」的语义，把 API 换成 yaw 增量即可，第一版不做。
- 默认时长：站立 5 秒，转身 3 秒。
- 实测几何数值：初始位姿躯干中心 1.28m、脚面 0.027m；模型时间步 0.002s。
- 控制器为纯 NumPy 实现的 iLQG（`action_iLQG.py`），无 MJPC 编译依赖；
  默认参数下约 12 倍慢动作（0.8 秒仿真 ≈ 10 秒真实时间），只求跑通、未做调优。

## 运行方式（目标 Linux 机器）

- 依赖：`mujoco`、`numpy`；导出视频还需要 `pillow` 和 `ffmpeg`。
  `pip install mujoco numpy pillow`
- 有显示器：`python main.py` 直接开 MuJoCo viewer。
- 无头运行：`python main.py --no-view`（打印最终姿态）。
- 留档：`python main.py --save-video out.mp4` 把轨迹渲染成视频。
- 调参：`--horizon --iters --replan --max-steps`。

## 当前状态 / TODO

- [x] MJPC humanoid 任务：复用 stand 残差 + 新增 heading 残差（处理 ±180 度环绕）。
- [x] `reward_script.py` 的执行环境：四个 stub 函数收集参数并写进任务。
- [x] `main.py`：加载脚本 → 运行 planner → viewer 渲染。
- [x] `env.py` / `action_iLQG.py`：环境封装与 iLQG planner。
- [ ] （可选）平衡稳定性与转向速度的权重/正则化调优。
- [ ] （可选）换成 MJPC Python 绑定或 mjx 以获得实时速度。
