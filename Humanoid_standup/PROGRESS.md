# 核心目标和要求
- 不要直接给我任何代码，而是一步一步引导我完成每一个算法和环境的编写
- 记录每一个算法的完成进度，以方便下一次唤醒的时候能载入
- 查看代码的时候包括检查注释，看看注释和代码本身有没有写对
- 完成的任务都是一样的(使用 gymnasium 的 Humanoid 让机器人向前行走)，但是希望通过不同的算法(例如PPO/DQN/DPO/GRPO算法来完成)
- 任务为使用gymnasium让人物移动

---

## 项目进度（下次唤醒从这里继续）

### 通用：环境（各算法共用）
- 目标：让 MuJoCo Humanoid 学会向前行走
- 基础：gymnasium 的 Humanoid 物理模型（库较新，自带 -v5 环境）；观测空间、动作空间、奖励函数全部自己编写
- 已确定的设计：
  - 观测：根姿态四元数(4) + 根线速度(3) + 根角速度(3) + 关节角(17) + 关节角速度(17) + 高度(1) ≈ 45 维（可选再加足底接触 2 维）
  - 动作：17 个关节力矩，Box(-0.4, 0.4)
  - 奖励：前进 1.0×vx + 存活 0.5 − 控制 0.1×Σa² − 摔倒 −5
- 状态：已完成（静态检查通过），待用户机器验收
  - humanoid.xml 已在项目根目录（gymnasium 官方同款；timestep=0.003、17 执行器、ctrlrange ±0.4、无 keyframe，reset 用 qpos0+噪声）
  - 环境代码在 PPO/env.py：45 维观测、±0.4 动作、5 子步仿真、摔倒 z<0.8 → terminated、1000 步 → truncated
  - 收尾待办：用户机器跑 check_env.py 验收；metadata 声称 render_modes 但未实现渲染（先改空）；get_obs_ 补 docstring；环境文件以后挪到根目录供四个算法共用

### PPO（Humanoid 向前行走）
- 训练范式：经典 PPO（Actor + Critic + GAE），不需要 RM / Reference / KL / SFT
- 进度：
  - [x] 第 1 步：任务与协作方式确认（引导式、不给现成代码、检查注释）
  - [x] 第 2 步：自定义环境（obs / action / reward）—— 静态检查通过，待用户机器验收
  - [x] 第 3 步：Actor + Critic 网络 —— 静态检查通过（forward 与 get_action_and_value 均正确）；遗留一处含糊注释 "# 获得新的" 待改
  - [x] 第 4 步：Rollout + GAE 优势估计
    - rollout 函数通过静态检查（PPO/PPO.py：七项返回 + next_value 逻辑正确）
    - reward/done append 已指定 dtype=float32（reward 行正确；done 行最终改为 torch.tensor(float(done), dtype=torch.float32)）
    - compute_gae 已由用户编写完成：边界用 next_value、lastgaelam 递推、dones[t] 截断、reverse、torch.stack().float() 均正确
    - 恒等式 returns - advantages = values 已用 3 步小数据手推验证（advantages=[1.1140,1.6157,1.6900]，returns=[1.1140,2.5157,3.4900]）
    - 遗留：用户机器上跑 3 步小测试打印三行；注释清理（# 获得新的 → 动作均值；GAE 注释简化）
  - [x] 第 5 步：PPO 更新（clip + entropy + critic loss）
    - update_ppo 初稿整体结构正确：clip、surr1/surr2、entropy bonus、critic MSE、loss 符号都对
    - ratio 已修复为 `torch.exp(new_log_probs - old_log_probs)`，静态看正确
    - 按用户要求先不单独做小数值验证，留到训练循环跑通后一起检查
  - [x] 第 6 步：训练循环与指标记录（静态完成，待用户机器冒烟运行）
    - 已形成“采样 → GAE → 多 epoch 小批量更新 → 记录/打印”的闭环
    - 待完成：
      - 用户机器跑 10~20 轮冒烟，确认无报错、loss 有限、无 NaN，episode 指标正常打印
      - 跑通后再补第 4/5 步的小数值验证
    - 已补充：PPO.py 增加 checkpoint 保存（每 5 轮 + 最终模型）；env.py 增加 render_mode 支持（human/rgb_array），并在每轮迭代末尾渲染一次
  - [ ] 第 7 步：评测与视频保存
  - [ ] 第 8 步：注释与代码一致性检查
- 会话记录：
  - 2026-09-09：立项；确定使用 Humanoid（-v5 库）并自定义 obs/action/reward；本地不装依赖，代码在此编写、用户环境运行
  - 2026-09-10：完成环境、ActorCritic 网络、rollout 函数；讲解 GAE 概念与公式；完成第 5/6 步静态编写并修正训练循环中的关键错误
- 下次唤醒从这继续：① 用户机器跑 10~20 轮训练冒烟测试并反馈指标 ② 补第 4/5 步小数值验证 ③ 进入第 7 步评测与视频保存

### DQN
- 未开始（DQN.md 仅有开头笔记）

### DPO / GRPO
- 未开始
