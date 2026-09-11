# PPO:
- PPO需要同时维护四个模型，每个模型都扮演着不同的角色
- 两个静态锚点和两个动态模型之间的协同迭代

## Actor Network
- 策略模型本身，会在训练的途中更新参数
- 初始数据来自于SFT(有监督微调)，需要一个SFT模型来避免强化学习难以收敛
- 克隆一个reference model(完全冻结不更新的模型)
- 最后部署的模型就是actor

## Critic Network
- 价值模型，用于评估当前的输出处于什么水平，给出未来的累计收益，会在训练的途中更新参数
- 初始数据来源于SFT/RM模型的拓展，但是将最后的LM Head(预测每个词的概率的层)去除，换成全新的Value Head，由于在SFT阶段根本不存在，所以它的参数权重无法从SFT继承，只能用随机数填充
- 为什么能够直接套用大部分的SFT？对于critic而言，也需要拥有像actor一样的理解能力
- 由于一开始输出的价值预测V(s)是随机乱猜的数值，在训练的初期需要快速收敛

## Reference Model
- 给AI策略加上一个安全绳，actor和ref模型会有相同的输入并且产生动作的概率，但是ref模型是会冻结参数，只进行前向传播，没有计算梯度，因此输出会有差值
- ref model相当于是那个一直没有受到奖励模型/奖励影响的模型，保持着原始的分布
- 如果actor和ref模型的差值较大，KL惩罚变回生效，使得actor不会改变太多
- 经过一段时间的训练后，ref中的权重又会更新的和actor一样

## Reward Model
- 由偏好数据(Pairwise data)预先训练，不会在训练的过程中更新参数(偏好数据是指会给出一个好和坏的回答/动作让它知道相对的标尺)
- 对于actor模型最后的结果进行整体的打分，输出奖励值

### Reward Model和Critic Network的区别
- Reward Model：代表着真实奖励/人类的偏好，负责对Actor生成的完整回答给与最终的客观打分，
- Critic Network：对于每一个生成的步骤进行打分，帮actor搞清楚这句话好不好，给出的是当前步的Value(包括未来的累计回报奖励)
- reward model就相当于是真实分数，给予的是这一步的真实奖励，而critic network就是这一步的value

## 算法的流程
1. 预训练：
- 输入大量的无标记数据来进行下一个token/动作的预测(获得跨形态的通用物理认知，训练的数据很杂包括可能有不同型号的机器人/不同的任务)
- 训练出base model

2. SFT有监督微调
- 输入的数据成为对话来进行进一步的训练/进行模仿学习(获得专属于这一个机器人/这个动作的动作)
- 使得模型获得专属的能力

3. 奖励模型训练(仅仅是PPO必需要)
- 训练出一个能够打出标量分数的模型/直接就是奖励的环境
- 在LLM中就是人类输入偏好数据

4. 核心训练
- 轨道采样(rollout)：使用旧策略网络与环境交互，搜集T个时间步的数据
    - 在整条(或整批)轨迹已经完全收集完毕后，拿已经发生过的历史数据倒推算出来的
    - 单步时序差分误差: $$\sigma_t = r_t + \gamma * V(s_{t+1}) - V(s_t)$$
    - 多步(广义)时序差分误差: $$A_t = \sigma_t + (\gamma \lambda) \sigma_{t+1} + (\gamma \lambda)^2 \sigma_{t+2} + \dots$$
- 随机打乱数据, 将收集到的数据D进行随机打乱, 准备进行小批量的随机梯度下降, 主要目标是为了减少更新的方差
- 策略模型更新: $$L^{\text{clip}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$
    - $A_t$ 为通过 GAE 算出的 $A_t$
    - 对于好动作（$A_t > 0$）：min 会在 $r_t$ 变大时选中截断项，防止模型太贪婪、更新幅度过大（限制上限）
    - 对于坏动作（$A_t < 0$）：如果模型误将坏动作的概率改得极高，min 会选中未截断项，解除截断限制，给予极大的梯度惩罚，强制模型迅速纠错（不设下限）
- 价值模型更新

## 为什么还需要RL(PPO之类的)
- SFT只能够教模型什么是正确的，却很难教模型什么是错误的
- 突破人类的标记上限

# 注意：
- 旧版本里 Actor 和 Critic 各有一条独立的 45→256→256 躯干(两个塔)，但是新的是两个头看同一组特征，节省参数，同时actor学习到的特征能够直接帮助critic
## 高斯策略和ε-greedy：
- 一般来说的ε-greedy是用概率来决定是探索(随机探索)还是选择旧有的步骤
- 高斯策略是在动作的基础上再加上一个小随机
- 高斯策略是在连续的数，例如在0-10之中要输出一个数字的类型，以μ为中心，σ定范围的采样
- σ也是一种可学习参数，也是一种慢慢衰减的ε，如果梯度发现探索不够，损失还大就会调大σ，如果找到门道的关节，梯度把它压小，动作变得精准
- $$\pi_\theta(a \mid s) = \frac{1}{\sqrt{2\pi\sigma_\theta^2(s)}} \exp\left( -\frac{(a - \mu_\theta(s))^2}{2\sigma_\theta^2(s)} \right)$$
    - 每一个动作输出的概率为多少
    - 正态分布的公式：$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$
    - 偏差平方项 $(a - \mu_\theta(s))^2$:衡量实际执行的动作 $a$ 距离网络预测中心 $\mu_\theta(s)$ 有多远

## 熵价值
- 前面两种都是强制对动作变化，而后面调用熵价值是更改每一个动作概率
- 靠反向传播（Backpropagation）与梯度更新
- 香农熵的公式为: $$\mathcal{S}[\pi_\theta](s_t) = -\sum_{i=1}^{K} p_i \ln(p_i)$$
    - 分布越倾斜，$\mathcal{S}$ 越小；分布越平均，$\mathcal{S}$ 越大
- 如果使用熵价值, PPO 的总 Loss 函数形式为：$$\text{Loss}_{\text{total}} = -\mathcal{L}^{\text{clip}} + c_1 \mathcal{L}^{VF} - c_2 \cdot \mathcal{S}[\pi_\theta]$$ 
    - 要让整体 Loss 越来越小，在数学上就必须让被减数 $c_2 \cdot \mathcal{S}$ 尽可能大
    - 损失函数中的系数 $c_2$ 通常被设得非常小, 正确动作带来的巨大收益降幅，远远大于失去熵奖励所带来的一点点 Loss 增加。在优化器的眼里，宁可牺牲这点熵，也要把正确动作的概率推上去
- $$\mathcal{S}[\pi_\theta](s_t) = -\sum_{a_t} \pi_\theta(a_t|s_t) \log \pi_\theta(a_t|s_t)$$

## 共享主干和完全独立独立成两个模型
- 共享主干: 如经典 RL、机器人控制、视觉游戏
    - 优点：节省显存，计算速度快。主干网能学习到既有利于“理解环境（估值）”又有利于“做出动作（决策）”的通用特征
    - 同一个共享网络能同时处理两种更新，靠“梯度的矢量叠加”
    - $$\boldsymbol{g}_{\text{shared}} = \boldsymbol{g}_{\text{policy}} + c_1 \cdot \boldsymbol{g}_{\text{value}}$$
- 完全独立独立成两个模型: 如 LLM RLHF，例如 DeepSeek/ChatGPT 训练
    - 大模型的文本生成和数值打分属于完全不同的任务维度。如果强行共享主干，价值训练的大梯度很容易产生“梯度干扰（Gradient Interference）”，直接破坏策略模型的语言表达与逻辑能力
    - 策略网络 $\theta$：只用公式 (5) 的策略损失 $$L^{\text{clip}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$ 去做梯度更新
    - 价值网络 $\phi$：只用: $$\mathcal{L}^{\text{VF}}(\phi) = \frac{1}{2} \mathbb{E} \left[ \left( V_\phi(s_t) - G_t \right)^2 \right]$$的均方误差损失 $L^{VF}(\phi)$ 去做梯度更新