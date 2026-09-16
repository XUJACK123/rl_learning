# DPO算法：
- 核心思想：从偏好数据学到奖励然后再使用RL优化策略，这两步直接变为一个对大模型权重直接求偏好分类损失

## 偏好与二分类
- 目的：希望能够找到一个值通过给出好和坏的判断的数据集(人类擅长这一点)来训练出一个能够发出绝对分值的奖励模型
- 在人类偏好的数据集中的x包括了人类偏好获胜的答案 $y_w$(win)和失败回答 $y_l$(lose)
- RLHF假设人类的偏好概率
- Bradley-Terry 模型(reward model，给win/loss打分的) $$P(y_w \succ y_l \vert{} x) = \sigma\Big(r(x, y_w) - r(x, y_l)\Big)$$
    - $y_w \succ y_l \mid x$：表示“在给定 $x$ 的条件下，$y_w$ 优于 $y_l$”这一偏好事件
    - $\sigma(z) = \frac{1}{1 + e^{-z}}$：Sigmoid 激活函数，用于将两者的得分差映射到 $(0, 1)$ 区间的概率值
    - 输入为$(x, y_w, y_l)$
    - $r(x, y)$：奖励模型（Reward Model）对 Prompt $x$ 与回答 $y$ 组合打出的隐式标量得分
    - 将相对的比较转化为分差概率，分为三种情况
        1. 若 $r(x, y_w) \gg r(x, y_l)$，得分差很大，概率 $\sigma(\Delta r)$ 趋近于 $1$
        2. 若 $r(x, y_w) \approx r(x, y_l)$，两者得分接近，概率趋近于 $0.5$（随机猜测）
        3. 若模型给出的分高低倒置（$r(x, y_w) < r(x, y_l)$），概率将低于 $0.5$
    - 高 Loss 会产生强烈的梯度，强制更新模型权重：要求模型下次见到类似 $y_w$ 的语义结构时把分数调高，见到 $y_l$ 的特征时把分数调低
    - 同时损失函数也会尽可能拉大好回答与坏回答的分差，如果模型给出的答案很类似，会让$\Delta r \approx 0$， 激活输出 $\sigma(0) = 0.5$。loss变大，如果给出的数字区别很大反而会让loss变小
- 最后的reward model的搭建：$$\mathcal{L}_{RM}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\Big(r_\psi(x, y_w) - r_\psi(x, y_l)\Big) \right]$$
    - $r_\psi(x, y_w)$ 与 $r_\psi(x, y_l)$：奖励模型分别对人类偏好回答（胜出者 $y_w$）与不偏好回答（失败者 $y_l$）打出的具体分数
    - $\log \sigma(\cdot)$：对数似然概率。因为概率在 $(0, 1)$ 之间，所以 $\log$ 值恒为负数
    - $-\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}$：前面的负号将“极大化正确预测概率”转化为“极小化 Loss”（即负对数似然损失 NLL），$\mathbb{E}$ 代表在整个偏好数据集 $\mathcal{D}$ 上求期望（在代码实现中就是对一个 Batch 求平均值）
    1. 如果模型给胜出回答打了高分、给失败回答打了低分，即 $r_\psi(x, y_w) \gg r_\psi(x, y_l)$，那么分差极大。此时 $\sigma(\Delta r) \approx 1$，$\log(1) = 0$，最终 $\text{Loss} \approx 0$。模型不需要进行大的梯度修正
    2. 如果模型把分打反了（$r_\psi(x, y_w) < r_\psi(x, y_l)$），$\sigma(\Delta r)$ 会小于 $0.5$，$\log$ 后的负值绝对值非常大。加上最外层的负号后，Loss 会陡增。梯度会强力推动模型抬高 $r_\psi(x, y_w)$ 的得分，同时压低 $r_\psi(x, y_l)$ 的得分