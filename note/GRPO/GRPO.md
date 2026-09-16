# GRPO的算法：
- 与PPO算法相比较，缺少了critic model(value network)

## 计算损失公式：
- 优势公式：$$A_i = \frac{r_i - \text{mean}(r_1, r_2, \dots, r_G)}{\text{std}(r_1, r_2, \dots, r_G)}$$
    - 对当前 Prompt $q$ 生成的 $G$ 个回答评分 $r_1, \dots, r_G$ 求均值与标准差
    - $r_i$ 代表模型生成的第 $i$ 个回答所获得的奖励得分
    - 计算loss的时候会把1-G个答案全部取出来然后分别计算所有的loss
- 损失公式：$$\mathcal{L}^{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\vert{}o_i\vert{}} \sum_{t=1}^{\vert{}o_i\vert{}} \left\{ \min\left( \frac{\pi_\theta(a_t\vert{}s_t)}{\pi_{\text{old}}(a_t\vert{}s_t)} A_i, \text{clip}\left(\frac{\pi_\theta(a_t\vert{}s_t)}{\pi_{\text{old}}(a_t\vert{}s_t)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{KL}[\pi_\theta \vert{}\vert{} \pi_{\text{ref}}] \right\}$$
    - Clipping 限制项：继承自 PPO 的重要性采样截断机制，限制新老策略比率在 $[1-\epsilon, 1+\epsilon]$ 之间，防止单步策略更新幅度过大导致训练崩塌
    - KL散度惩罚项，限制当前策略模型不要偏离ref模型太远
    -  $\frac{\pi_\theta(a_t\vert{}s_t)}{\pi_{\text{old}}(a_t\vert{}s_t)}$ 代表损失函数的计算是显性的遍历了序列中的每一个位置t，对每一个token的预测概率变化单独求损失
- GRPO中的 $\frac{1}{G}$ 和 $\frac{1}{\vert{}o_i\vert{}}$
    - $\frac{1}{\vert{}o_i\vert{}}$(序列长度归一化)：乘以 $\frac{1}{\vert{}o_i\vert{}}$ 相当于对序列内的 Token 取平均，确保每条回答长度对梯度的贡献是平等的
    - $\frac{1}{G}$(组大小归一化)：GRPO 是以“组（Group）”为采样单位的，一个 Prompt 生成 $G$ 个候选序列。$\frac{1}{G} \sum_{i=1}^G$ 表示对这 $G$ 个候选序列的损失求算术平均
- 损失函数式针对token级别的，而优势函数是seq级别的
    - 只有一个标量的优势值A，而这个值会被广播给序列中的所有token->模型无法细粒的分析哪个token是对的或者是说错的