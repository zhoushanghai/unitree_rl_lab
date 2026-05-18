# Proprioception: 接触后局部栅格建图（AMP + GRU）

> 文档说明：本文给出一个基础可落地方案，目标是在当前 AMP 任务中，利用机器人本体感知与接触信息，做“接触后局部占据栅格图”预测。本文只覆盖最小可用设计（网络结构、监督信号、训练损失、运行约束与调试），不展开大规模地图融合、多传感器融合和全局 SLAM。
<!-- 公式渲染约定：统一使用 $...$（行内）与 $$...$$（块级），避免 \(...\)、\[...\] 在部分 Markdown 渲染器中失效。 -->

**核心链路**：本体观测 + GRU 记忆 -> 共享特征 trunk -> 动作头与栅格头 -> PPO/AMP 主损失 + 栅格辅助损失

## 一、输入/输出

| 类型 | 符号 | 说明 |
|---|---|---|
| 输入 | $o_t$ | 当前时刻 policy 观测（角速度、重力投影、关节状态、目标相对位姿、上一步动作等） |
| 输入 | $h_{t-1}$ | GRU 隐状态，用于跨时刻记忆与里程计累积 |
| 输入（监督） | $\mathcal{P}_t = \{(x_i, y_i, v_i)\}$ | 环境中 APF 碰撞点槽（机体系），$v_i\in\{0,1\}$ 表示槽是否有效 |
| 输出1 | $a_t$ | 动作头输出（与现有 actor 动作接口一致） |
| 输出2 | $\mathbf{z}_t \in \mathbb{R}^{H\times W}$ | 栅格头输出 logits（每个格子一个 logit） |
| 推理图 | $\hat{\mathbf{M}}_t\in[0,1]^{H\times W}$ | 占据概率图，$\hat{\mathbf{M}}_t=\sigma(\mathbf{z}_t)$ |

## 二、机制/算法

### 2.1 共享 trunk + 双头

给定观测 $o_t$ 与上一时刻隐藏状态 $h_{t-1}$，先经共享主干得到中间特征：
$$
h_t = \text{GRU}(\phi(o_t), h_{t-1})
$$
其中：
- $\phi(\cdot)$：前端特征编码（MLP）
- $h_t$：共享时序特征，供两个 head 复用

动作头与栅格头分别为：
$$
a_t = f_{\text{act}}(h_t), \qquad \mathbf{z}_t = f_{\text{grid}}(h_t)
$$

### 2.2 栅格监督目标（接触后局部建图）

将 APF 有效碰撞点投影到局部栅格坐标系，得到二值监督图 $\mathbf{M}_t \in \{0,1\}^{H\times W}$。

建议局部地图（机体系，以机器人为中心）：
- 圆形地图半径：$r_{\text{map}} = 1.0$ m
- 栅格分辨率：$\Delta = 0.05$ m/cell
- 对应方形栅格范围：$x,y \in [-1.0, 1.0]$ m
- 对应栅格尺寸：$H=W=\frac{2r_{\text{map}}}{\Delta}=40$

单点栅格映射可写为：
$$
u_i = \left\lfloor \frac{x_i + r_{\text{map}}}{2r_{\text{map}}} \cdot W \right\rfloor,\quad
v_i = \left\lfloor \frac{y_i + r_{\text{map}}}{2r_{\text{map}}} \cdot H \right\rfloor
$$
点满足 $\sqrt{x_i^2+y_i^2}\le r_{\text{map}}$ 且索引有效时，置 $\mathbf{M}_t[v_i,u_i]=1$。
圆外区域（$x^2+y^2>r_{\text{map}}^2$）统一作为无效区域掩码，不参与有效占据监督。

### 2.3 训练目标

总损失：
$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{PPO}} + \mathcal{L}_{\text{AMP}} +
\lambda_{\text{grid}} \cdot \mathcal{L}_{\text{grid}}
$$

其中栅格损失用 BCE with logits：
$$
\mathcal{L}_{\text{grid}} =
\text{BCEWithLogits}\left(\mathbf{z}_t,\mathbf{M}_t\right)
$$

说明：
- $\mathcal{L}_{\text{PPO}}$ 与 $\mathcal{L}_{\text{AMP}}$ 保持原流程；
- 栅格头只作为辅助任务，不改变动作输出接口；
- $\lambda_{\text{grid}}$ 取小值，避免干扰 locomotion 主任务收敛。

## 三、参数表

| 参数 | 值/范围 | 说明 |
|---|---|---|
| $r_{\text{map}}$ | 1.0 m | 以机器人为中心的圆形局部地图半径 |
| $\Delta$ | 0.05 m/cell | 栅格分辨率 |
| $H, W$ | 40, 40 | 由 $2r_{\text{map}}/\Delta$ 得到的栅格尺寸 |
| $x_{\text{min}}, x_{\text{max}}$ | -1.0, 1.0 m | 机体系横向范围（方形包围盒） |
| $y_{\text{min}}, y_{\text{max}}$ | -1.0, 1.0 m | 机体系纵向范围（方形包围盒） |
| $\lambda_{\text{grid}}$ | 0.02 ~ 0.10 | 栅格辅助损失权重 |
| $\tau$（推理阈值） | 0.5（可调） | 将概率图二值化时的阈值 |
| GRU hidden dim | 256（当前） | actor 时序记忆容量 |
| rollout length | 24（当前） | 训练时序窗口长度（后续可加长） |

## 四、流程与约束

1. 环境步进后，更新 APF 碰撞点缓存，并保持槽位语义稳定（近到远排序）。
2. 由有效槽位生成当前时刻局部栅格监督 $\mathbf{M}_t$。
3. actor 前向：共享 trunk/GRU 得到 $h_t$，同时输出动作 $a_t$ 与栅格 logits $\mathbf{z}_t$。
4. PPO/AMP 按原流程更新；额外计算 $\mathcal{L}_{\text{grid}}$ 并加权到总损失。
5. 训练/评估中记录栅格指标（如 IoU、precision、recall），观察是否影响主任务回报。

**硬约束（必须显式保持）**：
- 动作通道接口不变：控制器仍只消费动作头输出；
- 角速度通道不因栅格任务被额外修改；
- 若当前无有效碰撞点，监督图可为空图，但需防止模型坍塌为“永远全零”；
- 栅格输出用于训练辅助，不直接替代 APF 控制逻辑。

## 五、Reward/训练目标（可选增强）

### 5.1 设计意图
- 主任务仍是稳定行走与导航；
- 栅格头用于提升“接触后环境结构记忆”，帮助策略形成更一致的避障行为。

### 5.2 训练建议
- **阶段1**：先保持原 PPO+AMP 稳定训练；
- **阶段2**：打开栅格辅助损失，小权重接入；
- **阶段3**：若图质量不足，优先增大时序窗口（例如 48/64）而不是先加深网络。

### 5.3 调参方向
- $\lambda_{\text{grid}}$ 增大：栅格更快收敛，但可能拉低 locomotion 回报；
- 分辨率 $H,W$ 增大：细节更好，但类别不平衡更明显、训练更难；
- 时序窗口增大：更利于“接触后累积建图”，但显存与训练时间上升。

## 六、可视化调试

建议每次实验至少检查以下对象：

1. **碰撞点与监督图一致性**
   - 可视化 APF 碰撞点（world）与投影后的局部栅格（base）。
   - 期望现象：有接触时，前方对应栅格出现占据热点。
   - 异常信号：碰撞点可见但监督图长期为空（通常是坐标系或索引越界错误）。

2. **预测图与监督图对齐**
   - 同步展示 $\mathbf{M}_t$（GT）与 $\hat{\mathbf{M}}_t$（Pred）。
   - 期望现象：接触后若干步内热点位置逐步对齐。
   - 异常信号：Pred 长期全零或全一（多为损失权重/类别不平衡问题）。

3. **主任务是否被干扰**
   - 观察 episode reward、跌倒率、速度跟踪误差、AMP style reward。
   - 期望现象：主任务指标基本稳定，栅格指标逐步提升。
   - 异常信号：一开栅格损失就明显退化（优先下调 $\lambda_{\text{grid}}$）。

## 参数速查表

| 模块 | 参数 | 值 |
|---|---|---|
| 栅格定义 | $r_{\text{map}}$ / $\Delta$ / $H\times W$ | $1.0$ m / $0.05$ m / $40\times40$ |
| 栅格窗口 | $x:[x_{\text{min}},x_{\text{max}}]$, $y:[y_{\text{min}},y_{\text{max}}]$ | $x:[-1,1]$ m, $y:[-1,1]$ m（圆外掩码） |
| 损失 | $\lambda_{\text{grid}}$ | 0.02 ~ 0.10 |
| 结构 | actor RNN | GRU, hidden=256 |
| 时序 | rollout length | 24（当前，可增至 48/64 试验） |
| APF 缓存 | $N_{\text{max}}$ | 10（当前配置） |

