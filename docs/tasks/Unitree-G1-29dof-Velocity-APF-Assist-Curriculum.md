# Unitree-G1-29dof-Velocity APF 外力辅助课程设计

> 文档说明：本文定义“首次出现 APF 速度后，对机身施加 APF 方向外力”的课程机制，用于缓解障碍接触后的停滞问题。本文只覆盖课程逻辑、公式与参数，不包含完整代码实现细节。

**核心链路**：`碰撞点出现 -> APF 速度有效 -> 机身施加 APF 方向辅助力 -> 速度跟踪达标(0.8) -> 辅助系数按1%逐步衰减`

适用范围：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py`（外力施加事件）
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`（课程等级更新）
- `config/config.yaml`（课程参数配置）

## 一、输入/输出

| 类型 | 符号 | 说明 |
|---|---|---|
| 输入 | $m_i$ | 第 $i$ 个碰撞点有效标记（来自碰撞缓存槽位） |
| 输入 | $v_{\text{apf}}^{xy}$ | APF 输出平面速度（base 系） |
| 输入 | $v^{xy}$ | 机器人当前平面速度（base 系） |
| 输入 | $R_{\text{track}}$ | 本回合线速度跟踪指标（建议使用 `track_lin_vel_xy` 的 episode 平均值） |
| 状态 | $\alpha$ | 辅助系数，连续取值区间 $[\alpha_{\text{min}}, 1]$ |
| 输出 | $F_{\text{assist}}^{xy}$ | 施加在机身上的辅助外力（world 或 base 后再转换） |

## 二、机制/算法

### 2.1 触发条件

定义 hazard/APF 激活：

$$
h = \mathbf{1}\left(\sum_i m_i > 0\right)\cdot \mathbf{1}\left(\|v_{\text{apf}}^{xy}\| > \epsilon_v\right)
$$

当 $h=1$ 时，执行辅助力计算；当 $h=0$ 时，辅助力清零。

### 2.2 辅助力方向与大小

方向对齐 APF 速度：

$$
\hat d_{\text{apf}} = \frac{v_{\text{apf}}^{xy}}{\|v_{\text{apf}}^{xy}\|+\epsilon_v}
$$

根据你提出的规则，辅助强度采用：

$$
r_{\text{assist}} = \|v_{\text{apf}}^{xy}\| \cdot k_{\text{assist}} \cdot \alpha
$$

其中 $\alpha$ 是课程衰减系数，初始值通常取 $1.0$，并在达标时按固定步长衰减。

$$
\alpha \leftarrow \max\left(\alpha_{\text{min}}, \alpha - \epsilon\right),\qquad \epsilon=0.01
$$

> 注：等价于“每次达标减少 1% 的辅助力度”。

将该强度映射为外力模长，并与方向组合：

$$
\|F_{\text{assist}}^{xy}\| = \operatorname{clip}(r_{\text{assist}}, 0, F_{\text{max}})
$$

$$
F_{\text{assist}}^{xy} = \|F_{\text{assist}}^{xy}\|\cdot \hat d_{\text{apf}}
$$

### 2.3 课程系数更新（复用速度课程逻辑，阈值 0.8）

更新逻辑与现有 `lin_vel_cmd_levels` 保持一致：在 episode 边界，用 `track_lin_vel_xy` 的回合均值与该项权重做比较。

$$
R_{\text{track}}
=
\frac{\operatorname{mean}\left(\text{episode\_sum}(\texttt{track\_lin\_vel\_xy})\right)}{T_{\text{episode}}}
$$

$$
\text{upgrade}=\mathbf{1}\left(R_{\text{track}} \ge w_{\text{track}}\cdot 0.8\right)
$$

在此基础上更新课程系数：

$$
\alpha \leftarrow
\begin{cases}
\max(\alpha_{\text{min}}, \alpha - 0.01), & \text{if } \text{upgrade}=1 \\
\alpha, & \text{otherwise}
\end{cases}
$$

其中 $w_{\text{track}}$ 为 `track_lin_vel_xy` 的 reward 权重（当前配置中为 `1.0`）。

## 三、参数表

| 参数 | 默认值建议 | 含义 | 调大效果 | 调小效果 |
|---|---:|---|---|---|
| `assist_force.enable` | `true` | 是否启用 APF 外力辅助课程 | 启用课程 | 关闭课程 |
| `assist_force.eps_speed_mps` | `1e-3` | APF 速度有效阈值 $\epsilon_v$ | 更严格触发 | 更容易触发 |
| `assist_force.k_assist` | `100.0` | 强度增益 $k_{\text{assist}}$ | 推动力更强 | 推动力更弱 |
| `assist_force.force_max_n` | `100.0` | 外力上限 $F_{\text{max}}$（N） | 更快脱困但更易扰动 | 更稳但可能推不动 |
| `assist_force.alpha_init` | `1.0` | 初始辅助系数 $\alpha_0$ | 初期帮助更大 | 初期帮助更小 |
| `assist_force.alpha_min` | `0.0` | 最小辅助系数 $\alpha_{\text{min}}$ | 保留更多外力 | 更接近纯自主 |
| `assist_force.decay_step` | `0.01` | 达标时单次衰减步长 $\epsilon$ | 退辅更快 | 退辅更慢 |
| `assist_force.upgrade_threshold` | `0.8` | 升级门槛 | 更难升级 | 更易升级 |

## 四、运行流程与约束

1. 每个控制步读取碰撞点有效标记与 $v_{\text{apf}}^{xy}$。  
2. 若 `hazard && APF速度有效`，计算 $F_{\text{assist}}^{xy}$ 并作用到机身（推荐 torso）。  
3. 若不满足触发条件，外力置零。  
4. 每个 episode 结束时，按 `track_lin_vel_xy` 与 `weight*0.8` 的同口径判定更新课程系数。  
5. 达标时 `alpha` 每次减 `0.01`；不达标时 `alpha` 停滞不变；到达 `alpha_min` 后保持不变。  

硬约束：

- 外力仅作用平面方向（XY），不直接注入 Z 力和额外扭矩。  
- 外力必须限幅到 `force_max_n`，避免训练不稳定。  
- 课程系数更新频率固定为 episode 边界，避免步级抖动。  

## 五、Reward/训练关联说明

本设计中 `r_assist` 主要作为“外力强度信号”，不要求直接并入总 reward。  
若后续希望并入 reward，可使用同式：

$$
r_{\text{total}} \leftarrow r_{\text{total}} + w_{\text{assist}}\cdot r_{\text{assist}}
$$

建议先仅作为外力课程启用，验证行为改善后再决定是否加入 reward，以避免目标函数耦合过重。

## 六、可视化调试清单

建议在训练日志中至少记录：

1. `hazard_active_ratio`：每步触发比例。  
2. `apf_speed_norm` 与 `assist_force_norm` 的均值/分位数。  
3. `assist_alpha` 的时间曲线。  
4. `track_lin_vel_xy` 的 episode 均值与是否达标。  
5. 站桩指标（如 `|v_xy| < threshold` 比例）是否下降。  

期望现象：

- 首次碰撞后更快恢复运动，`assist_force_norm` 在 hazard 区间内有响应。  
- 随 `alpha` 逐步下降（辅助力变小），机器人在障碍接触后的停滞时长仍保持较低。  

异常信号：

- 力长期饱和：`k_assist` 过大或 `force_max_n` 过小。  
- 明显冲撞/抖动：`k_assist` 过大，或未做平面限幅。  
- 几乎无效果：`k_assist` 过低，或 `alpha_init` 过低。  

## 参数速查表

| 模块 | 参数 | 建议值 |
|---|---|---:|
| 外力触发 | `eps_speed_mps` | `1e-3` |
| 外力强度 | `k_assist` | `100.0` |
| 外力强度 | `force_max_n` | `100.0` |
| 课程系数 | `alpha_init` | `1.0` |
| 课程系数 | `alpha_min` | `0.0` |
| 课程系数 | `decay_step` | `0.01` |
| 升级规则 | `upgrade_threshold` | `0.8` |

