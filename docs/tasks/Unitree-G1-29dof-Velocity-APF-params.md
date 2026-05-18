# Unitree-G1-29dof-Velocity APF 参数说明

> 文档说明：本文只解释当前实现中 APF 速度修正与碰撞点缓存参数的意义、公式和调参方向；不覆盖 reward 设计与网络结构。

**核心链路**：`Obstacle 接触 -> 碰撞点缓存 -> APF 排斥速度 -> 叠加 base_velocity -> 平面限幅输出`

适用代码：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py`

## 一、输入/输出

| 类型 | 符号 | 说明 |
|---|---|---|
| 输入 | $v_{\text{cmd},b}^{xy}, \omega_z$ | 速度采样器输出（base 系），APF 在其上做增量修正 |
| 输入 | $p, o_i$ | 机器人平面位置与第 $i$ 个碰撞点平面位置（world 系） |
| 输入 | $m_i$ | 碰撞点有效标记（来自缓存槽位） |
| 中间量 | $\Delta v_w, \Delta v_b$ | APF 排斥速度（world 系 / base 系） |
| 输出 | $v_{\text{out},b}^{xy}, \omega_z$ | APF 修正后的命令；仅改 $(vx, vy)$，$wz$ 保持采样值 |

## 二、机制/算法

### 2.1 核心公式

相对向量、距离、单位方向：

$$
r_i = p - o_i,\qquad
d_i = \|r_i\|,\qquad
\hat r_i = \frac{r_i}{d_i + \epsilon}
$$

距离权重窗：

$$
w(d_i)=\left[\max\left(1-\frac{\max(d_i-\rho,0)}{R},0\right)\right]^{\beta}
$$

世界系排斥速度（含有效标记 $m_i$）：

$$
\Delta v_w^{(\text{cur})}=\sum_i k \cdot m_i \cdot w(d_i)\cdot \hat r_i
$$

EMA 平滑：

$$
\Delta v_w \leftarrow \alpha \Delta v_w + (1-\alpha)\Delta v_w^{(\text{cur})}
$$

旋转到 base 系后叠加，并做平面模长限幅：

$$
v_{\text{raw},b}^{xy}=v_{\text{cmd},b}^{xy}+\Delta v_b
$$

$$
v_{\text{out},b}^{xy}
=v_{\text{raw},b}^{xy}
\cdot
\min\left(1,\frac{v_{\text{max}}}{\|v_{\text{raw},b}^{xy}\|+\epsilon}\right)
$$

其中 $v_{\text{max}}=\texttt{lin\_vel\_xy\_max}$。

### 2.2 符号定义

| 符号 | 含义 | 单位 |
|---|---|---|
| $p$ | 机器人平面位置（world） | m |
| $o_i$ | 第 $i$ 个碰撞点平面位置（world） | m |
| $r_i$ | 机器人到碰撞点的相对向量 | m |
| $d_i$ | 相对距离 | m |
| $\hat r_i$ | 排斥方向单位向量 | - |
| $m_i$ | 碰撞点有效标记（0/1） | - |
| $\Delta v_w$ | 世界系排斥速度 | m/s |
| $\Delta v_b$ | base 系排斥速度 | m/s |
| $v_{\text{cmd},b}^{xy}$ | 原始采样平面速度命令 | m/s |
| $v_{\text{out},b}^{xy}$ | APF 修正后平面速度命令 | m/s |

> 注意：当前实现只修正 $(vx, vy)$，不修正 $wz$。

## 三、参数表

### 3.1 APF 主参数（`ApfCfg`）

| 参数 | 默认值 | 范围建议 | 含义 | 调大效果 | 调小效果 |
|---|---:|---|---|---|---|
| `R` | `1.0` | `0.6~1.5` | 势场作用半径 | 更早感知、更远开始绕障 | 仅近距离起效 |
| `rho` | `0.2` | `0.1~0.4` | 近区偏移，近距离保持高权重 | 近障碍强排斥区更宽 | 更快进入衰减区 |
| `beta` | `2.0` | `1.0~3.0` | 距离衰减指数 | 边缘衰减更陡，更“硬” | 衰减更平滑，更“软” |
| `k` | `1.0` | `0.5~2.0` | 排斥增益 | 避障更积极 | 避障更保守 |
| `alpha` | `0.5` | `0.2~0.8` | EMA 平滑系数 | 更稳但更滞后 | 更灵敏但更易抖 |
| `lin_vel_xy_max` | `1.0` | `0.6~1.5` | 合成后平面速度上限 | 机动性更强 | 更稳更保守 |

### 3.2 APF 输入质量参数（碰撞缓存事件）

| 参数 | 默认值 | 含义 | 调参影响 |
|---|---:|---|---|
| `force_threshold` | `0.3` | 力阈值过滤接触噪声 | 过低会抖，过高会漏检 |
| `merge_distance_m` | `0.02` | 新旧点合并阈值 | 过小冗余点多，过大过度合并 |
| `keep_radius_m` | `1.0` | 历史点保留半径 | 过小记忆短，过大引入旧点干扰 |
| `max_points` | `10` | 每 env 最大缓存槽位 $N_{\text{max}}$ | 过小信息不足，过大开销上升 |

## 四、流程与约束

1. **碰撞采集**：`obstacle_contact_forces` 仅过滤 `Obstacle`，并记录 `contact_pos_w`。  
2. **缓存维护**：按 `keep_radius_m` 清理历史点；新点按“合并/写空槽/淘汰最远槽”更新。  
3. **APF 合成**：用缓存点计算 $\Delta v_w^{(\text{cur})}$，再做 EMA。  
4. **坐标转换**：仅用 yaw 将 $\Delta v_w$ 转到 base 系得到 $\Delta v_b$。  
5. **命令回写**：`v_cmd + Δv` 后做平面模长限幅并回写 `vel_command_b`。  

硬约束：

- APF 只修改 `vx, vy`，`wz` 不变。  
- 最终平面速度满足 $\|v_{\text{out},b}^{xy}\| \le \texttt{lin\_vel\_xy\_max}$。  
- 若无有效碰撞点，APF 增量自动趋近 0（仅保留 EMA 余量）。  

## 五、(可选) 训练目标关联说明

当前迁移阶段 APF 是“命令层修正”，主要影响策略观测中的 `velocity_commands`。  
若后续接入 APF 专项 reward，建议单独定义“意图 + 公式 + 调参”文档，避免与本参数文档耦合。

## 六、可视化调试清单

建议每次调参至少检查以下量（每项都可打印均值/分位数）：

1. `v_cmd_xy`、`delta_v_b`、`v_out_xy`。  
2. `collision_points_count`（每 env 有效点数）。  
3. `force_magnitude` 统计是否大量低于 `force_threshold`。  
4. 速度限幅触发率（`||v_raw|| > lin_vel_xy_max` 的比例）。  

期望现象：

- 障碍接触后 `delta_v_b` 有明显响应；离开障碍后平滑衰减。  
- `v_out_xy` 方向偏离障碍点方向，且不长期饱和。  

常见异常信号：

- **长时间抖动**：`alpha` 太小或 `force_threshold` 太低。  
- **几乎无避障**：`k` 太小、`R` 太小或 `force_threshold` 太高。  
- **过度绕行**：`k`/`rho` 偏大或 `lin_vel_xy_max` 偏高。  

## 参数速查表

| 模块 | 参数 | 默认值 |
|---|---|---:|
| APF | `R` | `1.0` |
| APF | `rho` | `0.2` |
| APF | `beta` | `2.0` |
| APF | `k` | `1.0` |
| APF | `alpha` | `0.5` |
| APF | `lin_vel_xy_max` | `1.0` |
| 碰撞缓存 | `force_threshold` | `0.3` |
| 碰撞缓存 | `merge_distance_m` | `0.02` |
| 碰撞缓存 | `keep_radius_m` | `1.0` |
| 碰撞缓存 | `max_points` | `10` |

