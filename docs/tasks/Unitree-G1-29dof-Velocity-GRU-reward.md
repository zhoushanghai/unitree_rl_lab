# Unitree-G1-29dof-Velocity-GRU 任务 Reward 说明

> 本文档说明任务 `Unitree-G1-29dof-Velocity-GRU` 的 MDP reward 设计：各项含义、计算公式、权重与典型量级。  
> 不包含 PPO 超参、网络结构或观测细节；GRU 任务与 `Unitree-G1-29dof-Velocity` **共用同一套 `RewardsCfg`**，仅观测与 actor 不同。

**核心链路**：速度指令 `base_velocity` → 策略输出关节位置增量 → 物理仿真 → 各 reward 项加权求和 → PPO 优化

**配置来源**：

| 文件 | 说明 |
|------|------|
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py` | `RewardsCfg` 定义 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py` | 项目自定义项（`energy`, `feet_gait`, `foot_clearance_reward`） |
| Isaac Lab `isaaclab.envs.mdp.rewards` | 通用惩罚/跟踪项 |
| Isaac Lab Tasks `velocity/mdp/rewards.py` | `track_lin_vel_xy_yaw_frame_exp`, `feet_slide` |

---

## 一、环境与时间尺度

| 参数 | 值 | 说明 |
|------|-----|------|
| `sim.dt` | 0.005 s | 物理步长 |
| `decimation` | 4 | 控制 decimation |
| 控制步长 $\Delta t$ | **0.02 s** | $0.005 \times 4$ |
| `episode_length_s` | 20.0 s | 单 episode 时长 |
| 每 episode 步数 | **~1000** | $20 / 0.02$ |

**总 reward（每控制步）**：

$$
r_t = \sum_i w_i \, f_i(s_t, a_t)
$$

其中 $w_i$ 为 `RewardTermCfg.weight`，$f_i$ 为该项原始返回值（未乘 weight 前）。

TensorBoard 中 `Episode_Reward/<term_name>` 一般为 **$w_i \cdot f_i$** 的 episode 累计或均值（取决于 Isaac Lab 日志实现）。

---

## 二、速度指令（与跟踪 reward 相关）

命令项 `base_velocity`（`UniformLevelVelocityCommandCfg`）：

| 阶段 | `lin_vel_x` | `lin_vel_y` | `ang_vel_z` |
|------|-------------|-------------|-------------|
| 训练初值 `ranges` | [-0.1, 0.1] | [-0.1, 0.1] | [-0.1, 0.1] |
| 课程上限 `limit_ranges` | [-0.5, 1.0] | [-0.3, 0.3] | [-0.2, 0.2] |
| 重采样周期 | 10 s | — | — |

符号：$\mathbf{v}^{cmd} = (v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd})$。

---

## 三、Reward 总表

| # | 配置名 | weight $w$ | 方向 | 函数 | 一句话 |
|---|--------|------------|------|------|--------|
| 1 | `track_lin_vel_xy` | +1.0 | 奖励 | `track_lin_vel_xy_yaw_frame_exp` | yaw 系平面线速度跟踪 |
| 2 | `track_ang_vel_z` | +0.5 | 奖励 | `track_ang_vel_z_exp` | 偏航角速度跟踪 |
| 3 | `alive` | +0.15 | 奖励 | `is_alive` | 未终止存活奖励 |
| 4 | `base_linear_velocity` | -2.0 | 惩罚 | `lin_vel_z_l2` | 抑制竖直速度 |
| 5 | `base_angular_velocity` | -0.05 | 惩罚 | `ang_vel_xy_l2` | 抑制 roll/pitch 角速度 |
| 6 | `joint_vel` | -0.001 | 惩罚 | `joint_vel_l2` | 关节速度正则 |
| 7 | `joint_acc` | -2.5e-7 | 惩罚 | `joint_acc_l2` | 关节加速度正则 |
| 8 | `action_rate` | -0.05 | 惩罚 | `action_rate_l2` | 动作变化率 |
| 9 | `dof_pos_limits` | -5.0 | 惩罚 | `joint_pos_limits` | 软限位违反 |
| 10 | `energy` | -2e-5 | 惩罚 | `energy` | 近似关节能耗 |
| 11 | `joint_deviation_arms` | -0.1 | 惩罚 | `joint_deviation_l1` | 手臂偏离默认姿态 |
| 12 | `joint_deviation_waists` | -1.0 | 惩罚 | `joint_deviation_l1` | 腰部偏离默认姿态 |
| 13 | `joint_deviation_legs` | -1.0 | 惩罚 | `joint_deviation_l1` | 髋 roll/yaw 偏离默认 |
| 14 | `flat_orientation_l2` | -5.0 | 惩罚 | `flat_orientation_l2` | 躯干水平 |
| 15 | `base_height` | -10.0 | 惩罚 | `base_height_l2` | 根高度 0.78 m |
| 16 | `gait` | +0.5 | 奖励 | `feet_gait` | 周期步态相位对齐 |
| 17 | `feet_slide` | -0.2 | 惩罚 | `feet_slide` | 支撑相脚滑移 |
| 18 | `feet_clearance` | +1.0 | 奖励 | `foot_clearance_reward` | 摆动相抬脚 |
| 19 | `undesired_contacts` | -1.0 | 惩罚 | `undesired_contacts` | 非脚部位触地 |

---

## 四、各项公式与含义

### 4.1 任务跟踪（正向）

#### `track_lin_vel_xy`（$w=1.0$）

将机体线速度投影到 **yaw 对齐机体系** 后与指令比较：

$$
\mathbf{v}_{xy}^{yaw} = R_{yaw}^{-1}(\mathbf{q}) \, \mathbf{v}_{w,xy}
$$

$$
f = \exp\left(-\frac{\|\mathbf{v}^{cmd}_{xy} - \mathbf{v}_{xy}^{yaw}\|^2}{\sigma^2}\right), \quad \sigma = \sqrt{0.25} = 0.5
$$

- **含义**：平面速度跟踪，误差越小越接近 1。
- **典型量级**：$f \in [0, 1]$，理想跟踪时加权贡献 **≈ +1.0/step**。

#### `track_ang_vel_z`（$w=0.5$）

$$
f = \exp\left(-\frac{(\omega_z^{cmd} - \omega_z)^2}{\sigma^2}\right), \quad \sigma = 0.5
$$

- **含义**：偏航角速度跟踪（机体系 $\omega_z$）。
- **典型量级**：$f \in [0, 1]$，理想时加权贡献 **≈ +0.5/step**。

#### `alive`（$w=0.15$）

$$
f = \mathbb{1}[\text{episode 未终止}]
$$

- **含义**：鼓励存活到 timeout，避免过早摔倒。
- **典型量级**：存活时每步 **+0.15**。

**任务向合计（理想跟踪 + 存活）**：约 **+1.65 / step**。

---

### 4.2 机身姿态与高度（惩罚）

| 配置名 | $w$ | 公式 $f$ | 说明 |
|--------|-----|----------|------|
| `base_linear_velocity` | -2.0 | $v_z^2$ | 机体坐标系竖直线速度，抑制蹦跳/下沉 |
| `base_angular_velocity` | -0.05 | $\omega_x^2 + \omega_y^2$ | 抑制躯干 roll/pitch 角速度 |
| `flat_orientation_l2` | -5.0 | $\|g_{xy}^{proj}\|^2$ | 投影重力 xy 分量，保持水平 |
| `base_height` | -10.0 | $(z - 0.78)^2$ | 世界系根高度，目标 **0.78 m** |

---

### 4.3 关节与动作正则（惩罚）

| 配置名 | $w$ | 公式 $f$ | 作用范围 |
|--------|-----|----------|----------|
| `joint_vel` | -0.001 | $\sum_j \dot{q}_j^2$ | 全部关节 |
| `joint_acc` | **-2.5e-7** | $\sum_j \ddot{q}_j^2$ | 全部关节 |
| `action_rate` | -0.05 | $\sum_k (a_{t,k} - a_{t-1,k})^2$ | 动作向量 |
| `dof_pos_limits` | -5.0 | 超出软限位的违反量之和 | 全部关节 |
| `energy` | -2e-5 | $\sum_j \|\dot{q}_j\| \cdot \|\tau_j\|$ | 速度×力矩近似功率 |

---

### 4.4 关节偏离默认姿态（惩罚，分组）

$$
f = \sum_{j \in \mathcal{J}} |q_j - q_j^{default}|
$$

| 配置名 | $w$ | 关节集合 $\mathcal{J}$ |
|--------|-----|------------------------|
| `joint_deviation_arms` | -0.1 | `.*_shoulder_.*_joint`, `.*_elbow_joint`, `.*_wrist_.*` |
| `joint_deviation_waists` | -1.0 | `waist.*` |
| `joint_deviation_legs` | -1.0 | `.*_hip_roll_joint`, `.*_hip_yaw_joint` |

- **设计意图**：腰、髋 roll/yaw 约束更强；手臂权重小，允许摆臂平衡。

---

### 4.5 步态与足部（正 + 负）

#### `gait`（$w=0.5$）

参数：`period=0.8` s，`offset=[0.0, 0.5]`（左右脚相差半周期），`threshold=0.55`，接触体 `.*ankle_roll.*`。

- 全局相位：$\phi = (t \bmod T) / T$
- 每腿期望支撑相：$\phi_i < 0.55$；与实际接触一致则该腿 +1
- $f \in \{0, 1, 2\}$（两腿各 1 分）
- **门控**：仅当 $\|\mathbf{v}^{cmd}\| > 0.1$ 时生效

最大加权贡献：$0.5 \times 2 =$ **+1.0 / step**。

#### `feet_clearance`（$w=1.0$）

$$
f = \exp\left(-\frac{1}{\sigma_{clr}} \sum_{foot} (z_{foot} - z_{tgt})^2 \cdot \tanh(\alpha \|\mathbf{v}_{foot,xy}\|)\right)
$$

| 参数 | 值 |
|------|-----|
| $z_{tgt}$ | 0.1 m |
| $\sigma_{clr}$ | 0.05 |
| $\alpha$ (`tanh_mult`) | 2.0 |
| 足端 | `.*ankle_roll.*` |

- **含义**：脚抬离目标高度且水平速度较大（摆动相）时给高 reward。
- **典型量级**：$f \in (0, 1]$，理想摆动时加权 **≈ +1.0**。

#### `feet_slide`（$w=-0.2$）

$$
f = \sum_{foot} \|\mathbf{v}_{foot,xy}\| \cdot \mathbb{1}[\text{接触力} > 1\,\text{N}]
$$

- **含义**：支撑相脚在地面滑动则惩罚。

#### `undesired_contacts`（$w=-1.0$）

$$
f = \#\{\text{body} : \|\mathbf{F}_{contact}\| > 1\,\text{N},\; \text{非脚踝}\}
$$

- **检测体**：`(?!.*ankle.*).*`（除脚踝外所有连杆）
- **含义**：膝、髋、躯干等触地计数惩罚。

---

## 五、设计意图与权重层次

```text
[主导] 速度跟踪 + 存活     w: +1.0, +0.5, +0.15
[强约束] 高度/姿态/限位    w: -10, -5, -5
[步态塑造] 步态+抬脚       w: +0.5, +1.0
[正则] 动作/速度/能耗      w: -0.05 ~ -2.5e-7
```

1. **跟踪优先**：`track_lin_vel_xy` 与 `track_ang_vel_z` 决定能否按指令行走。  
2. **形态安全**：`base_height`、`flat_orientation_l2`、`dof_pos_limits` 权重大，防倒、防姿态发散。  
3. **步态质量**：`gait` + `feet_clearance` 塑造周期步态；`feet_slide`、`undesired_contacts` 抑制滑步与跪地。  
4. **平滑与能耗**：小权重正则，避免动作抖动与过激力矩。

---

## 六、终止条件（与 `alive` 相关）

| 名称 | 条件 |
|------|------|
| `time_out` | episode 到时（非惩罚性终止） |
| `base_height` | 根高度 < 0.2 m |
| `bad_orientation` | 倾斜角 > 0.8 rad |

提前终止时 `alive` 为 0，且无法继续累积跟踪 reward。

---

## 七、调参建议

| 现象 | 可调项 | 建议方向 |
|------|--------|----------|
| 跟踪慢、速度误差大 | `track_lin_vel_xy` / `track_ang_vel_z` weight 或 `std` | 略增 weight 或略减 `std`（更陡的 exp） |
| 易摔倒、高度不稳 | `base_height`, `flat_orientation_l2` | 略增惩罚 weight |
| 脚拖地、滑步 | `feet_slide` | 略增 $|w|$（如 -0.2 → -0.3） |
| 膝/髋触地 | `undesired_contacts` | 略增 $|w|$ |
| 步态僵硬、抬脚不足 | `feet_clearance`, `gait` | 略增正向 weight |
| 动作抖动 | `action_rate`, `joint_vel` | 略增 $|w|$ |
| 手臂姿态怪异 | `joint_deviation_arms` | 略增 $|w|$ |

修改后需重新训练；仅改 weight 不改变 $f_i$ 定义。

---

## 八、可视化与调试

| 观察量 | 期望（训练后期） | 异常信号 |
|--------|------------------|----------|
| `Episode_Reward/track_lin_vel_xy` | 高且稳定 | 长期接近 0 → 不会走 |
| `Episode_Reward/base_height` | 绝对值小 | 持续很负 → 高度失控 |
| `Episode_Reward/undesired_contacts` | 接近 0 | 持续负 → 身体触地 |
| `Episode_Reward/feet_slide` | 绝对值小 | 很负 → 严重滑步 |
| `Episode_Reward/gait` | 随速度指令升高 | 长期为 0 → 步态相位未对齐 |
| episode length | 接近 1000 steps | 明显偏短 → 频繁 `base_height` / `bad_orientation` 终止 |

建议在 Isaac Sim 中同时查看：根高度、脚接触、速度指令与机体速度曲线。

---

## 九、参数速查表

| 模块 | 参数 | 值 |
|------|------|-----|
| 跟踪 | `track_lin_vel_xy` weight | 1.0 |
| 跟踪 | `track_ang_vel_z` weight | 0.5 |
| 跟踪 | `std`（两项共用） | $\sqrt{0.25}=0.5$ |
| 存活 | `alive` weight | 0.15 |
| 高度 | `target_height` | 0.78 m |
| 高度 | `base_height` weight | -10.0 |
| 步态 | `gait` period / offset | 0.8 s / [0, 0.5] |
| 步态 | `gait` threshold | 0.55 |
| 抬脚 | `feet_clearance` target / std | 0.1 m / 0.05 |
| 接触 | `undesired_contacts` threshold | 1 N |
| 控制 | $\Delta t$ | 0.02 s |

---

## 十、与 GRU 任务的关系

| 项目 | `Unitree-G1-29dof-Velocity` | `Unitree-G1-29dof-Velocity-GRU` |
|------|------------------------------|----------------------------------|
| `RewardsCfg` | 相同 | 相同 |
| 环境配置类 | `RobotEnvCfg` | `RobotGruObs1EnvCfg` |
| 观测 | `history_length=5`（默认） | `history_length=5` |
| Actor | MLP（`BasePPORunnerCfg`） | GRU（`G1GruActorPPORunnerCfg`） |

Reward 文档对两个任务均适用；差异仅在策略网络与任务注册名。
