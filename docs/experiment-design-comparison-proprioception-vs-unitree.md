# 实验设计对比：proprioception vs unitree_rl_lab

本文整理 `/home/hz/proprioception/proprioception` 与 `/home/hz/proprioception/unitree_rl_lab` 两套 Isaac Lab + RSL-RL 实验在**任务目标、场景、碰撞/APF、观测奖励、Play 可视化**上的差异，便于对照消融与复现实验。

---

## 1. 总体定位

| 维度 | **proprioception** | **unitree_rl_lab** |
|------|-------------------|-------------------|
| 核心任务 | **目标点导航 + 避障**：朝随机 goal 行走，途中出现障碍物，用 APF 修正速度指令 | **平面速度跟踪**：跟踪随机采样的 `(vx, vy, ωz)`，无导航目标 |
| 环境类 | 自定义 `G1NavEnv`（继承 `ManagerBasedRLEnv`） | 标准 `ManagerBasedRLEnv` |
| 注册任务 | `Template-G1-Proprioception-Vel-v0` / `...-Vel-Play-v0` | `Unitree-G1-29dof-Velocity` / `...-Velocity-GRU` |
| 机器人 | G1 29-DoF | G1 29-DoF（同族 URDF） |
| 主要实验轴 | 障碍物 + APF + goal 导航 + GRU actor | MLP vs GRU actor（**同一套平地 MDP**） |

**关系**：`unitree_rl_lab` 可视为**平地 locomotion 基线**；`proprioception` 在其 locomotion 能力之上叠加了**导航场景、刚体障碍、接触驱动的 APF 速度合成**与配套 debug 工具。

---

## 2. 场景与障碍物

### 2.1 proprioception（有障碍物）

**配置**：`source/proprioception/proprioception/tasks/proprioception_vel/nav_env_cfg.py` → `G1AmpSceneCfg`

- **地面**：`GroundPlaneCfg`（`/World/ground`）， deliberately **不用** `TerrainImporter`，避免地面接触被 APF 误判为障碍碰撞。
- **障碍物**：每个 parallel env 一个刚体立方体 `{ENV_REGEX_NS}/Obstacle`
  - 尺寸约 `0.25 × 0.25 × 1.0` m，质量 80 kg
- **目标标记**：红色球体 `GoalMarker`（仅可视化，无碰撞）
- **延迟出现逻辑**（`DelayedGoalObstacleCfg`，在 `nav_env.py` 实现）：
  1. Reset / 重采样 goal 时：障碍**瞬移到 goal 附近**（“藏起来”），不挡起步
  2. 当平面 `goal_distance` 首次低于每回合随机阈值 `[2.0, 5.0]` m：障碍**瞬移到机身水平 +x 前方** `0.8` m
  3. 每段 goal 只触发一次（`obstacle_spawned` 标志）

**Goal 采样**：以机器人当前 XY 为圆心、半径 `6.0` m 的圆周上均匀采样；每 `10–12` s 可重采样 goal（不结束 episode）。

### 2.2 unitree_rl_lab（无障碍物）

**配置**：`source/unitree_rl_lab/.../velocity_env_cfg.py` → `RobotSceneCfg`

- **地形**：`TerrainImporterCfg` + 生成器，当前子地形仅 **flat** `MeshPlaneTerrainCfg`
- **无** `Obstacle` / `GoalMarker` 等导航资产
- **命令**：`UniformLevelVelocityCommandCfg` 周期性随机 resample 速度，带 curriculum 扩范围

---

## 3. 碰撞检测与碰撞点记录

### 3.1 proprioception：障碍专用接触 + 碰撞点缓存

**传感器**（过滤到障碍物，避免地面误报）：

```python
# nav_env_cfg.py — obstacle_contact_forces
ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle"],
    track_contact_points=True,  # 读取真实接触点世界坐标
)
```

**检测条件**（`nav_env.py::_update_apf_points`）：

- 使用 **`force_matrix_w`**（已 filter 到 Obstacle），不用 `net_forces_w`（会含地面力）
- 接触有效：`‖F‖ > F_threshold`（默认 **0.3 N**）
- 从 sensor 读 **`contact_pos_w`**，过滤 NaN

**碰撞点缓存**（每 env 维护）：

| 状态量 | 形状 | 含义 |
|--------|------|------|
| `apf_points_w` | `(N_env, N_max, 3)` | 世界系接触点（APF 用 XY） |
| `apf_slot_valid` | `(N_env, N_max)` | 槽位是否有效 |
| `apf_delta_v` | `(N_env, 2)` | 世界系排斥速度 EMA |

**更新规则**（`ApfCollisionCfg`，默认 `N_max=10`，`R=1.0` m）：

1. **距离剔除**：点与机器人 XY 距离 `> R` 则清空（**无时间 TTL**）
2. **新接触**：力超阈值则写入；与已有点距离 `< d_merge`（0.02 m）则**合并平均**
3. **槽位满**：驱逐**离机器人最远**的点
4. **排序**：按 XY 距离**近→远**，供 critic 观测语义稳定

**Critic 特权观测**：`mdp/observations.py::apf_collision_slots_base_xy` — 每槽 `(x_b, y_b, valid)`；**Policy 组不包含 APF 槽位**。

另有一套 `contact_forces`（脚接触、Gallant 步态奖励等），与 APF 分离。

### 3.2 unitree_rl_lab：仅 locomotion 接触

- 单一 `ContactSensorCfg` 用于 **`undesired_contacts`**（惩罚非脚部位碰地）及步态奖励
- **无** 障碍过滤、**无** `track_contact_points`、**无** 碰撞点缓存张量

---

## 4. APF 人工势场与速度指令

仅在 **proprioception** 的 `G1NavEnv` 中实现；`unitree_rl_lab` **无 APF**。

### 4.1 导航基准速度 `v_cmd`

由 goal 误差在 **yaw-only** 机体系下 P 控制得到 `(vx, vy, ωz)`：

- `gain=2.5`，平面速度模长限制 `lin_vel_xy_max=1.0` m/s
- 距 goal `< 0.4` m 时 **导航线速度置零**（APF 排斥仍可叠加）

实现：`nav_env.py::_apply_goal_velocity_command`

### 4.2 APF 排斥与合成 `v_out`

对每个有效碰撞点 \(i\)（机体系 XY）：

- 距离 \(d_i\)，单位排斥方向 \(\hat{r}_i\)
- 权重：\(w(d_i) = \mathrm{clamp}\big(1 - \mathrm{clamp}(d_i-\rho,0)/R,\,0,\,1\big)^\beta\)，\(\rho=0.2\)，\(\beta=2\)
- 瞬时排斥：\(\Delta v = \sum_i k \cdot w(d_i) \cdot \hat{r}_i\)（默认 \(k=1\)）
- **EMA 平滑**：`apf_delta_v ← α·apf_delta_v + (1-α)·Δv`（\(\alpha=0.5\)）
- 合成：`v_raw = v_cmd + R_yaw · apf_delta_v`，再对 **平面模长** 限制到 `lin_vel_xy_max`（保持方向）

结果写入 `base_velocity` command buffer → 策略看到的 `velocity_commands` 已是 **APF 修正后** 的指令。

### 4.3 相关奖励

- `track_apf_velocity_reference_exp`（权重 2.0）：鼓励实际 base 线速度跟踪 `v_out`
- 另有 goal 距离、Gallant 步态、姿态等奖励项

**设计要点**：APF 是 **command 层参考 + critic 特权 + tracking reward**，不是 policy 直接输入的 exteroceptive map。

---

## 5. 观测与策略输入对比

| 观测项 | proprioception (policy) | unitree_rl_lab (policy) |
|--------|-------------------------|-------------------------|
| `velocity_commands` | **已被 env 写成 APF 合成速度** | 随机速度命令 |
| `goal_pos_in_base_frame` | ✅ 有 | ❌ 无 |
| `base_ang_vel`, `projected_gravity`, joints, `last_action` | ✅ | ✅ |
| APF 碰撞槽位 | ❌（仅 critic） | ❌ |
| `height_scan` | ❌ | 传感器存在，**观测项已注释** |
| history | 有 | `history_length=5` |

两者当前训练均可用 **GRU actor + MLP critic**（proprioception：`G1ProprioceptionVelRslRlRunnerCfg`；unitree：`G1GruActorPPORunnerCfg`）。

---

## 6. Play 可视化

### 6.1 proprioception：双层 debug

**A. Isaac 内建标记（env 内默认开启，有 GUI 时）**

- 文件：`nav_env.py` — `_setup_apf_visualizer` / `_render_apf_debug`
- 白球：缓存的 `apf_points_w`
- 箭头：蓝 `v_cmd`、红 `Δv`、黄 `v_out`、绿实测 `v_xy`
- 默认只看 `env_id=0`，建议 play 时 `--num_envs 1`

**B. Matplotlib 局部碰撞栅格（CLI 可选）**

- 文件：`apf_collision_grid_vis.py` — `ApfPlayDebugPanel`
- 开关：`scripts/rsl_rl/play.py --apf_collision_grid_vis`
- 参数示例：
  - `--apf_collision_grid_radius_m 1.0`（圆域半径）
  - `--apf_collision_grid_cell_m 0.05`（栅格分辨率 → 约 40×40）
  - `--apf_collision_grid_env_id 0`
- **左图**：机体系局部 occupancy heatmap（由 APF 点投影占格）+ 碰撞散点 + 速度箭头 + ωz 弧
- **右图**：vx / vy / ωz 的 cmd vs out vs measured 条形对比

示例（摘自 proprioception README）：

```bash
/home/hz/IsaacLab/isaaclab.sh -p .../proprioception/scripts/rsl_rl/play.py \
  --task Template-G1-Proprioception-Vel-Play-v0 \
  --num_envs 1 --real-time \
  --apf_collision_grid_vis \
  --checkpoint logs/.../model_XXXX.pt
```

### 6.2 unitree_rl_lab

- `play.py`：加载 checkpoint、可选 `--video` / `--real-time`
- `commands.base_velocity.debug_vis` 可显示**速度命令箭头**
- **无** APF 标记、**无** 局部碰撞栅格 matplotlib 面板
- `height_scanner` 的 `GridPatternCfg` 仅为**射线采样几何**，不是导航 occupancy map

---

## 7. 训练入口对照

| | proprioception | unitree_rl_lab |
|---|----------------|----------------|
| Train | `scripts/rsl_rl/train.py --task Template-G1-Proprioception-Vel-v0` | `unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity` |
| Play | `...-Vel-Play-v0` + 可选 `--apf_collision_grid_vis` | `unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-GRU` |
| 并行 env 数（默认） | Train 4096 / Play 32 | Train 4096 / Play 32 |
| Runner | `OnPolicyRunner` + PPO | 同左（**未接 AMP runner**） |

---

## 8. 数据流示意（proprioception 独有路径）

```mermaid
flowchart LR
  subgraph Scene
    G[Goal 采样]
    O[Obstacle 延迟瞬移]
  end
  subgraph Sensing
    CS[obstacle_contact_forces<br/>filter → Obstacle]
  end
  subgraph APF
    CP[碰撞点缓存<br/>merge / evict / sort]
    APF[APF 排斥 + EMA]
    VOUT[v_out = v_cmd + Δv]
  end
  subgraph RL
    CMD[velocity_commands]
    POL[Policy GRU]
    CRIT[Critic + APF slots]
  end
  G --> VCMD[v_cmd P 控制]
  O --> CS
  CS --> CP --> APF
  VCMD --> VOUT
  APF --> VOUT
  VOUT --> CMD --> POL
  CP --> CRIT
  VOUT --> CRIT
```

unitree_rl_lab 路径可简化为：`UniformVelocityCommand → Policy`，无 Obstacle / APF / Goal 支路。

---

## 9. 规划但未在当前代码落地的功能

`proprioception/docs/proprioception.md` 描述过：

- 用与 play 栅格相同规格（1 m 半径、0.05 m  cell、机体系）训练 **grid 预测头**，作辅助监督

当前仓库仅有 **栅格可视化 + critic 碰撞槽位**，**无** grid head 损失接入 `OnPolicyRunner`。

历史任务（`Template-G1-Proprioception-Goal-v0`、AMP+SKRL 等）在 README / `log.md` 中有记录，**当前 `tasks/` 树仅保留 `proprioception_vel`**。

---

## 10. 关键文件索引

### proprioception

| 主题 | 路径 |
|------|------|
| 场景 / APF / 障碍配置 | `tasks/proprioception_vel/nav_env_cfg.py` |
| Goal、障碍瞬移、APF、速度覆盖 | `tasks/proprioception_vel/nav_env.py` |
| Play 栅格可视化 | `tasks/proprioception_vel/apf_collision_grid_vis.py` |
| APF critic 观测 | `tasks/proprioception_vel/mdp/observations.py` |
| APF 跟踪奖励 | `tasks/proprioception_vel/mdp/rewards.py` |
| 设计文档 | `docs/apf-collision-points-vel-ref-reward.md`, `docs/baseline-velocity-command.md` |

### unitree_rl_lab

| 主题 | 路径 |
|------|------|
| 平地速度任务 MDP | `tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py` |
| 任务注册 | `tasks/locomotion/robots/g1/29dof/__init__.py` |
| PPO / GRU agent | `tasks/locomotion/agents/rsl_rl_ppo_cfg.py` |
| 奖励说明 | `docs/tasks/Unitree-G1-29dof-Velocity-GRU-reward.md` |

---

## 11. 一句话对照（便于记忆）

- **unitree_rl_lab**：在**无障碍平地**上学 **跟踪随机速度**；接触只服务步态与 undesired body contact。
- **proprioception**：在 **goal 导航** 中学走路，**延迟障碍**触发 **过滤接触** → **缓存碰撞点** → **APF 改速度指令**；critic 看碰撞槽位，play 可开 **局部碰撞栅格图** 与 Isaac 内 APF 箭头 debug。

---

*文档生成依据两个仓库当前 `main` 工作区源码；若分支名如 `test-noobstical` 等用于消融，以对应分支 checkout 为准。*
