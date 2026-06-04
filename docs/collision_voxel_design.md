# 碰撞点 → 体素地图：设计说明

> 状态：**定稿**（后处理脚本已实现）  
> 输入：`dataset/episode_XXXXX.npz` 中的 `collisions_json` + `root_pos_w` / `root_quat_w`  
> 目标：每个时刻 \(t\) 得到机器人周围的碰撞体素占用图；碰撞点**世界坐标持久化**，首次出现后一直参与体素化。

---

## 定稿参数（已确认）

| 项目 | 约定 |
|------|------|
| 水平中心 | **A. 根节点 XY**：`root_pos_w[t]` 的 \((x, y)\)，不用 z |
| 水平朝向 | `root_quat_w[t]` 的 **yaw**；网格随机器人旋转（\(+x\) 前，\(+y\) 左） |
| 地面 | 恒为世界系 **\(z = 0\)** |
| 高度范围 | **\(z \in [0,\, 1.5]\,\text{m}\)**（相对地面） |
| 分辨率 | **0.1 m**（10 cm） |
| 水平范围 | 局部 **\(x', y' \in [-1, 1]\,\text{m}\)**（各向 ±1 m） |
| 水平索引 | **方式 1**：\(i, j \in [0, 19]\)，中心在格点中间 |
| 持久化 | **不去重**：每步有效 `contact_position`（世界系）**全部追加**到列表 |
| 线速度过滤 | \(\|v_\text{cmd}\| < 0.3\,\text{m/s}\) 的 **整条 episode 丢弃**，不写入 `dataset_voxel/`（见下） |
| 占用规则 | 持久化点落入体素 → 该格为 **1**，否则 **0** |
| 输出形式 | **单独一套数据集**；每个文件 = **原始 NPZ 全部字段** + **局部体素字段**（见下） |
| 体素坐标系 | **非**世界系裁切；碰撞点世界坐标 → 变换到 **当前步机器人局部水平系（yaw）** 后体素化 |

**输出存储（单独数据集，在原始数据上叠加体素）**

| 项目 | 约定 |
|------|------|
| 目录 | 建议 `dataset_voxel/`（与 `dataset/` 并列），不覆盖原始 `episode_*.npz` |
| 文件名 | 与源文件一致，如 `dataset_voxel/episode_00001.npz` |
| 文件内容 | **复制** `dataset/episode_00001.npz` 中已有字段（`time`、`joint_pos`、`collisions_json`、`root_pos_w` 等）**原样保留** |
| 新增字段 | 仅多一个 **`collision_voxel`** |

| 字段 | Shape | 说明 |
|------|-------|------|
| `collision_voxel` | `(T, 20, 20, 15)` | `uint8`，`{0,1}`；**机器人局部**网格（水平随 yaw，\(z\) 相对地面 \([0,1.5]\)）；轴 `(x_forward, y_left, z_up)` |

`T` 与源轨迹步数一致（与 `time`、`joint_pos` 等第 0 维对齐）。实现：**离线后处理**读 `dataset/` → 写 `dataset_voxel/`。

**线速度过滤（定稿）**

- 速度指令：`command[t] = [v_x, v_y, \omega_z]`（采集时**整条轨迹内恒定**，取 `command[0]` 即可）。
- 线速度模长：\(\|v_\text{cmd}\| = \sqrt{v_x^2 + v_y^2}\)（**不含** \(\omega_z\)）。
- 当 **\(\|v_\text{cmd}\| < 0.3\,\text{m/s}\)**：**不生成** `dataset_voxel/episode_XXXXX.npz`（整条数据丢弃；`dataset/` 原文件保留）。
- 当 **\(\|v_\text{cmd}\| \ge 0.3\,\text{m/s}\)**：正常后处理并写入 `dataset_voxel/`。

**每步流程**（仅对通过线速度过滤的 episode）

```text
contacts_world ← 空（按 episode）
for t = 0 .. T-1:
    将 collisions[t] 中每条有效 contact_position（世界系）全部并入 contacts_world
    collision_voxel[t] = voxelize(contacts_world, root_pos_w[t], root_quat_w[t])
```

---

## 实现约定（编码按此执行）

### 索引与边界

- 局部范围：\(x', y' \in [-1,\, 1)\,\text{m}\)，\(z \in [0,\, 1.5)\,\text{m}\)（左闭右开）。
- 体素下标（方式 1，\(i,j,k \in [0,19]/[0,14]\)）：

```text
i = floor((x_local + 1.0) / 0.1)，clamp 到 [0, 19]
j = floor((y_local + 1.0) / 0.1)，clamp 到 [0, 19]
k = floor(z / 0.1)，clamp 到 [0, 14]
```

- 网格外碰撞点：丢弃，不进入持久化集合。

### 位姿与 yaw

- `root_quat_w`：**`[w, x, y, z]`**（与 `collision_data_collection.md` 一致）。
- yaw（绕世界 \(z\)）：`atan2(2(wz+xy), 1-2(y²+z²))`。
- 世界点 \(\mathbf{p}\) 转局部水平：先减根节点 XY，再按 yaw 旋转；\(+x\) 为前方，\(+y\) 为左侧。

### 持久化（无去重）

- 每步将 `collisions[t]` 里通过校验的记录**逐条追加**世界坐标，**不**做体素/距离去重（连续多步重复上报会保留多条）。
- 写 `collision_voxel[t]` 时，对**全部**持久化点用**当前步**位姿投影；多个世界点落入同一体素时，该格仍为 **1**。

### 输入过滤

- 跳过：`contact_position` 缺失/非 3D/含 NaN。
- 跳过：`contact_position_source` 存在且 **≠** `"contact_pos_w"`（无该字段时，仅要求坐标有效，兼容旧文件）。

### `command` 过滤说明

- 后处理脚本在读写前检查 \(\|v_{cmd}\|=\sqrt{v_x^2+v_y^2}\)（用 `command[0]`）：
  - **< 0.3 m/s**：**跳过该 episode**，`dataset_voxel/` 中**无对应文件**；
  - **≥ 0.3 m/s**：写入 `dataset_voxel/episode_XXXXX.npz`。

### 脚本

```bash
python scripts/rsl_rl/process_collision_voxels.py --input dataset --output dataset_voxel
# 可选：--min-cmd-speed 0.3
```

实现：`scripts/rsl_rl/voxel_collision_utils.py`、`scripts/rsl_rl/process_collision_voxels.py`。

---

## 1. 需求摘要

### 1.1 体素化

| 参数 | 取值 |
|------|------|
| 体素分辨率 | **0.1 m**（10 cm） |
| 水平范围 | 相对机器人中心，**前/后/左/右各 1 m**（即水平约 **2 m × 2 m**） |
| 高度范围 | 总高 **1.5 m** |
| 水平原点 | **根节点 XY**（见定稿参数） |
| 高度原点 | **地面** \(z=0\)，不是机器人质心高度 |
| 占用规则 | 若**持久化后的碰撞点**（世界坐标）落在某体素内，则该体素为 **1**，否则 **0** |

### 1.2 碰撞点持久化

| 项目 | 当前数据 | 目标逻辑 |
|------|----------|----------|
| 时间维度 | 仅碰撞当步 `collisions` 非空 | **首次出现后一直保留**（仅进入 `dataset_voxel` 的 episode） |
| 坐标 | 每步世界坐标 `contact_position` | 保存**全局世界坐标**（不变） |
| 用途 | — | 用截至 \(t\) 的持久化点集生成 `collision_voxel[t]` |

### 1.3 线速度过滤

**\(\|command_{xy}\| < 0.3\,\text{m/s}\)** 的 episode **不写入** `dataset_voxel/`；`dataset/` 原始 NPZ **全部保留**。

---

## 2. 局部体素 vs 全局裁切

`collision_voxel[t]` **不是**在世界系固定 3D 栅格上截取 ROI，而是：

1. 持久化碰撞点保存为 **世界坐标**；
2. 每步用 **当前** `root_pos_w[t]`、`root_quat_w[t]`（yaw）把点变到 **机器人局部水平系**；
3. 在局部 \([-1,1]^2 \times [0,1.5]\)（z 仍相对地面）内划 20×20×15 格得到占用。

因此每条 `dataset_voxel/episode_*.npz` 里：原始轨迹字段不变，**叠加**的是随时刻变化的 **ego-centric 局部体素序列**。

---

## 3. 体素网格明细

- 水平：\(2\,\text{m} \times 2\,\text{m}\)，分辨率 0.1 m → **20 × 20**；索引 \(i,j \in [0,19]\)，局部 \(x', y' \in [-1,1]\)。
- 竖直：\(z \in [0, 1.5]\,\text{m}\)，分辨率 0.1 m → **15** 层，索引 \(k \in [0,14]\)。
- 单步约 6000 cell；`collision_voxel[t].shape = (20, 20, 15)`。

---

## 4. 处理流程（推荐管线）

分两个阶段，便于先验证再改采集。

```text
原始 NPZ (collisions_json 仅碰撞当步有记录)
        │
        ▼
┌───────────────────────────────────────┐
│ 阶段 A：持久化碰撞点（按 episode）      │
│  维护列表 contacts_world[]           │
│  每步 t：合并当步全部有效点（§5）；低速 episode 在写入前剔除   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 阶段 B：每步体素化                      │
│  用 root_pos_w[t], root_quat_w[t]     │
│  将 contacts_world 中所有点 → 体素占用  │
└───────────────────────────────────────┘
        │
        ▼
输出：`dataset_voxel/episode_*.npz`（原始字段 + `collision_voxel`）
```

**不建议**在体素化完成前改采集脚本；先用 **离线后处理** 跑通 `episode_00001.npz`，再决定是否把持久化逻辑写进 `collect_data.py`。

---

## 4.1 单步体素化算法（概念）

对时间步 \(t\)：

1. **输入**
   - 机器人根位姿：\(T_{\text{root}}(t)\) ← `root_pos_w[t]`, `root_quat_w[t]`
   - 持久化碰撞点集：\(\mathcal{P}_t = \{ \mathbf{p}_k \in \mathbb{R}^3 \}\)（世界系），仅包含 **时间 \(\le t\)** 已出现的点

2. **对每个** \(\mathbf{p} = (p_x, p_y, p_z) \in \mathcal{P}_t\)：
   - 平移到以根节点 XY 为参考：\(\mathbf{p}' = \mathbf{p} - [r_x, r_y, 0]^T\)
   - 用 yaw 旋转到机体水平系：\(\mathbf{p}_\text{local} = R_z(\text{yaw})^{-1} \mathbf{p}'\)
   - 若 \(p_{\text{local},x} \in [-1, 1]\)、\(p_{\text{local},y} \in [-1, 1]\)、\(p_z \in [0, 1.5]\)，则落入对应体素，置 **1**

3. **输出** `collision_voxel[t, i, j, k] ∈ {0,1}`

多个碰撞点落在同一体素 → 仍为 1（占用合并）。

### 4.2 边界与数值

- `contact_position` 来自 `contact_pos_w`，为世界系；与 `root_pos_w` 同一坐标系，**无需**在体素化时再乘 body 变换（除水平 re-base + yaw 外）。
- 点在网格外：忽略（不裁剪进网格）。
- 建议对持久化点做 **\(|\mathbf{p} - \mathbf{r}_\text{root}|_\infty\)** 或距离预筛，避免明显离群点打满整块网格（采集阶段已有质心 3 m 过滤，后处理可再收紧）。

---

## 5. 持久化碰撞点

### 5.1 基本规则

```text
if sqrt(command[0]^2 + command[1]^2) < 0.3: 跳过本 episode，不写 dataset_voxel
contacts_world ← 空
for t in 0 .. T-1:
    for each 碰撞记录 in collisions[t]:
        将每条有效 contact_position (world) 追加到 contacts_world（不去重）
    collision_voxel[t] = voxelize(contacts_world, root_pose[t])
```

注意：持久化的**世界点列表**只增不减（多条可重复）；**同一格** `(i,j,k)` 在不同 \(t\) 可能因机体运动而变化。

### 5.2 Episode 边界

持久化列表 **每个 episode 独立清空**（不跨 `episode_00001` / `00002`）。与「一条轨迹一个文件」一致。

---

## 6. 与现有采集数据的关系

当前 `episode_00001.npz` 实测（示例）：

- 约 499 步，48 步有碰撞记录；
- `contact_position` 为世界系，`contact_position_source: contact_pos_w`；
- 碰撞多集中在连续若干步（同一点重复上报）。

因此：

1. **持久化 + 体素** 适合作为 **离线脚本** 从现有 `dataset/` 批量生成，无需立刻重采；
2. 若希望 NPZ 直接带 `collision_voxel`，可在采集端增加「在线累积 + 每步写体素」，但会增大 IO 与实现复杂度——建议第二期再做。

---

## 7. 后续步骤

1. ~~`process_collision_voxels.py`~~ ✅ 已实现。  
2. ~~体素可视化~~：`replay_voxel_dataset.py` 在 Isaac Lab 中回放 `dataset_voxel/` 并显示橙色占用格。  
3. （可选）采集端在线写 `collision_voxel`（第二期）。

---

## 8. 小结

定稿见文首：局部系体素化；**\(\|command_{xy}\| \ge 0.3\)** 的 episode 才写入 **`dataset_voxel/`**（原始字段 + **`collision_voxel`**），否则整条丢弃。
