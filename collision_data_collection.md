# G1 机器人碰撞与本体感知数据采集说明文档

本篇文档说明了如何利用训练好的 G1 机器人 Policy (`model_52800.pt`) 运行仿真，在给定的随机目标速度指令（采用最大速度范围）下，采集本体感知数据以及过滤后的有效碰撞力信息（排除小于 1.0 N 的微弱力），并获取碰撞点以及根节点（Root）的位姿，以方便在局部坐标系下计算碰撞位置与受力方向。

---

## 1. 采集流程设计 (Design Workflow)

1. **单次运行周期 (10s Episode)**：每个环境的单次运行周期（Episode）固定为 **10.0 秒**（仿真频率为 50 Hz，对应 500 个决策步）。
2. **单组随机速度控制**：在每次运行开始（Reset）时，系统会为机器人随机抽取一组目标速度指令（基于最大速度范围 `limit_ranges`）。在整个 10.0 秒的运行周期内，**该速度指令保持恒定**，不再重新采样。
3. **环境初始化与速度范围**：使用 `Unitree-G1-29dof-Velocity` 任务，配置采用最大随机速度范围 `limit_ranges`：
   * 前向/后向速度 $v_x \in [-0.5, 1.0] \text{ m/s}$
   * 横向移动速度 $v_y \in [-0.3, 0.3] \text{ m/s}$
   * 旋转角速度 $\omega_z \in [-0.2, 0.2] \text{ rad/s}$
4. **提前重置 (Early Reset)**：如果在 10s 内机器人因碰撞严重跌倒（触发基座高度过低或姿态倾角过大等终止条件），环境会提前 Reset，并重新生成一组随机速度开始新的 10s 循环。
5. **加载 Policy**：加载训练生成的 checkpoint 模型进行推理。
6. **仿真运行与步进**：在每个仿真步中，读取机器人的状态、根节点位姿和传感器数据。
7. **碰撞力与受力方向过滤**：通过 `contact_forces` 传感器获取各个 Body 上的 3D 碰撞力矢量 $\mathbf{f}_{world} = [f_x, f_y, f_z]$，当碰撞力大小 $\|\mathbf{f}_{world}\|_2 \ge 1.0\text{ N}$ 时，记录：
   * 碰撞发生的位置（Body 名称）
   * 碰撞力矢量（3D 方向与大小）
   * 碰撞点的 3D 世界坐标
8. **根节点位姿**：记录根节点的 3D 位置 `root_pos_w` 与四元数 `root_quat_w`，用于将世界坐标系下的碰撞点位置及碰撞受力方向转换到机器人局部坐标系下。
9. **IMU 差分模拟**：基于相邻步的线速度差进行数值微分，计算并保存基座的线加速度（含重力分量），以提供真实的 IMU 加速度计数据。
10. **数据存储**：将每个有效的时间步数据保存，仿真结束后统一输出为 `.pt` 文件。

---

## 2. 数据采集字段表 (Data Schema Table)

在每个仿真时间步中，我们将数据打包并统一记录为以下结构：

| 数据类别 (Category) | 字段名称 (Field) | 维度/格式 (Shape) | 字段说明 (Description) |
| :--- | :--- | :--- | :--- |
| **A. 机器人本体感知与控制信息**<br>(实机车载传感器直接可测数据) | `time` | `float` | 仿真当前的时间戳 (秒) |
| | `command` | `[v_x, v_y, w_z]` | 目标速度指令（控制命令输入，每 10s Episode 内恒定） |
| | `base_ang_vel` | `[w_x, w_y, w_z]` | 机器人基座在本体坐标系下的实际角速度 (IMU 陀螺仪，模型输入) |
| | `projected_gravity` | `[3]` | 重力向量在本体坐标系下的投影 (IMU 倾角，模型输入) |
| | `joint_pos` | `[29]` | 29个关节的当前角度位置 (模型输入) |
| | `joint_vel` | `[29]` | 29个关节的当前角速度 (模型输入) |
| | `last_action` | `[29]` | 上一步输出的关节控制动作 (模型输入) |
| | `base_lin_acc` | `[3]` | 机器人基座在本体坐标系下的线加速度 (IMU 加速度计，含重力分量) |
| **B. 全局特权信息**<br>(仿真引擎真值，实机通常需状态估计或无法获取) | `root_pos_w` | `[3]` | 机器人 Root 在世界坐标系下的 3D 坐标 `[x, y, z]` (用于局部坐标转换) |
| | `root_quat_w` | `[4]` | 机器人 Root 在世界坐标系下的四元数 `[w, x, y, z]` (用于局部坐标转换) |
| | `base_lin_vel` | `[v_x, v_y, v_z]` | 机器人基座在本体坐标系下的实际线速度 (特权观测，实机无法直接测量) |
| | `joint_torques` | `[29]` | 29个关节的当前电机实际输出力矩 (实机通常靠电流估算) |
| **C. 碰撞信息**<br>(受力大于 1.0 N) | `collisions` | `list` | 碰撞的详细信息列表（本步若无碰撞则为空列表 `[]`，格式见下文） |

#### 碰撞信息列表项格式 (collisions list item)
如果某个 Body 在当前步检测到的碰撞力 $\ge 1.0\text{ N}$，则在 `collisions` 列表中添加一项：
```python
{
    "body_name": str,       # 碰撞发生的位置（例如 "left_ankle_roll_link", "torso_link" 等）
    "force_magnitude": float, # 碰撞力大小（L2 范数，单位：牛顿 N）
    "contact_force_vector": [f_x, f_y, f_z], # 世界坐标系下 3D 碰撞力矢量（单位：牛顿 N）
    "contact_position": [x, y, z] # 碰撞点在世界坐标系下的 3D 坐标
}
```

---

## 3. 传感器与位姿调用机制 (Isaac Lab API)

我们将通过以下接口在每个时间步提取所需数据：

* **获取机器人资产与根节点位姿/力矩**：
  ```python
  robot = env.unwrapped.scene["robot"]
  root_pos_w = robot.data.root_pos_w  # 形状 (num_envs, 3)
  root_quat_w = robot.data.root_quat_w  # 形状 (num_envs, 4)
  joint_torques = robot.data.applied_torque  # 形状 (num_envs, 29)
  ```
* **IMU 传感器差分计算**：
  ```python
  # dt 为仿真决策周期 (通常为 0.02s)
  # 线加速度：包含重力投影分量以贴近真实加速度计测量
  base_lin_acc = (robot.data.root_lin_vel_b - last_root_lin_vel_b) / dt - robot.data.projected_gravity_b * 9.81
  ```
* **获取碰撞传感器**：
  ```python
  contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
  ```
* **碰撞力矢量**：`contact_sensor.data.net_forces_w`，其形状为 `(num_envs, num_bodies, 3)`。
* **碰撞力大小**：通过 L2 范数计算得到 `forces_mag = torch.norm(net_forces_w, dim=-1)`，其形状为 `(num_envs, num_bodies)`。
* **碰撞点 3D 世界坐标**：`contact_sensor.data.contact_pos_w`，其形状为 `(num_envs, num_bodies, 1, 3)`，第 $b$ 个 body 的碰撞点为 `contact_pos_w[env_idx, b, 0]`。
* **发生碰撞的 Body 映射**：`contact_sensor.body_names`。

---

## 4. 运行采集命令示例 (Command Example)

脚本写好后，可以使用如下命令在 Docker 中运行数据采集：

```bash
docker exec prop /home/hz/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/collect_data.py \
  --task Unitree-G1-29dof-Velocity \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-29_11-08-59_first-test/model_52800.pt \
  --num_steps 5000 \
  --output logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-29_11-08-59_first-test/collision_data.pt
```

---

## 5. 示例读取与局部坐标系转换代码 (Local Coordinate Transformation Example)

采集到的数据可以通过以下方式加载，并转换为机器人的局部（Body）坐标系位置与受力方向：

```python
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# 加载采集到的数据
data = torch.load("logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-29_11-08-59_first-test/collision_data.pt")

print(f"Total steps collected: {len(data)}")

for step_idx, step_data in enumerate(data):
    if len(step_data["collisions"]) > 0:
        # 获取机器人根节点的位置和姿态
        root_pos = np.array(step_data["root_pos_w"])
        root_quat = np.array(step_data["root_quat_w"]) # 四元数形式 [w, x, y, z] 或 [x, y, z, w]
        
        # 构造旋转矩阵以用于坐标转换 (这里以 scipy 为例，注意四元数系数顺序)
        # scipy 的 R.from_quat 默认输入顺序为 [x, y, z, w]
        # 如果 root_quat_w 存储的是 [w, x, y, z]，需要进行相应的调整：
        q_xyzw = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        r_world_to_body = R.from_quat(q_xyzw).inv()
        
        print(f"Step {step_idx} (Time: {step_data['time']:.2f}s):")
        print(f"  IMU Acc: {step_data['base_lin_acc']}")
        for col in step_data["collisions"]:
            pos_world = np.array(col["contact_position"])
            force_world = np.array(col["contact_force_vector"])
            
            # 1. 转换位置到局部（Body）坐标系：p_local = R_inv * (p_world - p_root)
            pos_local = r_world_to_body.apply(pos_world - root_pos)
            
            # 2. 转换受力方向到局部（Body）坐标系：f_local = R_inv * f_world （注意：受力是矢量，只旋转，不平移）
            force_local = r_world_to_body.apply(force_world)
            
            print(f"  Collision on [{col['body_name']}]:")
            print(f"    Force Magnitude: {col['force_magnitude']:.2f} N")
            print(f"    Local Force Vector (N): {force_local}")
            print(f"    Local Contact Position (m): {pos_local}")
        break
```
