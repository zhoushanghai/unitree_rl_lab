# Plan: Isaac Lab Minimal Humanoid Exploration Loop

- Date: 2026-03-08
- Slug: minimal-humanoid-exploration-loop
- Status: CURRENT

## Objective

本周目标是在 `unitree_rl_lab` 里跑通一个最小闭环任务壳子：
- 房间场景
- humanoid 机器人
- 全身本体观测、区域接触和 IMU 记录
- `local_voxel_gt`
- `explored_mask`
- 一个脚本化探索器与最小数据导出链路

本周交付物必须是“可运行、可导出、可验收”的任务壳子，而不是 RL 收敛结果。

## Naming Convention

从这一版计划开始，所有新任务统一采用 `ontology_perception` 相关命名。

建议约定如下：
- Python 包路径使用 `ontology_perception`
- 脚本目录使用 `scripts/ontology_perception/`
- 环境配置文件使用 `ontology_perception_env_cfg.py`
- gym task ID 使用 `Unitree-G1-29dof-Ontology-Perception-...`

推荐首版命名：
- 包根目录：`source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/`
- 机器人场景目录：`source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/robots/g1_room/`
- 环境配置：`ontology_perception_env_cfg.py`
- 数据采集脚本：`scripts/ontology_perception/collect_scripted_rollouts.py`
- 任务 ID：`Unitree-G1-29dof-Ontology-Perception-Room-v0`

术语备注：
- 按你的要求，本计划统一使用 `ontology_perception` 命名。
- 但从英文技术术语上说，“本体感知”更常见的对应词是 `proprioception`。
- 这不影响本轮命名落地，但后续如果你们希望对外发表、开源或写论文，建议再统一核对一次术语。

## Why This Repo

结论：本周工作应放在当前扩展仓库里做，不建议直接改 Isaac Lab 本体。

依据如下：
- 当前仓库在说明里明确写成“built on top of Isaac Lab”的独立项目，并要求与 Isaac Lab 分开 clone、在同一个 Python 环境里以 editable mode 安装，这说明它的定位就是任务层扩展，不是上游核心源码位点。参考 `README.md:11-38`。
- 当前扩展在 `source/unitree_rl_lab/config/extension.toml` 中把 `isaaclab`、`isaaclab_assets`、`isaaclab_rl`、`isaaclab_tasks` 声明为依赖，而不是把这些模块复制进来维护，说明这里天然适合承载“研究任务代码”，上游继续作为依赖。参考 `source/unitree_rl_lab/config/extension.toml:17-26`。
- 现有任务已经按 Isaac Lab extension 的常规方式在当前仓库里注册成 gym task，并直接用 `ManagerBasedRLEnv` 作为入口，例如 `Unitree-G1-29dof-Velocity`。参考 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py:3-10`。
- 现有 G1 任务已经在当前仓库内使用了 `InteractiveSceneCfg`、`ContactSensorCfg` 和 `ManagerBasedRLEnvCfg`，说明你的目标并不要求修改 Isaac Lab 本体才成立。参考 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:40-76` 和 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:358-387`。
- 当前仓库已经有现成的 G1 29 自由度 locomotion 任务，并且它直接复用了当前仓库自己的 `UNITREE_G1_29DOF_CFG` 资产配置，这与“继续沿用本仓库现有模型”完全一致。参考 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py:3-10`、`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:21-22`、`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:398-460`。
- Isaac Lab 本体已经提供你这周需要的核心底座，包括 Unitree H1/G1 资产配置、IMU 传感器配置与 contact sensor 机制，所以本周目标不需要往本体里补功能。参考 `/home/hz/tiangong/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:13-18`、`/home/hz/tiangong/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:184-381`、`/home/hz/tiangong/IsaacLab/source/isaaclab/test/sensors/test_imu.py:125-185`、`/home/hz/tiangong/IsaacLab/source/isaaclab/test/sensors/test_contact_sensor.py:267-310`。

## Final Recommendation

- 主开发位点放在当前仓库 `unitree_rl_lab`。
- Isaac Lab 本体只作为上游依赖和参考实现，不作为本周主要改动位点。
- 只有在以下情况出现时，才考虑去改 `/home/hz/tiangong/IsaacLab`：
  - 发现 Isaac Lab 传感器或场景 API 缺失，扩展层无法绕开。
  - 确认是 Isaac Lab 框架级 bug，而不是任务实现问题。
  - 确认是上游资产或模拟器行为问题，且无法在扩展层 patch。

额外建议：
- 本周默认继续复用当前仓库的 `UNITREE_G1_29DOF_CFG`，不要切到 Isaac Lab 本体的 H1 或其它替代资产。
- 原因是当前仓库已有 `Unitree-G1-29dof-Velocity` 任务在用这套 G1 资产与关节/执行器定义，直接复用它最符合你“沿用这个仓库当前模型”的要求。参考 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:21-22` 和 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:398-460`。
- 需要注意的是，这套 G1 资产当前依赖 `UNITREE_ROS_DIR` 的本地路径配置，所以本周应把它视为运行前置条件，而不是本周优先整改项。参考 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:20-22` 和 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:399-401`。
- 现有脚本会自动发现名字里带 `Unitree` 的 task，因此新任务 ID 最好继续保持 `Unitree-...` 前缀，避免额外改脚本。参考 `scripts/list_envs.py:59-92` 和 `scripts/rsl_rl/train.py:20-40`。

## Environment Strategy

本周主线建议采用：
- `ManagerBasedRLEnv`
- `InteractiveSceneCfg`
- task-local scene/sensor/mdp 模块
- task-local 记录器或数据导出工具

理由：
- 这和当前仓库已有任务结构一致，复用成本最低。
- 你的最初目标就是“manager-based env + InteractiveScene + 传感接入”，与现有代码基风格完全一致。
- 本周的核心风险是数据链路而不是算法训练，因此优先选“能复用现成脚手架”的方案。

保底策略：
- 如果 `local_voxel_gt` 和 `explored_mask` 的逐步记录在 manager-based 结构里变得过于别扭，不先去改 Isaac Lab 本体。
- 第一优先是增加一个 task-local recorder / collector。
- 只有在 manager-based 原型完成后仍明显阻碍开发时，才在当前仓库内切到 `DirectRLEnv` 分支验证，不要直接把需求压到 Isaac Lab 上游。

## Recommended Robot Choice

本周默认机器人锁定为当前仓库的 G1 29 自由度模型：
- 现有 G1 locomotion 任务已经直接使用 `UNITREE_G1_29DOF_CFG`，复用路径最短。参考 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:21-22`。
- 这套配置已经把初始姿态、关节 PD、踝关节和腰部/手臂执行器都定义好了，适合作为探索任务的本体基座。参考 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:398-460`。
- 你的后续实验本来就更贴近 G1 场景，因此不再用 H1 做过渡。

本周默认不切模型，不做 H1 对齐。

## Planned Layout

建议在当前仓库新增一个独立任务包，避免把研究代码混进现有 locomotion/mimic 任务里：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/__init__.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/robots/g1_room/__init__.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/robots/g1_room/scene_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/robots/g1_room/ontology_perception_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/mdp/observations.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/mdp/rewards.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/mdp/terminations.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/utils/voxel_map.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/utils/scripted_explorer.py`
- `scripts/ontology_perception/collect_scripted_rollouts.py`

各模块职责：
- `scene_cfg.py`：房间、箱子、悬空障碍、机器人和传感器挂载。
- `ontology_perception_env_cfg.py`：manager-based task 总配置，连接 scene / observations / rewards / terminations / events。
- `mdp/observations.py`：joint/base/IMU/contact 及 recorder 所需观测项。
- `utils/voxel_map.py`：局部体素 GT 与 explored mask 生成逻辑。
- `utils/scripted_explorer.py`：脚本化 primitive 调度。
- `collect_scripted_rollouts.py`：执行短 episode 并导出 `npz` / `pt`。

## Scope

本周范围内：
- 最小房间探索场景
- humanoid 本体与接触观测
- IMU 接入
- 局部体素 GT
- explored mask
- 脚本化探索器
- 最小数据集导出与人工检查

本周不做：
- RL 探索策略训练
- HOMIE 全量移植
- sim2real 接口
- Isaac Lab 本体结构性修改

## Checklist

- ✅ ~~在 `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/` 下建立独立任务包并完成 gym 注册~~
- ✅ ~~新任务 ID 采用 `Unitree-G1-29dof-Ontology-Perception-...` 命名~~
- ✅ ~~新任务默认复用当前仓库的 `UNITREE_G1_29DOF_CFG`~~
- ✅ ~~确认 `UNITREE_ROS_DIR` 指向的 G1 URDF 资源在当前机器可用~~
- ✅ ~~搭建房间场景，包含地面、两个箱体和一个悬空障碍~~
- [ ] headless 跑通 100 次 reset，确认初始姿态稳定且无 NaN
- ✅ ~~按身体区域挂接 contact sensors，并完成 body-name 对齐~~
- ✅ ~~挂接 base IMU~~
- ✅ ~~确认 base IMU 的线加速度、角速度随动作变化~~
- ✅ ~~输出 joint/base/action/IMU/contact 的完整观测字典~~
- ✅ ~~输出 `local_voxel_gt`~~
- ✅ ~~输出 `explored_mask`~~
- ✅ ~~写完脚本化探索器与数据导出脚本~~
- [ ] 采集至少 20 到 50 个短 episode，并完成一次人工可视化检查

## Weekly Execution Plan

### Phase A: Scene Shell

目标：
- 建一个固定房间
- 放两个大箱子
- 放一个悬空横杆或悬空板
- 放一个 humanoid

实现原则：
- 第一版先固定几何体位置，先保证 reset 一致性。
- 第一版不引入复杂随机化，只保留机器人初始位姿和必要的轻量 reset。
- 第一版只关注场景正确、接触正确、传感正确。

交付物：
- `scene_cfg.py`
- 场景截图
- 100 次 reset headless 通过

验收标准：
- 机器人能稳定站立
- 障碍位置正确
- reset 后场景一致

### Phase B: Whole-body Sensing

目标观测：
- `joint_pos`
- `joint_vel`
- `action`
- `base_pose`
- `base_lin_vel`
- `base_ang_vel`
- `imu_lin_acc`
- `imu_ang_vel`
- `contacts`

接触区域建议最少覆盖：
- `left_hand`
- `right_hand`
- `left_forearm`
- `right_forearm`
- `left_shin`
- `right_shin`
- `left_foot`
- `right_foot`
- `pelvis`

实现逻辑：
- IMU 使用 Isaac Lab 已有 `ImuCfg` 模式挂在 base 或等效 IMU link 上，先得到稳定的线加速度和角速度基线。参考 `/home/hz/tiangong/IsaacLab/source/isaaclab/test/sensors/test_imu.py:125-185`。
- Contact sensor 不再只保留一个全身正则表达式采样器，而是按 body region 分拆传感器，保证区域归因清晰。
- 这里“分区域单独挂 sensor”是结合本周目标做出的工程判断，不是 Isaac Lab API 的强制限制；目的只是让后续 `explored_mask` 和接触模式分析更直接。

交付物：
- 每步都能保存结构化观测字典
- 至少一次 hand/foot/pelvis 触碰验证录像或数值记录

验收标准：
- 脚接触地面和障碍时，对应足部 / 小腿区域有响应
- 手臂横扫接触箱子时，对应手/前臂区域有响应
- IMU 不是全零，并会随动作变化

### Phase C: Privileged Local Map

目标：
- 以 base 为中心切出局部体素块
- 导出 `local_voxel_gt`
- 导出 `explored_mask`

建议首版参数：
- 范围：`2.0m x 2.0m x 2.0m`
- 分辨率：`20^3` 或 `24^3`

实现逻辑：
- `local_voxel_gt` 本周先做二值或三值都可，但必须稳定可视化。
- `explored_mask` 第一版先用保守定义：
  - 接触邻域
  - 身体最近几步 swept 过的局部空间
  - 足底稳定支撑邻域
- 如果时间不够，至少保留“接触邻域 = explored”的首版闭环。

交付物：
- 单 episode 的 voxel GT 可视化
- 单 episode 的 explored mask 可视化

验收标准：
- 箱体与悬空障碍在局部体素里位置正确
- explored mask 不会无差别刷满整图

### Phase D: Scripted Explorer

建议第一版只做 3 个 primitive：
- 前进一步并停住
- 下蹲 / 起身
- 手臂横扫

可选第 4 个 primitive：
- 轻微 pelvis-lower / sit-probe

实现逻辑：
- 本周探索器的目标不是“聪明”，而是稳定、可复现、可触发不同 body region 接触模式。
- 该脚本最好独立于训练脚本，直接面向数据采集。

交付物：
- 固定顺序 rollout
- 数据导出为 `npz` 或 `pt`

验收标准：
- 同一配置可重复触发近似一致的接触模式
- 至少采到 20 到 50 个短 episode

## End-of-Week Acceptance

到周末必须拿到下面 5 个结果：

1. 一个能运行的 Isaac Lab 自定义任务壳子。
2. 一套稳定的全身观测接口。
3. 一个每步可导出的 `local_voxel_gt` 接口。
4. 一个每步可导出的 `explored_mask` 接口。
5. 一份最小数据集，供下周做 MLP baseline 或 map predictor baseline。

## Risks And Mitigations

- 风险：当前仓库的 G1 资产依赖 `UNITREE_ROS_DIR` 本地路径，换机器时可能失效。
  - 缓解：本周先沿用当前机器可运行的路径配置，把“资产路径参数化”降级为后续整理项，不阻塞最小闭环。

- 风险：manager-based 环境里自定义 map recorder 写起来不顺。
  - 缓解：先加 task-local recorder / collector，不先改上游。

- 风险：身体 link 名称和预期 patch 名称不一致。
  - 缓解：第一天先做 body name / sensor mapping 清点，先稳定命名再挂 sensor。

- 风险：初始站姿不稳导致 reset/采集链路一直被打断。
  - 缓解：先沿用当前仓库 G1 的既有初始站姿与执行器参数，只做固定房间和轻量 reset，先拿稳定闭环。

## Success Criteria

- Must-have:
  - 场景、传感器、voxel GT、explored mask、scripted explorer 全部跑通
  - 至少一段有效数据采集结果

- Nice-to-have:
  - 支持后续从当前 G1 ontology perception task 平滑复用到 G1 mimic / locomotion 数据链路
  - 初步 observed-region 指标统计

- Stop condition:
  - 如果在 manager-based 原型里，`local_voxel_gt` 或 `explored_mask` 连一个最小 episode 都无法稳定导出，则暂停堆功能，先做 recorder 简化或局部切换到当前仓库内的 `DirectRLEnv` 原型，不进入 Isaac Lab 本体改动。

## Progress

- 2026-03-08: 初始化当前计划文档，确定本周主开发位点放在 `unitree_rl_lab`，Isaac Lab 本体保持只读依赖位。
- 2026-03-08: 已创建 `ontology_perception` 独立任务包与 `Unitree-G1-29dof-Ontology-Perception-Room-v0` 注册，补齐 `scene_cfg.py`、`ontology_perception_env_cfg.py`、`mdp/`、`voxel_map.py`、`scripted_explorer.py` 和 `collect_scripted_rollouts.py`。
- 2026-03-08: 已完成静态编译检查；当前仍缺少 headless reset、IMU 数值变化和 episode 导出这三类仿真级验收。
- 2026-03-08: 已修复 URDF merge 导致的 `imu_in_pelvis` / `pelvis_contour_link` 传感器挂载失败，并在 `unitree_rl_lab` conda 环境中真实跑通 20-step headless rollout 与 20-step headless video 导出，确认 `episode_000.npz` 与 `.mp4` 产物都已生成。

## Decisions

- D1: 本周在当前扩展仓库实现新任务，不直接改 Isaac Lab 本体。
- D2: 主线采用 `ManagerBasedRLEnv + InteractiveSceneCfg`。
- D3: 默认使用当前仓库的 `UNITREE_G1_29DOF_CFG`，不在本周切换到 Isaac Lab 本体资产。
- D4: 本周先做脚本化探索器，不做 RL 探索策略训练。

## Open Questions

- `local_voxel_gt` 首版采用二值占据还是三值 `occupied/free/unknown`？
- 数据导出格式首版统一成 `npz` 还是 `pt`？
