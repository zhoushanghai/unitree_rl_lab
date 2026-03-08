# DEV_LOG

## 关键变更

### 2026-03-08 22:52 CST

- Action: 创建当前计划文档 `docs/@plan_minimal-humanoid-exploration-loop_2026-03-08.md`，用于承载本周 Isaac Lab 最小闭环任务方案。
- Details: 初版方案判断本周应在当前扩展仓库 `unitree_rl_lab` 实现新任务，Isaac Lab 本体只作为依赖与参考，不作为主要改动位点。该初版资产建议已在 22:58 的后续记录中根据用户决策更新为“沿用当前仓库 G1 模型”。
- Execution Record: 文档规划任务，无需执行训练或推理命令。

### 2026-03-08 22:58 CST

- Action: 更新当前计划文档，切换为“沿用当前仓库 G1 模型”的实施路线。
- Details: 根据用户新决策，方案不再建议使用 Isaac Lab 本体的 H1/H1 minimal 资产，改为直接复用当前仓库 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中的 `UNITREE_G1_29DOF_CFG`，并以已有 `Unitree-G1-29dof-Velocity` 任务结构作为 exploration task 的参考模板。
- Execution Record: 文档规划任务，无需执行训练或推理命令。

### 2026-03-08 23:02 CST

- Action: 更新当前计划文档中的新任务命名规范。
- Details: 根据用户要求，后续新任务统一采用 `ontology_perception` 相关命名，计划文档中的包路径、脚本路径、环境配置文件名和建议 gym task ID 已同步切换到 `ontology_perception` 命名体系。
- Execution Record: 文档规划任务，无需执行训练或推理命令。

### 2026-03-08 23:36 CST

- Action: 实现 G1 `ontology_perception` 最小闭环任务骨架，并接入场景、传感器、体素 GT、explored mask 与脚本化采集链路。
- Details: 新增 `source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception/` 任务包，注册 `Unitree-G1-29dof-Ontology-Perception-Room-v0`；在 `robots/g1_room/scene_cfg.py` 中复用当前仓库 `UNITREE_G1_29DOF_CFG`，搭建房间、两个箱体、悬空障碍、base IMU 和分区域 contact sensors；在 `ontology_perception_env_cfg.py` 中实现 manager-based 配置与 `policy / critic / collector` 观测组；在 `utils/voxel_map.py` 中实现 `local_voxel_gt` 与 `explored_mask`；在 `utils/scripted_explorer.py` 与 `scripts/ontology_perception/collect_scripted_rollouts.py` 中实现固定 primitive rollout 和数据导出；同时更新 `scripts/list_envs.py` 以纳入 `ontology_perception.robots` 的任务发现。
- Execution Record: `python -m compileall source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception scripts/ontology_perception scripts/list_envs.py`

### 2026-03-08 23:47 CST

- Action: 为 ontology perception 采集脚本补充可视化录像支持，并整理首批运行命令。
- Details: 更新 `scripts/ontology_perception/collect_scripted_rollouts.py`，新增 `--video` 与 `--video_length` 参数；录像文件名采用 `{TaskName}_{Timestamp}` 形式，输出到 `outputs/ontology_perception/.../videos/`，便于直接查看 scripted rollout 的效果。
- Execution Record: `./unitree_rl_lab.sh -l`
- Execution Record: `python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 420 --video --video_length 420`
- Execution Record: `python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 420 --video --video_length 420 --headless`

### 2026-03-08 23:50 CST

- Action: 修复 ontology perception 任务的运行期传感器绑定错误，并完成真实 rollout 验证。
- Details: `scene_cfg.py` 中的 `imu_in_pelvis` 和 `pelvis_contour_link` 会在当前 G1 URDF 导入时被 merge 到 `pelvis`，导致 IMU 和 pelvis contact 的 prim 路径不存在；现已改为直接挂到 `pelvis`。同时修复 `voxel_map.py` 中 `torch.clamp` 对 tensor 上界的兼容性问题。之后使用 `unitree_rl_lab` conda 环境实际跑通了 headless rollout 与 headless video 导出，生成了 `episode_000.npz` 和 `.mp4` 文件；抽样检查表明 `imu_lin_acc`、`imu_ang_vel` 非零且有变化，`explored_mask` 也已产生非空区域。
- Execution Record: `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --headless`
- Execution Record: `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --video --video_length 20 --headless`

### 2026-03-08 23:43 CST

- Action: 验证 ontology perception 任务的非 headless GUI 模式。
- Details: 使用当前 `unitree_rl_lab` conda 环境实际拉起了 Isaac Sim 窗口版 `collect_scripted_rollouts.py`，单环境、单 episode、120 step 能正常启动、进入场景并自动退出，没有新的运行期异常。
- Execution Record: `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 120`

### 2026-03-08 23:48 CST

- Action: 为 GUI 观察场景补充 `--keep_alive` 模式，避免 rollout 结束后立刻关窗。
- Details: 更新 `scripts/ontology_perception/collect_scripted_rollouts.py`，新增 `--keep_alive` 参数；在非 headless 模式下，rollout 和数据导出完成后不再立即 `env.close()`，而是持续渲染当前场景，方便手动拖动视角检查环境。关闭窗口或 `Ctrl+C` 后才退出。
- Execution Record: `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --keep_alive`

## 关键命令

- `python -m compileall source/unitree_rl_lab/unitree_rl_lab/tasks/ontology_perception scripts/ontology_perception scripts/list_envs.py`
- `./unitree_rl_lab.sh -l`
- `python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 420 --video --video_length 420`
- `python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 420 --video --video_length 420 --headless`
- `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --headless`
- `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --video --video_length 20 --headless`
- `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 120`
- `/home/hz/miniconda3/envs/unitree_rl_lab/bin/python scripts/ontology_perception/collect_scripted_rollouts.py --task Unitree-G1-29dof-Ontology-Perception-Room-v0 --num_envs 1 --episodes 1 --steps_per_episode 20 --keep_alive`
- **[2026-03-08]** `git commit`: feat(env): 新增本体感知闭环任务 / add ontology perception task

## 排错记录

- 2026-03-08: `python scripts/list_envs.py` 在当前 shell 下失败，报错 `ModuleNotFoundError: No module named 'gymnasium'`。结论：当前终端解释器不是项目可运行的 Isaac Lab Python 环境；本轮仅完成静态编译校验，未做真实任务导入与仿真验收。
- 2026-03-08: `collect_scripted_rollouts.py` 初次运行报错 `Failed to find a prim at path expression: /World/envs/env_.*/Robot/imu_in_pelvis`。根因是当前 G1 URDF 导入时 `imu_in_pelvis` 和 `pelvis_contour_link` 都被 merge 到 `pelvis`。修复方法：将 IMU 与 pelvis contact sensor 的 `prim_path` 都切换到 `.../Robot/pelvis`。
- 2026-03-08: `collect_scripted_rollouts.py` 第二次运行报错 `torch.clamp()` 参数组合不合法。根因是 `explored_mask` 里对 tensor 上界使用了 `min=int, max=tensor` 的混合写法。修复方法：改成 `torch.maximum + torch.minimum` 的显式张量裁剪。
