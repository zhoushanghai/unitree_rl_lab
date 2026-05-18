from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def reset_obstacle_spawn_timer_and_stash(
    env,
    env_ids: torch.Tensor,
    delay_range_s: tuple[float, float] = (1.0, 8.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    stash_z_offset_m: float = -5.0,
):
    """在 reset 阶段做两件事：
    1) 为每个 env 采样本回合唯一触发时刻（1~8s）；
    2) 先把障碍物藏到地下，避免开局立即阻挡机器人。
    """
    if env_ids.numel() == 0:
        return

    obstacle: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 懒初始化（只在首次调用时创建）：
    # - _obstacle_spawned: 该 env 在当前 episode 是否已完成“前方瞬移”
    # - _obstacle_spawn_time_s: 该 env 在当前 episode 的触发时刻（秒）
    # 这样不需要改 env 主循环，保持最小化侵入。
    if not hasattr(env, "_obstacle_spawned"):
        env._obstacle_spawned = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_obstacle_spawn_time_s"):
        env._obstacle_spawn_time_s = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # reset 后都标记为“尚未触发”。
    env._obstacle_spawned[env_ids] = False

    # 每个 episode、每个 env 独立采样触发时间 t ~ U(1, 8)。
    t_min, t_max = delay_range_s
    env._obstacle_spawn_time_s[env_ids] = torch.empty(len(env_ids), device=device).uniform_(t_min, t_max)

    # 障碍物“藏在地下”：
    # - XY 放在 env 原点
    # - Z 下沉 stash_z_offset_m（默认 -5m）
    # - 姿态置单位四元数，速度清零
    # 这样在触发前不会对训练产生碰撞干扰。
    root_state = obstacle.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += stash_z_offset_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def spawn_obstacle_forward_once(
    env,
    env_ids: torch.Tensor | None = None,
    forward_offset_m: float = 0.8,
    obstacle_half_height_m: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """到点后仅一次：将障碍物瞬移到机身“水平 +x”前方。

    关键约束：
    - 触发条件：episode_elapsed_time >= sampled_spawn_time
    - 位置规则：机身水平前方 forward_offset_m（默认 0.8m）
    - 每个 episode 每个 env 只触发一次
    """
    if (not hasattr(env, "_obstacle_spawned")) or (not hasattr(env, "_obstacle_spawn_time_s")):
        return

    # 用局部变量消除 Optional，便于静态检查。
    active_env_ids: torch.Tensor = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long) if env_ids is None else env_ids
    )
    if active_env_ids.numel() == 0:
        return

    # episode 已运行时间（秒）= step 计数 * step_dt
    elapsed_s = env.episode_length_buf.float() * env.step_dt
    to_spawn = (~env._obstacle_spawned[active_env_ids]) & (
        elapsed_s[active_env_ids] >= env._obstacle_spawn_time_s[active_env_ids]
    )
    if not torch.any(to_spawn):
        return

    spawn_env_ids = active_env_ids[to_spawn]
    obstacle: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # 从机器人根姿态四元数提取“世界系前向向量”的 XY 分量，并归一化。
    # 注意这里取的是“水平面前向”，故只用 XY，不使用 pitch/roll 的垂向影响。
    quat_w = robot.data.root_quat_w[spawn_env_ids]  # (w, x, y, z)
    qw, qx, qy, qz = quat_w.unbind(dim=-1)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    fwd_xy = fwd_xy / torch.linalg.norm(fwd_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)

    # 障碍瞬移目标位姿：
    # - XY: 机器人根位置 + 前向单位向量 * forward_offset_m
    # - Z : 贴地放置（env 地面高度 + 障碍半高）
    # - 姿态: 单位四元数
    # - 速度: 清零，避免瞬移遗留速度造成额外动力学扰动
    root_state = obstacle.data.default_root_state[spawn_env_ids].clone()
    root_state[:, 0:2] = robot.data.root_pos_w[spawn_env_ids, 0:2] + forward_offset_m * fwd_xy
    root_state[:, 2] = env.scene.env_origins[spawn_env_ids, 2] + obstacle_half_height_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(spawn_env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=spawn_env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=spawn_env_ids)

    # 标记本回合已触发，确保“只瞬移一次”。
    env._obstacle_spawned[spawn_env_ids] = True
