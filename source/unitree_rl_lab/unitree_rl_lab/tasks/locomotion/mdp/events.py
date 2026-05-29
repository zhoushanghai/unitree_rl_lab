from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def reset_obstacle_spawn_timer_and_stash(
    env,
    env_ids: torch.Tensor,
    delay_range_s: tuple[float, float] = (1.0, 4.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    stash_z_offset_m: float = -5.0,
):
    """reset 时采样障碍触发延时，并把障碍藏到地下。"""
    if env_ids.numel() == 0:
        return

    obstacle: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    device = env.device

    if not hasattr(env, "_obstacle_spawned"):
        env._obstacle_spawned = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_obstacle_spawn_time_s"):
        env._obstacle_spawn_time_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
    if not hasattr(env, "_obstacle_spawn_cycle_idx"):
        env._obstacle_spawn_cycle_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    env._obstacle_spawned[env_ids] = False
    env._obstacle_spawn_cycle_idx[env_ids] = 0

    t_min, t_max = delay_range_s
    env._obstacle_spawn_time_s[env_ids] = torch.empty(len(env_ids), device=device).uniform_(t_min, t_max)

    root_state = obstacle.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += stash_z_offset_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def spawn_obstacle_forward_once(
    env,
    env_ids: torch.Tensor,
    forward_offset_m: float = 0.8,
    lateral_offset_range_m: tuple[float, float] = (-0.2, 0.2),
    min_speed_for_velocity_dir: float = 0.05,
    obstacle_half_height_m: float = 0.5,
    delay_range_s: tuple[float, float] = (1.0, 4.0),
    command_refresh_interval_s: float = 10.0,
    stash_z_offset_m: float = -5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """到点后将障碍瞬移到速度方向前方（含左右随机偏移）。"""
    if (not hasattr(env, "_obstacle_spawned")) or (not hasattr(env, "_obstacle_spawn_time_s")):
        return
    if not hasattr(env, "_obstacle_spawn_cycle_idx"):
        env._obstacle_spawn_cycle_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    active_env_ids: torch.Tensor = env_ids
    if active_env_ids.numel() == 0:
        return

    elapsed_s = env.episode_length_buf.float() * env.step_dt

    # 与速度命令重采样周期对齐：每周期重采样延时并藏障碍，到点再瞬移一次。
    cycle_dt = max(float(command_refresh_interval_s), 1.0e-6)
    cycle_idx_all = torch.floor(elapsed_s / cycle_dt).long()
    current_cycle = cycle_idx_all[active_env_ids]
    cycle_changed = current_cycle != env._obstacle_spawn_cycle_idx[active_env_ids]
    if torch.any(cycle_changed):
        changed_env_ids = active_env_ids[cycle_changed]
        env._obstacle_spawn_cycle_idx[changed_env_ids] = cycle_idx_all[changed_env_ids]
        env._obstacle_spawned[changed_env_ids] = False

        t_min, t_max = delay_range_s
        env._obstacle_spawn_time_s[changed_env_ids] = torch.empty(len(changed_env_ids), device=env.device).uniform_(
            t_min, t_max
        )
        obstacle: RigidObject = env.scene[asset_cfg.name]
        stash_state = obstacle.data.default_root_state[changed_env_ids].clone()
        stash_state[:, 0:3] = env.scene.env_origins[changed_env_ids]
        stash_state[:, 2] += stash_z_offset_m
        stash_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(changed_env_ids), 1)
        stash_state[:, 7:13] = 0.0
        obstacle.write_root_pose_to_sim(stash_state[:, 0:7], env_ids=changed_env_ids)
        obstacle.write_root_velocity_to_sim(stash_state[:, 7:13], env_ids=changed_env_ids)

    elapsed_in_cycle = elapsed_s[active_env_ids] - current_cycle.float() * cycle_dt
    to_spawn = (~env._obstacle_spawned[active_env_ids]) & (
        elapsed_in_cycle >= env._obstacle_spawn_time_s[active_env_ids]
    )
    if not torch.any(to_spawn):
        return

    spawn_env_ids = active_env_ids[to_spawn]
    obstacle: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    vel_xy = robot.data.root_lin_vel_w[spawn_env_ids, :2]
    vel_xy_norm = torch.linalg.norm(vel_xy, dim=-1, keepdim=True)
    vel_dir_xy = vel_xy / vel_xy_norm.clamp(min=1.0e-6)

    quat_w = robot.data.root_quat_w[spawn_env_ids]
    qw, qx, qy, qz = quat_w.unbind(dim=-1)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_dir_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    fwd_dir_xy = fwd_dir_xy / torch.linalg.norm(fwd_dir_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)
    use_vel_dir = vel_xy_norm.squeeze(-1) >= float(min_speed_for_velocity_dir)
    dir_xy = torch.where(use_vel_dir.unsqueeze(-1), vel_dir_xy, fwd_dir_xy)

    lat_dir_xy = torch.stack((-dir_xy[:, 1], dir_xy[:, 0]), dim=-1)
    lat_dir_xy = lat_dir_xy / torch.linalg.norm(lat_dir_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)
    lat_min, lat_max = float(lateral_offset_range_m[0]), float(lateral_offset_range_m[1])
    lat_offset = torch.empty(len(spawn_env_ids), device=env.device).uniform_(lat_min, lat_max)

    root_state = obstacle.data.default_root_state[spawn_env_ids].clone()
    root_state[:, 0:2] = (
        robot.data.root_pos_w[spawn_env_ids, 0:2]
        + forward_offset_m * dir_xy
        + lat_offset.unsqueeze(-1) * lat_dir_xy
    )
    root_state[:, 2] = env.scene.env_origins[spawn_env_ids, 2] + obstacle_half_height_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(spawn_env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=spawn_env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=spawn_env_ids)

    env._obstacle_spawned[spawn_env_ids] = True
