from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def _lin_vel_cmd_curriculum_at_max(env, command_name: str = "base_velocity", atol: float = 1.0e-4) -> bool:
    """判断速度命令课程是否已扩到 limit_ranges（lin_vel_x / lin_vel_y 均达上限）。"""
    term = env.command_manager.get_term(command_name)
    ranges = term.cfg.ranges
    limits = term.cfg.limit_ranges
    checks = (
        (ranges.lin_vel_x[0], limits.lin_vel_x[0]),
        (ranges.lin_vel_x[1], limits.lin_vel_x[1]),
        (ranges.lin_vel_y[0], limits.lin_vel_y[0]),
        (ranges.lin_vel_y[1], limits.lin_vel_y[1]),
    )
    return all(abs(cur - lim) <= atol for cur, lim in checks)


def _stash_obstacle_underground(
    env,
    env_ids: torch.Tensor,
    obstacle: RigidObject,
    stash_z_offset_m: float,
):
    """将指定 env 的障碍藏到地下并清零速度。"""
    if env_ids.numel() == 0:
        return
    device = env.device
    root_state = obstacle.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += stash_z_offset_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


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

    _stash_obstacle_underground(env, env_ids, obstacle, stash_z_offset_m)


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
    require_lin_vel_cmd_at_max: bool = True,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """到点后将障碍瞬移到速度命令方向前方（含左右随机偏移）。"""
    if (not hasattr(env, "_obstacle_spawned")) or (not hasattr(env, "_obstacle_spawn_time_s")):
        return
    if not hasattr(env, "_obstacle_spawn_cycle_idx"):
        env._obstacle_spawn_cycle_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    active_env_ids: torch.Tensor = env_ids
    if active_env_ids.numel() == 0:
        return

    obstacle: RigidObject = env.scene[asset_cfg.name]

    # 速度课程未到最大时，障碍始终藏在地下。
    if require_lin_vel_cmd_at_max and (not _lin_vel_cmd_curriculum_at_max(env, command_name=command_name)):
        _stash_obstacle_underground(env, active_env_ids, obstacle, stash_z_offset_m)
        env._obstacle_spawned[active_env_ids] = False
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
        _stash_obstacle_underground(env, changed_env_ids, obstacle, stash_z_offset_m)

    elapsed_in_cycle = elapsed_s[active_env_ids] - current_cycle.float() * cycle_dt
    to_spawn = (~env._obstacle_spawned[active_env_ids]) & (
        elapsed_in_cycle >= env._obstacle_spawn_time_s[active_env_ids]
    )
    if not torch.any(to_spawn):
        return

    spawn_env_ids = active_env_ids[to_spawn]
    robot: Articulation = env.scene[robot_cfg.name]

    # 使用速度命令（机体系 vx, vy）旋转到世界系 XY，作为障碍放置方向。
    cmd_xy_b = env.command_manager.get_command(command_name)[spawn_env_ids, :2]
    cmd_xy_norm = torch.linalg.norm(cmd_xy_b, dim=-1, keepdim=True)
    cmd_dir_b = cmd_xy_b / cmd_xy_norm.clamp(min=1.0e-6)

    yaw = robot.data.heading_w[spawn_env_ids]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cmd_dir_xy = torch.stack(
        (
            cos_yaw * cmd_dir_b[:, 0] - sin_yaw * cmd_dir_b[:, 1],
            sin_yaw * cmd_dir_b[:, 0] + cos_yaw * cmd_dir_b[:, 1],
        ),
        dim=-1,
    )

    quat_w = robot.data.root_quat_w[spawn_env_ids]
    qw, qx, qy, qz = quat_w.unbind(dim=-1)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_dir_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    fwd_dir_xy = fwd_dir_xy / torch.linalg.norm(fwd_dir_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)
    use_cmd_dir = cmd_xy_norm.squeeze(-1) >= float(min_speed_for_velocity_dir)
    dir_xy = torch.where(use_cmd_dir.unsqueeze(-1), cmd_dir_xy, fwd_dir_xy)

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
