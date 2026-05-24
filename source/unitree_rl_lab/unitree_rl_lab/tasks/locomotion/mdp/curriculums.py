from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _is_speed_curriculum_maxed(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> bool:
    """检查线速度课程是否已达到配置上限（与障碍门控口径一致）。"""
    try:
        term = env.command_manager.get_term(command_name)
        ranges = term.cfg.ranges
        limit_ranges = term.cfg.limit_ranges
    except Exception:
        return False

    eps = 1.0e-6
    return (
        abs(float(ranges.lin_vel_x[0]) - float(limit_ranges.lin_vel_x[0])) <= eps
        and abs(float(ranges.lin_vel_x[1]) - float(limit_ranges.lin_vel_x[1])) <= eps
        and abs(float(ranges.lin_vel_y[0]) - float(limit_ranges.lin_vel_y[0])) <= eps
        and abs(float(ranges.lin_vel_y[1]) - float(limit_ranges.lin_vel_y[1])) <= eps
    )


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        # 课程升级双门槛：reward 达标 + 实际速度达到当前最大指令速度的 80%。
        current_max_speed = ranges.lin_vel_x[1]
        snapshot = getattr(env, "_lin_vel_diag_snapshot", None)
        if snapshot is not None and len(env_ids) > 0:
            idx = torch.tensor(env_ids, dtype=torch.long, device=env.device)
            mean_actual_speed = snapshot["actual_speed"][idx].mean().item()
        else:
            # 未记录诊断快照时，回退为只看 reward，避免阻塞课程。
            mean_actual_speed = current_max_speed

        speed_threshold_met = mean_actual_speed >= current_max_speed * 0.8
        if reward > reward_term.weight * 0.8 and speed_threshold_met:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def apf_assist_alpha_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    threshold_ratio: float = 0.8,
    decay_step: float = 0.01,
    alpha_min: float = 0.2,
    alpha_init: float = 1.0,
) -> torch.Tensor:
    """按速度课程同口径衰减 APF 外力辅助系数 alpha。"""
    # 惰性初始化全局 alpha：按训练全局共享，不随单个 env reset 回滚。
    if not hasattr(env, "_apf_assist_alpha"):
        env._apf_assist_alpha = torch.tensor(float(alpha_init), dtype=torch.float32, device=env.device)

    # 门控：速度课程未到上限（障碍尚未开始刷出）时，不衰减 alpha。
    if not _is_speed_curriculum_maxed(env, command_name="base_velocity"):
        env.apf_assist_alpha = env._apf_assist_alpha
        return env._apf_assist_alpha

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 仅在 episode 边界按课程规则更新：达标则衰减 1%，不达标则停滞。
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * float(threshold_ratio):
            env._apf_assist_alpha = torch.clamp(
                env._apf_assist_alpha - float(decay_step),
                min=float(alpha_min),
            )

    # 便于日志与调试读取当前课程系数。
    env.apf_assist_alpha = env._apf_assist_alpha
    return env._apf_assist_alpha
