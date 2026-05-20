from __future__ import annotations

import os
import csv
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        if reward > reward_term.weight * 0.8:
            old_lin_vel_x = list(ranges.lin_vel_x)
            old_lin_vel_y = list(ranges.lin_vel_y)

            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            new_lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            new_lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

            log_dir = getattr(env, "log_dir", None)
            if log_dir is not None and len(env_ids) > 0:
                os.makedirs(log_dir, exist_ok=True)
                csv_path = os.path.join(log_dir, "curriculum_upgrade_vel_track.csv")
                write_header = not os.path.exists(csv_path)

                snapshot = getattr(env, "_lin_vel_diag_snapshot", None)
                learning_iteration = getattr(env, "current_learning_iteration", -1)

                if snapshot is not None:
                    idx = torch.tensor(env_ids, dtype=torch.long, device=env.device)
                    raw_speed_envs = snapshot["raw_speed"][idx].cpu().tolist()
                    apf_speed_envs = snapshot["apf_speed"][idx].cpu().tolist()
                    actual_speed_envs = snapshot["actual_speed"][idx].cpu().tolist()
                    speed_error_envs = snapshot["speed_error"][idx].cpu().tolist()
                    tracking_reward_envs = snapshot["tracking_reward"][idx].cpu().tolist()
                    ep_rewards_envs = (env.reward_manager._episode_sums[reward_term_name][idx] / env.max_episode_length_s).cpu().tolist()

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow([
                                "iteration",
                                "env_id",
                                "cmd_speed_raw",
                                "cmd_speed_apf",
                                "actual_speed",
                                "speed_error",
                                "tracking_reward",
                                "episode_mean_reward",
                                "old_range_x_min", "old_range_x_max",
                                "old_range_y_min", "old_range_y_max",
                                "new_range_x_min", "new_range_x_max",
                                "new_range_y_min", "new_range_y_max"
                            ])
                        for i, env_id_val in enumerate(env_ids):
                            writer.writerow([
                                learning_iteration,
                                int(env_id_val),
                                raw_speed_envs[i],
                                apf_speed_envs[i],
                                actual_speed_envs[i],
                                speed_error_envs[i],
                                tracking_reward_envs[i],
                                ep_rewards_envs[i],
                                old_lin_vel_x[0], old_lin_vel_x[1],
                                old_lin_vel_y[0], old_lin_vel_y[1],
                                new_lin_vel_x[0], new_lin_vel_x[1],
                                new_lin_vel_y[0], new_lin_vel_y[1]
                            ])

            ranges.lin_vel_x = new_lin_vel_x
            ranges.lin_vel_y = new_lin_vel_y

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
