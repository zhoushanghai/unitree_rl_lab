"""Collect proprioception + obstacle-collision data from a trained G1 policy.

Usage (inside Docker):
    /home/hz/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/collect_data.py \\
        --task Unitree-G1-29dof-Velocity \\
        --checkpoint logs/rsl_rl/.../model_XXXXX.pt \\
        --num_episodes 100 \\
        --headless

Output:
    dataset/episode_00001.json, episode_00002.json, ...

每个 JSON 文件是一个列表，长度 = 该 episode 实际步数（最多 500 步）。
每步记录字段见 collision_data_collection.md。
"""

import os
import sys

# Force using the local rsl_rl directory instead of the prebundled package
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rsl_rl"),
)

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing — 独立定义，不调用 cli_args.add_rsl_rl_args()
# 原因：cli_args.py 中已注册 --checkpoint，重复注册会触发 ArgumentError。
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Collect collision data with a trained G1 policy.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Isaac Lab task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to run.")
parser.add_argument("--force_threshold", type=float, default=1.0, help="Minimum contact force (N) to record.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# AppLauncher 追加 --headless 等参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim FIRST，之后才能 import torch/gym 等
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下 import 必须在 AppLauncher 之后
# ---------------------------------------------------------------------------
import json
import math

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _to_list(tensor) -> list:
    """把任意形状的 tensor/ndarray 转成 Python list（JSON 可序列化）。"""
    if tensor is None:
        return []
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return list(tensor)


def collect_step_collisions(
    contact_sensor,
    env_idx: int,
    force_threshold: float,
) -> list:
    """从 obstacle_contact_forces 传感器提取当前步的碰撞信息列表。

    关键：使用 force_matrix_w（而非 net_forces_w）。
    - net_forces_w 是该 body 所有接触的合力，会包含地面接触，忽略 filter_prim_paths_expr。
    - force_matrix_w 形状 (num_envs, num_bodies, num_filtered_bodies, 3)，
      只包含与 filter_prim_paths_expr 中的物体（Obstacle）的接触力，过滤语义正确。

    Args:
        contact_sensor: obstacle_contact_forces 传感器实例。
        env_idx:        当前环境索引（单环境时为 0）。
        force_threshold: 碰撞力阈值（N），低于此值不记录。

    Returns:
        碰撞信息列表，每项 dict 包含：
            body_name, force_magnitude, contact_force_vector, contact_position
    """
    collisions = []

    # force_matrix_w: (num_envs, num_bodies, num_filtered_bodies, 3)
    # 只包含与 filter_prim_paths_expr（即 Obstacle）的接触力，不含地面等
    force_matrix_w = contact_sensor.data.force_matrix_w
    # contact_pos_w: (num_envs, num_bodies, M, 3)
    contact_pos_w = contact_sensor.data.contact_pos_w

    if force_matrix_w is None:
        return collisions

    # 对 filtered bodies 维度求和，得到该 robot body 与所有 filtered bodies 的合力
    # force_matrix_w[env, body, :, :] -> sum over filtered dim -> (num_bodies, 3)
    forces_this_env = force_matrix_w[env_idx].sum(dim=1)   # (num_bodies, 3)
    force_mags = torch.linalg.norm(forces_this_env, dim=-1)  # (num_bodies,)

    # 找出超过阈值的 body 索引
    hit_body_ids = (force_mags >= force_threshold).nonzero(as_tuple=True)[0]
    if hit_body_ids.numel() == 0:
        return collisions

    body_names = contact_sensor.body_names  # list[str], 长度 = num_bodies

    for b_idx in hit_body_ids:
        b_idx = int(b_idx.item())
        force_vec = forces_this_env[b_idx]  # (3,)
        mag = float(force_mags[b_idx].item())

        # 碰撞点坐标：取第一个接触点（索引 0）
        contact_pos = None
        if contact_pos_w is not None:
            pos_candidate = contact_pos_w[env_idx, b_idx, 0]  # (3,)
            if torch.isfinite(pos_candidate).all():
                contact_pos = _to_list(pos_candidate)

        collisions.append({
            "body_name": body_names[b_idx],
            "force_magnitude": round(mag, 4),
            "contact_force_vector": _to_list(force_vec),
            "contact_position": contact_pos if contact_pos is not None else [],
        })

    return collisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. 构造环境配置
    # 与训练保持完全一致：障碍出现逻辑不做任何修改。
    # 唯一的改动：
    #   - 多并行环境（由 args_cli.num_envs 指定）
    #   - 速度命令使用最大范围（limit_ranges），让 _is_speed_curriculum_maxed() 立即满足
    #   - resampling_time = episode_length = 10s，保证命令在整个 episode 内恒定
    # ------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 10.0
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)

    # 强制在数据采集时每次都生成障碍，不再需要“速度指令课程达到最大”的前置条件
    env_cfg.events.spawn_obstacle_forward_once.params["require_lin_vel_cmd_at_max"] = False

    # 修改障碍物生成时刻与偏移范围：2-4s生成，左右偏移 0.2m
    env_cfg.events.reset_obstacle_spawn.params["delay_range_s"] = (2.0, 4.0)
    env_cfg.events.spawn_obstacle_forward_once.params["delay_range_s"] = (2.0, 4.0)
    env_cfg.events.spawn_obstacle_forward_once.params["lateral_offset_range_m"] = (-0.2, 0.2)

    # 解决 omni.physx 在多环境下对 filter_prim_paths_expr 的广播/维度匹配问题：
    # PhysX 要求过滤路径的匹配项总数要么为 1 (广播到全部)，要么与 (num_envs * num_bodies) 完全一致（进行 1对1 配对）。
    # 当 num_envs > 1 时，`{ENV_REGEX_NS}/Obstacle` 会被解析为 num_envs 个匹配项（例如4个），与 128 个 body 既不为 1 也不相等，会导致报错。
    # 我们根据环境数量，动态为每个环境下的每一个 tracked body (G1 共有 32 个 body) 独立生成对应的障碍物过滤路径。
    num_bodies = 32
    filter_exprs = []
    for env_idx in range(args_cli.num_envs):
        filter_exprs.extend([f"/World/envs/env_{env_idx}/Obstacle"] * num_bodies)
    env_cfg.scene.obstacle_contact_forces.filter_prim_paths_expr = filter_exprs

    # ------------------------------------------------------------------
    # 2. 加载 agent 配置（只需要能构造 runner，不依赖 cli_args）
    # ------------------------------------------------------------------
    import importlib.metadata as meta_pkg
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = meta_pkg.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # ------------------------------------------------------------------
    # 3. 创建环境
    # ------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # 4. 加载 checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------------------------------------------
    # 5. 取得传感器引用
    # ------------------------------------------------------------------
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]

    # obstacle_contact_forces：只统计与 Obstacle 的接触，track_contact_points=True
    if "obstacle_contact_forces" not in raw_env.scene.sensors:
        raise RuntimeError(
            "Sensor 'obstacle_contact_forces' not found in scene. "
            "Please add it to RobotSceneCfg in velocity_env_cfg.py."
        )
    contact_sensor = raw_env.scene.sensors["obstacle_contact_forces"]

    # ------------------------------------------------------------------
    # 6. 数据采集循环
    # ------------------------------------------------------------------
    os.makedirs("dataset", exist_ok=True)
    force_threshold = args_cli.force_threshold
    num_envs = args_cli.num_envs

    # 用来存储每个环境正在采集的当前 episode 数据
    active_episodes = [[] for _ in range(num_envs)]
    saved_episodes_count = 0

    def _get_obs():
        """兼容 rsl-rl 各版本：get_observations() 可能返回 obs 或 (obs, extras)。"""
        result = env.get_observations()
        if isinstance(result, tuple):
            return result[0]
        return result

    obs = _get_obs()

    print(f"[INFO] Collecting {args_cli.num_episodes} episodes using {num_envs} environments (force threshold: {force_threshold} N) ...")

    # 每步决策步数 = decimation；物理 dt = sim.dt；step_dt = decimation * sim.dt
    step_dt = raw_env.step_dt
    cmd_term = raw_env.command_manager.get_term("base_velocity")

    while saved_episodes_count < args_cli.num_episodes:
        # --- 推理 ---
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        # 遍历所有并行环境，读取其状态
        for env_idx in range(num_envs):
            sim_time = float(raw_env.episode_length_buf[env_idx].item()) * step_dt

            # 速度命令
            command = _to_list(cmd_term.vel_command_b[env_idx])  # [vx, vy, wz]

            # 本体感知
            base_ang_vel   = _to_list(robot.data.root_ang_vel_b[env_idx])
            projected_grav = _to_list(robot.data.projected_gravity_b[env_idx])
            joint_pos      = _to_list(robot.data.joint_pos[env_idx])
            joint_vel      = _to_list(robot.data.joint_vel[env_idx])
            last_action    = _to_list(raw_env.action_manager.action[env_idx])

            # 特权信息
            root_pos_w   = _to_list(robot.data.root_pos_w[env_idx])
            root_quat_w  = _to_list(robot.data.root_quat_w[env_idx])  # [w, x, y, z]
            base_lin_vel = _to_list(robot.data.root_lin_vel_b[env_idx])
            joint_torques = _to_list(robot.data.applied_torque[env_idx])

            # 碰撞信息
            collisions = collect_step_collisions(contact_sensor, env_idx, force_threshold)

            step_data = {
                "time":              round(sim_time, 4),
                "command":           command,
                "base_ang_vel":      base_ang_vel,
                "projected_gravity": projected_grav,
                "joint_pos":         joint_pos,
                "joint_vel":         joint_vel,
                "last_action":       last_action,
                "root_pos_w":        root_pos_w,
                "root_quat_w":       root_quat_w,
                "base_lin_vel":      base_lin_vel,
                "joint_torques":     joint_torques,
                "collisions":        collisions,
            }

            # 如果当前环境发生了 Done (重置)
            if dones[env_idx]:
                # 如果这个环境之前有数据，说明它刚刚完成了一个 episode，保存它！
                if len(active_episodes[env_idx]) > 0:
                    saved_episodes_count += 1
                    if saved_episodes_count <= args_cli.num_episodes:
                        filename = f"dataset/episode_{saved_episodes_count:05d}.json"
                        with open(filename, "w") as f:
                            json.dump(active_episodes[env_idx], f, indent=2)

                        n_collision_steps = sum(1 for s in active_episodes[env_idx] if s["collisions"])
                        print(
                            f"[INFO] Episode {saved_episodes_count:4d}/{args_cli.num_episodes} | "
                            f"env={env_idx:2d} | steps={len(active_episodes[env_idx])} | "
                            f"collision_steps={n_collision_steps} | saved → {filename}"
                        )
                # 清空该环境的缓存，并开始新的一轮记录（首步为当前重置后的状态）
                active_episodes[env_idx] = [step_data]
            else:
                # 正常步，直接追加
                active_episodes[env_idx].append(step_data)

        # 再次检查是否完成了所有的采集任务（防止在多环境同时 done 时溢出）
        if saved_episodes_count >= args_cli.num_episodes:
            break

    print(f"[INFO] Done. {saved_episodes_count} episodes saved to dataset/")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
