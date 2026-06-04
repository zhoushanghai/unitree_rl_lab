"""Collect proprioception + obstacle-collision data from a trained G1 policy.

Usage (inside Docker):
    python scripts/rsl_rl/collect_data.py \\
        --task Unitree-G1-29dof-Velocity \\
        --checkpoint logs/rsl_rl/.../model_XXXXX.pt \\
        --num_episodes 100 \\
        --headless

    # 多卡：每张 GPU 起一个独立 Isaac 进程，episode 编号预先划分，避免写盘冲突
    python scripts/rsl_rl/collect_data.py ... --gpu_ids 0,1 --num_envs 64 --headless

Output:
    dataset/episode_00001.npz, episode_00002.npz, ...
    若 dataset/ 已有文件，从最大编号续接；每次运行再写入 --num_episodes 条新数据。

每个 NPZ 文件保存为一个压缩的字典，其中数值类型保存为 float32 数组，碰撞列表序列化为 JSON 字符串保存。
每步记录字段见 collision_data_collection.md。
"""

import argparse
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Force using the local rsl_rl directory instead of the prebundled package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "rsl_rl"))
sys.path.insert(0, _SCRIPT_DIR)

# 须在 AppLauncher 之前导入（仅 numpy，无 Isaac 依赖）
from voxel_collision_utils import DEFAULT_MIN_CMD_SPEED  # noqa: E402


def _existing_episode_count(dataset_dir: str) -> int:
    """扫描 dataset 目录中 episode_XXXXX.npz，返回已有最大编号（无文件则为 0）。"""
    if not os.path.isdir(dataset_dir):
        return 0
    max_idx = 0
    for name in os.listdir(dataset_dir):
        if not (name.startswith("episode_") and name.endswith(".npz")):
            continue
        try:
            max_idx = max(max_idx, int(name[len("episode_") : -len(".npz")]))
        except ValueError:
            continue
    return max_idx


def _parse_gpu_ids(gpu_ids: str) -> list[int]:
    ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
    if not ids:
        raise ValueError("--gpu_ids 不能为空")
    return ids


def _split_episode_quota(total: int, num_workers: int) -> list[int]:
    """将 total 条 episode 均分到 num_workers（余数给前几个 worker）。"""
    base, rem = divmod(total, num_workers)
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


def _filter_argv(argv: list[str], remove_flags: set[str]) -> list[str]:
    """从 argv 中去掉指定 flag 及其参数值。"""
    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in remove_flags:
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            continue
        if any(arg.startswith(f"{f}=") for f in remove_flags):
            continue
        out.append(arg)
    return out


def _launch_multi_gpu_workers(args_cli) -> None:
    """父进程：按 GPU 划 episode 区间，子进程各绑一张卡（CUDA_VISIBLE_DEVICES）。"""
    gpu_ids = _parse_gpu_ids(args_cli.gpu_ids)
    quotas = _split_episode_quota(args_cli.num_episodes, len(gpu_ids))
    dataset_dir = "dataset"
    ep_base = _existing_episode_count(dataset_dir)

    # 去掉 --gpu_ids，子进程以单卡 worker 方式启动
    worker_argv = _filter_argv(sys.argv[1:], {"--gpu_ids"})
    script = os.path.abspath(sys.argv[0])
    procs: list[subprocess.Popen] = []
    next_ep = ep_base

    print(
        f"[INFO] Multi-GPU collect: gpus={gpu_ids}, total_new_episodes={args_cli.num_episodes}, "
        f"quotas={quotas}, resume_after=episode_{ep_base:05d}"
    )

    for worker_i, (gpu_id, quota) in enumerate(zip(gpu_ids, quotas)):
        if quota <= 0:
            continue
        episode_start = next_ep + 1
        cmd = [
            sys.executable,
            script,
            *worker_argv,
            "--episode_start",
            str(episode_start),
            "--num_episodes",
            str(quota),
            "--_worker_gpu",
            str(worker_i),
            "--device",
            "cuda:0",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[INFO] Spawn worker {worker_i}: physical_gpu={gpu_id}, "
            f"episodes {episode_start:05d}..{episode_start + quota - 1:05d}, "
            f"num_envs={args_cli.num_envs}"
        )
        procs.append(subprocess.Popen(cmd, env=env))
        next_ep += quota

    failed = 0
    for worker_i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            print(f"[ERROR] Worker {worker_i} exited with code {rc}")
            failed += 1
    if failed:
        raise SystemExit(1)
    print(f"[INFO] All {len(procs)} GPU workers finished.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect collision data with a trained G1 policy.")
    parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Isaac Lab task name.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to run.")
    parser.add_argument("--force_threshold", type=float, default=1.0, help="Minimum contact force (N) to record.")
    parser.add_argument(
        "--min-cmd-speed",
        type=float,
        default=DEFAULT_MIN_CMD_SPEED,
        help="Episode 开始时若 sqrt(vx^2+vy^2) 低于此值则重采样速度命令（与体素后处理 --min-cmd-speed 一致）。",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="逗号分隔物理 GPU 编号，如 0,1。将启动多进程，每卡一个 Isaac 实例；"
        "episode 编号在启动时划分，避免写盘冲突。",
    )
    parser.add_argument(
        "--episode_start",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker_gpu",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    return parser


# ---------------------------------------------------------------------------
# Argument parsing — 须在 AppLauncher 之前；多卡时父进程直接 spawn 后退出
# ---------------------------------------------------------------------------
parser = _build_arg_parser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.gpu_ids and args_cli._worker_gpu is None:
    _gpu_list = _parse_gpu_ids(args_cli.gpu_ids)
    if len(_gpu_list) == 1:
        # 单卡：只绑定可见 GPU，不额外 spawn 子进程
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_list[0])
    else:
        _launch_multi_gpu_workers(args_cli)
        raise SystemExit(0)

# Launch Isaac Sim FIRST，之后才能 import torch/gym 等
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下 import 必须在 AppLauncher 之后
# ---------------------------------------------------------------------------
import json
import math

import gymnasium as gym
import numpy as np
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

import importlib

from collision_contact_utils import resolve_contact_position_from_contact_pos_w

_obstacle_sensors_mod = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_contact_sensors"
)
G1_OBSTACLE_CONTACT_BODY_NAMES = _obstacle_sensors_mod.G1_OBSTACLE_CONTACT_BODY_NAMES
obstacle_contact_sensor_name = _obstacle_sensors_mod.obstacle_contact_sensor_name


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
    obstacle_sensors: dict,
    robot,
    env_idx: int,
    force_threshold: float,
    body_name_to_idx: dict[str, int],
) -> list:
    """遍历各连杆独立 obstacle 传感器，汇总当前步碰撞（filter 维恒为 0）。"""
    collisions = []
    body_pos_w = robot.data.body_pos_w[env_idx]
    filter_idx = 0  # 每传感器仅 filter=["{ENV_REGEX_NS}/Obstacle"]

    for link_name, sensor in obstacle_sensors.items():
        force_matrix_w = sensor.data.force_matrix_w
        if force_matrix_w is None:
            continue

        force_vec = force_matrix_w[env_idx, 0, filter_idx]
        mag = float(torch.linalg.norm(force_vec).item())
        if mag < force_threshold:
            continue

        contact_pos: list = []
        pos_source = "invalid"
        contact_pos_w = sensor.data.contact_pos_w
        if contact_pos_w is not None:
            b_idx = body_name_to_idx[link_name]
            contact_pos, pos_source = resolve_contact_position_from_contact_pos_w(
                body_pos_w[b_idx], contact_pos_w[env_idx, 0, filter_idx]
            )

        collisions.append({
            "body_name": link_name,
            "force_magnitude": round(mag, 4),
            "contact_force_vector": _to_list(force_vec),
            "contact_position": contact_pos,
            "contact_position_source": pos_source,
        })

    return collisions


def ensure_min_command_xy_speed(cmd_term, env_ids: list[int], min_speed: float, max_tries: int = 128) -> None:
    """对每个 env 重复采样速度命令，直到平面线速度模长 >= min_speed。

    在 episode 开始时调用即可：collect 已将 resampling_time 对齐 episode 长度，整条轨迹 command 恒定。
    """
    if not env_ids:
        return
    for _ in range(max_tries):
        vel_xy = cmd_term.vel_command_b[env_ids, :2]
        slow_mask = torch.linalg.norm(vel_xy, dim=-1) < min_speed
        if not bool(slow_mask.any().item()):
            return
        slow_ids = [env_ids[i] for i in range(len(env_ids)) if slow_mask[i].item()]
        cmd_term._resample_command(slow_ids)
        cmd_term._update_command()
    vel_xy = cmd_term.vel_command_b[env_ids, :2]
    still_slow = torch.linalg.norm(vel_xy, dim=-1) < min_speed
    if bool(still_slow.any().item()):
        slow_list = [env_ids[i] for i in range(len(env_ids)) if still_slow[i].item()]
        print(
            f"[WARN] env {slow_list} 在 {max_tries} 次重采样后线速度仍 < {min_speed} m/s，"
            "将按当前命令继续采集。"
        )


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
    # 采集不接受“站立”零速命令，避免与 min-cmd-speed 重采样冲突
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # 强制在数据采集时每次都生成障碍，不再需要“速度指令课程达到最大”的前置条件
    env_cfg.events.spawn_obstacle_forward_once.params["require_lin_vel_cmd_at_max"] = False

    # 修改障碍物生成时刻与偏移范围：2-4s生成，左右偏移 0.2m
    env_cfg.events.reset_obstacle_spawn.params["delay_range_s"] = (2.0, 4.0)
    env_cfg.events.spawn_obstacle_forward_once.params["delay_range_s"] = (2.0, 4.0)
    env_cfg.events.spawn_obstacle_forward_once.params["lateral_offset_range_m"] = (-0.2, 0.2)

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

    # 各连杆独立 obstacle 传感器（obstacle_contact_<link>）
    obstacle_sensors = {}
    for link_name in G1_OBSTACLE_CONTACT_BODY_NAMES:
        key = obstacle_contact_sensor_name(link_name)
        if key not in raw_env.scene.sensors:
            raise RuntimeError(
                f"Sensor '{key}' not found. "
                "请在 velocity_env_cfg.RobotSceneCfg 中注册 per-link obstacle 接触传感器。"
            )
        obstacle_sensors[link_name] = raw_env.scene.sensors[key]

    num_envs = args_cli.num_envs
    body_name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    missing = [ln for ln in G1_OBSTACLE_CONTACT_BODY_NAMES if ln not in body_name_to_idx]
    if missing:
        raise RuntimeError(f"Robot 缺少连杆名（与传感器配置不一致）: {missing[:5]}...")

    n_sensors = len(obstacle_sensors)
    sample = next(iter(obstacle_sensors.values()))
    filter_count = sample.data.force_matrix_w.shape[2] if sample.data.force_matrix_w is not None else 0
    print(
        f"[INFO] Per-link obstacle sensors: {n_sensors}, num_envs={num_envs}, "
        f"filter_count={filter_count} (期望 1，即仅 Obstacle)"
    )
    if sample.data.contact_pos_w is None:
        print("[WARN] contact_pos_w is None; 将无法写入真实接触点坐标。")
    print("[INFO] contact_position 仅来自 contact_pos_w（连杆-障碍物接触点平均），无效时留空。")

    # ------------------------------------------------------------------
    # 6. 数据采集循环
    # ------------------------------------------------------------------
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    force_threshold = args_cli.force_threshold

    # episode 编号：多卡 worker 用父进程划好的区间；单卡则扫描 dataset 续接
    if args_cli.episode_start is not None:
        existing_count = args_cli.episode_start - 1
        target_count = existing_count + args_cli.num_episodes
        worker_tag = (
            f"worker={args_cli._worker_gpu} cuda={args_cli.device} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})"
        )
        print(
            f"[INFO] GPU worker [{worker_tag}]: episodes "
            f"{args_cli.episode_start:05d}..{target_count:05d} ({args_cli.num_episodes} new)"
        )
    else:
        existing_count = _existing_episode_count(dataset_dir)
        target_count = existing_count + args_cli.num_episodes

    # 用来存储每个环境正在采集的当前 episode 数据
    active_episodes = [[] for _ in range(num_envs)]
    saved_episodes_count = existing_count

    def _get_obs():
        """兼容 rsl-rl 各版本：get_observations() 可能返回 obs 或 (obs, extras)。"""
        result = env.get_observations()
        if isinstance(result, tuple):
            return result[0]
        return result

    obs = _get_obs()

    if args_cli.episode_start is None:
        if existing_count > 0:
            print(
                f"[INFO] Found {existing_count} episodes in {dataset_dir}/, "
                f"appending {args_cli.num_episodes} more → episode_{existing_count + 1:05d}.npz ..."
            )
        else:
            print(
                f"[INFO] Collecting {args_cli.num_episodes} episodes using {num_envs} environments "
                f"(force threshold: {force_threshold} N) ..."
            )

    # 每步决策步数 = decimation；物理 dt = sim.dt；step_dt = decimation * sim.dt
    step_dt = raw_env.step_dt
    cmd_term = raw_env.command_manager.get_term("base_velocity")
    min_cmd_speed = args_cli.min_cmd_speed
    print(f"[INFO] min_cmd_speed={min_cmd_speed} m/s（episode 开始时不足则重采样 command）")

    # 并行环境创建后的首条 episode 也需满足线速度下限
    ensure_min_command_xy_speed(cmd_term, list(range(num_envs)), min_cmd_speed)

    while saved_episodes_count < target_count:
        # --- 推理 ---
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        # 遍历所有并行环境，读取其状态
        for env_idx in range(num_envs):
            sim_time = float(raw_env.episode_length_buf[env_idx].item()) * step_dt

            # 速度命令（episode 起点已在 done 分支重采样至 >= min_cmd_speed）
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
            collisions = collect_step_collisions(
                obstacle_sensors,
                robot,
                env_idx,
                force_threshold,
                body_name_to_idx,
            )

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
                    if saved_episodes_count <= target_count:
                        filename = f"{dataset_dir}/episode_{saved_episodes_count:05d}.npz"
                        
                        episode_data = active_episodes[env_idx]
                        npz_data = {}
                        
                        # 转换数值类型字段为 NumPy array
                        numerical_keys = [
                            "time", "command", "base_ang_vel", "projected_gravity",
                            "joint_pos", "joint_vel", "last_action", "root_pos_w",
                            "root_quat_w", "base_lin_vel", "joint_torques"
                        ]
                        for key in numerical_keys:
                            npz_data[key] = np.array([step[key] for step in episode_data], dtype=np.float32)
                        
                        # 变长且包含字符串/字典的 collisions 采用 json 序列化成一个字符串保存
                        collisions = [step["collisions"] for step in episode_data]
                        npz_data["collisions_json"] = np.array(json.dumps(collisions))
                        
                        np.savez_compressed(filename, **npz_data)

                        n_collision_steps = sum(1 for s in episode_data if s["collisions"])
                        run_idx = saved_episodes_count - existing_count
                        print(
                            f"[INFO] Episode {saved_episodes_count:5d} "
                            f"(run {run_idx}/{args_cli.num_episodes}) | "
                            f"env={env_idx:2d} | steps={len(episode_data)} | "
                            f"collision_steps={n_collision_steps} | saved → {filename}"
                        )
                # 新 episode：保证本段轨迹的速度命令线速度 >= min_cmd_speed
                ensure_min_command_xy_speed(cmd_term, [env_idx], min_cmd_speed)
                step_data["command"] = _to_list(cmd_term.vel_command_b[env_idx])
                active_episodes[env_idx] = [step_data]
            else:
                # 正常步，直接追加
                active_episodes[env_idx].append(step_data)

        # 再次检查是否完成了所有的采集任务（防止在多环境同时 done 时溢出）
        if saved_episodes_count >= target_count:
            break

    new_saved = saved_episodes_count - existing_count
    print(f"[INFO] Done. {new_saved} new episodes saved to {dataset_dir}/ (total: {saved_episodes_count})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
