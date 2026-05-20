# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help="Export policy to JIT/ONNX during play (disabled by default to keep logs clean).",
)
parser.add_argument(
    "--apf_collision_grid_vis",
    action="store_true",
    default=False,
    help="Open matplotlib APF collision grid debug panel.",
)
parser.add_argument(
    "--apf_collision_grid_radius_m",
    type=float,
    default=1.0,
    help="Circular local map radius in meters.",
)
parser.add_argument(
    "--apf_collision_grid_cell_m",
    type=float,
    default=0.05,
    help="Grid cell size in meters (r=1.0, cell=0.05 => ~40x40).",
)
parser.add_argument(
    "--apf_collision_grid_env_id",
    type=int,
    default=0,
    help="Parallel env index to visualize.",
)
parser.add_argument(
    "--apf_collision_grid_every",
    type=int,
    default=1,
    help="Refresh panel every N simulation steps.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _focus_camera_on_robot(gym_env) -> bool:
    """Best-effort camera focus for play startup.

    Try multiple common camera APIs across IsaacLab versions to keep the robot
    visible at startup. Failures are swallowed to avoid breaking play.
    """
    try:
        core = gym_env.unwrapped
        robot = core.scene["robot"]
        root = robot.data.root_pos_w[0, :3].detach().cpu()
        lookat = (float(root[0]), float(root[1]), float(root[2] + 0.8))
        eye = (float(root[0] + 2.8), float(root[1] + 2.8), float(root[2] + 1.8))
    except Exception:
        return False

    candidates = [core, getattr(core, "sim", None), getattr(core, "viewer", None)]
    for obj in candidates:
        if obj is None:
            continue
        for method_name in ("set_camera_view", "set_camera_pose", "set_view"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            # Try both keyword and positional calling conventions.
            try:
                method(eye=eye, target=lookat)
                return True
            except Exception:
                pass
            try:
                method(eye, lookat)
                return True
            except Exception:
                pass
    return False


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    # specify directory for logging experiments
    experiment_name = agent_cfg.experiment_name
    log_root_path = os.path.join("logs", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 可选：APF 局部碰撞栅格调试面板（matplotlib）
    apf_grid_vis = None
    if args_cli.apf_collision_grid_vis:
        try:
            # 仅使用本仓库内的 APF 面板实现（已复制 proprioception 原版代码）。
            panel_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.apf_collision_grid_vis")
            ApfPlayDebugPanel = getattr(panel_mod, "ApfPlayDebugPanel")
            apf_grid_vis = ApfPlayDebugPanel(
                radius_m=args_cli.apf_collision_grid_radius_m,
                cell_m=args_cli.apf_collision_grid_cell_m,
                env_id=args_cli.apf_collision_grid_env_id,
                update_every=args_cli.apf_collision_grid_every,
            )
            print("[INFO] APF collision grid panel enabled. Close the figure window to hide it.")
        except Exception as e:
            print(f"[WARNING] APF collision grid panel could not start: {e}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # 先尝试完整加载；若遇到 critic 结构不匹配（常见于增加/移除 privileged obs 后），
    # 自动降级为“仅加载 actor”以支持旧 checkpoint 的推理播放。
    try:
        runner.load(resume_path)
    except RuntimeError as err:
        err_text = str(err)
        critic_mismatch = (
            "Error(s) in loading state_dict for MLPModel" in err_text
            and "size mismatch" in err_text
            and ("critic" in err_text.lower() or "mlp.0.weight" in err_text)
        )
        if not critic_mismatch:
            raise
        print("[WARN] Critic state_dict shape mismatch detected. Falling back to actor-only checkpoint loading for play.")
        runner.load(
            resume_path,
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=False,
            map_location=agent_cfg.device,
        )

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 默认不导出策略，避免 play 过程中出现 ONNX/GRU 相关告警干扰观察。
    # 如确需导出，可显式传入 --export_policy。
    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        try:
            runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
        except Exception as export_err:
            # 导出失败不应阻断播放主流程；打印告警便于后续单独排查导出兼容性。
            print(f"[WARN] Policy export skipped due to compatibility error: {export_err}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    # 启动阶段强制对焦机器人，避免部分版本/窗口下默认视角偏离主体。
    camera_focused = False
    for _ in range(3):
        camera_focused = _focus_camera_on_robot(env) or camera_focused
        if camera_focused:
            break
    timestep = 0
    sim_step_idx = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            if apf_grid_vis is not None:
                apf_grid_vis.step(env)
            # 前几步重复一次对焦，覆盖可能的首帧视角重置。
            if sim_step_idx < 20 and not camera_focused:
                camera_focused = _focus_camera_on_robot(env) or camera_focused
        sim_step_idx += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if apf_grid_vis is not None:
        apf_grid_vis.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
