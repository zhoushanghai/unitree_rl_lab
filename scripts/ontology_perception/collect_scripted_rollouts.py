"""
Purpose: Collect scripted ontology perception rollouts from the G1 room task and export per-episode datasets.
Main contents: Isaac Lab app launch, optional video recording, scripted action loop, collector-group export, and post-rollout scene inspection.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect scripted ontology perception rollouts.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-Ontology-Perception-Room-v0",
    help="Task name to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only env_0 is exported.")
parser.add_argument("--episodes", type=int, default=4, help="Number of episodes to record.")
parser.add_argument("--steps_per_episode", type=int, default=420, help="Maximum environment steps per episode.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=420, help="Maximum number of steps to keep in the video.")
parser.add_argument(
    "--keep_alive",
    action="store_true",
    default=False,
    help="Keep the GUI window open after the rollout so the scene can be inspected manually.",
)
parser.add_argument(
    "--save_format",
    type=str,
    choices=("npz", "pt"),
    default="npz",
    help="Dataset serialization format.",
)
parser.add_argument("--output_dir", type=str, default=None, help="Optional export directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.ontology_perception.utils import ScriptedExplorer  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return np.asarray(value)


def _save_episode(output_file: str, episode_data: dict[str, list], save_format: str) -> None:
    if save_format == "npz":
        array_data = {key: np.stack(value, axis=0) for key, value in episode_data.items()}
        np.savez_compressed(output_file, **array_data)
        return

    tensor_data = {key: torch.from_numpy(np.stack(value, axis=0)) for key, value in episode_data.items()}
    torch.save(tensor_data, output_file)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric if hasattr(args_cli, "disable_fabric") else None,
        entry_point_key="play_env_cfg_entry_point",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args_cli.output_dir or os.path.join("outputs", "ontology_perception", f"{args_cli.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(output_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"{args_cli.task}_{timestamp}",
        }
        print("[INFO] Recording scripted ontology perception video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.unwrapped.scene.update(dt=0.0)
    explorer = ScriptedExplorer(env.unwrapped)

    obs, _ = env.reset()
    del obs

    for episode_idx in range(args_cli.episodes):
        episode_data = defaultdict(list)
        env.reset()

        for step_idx in range(args_cli.steps_per_episode):
            action = explorer.action(step_idx, env.unwrapped.num_envs)
            _, reward, terminated, truncated, extras = env.step(action)
            collector_obs = env.unwrapped.observation_manager.compute_group("collector")

            for key, value in collector_obs.items():
                episode_data[key].append(_to_numpy(value[0]))
            episode_data["reward"].append(_to_numpy(reward[0]))
            episode_data["terminated"].append(_to_numpy(terminated[0]))
            episode_data["truncated"].append(_to_numpy(truncated[0]))

            if bool((terminated | truncated)[0].item()):
                del extras
                break

        extension = "npz" if args_cli.save_format == "npz" else "pt"
        output_file = os.path.join(output_dir, f"episode_{episode_idx:03d}.{extension}")
        _save_episode(output_file, episode_data, args_cli.save_format)

    if args_cli.keep_alive and not getattr(args_cli, "headless", False):
        print("[INFO] Rollout finished. The window will stay open for inspection.")
        print("[INFO] Drag the viewport as needed, then close the window or press Ctrl+C in the terminal to exit.")
        try:
            while simulation_app.is_running() and not simulation_app.is_exiting():
                env.unwrapped.sim.render()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("[INFO] Inspection interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
