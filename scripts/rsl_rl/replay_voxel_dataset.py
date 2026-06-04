"""在 Isaac Lab 中回放 dataset_voxel，并可视化 collision_voxel 占用格（与 replay_dataset 类似）。

用法（Docker 内，需图形界面时不要加 --headless）:
    python scripts/rsl_rl/replay_voxel_dataset.py \\
        --file dataset_voxel/episode_00002.npz --real-time
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay dataset_voxel with collision_voxel markers in Isaac Sim.")
parser.add_argument("--file", "-f", type=str, required=True, help="dataset_voxel/episode_XXXXX.npz")
parser.add_argument("--real-time", action="store_true", help="按录制 time 字段间隔播放。")
parser.add_argument(
    "--show-contacts",
    action="store_true",
    help="同时显示 collisions_json 中的碰撞红点（replay_dataset 同款）。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from collision_contact_utils import resolve_contact_position_from_contact_pos_w
from voxel_collision_utils import VOXEL_RES_M, occupied_voxel_centers_world

# 体素占用：半透明小方块；碰撞点：红球
VOXEL_MARKER_SCALE = (VOXEL_RES_M * 0.92, VOXEL_RES_M * 0.92, VOXEL_RES_M * 0.92)
CONTACT_DOT_RADIUS = 0.05


@configclass
class ReplayVoxelSceneCfg(InteractiveSceneCfg):
    """单环境 G1 + 地面（与 replay_dataset 一致）。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.4)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _parse_collisions_json(raw) -> list:
    if hasattr(raw, "item"):
        try:
            text = raw.item()
        except ValueError:
            text = str(raw)
    else:
        text = str(raw)
    return json.loads(text)


def load_episode(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "collision_voxel" not in data:
        raise KeyError(f"'collision_voxel' missing in {npz_path}（请用 dataset_voxel/ 或先跑 process_collision_voxels.py）")

    return {
        "path": npz_path,
        "num_steps": int(data["time"].shape[0]),
        "time": data["time"].astype(np.float32),
        "joint_pos": data["joint_pos"].astype(np.float32),
        "joint_vel": data["joint_vel"].astype(np.float32),
        "root_pos_w": data["root_pos_w"].astype(np.float32),
        "root_quat_w": data["root_quat_w"].astype(np.float32),
        "collision_voxel": data["collision_voxel"].astype(np.uint8),
        "collisions": _parse_collisions_json(data["collisions_json"]) if "collisions_json" in data else [],
    }


def _resolve_contact_positions(step_collisions: list, robot: Articulation) -> list[np.ndarray]:
    body_name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    body_pos_w = robot.data.body_pos_w[0].detach().cpu().numpy()
    out = []
    for hit in step_collisions:
        body_name = hit.get("body_name")
        if not body_name or body_name not in body_name_to_idx:
            continue
        stored = hit.get("contact_position") or []
        if len(stored) != 3:
            continue
        pos, _ = resolve_contact_position_from_contact_pos_w(body_pos_w[body_name_to_idx[body_name]], stored)
        if len(pos) == 3:
            out.append(np.array(pos, dtype=np.float32))
    return out


def run_replay(sim: SimulationContext, scene: InteractiveScene, episode: dict):
    robot: Articulation = scene["robot"]
    device = sim.device
    sim_dt = sim.get_physics_dt()
    env_origin = scene.env_origins[0]

    marker_cfgs = {
        "voxel": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.05), opacity=0.85),
        ),
    }
    if args_cli.show_contacts:
        marker_cfgs["contact"] = sim_utils.SphereCfg(
            radius=CONTACT_DOT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )

    markers = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/voxel_replay/markers", markers=marker_cfgs)
    )

    num_steps = episode["num_steps"]
    voxels = episode["collision_voxel"]
    steps_with_vox = int(np.any(voxels.reshape(num_steps, -1), axis=1).sum())
    print(
        f"[INFO] Replaying {os.path.basename(episode['path'])} | steps={num_steps} | "
        f"steps_with_voxel={steps_with_vox} | real_time={args_cli.real_time} | show_contacts={args_cli.show_contacts}"
    )

    for step_idx in range(num_steps):
        if not simulation_app.is_running():
            return

        root_states = robot.data.default_root_state.clone()
        root_pos = torch.tensor(episode["root_pos_w"][step_idx], device=device, dtype=torch.float32)
        root_quat = torch.tensor(episode["root_quat_w"][step_idx], device=device, dtype=torch.float32)
        root_states[0, :3] = root_pos + env_origin
        root_states[0, 3:7] = root_quat
        root_states[0, 7:13] = 0.0

        joint_pos = torch.tensor(episode["joint_pos"][step_idx], device=device, dtype=torch.float32).unsqueeze(0)
        joint_vel = torch.tensor(episode["joint_vel"][step_idx], device=device, dtype=torch.float32).unsqueeze(0)

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        # 体素中心：局部网格 → 当前步世界坐标
        vol = voxels[step_idx]
        root_np = episode["root_pos_w"][step_idx]
        quat_np = episode["root_quat_w"][step_idx]
        world_centers = occupied_voxel_centers_world(vol, root_np, quat_np)
        if world_centers.shape[0] > 0:
            world_centers = world_centers + env_origin.detach().cpu().numpy()

        n_vox = int(world_centers.shape[0])
        n_contact = 0
        if args_cli.show_contacts and step_idx < len(episode["collisions"]):
            contact_pts = _resolve_contact_positions(episode["collisions"][step_idx], robot)
            if contact_pts:
                contact_pts = [p + env_origin.detach().cpu().numpy() for p in contact_pts]
            n_contact = len(contact_pts)
        else:
            contact_pts = []

        sim_t = float(episode["time"][step_idx])
        pct = 100.0 * (step_idx + 1) / num_steps
        print(
            f"\r[INFO] 进度 {step_idx + 1}/{num_steps} ({pct:5.1f}%) | t={sim_t:.2f}s | "
            f"voxels={n_vox} | contacts={n_contact}",
            end="",
            flush=True,
        )

        if n_vox == 0 and n_contact == 0:
            markers.set_visibility(False)
        else:
            markers.set_visibility(True)
            translations = []
            orientations = []
            indices = []
            scales = []

            if n_vox > 0:
                translations.append(torch.tensor(world_centers, device=device, dtype=torch.float32))
                quat_v = torch.zeros(n_vox, 4, device=device)
                quat_v[:, 0] = 1.0
                orientations.append(quat_v)
                # visualize() 要求 scales 为 (M, 3)，不是 (M, 4)
                sx, sy, sz = VOXEL_MARKER_SCALE
                scale_v = torch.tensor([[sx, sy, sz]] * n_vox, device=device, dtype=torch.float32)
                scales.append(scale_v)
                indices.extend([0] * n_vox)

            if n_contact > 0:
                translations.append(torch.tensor(np.stack(contact_pts), device=device, dtype=torch.float32))
                quat_c = torch.zeros(n_contact, 4, device=device)
                quat_c[:, 0] = 1.0
                orientations.append(quat_c)
                scales.append(torch.ones(n_contact, 3, device=device))
                indices.extend([1] * n_contact)

            markers.visualize(
                translations=torch.cat(translations, dim=0),
                orientations=torch.cat(orientations, dim=0),
                scales=torch.cat(scales, dim=0),
                marker_indices=indices,
            )

        lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(lookat + np.array([2.5, 2.5, 0.6]), lookat)

        if args_cli.real_time and step_idx > 0:
            dt = float(episode["time"][step_idx] - episode["time"][step_idx - 1])
            if dt > 0.0:
                time.sleep(dt)

    print()
    print("[INFO] 回放完成（仅播放一次）。")


def main():
    episode = load_episode(args_cli.file)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(ReplayVoxelSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_replay(sim, scene, episode)


if __name__ == "__main__":
    main()
    simulation_app.close()
