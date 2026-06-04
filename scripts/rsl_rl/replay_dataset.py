"""回放 dataset 中采集的 episode（录制的关节/根位姿），并可视化碰撞点。

用法（Docker 内，需图形界面时不要加 --headless）:
    python scripts/rsl_rl/replay_dataset.py \\
        --file dataset/episode_00001.npz --real-time
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

parser = argparse.ArgumentParser(description="Replay collected G1 dataset episodes with collision markers.")
parser.add_argument("--file", "-f", type=str, required=True, help="Path to episode_XXXXX.npz.")
parser.add_argument("--real-time", action="store_true", help="按录制 time 字段间隔播放。")
parser.add_argument("--show-force", action="store_true", help="在碰撞点处绘制力方向箭头。")

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

# 碰撞点小红球半径（米）
CONTACT_DOT_RADIUS = 0.05


@configclass
class ReplayDatasetSceneCfg(InteractiveSceneCfg):
    """单环境 G1 回放场景（与训练任务类似的可见地面）。"""

    # 使用 TerrainImporter 平面，比 GroundPlaneCfg 在回放中更稳定可见
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
    """从 NPZ 字段解析每步碰撞列表。"""
    if hasattr(raw, "item"):
        try:
            text = raw.item()
        except ValueError:
            text = str(raw)
    else:
        text = str(raw)
    return json.loads(text)


def _quat_wxyz_from_direction(direction: np.ndarray, ref_axis: np.ndarray | None = None) -> np.ndarray:
    """将 ref_axis（默认 +X）旋转到 direction，返回 wxyz 四元数。"""
    if ref_axis is None:
        ref_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d = direction.astype(np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    d /= norm
    ref = ref_axis / np.linalg.norm(ref_axis)
    dot = float(np.clip(np.dot(ref, d), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -0.9999:
        # 反向：绕任意垂直轴转 pi
        ortho = np.array([0.0, 1.0, 0.0]) if abs(ref[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        axis = np.cross(ref, ortho)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis.tolist()], dtype=np.float32)
    axis = np.cross(ref, d)
    axis /= np.linalg.norm(axis)
    half_angle = np.arccos(dot) * 0.5
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def load_episode(npz_path: str) -> dict:
    """加载单条 episode 轨迹与碰撞序列。"""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Episode file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "collisions_json" not in data:
        raise KeyError(f"'collisions_json' missing in {npz_path}")

    collisions = _parse_collisions_json(data["collisions_json"])
    num_steps = len(data["time"])

    episode = {
        "path": npz_path,
        "num_steps": num_steps,
        "time": data["time"].astype(np.float32),
        "joint_pos": data["joint_pos"].astype(np.float32),
        "joint_vel": data["joint_vel"].astype(np.float32),
        "root_pos_w": data["root_pos_w"].astype(np.float32),
        "root_quat_w": data["root_quat_w"].astype(np.float32),
        "collisions": collisions,
    }
    return episode


def _resolve_contact_positions(
    step_collisions: list,
    robot: Articulation,
    env_idx: int = 0,
) -> list[tuple[np.ndarray, dict]]:
    """解析碰撞点：与采集相同逻辑，避免红点落在刚体质心（人体内）。"""
    body_name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    body_pos_w = robot.data.body_pos_w[env_idx].detach().cpu().numpy()
    resolved: list[tuple[np.ndarray, dict]] = []

    for hit in step_collisions:
        body_name = hit.get("body_name")
        if not body_name or body_name not in body_name_to_idx:
            continue
        body_idx = body_name_to_idx[body_name]
        stored_pos = hit.get("contact_position") or []
        if len(stored_pos) != 3:
            continue

        pos, _ = resolve_contact_position_from_contact_pos_w(body_pos_w[body_idx], stored_pos)
        if len(pos) != 3:
            continue
        resolved.append((np.array(pos, dtype=np.float32), hit))

    return resolved


def run_replay(sim: SimulationContext, scene: InteractiveScene, episode: dict):
    robot: Articulation = scene["robot"]
    device = sim.device
    sim_dt = sim.get_physics_dt()

    # 碰撞点：小红点
    marker_cfgs = {
        "contact": sim_utils.SphereCfg(
            radius=CONTACT_DOT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
    if args_cli.show_force:
        marker_cfgs["force"] = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.15, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 1.0)),
        )

    contact_markers = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/dataset_replay/contacts", markers=marker_cfgs)
    )

    num_steps = episode["num_steps"]
    collision_steps = sum(1 for s in episode["collisions"] if len(s) > 0)
    print(
        f"[INFO] Replaying {os.path.basename(episode['path'])} | steps={num_steps} | "
        f"collision_steps={collision_steps} | real_time={args_cli.real_time}"
    )

    for step_idx in range(num_steps):
        if not simulation_app.is_running():
            return

        # --- 机器人状态回放 ---
        root_states = robot.data.default_root_state.clone()
        root_pos = torch.tensor(episode["root_pos_w"][step_idx], device=device, dtype=torch.float32)
        root_quat = torch.tensor(episode["root_quat_w"][step_idx], device=device, dtype=torch.float32)
        root_states[0, :3] = root_pos + scene.env_origins[0]
        root_states[0, 3:7] = root_quat

        # 回放采集时的关节/根状态（与碰撞记录一致）
        joint_pos = torch.tensor(episode["joint_pos"][step_idx], device=device, dtype=torch.float32).unsqueeze(0)
        joint_vel = torch.tensor(episode["joint_vel"][step_idx], device=device, dtype=torch.float32).unsqueeze(0)

        # 根速度置零：纯运动学展示，避免物理积分漂移
        root_states[0, 7:13] = 0.0

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        # --- 碰撞点可视化（须在 scene.update 之后读取 body 位姿）---
        step_collisions = episode["collisions"][step_idx]
        resolved_hits = _resolve_contact_positions(step_collisions, robot)

        # 终端进度（单行刷新）
        sim_t = float(episode["time"][step_idx])
        pct = 100.0 * (step_idx + 1) / num_steps
        print(
            f"\r[INFO] 进度 {step_idx + 1}/{num_steps} ({pct:5.1f}%) | "
            f"t={sim_t:.2f}s | 当步碰撞={len(resolved_hits)}",
            end="",
            flush=True,
        )

        if resolved_hits:
            contact_markers.set_visibility(True)
            contact_positions = [p for p, _ in resolved_hits]
            pos_t = torch.tensor(np.stack(contact_positions), device=device, dtype=torch.float32)
            quat_t = torch.zeros(len(contact_positions), 4, device=device)
            quat_t[:, 0] = 1.0

            if args_cli.show_force:
                force_positions = []
                force_quats = []
                force_scales = []
                for world_pos, hit in resolved_hits:
                    force_vec = np.array(hit.get("contact_force_vector") or [0.0, 0.0, 0.0], dtype=np.float64)
                    if np.linalg.norm(force_vec) < 1e-6:
                        continue
                    force_positions.append(world_pos)
                    force_quats.append(_quat_wxyz_from_direction(force_vec))
                    mag = min(float(hit.get("force_magnitude", np.linalg.norm(force_vec))), 80.0)
                    force_scales.append(np.array([0.05 + 0.01 * mag, 1.0, 1.0, 1.0], dtype=np.float32))

                if force_positions:
                    all_pos = torch.cat([pos_t, torch.tensor(np.stack(force_positions), device=device)], dim=0)
                    all_quat = torch.cat(
                        [quat_t, torch.tensor(np.stack(force_quats), device=device, dtype=torch.float32)], dim=0
                    )
                    indices = [0] * len(contact_positions) + [1] * len(force_positions)
                    scales = torch.ones(len(all_pos), 4, device=device)
                    scales[len(contact_positions) :] = torch.tensor(np.stack(force_scales), device=device)
                    contact_markers.visualize(
                        translations=all_pos, orientations=all_quat, scales=scales, marker_indices=indices
                    )
                else:
                    contact_markers.visualize(
                        translations=pos_t, orientations=quat_t, marker_indices=[0] * len(pos_t)
                    )
            else:
                contact_markers.visualize(
                    translations=pos_t, orientations=quat_t, marker_indices=[0] * len(pos_t)
                )
        else:
            contact_markers.set_visibility(False)

        # 相机跟随
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

    scene_cfg = ReplayDatasetSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_replay(sim, scene, episode)


if __name__ == "__main__":
    main()
    simulation_app.close()
