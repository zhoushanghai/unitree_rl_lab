"""
Purpose: Observation terms for ontology perception tasks.
Main contents: proprioception, IMU, body-region contact, and privileged voxel-map observation helpers.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.imu import Imu

from ..utils.voxel_map import compute_explored_mask, compute_local_voxel_gt

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w[:, :3], asset.data.root_quat_w), dim=-1)


def imu_lin_acc(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu")) -> torch.Tensor:
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.lin_acc_b


def imu_ang_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu")) -> torch.Tensor:
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ang_vel_b


def contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w.reshape(env.num_envs, -1)


def contact_magnitude(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    force = contact_force(env, sensor_cfg=sensor_cfg).view(env.num_envs, -1, 3)
    magnitude = torch.linalg.norm(force, dim=-1)
    return magnitude


def local_voxel_gt(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_local_voxel_gt(env).flatten(start_dim=1)


def explored_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_explored_mask(env).flatten(start_dim=1)

