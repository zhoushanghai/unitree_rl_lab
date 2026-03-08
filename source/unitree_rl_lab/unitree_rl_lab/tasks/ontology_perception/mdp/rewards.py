"""
Purpose: Reward helpers for ontology perception tasks.
Main contents: simple alive, upright, contact-penalty, and exploration-progress shaping terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..utils.voxel_map import compute_explored_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def upright_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)


def contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 10.0) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = torch.linalg.norm(sensor.data.net_forces_w.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=-1)
    return (contact_force > threshold).float()


def explored_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    explored = compute_explored_mask(env).flatten(start_dim=1)
    return explored.mean(dim=-1)

