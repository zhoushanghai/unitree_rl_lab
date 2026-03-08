"""
Purpose: Termination helpers for ontology perception tasks.
Main contents: room-boundary checks used alongside standard Isaac Lab timeout and fall conditions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def outside_room_bounds(
    env: ManagerBasedRLEnv,
    limit_x: float = 1.8,
    limit_y: float = 1.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w[:, :2]
    return torch.logical_or(torch.abs(pos_w[:, 0]) > limit_x, torch.abs(pos_w[:, 1]) > limit_y)

