"""
Purpose: Compute local voxel occupancy and explored-mask tensors for ontology perception tasks.
Main contents: voxel grid initialization, room cuboid projection, and short-horizon explored-point buffering.
"""

from __future__ import annotations

import itertools

import torch
from isaaclab.assets import Articulation, RigidObject

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from ..robots.g1_room.scene_cfg import (
    CONTACT_SENSOR_NAMES,
    CONTACT_SENSOR_TO_BODY_NAME,
    EXPLORATION_BODY_NAMES,
    LOCAL_VOXEL_RESOLUTION,
    LOCAL_VOXEL_SIZE,
    ROOM_OBJECT_SPECS,
    SUPPORT_SENSOR_NAMES,
)

MAX_RECENT_POINTS = 512
CONTACT_FORCE_THRESHOLD = 5.0


def _ensure_state(env) -> dict[str, torch.Tensor | list[int] | dict[str, int] | int]:
    if hasattr(env, "_ontology_perception_state") and env._ontology_perception_state is not None:
        return env._ontology_perception_state

    device = env.device
    dtype = torch.float32
    x_cells, y_cells, z_cells = LOCAL_VOXEL_RESOLUTION
    x_size, y_size, z_size = LOCAL_VOXEL_SIZE

    x_coords = torch.linspace(-x_size / 2 + x_size / (2 * x_cells), x_size / 2 - x_size / (2 * x_cells), x_cells, device=device, dtype=dtype)
    y_coords = torch.linspace(-y_size / 2 + y_size / (2 * y_cells), y_size / 2 - y_size / (2 * y_cells), y_cells, device=device, dtype=dtype)
    z_coords = torch.linspace(-z_size / 2 + z_size / (2 * z_cells), z_size / 2 - z_size / (2 * z_cells), z_cells, device=device, dtype=dtype)
    grid_local = torch.stack(torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij"), dim=-1).reshape(-1, 3)

    robot: Articulation = env.scene["robot"]
    exploration_body_ids, _ = robot.find_bodies(list(EXPLORATION_BODY_NAMES), preserve_order=True)
    contact_body_ids = {name: robot.find_bodies(body_name)[0][0] for name, body_name in CONTACT_SENSOR_TO_BODY_NAME.items()}
    neighbor_offsets = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), device=device, dtype=torch.long)
    resolution = torch.tensor(LOCAL_VOXEL_RESOLUTION, device=device, dtype=torch.long)

    env._ontology_perception_state = {
        "grid_local": grid_local,
        "neighbor_offsets": neighbor_offsets,
        "resolution": resolution,
        "exploration_body_ids": exploration_body_ids,
        "contact_body_ids": contact_body_ids,
        "recent_points": torch.zeros((env.num_envs, MAX_RECENT_POINTS, 3), device=device, dtype=dtype),
        "point_counts": torch.zeros(env.num_envs, device=device, dtype=torch.long),
        "last_update_step": -1,
    }
    return env._ontology_perception_state


def _clear_reset_envs(env, state: dict[str, torch.Tensor | int]) -> None:
    reset_env_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() == 0:
        return
    state["recent_points"][reset_env_ids] = 0.0
    state["point_counts"][reset_env_ids] = 0


def _append_recent_points(state: dict[str, torch.Tensor | int], points: torch.Tensor, valid_mask: torch.Tensor) -> None:
    recent_points: torch.Tensor = state["recent_points"]
    point_counts: torch.Tensor = state["point_counts"]
    max_points = recent_points.shape[1]

    for env_idx in range(points.shape[0]):
        valid_points = points[env_idx][valid_mask[env_idx]]
        if valid_points.numel() == 0:
            continue

        if valid_points.shape[0] >= max_points:
            recent_points[env_idx] = valid_points[-max_points:]
            point_counts[env_idx] = max_points
            continue

        existing_count = int(point_counts[env_idx].item())
        total_count = existing_count + valid_points.shape[0]
        if total_count <= max_points:
            recent_points[env_idx, existing_count:total_count] = valid_points
            point_counts[env_idx] = total_count
            continue

        overflow = total_count - max_points
        recent_points[env_idx] = torch.roll(recent_points[env_idx], shifts=-overflow, dims=0)
        insert_start = max_points - valid_points.shape[0]
        recent_points[env_idx, insert_start:] = valid_points
        point_counts[env_idx] = max_points


def _update_recent_points(env) -> None:
    state = _ensure_state(env)
    current_step = int(env.common_step_counter)
    if state["last_update_step"] == current_step:
        return

    _clear_reset_envs(env, state)

    robot: Articulation = env.scene["robot"]
    body_points = robot.data.body_pos_w[:, state["exploration_body_ids"], :3]
    body_valid = torch.ones(body_points.shape[:2], device=env.device, dtype=torch.bool)

    contact_points = []
    contact_valid = []
    for sensor_name in CONTACT_SENSOR_NAMES:
        sensor = env.scene.sensors[sensor_name]
        forces = sensor.data.net_forces_w.reshape(env.num_envs, -1, 3)
        magnitudes = torch.linalg.norm(forces, dim=-1).amax(dim=-1)
        point = robot.data.body_pos_w[:, state["contact_body_ids"][sensor_name], :3].unsqueeze(1).clone()
        if sensor_name in SUPPORT_SENSOR_NAMES:
            point[..., 2] -= 0.04
        contact_points.append(point)
        contact_valid.append((magnitudes > CONTACT_FORCE_THRESHOLD).unsqueeze(1))

    stacked_contact_points = torch.cat(contact_points, dim=1)
    stacked_contact_valid = torch.cat(contact_valid, dim=1)
    all_points = torch.cat([body_points, stacked_contact_points], dim=1)
    all_valid = torch.cat([body_valid, stacked_contact_valid], dim=1)

    _append_recent_points(state, all_points, all_valid)
    state["last_update_step"] = current_step


def compute_local_voxel_gt(env) -> torch.Tensor:
    state = _ensure_state(env)
    robot: Articulation = env.scene["robot"]
    grid_local: torch.Tensor = state["grid_local"]
    num_envs = env.num_envs
    num_points = grid_local.shape[0]
    voxel = torch.zeros((num_envs, num_points), device=env.device, dtype=torch.float32)

    for env_idx in range(num_envs):
        base_quat = robot.data.root_quat_w[env_idx].unsqueeze(0).expand(num_points, -1)
        base_pos = robot.data.root_pos_w[env_idx, :3]
        world_points = quat_apply(base_quat, grid_local) + base_pos

        occupied = torch.zeros(num_points, device=env.device, dtype=torch.bool)
        for object_name, spec in ROOM_OBJECT_SPECS.items():
            object_asset: RigidObject = env.scene[object_name]
            center = object_asset.data.root_pos_w[env_idx, :3]
            half_extent = torch.tensor(spec.size, device=env.device, dtype=torch.float32) * 0.5
            occupied |= torch.all(torch.abs(world_points - center) <= half_extent + 1.0e-6, dim=-1)

        voxel[env_idx] = occupied.float()

    return voxel.view(num_envs, *LOCAL_VOXEL_RESOLUTION)


def compute_explored_mask(env) -> torch.Tensor:
    _update_recent_points(env)
    state = _ensure_state(env)
    robot: Articulation = env.scene["robot"]
    recent_points: torch.Tensor = state["recent_points"]
    point_counts: torch.Tensor = state["point_counts"]
    neighbor_offsets: torch.Tensor = state["neighbor_offsets"]
    resolution: torch.Tensor = state["resolution"]

    x_cells, y_cells, z_cells = LOCAL_VOXEL_RESOLUTION
    x_size, y_size, z_size = LOCAL_VOXEL_SIZE
    cell_size = torch.tensor(
        (x_size / x_cells, y_size / y_cells, z_size / z_cells),
        device=env.device,
        dtype=torch.float32,
    )
    half_extent = torch.tensor((x_size / 2, y_size / 2, z_size / 2), device=env.device, dtype=torch.float32)
    mask = torch.zeros((env.num_envs, x_cells * y_cells * z_cells), device=env.device, dtype=torch.float32)

    for env_idx in range(env.num_envs):
        count = int(point_counts[env_idx].item())
        if count == 0:
            continue

        points_w = recent_points[env_idx, :count]
        base_pos = robot.data.root_pos_w[env_idx, :3]
        base_quat = robot.data.root_quat_w[env_idx].unsqueeze(0).expand(count, -1)
        points_local = quat_apply_inverse(base_quat, points_w - base_pos)
        in_bounds = torch.all(torch.abs(points_local) <= half_extent + 1.0e-6, dim=-1)
        if not torch.any(in_bounds):
            continue

        valid_points = points_local[in_bounds]
        cell_indices = torch.floor((valid_points + half_extent) / cell_size).long()
        cell_indices = torch.maximum(cell_indices, torch.zeros_like(cell_indices))
        cell_indices = torch.minimum(cell_indices, (resolution - 1).view(1, 3))

        for offset in neighbor_offsets:
            expanded_idx = cell_indices + offset
            is_valid_idx = torch.logical_and(expanded_idx >= 0, expanded_idx < resolution)
            is_valid_idx = torch.all(is_valid_idx, dim=-1)
            if not torch.any(is_valid_idx):
                continue
            expanded_idx = expanded_idx[is_valid_idx]
            flat_idx = (
                expanded_idx[:, 0] * (y_cells * z_cells)
                + expanded_idx[:, 1] * z_cells
                + expanded_idx[:, 2]
            )
            mask[env_idx, flat_idx] = 1.0

    return mask.view(env.num_envs, *LOCAL_VOXEL_RESOLUTION)
