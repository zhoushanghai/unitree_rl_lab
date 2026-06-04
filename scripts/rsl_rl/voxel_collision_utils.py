"""碰撞点持久化与局部体素网格（见 docs/collision_voxel_design.md）。"""

from __future__ import annotations

import numpy as np

# 与 design 定稿一致
VOXEL_RES_M = 0.1
GRID_NX = GRID_NY = 20
GRID_NZ = 15
LOCAL_XY_MIN = -1.0
LOCAL_XY_MAX = 1.0
Z_MIN = 0.0
Z_MAX = 1.5
DEFAULT_MIN_CMD_SPEED = 0.3


def command_xy_speed(command_row: np.ndarray) -> float:
    """command[t] = [vx, vy, wz]，线速度模长不含 wz。"""
    c = np.asarray(command_row, dtype=np.float64).reshape(-1)
    return float(np.hypot(c[0], c[1]))


def yaw_from_root_quat_w(quat_wxyz: np.ndarray) -> float:
    """root_quat_w = [w, x, y, z]，提取绕世界 z 轴 yaw（与 scipy 约定一致）。"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def world_point_to_voxel_index(
    point_w: np.ndarray,
    root_pos_w: np.ndarray,
    yaw: float,
) -> tuple[int, int, int] | None:
    """世界系碰撞点 → 局部体素 (i,j,k)；网格外返回 None。"""
    px, py, pz = float(point_w[0]), float(point_w[1]), float(point_w[2])
    rx, ry = float(root_pos_w[0]), float(root_pos_w[1])

    dx, dy = px - rx, py - ry
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    # 机体前向 +x、左侧 +y
    x_local = cos_y * dx + sin_y * dy
    y_local = -sin_y * dx + cos_y * dy

    if not (LOCAL_XY_MIN <= x_local < LOCAL_XY_MAX and LOCAL_XY_MIN <= y_local < LOCAL_XY_MAX):
        return None
    if not (Z_MIN <= pz < Z_MAX):
        return None

    i = int(np.floor((x_local - LOCAL_XY_MIN) / VOXEL_RES_M))
    j = int(np.floor((y_local - LOCAL_XY_MIN) / VOXEL_RES_M))
    k = int(np.floor((pz - Z_MIN) / VOXEL_RES_M))
    i = int(np.clip(i, 0, GRID_NX - 1))
    j = int(np.clip(j, 0, GRID_NY - 1))
    k = int(np.clip(k, 0, GRID_NZ - 1))
    return i, j, k


def is_valid_collision_record(record: dict) -> bool:
    """跳过空位姿、invalid source。"""
    pos = record.get("contact_position")
    if pos is None or len(pos) != 3:
        return False
    arr = np.asarray(pos, dtype=np.float64)
    if not np.isfinite(arr).all():
        return False
    src = record.get("contact_position_source")
    if src is not None and src != "contact_pos_w":
        return False
    return True


def local_voxel_center_m(i: int, j: int, k: int) -> np.ndarray:
    """体素 (i,j,k) 中心在局部水平系 + 世界 z 下的坐标 (x前, y左, z_up)。"""
    x = LOCAL_XY_MIN + (i + 0.5) * VOXEL_RES_M
    y = LOCAL_XY_MIN + (j + 0.5) * VOXEL_RES_M
    z = Z_MIN + (k + 0.5) * VOXEL_RES_M
    return np.array([x, y, z], dtype=np.float64)


def occupied_voxel_centers_world(
    voxel_grid: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
) -> np.ndarray:
    """将单步 collision_voxel (20,20,15) 占用格中心变换到世界系，返回 (N,3)。"""
    ii, jj, kk = np.where(voxel_grid > 0)
    if ii.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    yaw = yaw_from_root_quat_w(root_quat_w)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rx, ry = float(root_pos_w[0]), float(root_pos_w[1])

    points = []
    for i, j, k in zip(ii.tolist(), jj.tolist(), kk.tolist()):
        local = local_voxel_center_m(i, j, k)
        dx = local[0] * cos_y - local[1] * sin_y
        dy = local[0] * sin_y + local[1] * cos_y
        points.append([rx + dx, ry + dy, local[2]])

    return np.asarray(points, dtype=np.float32)


def voxel_indices_at_pose(
    contacts_world: list[np.ndarray],
    root_pos_w: np.ndarray,
    yaw: float,
) -> set[tuple[int, int, int]]:
    """将持久化世界点集投影到当前位姿下的占用体素索引集合。"""
    occupied: set[tuple[int, int, int]] = set()
    for pt in contacts_world:
        idx = world_point_to_voxel_index(pt, root_pos_w, yaw)
        if idx is not None:
            occupied.add(idx)
    return occupied


def build_collision_voxel_sequence(
    collisions_per_step: list,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
) -> np.ndarray:
    """生成 (T, 20, 20, 15) uint8 体素序列（线速度过滤在写入前按 episode 剔除）。"""
    t_steps = root_pos_w.shape[0]
    voxels = np.zeros((t_steps, GRID_NX, GRID_NY, GRID_NZ), dtype=np.uint8)
    contacts_world: list[np.ndarray] = []

    for t in range(t_steps):
        yaw = yaw_from_root_quat_w(root_quat_w[t])
        root_t = root_pos_w[t]

        # 持久化：当步每条有效碰撞记录的世界坐标全部追加（不做体素/距离去重）
        for rec in collisions_per_step[t] if t < len(collisions_per_step) else []:
            if not is_valid_collision_record(rec):
                continue
            contacts_world.append(np.asarray(rec["contact_position"], dtype=np.float64))

        # 用当前步位姿将累积世界点重投影到局部网格（同格多点仍只占 1 格）
        for idx in voxel_indices_at_pose(contacts_world, root_t, yaw):
            voxels[t, idx[0], idx[1], idx[2]] = 1

    return voxels
