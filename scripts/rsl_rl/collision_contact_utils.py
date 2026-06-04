"""从 ContactSensor.contact_pos_w 解析碰撞点世界坐标。"""

from __future__ import annotations

import numpy as np

# 仅拒绝明显离谱的读数（相对连杆质心过远）
MAX_COM_DISTANCE_M = 3.0


def _to_numpy(vec) -> np.ndarray:
    """兼容 CUDA tensor / list，转为 (3,) float32。"""
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(3)


def contact_filter_index(env_idx: int, b_idx: int, num_bodies: int, num_envs: int, filter_count: int) -> int:
    """推断 force_matrix_w / contact_pos_w 的 filter 维下标（与力、位置同一列）。"""
    if filter_count == num_envs * num_bodies:
        # 每 env×body 一条 obstacle 路径（1 对 1）
        return env_idx * num_bodies + b_idx
    if filter_count == num_envs:
        # 每环境一个 Obstacle filter，该 env 下所有 body 共用此列
        return env_idx
    if filter_count == num_bodies:
        return b_idx
    if filter_count == 1:
        return 0
    # 回退：按扁平索引，避免越界
    return min(env_idx * num_bodies + b_idx, filter_count - 1)


def resolve_contact_position_from_contact_pos_w(
    body_pos_w,
    contact_pos_w,
) -> tuple[list[float], str]:
    """使用 contact_pos_w；仅拒绝 NaN 或相对质心过远的异常值。"""
    body = _to_numpy(body_pos_w)
    pos = _to_numpy(contact_pos_w)

    if not np.isfinite(pos).all():
        return [], "invalid"

    if float(np.linalg.norm(pos - body)) > MAX_COM_DISTANCE_M:
        return [], "invalid"

    return pos.tolist(), "contact_pos_w"
