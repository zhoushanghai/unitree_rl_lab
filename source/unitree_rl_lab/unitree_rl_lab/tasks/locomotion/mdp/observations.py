from __future__ import annotations

import torch
from typing import TYPE_CHECKING
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apf_raw_velocity_commands(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """返回“输入给模型”的原始速度命令（APF 修正前）。

    设计目的（按当前迁移需求）：
    - 控制执行与 reward：使用 APF 修正后的新命令（events.py 中回写到 vel_command_b）。
    - 模型观测输入：使用 APF 修正前的旧命令，避免策略直接看到避障后的目标速度。

    数据来源：
    - 首选 `env._apf_v_cmd_b`（在 `apply_apf_to_velocity_command` 中每步缓存的旧命令）。
    - 若缓存尚未建立（例如训练刚启动第一步），降级为当前 command manager 命令，保证流程不断。
    """
    # 常规路径：APF 事件会在每步先缓存“旧 cmd”，这里直接返回该缓存供模型使用。
    if hasattr(env, "_apf_v_cmd_b") and env._apf_v_cmd_b.shape[0] == env.num_envs:
        return env._apf_v_cmd_b

    # 兜底路径：防止首步/异常时缓存还未创建。
    return env.command_manager.get_command(command_name)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def obstacle_collision_slots_base_xy(env: ManagerBasedRLEnv, max_points: int = 10) -> torch.Tensor:
    """返回给 Critic 的碰撞点特权观测：每槽 (x_b, y_b, valid) 交错展开。

    输出形状：
    - (num_envs, 3 * max_points)
    - 排列方式: [x0, y0, v0, x1, y1, v1, ...]

    语义约定：
    - x_b, y_b 为“机体系”下的碰撞点坐标（由世界系点减根位置后再做 quat_apply_inverse）。
    - valid∈{0,1}，无效槽位时 xy 会被强制置零，避免给 Critic 注入噪声。
    """
    n = env.num_envs
    device = env.device
    out = torch.zeros((n, 3 * max_points), dtype=torch.float32, device=device)

    # 若缓存尚未初始化（例如首步或未启用碰撞缓存事件），返回全零占位，保证训练流程稳定。
    if (not hasattr(env, "_obstacle_collision_points_w")) or (not hasattr(env, "_obstacle_collision_slot_valid")):
        return out

    points_w = env._obstacle_collision_points_w[:, :max_points, :3]  # (n, K, 3)
    valid = env._obstacle_collision_slot_valid[:, :max_points]  # (n, K)
    robot = env.scene["robot"]

    # 世界系点 -> 机体系偏移：先减 root_pos_w，再按 root_quat_w 做 inverse 旋转。
    root_pos = robot.data.root_pos_w.unsqueeze(1).expand(-1, max_points, -1)
    offsets_w = points_w - root_pos
    quat_flat = robot.data.root_quat_w.unsqueeze(1).expand(-1, max_points, -1).reshape(n * max_points, 4)
    offsets_flat = offsets_w.reshape(n * max_points, 3)
    offsets_b = quat_apply_inverse(quat_flat, offsets_flat).reshape(n, max_points, 3)

    v = valid.to(dtype=torch.float32)
    xy = offsets_b[:, :, :2] * v.unsqueeze(-1)
    slots = torch.stack((xy[:, :, 0], xy[:, :, 1], v), dim=-1)
    return slots.reshape(n, 3 * max_points)
