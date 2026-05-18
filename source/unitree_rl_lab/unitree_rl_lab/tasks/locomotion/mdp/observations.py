from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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
