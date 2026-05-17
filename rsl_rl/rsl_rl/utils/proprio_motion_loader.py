# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proprioception 风格的 reference motion 加载（YAML 清单 + 每条轨迹 ``.npz``）。

与设计文档中的约定对齐：
    - YAML 顶层 ``motion_files``：每条可为 ``{file, weight}`` 或单独路径字符串；
    - ``npz`` 含 ``dof_names`` / ``key_body_names`` / ``fps`` / ``dof_positions`` /
      ``dof_velocities`` / ``key_body_positions_local``；

Expert 侧的 AMP 行向量定义为（与 Isaac Lab proprioception ``_compute_amp_observations`` 一致）::

    concat( dof_positions[子集], dof_velocities[子集],
            key_body_positions_local[子集].reshape(T, -1) )

判别器需要的是 ``(state, next_state)`` 对：本模块在每条 clip 内部用 ``t→t+1`` 构造，
最后帧不自连到另一条轨迹（末尾 hold），避免错误的跨片段转移。
"""

from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import torch
import yaml


class ProprioceptionMotionExpert:
    """在内存中建好所有 clip 展平后的 ``all_obs`` / ``all_next_obs``，并按权重抽样 expert batch。

    与 ``amp-rsl-rl`` 的 ``AMPLoader.feed_forward_generator`` 角色相同：产出
    ``(expert_state, expert_next_state)``，维度均为 ``amp_obs_dim``，
    供 ``AMPPPO.update`` 与 policy replay batch 对齐使用。
    """

    def __init__(
        self,
        motion_yaml_path: str,
        amp_obs_dim: int,
        device: str | torch.device,
        motion_dof_indexes: Sequence[int] | None = None,
        motion_key_body_indexes: Sequence[int] | None = None,
        motion_project_root: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        # 用户配置的 AMP 单行维数，必须与 env 里算的 amp 向量宽度一致。
        self.amp_obs_dim = int(amp_obs_dim)

        if not os.path.isfile(motion_yaml_path):
            raise FileNotFoundError(motion_yaml_path)

        # 相对路径锚点：若有 ``motion_project_root`` 则用之；否则沿目录向上找到含 ``datasets`` 的根（与 proprioception MotionLoader 一致）。
        project_root = motion_project_root or self._find_project_root(os.path.dirname(os.path.abspath(motion_yaml_path)))
        entries = self._parse_manifest(motion_yaml_path, project_root)
        # 采样权重归一化为概率分布，用于按帧加权（每条轨迹内部的帧权 = 轨迹权 / 轨迹长度）。
        weights = self._normalize_weights(np.array([w for _, w in entries], dtype=np.float64))

        obs_segments: list[np.ndarray] = []
        weight_frames: list[torch.Tensor] = []

        for (npz_path, w), motion_weight in zip(entries, weights, strict=True):
            data = np.load(npz_path)
            # 首条轨迹定 ``dof/key_body/fps`` 模板，后续必须与之一致。
            self._validate_first_or_match(data, npz_path)
            feat = self._rows_to_amp_features(
                data,
                motion_dof_indexes,
                motion_key_body_indexes,
            )
            # 与子集拼接后的真实维数必须与配置吻合，否则会与判别器输入维错位。
            if feat.shape[1] != self.amp_obs_dim:
                raise ValueError(
                    f"AMP feature dim {feat.shape[1]} from {npz_path} != configured amp_obs_dim={self.amp_obs_dim}. "
                    "Check motion_dof_indexes / motion_key_body_indexes or cfg['algorithm']['amp_obs_dim'].",
                )
            obs_segments.append(feat)
            L = feat.shape[0]
            # 该轨迹上每一帧的未归一化先验权重 chunk；后面拼成全局 per_frame_weights 再归一化。
            weight_frames.append(torch.full((L,), float(motion_weight) / float(L), device=self.device, dtype=torch.float32))

        # 将所有 clip 首尾相接成一条大张量，行索引隐含「属于哪段轨迹」已由 next 的计算方式保证不穿轨。
        all_obs_np = np.concatenate(obs_segments, axis=0)
        self.all_obs = torch.tensor(all_obs_np, dtype=torch.float32, device=self.device)
        next_rows = []
        for seg in obs_segments:
            L = seg.shape[0]
            idx = np.arange(L, dtype=np.int64)
            nxt = idx + 1
            # 最后一帧的「下一帧」仍为最后一帧，避免索引越界或与下一条 clip 首帧相连。
            nxt[-1] = nxt[-1] - 1  # hold last frame (no cross-clip wrap)
            next_rows.append(seg[nxt])
        self.all_next_obs = torch.tensor(np.concatenate(next_rows, axis=0), dtype=torch.float32, device=self.device)

        per_frame = torch.cat(weight_frames)
        self.per_frame_weights = per_frame / per_frame.sum()

    def _validate_first_or_match(self, data: np.lib.npyio.NpzFile, source: str) -> None:
        """第一次加载：缓存名字表与 dt；之后的 npz 必须与之严格一致。"""
        if not hasattr(self, "_dof_names"):
            self._dof_names = data["dof_names"].tolist()
            self._key_body_names = data["key_body_names"].tolist()
            self.dt = 1.0 / float(data["fps"])
            return
        if data["dof_names"].tolist() != self._dof_names:
            raise ValueError(f"Motion DOF names do not match: {source}")
        if data["key_body_names"].tolist() != self._key_body_names:
            raise ValueError(f"Motion key body names do not match: {source}")
        if not np.isclose(1.0 / float(data["fps"]), self.dt):
            raise ValueError(f"Motion fps does not match first clip: {source}")

    @staticmethod
    def _rows_to_amp_features(
        data: np.lib.npyio.NpzFile,
        dof_indexes: Sequence[int] | None,
        key_body_indexes: Sequence[int] | None,
    ) -> np.ndarray:
        """构造单帧 AMP 特征矩阵 ``(T, F)``，与 proprioception ``torch.cat`` 语义一致。"""
        dof_positions = np.asarray(data["dof_positions"], dtype=np.float64)
        dof_velocities = np.asarray(data["dof_velocities"], dtype=np.float64)
        kbl = np.asarray(data["key_body_positions_local"], dtype=np.float64)
        T = dof_positions.shape[0]

        # 可选：只对齐机器人子集自由度（须与仿真里 motion_dof_indexes 一致）。
        if dof_indexes is not None:
            di = list(dof_indexes)
            dof_positions = dof_positions[:, di]
            dof_velocities = dof_velocities[:, di]

        nb = len(data["key_body_names"])
        # 离线数据可为 (T, nb*3) 或 (T, nb, 3)，统一到 (T, nb, 3) 再做刚体子集选取。
        if kbl.ndim == 2:
            if kbl.shape[1] % 3 != 0:
                raise ValueError("key_body_positions_local last dim must be multiple of 3")
            if kbl.shape[1] // 3 != nb:
                raise ValueError("key_body_positions_local body count does not match key_body_names")
            kbl = kbl.reshape(T, nb, 3)
        elif kbl.ndim == 3:
            if kbl.shape[1] != nb:
                raise ValueError("key_body_positions_local middle dim does not match key_body_names")
        else:
            raise ValueError(f"Unexpected key_body_positions_local shape: {kbl.shape}")

        if key_body_indexes is not None:
            bi = list(key_body_indexes)
            kbl = kbl[:, bi, :]

        kbl_flat = kbl.reshape(T, -1)
        return np.concatenate([dof_positions, dof_velocities, kbl_flat], axis=-1).astype(np.float32)

    @staticmethod
    def _find_project_root(start_dir: str) -> str:
        """自下而上寻找包含 ``datasets`` 的目录作为相对路径的工程根。"""
        current_dir = os.path.abspath(start_dir)
        while True:
            if os.path.isdir(os.path.join(current_dir, "datasets")):
                return current_dir
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                return os.path.abspath(start_dir)
            current_dir = parent_dir

    @staticmethod
    def _parse_manifest(config_file: str, project_root: str) -> list[tuple[str, float]]:
        """解析 YAML，得到 ``(npz绝对路径, 权重)`` 列表。"""
        with open(config_file) as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}
        raw_entries = config.get("motion_files", [])
        weights = config.get("weights")
        resolved: list[tuple[str, float]] = []
        for index, raw_entry in enumerate(raw_entries):
            if isinstance(raw_entry, str):
                weight = 1.0 if weights is None else float(weights[index])
                path = raw_entry
            else:
                path = raw_entry.get("file", raw_entry.get("path"))
                weight = float(raw_entry.get("weight", 1.0))
            if not path:
                raise ValueError(f"Invalid motion_files entry in {config_file}: {raw_entry}")
            abs_path = path if os.path.isabs(path) else os.path.join(project_root, path)
            resolved.append((abs_path, weight))
        if not resolved:
            raise ValueError(f"No motion_files entries in {config_file}")
        return resolved

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("Motion weights must be non-negative with at least one positive value")
        return weights / np.sum(weights)

    @property
    def feature_dim(self) -> int:
        return self.amp_obs_dim

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """按 ``per_frame_weights`` 有放回地抽帧索引，切分为与 PPO AMP policy gen 对齐的若干个 mini-batch。

        必须与 ``AMPPPO.update`` 里 ``num_mini_batch * mini_batch_size`` 与 policy replay 侧一致，
        否则 ``zip(generator, ..., ...)`` 会提前耗尽或语义错位。
        """
        total = num_mini_batch * mini_batch_size
        idx = torch.multinomial(self.per_frame_weights, total, replacement=True)
        obs = self.all_obs[idx]
        nxt = self.all_next_obs[idx]
        for i in range(num_mini_batch):
            sl = slice(i * mini_batch_size, (i + 1) * mini_batch_size)
            yield obs[sl], nxt[sl]
