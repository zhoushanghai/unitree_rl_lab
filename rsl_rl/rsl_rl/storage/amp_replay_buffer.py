# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""仿真侧 AMP 转移的环形缓冲。

AMP 判别器对每个样本需要的输入形如 ``concat(amp_obs_t, amp_obs_{t+1})``，因此 replay 必须成对存储。
本缓冲与 trajectory 清空无关：**不清 rollout 时在每步 insert**，仅在整次 ``update`` 后用其中的样本与 expert 对齐训练。
"""

from __future__ import annotations

from collections.abc import Generator

import torch


class AMPReplayBuffer:
    """FIFO 环形缓冲：``insert(states=b_t, next_states=b_{t+1})``，形状均为 ``[num_envs, amp_obs_dim]``。

    与 ``RolloutStorage`` 分工：
        - RolloutStorage：PPO 的 policy/critic 所需整条 horizon；
        - 本缓冲：仅用 AMP 特征维，容量独立（通常数万～十万级），可被覆盖写满后继续环形覆盖最早数据。
    """

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.buffer_size = buffer_size

        # 预分配避免训练循环中频繁 cat / 拷贝。
        self.states = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)

        self.step = 0  # 下一个写入位置
        self.num_samples = 0  # 已有效样本数，最大为 buffer_size

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """将一批配对写入环形区；批量跨越尾部时拆成两段写。"""
        states = states.to(self.device)
        next_states = next_states.to(self.device)

        batch_size = states.shape[0]
        end = self.step + batch_size

        if end <= self.buffer_size:
            self.states[self.step : end] = states
            self.next_states[self.step : end] = next_states
        else:
            # 环形回绕：先填满 [step, buffer_size)，再从 0 写 remainder。
            first_part = self.buffer_size - self.step
            self.states[self.step :] = states[:first_part]
            self.next_states[self.step :] = next_states[:first_part]
            remainder = batch_size - first_part
            self.states[:remainder] = states[first_part:]
            self.next_states[:remainder] = next_states[first_part:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + batch_size)

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        allow_replacement: bool = True,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """为 ``AMPPPO.update`` 产出 ``num_mini_batch`` 个 ``(policy_amp_s, policy_amp_s')``。

        Parameters
        ----------
        allow_replacement
            当真：请求样本数多于当前缓冲存量时允许有放回抽样（训练早期缓冲未满时常为真）。
            当假：直接报错，便于排查 batch 对齐或 insert 遗漏。
        """
        total = num_mini_batch * mini_batch_size

        if total > self.num_samples:
            if not allow_replacement:
                raise ValueError(
                    f"AMP replay: requested {total} samples but buffer has only {self.num_samples}",
                )
            # 通过重复打乱区间再取模实现有放回打乱索引。
            cycles = (total + self.num_samples - 1) // self.num_samples
            big_size = self.num_samples * cycles
            big_perm = torch.randperm(big_size, device=self.device)
            indices = big_perm[:total] % self.num_samples
        else:
            indices = torch.randperm(self.num_samples, device=self.device)[:total]

        for i in range(num_mini_batch):
            batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            yield self.states[batch_idx], self.next_states[batch_idx]

    def __len__(self) -> int:
        return self.num_samples
