# Copyright (c) 2025, Istituto Italiano di Tecnologia
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Discriminator structure and AMP losses adapted from ami-iit/amp-rsl-rl (BSD-3-Clause).

"""AMP 判别器模块。

输入约定（与 amp-rsl-rl 一致）：
    ``x = concat(state, next_state)``，其中 ``state`` / ``next_state`` 各为单行 AMP 特征，
    ``input_dim`` = ``2 * amp_obs_dim``.

前向时对两半分别可做经验归一化（``EmpiricalNormalization``），再送入共享 ``trunk + linear``.
``forward`` 里合并后再过 trunk：这样判别器对 ``(s, s')`` 联合分布打分。

损失：
    - MSE（默认，ultra_run 同款）：policy 片段标签 -1，expert 标签 +1，并配 ``(||∇D||-0)^2`` 梯度惩罚；
    - 可选 Wasserstein 变种（实验性）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd

from rsl_rl.modules.normalization import EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """判别 ``concat(state, next_state)`` 来自 expert replay 还是 policy AMP replay。"""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: str | torch.device = "cpu",
        loss_type: str = "MSE",
        eta_wgan: float = 0.3,
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        layers: list[nn.Module] = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        self.linear = nn.Linear(final_in_dim, 1)

        self.empirical_normalization = empirical_normalization
        # 两半输入各自长度为 amp_obs_dim；归一化在单步向量上累积统计。
        amp_obs_dim = input_dim // 2
        if empirical_normalization:
            self.amp_normalizer: nn.Module = EmpiricalNormalization(shape=[amp_obs_dim])
        else:
            self.amp_normalizer = nn.Identity()

        self.to(self.device)
        self.train()
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type if loss_type is not None else "MSE"
        if self.loss_type == "MSE":
            self.loss_fun = nn.MSELoss()
        elif self.loss_type == "Wasserstein":
            self.loss_fun = None
            self.eta_wgan = eta_wgan
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported: 'MSE', 'Wasserstein'.",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对两半 AMP 向量分别 normalize，再拼接回整块后过 MLP。"""
        state, next_state = torch.split(x, self.input_dim // 2, dim=-1)
        state = self.amp_normalizer(state)
        next_state = self.amp_normalizer(next_state)
        x = torch.cat([state, next_state], dim=-1)

        h = self.trunk(x)
        if self.use_minibatch_std:
            # 把一个标量级别的 batch 内特征 std 广播到每条样本，缓解判别器对小 batch 的过拟合。
            s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)
        return self.linear(h)

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        s = h.float().std(dim=0, unbiased=False).mean()
        return s.expand(h.shape[0], 1).to(h.dtype)

    def predict_reward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Rollout 内风格奖励（TienKung-Lab 二次型，中心在 expert 目标 logit=1）。

        ``style = reward_scale * clamp(1 - 0.25 * (d - 1)^2, min=0)``：仅当 ``d`` 接近 1（像 expert）时分高；
        policy 被判假（``d`` 明显小于 0）时 reward→0，与 MSE(-1/+1) 标签语义一致。
        """
        with torch.inference_mode():
            was_training = self.training
            self.eval()
            d = self.forward(torch.cat([state, next_state], dim=-1))

            if self.loss_type == "Wasserstein":
                d = torch.tanh(self.eta_wgan * d)
                reward = self.reward_scale * torch.exp(d)
            else:
                # expert 训练标签为 1；``d=1`` 时 style 取满 ``reward_scale``，远离 1 则为 0。
                reward = self.reward_scale * torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)

            if was_training:
                self.train()
            return reward.squeeze()

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """在 optimizer.step 之后、用 **未代入 forward 的另一份 raw** batch 更新 running mean/var（若启用）。"""
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch)  # type: ignore[union-attr]

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        sample_amp_expert: tuple[torch.Tensor, torch.Tensor],
        sample_amp_policy: tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(amp_classification_loss, grad_pen_loss)``；后者在 ``compute_grad_pen`` 中对 expert 一端构造。"""
        # 梯度惩罚分支里对传入的拼接专家样本做 normalize（与判别器内部的 forward 语义对齐）。
        sample_amp_expert = tuple(self.amp_normalizer(s) for s in sample_amp_expert)  # type: ignore[union-attr]
        sample_amp_policy = tuple(self.amp_normalizer(s) for s in sample_amp_policy)  # type: ignore[union-attr]
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            policy_states=sample_amp_policy,
            lambda_=lambda_,
        )
        if self.loss_type == "MSE":
            # ultra_run 风格：expert 目标 +1，policy 目标 -1。
            expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
        elif self.loss_type == "Wasserstein":
            amp_loss = self.wgan_loss(policy_d=policy_d, expert_d=expert_d)
        else:
            raise ValueError(self.loss_type)

        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: tuple[torch.Tensor, torch.Tensor],
        policy_states: tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10,
    ) -> torch.Tensor:
        expert = torch.cat(expert_states, -1)

        if self.loss_type == "Wasserstein":
            # 混合真假插值点上对输出求梯度范数正则（WGAN-GP 类）。
            policy = torch.cat(policy_states, -1)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            alpha = alpha.expand_as(expert)
            data = alpha * expert + (1 - alpha) * policy
            data = data.detach().requires_grad_(True)
            h = self.trunk(data)
            if self.use_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
                h = torch.cat([h, s], dim=-1)
            scores = self.linear(h)
            grad = autograd.grad(
                outputs=scores,
                inputs=data,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()

        if self.loss_type == "MSE":
            # ultra_run 风格：对 expert 样本做 ``lambda * (||∇D||-0)^2``，不乘 0.5 系数。
            data = expert.detach().requires_grad_(True)
            h = self.trunk(data)
            if self.use_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
                h = torch.cat([h, s], dim=-1)
            scores = self.linear(h)

            grad = autograd.grad(
                outputs=scores,
                inputs=data,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()

        raise ValueError(self.loss_type)

    def wgan_loss(self, policy_d: torch.Tensor, expert_d: torch.Tensor) -> torch.Tensor:
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()
