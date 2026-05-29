# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + AMP：在标准 PPO 的 ``update`` 中插入判别器与 proprioception Expert 数据。

数据流（单个 learn iteration 内）：
    1. Rollout：``act`` → env → ``process_env_step``；并行地 ``act_amp`` → ``process_amp_step`` 写入 ``AMPReplayBuffer``。
    2. ``compute_returns``：仅对 PPO 的 ``RolloutStorage``。
    3. ``update``：对每个 mini-batch，
       - 重算 surrogate / value（与父类 PPO 一致）；
       - 同步从 ``amp_storage``、`ProprioceptionMotionExpert` 各取同等形状 batch；
       - 判别器在一次 forward 里拼接 policy 与 expert 两半，计算 ``amp_loss + grad_pen``；
       - ``loss = ppo_loss + amp_loss + grad_pen``，**一次 backward**；
       - 只对 actor/critic（及本实现中加上的判别器）做 grad clip / ``optimizer.step``。

与父类差异：
    - 使用 **单个 Adam**（多 param group）包含 actor、critic、判别器 trunk/head；
    - **RNN**：与 ``amp-rsl-rl`` 一致——PPO 用 ``recurrent_mini_batch_generator``；AMP policy/expert 仍用
      ``num_envs * horizon // num_mini_batches`` 的扁平 batch 与 PPO 迭代步数 ``zip``；
    - **仍不支持** RND、symmetry（与 AMP 单优化器组合未实现）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.utils.proprio_motion_loader import ProprioceptionMotionExpert


class AMPPPO(PPO):
    """Proximal Policy Optimization + Adversarial Motion Priors（判别器 + Expert YAML/npz）."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: AMPDiscriminator,
        amp_expert: ProprioceptionMotionExpert,
        amp_replay_buffer_size: int,
        amp_grad_pen_lambda: float = 10.0,
        amp_discriminator_trunk_weight_decay: float = 1e-4,
        amp_discriminator_head_weight_decay: float = 1e-2,
        **ppo_kwargs,
    ) -> None:
        # 先构造父类 PPO（会创建仅含 actor/critic 的 optimizer，下面立刻替换为含判别器的版本）。
        super().__init__(actor, critic, storage, **ppo_kwargs)

        self.discriminator = discriminator.to(self.device)
        self.amp_expert = amp_expert
        self.amp_grad_pen_lambda = amp_grad_pen_lambda
        self.amp_obs_dim = amp_expert.amp_obs_dim

        self.amp_storage = AMPReplayBuffer(
            obs_dim=self.amp_obs_dim,
            buffer_size=amp_replay_buffer_size,
            device=self.device,
        )
        # 供 runner 在 ``Logger.log`` 中写入本轮 update 的 policy / expert logit 均值。
        self._last_amp_log_scalars: dict[str, float] = {}
        # 配对缓存：上一轮 ``act_amp`` 写入的状态，待用 ``process_amp_step(next)`` flush。
        self._amp_pending: torch.Tensor | None = None

        # AMP 惯例：判别器与 policy 同步更新（amp-rsl-rl 同款），故合并进同一 optimizer；
        # trunk/head 使用不同 weight decay 以稳定判别器。
        opt_name = ppo_kwargs.get("optimizer", "adam")
        self.optimizer = resolve_optimizer(opt_name)(
            [
                {"params": self.actor.parameters(), "name": "actor"},
                {"params": self.critic.parameters(), "name": "critic"},
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": amp_discriminator_trunk_weight_decay,
                    "name": "amp_trunk",
                },
                {
                    "params": self.discriminator.linear.parameters(),
                    "weight_decay": amp_discriminator_head_weight_decay,
                    "name": "amp_head",
                },
            ],
            lr=self.learning_rate,
        )  # type: ignore[arg-type]

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        """在 ``env.step`` **之前** 调用：记下当前决策时刻的 AMP 向量 ``amp_t``."""
        self._amp_pending = amp_obs.detach()

    def process_amp_step(self, next_amp_obs: torch.Tensor) -> None:
        """在拿到 ``step`` 后新观测对应的 ``amp_{t+1}`` 时写入一条 policy AMP 转移。"""
        if self._amp_pending is None:
            raise RuntimeError("act_amp() must be called before each process_amp_step().")
        self.amp_storage.insert(self._amp_pending, next_amp_obs.detach())
        self._amp_pending = None

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def update(self) -> dict[str, float]:
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("AMPPPO does not compose with RND or symmetry extensions in this revision.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        sum_policy_logit = 0.0
        sum_expert_logit = 0.0
        n_amp_log_batches = 0

        # 与父类 PPO、ami-iit/amp-rsl-rl AMP_PPO：RNN 走轨迹 + mask + hidden_states。
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 与 amp-rsl-rl 相同：AMP 每步样本数与展平后每个 PPO mini-batch 的转移条数一致（与是否 RNN 无关）。
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        num_amp_batches = self.num_learning_epochs * self.num_mini_batches
        amp_policy_gen = self.amp_storage.feed_forward_generator(num_amp_batches, mini_batch_size, allow_replacement=True)
        amp_expert_gen = self.amp_expert.feed_forward_generator(num_amp_batches, mini_batch_size)

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_gen, amp_expert_gen):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore[arg-type]

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore[arg-type]
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # --------- AMP：policy / expert 转移对 ----------
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            policy_state = policy_state.to(self.device)
            policy_next_state = policy_next_state.to(self.device)
            expert_state = expert_state.to(self.device)
            expert_next_state = expert_next_state.to(self.device)

            # 供 ``update_normalization``：与 forward 用同一批数据但未误用已归一化张量累积统计。
            policy_state_raw = policy_state.detach().clone()
            policy_next_state_raw = policy_next_state.detach().clone()
            expert_state_raw = expert_state.detach().clone()
            expert_next_state_raw = expert_next_state.detach().clone()

            b_pol = policy_state.size(0)
            # 一半 batch 来自 policy AMP，另一半来自 expert，拼成判别器单次 forward（提高 GPU 吞吐）。
            disc_in = torch.cat(
                [
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ],
                dim=0,
            )
            discriminator_output = self.discriminator(disc_in)
            policy_d, expert_d = discriminator_output[:b_pol], discriminator_output[b_pol:]

            with torch.no_grad():
                n_amp_log_batches += 1
                sum_policy_logit += float(policy_d.mean().item())
                sum_expert_logit += float(expert_d.mean().item())

            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                sample_amp_expert=(expert_state, expert_next_state),
                sample_amp_policy=(policy_state, policy_next_state),
                lambda_=self.amp_grad_pen_lambda,
            )

            loss = ppo_loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            # 判别器也 clip，避免 AMP 分支梯度爆炸拖垮整张图（阈值与 actor/critic 共用 max_grad_norm）。
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.discriminator.update_normalization(
                expert_state_raw,
                expert_next_state_raw,
                policy_state_raw,
                policy_next_state_raw,
            )

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates

        self.storage.clear()

        n_log = max(n_amp_log_batches, 1)
        self._last_amp_log_scalars = {
            "Amp/update/policy_logit_mean": sum_policy_logit / n_log,
            "Amp/update/expert_logit_mean": sum_expert_logit / n_log,
        }

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
        }

    def save(self) -> dict:
        saved = super().save()
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "discriminator": True,
            }

        load_cfg_merged = dict(load_cfg)
        load_discriminator = load_cfg_merged.pop("discriminator", True)

        iteration_loaded = super().load(loaded_dict, load_cfg_merged, strict)

        if load_discriminator and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        return iteration_loaded

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        model_params = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self.discriminator.state_dict(),
        ]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """DDP：对 actor、critic、判别器的梯度做 **同序** all_reduce 再写回各自 param.grad。"""
        if not self.is_multi_gpu:
            return
        params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        params = [p for p in params]
        grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is not None:
                n = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + n].view_as(param.grad.data))
                offset += n

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """从 ``train_cfg`` 装配 ``AMPPPO``：必填 ``algorithm.amp_motion_yaml``、``algorithm.amp_obs_dim``及 ``discriminator`` 段落。"""

        if cfg["algorithm"].get("rnd_cfg"):
            raise ValueError("AMPPPO does not support rnd_cfg.")
        if cfg["algorithm"].get("symmetry_cfg"):
            raise ValueError("AMPPPO does not support symmetry_cfg.")

        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]

        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        if cfg["algorithm"].get("rnd_cfg"):
            raise ValueError("AMPPPO does not support rnd_cfg after config resolution.")
        if cfg["algorithm"].get("symmetry_cfg"):
            raise ValueError("AMPPPO does not support symmetry_cfg after config resolution.")

        # ---- AMP / Expert 专有字段（从 algorithm 剔除后余下为标准 PPO 超参）----
        amp_motion_yaml = cfg["algorithm"].pop("amp_motion_yaml")
        amp_obs_dim = int(cfg["algorithm"].pop("amp_obs_dim"))
        amp_replay_buffer_size = int(cfg["algorithm"].pop("amp_replay_buffer_size", 100_000))
        amp_grad_pen_lambda = float(cfg["algorithm"].pop("amp_grad_pen_lambda", 10.0))
        amp_trunk_wd = float(cfg["algorithm"].pop("amp_discriminator_trunk_weight_decay", 1e-4))
        amp_head_wd = float(cfg["algorithm"].pop("amp_discriminator_head_weight_decay", 1e-2))
        dof_idx = cfg["algorithm"].pop("motion_dof_indexes", None)
        kb_idx = cfg["algorithm"].pop("motion_key_body_indexes", None)
        motion_root = cfg["algorithm"].pop("amp_motion_project_root", None)

        disc_cfg = dict(cfg["discriminator"])
        disc_cfg.pop("class_name", None)
        hidden_dims = disc_cfg.pop("hidden_dims")
        reward_scale = float(disc_cfg.pop("reward_scale", 2.0))
        loss_type = disc_cfg.pop("loss_type", "BCEWithLogits")
        empirical_norm = bool(disc_cfg.pop("empirical_normalization", False))

        expert = ProprioceptionMotionExpert(
            motion_yaml_path=amp_motion_yaml,
            amp_obs_dim=amp_obs_dim,
            device=device,
            motion_dof_indexes=list(dof_idx) if dof_idx is not None else None,
            motion_key_body_indexes=list(kb_idx) if kb_idx is not None else None,
            motion_project_root=motion_root,
        )

        # 判别器输入维 = 2 * 单行 AMP（state 与 next_state 拼接）。
        discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim * 2,
            hidden_layer_sizes=hidden_dims,
            reward_scale=reward_scale,
            device=device,
            loss_type=loss_type,
            empirical_normalization=empirical_norm,
            **disc_cfg,
        ).to(device)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")  # noqa: T201
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[assignment]

        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")  # noqa: T201

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: AMPPPO = alg_class(
            actor,
            critic,
            storage,
            discriminator=discriminator,
            amp_expert=expert,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_grad_pen_lambda=amp_grad_pen_lambda,
            amp_discriminator_trunk_weight_decay=amp_trunk_wd,
            amp_discriminator_head_weight_decay=amp_head_wd,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        alg.compile(cfg.get("torch_compile_mode"))
        return alg
