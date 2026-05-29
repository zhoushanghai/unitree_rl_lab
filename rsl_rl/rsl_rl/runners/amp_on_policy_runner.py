# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 专用的 On-policy runner。

相较于 ``OnPolicyRunner``，在每步环境里额外维护一条 **AMP 时间序列**，并调用：
    - ``alg.act_amp(amp_t)``：在 ``step`` **前**记下当前 AMP；
    - ``alg.process_amp_step(amp_{t+1})``：在 ``step`` **后**把 ``(amp_t, amp_{t+1})`` 塞进 ``AMPReplayBuffer``。

可选地，按 ``discriminator.predict_reward`` 把 **style 奖励** 与任务奖励凸组合（详见 ``train_cfg[\"amp_runner\"]``）。
AMP 向量来源优先级：
    1. ``observations[observation_key]``（例如 TensorDict 里单独一组 ``\"amp\"``）；
    2. ``extras[\"amp_obs\"]`` 等 Isaac Lab 常见写法。
"""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


def extract_amp_observation(
    observations: TensorDict,
    extras: dict,
    *,
    observation_key: str,
    extras_key: str | None,
    device: str | torch.device,
) -> torch.Tensor:
    """从观测或 extras 中取形状 ``[num_envs, amp_dim]`` 的 AMP 张量。"""
    if observation_key in observations.keys():  # type: ignore[arg-type]
        tens = observations.get(observation_key)
        if tens is None:
            raise KeyError(f'TensorDict key {observation_key!r} is missing or None.')
        return tens.clone().detach().to(device=device, dtype=torch.float32)
    if extras_key is not None and extras_key in extras:
        tensor = extras[extras_key].clone().detach().to(device=device, dtype=torch.float32)
        # 容错：单行 AMP 向量扩成 batch 维（通常不应发生）。
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor
    ek = repr(extras_key)
    raise KeyError(
        f"AMP observations not found: TensorDict[{observation_key!r}] absent and extras[{ek}] unavailable; "
        'set ``amp_runner.observation_key`` / ``extras_key`` in ``train_cfg``.',
    )


class AmpOnPolicyRunner(OnPolicyRunner):
    """构建 ``AMPPPO.construct_algorithm`` 并运行带 AMP 钩子的 ``learn`` 循环。"""

    alg: AMPPPO

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        self._configure_multi_gpu()

        obs = self.env.get_observations()

        alg_class: type = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore[assignment]
        if not hasattr(alg_class, "construct_algorithm"):
            raise TypeError(f"{alg_class.__name__} must implement construct_algorithm (use AMPPPO).")
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """单次迭代内：对每个 env step 同时推进 PPO 转移与 AMP 转移。"""

        amp_cfg = self.cfg.get("amp_runner", {})
        observation_key = str(amp_cfg.get("observation_key", "amp"))
        extras_key_amp = amp_cfg.get("extras_key", "amp_obs")
        use_style_reward = bool(amp_cfg.get("use_style_reward", True))
        # style_coef=0.5 ↔ 与原 amp-rsl-rl 演示里「一半任务一半风格」对齐，可按需改写或关闭 ``use_style_reward``。
        style_coef = float(amp_cfg.get("style_reward_coef", 0.5))

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        # 第一轮循环入口处：用上一步 reset 之后的 AMP（与 amp-rsl-rl 在进入 rollout 前的 ``amp_obs = obs[\"amp\"]`` 一致）。
        amp_obs = extract_amp_observation(
            obs, {}, observation_key=observation_key, extras_key=extras_key_amp, device=self.device
        )

        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")  # noqa: T201
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout 内按步累计 env 任务奖励与 style 标量，供 TensorBoard 分项展示（与 ``Train/mean_reward`` 的回合累计不同）。
            sum_task_reward = 0.0
            sum_style_reward = 0.0
            n_rollout_steps = 0
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)

                    # 与 AMP 论文/amp-rsl-rl 节拍一致：在施加动作前先缓存本步 AMP，再与环境交互。
                    self.alg.act_amp(amp_obs.to(self.device))

                    obs0, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    obs = obs0.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    extras = extras or {}

                    next_amp = extract_amp_observation(
                        obs, extras, observation_key=observation_key, extras_key=extras_key_amp, device=self.device
                    )

                    # 混合前记录任务奖励；style 为 ``predict_reward`` 的原始尺度（与 ``style_reward_coef`` 无关）。
                    sum_task_reward += float(rewards.float().mean().item())
                    if use_style_reward:
                        # predict_reward 在 ``train_mode`` 的判别器上跑 forward；rollout 子图仍可无梯度累积。
                        style = self.alg.discriminator.predict_reward(amp_obs, next_amp)
                        if rewards.dim() >= 2 and style.dim() == 1:
                            style = style.unsqueeze(-1).expand_as(rewards)
                        elif rewards.dim() == style.dim():
                            pass
                        else:
                            style = style.view_as(rewards)
                        sum_style_reward += float(style.float().mean().item())
                        rewards = (1.0 - style_coef) * rewards + style_coef * style.to(rewards.dtype)
                    n_rollout_steps += 1

                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.alg.process_amp_step(next_amp)
                    amp_obs = next_amp

                    # 当前 AMPPPO 不带 RND；占位保持 Logger API 与原 runner 对齐。
                    intrinsic_rewards = None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            extra_scalars: dict[str, float] | None = None
            if n_rollout_steps > 0:
                extra_scalars = {
                    "Amp/mean_task_reward_step": sum_task_reward / n_rollout_steps,
                }
                if use_style_reward:
                    extra_scalars["Amp/mean_style_reward_step"] = sum_style_reward / n_rollout_steps
            if isinstance(self.alg, AMPPPO) and self.alg._last_amp_log_scalars:
                if extra_scalars is None:
                    extra_scalars = {}
                extra_scalars.update(self.alg._last_amp_log_scalars)

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if bool(self.cfg["algorithm"].get("rnd_cfg")) else None,
                extra_scalars=extra_scalars,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"),  # type: ignore[arg-type]
            )
            self.logger.stop_logging_writer()
