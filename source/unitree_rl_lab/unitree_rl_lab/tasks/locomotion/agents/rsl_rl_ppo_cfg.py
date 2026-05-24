# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000000
    save_interval = 300
    experiment_name = ""  # same as task name
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1GruActorPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner config for G1 velocity task with GRU actor."""

    def __post_init__(self):
        # 关键点：仅把 actor 的模型类型切到 RNNModel，critic 继续使用 MLP，减少改动面。
        self.actor.class_name = "RNNModel"
        # 关键点：显式指定使用 GRU（不是 LSTM）。
        self.actor.rnn_type = "gru"
        self.actor.rnn_hidden_dim = 256
        self.actor.rnn_num_layers = 1
        # True: MLP_in = concat(gru_hidden, obs)；False: MLP_in = gru_hidden only
        self.actor.rnn_concat_obs = True
        # 仅让碰撞点进入 GRU 分支，不进入后续 MLP skip。
        # 当前 policy history_length=5，且 obstacle_collision_slots=10*(x,y,valid)=30 dims/帧，
        # 因此需要从 obs skip 尾部排除 30*5=150 dims。
        self.actor.rnn_concat_obs_exclude_tail_dims = 150
