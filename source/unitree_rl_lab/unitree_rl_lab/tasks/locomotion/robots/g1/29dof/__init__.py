import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        # 关键点：保留原始 Velocity 任务继续使用 MLP actor，避免影响已有实验。
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-GRU",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 关键点：GRU 任务绑定到 1 帧观测环境配置，避免影响默认 Velocity 任务。
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotGruObs1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotGruObs1PlayEnvCfg",
        # 关键点：新增独立任务使用 GRU actor，便于与 MLP 基线并行对比。
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G1GruActorPPORunnerCfg",
    },
)
