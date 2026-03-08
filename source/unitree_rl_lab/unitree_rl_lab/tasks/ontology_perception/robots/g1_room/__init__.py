"""
Purpose: Register the G1 ontology perception room task with Gym.
Main contents: environment entry points for training and play/collection configurations.
"""

import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Ontology-Perception-Room-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ontology_perception_env_cfg:G1OntologyPerceptionEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ontology_perception_env_cfg:G1OntologyPerceptionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

