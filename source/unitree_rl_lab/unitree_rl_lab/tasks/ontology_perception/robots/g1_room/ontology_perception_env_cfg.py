"""
Purpose: Configure the manager-based G1 ontology perception room environment.
Main contents: action, observation, reward, event, and termination managers for the fixed-room sensing task.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.ontology_perception import mdp
from unitree_rl_lab.tasks.ontology_perception.robots.g1_room.scene_cfg import CONTACT_SENSOR_NAMES, G1OntologyPerceptionSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for ontology perception."""

    JointPositionAction = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.20,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups for policy, critic, and collection."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        left_hand_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("left_hand_contact")})
        right_hand_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("right_hand_contact")})
        left_forearm_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("left_forearm_contact")})
        right_forearm_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("right_forearm_contact")})
        left_shin_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("left_shin_contact")})
        right_shin_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("right_shin_contact")})
        left_foot_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("left_foot_contact")})
        right_foot_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("right_foot_contact")})
        pelvis_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("pelvis_contact")})
        torso_contact = ObsTerm(func=mdp.contact_magnitude, params={"sensor_cfg": SceneEntityCfg("torso_contact")})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        base_pose = ObsTerm(func=mdp.base_pose)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        local_voxel_gt = ObsTerm(func=mdp.local_voxel_gt)
        explored_mask = ObsTerm(func=mdp.explored_mask)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CollectorCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        action = ObsTerm(func=base_mdp.last_action)
        base_pose = ObsTerm(func=mdp.base_pose)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        left_hand_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("left_hand_contact")})
        right_hand_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("right_hand_contact")})
        left_forearm_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("left_forearm_contact")})
        right_forearm_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("right_forearm_contact")})
        left_shin_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("left_shin_contact")})
        right_shin_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("right_shin_contact")})
        left_foot_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("left_foot_contact")})
        right_foot_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("right_foot_contact")})
        pelvis_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("pelvis_contact")})
        torso_contact = ObsTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("torso_contact")})
        local_voxel_gt = ObsTerm(func=mdp.local_voxel_gt)
        explored_mask = ObsTerm(func=mdp.explored_mask)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    collector: CollectorCfg = CollectorCfg()


@configclass
class EventCfg:
    """Reset events for ontology perception."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Lightweight shaping for ontology perception rollouts."""

    alive = RewTerm(func=mdp.alive_bonus, weight=0.5)
    upright = RewTerm(func=mdp.upright_bonus, weight=0.5)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    torso_contact = RewTerm(
        func=mdp.contact_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("torso_contact"), "threshold": 25.0},
    )


@configclass
class TerminationsCfg:
    """Termination conditions for ontology perception."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_height = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    torso_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("torso_contact"), "threshold": 30.0},
    )
    outside_room = DoneTerm(func=mdp.outside_room_bounds, params={"limit_x": 1.8, "limit_y": 1.8})


@configclass
class G1OntologyPerceptionEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based G1 room task for ontology perception rollouts and data collection."""

    scene: G1OntologyPerceptionSceneCfg = G1OntologyPerceptionSceneCfg(
        num_envs=16,
        env_spacing=6.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (6.5, 6.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.7)

        for sensor_name in CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        self.scene.base_imu.update_period = self.sim.dt


@configclass
class G1OntologyPerceptionPlayEnvCfg(G1OntologyPerceptionEnvCfg):
    """Smaller play/collection variant for single-scene debugging and dataset export."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 6.0

