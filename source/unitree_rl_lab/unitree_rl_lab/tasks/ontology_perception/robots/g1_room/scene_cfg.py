"""
Purpose: Define the fixed room scene, G1 robot asset, and body-region sensors for the ontology perception task.
Main contents: room obstacle constants, voxel grid settings, sensor name mappings, and InteractiveSceneCfg setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as UNITREE_G1_29DOF_CFG


@dataclass(frozen=True)
class CuboidSpec:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    color: tuple[float, float, float]


LOCAL_VOXEL_SIZE = (2.0, 2.0, 2.0)
LOCAL_VOXEL_RESOLUTION = (20, 20, 20)

ROOM_OBJECT_SPECS: dict[str, CuboidSpec] = {
    "floor": CuboidSpec("floor", (4.0, 4.0, 0.1), (0.0, 0.0, -0.05), (0.52, 0.52, 0.55)),
    "left_wall": CuboidSpec("left_wall", (4.0, 0.1, 1.6), (0.0, 1.95, 0.8), (0.83, 0.82, 0.79)),
    "right_wall": CuboidSpec("right_wall", (4.0, 0.1, 1.6), (0.0, -1.95, 0.8), (0.83, 0.82, 0.79)),
    "front_wall": CuboidSpec("front_wall", (0.1, 4.0, 1.6), (1.95, 0.0, 0.8), (0.83, 0.82, 0.79)),
    "back_wall": CuboidSpec("back_wall", (0.1, 4.0, 1.6), (-1.95, 0.0, 0.8), (0.83, 0.82, 0.79)),
    "box_left": CuboidSpec("box_left", (0.45, 0.45, 0.70), (0.95, 0.62, 0.35), (0.68, 0.44, 0.25)),
    "box_right": CuboidSpec("box_right", (0.65, 0.45, 0.90), (1.10, -0.70, 0.45), (0.30, 0.55, 0.70)),
    "overhead_bar": CuboidSpec("overhead_bar", (0.20, 1.20, 0.15), (1.05, 0.0, 1.45), (0.58, 0.62, 0.70)),
}

CONTACT_SENSOR_TO_BODY_NAME: dict[str, str] = {
    "left_hand_contact": "left_rubber_hand",
    "right_hand_contact": "right_rubber_hand",
    "left_forearm_contact": "left_elbow_link",
    "right_forearm_contact": "right_elbow_link",
    "left_shin_contact": "left_knee_link",
    "right_shin_contact": "right_knee_link",
    "left_foot_contact": "left_ankle_roll_link",
    "right_foot_contact": "right_ankle_roll_link",
    "pelvis_contact": "pelvis",
    "torso_contact": "torso_link",
}

CONTACT_SENSOR_NAMES = tuple(CONTACT_SENSOR_TO_BODY_NAME.keys())
EXPLORATION_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
)
SUPPORT_SENSOR_NAMES = ("left_foot_contact", "right_foot_contact")


def _static_cuboid_cfg(spec: CuboidSpec) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.position),
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, roughness=0.6),
        ),
    )


@configclass
class G1OntologyPerceptionSceneCfg(InteractiveSceneCfg):
    """Room scene with static obstacles, G1 robot, and body-region sensors."""

    floor = _static_cuboid_cfg(ROOM_OBJECT_SPECS["floor"])
    left_wall = _static_cuboid_cfg(ROOM_OBJECT_SPECS["left_wall"])
    right_wall = _static_cuboid_cfg(ROOM_OBJECT_SPECS["right_wall"])
    front_wall = _static_cuboid_cfg(ROOM_OBJECT_SPECS["front_wall"])
    back_wall = _static_cuboid_cfg(ROOM_OBJECT_SPECS["back_wall"])
    box_left = _static_cuboid_cfg(ROOM_OBJECT_SPECS["box_left"])
    box_right = _static_cuboid_cfg(ROOM_OBJECT_SPECS["box_right"])
    overhead_bar = _static_cuboid_cfg(ROOM_OBJECT_SPECS["overhead_bar"])

    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    base_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        gravity_bias=(0.0, 0.0, 9.81),
        debug_vis=False,
    )

    left_hand_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_rubber_hand", history_length=3, force_threshold=1.0)
    right_hand_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_rubber_hand", history_length=3, force_threshold=1.0)
    left_forearm_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_elbow_link", history_length=3, force_threshold=1.0)
    right_forearm_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_elbow_link", history_length=3, force_threshold=1.0)
    left_shin_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_knee_link", history_length=3, force_threshold=1.0)
    right_shin_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_knee_link", history_length=3, force_threshold=1.0)
    left_foot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link", history_length=3, force_threshold=1.0, track_air_time=True)
    right_foot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link", history_length=3, force_threshold=1.0, track_air_time=True)
    pelvis_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/pelvis", history_length=3, force_threshold=1.0)
    torso_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link", history_length=3, force_threshold=1.0)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.78, 0.80, 0.84), intensity=3200.0),
    )
