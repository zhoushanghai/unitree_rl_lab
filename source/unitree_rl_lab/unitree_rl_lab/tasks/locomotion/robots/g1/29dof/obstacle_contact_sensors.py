"""G1 各连杆独立 obstacle 接触传感器（filter 维恒为 1，contact_pos_w 下标固定为 0）。"""

from isaaclab.sensors import ContactSensorCfg

# 与 URDF 中带碰撞的连杆一致（不含 imu / 相机 / 手等）
G1_OBSTACLE_CONTACT_BODY_NAMES: tuple[str, ...] = (
    # pelvis_contour_link 在 URDF 有碰撞，但导入后无独立 contact reporter，不能挂传感器
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


def obstacle_contact_sensor_name(link_name: str) -> str:
    """场景传感器字段名，例如 obstacle_contact_left_knee_link。"""
    return f"obstacle_contact_{link_name}"


def make_obstacle_contact_sensor_cfg(link_name: str) -> ContactSensorCfg:
    """单连杆传感器：只统计该 link 与 Obstacle 的接触。"""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle"],
        history_length=3,
        track_air_time=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=4,
    )


def iter_obstacle_contact_sensors():
    """(scene_key, link_name, cfg) 迭代。"""
    for link in G1_OBSTACLE_CONTACT_BODY_NAMES:
        yield obstacle_contact_sensor_name(link), link, make_obstacle_contact_sensor_cfg(link)
