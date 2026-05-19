import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


@configclass
class ApfCfg:
    """APF 参数配置（用于速度命令修正）。"""

    # 势场窗口参数
    R: float = 1.0  # 势场生效半径（m）
    rho: float = 0.2  # 近区偏移（m），d<=rho 时权重保持高值
    beta: float = 2.0  # 权重衰减指数
    k: float = 1.0  # 排斥强度系数
    alpha: float = 0.9 # EMA 平滑系数

    # 排斥增量平面模长上限（m/s）：限制 ||Δv_xy||
    delta_vel_xy_max: float = 1.0
    # 合成后平面速度模长上限（m/s）：限制 ||v_cmd + Δv||
    lin_vel_xy_max: float = 1.0

    # 无有效碰撞点时如何处理 APF 的 EMA 内部状态：
    # - "zero" : 直接清零（推荐，且能保证无碰撞时 v_out 严格等于 v_cmd）
    # - "decay": 按保留系数衰减（保留少量历史惯性）
    no_contact_delta_mode: str = "zero"
    no_contact_delta_decay_factor: float = 0.5


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 每个并行环境一个障碍物：
    # - 这里只负责“资产声明”（尺寸、质量、材质、碰撞体）。
    # - 真正的“出现时机与位置”由 EventCfg 中的 reset/interval 事件控制。
    # - init_state 放在地下，仅用于首次加载时避免挡路。
    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 障碍专用接触传感器（碰撞检测 + 碰撞点记录入口）：
    # - prim_path 仍覆盖机器人各刚体
    # - 但通过 filter_prim_paths_expr 强制只统计与 Obstacle 的接触
    # - 开启 track_contact_points 后可直接读取 contact_pos_w（真实碰撞点坐标）
    obstacle_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle"],
        history_length=3,
        track_air_time=False,
        track_contact_points=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
    # reset 阶段：
    # 1) 为每个 env 独立采样一次触发时刻 t ~ U(1, 8)s；
    # 2) 障碍先藏到地下，确保开局是“纯速度跟踪”，不被障碍干扰。
    reset_obstacle_spawn = EventTerm(
        func=mdp.reset_obstacle_spawn_timer_and_stash,
        mode="reset",
        params={
            "delay_range_s": (1.0, 4.0),
            "asset_cfg": SceneEntityCfg("obstacle"),
            "stash_z_offset_m": -5.0,
        },
    )
    # reset 时清空碰撞点缓存（每个 env 都从空缓存开始）。
    reset_obstacle_collision_points = EventTerm(
        func=mdp.reset_obstacle_collision_point_cache,
        mode="reset",
        params={
            "max_points": 10,
        },
    )

    # interval
    # interval 阶段高频轮询（50Hz，与控制频率 step_dt=0.02s 对齐）：
    # - 每个“速度命令刷新周期”开始时重采样一次障碍触发延时；
    # - 当某 env 在该周期内到达采样触发时刻后，执行一次瞬移；
    # - 瞬移目标优先取“当前平面速度方向前方 0.8m + 左右随机偏移”；
    # - 低速时回退到机身前向，侧向偏移仍生效；
    # - 事件函数内部打标记，保证“每周期每 env 只触发一次”。
    spawn_obstacle_forward_once = EventTerm(
        func=mdp.spawn_obstacle_forward_once,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "forward_offset_m": 0.8,
            "lateral_offset_range_m": (-0.2, 0.2),
            "min_speed_for_velocity_dir": 0.05,
            "obstacle_half_height_m": 0.5,
            "delay_range_s": (1.0, 4.0),
            "command_refresh_interval_s": 10.0,
            "stash_z_offset_m": -5.0,
            "asset_cfg": SceneEntityCfg("obstacle"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # interval 高频记录：从障碍专用 contact sensor 更新碰撞点缓存。
    # 这里只做检测与记录，不施加 APF、不改速度命令。
    update_obstacle_collision_points = EventTerm(
        func=mdp.update_obstacle_collision_point_cache,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_contact_forces"),
            "robot_cfg": SceneEntityCfg("robot"),
            "max_points": 10,
            "force_threshold": 1.0,
            "merge_distance_m": 0.05,
            "keep_radius_m": 1.0,
        },
    )
    # reset 时清空 APF 内部状态（EMA 缓存 + 调试变量）。
    reset_apf_state = EventTerm(
        func=mdp.reset_apf_velocity_state,
        mode="reset",
        params={},
    )
    # interval 同步执行 APF 命令计算：
    # - 输入：当前 sampled velocity command + 碰撞点缓存
    # - 输出：APF 修正后的速度缓存（不覆盖原始采样命令）
    apply_apf_to_base_velocity = EventTerm(
        func=mdp.apply_apf_to_velocity_command,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "base_velocity",
            "robot_cfg": SceneEntityCfg("robot"),
            "k": 1.0,
            "R": 1.0,
            "rho": 0.2,
            "beta": 2.0,
            "alpha": 0.5,
            "lin_vel_xy_max": 1.0,
            "no_contact_delta_mode": "zero",
            "no_contact_delta_decay_factor": 0.5,
        },
    )
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 5.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 关键：模型输入使用 APF 修正前的原始采样命令（旧 cmd），
        # 与控制执行/奖励所使用的新 cmd（APF 后）解耦。
        velocity_commands = ObsTerm(func=mdp.apf_raw_velocity_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # Critic 同样对齐“输入模型看旧 cmd”的约定，避免 actor/critic 命令语义不一致。
        velocity_commands = ObsTerm(func=mdp.apf_raw_velocity_commands, params={"command_name": "base_velocity"})
        # 与 Actor 保持一致：Critic 也输入同一份碰撞点槽位，稳定 AC 观测语义。
        obstacle_collision_slots = ObsTerm(
            func=mdp.obstacle_collision_slots_base_xy,
            params={"max_points": 10},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class GruObservationsCfg(ObservationsCfg):
    """GRU 专用观测：在 policy 输入中额外加入碰撞点槽位。"""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # 仅 GRU 策略输入碰撞点：每槽 (x_b, y_b, valid)。
        obstacle_collision_slots = ObsTerm(
            func=mdp.obstacle_collision_slots_base_xy,
            params={"max_points": 10},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        # 关键点：reward 跟踪 APF 后目标速度；原始命令保持不变给模型观测使用。
        func=mdp.track_lin_vel_xy_yaw_frame_exp_apf,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "use_apf_command": True},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_apf,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "use_apf_command": True},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "use_apf_command": True,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    apf: ApfCfg = ApfCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.obstacle_contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 将 APF 配置同步到事件参数，避免一处改参数另一处遗漏。
        self.events.apply_apf_to_base_velocity.params["k"] = self.apf.k
        self.events.apply_apf_to_base_velocity.params["R"] = self.apf.R
        self.events.apply_apf_to_base_velocity.params["rho"] = self.apf.rho
        self.events.apply_apf_to_base_velocity.params["beta"] = self.apf.beta
        self.events.apply_apf_to_base_velocity.params["alpha"] = self.apf.alpha
        self.events.apply_apf_to_base_velocity.params["delta_vel_xy_max"] = self.apf.delta_vel_xy_max
        self.events.apply_apf_to_base_velocity.params["lin_vel_xy_max"] = self.apf.lin_vel_xy_max
        self.events.apply_apf_to_base_velocity.params["no_contact_delta_mode"] = self.apf.no_contact_delta_mode
        self.events.apply_apf_to_base_velocity.params["no_contact_delta_decay_factor"] = (
            self.apf.no_contact_delta_decay_factor
        )
        # 将障碍刷新周期与命令重采样周期对齐，满足“命令刷新后障碍也刷新”的需求。
        self.events.spawn_obstacle_forward_once.params["command_refresh_interval_s"] = (
            self.commands.base_velocity.resampling_time_range[0]
        )

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Play 默认单环境，避免开局相机落在多环境阵列中导致“看不到机器人”。
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # 开启 IsaacLab 命令可视化，恢复 play 里的速度箭头显示。
        # 注意：部分环境下可能重新出现 /Visuals/Command/* 的告警日志。
        self.commands.base_velocity.debug_vis = True
        # Play 默认视角：固定看向 env_0 附近机器人，避免启动时看不到主体。
        # eye/lookat 均为世界坐标；该设置只影响可视化，不影响训练/控制逻辑。
        # 正上方俯视：相机位于机器人上空，便于观察平面轨迹与局部避障关系。
        self.viewer.eye = (0.0, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class RobotGruObs1EnvCfg(RobotEnvCfg):
    """G1 GRU task env cfg with single-frame observations."""
    # 仅 GRU 任务使用带碰撞点输入的 policy 观测配置；默认 MLP 任务不包含该输入。
    observations: GruObservationsCfg = GruObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 关键点：GRU 任务仅使用当前帧观测（history_length=1），时序信息由 RNN hidden state 负责。
        self.observations.policy.history_length = 5
        self.observations.critic.history_length = 5


@configclass
class RobotGruObs1PlayEnvCfg(RobotPlayEnvCfg):
    """Play cfg for G1 GRU task with single-frame observations."""
    # Play 也保持与 GRU 训练一致：使用带碰撞点输入的 policy 观测配置。
    observations: GruObservationsCfg = GruObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 关键点：推理/播放阶段保持与训练一致，同样使用 1 帧观测输入。
        self.observations.policy.history_length = 5
        self.observations.critic.history_length = 5
