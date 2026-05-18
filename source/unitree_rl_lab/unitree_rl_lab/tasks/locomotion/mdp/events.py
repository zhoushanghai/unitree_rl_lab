from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def reset_obstacle_spawn_timer_and_stash(
    env,
    env_ids: torch.Tensor,
    delay_range_s: tuple[float, float] = (1.0, 8.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    stash_z_offset_m: float = -5.0,
):
    """在 reset 阶段做两件事：
    1) 为每个 env 采样本回合唯一触发时刻（1~8s）；
    2) 先把障碍物藏到地下，避免开局立即阻挡机器人。
    """
    if env_ids.numel() == 0:
        return

    obstacle: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 懒初始化（只在首次调用时创建）：
    # - _obstacle_spawned: 该 env 在当前 episode 是否已完成“前方瞬移”
    # - _obstacle_spawn_time_s: 该 env 在当前 episode 的触发时刻（秒）
    # 这样不需要改 env 主循环，保持最小化侵入。
    if not hasattr(env, "_obstacle_spawned"):
        env._obstacle_spawned = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_obstacle_spawn_time_s"):
        env._obstacle_spawn_time_s = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # reset 后都标记为“尚未触发”。
    env._obstacle_spawned[env_ids] = False

    # 每个 episode、每个 env 独立采样触发时间 t ~ U(1, 8)。
    t_min, t_max = delay_range_s
    env._obstacle_spawn_time_s[env_ids] = torch.empty(len(env_ids), device=device).uniform_(t_min, t_max)

    # 障碍物“藏在地下”：
    # - XY 放在 env 原点
    # - Z 下沉 stash_z_offset_m（默认 -5m）
    # - 姿态置单位四元数，速度清零
    # 这样在触发前不会对训练产生碰撞干扰。
    root_state = obstacle.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += stash_z_offset_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def spawn_obstacle_forward_once(
    env,
    env_ids: torch.Tensor | None = None,
    forward_offset_m: float = 0.8,
    obstacle_half_height_m: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """到点后仅一次：将障碍物瞬移到机身“水平 +x”前方。

    关键约束：
    - 触发条件：episode_elapsed_time >= sampled_spawn_time
    - 位置规则：机身水平前方 forward_offset_m（默认 0.8m）
    - 每个 episode 每个 env 只触发一次
    """
    if (not hasattr(env, "_obstacle_spawned")) or (not hasattr(env, "_obstacle_spawn_time_s")):
        return

    # 用局部变量消除 Optional，便于静态检查。
    active_env_ids: torch.Tensor = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long) if env_ids is None else env_ids
    )
    if active_env_ids.numel() == 0:
        return

    # episode 已运行时间（秒）= step 计数 * step_dt
    elapsed_s = env.episode_length_buf.float() * env.step_dt
    to_spawn = (~env._obstacle_spawned[active_env_ids]) & (
        elapsed_s[active_env_ids] >= env._obstacle_spawn_time_s[active_env_ids]
    )
    if not torch.any(to_spawn):
        return

    spawn_env_ids = active_env_ids[to_spawn]
    obstacle: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # 从机器人根姿态四元数提取“世界系前向向量”的 XY 分量，并归一化。
    # 注意这里取的是“水平面前向”，故只用 XY，不使用 pitch/roll 的垂向影响。
    quat_w = robot.data.root_quat_w[spawn_env_ids]  # (w, x, y, z)
    qw, qx, qy, qz = quat_w.unbind(dim=-1)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    fwd_xy = fwd_xy / torch.linalg.norm(fwd_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)

    # 障碍瞬移目标位姿：
    # - XY: 机器人根位置 + 前向单位向量 * forward_offset_m
    # - Z : 贴地放置（env 地面高度 + 障碍半高）
    # - 姿态: 单位四元数
    # - 速度: 清零，避免瞬移遗留速度造成额外动力学扰动
    root_state = obstacle.data.default_root_state[spawn_env_ids].clone()
    root_state[:, 0:2] = robot.data.root_pos_w[spawn_env_ids, 0:2] + forward_offset_m * fwd_xy
    root_state[:, 2] = env.scene.env_origins[spawn_env_ids, 2] + obstacle_half_height_m
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(spawn_env_ids), 1)
    root_state[:, 7:13] = 0.0
    obstacle.write_root_pose_to_sim(root_state[:, 0:7], env_ids=spawn_env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=spawn_env_ids)

    # 标记本回合已触发，确保“只瞬移一次”。
    env._obstacle_spawned[spawn_env_ids] = True


def reset_obstacle_collision_point_cache(
    env,
    env_ids: torch.Tensor,
    max_points: int = 10,
):
    """reset 时初始化/清空每个 env 的碰撞点缓存。

    缓存定义（挂在 env 对象上，后续 APF/可视化可直接复用）：
    - _obstacle_collision_points_w: (num_envs, max_points, 3) 世界系碰撞点
    - _obstacle_collision_slot_valid: (num_envs, max_points) 槽位有效标记
    """
    if env_ids.numel() == 0:
        return

    device = env.device
    num_envs = env.num_envs

    # 懒初始化：首次进入时为全部并行环境分配缓存。
    if (not hasattr(env, "_obstacle_collision_points_w")) or (
        env._obstacle_collision_points_w.shape[1] != max_points
    ):
        env._obstacle_collision_points_w = torch.zeros((num_envs, max_points, 3), dtype=torch.float32, device=device)
        env._obstacle_collision_slot_valid = torch.zeros((num_envs, max_points), dtype=torch.bool, device=device)

    # 仅清理本次 reset 的 env，避免误改其他仍在运行的 env 状态。
    env._obstacle_collision_points_w[env_ids] = 0.0
    env._obstacle_collision_slot_valid[env_ids] = False


def update_obstacle_collision_point_cache(
    env,
    env_ids: torch.Tensor | None = None,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_contact_forces"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_points: int = 10,
    force_threshold: float = 0.3,
    merge_distance_m: float = 0.02,
    keep_radius_m: float = 1.0,
):
    """从障碍专用接触传感器更新碰撞点缓存（仅记录，不做 APF）。

    算法步骤（每次 interval）：
    1) 先按机器人平面距离清理历史点（超过 keep_radius_m 直接失效）
    2) 从 filtered contact sensor 读取 force_matrix_w + contact_pos_w
    3) 力阈值过滤 + NaN 过滤后，逐点写入缓存：
       - 与已有点距离小于 merge_distance_m：做均值合并
       - 否则写入空槽；若满则淘汰“离机器人最远”的槽
    4) 最后按与机器人 XY 距离近->远排序，保持槽位语义稳定
    """
    if (not hasattr(env, "_obstacle_collision_points_w")) or (not hasattr(env, "_obstacle_collision_slot_valid")):
        # 防御式初始化：正常流程会在 reset 事件里初始化，这里兜底避免顺序变更导致崩溃。
        all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        reset_obstacle_collision_point_cache(env=env, env_ids=all_env_ids, max_points=max_points)

    active_env_ids: torch.Tensor = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long) if env_ids is None else env_ids
    )
    if active_env_ids.numel() == 0:
        return

    # 读取缓存引用，减少后续长变量名噪音。
    points_w = env._obstacle_collision_points_w
    valid = env._obstacle_collision_slot_valid

    robot: Articulation = env.scene[robot_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2]

    # Step-1: 按距离淘汰历史碰撞点（无时间 TTL，仅距离生命周期）。
    dist_to_robot = torch.linalg.norm(points_w[active_env_ids, :, :2] - robot_xy[active_env_ids].unsqueeze(1), dim=-1)
    keep_mask = dist_to_robot <= keep_radius_m
    valid[active_env_ids] &= keep_mask
    points_w[active_env_ids] = points_w[active_env_ids].masked_fill(~valid[active_env_ids].unsqueeze(-1), 0.0)

    # Step-2: 读取“只过滤到 Obstacle”的接触传感器。
    if sensor_cfg.name not in env.scene.sensors:
        return
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    forces = contact_sensor.data.force_matrix_w  # 期望形状: (num_envs, num_bodies, M, 3)
    contact_pos_w = contact_sensor.data.contact_pos_w  # 期望形状: (num_envs, num_bodies, M, 3)
    if (forces is None) or (contact_pos_w is None):
        return

    # Step-3: 力阈值筛选，只保留真实接触而非数值噪声。
    force_mags = torch.linalg.norm(forces, dim=-1)  # (num_envs, num_bodies, M)
    strong_contacts = force_mags > force_threshold
    env_idx, body_idx, filter_idx = strong_contacts.nonzero(as_tuple=True)
    if env_idx.numel() == 0:
        return

    # 仅处理 active_env_ids，避免被传入子集 env_ids 时误写其它环境。
    active_mask = torch.isin(env_idx, active_env_ids)
    if not torch.any(active_mask):
        return
    env_idx = env_idx[active_mask]
    body_idx = body_idx[active_mask]
    filter_idx = filter_idx[active_mask]

    new_points = contact_pos_w[env_idx, body_idx, filter_idx, :]  # (N, 3)
    finite_mask = torch.isfinite(new_points).all(dim=-1)
    if not torch.any(finite_mask):
        return
    env_idx = env_idx[finite_mask]
    new_points = new_points[finite_mask]

    # Step-3 continued: 逐个新点做“合并/写入/淘汰”。
    for one_env_id, one_pt in zip(env_idx, new_points):
        env_id = int(one_env_id.item())
        cur_valid = valid[env_id]
        cur_pts = points_w[env_id]

        # 先尝试与已有点合并（仅比较 XY，符合后续 APF 的平面语义）。
        if torch.any(cur_valid):
            active_slots = cur_valid.nonzero(as_tuple=True)[0]
            d_active = torch.linalg.norm(cur_pts[active_slots, :2] - one_pt[:2], dim=-1)
            min_d, min_pos = torch.min(d_active, dim=0)
            if min_d < merge_distance_m:
                merge_slot = active_slots[min_pos]
                cur_pts[merge_slot] = 0.5 * (cur_pts[merge_slot] + one_pt)
                continue

        # 无可合并点：优先写空槽，满了则淘汰最远槽。
        empty_slots = (~cur_valid).nonzero(as_tuple=True)[0]
        if empty_slots.numel() > 0:
            target_slot = empty_slots[0]
        else:
            d_all = torch.linalg.norm(cur_pts[:, :2] - robot_xy[env_id].unsqueeze(0), dim=-1)
            d_all = torch.where(cur_valid, d_all, torch.full_like(d_all, float("-inf")))
            target_slot = torch.argmax(d_all)
        cur_pts[target_slot] = one_pt
        cur_valid[target_slot] = True

    # Step-4: 按最近优先排序（invalid 用 +inf 作为排序键，自动沉到后面）。
    n = active_env_ids.shape[0]
    slot_count = valid.shape[1]
    dist_sorted_key = torch.linalg.norm(points_w[active_env_ids, :, :2] - robot_xy[active_env_ids].unsqueeze(1), dim=-1)
    dist_sorted_key = torch.where(valid[active_env_ids], dist_sorted_key, torch.full_like(dist_sorted_key, float("inf")))
    order = torch.argsort(dist_sorted_key, dim=-1)
    batch = torch.arange(n, device=env.device).unsqueeze(1).expand(-1, slot_count)
    points_w[active_env_ids] = points_w[active_env_ids][batch, order]
    valid[active_env_ids] = valid[active_env_ids][batch, order]
    points_w[active_env_ids] = points_w[active_env_ids].masked_fill(~valid[active_env_ids].unsqueeze(-1), 0.0)
