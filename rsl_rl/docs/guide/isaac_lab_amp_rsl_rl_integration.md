# Isaac Lab 自写环境与 AMP + rsl_rl：数据管线与训练侧分工

本文说明如何在 **Isaac Lab + 自写 `env`** 的前提下，**reference motion 的磁盘格式与 proprioception 工程一致**：`/home/hz/proprioception/proprioception` 中的 YAML 清单 + `.npz` 约定（由任务下的 `MotionLoader` 读取，例如 `tasks/proprioception_vel/mdp/motions/motion_loader.py`），同时在 **runner / `rsl_rl` 改动方式** 上对齐 [`amp-rsl-rl`](https://github.com/ami-iit/amp-rsl-rl)：**判别器、`expert` 与 `policy` 的 AMP 样本缓冲、以及与 PPO 同一优化步内的 loss 拼装**。

要点：**reference motion 格式 = proprioception（YAML + npz）；算法接线 = `amp-rsl-rl`。** 二者通过「AMP 特征向量」的**维度与语义契约**对齐，与 `amp-rsl-rl` 的 pickle `.npy` 数据集无直接兼容要求。

---

## 1. 总体架构

| 层次 | 职责 | 与 `amp-rsl-rl` 的关系 |
|------|------|------------------------|
| **数据** | **proprioception 同款**：YAML `motion_files` + 每条轨迹一个 `.npz`（字段见 §2） | 替代其 `AMPLoader` + pickle dict `.npy`；**接口目标一致**：给出可用于训练的 `(state_amp, next_state_amp)` |
| **Isaac Lab `env`** | 仿真步进、任务观测、奖励；从仿真状态构造 **policy 侧的 AMP 特征** `amp_obs`，并在步进边界写入 runner 约定的 buffer | 语义对齐 `MotionData.get_amp_dataset_obs` 一类逻辑：**同一套特征定义**用于 expert 与 policy |
| **Runner / 算法** | PPO rollout；每步收集 `amp_obs` 与 `amp_obs'` 写入 **policy AMP replay**；每 PPO mini-batch **同步**从 expert 发生器与 policy replay **取同尺寸 batch**，拼判别器输入并反传 `ppo_loss + amp_loss + grad_pen_loss` | 流程对齐 `amp_rsl_rl.algorithms.amp_ppo.AMP_PPO` |

---

## 2. Reference motion 格式（对齐 `/home/hz/proprioception/proprioception`）

以下与 proprioception 中 `MotionLoader` 的实现一致（清单解析、`_load_npz`、多轨迹校验逻辑可直接参照该文件）。

### 2.1 YAML 清单

- 顶层键 **`motion_files`**：条目列表。
- 每条可为：
  - 字符串路径；或
  - 映射：`file`（或 `path`）+ `weight`（默认 `1.0`）。
- 若 `motion_files` 为字符串列表，亦可配合顶层 **`weights`** 数组按索引给权（解析逻辑见 `_parse_motion_manifest`）。
- 相对路径相对于 **含 `datasets` 目录的工程根**：loader 会从 YAML 所在目录向上查找带有 `datasets` 的目录作为 root，再 `_resolve_path`。**建议**像你本地示例 `datasets/proprioception_reference_motions.yaml` 一样，把条目里的 `file` 写成相对该 root 的路径。

权重在 **`MotionLoader` 内归一化**为多轨迹采样分布。

### 2.2 每条轨迹 `.npz` 必备键与约束

`np.load` 后须包含（与 `_validate_motion` 一致）：

| 键 | 含义 |
|----|------|
| `dof_names` | DOF 名字列表（首条轨迹确定顺序；后续轨迹必须与首条 **完全一致**） |
| `key_body_names` | 关键刚体名字列表（同上，多轨迹必须一致） |
| `fps` | 标量帧率；内部用 `dt = 1/fps`，**所有轨迹的 `fps` 须与首条一致**（与仿真步长对齐） |
| `dof_positions` | 形状 `(T, num_dof)` |
| `dof_velocities` | 形状 `(T, num_dof)` |
| `key_body_positions_local` | 与关键刚体局部位置对应的逐帧数组（与 proprioception 中用法一致） |

`MotionLoader.sample` 按帧索引返回 `(dof_positions, dof_velocities, key_body_positions_local)`，供 env 侧拼 AMP 观测或中间特征。

### 2.3 Expert 侧如何接到 `amp-rsl-rl` 式训练

**Expert 发生器**（对应 `amp-rsl-rl` 的 `AMPLoader.feed_forward_generator`）在 **不改变上述磁盘格式** 的前提下应：

1. 用与 proprioception 相同的加载/加权采样方式从多轨迹取 `(t, t+1)` 帧对；
2. 用 **与 Isaac Lab `env` 完全相同** 的函数把「单帧仿真态 / 单帧 npz 态」映射为 **单行 AMP 特征** `state`、`next_state`（见 §3）。

只要 **`state`/`next_state` 与仿真侧 `amp_obs` 同维同序**，磁盘上即维持 proprioception 格式，无需向 `amp-rsl-rl` 的 `.npy` 看齐。

---

## 3. Isaac Lab `env` 侧：AMP 特征契约

### 3.1 谁来算特征

- **仿真中（policy）**：每个物理步（或每个 policy 决策步）用 **当前仿真状态** 计算 `amp_obs`，形状 `[num_envs, amp_dim]`。
- **数据中（expert）**：从 **proprioception 格式 `.npz`** 经 `MotionLoader` 取出 `dof_positions` / `dof_velocities` / `key_body_positions_local`，再经与仿真侧 **同一套** 索引（如 `motion_dof_indexes`、`motion_key_body_indexes`）与 **同一函数**（如 proprioception `amp_env` 中的 `_compute_amp_observations`）打成 `state`/`next_state`。

### 3.2 特征定义：以 proprioception 为准，与 `amp-rsl-rl` 仅概念对照

`amp-rsl-rl` 的 `MotionData.get_amp_dataset_obs` 是 **另一套** 特征拼接（关节 + 基座局部线/角速度）。你采用 proprioception 数据时，**AMP 向量定义应以 Isaac Lab env 里实际使用的 `_compute_amp_observations`（及关键刚体局部坐标的计算方式）为准**。

- **expert 与 policy 必须共用同一构造函数、同一 DOF/刚体子集、同一归一化（若有）**。
- 四元数、参考系与 `wxyz`/`xyzw` 约定以仿真与 proprioception 导出脚本为准，并在实现里固定下来。

### 3.3 Runner 钩子（与 `AMP_PPO` 一致的节奏）

在每步中与 `amp-rsl-rl` 对齐的调用节奏为：

1. **`act_amp(amp_obs_t)`**：在施加动作前或取得 `obs_t` 后，缓存当前 AMP 特征（上一时刻到当前时刻配对中的「当前时刻」）。
2. 环境执行 `step`。
3. **`process_amp_step(amp_obs_{t+1})`**：将 `(amp_obs_t, amp_obs_{t+1})` 作为一条 transition **写入 policy AMP replay buffer**。

奖励中的 style 项若在 `env` 内使用判别器 logits，需注意 **是否与训练端判别器共用网络与归一化**；常见做法是 **`env` 仅消费 runner 下发的标量或由单独模块计算**，避免与 `update()` 内判别器两步不一致。若必须由 `env` 内访问判别器，应保持与 runner 相同的 **eval 模式与归一化状态**。

---

## 4. Runner / `rsl_rl`：PPO + AMP 的 batch 拼装

以下与 `AMP_PPO.update()` 的流程一致（见 `amp_rsl_rl/algorithms/amp_ppo.py`）。

### 4.1 三个数据流

1. **`RolloutStorage`**：常规 PPO 所需的 `TensorDict` 观测、动作、回报、advantage 等。
2. **`ReplayBuffer`（policy AMP）**：存储仿真产生的 `(state, next_state)`，`state` 与 `next_state` 各占 **判别器输入维度的一半**（因为判别器输入为 `concat(state, next_state)`）。
3. **Expert AMP 发生器**：从 YAML/`.npz` 管线采样， yielding 与上面相同形状的 `(expert_state, expert_next_state)`。

### 4.2 每个 PPO mini-batch 内如何拼接

对每个对齐的 mini-batch（`zip(rollout_generator, amp_policy_generator, amp_expert_generator)`）：

1. 计算 **PPO 部分**：surrogate、value（及 entropy 等）。
2. 取：
   - `policy_state`, `policy_next_state`
   - `expert_state`, `expert_next_state`
3. 构造判别器输入（与 `AMP_PPO` 相同）：
   - `fake = concat(policy_state, policy_next_state, dim=-1)`
   - `real = concat(expert_state, expert_next_state, dim=-1)`
   - `disc_in = concat(fake, real, dim=0)`，`B_pol = fake.size(0)`
4. 前向：`d_out = discriminator(disc_in)`，再拆成 policy 段与 expert 段。
5. **Loss**：`loss = ppo_loss + amp_loss + grad_pen_loss`（梯度惩罚等形式以你所用 `Discriminator.compute_loss` 为准）。
6. **一次 `optimizer.step()`** 同时更新 actor-critic 与 discriminator 参数（或按项目拆optimizer，但总原则与参考实现一致：**同一更新相位内 AMP 与 PPO 协同**）。
7. **判别器输入归一化**：若在 `compute_loss` 之后调用 `discriminator.update_normalization(...)`，应对 **未经错误归一化的 raw** expert/policy 样本做统计（参考实现使用 `detach().clone()` 的 raw 张量）。

### 4.3 batch 尺寸对齐

参考实现令：

- `amp_policy_generator` 与 `amp_expert_generator` 的 **mini-batch 数量与大小** 与 PPO mini-batch 设置一致（`num_learning_epochs * num_mini_batches` 与每条 batch 的样本数formula 与 rollout 总规模、`num_mini_batches` 相关联）。

你若修改 horizon 或 `num_envs`，需同步调整发生器，避免出现 **长度不齐的 `zip` 提前结束**。

---

## 5. 维度与接口检查清单

在第一次能跑通训练前建议逐项核对：

- [ ] `amp_obs` 在 `env` 内行向量维度 = `amp_dim`，且 **expert 发生器输出同 `amp_dim`**。
- [ ] 判别器 `input_dim == 2 * amp_dim`（若采用 `concat(state, next_state)` 单样本约定）。
- [ ] Expert 的 `next_state` 与 policy 的 `next_state` **同一时刻语义**（均为 `t`→`t+1` 的仿真步）。
- [ ] YAML 权重归一化、多轨迹 `fps` 一致；与仿真 `dt` 一致或可解释的重采样策略。
- [ ] policy AMP replay **容量**、`insert` 频率与 **`clear()`** 时机与 `RolloutStorage` 一致（参考实现在 `update` 末尾 `storage.clear()`；AMP replay **通常不因回合结束而整条清空**，除非你有意为之）。
- [ ] `Discriminator.update_normalization` 使用的样本与训练前向所用特征 **统计口径一致**。

---

## 6. 小结

- **Reference motion**：**proprioception 格式**——YAML `motion_files` + 满足 §2 的 `.npz`；用它喂 **expert AMP 迷你 batch**。
- **`env`**：Isaac Lab 仿真下构造与上述 **同一 AMP 语义** 的 `amp_obs`，并按 runner 约定 **步步写入 policy AMP replay**。
- **Runner**：按 **`AMP_PPO`** 的方式把 **PPO minibatch × expert AMP × policy AMP** 绑在同一更新步，并保持 **判别器输入 = 双倍 AMP 特征维** 的配对约定。

与 `amp-rsl-rl` 的差异在于 **reference 磁盘格式与 AMP 观测的具体拼接**（proprioception 管线）；**训练流程与 batch 拼装**仍可与其对齐。

---

## 7. 本仓库中的对应实现（`rsl_rl`）

以下内容已并入本库（依赖 **PyYAML**，见 `pyproject.toml`），便于 Isaac Lab `env` 与 proprioception 数据集直接对接：

| 组件 | 模块路径 |
|------|-----------|
| PPO + AMP 更新 (`update` 内三方 batch 对齐) | `rsl_rl.algorithms.amp_ppo.AMPPPO` |
| 训练循环：`act_amp` / `process_amp_step` / 可选 style reward | `rsl_rl.runners.amp_on_policy_runner.AmpOnPolicyRunner` |
| YAML + `.npz` Expert 采样 `(state, next_state)` | `rsl_rl.utils.proprio_motion_loader.ProprioceptionMotionExpert` |
| AMP 判别器 | `rsl_rl.modules.amp_discriminator.AMPDiscriminator` |
| 策略侧 AMP 环形缓冲 | `rsl_rl.storage.amp_replay_buffer.AMPReplayBuffer` |

### `train_cfg` 骨架（节选）

- `algorithm.class_name`: `rsl_rl.algorithms.amp_ppo.AMPPPO`
- **`algorithm`** 必含：`amp_motion_yaml`、`amp_obs_dim`；可选 `amp_replay_buffer_size`、`motion_dof_indexes`、`motion_key_body_indexes`、`amp_motion_project_root`
- **`discriminator`**：`hidden_dims`（如 `[1024, 512]`）、`reward_scale`、`loss_type`、`empirical_normalization` 等（与 `amp-rsl-rl` 习惯一致，`hidden_dims` 即隐层宽度列表）
- **`amp_runner`**（传给 `AmpOnPolicyRunner`）：`observation_key`（默认 `amp`， TensorDict 中的 AMP 向量）、`extras_key`（默认 `amp_obs`，与 Isaac `extras` 对齐）、`use_style_reward`、`style_reward_coef`

环境侧须在 **每一步** 提供与 Expert 一致的 AMP 向量：优先写入 TensorDict 的 `observation_key`，否则写入 `extras[extras_key]`；维度须等于 `amp_obs_dim`。

**限制（当前 revision）**：`AMPPPO` 与 **RND、对称增强** 不组合。**RNN 策略**已按 `amp-rsl-rl` 方式支持：PPO 使用 `RolloutStorage.recurrent_mini_batch_generator`，AMP policy/expert 仍用与 `num_envs * num_steps_per_env // num_mini_batches` 对齐的扁平 batch 与之一同 `zip` 更新。
