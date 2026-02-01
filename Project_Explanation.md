# unitree_rl_lab 项目说明

## 1. 项目简介
**unitree_rl_lab** 是一个基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 构建的强化学习（Reinforcement Learning, RL）环境集合，专门用于 Unitree（宇树科技）的机器人。该项目旨在提供从仿真训练到真机部署的全套解决方案。

### 核心功能
*   **基于 Isaac Sim**: 利用 NVIDIA Isaac Sim 强大的物理仿真能力。
*   **多环境支持**: 提供了 Isaac Lab（开发/训练）、Mujoco（Sim2Sim 验证）和物理真机（Sim2Real 部署）的统一工作流。
*   **支持机器人**: 目前支持 Unitree **Go2** (四足), **H1** (双足/人形), 和 **G1-29dof** (人形) 机器人。

## 2. 目录结构说明

项目的核心文件结构如下：

*   **`source/`**: 包含项目的核心 Python 源码包 `unitree_rl_lab`，定义了机器人的资产（assets）、环境（envs）和任务配置。
*   **`scripts/`**: 包含用于训练和推理的脚本。
    *   `scripts/rsl_rl/train.py`: 启动 RL 训练。
    *   `scripts/rsl_rl/play.py`: 加载训练好的模型进行推理演示。
*   **`deploy/`**: 包含用于部署到真机或 Mujoco 仿真器的 C++ 代码和工具。
    *   支持编译生成机器人控制器。
*   **`logs/`**: 默认存放训练日志和模型检查点的目录。
*   **`unitree_rl_lab.sh`**: 项目提供的便捷管理脚本，用于安装依赖、列出任务和运行环境。

## 3. 安装与配置
项目依赖于 Isaac Lab。主要安装步骤包括：
1.  安装 Isaac Lab。
2.  克隆本项目并安装依赖：
    ```bash
    ./unitree_rl_lab.sh -i
    ```
3.  配置机器人资源（URDF 或 USD 文件），通常需要从 HuggingFace 或 GitHub 下载 Unitree 的模型文件，并在 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置路径。

## 4. 使用流程

### 4.1 训练 (Training)
使用 `scripts/rsl_rl/train.py` 或快捷脚本启动训练任务：
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

### 4.2 推理/回放 (Inference/Play)
查看训练效果：
```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

### 4.3 部署 (Deployment)
部署流程通常包含两个阶段：
1.  **Sim2Sim (Mujoco)**: 在 Mujoco 仿真器中验证 Isaac Sim 训练的策略，确保控制器的鲁棒性。
2.  **Sim2Real (Real Robot)**: 将验证后的策略部署到物理机器人上。需要编译 `deploy/` 下的 C++ 控制程序 (`g1_ctrl` 等)。

## 5. Sim2Sim 按键操作指南

### 5.1 g1_ctrl 终端（键盘控制）

需要在运行 `./g1_ctrl` 的终端窗口中按键：

| 按键 | 功能 |
|------|------|
| `1` | 进入 FixStand（站立模式） |
| `2` | 进入 Velocity（RL 控制模式） |
| `0` | 返回 Passive（安全模式） |
| `W` | 前进 |
| `S` | 后退 |
| `A` | 左平移 |
| `D` | 右平移 |
| `Q` | 左旋转 |
| `E` | 右旋转 |

### 5.2 MuJoCo 仿真窗口

需要点击 MuJoCo 仿真窗口使其获得焦点后按键：

| 按键 | 功能 |
|------|------|
| `7` | 放下机器人（降低高度） |
| `8` | 吊起机器人（升高高度） |
| `9` | 开/关虚拟挂带 |
| `Space` | 暂停/继续仿真 |
| 鼠标左键拖拽 | 旋转视角 |
| 鼠标右键拖拽 | 平移视角 |
| 滚轮 | 缩放视角 |

### 5.3 操作流程

```
启动 → Passive → [按1] → FixStand → [MuJoCo按8放下] 
                                          ↓
                              [按2] → RL控制 → [MuJoCo按9松挂带]
                                          ↓
                                    [WASD/QE控制移动]
                                          ↓
                              [按0] → 返回 Passive（安全）
```

## 6. 策略文件配置

### 6.1 策略目录结构

`g1_ctrl` 使用的是 **ONNX 格式**的策略模型（不是 PT 文件）：

```
deploy/robots/g1_29dof/config/policy/velocity/
└── v0/                      # 版本目录（自动选择最新的）
    ├── exported/
    │   └── policy.onnx      # ← 实际使用的模型文件
    └── params/
        └── deploy.yaml      # 观测/动作配置
```

### 6.2 指定策略路径

在 `deploy/robots/g1_29dof/config/config.yaml` 中配置：

```yaml
Velocity:
  # 方式1: 使用相对路径（相对于 config 目录）
  policy_dir: config/policy/velocity
  
  # 方式2: 使用训练输出的日志目录
  # policy_dir: ../../../logs/rsl_rl/unitree_g1_29dof_velocity
```

### 6.3 导出 ONNX 模型

训练完成后，运行 `play.py` 会自动导出 ONNX：

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

导出位置：`logs/rsl_rl/<experiment>/exported/policy.onnx`

然后将 `exported/` 目录和 `params/deploy.yaml` 复制到策略目录即可。

## 7. 参考资源
*   **IsaacLab**: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
*   **Unitree Robotics**: [https://www.unitree.com/](https://www.unitree.com/)

