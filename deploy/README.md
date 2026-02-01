# Unitree RL Lab 部署模块 (Deploy)

本目录包含将 IsaacLab 训练的强化学习策略部署到 Unitree 真实机器人上的 C++ 控制代码。

## 目录结构

```
deploy/
├── include/           # 头文件库
│   ├── FSM/           # 有限状态机(FSM)控制系统
│   ├── isaaclab/      # IsaacLab 环境的 C++ 移植
│   ├── param.h        # 参数解析和日志系统
│   └── unitree_*.h    # Unitree 机器人抽象接口
├── robots/            # 各机器人平台的控制器实现
│   ├── go2/           # Unitree Go2 四足机器人
│   ├── go2w/          # Go2 轮式版本
│   ├── b2/            # Unitree B2 机器人
│   ├── g1_23dof/      # G1 人形机器人 (23自由度)
│   ├── g1_29dof/      # G1 人形机器人 (29自由度)
│   ├── h1/            # Unitree H1 人形机器人
│   └── h1_2/          # H1 第二代版本
└── thirdparty/        # 第三方依赖
    └── onnxruntime-linux-x64-1.22.0/  # ONNX Runtime 推理引擎
```

## 核心组件

### 1. 有限状态机 (FSM) 控制系统

位于 `include/FSM/`，实现机器人状态管理：

| 状态           | 功能描述                           |
| -------------- | ---------------------------------- |
| `Passive`      | 被动模式，电机阻尼控制             |
| `FixStand`     | 固定站立，移动到预设姿态           |
| `RLBase`       | 强化学习控制，执行 ONNX 策略推理   |
| `Mimic`        | 动作模仿（仅 G1 人形机器人）       |

状态转换通过遥控器手柄触发，例如：
- `LT + A` → 进入 FixStand
- `Start` → 启动 RL 控制

### 2. IsaacLab C++ 移植 (`include/isaaclab/`)

将 Python 训练环境的关键组件移植为 C++：

```
isaaclab/
├── algorithms/        # 策略推理 (OrtRunner - ONNX Runtime)
├── envs/              # RL 环境框架 (ManagerBasedRLEnv)
├── manager/           # 观测管理器 + 动作管理器
├── assets/            # 机器人关节抽象
└── devices/           # 输入设备 (键盘/遥控器)
```

**核心类**:
- `OrtRunner`: 加载 ONNX 模型，执行策略推理
- `ManagerBasedRLEnv`: 管理观测和动作流水线
- `ObservationManager`: 计算传感器观测值
- `ActionManager`: 处理并输出关节目标

### 3. 机器人配置 (`robots/<robot>/config/`)

每个机器人目录包含：

```
robots/go2/
├── CMakeLists.txt     # 构建配置
├── main.cpp           # 入口程序
├── config/
│   └── config.yaml    # FSM 配置 + 策略路径
├── include/           # 机器人特定头文件
└── src/               # 实现代码
```

**config.yaml 示例**:

```yaml
FSM:
  _:  # 启用的状态
    Passive: { id: 1 }
    FixStand: { id: 2 }
    Velocity: { id: 3, type: RLBase }
  
  Velocity:
    transitions:
      Passive: LT + B.on_pressed
    policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity  # 策略目录
```

## 构建与运行

### 依赖

- Unitree SDK2
- ONNX Runtime 1.22.0
- Boost (program_options)
- yaml-cpp
- Eigen3
- spdlog

### 编译

```bash
cd deploy/robots/go2
mkdir build && cd build
cmake ..
make
```

### 运行

```bash
./go2_ctrl                    # 默认配置
./go2_ctrl --network eth0     # 指定网络接口
./go2_ctrl --log              # 启用日志记录
```

### 操作流程

1. 启动程序，机器人进入 `Passive` 模式
2. 按 `L2 + A` 进入 `FixStand` 站立
3. 按 `Start` 切换到 RL 控制模式
4. 使用遥控器摇杆控制速度指令

## 策略部署

将训练好的 ONNX 模型放入策略目录：

```
logs/rsl_rl/unitree_go2_velocity/
└── <timestamp>/
    └── exported/
        ├── policy.onnx       # 策略网络
        └── params.yaml       # 环境参数
```

策略目录由 `config.yaml` 中的 `policy_dir` 指定。

## 支持的功能

| 功能     | Go2 | B2  | G1  | H1  |
| -------- | --- | --- | --- | --- |
| 速度控制 | ✅  | ✅  | ✅  | ✅  |
| 动作模仿 | ❌  | ❌  | ✅  | ❌  |

## 安全注意事项

> [!CAUTION]
> - 首次测试请使用悬吊装置
> - 确保周围有足够空间
> - 熟悉紧急停止操作 (`L2 + B` 回到 Passive)
> - 检查策略参数与实际机器人匹配

## 相关文件链接

- [算法推理](file:///home/hz/isaaclab/unitree_rl_lab/deploy/include/isaaclab/algorithms/algorithms.h) - ONNX 策略执行
- [RL 环境](file:///home/hz/isaaclab/unitree_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h) - 环境管理器
- [FSM 控制](file:///home/hz/isaaclab/unitree_rl_lab/deploy/include/FSM/CtrlFSM.h) - 状态机核心
- [Go2 配置](file:///home/hz/isaaclab/unitree_rl_lab/deploy/robots/go2/config/config.yaml) - 示例配置
