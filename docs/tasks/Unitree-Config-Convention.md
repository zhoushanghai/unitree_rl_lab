# Unitree 配置读取与调用规范（草案）

> 文档说明：本文定义项目内配置文件（YAML）如何组织、如何读取、如何在代码中调用，目标是减少重复读取逻辑、统一参数命名和覆盖行为。本文只讨论“读取与调用规范”，不展开具体 reward/网络参数设计。

**核心链路**：`config/config.yaml -> config/loader.py -> config/__init__.py -> 业务代码点访问参数`

## 一、输入/输出

| 类型 | 名称 | 说明 |
|---|---|---|
| 输入 | `config/config.yaml` | 项目级统一参数源（人可读、可版本化） |
| 输入 | `config/loader.py` | 项目根目录全局配置工具（递归读取 + 点访问映射） |
| 输出 | `config/__init__.py` | 全局配置入口，导出 `CONFIG` |
| 输出 | 业务代码读取方式 | `from <project_config_entry> import CONFIG` 后 `CONFIG.apf.rho` 读取 |

## 二、机制/规范

### 2.1 文件分层

1. **全局工具层**（项目级统一复用）  
   `config/loader.py`
2. **全局入口层**（统一导出）  
   `config/__init__.py`
3. **参数层**（业务可调）  
   `config/config.yaml`

### 2.2 读取接口（统一约定）

- 业务文件不得自行实现 YAML 读取（禁止重复 `Path + yaml.safe_load + try/except`）。
- 业务文件只通过项目统一配置入口 import（入口名字可按项目定义）：
  - `CONFIG`（递归映射后的对象）
- 取值统一使用点访问：
  - `value = CONFIG.apf.rho`
  - `value = CONFIG.obstacle_spawn.interval_s`
- `dict.get()` 仅允许出现在 `config/loader.py` 内部，不允许散落在业务代码中。

### 2.3 参数结构（硬约束）

统一采用**两层结构**（仅 `section -> key`）：

```yaml
apf:
  R: 1.0
  rho: 0.2

obstacle_spawn:
  delay_range_s: [1.0, 4.0]
  interval_s: 0.02
```

规则：

1. 顶层只能是业务模块 section（如 `apf`、`obstacle_spawn`）。
2. section 下直接放参数 key/value，不再嵌套第 3 层。
3. 若必须扩展复杂结构，先在设计评审中说明原因，再新增特例。

### 2.4 覆盖优先级

统一采用以下优先级（高 -> 低）：

1. YAML 中显式配置值
2. 代码当前字段值（类默认或运行时已有值）

说明：当前阶段不引入“多层 override 文件”（如 base/task/local）以保持简单；后续需要时再扩展。

## 三、参数与命名约束

| 约束项 | 规则 | 目的 |
|---|---|---|
| 参数层级 | 仅两层（`section -> key`） | 降低复杂度与维护成本 |
| section 命名 | 小写下划线（如 `apf`, `obstacle_spawn`） | 易搜索、易复用 |
| key 命名 | 小写下划线（如 `delta_vel_xy_max`） | 与 Python 字段一致 |
| 单位信息 | 物理量在注释中标单位（`m`, `s`, `m/s`） | 降低误调参风险 |
| 注释风格 | 行尾注释 + section 前一行块注释 | 可读且紧凑 |
| 默认回退 | 在全局工具/任务薄封装中处理 | 业务调用保持简洁 |

## 四、流程与约束

1. 新增参数时，先在 `config.yaml` 增加 key 与注释。
2. 在 `config/loader.py` 中确认该 section 已映射到 `CONFIG`。
3. 在业务配置文件（如 `velocity_env_cfg.py`）中用 `CONFIG.xxx.yyy` 读取。
4. 保持两层结构，不引入深层嵌套。
5. 保持“读取代码集中、业务代码薄调用”原则，不在多个文件复制 loader。

**硬约束（必须保持）**：

- 禁止在业务文件中直接读 YAML 文件路径。
- 禁止同一参数在多个位置用不同 key 名称。
- 禁止把“配置读取失败”变成硬崩溃（应回退默认值，便于开发调试）。

## 五、迁移建议（当前项目）

| 阶段 | 动作 | 范围 |
|---|---|---|
| Phase 1 | 已接入 APF / 障碍刷新参数 | `velocity_env_cfg.py` |
| Phase 2 | 迁移碰撞缓存参数（`force_threshold` 等） | `EventCfg` 相关 |
| Phase 3 | 迁移观测/训练辅助参数 | `ObservationsCfg` / runner cfg |
| Phase 4 | 视需要引入 base/task/local 覆盖层 | 全项目 |

## 参数速查表

| 模块 | 参数入口 | 使用方式 |
|---|---|---|
| 全局工具 | `config/loader.py` | 统一读取 YAML、递归映射为点访问对象 |
| 全局入口 | `config/__init__.py` | 导出 `CONFIG`（全局配置入口） |
| 业务调用 | `velocity_env_cfg.py` 等 | `from <project_config_entry> import CONFIG` 后 `CONFIG.apf.rho` |
