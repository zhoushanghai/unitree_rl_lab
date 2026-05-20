# G1 29dof 本地模型（URDF + meshes）

训练 `Unitree-G1-29dof-*` 任务时使用本目录，**无需**安装或 clone `unitree_ros`。

## 目录要求

- `g1_29dof_rev_1_0.urdf`
- `meshes/`（与 URDF 内 `meshdir="meshes"` 相对路径一致）

## 若目录缺失

从 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) 拷贝：

```bash
cp -a /path/to/unitree_ros/robots/g1_description/* \
  "$(dirname "$0")/"
```

或设置环境变量回退到外部 clone：

```bash
export UNITREE_ROS_DIR=/path/to/unitree_ros
```

## 说明

- `meshes/*.STL` 已纳入本仓库（根目录 `.gitignore` 对其它路径仍忽略 STL）。
- 仅需 G1 时不必拉取整个 `unitree_ros` 仓库。
