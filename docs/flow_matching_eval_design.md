# Flow Matching Voxel 模型评估方案设计

本项目旨在设计并实现一个评估脚本 `scripts/flow_matching/evaluate.py`，用于测试和量化训练好的 Flow Matching 3D Voxel 预测模型（`checkpoints/flow_model_step2100.pth`）在测试集 `dataset_voxel_test` 上的性能。

评估脚本将包含以下三种版本的推理测试，并在评估后将这三种预测结果保存至新建的数据集中。

## 1. 评估模式设计

### 1.1 单帧推理测试 (Single-Frame Inference)
* **原理**：以第 $t$ 帧的**真实体素网格 (Ground Truth)** 作为输入 `v_prev`，并结合前 50 帧的历史本体感知序列 `c_seq`，通过模型进行单步积分，预测第 $t+1$ 帧的体素网格。
* **条件变量设计**：本脚本将严格按照实际训练代码的 122 维条件拼接方式：
  * 角速度 `base_ang_vel` (3) + 重力投影 `projected_gravity` (3) + 关节位置 `joint_pos` (29) + 关节速度 `joint_vel` (29) + 上一步动作 `last_action` (29) + **关节力矩 `joint_torques` (29)** = **122维**。
* **保存字段**：`collision_voxel_pred_single`

### 1.2 二值化自回归推理测试 (Autoregressive Binarized Inference)
* **原理**：针对每个 Episode，仅在第 $0$ 帧使用真实的体素网格 $V_0^{\text{gt}}$。在后续的每个时间步 $t$：
  * 将上一步预测出的三维矩阵 $x_1$ 经过 `> 0.5` **二值化处理**（还原成 0/1 网格），作为下一步的 `v_prev` 输入模型。
  * 结合真实的本体感知序列 `c_seq`，推演下一时刻。
* **目的**：测试满足训练分布（输入全为二值网格）下的自回归长时稳定性。
* **保存字段**：`collision_voxel_pred_auto_bin`

### 1.3 连续值自回归推理测试 (Autoregressive Continuous Inference)
* **原理**：同样仅在第 $0$ 帧使用 $V_0^{\text{gt}}$。但在后续的每个时间步 $t$：
  * 直接将上一步模型预测出的**原始连续概率/流形值** $x_1$（不经阈值截断，保留浮点数值）作为输入 `v_prev`。
  * 结合真实的本体感知序列 `c_seq`，推演下一时刻。
* **目的**：对比分析“连续概率值直接传递”是否比“二值化截断”更能保留边界渐变信息，抑或更容易导致模糊度快速扩散与崩溃。
* **保存字段**：`collision_voxel_pred_auto_cont`

---

## 2. 核心数学与算法细节

### 2.1 ODE 积分 (Euler 方法)
Flow Matching 模型预测的是体素的流场变化率（速度向量场） $v_\theta(x_s, s, c)$。为了从时刻 $s=0$ (上一帧 `v_prev`) 得到时刻 $s=1$ (当前帧 `v_curr`) 的预测，我们需要对如下 ODE 进行数值积分：
$$ \frac{d x_s}{d s} = v_\theta(x_s, s, c) $$

在推理时，我们采用 **Euler 积分法**，将其离散化为 $N$ 步（默认 $N=10$）：
1. 令 $x_0 = v_{\text{prev}}$ 并且 $ds = \frac{1}{N}$。
2. 循环对于 $i = 0, 1, \dots, N-1$：
   * $s_i = \frac{i}{N}$
   * 计算模型速度：$v_i = \text{model}(x_{s_i}, s_i, \text{c\_seq})$
   * 步进：$x_{s_{i+1}} = x_{s_i} + v_i \cdot ds$
3. 得到积分终点 $x_1$。
4. **二值化过滤**：计算指标时，所有三个版本最终均通过 `> 0.5` 阈值处理，得到供对比的二值网格。

---

## 3. 数据集备份与三轨数据保存

评估脚本在对测试集进行推理时，将采取如下数据保存机制：
* **输出路径**：默认在宿主机/容器内的 `dataset_voxel_test_pred/` 文件夹下保存。
* **数据复制与新增**：遍历并复制 `dataset_voxel_test/` 中的所有原始变量（包括 `collision_voxel` 和 122 维 proprioception 系列数据），并在新的 `.npz` 中额外打包以下三个预测序列：
  1. `collision_voxel_pred_single`
  2. `collision_voxel_pred_auto_bin`
  3. `collision_voxel_pred_auto_cont`
  （注：这三个矩阵的形状均与原始的 `collision_voxel` 相同，即 `(T, 20, 20, 15)`）
* **效果**：生成的新数据集完全兼容原生的回放脚本（如 `replay_voxel_dataset.py`），支持后期非常直观的侧向对比及二次加载分析。

---

## 4. 评测指标 (Metrics)

对于每个预测步，我们将二值化后的预测值与真实网格 $V^{\text{gt}}$ 进行像素级对比，计算以下指标：

* **IoU (交并比)**:
  $$ \text{IoU} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{pred}} \cup V^{\text{gt}}|, 1)} $$
* **Precision (精确率)**:
  $$ \text{Precision} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{pred}}|, 1)} $$
* **Recall (召回率)**:
  $$ \text{Recall} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{gt}}|, 1)} $$
* **F1-Score**:
  $$ \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\max(\text{Precision} + \text{Recall}, 1e-8)} $$

**特殊边界处理**：如果某帧中 $V^{\text{gt}}$ 和 $V^{\text{pred}}$ 均完全无障碍物（全空网格），则判定该步预测完全正确，IoU 和 F1 直接记为 $1.0$。

---

## 5. 输出产物与可视化

评估脚本运行结束后，将在指定的输出目录（如 `evaluation_results/`）生成以下内容：
1. **控制台/日志统计**：输出全部三种推理模式在整个测试集上的平均 IoU、Precision、Recall 和 F1-Score（包含标准差）。
2. **自回归衰减曲线 (`metrics_trend.png`)**：绘制两种自回归推理模式（二值化 vs 连续值）下，各项指标随时间步（Time Step, $0 \to T$）的变化曲线，直观对比哪种自回归方式更稳定。
3. **3D 体素对比可视化 (`voxel_comparison_step_*.png`)**：
   * 选取特定 Episode 的典型时间步，使用 matplotlib 的 3D 网格绘图功能绘制多栏对比图，以定性分析障碍物的还原情况。
��率值的模糊误差随着时间步呈指数级扩散。

---

## 3. 评测指标 (Metrics)

对于每个预测步，我们将二值化后的预测值 $V^{\text{pred}} \in \{0, 1\}^{20 \times 20 \times 15}$ 与真实网格 $V^{\text{gt}} \in \{0, 1\}^{20 \times 20 \times 15}$ 进行像素级对比，计算以下指标：

* **IoU (交并比)**:
  $$ \text{IoU} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{pred}} \cup V^{\text{gt}}|, 1)} $$
* **Precision (精确率)**:
  $$ \text{Precision} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{pred}}|, 1)} $$
* **Recall (召回率)**:
  $$ \text{Recall} = \frac{|V^{\text{pred}} \cap V^{\text{gt}}|}{\max(|V^{\text{gt}}|, 1)} $$
* **F1-Score**:
  $$ \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\max(\text{Precision} + \text{Recall}, 1e-8)} $$

**特殊边界处理**：如果某帧中 $V^{\text{gt}}$ 和 $V^{\text{pred}}$ 均完全无障碍物（全空网格），则判定该步预测完全正确，IoU 和 F1 直接记为 $1.0$。

---

## 4. 输出产物与可视化

评估脚本运行结束后，将在指定的输出目录（如 `evaluation_results/`）生成以下内容：
1. **控制台/日志统计**：输出单帧模式和自回归模式在整个测试集上的平均 IoU、Precision、Recall 和 F1-Score（包含标准差）。
2. **自回归衰减曲线 (`metrics_trend.png`)**：绘制自回归推理模式下，各项指标随时间步（Time Step, $0 \to T$）的变化曲线，直观展示误差漂移速度。
3. **3D 体素对比可视化 (`voxel_comparison_step_*.png`)**：
   * 选取特定 Episode 的典型时间步，使用 matplotlib 的 3D 网格绘图功能绘制三栏对比图：
     * **左栏**：Ground Truth
     * **中栏**：Single-Frame Prediction
     * **右栏**：Autoregressive Prediction
   * 用于直观定性评估障碍物的形状恢复程度和动态跟踪情况。
