# RL 训练日志自动可视化与分析指南

本指南说明了在您向 AI 助手（Antigravity）提供实验日志路径时，系统是如何自动处理并为您画图和更新分析报告的，以及您自己如何直接运行相关脚本。

---

## 1. 如何通过 AI 助手快速画图？

您在聊天栏里只需提供**日志文件夹的路径**，例如：

> **您发送**：  
> `logs/GRU/0521_0127_strict` 帮我把这个 log 的图画出来。  
> *（或者直接发送路径即可）*

### AI 助手的自动处理流程：
1. **调用绘图 Skill**：AI 助手会直接在后台对该目录运行 `plot_all.py` 脚本。
2. **生成并导出图表**：脚本将自动在您的日志目录以及 IDE 历史会话中生成所有指标和对比图。
3. **更新分析报告**：AI 助手会自动刷新 [training_log_analysis.md](file:///home/hz/.gemini/antigravity-ide/brain/652857d2-34b6-4edb-92bf-dc8e054545e9/training_log_analysis.md)，您可以直接在编辑器右侧查看该报告以及交互式轮播图。

---

## 2. 如何自己在本地终端中运行绘图？

项目内置了统一的可视化脚本 `scripts/plot_all.py`，支持命令行参数。

### 运行环境准备：
请确保使用的是 `unitree_rl_lab` Conda 环境：
```bash
conda activate unitree_rl_lab
```

### 运行命令格式：
```bash
python scripts/plot_all.py --log_dir <您的实验日志目录相对路径或绝对路径>
```

#### 示例：
```bash
python scripts/plot_all.py --log_dir logs/GRU/0521_0129_speed
```

---

## 3. 生成的图表产物说明

运行后，您的日志目录下会生成以下图表：

| 图像文件名 | 描述 |
| :--- | :--- |
| `training_metrics_summary.png` | **TensorBoard 训练总览**：展示平均奖励、速度追踪奖励、速度误差以及课程（Curriculum）等级的变化趋势。 |
| `vel_tracking_accuracy.png` | **命令 vs 实际速度散点图**：展示不同迭代次数下，机器人实际速度与命令速度的对齐程度（越贴近 $y=x$ 对角线越好）。 |
| `vel_tracking_distribution_trends.png` | **误差与奖励分布箱线图**：展示 4096 个并行环境中误差和追踪奖励随训练步数收敛的趋势。 |
| `lin_vel_track_iter_{iter}.png` | **单次迭代环境追踪曲线**：按命令速度由低到高对所有并行环境排序，查看特定迭代下的详细速度对齐和追踪奖励。 |
| `curriculum_vel_track_iter_{iter}.png` | **课程升级点追踪曲线**：在课程系统**触发升级的瞬间**，记录当时触发升级的几个环境的详细对齐曲线。 |

---

## 4. 如何在 IDE 中预览图表？

如果您通过 AI 生成了图表，可以在右侧的 [training_log_analysis.md](file:///home/hz/.gemini/antigravity-ide/brain/652857d2-34b6-4edb-92bf-dc8e054545e9/training_log_analysis.md) 报告中直接阅读。
报告中已经内置了 Markdown Carousel（轮播图），您只需点击左右切换，即可观察到不同训练迭代或课程升级时速度追踪性能的动态进化过程。
