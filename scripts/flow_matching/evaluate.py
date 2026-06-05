import os
import sys
import argparse
import glob
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 确保能正确导入同一目录下的 model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import VoxelFlowNet

def parse_args():
    parser = argparse.ArgumentParser(description="Flow Matching Voxel Evaluation Script")
    parser.add_argument("--data_dir", type=str, default="dataset_voxel_test", help="Test dataset directory containing episode_*.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flow_model_step2100.pth", help="Path to the trained flow matching model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save metric plots and 3D visualizations")
    parser.add_argument("--pred_dir", type=str, default="dataset_voxel_test_pred", help="Directory to save the copy of dataset with prediction results")
    parser.add_argument("--num_steps", type=str, default="10", help="Number of Euler integration steps (can be parsed as int)")
    parser.add_argument("--threshold", type=str, default="0.5", help="Binarization threshold (can be parsed as float)")
    parser.add_argument("--save_pred_dataset", action="store_true", default=True, help="Whether to save the dataset copy with predictions")
    parser.add_argument("--plot_3d", action="store_true", default=True, help="Whether to generate and save 3D voxel plot visualizations")
    return parser.parse_args()

def euler_integration(model, v_prev, c_seq, num_steps, device):
    """
    使用 Euler 积分法从 s=0 到 s=1 积分 ODE
    v_prev: (1, 1, 20, 20, 15)
    c_seq: (1, 122, 50)
    """
    x = v_prev.clone().to(device)
    ds = 1.0 / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            s_val = i * ds
            s_tensor = torch.tensor([s_val], device=device, dtype=torch.float32)
            # v_theta 预测的是变化率/流速
            v = model(x, s_tensor, c_seq)
            x = x + v * ds
    return x

def compute_metrics(pred, gt):
    """
    计算二值网格的 IoU, Precision, Recall 和 F1-Score
    pred, gt: 形状均为 (20, 20, 15) 的 numpy 二值数组 (0 或 1)
    """
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(np.clip(pred_bin + gt_bin, 0.0, 1.0))
    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    
    # 边界情况：若真地值和预测值均完全没有障碍物（全空）
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 1.0, 1.0
        
    iou = float(intersection / max(union, 1.0))
    precision = float(intersection / max(pred_sum, 1.0))
    recall = float(intersection / max(gt_sum, 1.0))
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
        
    return iou, precision, recall, f1

def save_evaluated_episode(src_path, dest_dir, pred_single, pred_auto_bin, pred_auto_cont):
    """
    复制原始数据，并把三轨预测体素保存到新的 npz 文件中
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    
    # 读取原始的所有数据
    orig_data = dict(np.load(src_path))
    
    # 注入新的预测数据
    orig_data['collision_voxel_pred_single'] = pred_single.astype(np.float32)
    orig_data['collision_voxel_pred_auto_bin'] = pred_auto_bin.astype(np.float32)
    orig_data['collision_voxel_pred_auto_cont'] = pred_auto_cont.astype(np.float32)
    
    # 重新保存
    np.savez_compressed(dest_path, **orig_data)

def visualize_3d_voxels(gt_voxel, pred_single, pred_auto_bin, pred_auto_cont, ep_idx, step_idx, output_dir):
    """
    绘制并保存 3D 体素的四栏对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(24, 6))
    
    titles = [
        "Ground Truth", 
        "Single-Frame Prediction", 
        "Autoregressive Binarized", 
        "Autoregressive Continuous"
    ]
    
    voxels_list = [
        gt_voxel > 0.5, 
        pred_single > 0.5, 
        pred_auto_bin > 0.5, 
        pred_auto_cont > 0.5
    ]
    
    for idx, (voxels, title) in enumerate(zip(voxels_list, titles)):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        
        # 为了防止全空体素绘制报错，只在有障碍物时画 voxels
        if voxels.any():
            # 统一配色：绿色代表真地，蓝色/青色系列代表预测
            color = 'forestgreen' if idx == 0 else ('royalblue' if idx == 1 else ('darkturquoise' if idx == 2 else 'mediumpurple'))
            ax.voxels(voxels, edgecolor='k', facecolors=color, alpha=0.6)
            
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 15)
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"voxel_comparison_ep{ep_idx}_step{step_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved 3D voxel visualization to {save_path}")

def main():
    args = parse_args()
    num_steps = int(args.num_steps)
    threshold = float(args.threshold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data directory: {args.data_dir}")
    print(f"Num Euler integration steps: {num_steps}")
    print(f"Threshold: {threshold}")
    print("=" * 60)
    
    # 1. 实例化并加载模型
    model = VoxelFlowNet().to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    model.eval()
    
    # 2. 获取测试 Episode 文件列表
    files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.npz")))
    if not files:
        print(f"Error: No episode_*.npz files found in {args.data_dir}")
        sys.exit(1)
    
    print(f"Found {len(files)} test episodes.")
    
    # 记录全局指标
    all_metrics = {
        'single': {'iou': [], 'precision': [], 'recall': [], 'f1': []},
        'auto_bin': {'iou': [], 'precision': [], 'recall': [], 'f1': []},
        'auto_cont': {'iou': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    # 记录随时间步的指标（用于绘制衰减趋势）
    step_metrics_auto_bin = defaultdict(list)
    step_metrics_auto_cont = defaultdict(list)
    
    # 3. 逐个 Episode 进行评测
    for ep_idx, fpath in enumerate(files):
        print(f"Evaluating Episode {ep_idx + 1}/{len(files)}: {os.path.basename(fpath)}")
        
        # 加载单条 Episode 的所有时间步数据
        data = np.load(fpath)
        gt_voxels = data['collision_voxel'].astype(np.float32)  # (T, 20, 20, 15)
        T = gt_voxels.shape[0]
        
        base_ang_vel = data['base_ang_vel']
        projected_gravity = data['projected_gravity']
        joint_pos = data['joint_pos']
        joint_vel = data['joint_vel']
        last_action = data['last_action']
        joint_torques = data['joint_torques']
        
        # 拼接 122 维条件
        condition = np.concatenate([
            base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action, joint_torques
        ], axis=-1).astype(np.float32)  # (T, 122)
        
        # 初始化预测缓存
        pred_voxels_single = np.zeros_like(gt_voxels)
        pred_voxels_auto_bin = np.zeros_like(gt_voxels)
        pred_voxels_auto_cont = np.zeros_like(gt_voxels)
        
        # 第 0 步统一使用 Ground Truth 起始体素
        pred_voxels_single[0] = gt_voxels[0]
        pred_voxels_auto_bin[0] = gt_voxels[0]
        pred_voxels_auto_cont[0] = gt_voxels[0]
        
        # 用于自回归状态流转的中间变量
        # 增加 Batch 与 Channel 维度 -> (1, 1, 20, 20, 15)
        v_prev_auto_bin = torch.from_numpy(gt_voxels[0:1]).unsqueeze(0).to(device)
        v_prev_auto_cont = torch.from_numpy(gt_voxels[0:1]).unsqueeze(0).to(device)
        
        for t in range(T - 1):
            # A. 提取 50 帧历史 proprioception 并 padding
            start_idx = max(0, t - 50 + 1)
            c_seq_actual = condition[start_idx : t + 1]
            actual_len = c_seq_actual.shape[0]
            if actual_len < 50:
                pad_len = 50 - actual_len
                first_frame = c_seq_actual[0:1]
                pad_seq = np.repeat(first_frame, pad_len, axis=0)
                c_seq = np.concatenate([pad_seq, c_seq_actual], axis=0)
            else:
                c_seq = c_seq_actual
                
            # 转置为 (122, 50) 并添加 Batch 维度 -> (1, 122, 50)
            c_seq_t = torch.from_numpy(c_seq.T).unsqueeze(0).to(device)
            
            # --- 1. 单帧推理 (Single-Frame Track) ---
            v_prev_single = torch.from_numpy(gt_voxels[t:t+1]).unsqueeze(0).to(device)
            pred_single_out = euler_integration(model, v_prev_single, c_seq_t, num_steps, device)
            pred_single_np = pred_single_out.squeeze(0).squeeze(0).cpu().numpy()
            pred_voxels_single[t + 1] = pred_single_np
            
            # --- 2. 二值化自回归推理 (Autoregressive Binarized Track) ---
            pred_auto_bin_out = euler_integration(model, v_prev_auto_bin, c_seq_t, num_steps, device)
            pred_auto_bin_np = pred_auto_bin_out.squeeze(0).squeeze(0).cpu().numpy()
            pred_voxels_auto_bin[t + 1] = pred_auto_bin_np
            # 更新下一步状态：二值化后作为输入
            v_prev_auto_bin = (pred_auto_bin_out > threshold).float()
            
            # --- 3. 连续值自回归推理 (Autoregressive Continuous Track) ---
            pred_auto_cont_out = euler_integration(model, v_prev_auto_cont, c_seq_t, num_steps, device)
            pred_auto_cont_np = pred_auto_cont_out.squeeze(0).squeeze(0).cpu().numpy()
            pred_voxels_auto_cont[t + 1] = pred_auto_cont_np
            # 更新下一步状态：不经二值化，直接输入连续值
            v_prev_auto_cont = pred_auto_cont_out.clone()
            
            # --- 计算这一步的指标并记录 ---
            iou_s, prec_s, rec_s, f1_s = compute_metrics(pred_single_np, gt_voxels[t + 1])
            iou_b, prec_b, rec_b, f1_b = compute_metrics(pred_auto_bin_np, gt_voxels[t + 1])
            iou_c, prec_c, rec_c, f1_c = compute_metrics(pred_auto_cont_np, gt_voxels[t + 1])
            
            # 记录全局汇总
            all_metrics['single']['iou'].append(iou_s)
            all_metrics['single']['precision'].append(prec_s)
            all_metrics['single']['recall'].append(rec_s)
            all_metrics['single']['f1'].append(f1_s)
            
            all_metrics['auto_bin']['iou'].append(iou_b)
            all_metrics['auto_bin']['precision'].append(prec_b)
            all_metrics['auto_bin']['recall'].append(rec_b)
            all_metrics['auto_bin']['f1'].append(f1_b)
            
            all_metrics['auto_cont']['iou'].append(iou_c)
            all_metrics['auto_cont']['precision'].append(prec_c)
            all_metrics['auto_cont']['recall'].append(rec_c)
            all_metrics['auto_cont']['f1'].append(f1_c)
            
            # 记录时序指标，推演的时间步为 t + 1
            step_metrics_auto_bin[t + 1].append(f1_b)
            step_metrics_auto_cont[t + 1].append(f1_c)
            
        # 4. 可选：保存到评估后的新数据集中
        if args.save_pred_dataset:
            save_evaluated_episode(
                fpath, args.pred_dir, 
                pred_voxels_single, pred_voxels_auto_bin, pred_voxels_auto_cont
            )
            
        # 5. 可选：针对前 2 个 Episode，保存特定帧（如步数 10, 30, 50）的三维对比可视化图像
        if args.plot_3d and ep_idx < 2:
            visualize_steps = [min(10, T - 1), min(30, T - 1), min(50, T - 1)]
            for step in set(visualize_steps):
                if step < T:
                    visualize_3d_voxels(
                        gt_voxels[step], 
                        pred_voxels_single[step], 
                        pred_voxels_auto_bin[step], 
                        pred_voxels_auto_cont[step], 
                        ep_idx + 1, step, args.output_dir
                    )
                    
    # 6. 计算和输出汇总平均指标
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    for track in ['single', 'auto_bin', 'auto_cont']:
        print(f"Track: {track.upper()}")
        for metric in ['iou', 'precision', 'recall', 'f1']:
            vals = all_metrics[track][metric]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"  {metric.ljust(10)}: {mean_val:.4f} ± {std_val:.4f}")
        print("-" * 60)
        
    # 7. 绘制指标随时间步的衰减曲线图
    os.makedirs(args.output_dir, exist_ok=True)
    max_step = max(max(step_metrics_auto_bin.keys(), default=0), max(step_metrics_auto_cont.keys(), default=0))
    steps_list = sorted(list(step_metrics_auto_bin.keys()))
    
    avg_f1_bin = [np.mean(step_metrics_auto_bin[s]) for s in steps_list]
    avg_f1_cont = [np.mean(step_metrics_auto_cont[s]) for s in steps_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, avg_f1_bin, label='Autoregressive Binarized (Threshold 0.5)', color='darkturquoise', linewidth=2)
    plt.plot(steps_list, avg_f1_cont, label='Autoregressive Continuous (Raw Float)', color='mediumpurple', linewidth=2)
    plt.xlabel('Prediction Step (Time)', fontsize=12)
    plt.ylabel('Average F1-Score', fontsize=12)
    plt.title('Voxel Prediction Quality Decay over Time Steps', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    decay_chart_path = os.path.join(args.output_dir, "metrics_trend.png")
    plt.savefig(decay_chart_path, dpi=150)
    plt.close()
    print(f"\nSaved metrics decay trend chart to: {decay_chart_path}")
    print(f"Saved copy of evaluated dataset with predictions to: {args.pred_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
