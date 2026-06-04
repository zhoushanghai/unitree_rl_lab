import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoxelFlowDataset
from model import VoxelFlowNet

def compute_loss(model, c_seq, v_prev, v_curr, dt_map, device):
    """
    计算基于配对数据的条件流匹配 (Rectified Flow) 损失，包含基于距离场的权重惩罚
    """
    B = v_prev.shape[0]
    
    # 1. 采样虚拟时间 s ~ U(0, 1)
    s = torch.rand(B, device=device)
    
    # 将 s reshape 为 (B, 1, 1, 1, 1) 用于插值广播
    s_view = s.view(B, 1, 1, 1, 1)
    
    # 2. 构建前向概率路径: x_s = (1-s)*v_prev + s*v_curr
    x_s = (1.0 - s_view) * v_prev + s_view * v_curr
    
    # 3. 计算真实目标流速: u_s = v_curr - v_prev
    target_v = v_curr - v_prev
    
    # 4. 模型预测流速: v_theta(x_s, s, c)
    pred_v = model(x_s, s, c_seq)
    
    # 5. 基础 MSE 误差
    mse = (pred_v - target_v) ** 2
    
    # 6. 计算基于 EDT 的惩罚权重矩阵 W(x)
    # 找到有真实障碍物的地方 (包括 t-1 和 t 时刻)，给予基础高权重 10.0
    mask_obs = ((v_curr + v_prev) > 0.1).float()
    mask_air = 1.0 - mask_obs
    
    # 距离惩罚系数 alpha
    alpha = 2.0 
    
    # 有障碍物区域权重为 10，空气区域权重随 dt_map (距离场) 线性增长
    weight = mask_obs * 10.0 + mask_air * (1.0 + alpha * dt_map)
    
    # 7. 加权均方误差
    loss = torch.mean(weight * mse)
    
    return loss

def train():
    # 配置
    data_dir = "/home/hz/project/dataset_voxel" # 适配 Docker 环境
    if not os.path.exists(data_dir):
        # Fallback 到宿主机路径
        data_dir = "/home/hz/proprioception/unitree_rl_lab/dataset_voxel"
        
    # 针对 100M 大模型调整超参数：
    # 1. 显卡为 A6000 48GB，具有顶级的显存容量，当前 64 的 batch 只占了 19G。我们直接翻倍拉满到 batch_size=128！
    # 2. 模型容量变大后，学习率稍微调低一点以保证平稳收敛 (从 3e-4 降到 1e-4)
    batch_size = 128
    num_epochs = 50
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 数据集和 DataLoader
    dataset = VoxelFlowDataset(data_dir=data_dir, seq_len=50, is_train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 模型和优化器
    model = VoxelFlowNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 自适应学习率调度器：基于 Loss 监控的自动降衰 (ReduceLROnPlateau)
    # 当训练效果 (Loss) 停滞不前时，自动将学习率降低一半 (factor=0.5)，帮助模型跳出局部最优
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    
    # 创建保存目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [lr:{current_lr:.1e}]")
        
        for batch in pbar:
            c_seq = batch['c_seq'].to(device)
            v_prev = batch['v_prev'].to(device)
            v_curr = batch['v_curr'].to(device)
            dt_map = batch['dt_map'].to(device)
            
            optimizer.zero_grad()
            
            loss = compute_loss(model, c_seq, v_prev, v_curr, dt_map, device)
            
            loss.backward()
            
            # 梯度裁剪防梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # 步进自适应学习率 (根据实际跑出来的 Loss 来决定要不要降学习率)
        scheduler.step(avg_loss)
        
        # 每 5 个 epoch 保存一次权重
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"checkpoints/flow_model_ep{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
