import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

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
    
    # 6. 计算掩码矩阵 (完全按照你的最新逻辑)
    # (1) 首先关注本来有碰撞点的位置 (全量实体)
    mask_exist_raw = ((v_curr + v_prev) > 0.5).float()
    
    # (2) 往外延伸 1 个格子，作为冗余来判断 (包含实体 + 冗余层)
    mask_obs_dilated = F.max_pool3d(mask_exist_raw, kernel_size=3, stride=1, padding=1)
    
    # (3) 没有障碍物的地方，就是除去上面所有的剩余的地方 (纯空气)
    mask_air = 1.0 - mask_obs_dilated
    
    # (4) 单独关注变化的格子 (纯动态，针对最精确的变化点)
    mask_dynamic = (torch.abs(v_curr - v_prev) > 0.5).float()
    
    # 距离惩罚系数 alpha
    alpha = 2.0 
    
    # 7. 分区域计算平均误差 (依然采用解耦平均法，防止空气主导)
    # A. 碰撞点及冗余区域 Loss (保证结构完整，赋予基础权重 10.0)
    loss_obs = torch.sum(mask_obs_dilated * mse) / (torch.sum(mask_obs_dilated) + 1e-8)
    
    # B. 剩余空气区域 Loss (保持背景干净，带距离场惩罚，基础权重 1.0)
    weight_air = mask_air * (1.0 + alpha * dt_map)
    loss_air = torch.sum(weight_air * mse) / (torch.sum(mask_air) + 1e-8)
    
    # C. 单独关注的变化格子 Loss (给运动轨迹单独额外计算 Loss，赋予高权重 20.0)
    loss_dynamic = torch.sum(mask_dynamic * mse) / (torch.sum(mask_dynamic) + 1e-8)
    
    # 8. 组合最终 Loss (三部分相加)
    loss = 10.0 * loss_obs + 1.0 * loss_air + 20.0 * loss_dynamic
    
    return loss, loss_obs, loss_air, loss_dynamic

def train():
    # 初始化 wandb
    wandb.init(
        project="unitree-g1-voxel-flow",
        name="100M_Flow_Matching",
        config={
            "batch_size": 128,
            "learning_rate": 1e-4,
            "epochs": 50,
            "model_size": "100M",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau"
        }
    )

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
    
    # 让 wandb 深度监听模型！
    # 这会自动记录每一层神经网络权重的直方图 (Histograms) 和反向传播的梯度 (Gradients)
    wandb.watch(model, log="all", log_freq=100)
    
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
            
            loss, loss_obs, loss_air, loss_dynamic = compute_loss(model, c_seq, v_prev, v_curr, dt_map, device)
            
            loss.backward()
            
            # 梯度裁剪防梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # 将 Step 的数据实时上传到 wandb 仪表盘
            wandb.log({
                "train/step_loss": loss.item(),
                "train/loss_obs_raw": loss_obs.item(),
                "train/loss_air_raw": loss_air.item(),
                "train/loss_dynamic_raw": loss_dynamic.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            })
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # 步进自适应学习率 (根据实际跑出来的 Loss 来决定要不要降学习率)
        scheduler.step(avg_loss)
        
        # 将 Epoch 的汇总数据记录到 wandb
        wandb.log({
            "train/epoch_loss": avg_loss,
            "epoch": epoch + 1
        })
        
        # 每 5 个 epoch 保存一次权重
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"checkpoints/flow_model_ep{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
    # 结束 wandb 监控
    wandb.finish()

if __name__ == "__main__":
    train()
