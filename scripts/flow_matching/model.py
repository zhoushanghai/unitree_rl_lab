import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionEncoder(nn.Module):
    """
    编码过去 50 帧的本体感知信息 (Batch, 93, 50)
    """
    def __init__(self, in_channels=93, out_dim=512):
        super().__init__()
        # 时序降采样：50 -> 25 -> 13 -> 7
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
        )
        # 经过 3 层 stride=2，长度从 50 -> 7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, 93, 50)
        feat = self.conv_net(x)
        out = self.fc(feat)
        return out

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_channels * 2)
        
    def forward(self, x, cond):
        # cond: (B, cond_dim)
        # proj: (B, num_channels * 2)
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        # reshape for broadcast (B, C, 1, 1, 1)
        scale = scale.view(-1, scale.size(1), 1, 1, 1)
        shift = shift.view(-1, shift.size(1), 1, 1, 1)
        return x * (1 + scale) + shift

class ResnetBlock3D(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_c)
        self.act1 = nn.SiLU()
        
        self.film = FiLMLayer(cond_dim, out_c)
        
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_c)
        self.act2 = nn.SiLU()
        
        if in_c != out_c:
            self.shortcut = nn.Conv3d(in_c, out_c, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film(h, cond)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.shortcut(x)

class VoxelFlowNet(nn.Module):
    def __init__(self, in_channels=1, cond_dim=512, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        self.cond_encoder = ConditionEncoder(in_channels=93, out_dim=cond_dim)
        
        total_cond_dim = cond_dim + time_emb_dim
        
        # Encoder (Downsampling)
        # Input: 20x20x15
        self.enc1 = ResnetBlock3D(in_channels, 256, total_cond_dim)
        self.down1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1) 
        # -> 10x10x8
        
        self.enc2 = ResnetBlock3D(512, 512, total_cond_dim)
        self.down2 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1)
        # -> 5x5x4
        
        # Bottleneck
        self.mid = ResnetBlock3D(1024, 1024, total_cond_dim)
        
        # Decoder (Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResnetBlock3D(1024 + 512, 512, total_cond_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResnetBlock3D(512 + 256, 256, total_cond_dim)
        
        self.final_conv = nn.Conv3d(256, in_channels, kernel_size=3, padding=1)

    def forward(self, x_s, s, c_seq):
        """
        x_s: (B, 1, 20, 20, 15) 当前虚拟时间的体素流形
        s: (B,) 虚拟时间 0~1
        c_seq: (B, 93, 50) 历史本体感知序列
        """
        # 1. 编码时间和条件
        t_emb = self.time_mlp(s)
        c_emb = self.cond_encoder(c_seq)
        cond = torch.cat([t_emb, c_emb], dim=-1) # (B, 256 + 512)
        
        # pad z from 15 to 16 for cleaner downsampling
        # (B, C, X, Y, Z) -> pad back of Z
        x_pad = F.pad(x_s, (0, 1, 0, 0, 0, 0)) # 变为 (20, 20, 16)
        
        # Encoder
        e1 = self.enc1(x_pad, cond) # (32, 20, 20, 16)
        h = self.down1(e1)          # (64, 10, 10, 8)
        
        e2 = self.enc2(h, cond)     # (64, 10, 10, 8)
        h = self.down2(e2)          # (128, 5, 5, 4)
        
        # Bottleneck
        h = self.mid(h, cond)       # (128, 5, 5, 4)
        
        # Decoder
        h = self.up1(h)             # (128, 10, 10, 8)
        h = torch.cat([h, e2], dim=1)
        h = self.dec1(h, cond)      # (64, 10, 10, 8)
        
        h = self.up2(h)             # (64, 20, 20, 16)
        h = torch.cat([h, e1], dim=1)
        h = self.dec2(h, cond)      # (32, 20, 20, 16)
        
        out = self.final_conv(h)    # (1, 20, 20, 16)
        
        # 截掉之前 pad 的那一层 Z
        out = out[:, :, :, :, :15]  # 还原回 (1, 20, 20, 15)
        
        return out

if __name__ == "__main__":
    # 测试前向传播维度
    model = VoxelFlowNet()
    x_s = torch.randn(2, 1, 20, 20, 15)
    s = torch.rand(2)
    c_seq = torch.randn(2, 93, 50)
    
    out = model(x_s, s, c_seq)
    print("Input shape:", x_s.shape)
    print("Output shape:", out.shape)
    print("Model parameter count:", sum(p.numel() for p in model.parameters()))
