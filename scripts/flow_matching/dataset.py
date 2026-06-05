import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt

class VoxelFlowDataset(Dataset):
    def __init__(self, data_dir, seq_len=50, is_train=True):
        """
        Flow Matching 3D Voxel Dataset
        
        Args:
            data_dir: 包含 episode_*.npz 的 dataset_voxel/ 目录路径
            seq_len: 条件序列长度 (默认 50 帧)
            is_train: 是否是训练模式（决定是否启用数据增强）
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.is_train = is_train
        
        # 寻找所有的 npz 文件
        self.files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        
        # 记录每条有效序列的索引，构建扁平化的全局映射
        self.index_map = []  
        # 不再在内存中持有所有数据 (防止超过 10000 个文件撑爆系统 RAM)
        # self.episodes = [] 
        
        print(f"Scanning {len(self.files)} episodes from {data_dir} to build index...")
        for file_idx, fpath in enumerate(self.files):
            # 仅读取一次以获取长度，随后释放内存
            with np.load(fpath) as data:
                voxel = data['collision_voxel'] 
                T = voxel.shape[0]
                
            if T > 1:
                # 滑动窗口构建索引
                # 我们需要 V[t] 作为 V_prev，V[t+1] 作为 V_curr
                # 即便 t < 49 (不够 50 帧历史)，也可以通过向前 Padding 来处理
                for t in range(0, T - 1):
                    self.index_map.append((file_idx, t))

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        fpath = self.files[file_idx]
        
        # 懒加载 (Lazy Loading): 用到这帧数据时再去硬盘里读取，防止 OOM
        with np.load(fpath) as data:
            base_ang_vel = data['base_ang_vel']
            projected_gravity = data['projected_gravity']
            joint_pos = data['joint_pos']
            joint_vel = data['joint_vel']
            last_action = data['last_action']
            joint_torques = data['joint_torques']
            
            condition = np.concatenate([
                base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action, joint_torques
            ], axis=-1).astype(np.float32)
            
            voxel = data['collision_voxel'].astype(np.float32)
        
        # 1. 提取条件序列
        start_idx = max(0, t - self.seq_len + 1)
        c_seq_actual = condition[start_idx : t + 1] # shape (L, 122), L <= 50
        
        # 如果长度不足 50 帧 (例如 t=0 时，只有 1 帧)，使用第 0 帧进行向前重复填充 (Replicate Padding)
        actual_len = c_seq_actual.shape[0]
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            first_frame = c_seq_actual[0:1] # shape (1, 122)
            pad_seq = np.repeat(first_frame, pad_len, axis=0) # shape (pad_len, 122)
            c_seq = np.concatenate([pad_seq, c_seq_actual], axis=0) # shape (50, 122)
        else:
            c_seq = c_seq_actual
            
        # 转置为 (Channels=122, SeqLen=50) 适配 PyTorch 1D Conv 的输入习惯
        c_seq = c_seq.T
        
        # 2. 提取连续两帧体素
        v_prev = voxel[t]       # (20, 20, 15)
        v_curr = voxel[t + 1]   # (20, 20, 15)
        
        # 增加 Channel 维度 -> (1, 20, 20, 15)
        v_prev = np.expand_dims(v_prev, axis=0)
        v_curr = np.expand_dims(v_curr, axis=0)
        
        # 3. 计算 D_t (3D EDT 距离场)
        # scipy EDT 会计算所有非 0 元素到最近 0 元素的欧氏距离。
        # 因为真实障碍物的值是 1，我们需要算离 1 的距离，所以传入 (1.0 - v_curr)。
        if v_curr.max() > 0:
            dt_map = distance_transform_edt(1.0 - v_curr[0])
        else:
            # 如果当前帧视野内完全没有任何障碍物，赋予一个比较大的基础惩罚距离
            dt_map = np.full_like(v_curr[0], 20.0) 
            
        dt_map = np.expand_dims(dt_map, axis=0).astype(np.float32)
        
        # 转换为 Tensor
        c_seq_t = torch.from_numpy(c_seq)
        v_prev_t = torch.from_numpy(v_prev)
        v_curr_t = torch.from_numpy(v_curr)
        dt_map_t = torch.from_numpy(dt_map)
        
        # 4. 数据增强 (Domain Randomization & Denoising)
        if self.is_train:
            # 1. 条件序列加噪
            # 修改：将 0.1 降为 0.02。
            # 原因：0.1 弧度的关节位置误差相当于 5.7 度，对高精度编码器来说大得离谱。0.02 是更符合物理现实的传感器底噪。
            c_seq_t = c_seq_t + torch.randn_like(c_seq_t) * 0.02
            
            # 2. 源体素加噪
            # 维持 0.1 不变。因为体素原值是 0 或 1。加 0.1 噪声后，值域大概在 [-0.3, 0.3] 和 [0.7, 1.3] 浮动，
            # 仍然能被 0.5 的阈值完美切分开，这刚好能模拟自回归推理时的模糊误差。
            v_prev_t = v_prev_t + torch.randn_like(v_prev_t) * 0.1
            
        return {
            'c_seq': c_seq_t,     # (122, 50)
            'v_prev': v_prev_t,   # (1, 20, 20, 15)
            'v_curr': v_curr_t,   # (1, 20, 20, 15)
            'dt_map': dt_map_t    # (1, 20, 20, 15)
        }

if __name__ == "__main__":
    # 极简测试逻辑 (兼容 docker 内部路径和外部路径)
    data_path_host = "/home/hz/proprioception/unitree_rl_lab/dataset_voxel"
    data_path_docker = "/home/hz/project/dataset_voxel"
    
    data_path = data_path_docker if os.path.exists(data_path_docker) else data_path_host
    
    if os.path.exists(data_path):
        dataset = VoxelFlowDataset(data_dir=data_path, seq_len=50, is_train=True)
        print(f"✅ Total samples extracted: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"🔸 c_seq  shape: {sample['c_seq'].shape} | dtype: {sample['c_seq'].dtype}")
            print(f"🔸 v_prev shape: {sample['v_prev'].shape} | dtype: {sample['v_prev'].dtype}")
            print(f"🔸 v_curr shape: {sample['v_curr'].shape} | dtype: {sample['v_curr'].dtype}")
            print(f"🔸 dt_map shape: {sample['dt_map'].shape} | dtype: {sample['dt_map'].dtype}")
    else:
        print(f"❌ Data directory not found: {data_path}")
