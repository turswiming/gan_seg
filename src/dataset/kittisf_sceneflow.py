import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random


class KittisfSceneFlowDataset(Dataset):
    """
    Kittisf Scene Flow Dataset
    读取 kittisf 数据并返回与 AV2SceneFlowZoo 相同的格式
    """
    
    def __init__(self, 
                 data_root: str = "/workspace/kittisf/processed",
                 split: str = "train",  # "train" or "val"
                 num_points: int = 8192,
                 seed: int = 42):
        """
        Args:
            data_root: kittisf 数据根目录
            split: 数据集分割 ("train" or "val")
            num_points: 下采样后的点数
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.seed = seed
        
        
        # 获取所有可用的序列ID (000000 到 000199)
        all_sequence_ids = [f"{i:06d}" for i in range(200)]
        
        # 固定分割：前160个为train，后40个为val
        if split == "train":
            self.sequence_ids = all_sequence_ids[:160]  # 000000-000159
        else:  # val
            self.sequence_ids = all_sequence_ids[160:]  # 000160-000199
        
        print(f"KittisfSceneFlowDataset - {split}: {len(self.sequence_ids)} sequences")
    
    def __len__(self):
        return len(self.sequence_ids)
    
    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        sequence_path = self.data_root / sequence_id
        
        # 检查数据是否存在
        pc1_path = sequence_path / "pc1.npy"
        pc2_path = sequence_path / "pc2.npy"
        segm_path = sequence_path / "segm.npy"
        
        if not all([pc1_path.exists(), pc2_path.exists(), segm_path.exists()]):
            raise FileNotFoundError(f"Missing data files for sequence {sequence_id}")
        
        # 加载数据
        pc1 = np.load(pc1_path)  # (N, 3)
        pc2 = np.load(pc2_path)  # (N, 3)
        segm = np.load(segm_path)  # (N,)
        
        # 计算 flow = pc2 - pc1
        flow = pc2 - pc1  # (N, 3)
        
        # 下采样到固定点数
        pc1, pc2, flow, segm = self._downsample_points(pc1, pc2, flow, segm)
        
        # 转换为 torch tensor
        point_cloud_first = torch.from_numpy(pc1).float()
        point_cloud_next = torch.from_numpy(pc2).float()
        flow = torch.from_numpy(flow).float()
        mask = torch.from_numpy(segm).long()
        
        # 返回与 AV2SceneFlowZoo 相同的格式
        sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_next": point_cloud_next,
            "flow": flow,
            "mask": mask
        }
        
        return sample
    
    def _downsample_points(self, pc1, pc2, flow, segm):
        """
        下采样到固定点数
        """
        num_available = pc1.shape[0]
        
        if num_available >= self.num_points:
            # 随机选择点
            indices = np.random.choice(num_available, self.num_points, replace=False)
            indices = np.sort(indices)  # 保持顺序
        else:
            # 如果点数不够，重复采样
            indices = np.random.choice(num_available, self.num_points, replace=True)
            indices = np.sort(indices)
        
        return pc1[indices], pc2[indices], flow[indices], segm[indices]



