import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from OGCModel.icp_util import icp


def downsample_points(pc1, pc2, flow, segm, num_points=1024):
    """
    下采样到固定点数
    """
    num_available = pc1.shape[0]

    if num_available >= num_points:
        # 随机选择点
        indices = np.random.choice(num_available, num_points, replace=False)
        indices = np.sort(indices)  # 保持顺序
    else:
        # 如果点数不够，重复采样
        indices = np.random.choice(num_available, num_points, replace=True)
        indices = np.sort(indices)

    return pc1[indices], pc2[indices], flow[indices], segm[indices]


def get_global_transform_matrix(pc1, pc2, flow=None):
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.cpu().numpy()
        pc2 = pc2.cpu().numpy()
        flow = flow.cpu().numpy() if flow is not None else None
    ground_mask_pc1 = pc1[:, 1] < -1.4
    ground_mask_pc2 = pc2[:, 1] < -1.4

    # get flow transform matrix
    # assume nearly all flow woule be zero, to we try to optimiza a transform matrix that make flow be zero

    homogeneous_pc1 = np.hstack((pc1, np.ones((pc1.shape[0], 1))))
    if flow is not None:
        homogeneous_flowaddpc1 = np.hstack((flow + pc1, np.ones((flow.shape[0], 1))))
        result = np.linalg.lstsq(homogeneous_pc1, homogeneous_flowaddpc1, rcond=None)
        initial_transform_matrix = result[0]
    else:
        initial_transform_matrix = np.eye(4)

    pc1_without_ground = pc1[~ground_mask_pc1]
    pc2_without_ground = pc2[~ground_mask_pc2]
    if pc1_without_ground.shape[0] > pc2_without_ground.shape[0]:
        pc1_without_ground = pc1_without_ground[: pc2_without_ground.shape[0]]
    elif pc1_without_ground.shape[0] < pc2_without_ground.shape[0]:
        pc2_without_ground = pc2_without_ground[: pc1_without_ground.shape[0]]

    icp_pc1 = pc1_without_ground.copy()
    icp_pc2 = pc2_without_ground.copy()
    # temp downsample to 1024 points
    icp_pc1, _, _, _ = downsample_points(icp_pc1, icp_pc1, icp_pc1, icp_pc1, 1024)
    icp_pc2, _, _, _ = downsample_points(icp_pc2, icp_pc2, icp_pc2, icp_pc2, 1024)

    T, distances, i = icp(
        icp_pc1, icp_pc2, initial_transform_matrix.T, max_iterations=50
    )
    return T


class KittisfSceneFlowDataset(Dataset):
    """
    Kittisf Scene Flow Dataset
    读取 kittisf 数据并返回与 AV2SceneFlowZoo 相同的格式
    """

    def __init__(
        self,
        data_root: str = "/workspace/kittisf_downwampled/kittisf_downsampled/data",
        split: str = "train",  # "train" or "val"
        num_points: int = 8192,
        seed: int = 42,
        augmentation: bool = False,
    ):
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
        self.cache = {}
        self.augmentation = augmentation
        self.fix_to_global_coord = True

        # 获取所有可用的序列ID (000000 到 000199)
        all_sequence_ids = [f"{i:06d}" for i in range(200)]

        # 固定分割：前160个为train，后40个为val
        if split == "train":
            self.sequence_ids = all_sequence_ids[:100]  # 000000-000099
        else:  # val
            self.sequence_ids = all_sequence_ids[100:]  # 000100-000199

        print(f"KittisfSceneFlowDataset - {split}: {len(self.sequence_ids)} sequences")

    def __len__(self):

        return len(self.sequence_ids)

    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        is_reverse = False

        sequence_path = self.data_root / sequence_id

        # 检查数据是否存在
        pc1_path = sequence_path / f"pc{1 if is_reverse else 2}.npy"
        pc2_path = sequence_path / f"pc{2 if is_reverse else 1}.npy"
        flow1_path = sequence_path / f"flow{1 if is_reverse else 2}.npy"
        segm_path = sequence_path / f"segm{1 if is_reverse else 2}.npy"
        flow = np.load(flow1_path)  # (N, 3)
        segm = np.load(segm_path)  # (N,)
        # 加载数据
        pc1 = np.load(pc1_path)  # (N, 3)
        pc2 = np.load(pc2_path)  # (N, 3)
        # decentralize point cloud

        # if self.augmentation:
        #     pc1, pc2, flow = self.augment_transform(pc1, pc2, flow)

        if self.fix_to_global_coord and self.split == "train" and idx % 2 == 0:
            T = get_global_transform_matrix(pc1, pc2, flow)
            rot, transl = T[:3, :3], T[:3, 3].transpose()

            flow = np.einsum("ij,nj->ni", rot.T, flow + pc1 - transl) - pc1.copy()
            pc1 = np.einsum("ij,nj->ni", rot, pc1) + transl
            pc1 = pc1.astype(np.float32)

        flow = torch.from_numpy(flow)
        point_cloud_first = torch.from_numpy(pc1).float()
        point_cloud_next = torch.from_numpy(pc2).float()
        mask = torch.from_numpy(segm).long()
        # mean_point = (torch.mean(point_cloud_first,0)+torch.mean(point_cloud_next,0))/2*torch.tensor([1,0,1])
        # point_cloud_first = point_cloud_first - mean_point
        # point_cloud_next = point_cloud_next - mean_point
        # #downsample to num_points
        point_cloud_first, point_cloud_next, flow, mask = downsample_points(
            point_cloud_first, point_cloud_next, flow, mask, self.num_points
        )
        print(f"point_cloud_first shape: {point_cloud_first.shape}")
        # 返回与 AV2SceneFlowZoo 相同的格式
        sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_next": point_cloud_next,
            "flow": flow,
            "mask": mask,
            # "icp_distances": np.mean(distances),
            "sequence_id": sequence_id,
        }

        return sample
