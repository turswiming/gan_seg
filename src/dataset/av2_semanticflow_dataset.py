"""
AV2 Dataset implementation for SemanticFlow-style preprocessed data.

This module implements a PyTorch dataset loader for AV2 data that has been
preprocessed using the preprocess_av2_semanticflow_style.py script.

The loader reads from index.pkl and .h5 files in the same format as SemanticFlow.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class AV2SemanticFlowDataset(Dataset):
    """
    Dataset class for loading SemanticFlow-style preprocessed AV2 data.

    This dataset reads from an index.pkl file and corresponding .h5 files,
    following the same format used in the SemanticFlow project.

    Attributes:
        root_dir (Path): Root directory containing preprocessed data
        split (str): Dataset split ('train', 'val', 'test')
        data_index (List[Tuple[str, str]]): List of (scene_id, timestamp) pairs
        h5_cache (Dict[str, h5py.File]): Cache of opened h5 files
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 with_flow: bool = True,
                 point_cloud_range: Tuple[float, float, float, float, float, float] = (
                         -48.0, -48.0, -2.5, 48.0, 48.0, 2.5
                 ),
                 max_points: int = 8192,
                 use_random_sampling: bool = True):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of preprocessed data
            split (str): Dataset split ('train', 'val', 'test')
            with_flow (bool): Whether to load flow data (for supervised learning)
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) for cropping
            max_points (int): Maximum number of points to sample
            use_random_sampling (bool): Whether to use random sampling (True) or FPS
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.with_flow = with_flow
        self.point_cloud_range = point_cloud_range
        self.max_points = max_points
        self.use_random_sampling = use_random_sampling

        # Load index
        index_path = self.root_dir / 'index_total.pkl'
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, 'rb') as f:
            self.data_index: List[List[str]] = pickle.load(f)

        print(f"Loaded {len(self.data_index)} samples from {split} split")

        # Cache for h5 files
        self.h5_cache = {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_index) - 1  # -1 for consecutive pairs

    def _load_h5_file(self, scene_id: str) -> h5py.File:
        """Load or get cached h5 file."""
        if scene_id not in self.h5_cache:
            file_path = self.root_dir / f"{scene_id}.h5"
            if not file_path.exists():
                raise FileNotFoundError(f"H5 file not found: {file_path}")
            self.h5_cache[scene_id] = h5py.File(file_path, 'r')
        return self.h5_cache[scene_id]

    def _crop_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        """Crop point cloud to specified range."""
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        mask = torch.all(torch.stack([
            points[:, 0] >= x_min,
            points[:, 0] <= x_max,
            points[:, 1] >= y_min,
            points[:, 1] <= y_max,
            points[:, 2] >= z_min,
            points[:, 2] <= z_max
        ], dim=0), dim=0)

        return mask

    def _sample_points(self, points: torch.Tensor, features: Dict[str, torch.Tensor]) -> Dict:
        """Sample points to max_points using random sampling or FPS."""
        num_points = points.shape[0]

        if num_points <= self.max_points:
            # Pad with zeros
            padding = self.max_points - num_points
            if padding > 0:
                indices = torch.cat([
                    torch.arange(num_points),
                    torch.zeros(padding, dtype=torch.long)
                ])
                points = points[indices]
                for key in features:
                    if features[key] is not None:
                        features[key] = features[key][indices]
        else:
            # Sample points
            if self.use_random_sampling:
                # Random sampling
                indices = torch.randperm(num_points)[:self.max_points]
            else:
                # FPS (simplified - randomly select seed points)
                # NOTE: Full FPS implementation would be more complex
                indices = torch.randperm(num_points)[:self.max_points]

            points = points[indices]
            for key in features:
                if features[key] is not None:
                    features[key] = features[key][indices]

        return points, features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            dict: Dictionary containing:
                - point_cloud_first: First frame point cloud [N, 3]
                - point_cloud_second: Second frame point cloud [N, 3]
                - flow: Scene flow vectors [N, 3]
                - ground_mask: Ground point mask [N]
                - flow_is_valid: Flow validity mask [N]
                - flow_category_indices: Object categories [N]
                - ego_motion: Ego motion SE3 matrix [4, 4]
                - eval_mask: Evaluation mask [N] (if available)
        """
        # Get consecutive frame pair
        scene_id_0, timestamp_0 = self.data_index[idx]
        scene_id_1, timestamp_1 = self.data_index[idx + 1]

        # Check if same scene
        if scene_id_0 != scene_id_1:
            # Different scenes, skip to next valid pair
            return self.__getitem__(idx + 1)

        scene_id = scene_id_0

        # Load h5 file
        h5_file = self._load_h5_file(scene_id)

        # Load first frame
        group_0 = h5_file[timestamp_0]
        pc0 = torch.from_numpy(np.array(group_0['lidar'])).float()
        ground_mask_0 = torch.from_numpy(np.array(group_0['ground_mask'])).bool()
        pose_0 = torch.from_numpy(np.array(group_0['pose'])).float()

        # Load second frame
        group_1 = h5_file[timestamp_1]
        pc1 = torch.from_numpy(np.array(group_1['lidar'])).float()
        ground_mask_1 = torch.from_numpy(np.array(group_1['ground_mask'])).bool()

        # Load flow data if available
        flow = None
        flow_is_valid = None
        flow_category = None
        ego_motion = None

        if self.with_flow and self.split in ['train', 'val']:
            if 'flow' in group_0:
                flow = torch.from_numpy(np.array(group_0['flow'])).float()
                flow_is_valid = torch.from_numpy(np.array(group_0['flow_is_valid'])).bool()
                flow_category = torch.from_numpy(np.array(group_0['flow_category_indices'])).long()

                # Get ego motion for frame 0→1
                if 'ego_motion' in group_0:
                    ego_motion = torch.from_numpy(np.array(group_0['ego_motion'])).float()
                else:
                    # Compute from poses
                    ego_motion = torch.eye(4)

        # Load eval mask if available
        eval_mask = None
        if 'eval_mask' in group_0:
            eval_mask = torch.from_numpy(np.array(group_0['eval_mask'])).bool()

        # Crop point clouds
        crop_mask_0 = self._crop_point_cloud(pc0)
        pc0 = pc0[crop_mask_0]
        ground_mask_0 = ground_mask_0[crop_mask_0]
        if flow is not None:
            flow = flow[crop_mask_0]
            flow_is_valid = flow_is_valid[crop_mask_0]
            flow_category = flow_category[crop_mask_0]
        if eval_mask is not None:
            eval_mask = eval_mask[crop_mask_0]

        crop_mask_1 = self._crop_point_cloud(pc1)
        pc1 = pc1[crop_mask_1]
        ground_mask_1 = ground_mask_1[crop_mask_1]

        # Match point clouds (simple implementation)
        # In practice, you might want to use nearest neighbor matching
        min_points = min(pc0.shape[0], pc1.shape[0])
        pc0 = pc0[:min_points]
        pc1 = pc1[:min_points]
        ground_mask_0 = ground_mask_0[:min_points]
        ground_mask_1 = ground_mask_1[:min_points]
        if flow is not None:
            flow = flow[:min_points]
            flow_is_valid = flow_is_valid[:min_points]
            flow_category = flow_category[:min_points]
        if eval_mask is not None:
            eval_mask = eval_mask[:min_points]

        # Sample points
        features = {
            'ground_mask': ground_mask_0,
            'flow_is_valid': flow_is_valid,
            'flow_category': flow_category,
            'eval_mask': eval_mask,
        }

        pc0, features = self._sample_points(pc0, features)
        ground_mask_0 = features['ground_mask']
        flow_is_valid = features['flow_is_valid']
        flow_category = features['flow_category']
        eval_mask = features['eval_mask']

        # Do the same for second frame
        if pc1.shape[0] > self.max_points:
            indices = torch.randperm(pc1.shape[0])[:self.max_points]
            pc1 = pc1[indices]
            ground_mask_1 = ground_mask_1[indices]

        # Prepare output
        sample = {
            'point_cloud_first': pc0,
            'point_cloud_second': pc1,
            'ground_mask': ground_mask_0,
            'ground_mask_next': ground_mask_1,
        }

        if flow is not None:
            sample.update({
                'flow': flow,
                'flow_is_valid': flow_is_valid,
                'flow_category': flow_category,
                'ego_motion': ego_motion,
            })

        if eval_mask is not None:
            sample['eval_mask'] = eval_mask

        sample['sequence_id'] = scene_id
        sample['timestamp'] = timestamp_0

        return sample

    def close(self):
        """Close all cached h5 files."""
        for h5_file in self.h5_cache.values():
            h5_file.close()
        self.h5_cache.clear()

    def __del__(self):
        """Destructor to ensure files are closed."""
        self.close()


class AV2SemanticFlowDataModule:
    """
    PyTorch Lightning-style DataModule for AV2 SemanticFlow dataset.

    Handles train/val/test splits and data loading.
    """

    def __init__(self,
                 root_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 max_points: int = 8192,
                 point_cloud_range: Tuple[float, float, float, float, float, float] = (
                         -48.0, -48.0, -2.5, 48.0, 48.0, 2.5
                 )):
        """
        Initialize the DataModule.

        Args:
            root_dir (str): Root directory of preprocessed data
            batch_size (int): Batch size for data loading
            num_workers (int): Number of worker processes
            max_points (int): Maximum number of points per sample
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_points = max_points
        self.point_cloud_range = point_cloud_range

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split."""
        if stage == 'fit' or stage is None:
            self.train_dataset = AV2SemanticFlowDataset(
                root_dir=self.root_dir,
                split='train',
                max_points=self.max_points,
                point_cloud_range=self.point_cloud_range
            )
            self.val_dataset = AV2SemanticFlowDataset(
                root_dir=self.root_dir,
                split='val',
                max_points=self.max_points,
                point_cloud_range=self.point_cloud_range
            )

        if stage == 'test' or stage is None:
            self.test_dataset = AV2SemanticFlowDataset(
                root_dir=self.root_dir,
                split='test',
                with_flow=False,  # No flow for test set
                max_points=self.max_points,
                point_cloud_range=self.point_cloud_range
            )

    def train_dataloader(self):
        """Return training dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        """Return test dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


if __name__ == "__main__":
    # Test the dataset
    root_dir = "/workspace/av2data/preprocessed"

    dataset = AV2SemanticFlowDataset(
        root_dir=root_dir,
        split='val',
        max_points=8192
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
