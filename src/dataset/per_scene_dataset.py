"""
Per Scene Dataset implementation for scene flow prediction.

This module implements a dataset loader that processes individual scenes for
scene flow prediction, handling depth images and segmentation masks.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample

class PerSceneDataset(nn.Module):
    """
    Dataset class for loading and processing individual scenes.
    
    This class handles loading and processing of depth images and segmentation masks
    for scene flow prediction, with support for motion-based point filtering.
    
    Attributes:
        traj (dict): Trajectory data containing point cloud information
        movement_mask (torch.Tensor): Mask for points with significant movement
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
    """
    
    def __init__(self):
        """
        Initialize the per scene dataset loader.
        """
        super(PerSceneDataset, self).__init__()
        self.traj = None
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return 1
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            dict: A dictionary containing:
                - point_cloud_first (torch.Tensor): First frame point cloud [N, 3]
                - point_cloud_second (torch.Tensor): Second frame point cloud [N, 3]
                - flow (torch.Tensor): Ground truth flow vectors [N, 3]
        """
        start = 0
        end = 1
        if self.traj is None:
            # Load scene data
            metadata_path = "/home/lzq/workspace/gan_seg/dataset/0/metadata.json"
            dep_img_path = f"/home/lzq/workspace/gan_seg/dataset/0/depth_{start:05d}.tiff"
            seg_img_path = f"/home/lzq/workspace/gan_seg/dataset/0/segmentation_{start:05d}.png"
            self.traj = process_one_sample(metadata_path, dep_img_path, seg_img_path, f=start)
            
            # Load second frame
            dep_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/depth_{end:05d}.tiff"
            seg_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/segmentation_{end:05d}.png"
            traj2 = process_one_sample(metadata_path, dep_img_path_2, seg_img_path_2, f=end)
            
            # Calculate movement masks
            movement = self.traj[end] - self.traj[start]
            movement = torch.tensor(movement)
            self.movement_mask = torch.norm(movement, dim=1) > 0.01

            movement2 = traj2[end] - traj2[start]
            movement2 = torch.tensor(movement2)
            self.movement_mask2 = torch.norm(movement2, dim=1) > 0.01
            
            # Extract point clouds
            self.point_cloud_first = self.traj[start][self.movement_mask]
            self.point_cloud_second = traj2[end][self.movement_mask2]
            
        # Prepare sample
        sample = {
            "point_cloud_first": self.point_cloud_first,
            "point_cloud_second": self.point_cloud_second,
            "flow": (self.traj[end] - self.traj[start])[self.movement_mask]
        }

        return sample