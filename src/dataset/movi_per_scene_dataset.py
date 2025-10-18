"""
Per Scene Dataset implementation for scene flow prediction.

This module implements a dataset loader that processes individual scenes for
scene flow prediction, handling depth images and segmentation masks.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample,rgb_array_to_int32
import cv2
from utils.visualization_utils import remap_instance_labels
class MOVIPerSceneDataset(nn.Module):
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
    
    def __init__(self, dataset_path=None, motion_threshold=0.01):
        """
        Initialize the per scene dataset loader.
        
        Args:
            dataset_path (str): Path to the MOVI-F dataset directory
            motion_threshold (float): Threshold for motion filtering
        """
        super(MOVIPerSceneDataset, self).__init__()
        self.traj = None
        self.dataset_path = dataset_path
        self.motion_threshold = motion_threshold
        
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
            self.traj = torch.from_numpy(self.traj).to(torch.float32)
            #read segmentation image
            seg_img = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
            seg_img = rgb_array_to_int32(seg_img)
            seg_img = torch.from_numpy(seg_img).to(torch.int32)
            seg_img = remap_instance_labels(seg_img)
            self.seg_img = seg_img.reshape(-1)
            # Load second frame
            dep_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/depth_{end:05d}.tiff"
            seg_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/segmentation_{end:05d}.png"
            traj2 = process_one_sample(metadata_path, dep_img_path_2, seg_img_path_2, f=end)
            traj2 = torch.from_numpy(traj2).to(torch.float32)
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
            "flow": (self.traj[end] - self.traj[start])[self.movement_mask],
            "dynamic_instance_mask": self.seg_img[self.movement_mask],
        }

        return sample