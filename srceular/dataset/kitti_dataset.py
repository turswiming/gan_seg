"""
KITTI Dataset implementation for scene flow prediction.

This module implements the KITTI dataset loader, which handles loading and preprocessing
of point cloud data from the KITTI dataset format.
"""

import os
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import random

def compress_label_id(segms):
    """
    Compress the object-id in segmentation to consecutive numbers starting from 0 (0, 1, 2, ...).
    :param segm: (B, N).
    :return:
        segm_cpr: (B, N).
    """
    segms_cpr = []
    for b in range(segms.shape[0]):
        segm = segms[b]
        _, segm_cpr = np.unique(segm, return_inverse=True)
        segms_cpr.append(segm_cpr)
    segms_cpr = np.stack(segms_cpr, 0)
    return segms_cpr

def random_sample_points(points, labels, flows, num_points):
    """
    Perform voxel downsampling followed by random sampling to obtain a fixed number of points.
    
    Args:
        points (np.ndarray): Point cloud data [N, 3]
        labels (np.ndarray): Point labels [N]
        flows (np.ndarray): Flow vectors [N, 3]
        num_points (int): Number of points to sample
        
    Returns:
        tuple: (sampled_points, sampled_labels, sampled_flows)
    """
    
    # Random sampling
    num_total_points = points.shape[0]
    if num_total_points > num_points:
        indices = np.random.choice(num_total_points, num_points, replace=False)
    else:
        indices = np.random.choice(num_total_points, num_points, replace=True)
        
    return points[indices], labels[indices], flows[indices]

class KITTIPerSceneDataset(nn.Module):
    """
    Dataset class for loading and processing KITTI dataset.
    
    This class handles loading point cloud data from KITTI format files and provides
    the necessary preprocessing for scene flow prediction.
    
    Attributes:
        data_root (str): Root directory containing the dataset
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
        flow (torch.Tensor): Ground truth flow vectors
        fixed_scene_id (str): If set, always return this specific scene
        num_points (int): Number of points to sample from each point cloud
    """
    
    def __init__(self, data_root, downsampled=False, fixed_scene_id=None, num_points=8192):
        """
        Initialize the KITTI dataset loader.
        
        Args:
            data_root (str): Root directory containing the dataset
            downsampled (bool): Whether to use downsampled version of the data
            fixed_scene_id (str, optional): If set, always return this specific scene.
                                          Format should be like "000000". Defaults to None.
            num_points (int): Number of points to sample from each point cloud
        """
        super(KITTIPerSceneDataset, self).__init__()
        
        # Initialize dataset paths
        if downsampled:
            self.data_root = osp.join(data_root, 'data')
        else:
            self.data_root = osp.join(data_root, 'processed')
            
        self.fixed_scene_id = fixed_scene_id
        self.num_points = num_points
        
        # Get all scene directories
        self.scene_dirs = []
        for scene_id in os.listdir(self.data_root):
            scene_path = osp.join(self.data_root, scene_id)
            if osp.isdir(scene_path):
                self.scene_dirs.append(scene_id)
                
        if not self.scene_dirs:
            raise RuntimeError(f"No scene directories found in {self.data_root}")
            
        if fixed_scene_id is not None:
            if fixed_scene_id not in self.scene_dirs:
                raise ValueError(f"Specified fixed_scene_id '{fixed_scene_id}' not found in dataset")
            print(f"Dataset will always return scene: {fixed_scene_id}")
            
        self.downsampled = downsampled
        self.point_cloud_first = None
        self.point_cloud_second = None
        self.flow = None
        self.dynamic_instance_mask = None
        self.current_scene = None
        
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
                - dynamic_instance_mask (torch.Tensor): Instance segmentation mask [N]
        """
        if self.point_cloud_first is None or self.fixed_scene_id is None:
            # Select scene - either fixed or random
            print(f"Loading scene {self.current_scene} from {self.data_root}")
            if self.fixed_scene_id is not None:
                self.current_scene = self.fixed_scene_id
            else:
                self.current_scene = random.choice(self.scene_dirs)
                
            scene_path = osp.join(self.data_root, self.current_scene)
            
            # Load point clouds
            pc1 = np.load(osp.join(scene_path, 'pc1.npy'))
            pc2 = np.load(osp.join(scene_path, 'pc2.npy'))
            
            # Load segmentation and flow
            if self.downsampled:
                segm1 = np.load(osp.join(scene_path, 'segm1.npy'))
                flow = np.load(osp.join(scene_path, 'flow1.npy'))
            else:
                segm = np.load(osp.join(scene_path, 'segm.npy'))
                segm1 = segm
                flow = pc2 - pc1
            
            # Downsample points to fixed number
            pc1, segm1, flow = random_sample_points(pc1, segm1, flow, self.num_points)
            pc2 = pc2[np.random.choice(pc2.shape[0], self.num_points, replace=True)]  # Sample second frame independently
            
            # Convert to torch tensors
            self.point_cloud_first = torch.from_numpy(pc1).float()
            self.point_cloud_second = torch.from_numpy(pc2).float()
            self.flow = torch.from_numpy(flow).float()
            
            # Process segmentation mask
            segm1 = np.reshape(segm1, -1)
            segm1 = compress_label_id(segm1[None, :])[0]  # Add and remove batch dimension
            self.dynamic_instance_mask = torch.from_numpy(segm1).long()
            if self.fixed_scene_id is not None:
                self.point_cloud_first = self.point_cloud_first.to('cuda')
                self.point_cloud_second = self.point_cloud_second.to('cuda')
                self.flow = self.flow.to('cuda')
                self.dynamic_instance_mask = self.dynamic_instance_mask.to('cuda')
            
        # Prepare sample dictionary
        sample = {
            "point_cloud_first": self.point_cloud_first,
            "point_cloud_second": self.point_cloud_second,
            "flow": self.flow,
            "dynamic_instance_mask": self.dynamic_instance_mask,
        }
        
        return sample 