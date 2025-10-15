"""
KITTI Sequence Dataset implementation for scene flow prediction.

This module implements the KITTI dataset loader for continuous sequences,
which handles loading and preprocessing of point cloud data from KITTI format files.
"""

import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import random
from typing import Dict, List, Optional

cache = {}


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


class KITTISequenceDataset(nn.Module):
    """
    Dataset class for loading and processing KITTI dataset sequences.
    
    This class handles loading point cloud data from KITTI format files and provides
    the necessary preprocessing for scene flow prediction in continuous sequences.
    
    Attributes:
        data_root (str): Root directory containing the dataset
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
        flow (torch.Tensor): Ground truth flow vectors
        sequence_data (dict): Loaded sequence data
        sequence_length (int): Length of the sequence
    """
    
    def __init__(self, data_root: str, max_k: int = 1, num_points: int = 8192, 
                 downsampled: bool = False, motion_threshold: float = 0.05):
        """
        Initialize the KITTI sequence dataset loader.
        
        Args:
            data_root (str): Root directory containing the dataset
            max_k (int): Maximum k value for sequence length
            num_points (int): Number of points to sample from each point cloud
            downsampled (bool): Whether to use downsampled version of the data
            motion_threshold (float): Threshold for motion filtering
        """
        super(KITTISequenceDataset, self).__init__()
        self.data_root = data_root
        self.max_k = max_k
        self.num_points = num_points
        self.downsampled = downsampled
        self.motion_threshold = motion_threshold
        self.sequence_data = {}
        self.sequence_length = 0
        self.fixed_scene_idx = None
        
        # Set up data paths
        if downsampled:
            self.data_root = osp.join(data_root, 'data')
        else:
            self.data_root = osp.join(data_root, 'processed')
            
        # Load sequence data
        self._load_sequence_data()
        
    def _load_sequence_data(self):
        """
        Load all scenes in the dataset directory.
        """
        # Get all scene directories
        self.scene_dirs = []
        for scene_id in os.listdir(self.data_root):
            scene_path = osp.join(self.data_root, scene_id)
            if osp.isdir(scene_path):
                self.scene_dirs.append(scene_id)
                
        if not self.scene_dirs:
            raise RuntimeError(f"No scene directories found in {self.data_root}")
            
        self.sequence_length = len(self.scene_dirs)
        
        # Pre-load all scenes
        for i, scene_id in enumerate(self.scene_dirs):
            scene_path = osp.join(self.data_root, scene_id)
            
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
            
            # Process segmentation mask
            segm1 = np.reshape(segm1, -1)
            segm1 = compress_label_id(segm1[None, :])[0]  # Add and remove batch dimension
            
            self.sequence_data[i] = {
                'pc1': pc1,
                'pc2': pc2,
                'segm1': segm1,
                'flow': flow,
                'scene_id': scene_id,
                'scene_path': scene_path
            }
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return self.sequence_length - 1 - self.max_k
    
    def get_item(self, idx):
        """
        Get a specific item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            dict: Sample data dictionary
        """
        return self.prepare_item(idx, from_manual=True)
    
    def prepare_item(self, idx, from_manual=False):
        """
        Prepare a sample item from the dataset.
        
        Args:
            idx (int): Index of the sample
            from_manual (bool): Whether this is called manually
            
        Returns:
            dict: Sample data dictionary
        """
        if idx < self.max_k and not from_manual:
            return {}
        if self.fixed_scene_idx is not None and not from_manual:
            idx = self.fixed_scene_idx
            
        # Randomly choose sequence length
        longseq = random.random() < 0.5
        if longseq:
            k = self.max_k
        else:
            k = 1
            
        # Check cache
        if idx in cache:
            cache[idx]["k"] = k
            return cache[idx]
            
        # Get scene data
        if idx >= self.sequence_length - 1:
            idx = self.sequence_length - 2
            
        first_scene = self.sequence_data[idx]
        second_scene = self.sequence_data[idx + 1]
        
        # Extract point clouds
        pc1 = first_scene['pc1']
        pc2 = second_scene['pc1']  # Use first frame of next scene
        
        # Calculate flow between scenes
        flow = pc2 - pc1
        
        # Calculate motion mask
        motion_mask = np.linalg.norm(flow, axis=1) > self.motion_threshold
        
        # Apply motion filtering
        valid_mask = motion_mask
        
        # Filter point clouds and flow
        pc1_filtered = pc1[valid_mask]
        pc2_filtered = pc2[valid_mask]
        flow_filtered = flow[valid_mask]
        
        # Get segmentation masks
        segm1 = first_scene['segm1'][valid_mask]
        segm2 = second_scene['segm1'][valid_mask]
        
        # Downsample points to fixed number
        pc1_sampled, segm1_sampled, flow_sampled = random_sample_points(
            pc1_filtered, segm1, flow_filtered, self.num_points
        )
        pc2_sampled = pc2_filtered[np.random.choice(
            pc2_filtered.shape[0], self.num_points, replace=True
        )]
        
        # Convert to torch tensors
        point_cloud_first = torch.from_numpy(pc1_sampled).float()
        point_cloud_second = torch.from_numpy(pc2_sampled).float()
        flow = torch.from_numpy(flow_sampled).float()
        
        # Create dynamic instance mask
        dynamic_instance_mask = torch.from_numpy(segm1_sampled).long()
        
        # Create background/foreground masks
        background_static_mask = (segm1_sampled == 0) & (~motion_mask[valid_mask][np.random.choice(
            motion_mask[valid_mask].shape[0], self.num_points, replace=True
        )])
        foreground_static_mask = (segm1_sampled != 0) & (~motion_mask[valid_mask][np.random.choice(
            motion_mask[valid_mask].shape[0], self.num_points, replace=True
        )])
        foreground_dynamic_mask = (segm1_sampled != 0) & motion_mask[valid_mask][np.random.choice(
            motion_mask[valid_mask].shape[0], self.num_points, replace=True
        )]
        
        # Prepare sample
        sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_second": point_cloud_second,
            "flow": flow,
            'dynamic_instance_mask': dynamic_instance_mask,
            'background_static_mask': torch.from_numpy(background_static_mask).bool(),
            'foreground_static_mask': torch.from_numpy(foreground_static_mask).bool(),
            'foreground_dynamic_mask': torch.from_numpy(foreground_dynamic_mask).bool(),
            "idx": idx,
            "total_frames": self.sequence_length,
            "self": self,
            "k": k,
            "ego_motion": torch.eye(4),  # Identity matrix for ego motion (not applicable for KITTI)
        }
        
        cache[idx] = sample
        return sample
    
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
                - dynamic_instance_mask (torch.Tensor): Dynamic instance segmentation mask
                - background_static_mask (torch.Tensor): Background static points mask
                - foreground_static_mask (torch.Tensor): Foreground static points mask
                - foreground_dynamic_mask (torch.Tensor): Foreground dynamic points mask
        """
        return self.prepare_item(idx, from_manual=False)


class KITTIPerSceneDataset(KITTISequenceDataset):
    """
    Dataset class that inherits from KITTISequenceDataset but always returns a fixed scene.
    
    This class extends KITTISequenceDataset to always return the same fixed scene when accessed
    through __getitem__, while still supporting get_item(idx) for accessing specific scenes.
    
    Attributes:
        fixed_scene_idx (int): Fixed scene index to always return
    """
    
    def __init__(self, data_root: str, max_k: int = 1, num_points: int = 8192, 
                 downsampled: bool = False, motion_threshold: float = 0.05, 
                 fixed_scene_idx: int = 5):
        """
        Initialize the KITTI Per Scene Dataset.
        
        Args:
            data_root (str): Root directory containing the dataset
            max_k (int): Maximum k value for sequence
            num_points (int): Number of points to sample from each point cloud
            downsampled (bool): Whether to use downsampled version of the data
            motion_threshold (float): Threshold for motion filtering
            fixed_scene_idx (int): Fixed scene index to always return (default: 5)
        """
        # Initialize parent class
        super(KITTIPerSceneDataset, self).__init__(
            data_root, max_k, num_points, downsampled, motion_threshold
        )
        self.fixed_scene_idx = fixed_scene_idx
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset (always 1 for fixed scene)
        """
        return 1
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset. Always returns the fixed scene.
        
        Args:
            idx (int): Index (ignored, always returns fixed scene)
            
        Returns:
            dict: A dictionary containing scene data for the fixed scene
        """
        return self.prepare_item(self.fixed_scene_idx, from_manual=False)

