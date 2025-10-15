"""
MOVI-F Sequence Dataset implementation for scene flow prediction.

This module implements the MOVI-F dataset loader for continuous sequences,
which handles loading and preprocessing of point cloud data from MOVI-F format files.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample, rgb_array_to_int32
import cv2
import os
import random
import json
from typing import Dict, List, Optional

cache = {}


def remap_instance_labels(labels):
    """
    Remap arbitrary integer labels to consecutive label numbers starting from 0.
    
    For example: [0,1,8,1] -> [0,1,2,1]
    
    Args:
        labels (torch.Tensor): Input label tensor with arbitrary integer values
        
    Returns:
        torch.Tensor: Remapped label tensor with consecutive integers starting from 0
    """
    unique_labels = torch.unique(labels)
    mapping = {label.item(): idx for idx, label in enumerate(sorted(unique_labels))}
    # Create new label tensor
    remapped = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        remapped[labels == old_label] = new_label
        
    return remapped


class MOVIFSequenceDataset(nn.Module):
    """
    Dataset class for loading and processing MOVI-F dataset sequences.
    
    This class handles loading point cloud data from MOVI-F format files and provides
    the necessary preprocessing for scene flow prediction in continuous sequences.
    
    Attributes:
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
        flow (torch.Tensor): Ground truth flow vectors
        sequence_data (dict): Loaded sequence data
        sequence_length (int): Length of the sequence
    """
    
    def __init__(self, dataset_path: str, max_k: int = 1, motion_threshold: float = 0.01):
        """
        Initialize the MOVI-F sequence dataset loader.
        
        Args:
            dataset_path (str): Path to the MOVI-F dataset directory
            max_k (int): Maximum k value for sequence length
            motion_threshold (float): Threshold for motion filtering
        """
        super(MOVIFSequenceDataset, self).__init__()
        self.dataset_path = dataset_path
        self.max_k = max_k
        self.motion_threshold = motion_threshold
        self.sequence_data = {}
        self.sequence_length = 0
        self.fixed_scene_idx = None
        
        # Load sequence data
        self._load_sequence_data()
        
    def _load_sequence_data(self):
        """
        Load all frames in the sequence from the dataset directory.
        """
        # Find all depth and segmentation files
        depth_files = sorted([f for f in os.listdir(self.dataset_path) if f.startswith('depth_') and f.endswith('.tiff')])
        seg_files = sorted([f for f in os.listdir(self.dataset_path) if f.startswith('segmentation_') and f.endswith('.png')])
        
        if not depth_files or not seg_files:
            raise ValueError(f"No depth or segmentation files found in {self.dataset_path}")
            
        self.sequence_length = len(depth_files)
        
        # Load metadata
        metadata_path = os.path.join(self.dataset_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Pre-load all frames
        for i in range(self.sequence_length):
            depth_path = os.path.join(self.dataset_path, depth_files[i])
            seg_path = os.path.join(self.dataset_path, seg_files[i])
            
            # Load trajectory data using process_one_sample
            traj = process_one_sample(metadata_path, depth_path, seg_path, f=i)
            traj = torch.from_numpy(traj).to(torch.float32)
            
            # Load segmentation image
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            seg_img = rgb_array_to_int32(seg_img)
            seg_img = torch.from_numpy(seg_img).to(torch.int32)
            seg_img = remap_instance_labels(seg_img)
            seg_img = seg_img.reshape(-1)
            
            self.sequence_data[i] = {
                'trajectory': traj,
                'segmentation': seg_img,
                'depth_path': depth_path,
                'seg_path': seg_path
            }
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return self.sequence_length - 2 - self.max_k
    
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
            
        # Get frame data
        if idx >= self.sequence_length - 1:
            idx = self.sequence_length - 2
            
        first_frame = self.sequence_data[idx]
        second_frame = self.sequence_data[idx + 1]
        
        # Extract point clouds from trajectory data
        # trajectory shape: [num_frames, num_points, 3]
        point_cloud_first = first_frame['trajectory'][idx]  # First timestep of current frame
        point_cloud_second = first_frame['trajectory'][idx+1]  # First timestep of next frame
        
        # Calculate flow (movement between frames)
        flow = point_cloud_second - point_cloud_first
        
        # Calculate motion mask
        motion_mask = torch.norm(flow, dim=1) > self.motion_threshold
        
        # Apply motion filtering
        valid_mask = motion_mask
        
        # Filter point clouds and flow
        point_cloud_first = point_cloud_first[valid_mask]
        point_cloud_second = point_cloud_second[valid_mask]
        flow = flow[valid_mask]
        
        # Get segmentation masks
        seg_first = first_frame['segmentation'][valid_mask]
        seg_second = second_frame['segmentation'][valid_mask]
        
        # Create dynamic instance mask (points that moved significantly)
        dynamic_instance_mask = seg_first
        
        # Create background/foreground masks
        background_static_mask = (seg_first == 0) & (~motion_mask[valid_mask])
        foreground_static_mask = (seg_first != 0) & (~motion_mask[valid_mask])
        foreground_dynamic_mask = (seg_first != 0) & motion_mask[valid_mask]
        
        # Prepare sample
        sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_second": point_cloud_second,
            "flow": flow,
            'dynamic_instance_mask': dynamic_instance_mask,
            'background_static_mask': background_static_mask,
            'foreground_static_mask': foreground_static_mask,
            'foreground_dynamic_mask': foreground_dynamic_mask,
            "idx": idx,
            "total_frames": self.sequence_length,
            "self": self,
            "k": k,
            "ego_motion": torch.eye(4),  # Identity matrix for ego motion (not applicable for MOVI-F)
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


class MOVIFPerSceneDataset(MOVIFSequenceDataset):
    """
    Dataset class that inherits from MOVIFSequenceDataset but always returns a fixed scene.
    
    This class extends MOVIFSequenceDataset to always return the same fixed scene when accessed
    through __getitem__, while still supporting get_item(idx) for accessing specific scenes.
    
    Attributes:
        fixed_scene_idx (int): Fixed scene index to always return
    """
    
    def __init__(self, dataset_path: str, max_k: int = 1, motion_threshold: float = 0.01, fixed_scene_idx: int = 5):
        """
        Initialize the MOVI-F Per Scene Dataset.
        
        Args:
            dataset_path (str): Path to the MOVI-F dataset directory
            max_k (int): Maximum k value for sequence
            motion_threshold (float): Threshold for motion filtering
            fixed_scene_idx (int): Fixed scene index to always return (default: 5)
        """
        # Initialize parent class
        super(MOVIFPerSceneDataset, self).__init__(dataset_path, max_k, motion_threshold)
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

