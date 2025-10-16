"""
AV2 Dataset implementation for scene flow prediction.

This module implements the AV2 dataset loader, which handles loading and preprocessing
of point cloud data from the AV2 dataset format.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample
from .datasetutil.av2 import read_av2_scene
import random
cache = {}



class AV2SequenceDataset(nn.Module):
    """
    Dataset class for loading and processing AV2 dataset.
    
    This class handles loading point cloud data from AV2 format files and provides
    the necessary preprocessing for scene flow prediction.
    
    Attributes:
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
        flow (torch.Tensor): Ground truth flow vectors
    """
    
    def __init__(self, fix_ego_motion=True, max_k=1, apply_ego_motion=True, 
                 train_scene_path=None, test_scene_path=None, motion_threshold=0.05):
        """
        Initialize the AV2 dataset loader.
        
        Args:
            fix_ego_motion (bool): Whether to fix ego motion
            max_k (int): Maximum k value for sequence length
            apply_ego_motion (bool): Whether to apply ego motion
            train_scene_path (str): Path to training scene file
            test_scene_path (str): Path to test scene file
            motion_threshold (float): Threshold for motion filtering
        """
        super(AV2SequenceDataset, self).__init__()
        self.point_cloud_first = None
        self.apply_ego_motion = apply_ego_motion
        self.motion_threshold = motion_threshold
        
        # Use provided paths or fallback to default
        self.av2_scene_path = train_scene_path or "/workspace/gan_seg/demo_data/demo/train/8de6abb6-6589-3da7-8e21-6ecc80004a36.h5"
        self.av2_test_scene_path = test_scene_path or "/workspace/gan_seg/demo_data/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5"
        
        if self.apply_ego_motion:
            assert fix_ego_motion, "fix_ego_motion must be True when apply_ego_motion is True"
        self.av2_dataset = read_av2_scene(self.av2_scene_path, apply_ego_motion=apply_ego_motion)
        self.sequence_length = len(list(self.av2_dataset.keys()))
        self.fix_ego_motion = fix_ego_motion
        self.max_k = max_k
        self.fixed_scene_idx = None

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return self.sequence_length - 2 - self.max_k
    def get_item(self, idx):
        return self.prepare_item(idx,from_manual=True)

    def prepare_item(self, idx,from_manual=False):
        if idx < self.max_k and not from_manual:
            return {}
        if self.fixed_scene_idx is not None and not from_manual:
            idx = self.fixed_scene_idx
        longseq = random.random() < 0.5
        if longseq:
            k = self.max_k
        else:
            k = 1
        keys = list(self.av2_dataset.keys())
        if idx in cache:
            cache[idx]["k"] = k
            return cache[idx]
        first_key = keys[idx]
        first_value = self.av2_dataset[first_key]
        valid_mask = first_value["flow_is_valid"]
        cropped_mask = first_value["point_cloud_first_cropped_mask"]
        dynamic_mask = first_value["flow_category"] != 0
        ground_mask = first_value["ground_mask"]
        valid_mask = valid_mask & dynamic_mask & (~ ground_mask) & cropped_mask
        point_cloud_first = first_value["point_cloud_first"][valid_mask]
        ego_motion = first_value["ego_motion"]
        
        flow = first_value["flow"]
        if not self.apply_ego_motion:
            flow = torch.matmul(flow - ego_motion[:3, 3], ego_motion[:3, :3].T)
        motion_mask = torch.linalg.norm(flow, dim=1) > self.motion_threshold
        if self.fix_ego_motion:
            pass
        else:
            flow = first_value["flow"]
        flow = flow[valid_mask]
        dynamic_instance_mask = (motion_mask*first_value["label"])[valid_mask]
        """

        Category	Description
        Foreground/Background	A point belongs to the foreground if it is contained in the bounding box of any tracked object.
        Dynamic/Static	A point is dynamic if it is moving faster than 0.5 m/s in the world frame. Since each pair of sweeps spans 0.1s, this is equivalent to a point having a flow vector with a norm of at least 0.05m once ego-motion has been removed.
        """
        background_static_mask = first_value["label"] == 0
        foreground_static_mask = (first_value["label"] != 0) & (~motion_mask)
        foreground_dynamic_mask = (first_value["label"] != 0) & motion_mask
        background_static_mask = background_static_mask[valid_mask]
        foreground_static_mask = foreground_static_mask[valid_mask]
        foreground_dynamic_mask = foreground_dynamic_mask[valid_mask]

        # Prepare sample
        sample = {
            "point_cloud_first": point_cloud_first,
            "flow": flow,
            'dynamic_instance_mask': dynamic_instance_mask,
            'background_static_mask': background_static_mask,
            'foreground_static_mask': foreground_static_mask,
            'foreground_dynamic_mask': foreground_dynamic_mask,
            "idx": idx,
            "total_frames": self.sequence_length,
            "self": self,
            "k": k,
            "ego_motion": ego_motion,

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
        """
        
        return self.prepare_item(idx,from_manual=False)


class AV2PerSceneDataset(AV2SequenceDataset):
    """
    Dataset class that inherits from AV2SequenceDataset but always returns a fixed scene.
    
    This class extends AV2SequenceDataset to always return the same fixed scene when accessed
    through __getitem__, while still supporting get_item(idx) for accessing specific scenes.
    
    Attributes:
        fixed_scene_idx (int): Fixed scene index to always return
    """
    
    def __init__(self, fix_ego_motion=True, apply_ego_motion=True, fixed_scene_idx=5):
        """
        Initialize the AV2PerSceneDataset.
        
        Args:
            fix_ego_motion (bool): Whether to fix ego motion
            max_k (int): Maximum k value for sequence
            apply_ego_motion (bool): Whether to apply ego motion
            fixed_scene_idx (int): Fixed scene index to always return (default: 5)
        """
        # Initialize parent class
        super(AV2PerSceneDataset, self).__init__(fix_ego_motion, 1, apply_ego_motion)
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
    