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

class AV2ValDataset(nn.Module):
    """
    The same as AV2PerSceneDataset in /gan_seg/src/dataset/av2_dataset.py
    Dataset class for loading and processing AV2 dataset.
    
    This class handles loading point cloud data from AV2 format files and provides
    the necessary preprocessing for scene flow prediction.
    
    Attributes:
        point_cloud_first (torch.Tensor): First frame point cloud
        point_cloud_second (torch.Tensor): Second frame point cloud
        flow (torch.Tensor): Ground truth flow vectors
    """
    
    def __init__(self):
        """
        Initialize the AV2 dataset loader.
        """
        super(AV2ValDataset, self).__init__()
        self.point_cloud_first = None
        self.av2_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/train/8de6abb6-6589-3da7-8e21-6ecc80004a36.h5"
        self.av2_test_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5"
        self.av2_dataset = read_av2_scene(self.av2_scene_path)
        self.sequence_length = len(list(self.av2_dataset.keys()))

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return self.sequence_length - 1

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

        
        # Process first frame
        keys = list(self.av2_dataset.keys())
        first_key = keys[idx]
        first_value = self.av2_dataset[first_key]
        valid_mask = first_value["flow_is_valid"]
        dynamic_mask = first_value["flow_category"] != 0
        ground_mask = first_value["ground_mask"]
        valid_mask = valid_mask & dynamic_mask & (~ ground_mask)
        point_cloud_first = first_value["point_cloud_first"][valid_mask]
        ego_motion = first_value["ego_motion"]
        
        # Process second frame
        second_key = keys[idx+1]
        second_value = self.av2_dataset[second_key]
        valid_mask_second = second_value["flow_is_valid"]
        dynamic_mask_second = second_value["flow_category"] != 0
        ground_mask_second = second_value["ground_mask"]
        valid_mask_second = valid_mask_second & dynamic_mask_second & (~ ground_mask_second)
        point_cloud_second = second_value["point_cloud_first"][valid_mask_second]
        #apply ego motion
        point_cloud_second = torch.matmul(point_cloud_second - ego_motion[:3, 3], ego_motion[:3, :3].T)

        flow = first_value["flow"]
        flow = torch.matmul(flow - ego_motion[:3, 3], ego_motion[:3, :3].T)
        motion_mask = torch.linalg.norm(flow, dim=1) > 0.05

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

        #concat timestamp, from 0 to 1
        timestamp1 = torch.ones((point_cloud_first.shape[0], 1))
        timestamp1 *= idx/self.sequence_length
        timestamp2 = torch.ones((point_cloud_second.shape[0], 1))
        timestamp2 *= (idx+1)/self.sequence_length
        point_cloud_first = torch.cat((point_cloud_first, timestamp1), dim=1)
        point_cloud_second = torch.cat((point_cloud_second, timestamp2), dim=1)
        # Prepare sample
        sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_second": point_cloud_second,
            "flow": flow,
            'dynamic_instance_mask': dynamic_instance_mask,
            'background_static_mask': background_static_mask,
            'foreground_static_mask': foreground_static_mask,
            'foreground_dynamic_mask': foreground_dynamic_mask,
        }

        return sample
    