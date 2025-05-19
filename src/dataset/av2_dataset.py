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

class AV2PerSceneDataset(nn.Module):
    """
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
        super(AV2PerSceneDataset, self).__init__()
        self.point_cloud_first = None
        
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
        if self.point_cloud_first is None:
            # Load AV2 scene data
            av2_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/train/8de6abb6-6589-3da7-8e21-6ecc80004a36.h5"
            av2_test_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5"
            av2_dataset = read_av2_scene(av2_scene_path)
            
            # Process first frame
            keys = list(av2_dataset.keys())
            first_key = keys[0]
            first_value = av2_dataset[first_key]
            valid_mask = first_value["flow_is_valid"]
            ground_mask = first_value["ground_mask"]
            dynamic_mask = first_value["flow_category"] != 0
            valid_mask = valid_mask & dynamic_mask
            self.point_cloud_first = first_value["point_cloud_first"][valid_mask]

            # Process second frame
            second_key = keys[1]
            second_value = av2_dataset[second_key]
            valid_mask_second = second_value["flow_is_valid"]
            dynamic_mask_second = second_value["flow_category"] != 0
            valid_mask_second = valid_mask_second & dynamic_mask_second
            self.point_cloud_second = second_value["point_cloud_first"][valid_mask_second]
            flow = first_value["flow"]
            self.flow = flow[valid_mask]
            self.dynamic_instance_mask = first_value["flow_category"][valid_mask]
        # Prepare sample
        sample = {
            "point_cloud_first": self.point_cloud_first,
            "point_cloud_second": self.point_cloud_second,
            "flow": self.flow,
            'dynamic_instance_mask': self.dynamic_instance_mask,
        }

        return sample