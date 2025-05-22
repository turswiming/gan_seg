"""
Mask Prediction Model implementation for point cloud segmentation.

This module implements a learnable mask predictor that assigns points to different
slots (segments) in a point cloud. It uses a parameter-based approach where the
mask weights are directly optimized during training.
"""

import torch
from torch import nn
from torch.nn import functional as F

class OptimizedMaskPredictor(nn.Module):
    """
    Learnable mask predictor for point cloud segmentation.
    
    This model learns to segment point clouds by optimizing a set of parameters
    that represent soft assignments of points to different slots (segments).
    The assignments are normalized using softmax to ensure they sum to 1 across slots.
    
    Attributes:
        slot_num (int): Number of segmentation slots (masks)
        point_length (int): Number of points in the point cloud
        device (torch.device): Device to perform computations on
        tensor3d (nn.Parameter): Learnable parameters for mask prediction [K, N]
    """
    
    def __init__(self, slot_num=1, point_length=65536):
        """
        Initialize the mask predictor.
        
        Args:
            slot_num (int): Number of segmentation slots
            point_length (int): Number of points to process
        """
        super(OptimizedMaskPredictor, self).__init__()
        self.slot_num = slot_num
        self.point_length = point_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.randn((slot_num, point_length), device=self.device)
        softmaxed_tensor = F.softmax(tensor, dim=0)
        self.tensor2d = torch.nn.Parameter(softmaxed_tensor, requires_grad=True)
    
    def forward(self, inputs):
        """
        Predict segmentation masks for a batch of point clouds.
        
        Args:
            inputs (dict): Input dictionary containing:
                - point_cloud_first (torch.Tensor): First point cloud [B, N, 3]
                
        Returns:
            torch.Tensor: Predicted segmentation masks [B, K, N] where:
                - B is the batch size
                - K is the number of slots
                - N is the number of points
        """
        # Repeat the parameter tensor for each batch item
        return self.tensor2d 
    
class Neural_Mask_Prior(torch.nn.Module):
    def __init__(self, dim_x=3, slot_num=10, filter_size=128, act_fn='sigmoid', layer_size=8, dropout=0.2):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, slot_num))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, slot_num)))

    def forward(self, x):
        """ points -> features
            [N, 3] -> [slot_num, N]
        """
        layer_num = 0
        for layer in self.nn_layers:
            layer_num += 1
            print(f"layer_num: {layer_num}, x_std{x.std()}, x_mean: {x.mean()}")
            x = layer(x)
        x = F.softmax(x, dim=0)
        return x.permute(1, 0)
    