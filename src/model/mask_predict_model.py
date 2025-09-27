"""
Mask Prediction Model implementation for point cloud segmentation.

This module implements a learnable mask predictor that assigns points to different
slots (segments) in a point cloud. It uses a parameter-based approach where the
mask weights are directly optimized during training.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .eulerflow_raw_mlp import EulerFlowMLP, ActivationFn, QueryDirection

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
    """
    Neural network based mask predictor for point cloud segmentation.
    
    This model uses a multi-layer perceptron to learn point cloud segmentation
    by mapping 3D coordinates to slot assignment probabilities. The network
    architecture is configurable in terms of depth, width, and activation functions.
    
    Attributes:
        layer_size (int): Number of hidden layers
        nn_layers (nn.ModuleList): List of neural network layers
    """

    def __init__(self, input_dim=3, slot_num=10, filter_size=128, act_fn='sigmoid', layer_size=8, dropout=0.2):
        """
        Initialize the neural mask predictor.
        
        Args:
            dim_x (int): Input dimension (default: 3 for xyz coordinates)
            slot_num (int): Number of segmentation slots
            filter_size (int): Width of hidden layers
            act_fn (str): Activation function ('relu' or 'sigmoid')
            layer_size (int): Number of hidden layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == "leakyrelu":
                self.nn_layers.append(torch.nn.LeakyReLU())
                #add normalization
            # self.nn_layers.append(torch.nn.BatchNorm1d(filter_size))
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == "leakyrelu":
                    self.nn_layers.append(torch.nn.LeakyReLU())
            self.nn_layers.append(torch.nn.Linear(filter_size, slot_num))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, slot_num)))

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input point cloud coordinates [N, 3]
            
        Returns:
            torch.Tensor: Predicted slot assignment probabilities [slot_num, N]
        """
        layer_num = 0
        for layer in self.nn_layers:
            layer_num += 1
            # print(f"layer_num: {layer_num}, x_std{x.std()}, x_mean: {x.mean()}")
            x = layer(x)
        # x = F.softmax(x, dim=1)
        return x.permute(1, 0)
    

class EulerMaskMLP(EulerFlowMLP):
    def __init__(
        self, 
        slot_num=10, 
        filter_size=128, 
        act_fn: ActivationFn = ActivationFn.LEAKYRELU,
        layer_size=8, 
        use_normalization: bool = False,
        normalization_type: str = "group_norm",  # 默认使用group_norm节省内存
    ):
        super().__init__(
            output_dim=slot_num,
            latent_dim=filter_size,
            act_fn=act_fn,
            num_layers=layer_size,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
        )
        self.nn_layers = torch.nn.Sequential(self.nn_layers)

    def forward(self, pc, idx, total_entries):
        return super().forward(pc, idx, total_entries, QueryDirection.FORWARD)