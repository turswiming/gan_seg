"""
Scene Flow Prediction Models implementation.

This module implements various models for predicting scene flow in point clouds,
including a standard neural network approach and a parameter optimization approach.
"""

import torch
from torch import nn
from torch.nn import functional as F

    
class Neural_Prior(torch.nn.Module):
    """
    Neural prior model for scene flow prediction.
    
    This model uses a multi-layer perceptron as a prior for scene flow prediction,
    mapping point coordinates to flow vectors through a series of learned transformations.
    
    Attributes:
        layer_size (int): Number of hidden layers
        nn_layers (nn.ModuleList): List of neural network layers
    """
    
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        """
        Initialize the neural prior model.
        
        Args:
            dim_x (int): Input and output dimension (default: 3 for xyz coordinates)
            filter_size (int): Width of hidden layers
            act_fn (str): Activation function ('relu' or 'sigmoid')
            layer_size (int): Number of hidden layers
        """
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == "leakyrelu":
                self.nn_layers.append(torch.nn.LeakyReLU())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == "leakyrelu":
                    self.nn_layers.append(torch.nn.LeakyReLU())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input point cloud coordinates [N, 3]
            
        Returns:
            torch.Tensor: Predicted flow vectors [N, 3]
        """
        for layer in self.nn_layers:
            x = layer(x)
        return x
    
class OptimizedFLowPredictor(torch.nn.Module):
    """
    Parameter-based flow predictor using direct optimization.
    
    This model directly optimizes a set of parameters representing flow vectors
    for each point, without using a neural network architecture. This approach
    can be effective for scenarios where direct optimization is preferred over
    learned features.
    
    Attributes:
        pointSize (int): Number of points to predict flow for
        dim (int): Dimension of flow vectors (typically 3 for xyz)
        init_noise (nn.Parameter): Learnable flow parameters
    """
    
    def __init__(self, dim=3, pointSize=128):
        """
        Initialize the optimized flow predictor.
        
        Args:
            dim (int): Dimension of flow vectors (default: 3)
            pointSize (int): Number of points to predict flow for
        """
        super().__init__()
        self.pointSize = pointSize
        self.dim = dim
        init_noise = torch.randn((pointSize, dim))*0.1
        self.init_noise = torch.nn.Parameter(init_noise, requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass returning the optimized flow vectors.
        
        Args:
            x (torch.Tensor): Input point cloud coordinates [N, 3]
                Note: The input is not used in this model as flow vectors
                are directly optimized parameters.
            
        Returns:
            torch.Tensor: Predicted flow vectors [N, 3]
        """
        return self.init_noise