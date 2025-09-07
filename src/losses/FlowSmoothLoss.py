"""
Flow Smoothness Loss implementation for scene flow prediction.

This module implements a parametric flow smoothness loss that encourages spatially
coherent flow predictions within segmented regions. It uses a quadratic flow
approximation approach to ensure smooth transitions in the predicted flow field.
"""

import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .ChamferDistanceLoss import ChamferDistanceLoss

class ScaleGradient(torch.autograd.Function):
    """
    Custom autograd function for gradient scaling during backpropagation.
    
    This function allows for controlled gradient flow by scaling gradients
    during the backward pass while leaving the forward pass unchanged.
    """
    
    @staticmethod
    def forward(ctx, input, scale):
        """
        Forward pass: store scale and return input unchanged.
        
        Args:
            ctx: Context object to store information for backward pass
            input (torch.Tensor): Input tensor
            scale (float): Scale factor for gradients
            
        Returns:
            torch.Tensor: Input tensor unchanged
        """
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: scale the gradients.
        
        Args:
            ctx: Context object containing the scale factor
            grad_output (torch.Tensor): Incoming gradient
            
        Returns:
            tuple: (Scaled gradient, None for scale parameter)
        """
        return grad_output * ctx.scale, None

def normalize_global(x):
    """
    Normalize tensor globally using standard deviation.
    
    Args:
        x (torch.Tensor): Input tensor to normalize
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x

def normalize_useing_other(x, points):
    """
    Normalize tensor using standard deviation from another tensor.
    
    Args:
        x (torch.Tensor): Tensor to normalize
        points (torch.Tensor): Tensor to compute normalization statistics from
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = points.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    return x/std

class FlowSmoothLoss():
    """
    Flow smoothness loss using parametric quadratic flow approximation.
    
    This loss encourages smooth flow fields within segmented regions by fitting
    a quadratic model to the flow in each segment and penalizing deviations
    from this model.
    
    The loss is computed as:
    Lf(M|F) = sum_k || Fk - F̂k ||^2_F
    where F̂k = Ek θ̂k and θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
    
    Attributes:
        device (torch.device): Device to perform computations on
        criterion (nn.Module): MSE loss function for reconstruction error
        chamferDistanceLoss (ChamferDistanceLoss): For additional distance metrics
    """

    def __init__(self, device,flow_smooth_loss_config):
        """
        Initialize the Flow Smoothness Loss.
        
        Args:
            device (torch.device): Device to perform computations on
        """
        self.device=device
        each_mask_item_gradient = flow_smooth_loss_config.each_mask_item.relative_gradient
        sum_mask_item_gradient = flow_smooth_loss_config.sum_mask_item.relative_gradient
        self.each_mask_item_gradient = each_mask_item_gradient/(each_mask_item_gradient+sum_mask_item_gradient)
        self.sum_mask_item_gradient = sum_mask_item_gradient/(each_mask_item_gradient+sum_mask_item_gradient)
        self.each_mask_item_loss = flow_smooth_loss_config.each_mask_item.criterion
        self.sum_mask_item_loss = flow_smooth_loss_config.sum_mask_item.criterion
        if self.each_mask_item_loss in ["L1", "l1"]:
            self.each_mask_criterion = nn.L1Loss(reduction="mean").to(self.device)
        elif self.each_mask_item_loss in ["L2", "l2"]:
            self.each_mask_criterion = nn.MSELoss(reduction="mean").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.each_mask_item_loss}")
        if self.sum_mask_item_loss in ["L1", "l1"]:
            self.sum_mask_criterion = nn.L1Loss(reduction="mean").to(self.device)
        elif self.sum_mask_item_loss in ["L2", "l2"]:
            self.sum_mask_criterion = nn.MSELoss(reduction="mean").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.sum_mask_item_loss}")
        self.chamferDistanceLoss = ChamferDistanceLoss()
        pass

    def __call__(self, sample, mask, flow):
        """
        Compute the flow smoothness loss.
        
        Args:
            sample (dict): Input data containing:
                - point_cloud_first (list[torch.Tensor]): List of first frame point clouds [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]
            
        Returns:
            torch.Tensor: Computed smoothness loss averaged across the batch
        """
        return self.loss(sample, mask, flow)
        
    def loss(self, sample, mask, flow):
        """
        Core loss computation function.
        
        For each segment in the mask, fits a quadratic model to the flow
        and penalizes deviations from this model.
        
        Args:
            sample (dict): Input data containing:
                - point_cloud_first (list[torch.Tensor]): List of first frame point clouds [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]
            
        Returns:
            torch.Tensor: Average reconstruction loss across all batches
        """
        batch_size = len(sample["point_cloud_first"])
        point_position = [item.to(self.device) for item in sample["point_cloud_first"]]
        scene_flows = flow
        
        total_loss = 0.0
        for b in range(batch_size):
            # Get batch data
            point_position_b = point_position[b]  # (N, 3)
            scene_flow_b = scene_flows[b]  # (N, 3)
            mask_b = mask[b]  # (K, N)
            
            # Process mask
            mask_b = ScaleGradient.apply(mask_b, 1)
            mask_binary_b = F.softmax(mask_b, dim=0)  # (K, N)
            
            # Normalize flow
            scene_flow_b = normalize_useing_other(scene_flow_b, scene_flow_b)
            
            # Construct embedding
            coords = self.construct_embedding(point_position_b)  # (N, 5)
            
            # Initialize flow reconstruction
            flow_reconstruction = torch.zeros_like(scene_flow_b)  # (N, 3)
            
            # Per-slot reconstruction
            K = mask_b.shape[0]
            reconstruction_loss = 0
            
            for k in range(K):
                mk = mask_binary_b[k].unsqueeze(-1)  # (N, 1)
                
                Ek = coords * mk  # Apply mask to embedding
                Fk = scene_flow_b * mk  # Apply mask to flow
                # print(f"Ek {Ek.shape}, Fk {Fk.shape}")
                # Solve for parameters
                theta_k = torch.linalg.lstsq(Ek, Fk).solution
                
                # Reconstruct flow
                Fk_hat = Ek @ theta_k
                flow_reconstruction += Fk_hat  # (N, 3)

                reconstruction_loss = self.each_mask_criterion(Fk_hat,Fk)
                total_loss += reconstruction_loss*self.each_mask_item_gradient
            reconstruction_loss = self.sum_mask_criterion(scene_flow_b, flow_reconstruction)
            total_loss += reconstruction_loss*self.sum_mask_item_gradient
            # Compute reconstruction loss
            # with torch.no_grad():
            #     flow_reconstruction = flow_reconstruction.detach()
            # reconstruction_loss = torch.pow(torch.log((scene_flow_b+1e8)/(flow_reconstruction+1e8)), 2).mean()
        
        # Return average loss
        return total_loss / batch_size

    @torch.no_grad()
    def construct_embedding(self, point_position):
        """
        Construct point coordinate embedding [x, y, z, 1, 1].
        
        Creates an embedding that allows for quadratic flow approximation
        by augmenting the 3D coordinates with constant terms.
        
        Args:
            point_position (torch.Tensor): Point cloud positions [N, 3]
            
        Returns:
            torch.Tensor: Embedding vectors [N, 5]
        """
        x = point_position[..., 0].view(-1)
        y = point_position[..., 1].view(-1)
        z = point_position[..., 2].view(-1)
        # shape (N, 4)
        emb = torch.stack([
            x, y, z,
            # x*x*x,
            # y*y*y,
            # z*z*z, 
            # x*x*x*x*x,
            # y*y*y*y*y,
            # z*z*z*z*z,
            # x*y,
            # y*z,
            # x*z,
            torch.ones_like(x),
            ], dim=1)
        return emb