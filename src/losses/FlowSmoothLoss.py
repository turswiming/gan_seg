"""
Flow Smoothness Loss implementation for scene flow prediction.

This module implements a parametric flow smoothness loss that encourages spatially
coherent flow predictions within segmented regions. It uses a quadratic flow
approximation approach to ensure smooth transitions in the predicted flow field.

Key Optimizations:
- Parallel matrix operations for k-dimensional slots processing
- Batch subdivision strategy: automatically selects optimal batch sizes (5, 4, or 3)
  based on K to maximize GPU utilization and memory efficiency
- Vectorized loss computation to eliminate sequential loops
"""

import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
            self.each_mask_criterion = nn.L1Loss(reduction="sum").to(self.device)
        elif self.each_mask_item_loss in ["L2", "l2"]:
            self.each_mask_criterion = nn.MSELoss(reduction="sum").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.each_mask_item_loss}")
        if self.sum_mask_item_loss in ["L1", "l1"]:
            self.sum_mask_criterion = nn.L1Loss(reduction="sum").to(self.device)
        elif self.sum_mask_item_loss in ["L2", "l2"]:
            self.sum_mask_criterion = nn.MSELoss(reduction="sum").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.sum_mask_item_loss}")
        pass

    def __call__(self, point_position, mask, flow):
        """
        Compute the flow smoothness loss.
        
        Args:
            point_position (list[torch.Tensor]): List of point positions, each of shape [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]
            
        Returns:
            torch.Tensor: Computed smoothness loss averaged across the batch
        """
        return self.loss(point_position, mask, flow)
        
    def get_optimal_batch_size(self, K):
        """根据K选择最优的batch size (5, 4, 3的倍数)"""
        # 优先选择能整除K的最大batch size
        divisors = [5, 4, 3]
        for batch_size in divisors:
            if K % batch_size == 0:
                return batch_size
    
        return K  # 对于很小的K值，直接全部处理
    def loss(self, point_position, mask, flow):
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
        batch_size = len(mask)
        point_position = [item.to(self.device) for item in point_position]
        scene_flows = flow
        
        total_loss = 0.0
        for b in range(batch_size):
            one_batch_loss = 0.0
            N = point_position[b].shape[0]
            # Get batch data
            point_position_b = point_position[b]  # (N, 3)
            scene_flow_b = scene_flows[b]  # (N, 3)
            mask_b = mask[b]  # (K, N)
            
            # Process mask
            mask_b = ScaleGradient.apply(mask_b.clone(), 1)
            mask_binary_b = F.softmax(mask_b, dim=0)  # (K, N)
            mask_binary_b = mask_binary_b / pow(mask_binary_b.std(),0.5)
            scene_flow_b = scene_flow_b / pow(scene_flow_b.std(),1.5)
            # Normalize flow
            scene_flow_b = normalize_useing_other(scene_flow_b, point_position_b)
            scene_flow_b = ScaleGradient.apply(scene_flow_b.clone(),0.001)
            # Construct embedding
            coords = self.construct_embedding(point_position_b)  # (N, 5)
            
            # Initialize flow reconstruction
            flow_reconstruction = torch.zeros_like(scene_flow_b)  # (N, 3)
            
            # Per-slot reconstruction using parallel matrix operations
            K = mask_b.shape[0]
            reconstruction_loss = 0
            
            # Batch processing with K subdivision for acceleration
            # 根据K的大小选择最佳的batch_size进行分批计算

            
            batch_size_k = self.get_optimal_batch_size(K)
            num_batches = (K + batch_size_k - 1) // batch_size_k  # 向上取整
            
            # 分批处理K个slots
            for batch_idx in range(num_batches):
                start_k = batch_idx * batch_size_k
                end_k = min(start_k + batch_size_k, K)
                current_batch_size = end_k - start_k
                
                # 获取当前批次的mask
                mask_batch = mask_binary_b[start_k:end_k]  # (current_batch_size, N)
                mask_expanded = mask_batch.unsqueeze(-1)  # (current_batch_size, N, 1)
                
                # 扩展coords和scene_flow到当前批次大小
                coords_expanded = coords.unsqueeze(0).expand(current_batch_size, -1, -1)  # (current_batch_size, N, 5)
                scene_flow_expanded = scene_flow_b.unsqueeze(0).expand(current_batch_size, -1, -1)  # (current_batch_size, N, 3)
                
                # 批量应用masks到embeddings和flows
                Ek_batch = coords_expanded * mask_expanded  # (current_batch_size, N, 5)
                Fk_batch = scene_flow_expanded * mask_expanded  # (current_batch_size, N, 3)
                #add a small noise to the Fk_batch
                Fk_batch = Fk_batch + torch.randn_like(Fk_batch) * 1e-6
                # 批量线性最小二乘求解
                theta_batch = torch.linalg.lstsq(Ek_batch, Fk_batch, driver="gels").solution  # (current_batch_size, 5, 3)
                
                # 检查NaN值
                valid_mask = ~torch.isnan(theta_batch).any(dim=[1, 2])  # (current_batch_size,)
                
                if not valid_mask.any():
                    continue
                    # 批量重建flows
                Fk_hat_batch = torch.bmm(Ek_batch, theta_batch)  # (current_batch_size, N, 3)
                
                # 只对有效的slots进行累加
                valid_Fk_hat = Fk_hat_batch[valid_mask]  # (valid_count, N, 3)
                flow_reconstruction += valid_Fk_hat.sum(dim=0)  # (N, 3)
                
                # 计算有效slots的重建损失
                valid_Fk = Fk_batch[valid_mask]  # (valid_count, N, 3)
                
                # 向量化损失计算
                if self.each_mask_item_loss in ["L1", "l1"]:
                    batch_reconstruction_loss = torch.sum(torch.abs(valid_Fk_hat - valid_Fk))
                elif self.each_mask_item_loss in ["L2", "l2"]:
                    batch_reconstruction_loss = torch.sum((valid_Fk_hat - valid_Fk) ** 2)
                
                one_batch_loss += batch_reconstruction_loss * self.each_mask_item_gradient / N
            one_batch_loss = one_batch_loss * K
            reconstruction_loss = self.sum_mask_criterion(scene_flow_b, flow_reconstruction)
            one_batch_loss += reconstruction_loss*self.sum_mask_item_gradient /N
            total_loss += one_batch_loss
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