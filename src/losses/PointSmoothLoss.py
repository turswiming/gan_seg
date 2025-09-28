from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

from pointnet2.pointnet2 import *

"""
Point Smoothness Loss implementation for point cloud segmentation.

This module implements a comprehensive smoothness loss for point cloud segmentation,
combining k-nearest neighbors (KNN) and ball query approaches to encourage spatially
coherent segmentation masks. The loss penalizes rapid changes in segmentation
probabilities between neighboring points using two complementary neighborhood
definitions.
"""

"""
  smooth_loss_params:
    w_knn: 3.
    w_ball_q: 1.
    knn_loss_params:
      k: 8
      radius: 0.1
      loss_norm: 1
    ball_q_loss_params:
      k: 16
      radius: 0.2
      loss_norm: 1
"""
class KnnLoss(nn.Module):
    """
    K-nearest neighbors based smoothness loss component.
    
    This loss encourages consistent segmentation masks among k-nearest neighbors
    within a specified radius, using either L1/L2 distance or cross-entropy to
    measure differences in segmentation probabilities.
    
    Attributes:
        k (int): Number of nearest neighbors to consider
        radius (float): Maximum distance for neighbor consideration
        cross_entropy (bool): Whether to use cross-entropy instead of norm distance
        loss_norm (int): Norm to use for distance calculation (1 for L1, 2 for L2)
    """
    
    def __init__(self, k=8, radius=0.1, cross_entropy=False, loss_norm=1, **kwargs):
        """
        Initialize the KNN Loss component.
        
        Args:
            k (int): Number of nearest neighbors
            radius (float): Maximum neighbor distance
            cross_entropy (bool): Use cross-entropy loss instead of norm
            loss_norm (int): Which norm to use (1 or 2)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        Compute the KNN-based smoothness loss.
        
        Args:
            pc (torch.Tensor): Point cloud coordinates [B, N, 3]
            mask (torch.Tensor): Segmentation mask [B, N, K]
            
        Returns:
            torch.Tensor: Computed KNN smoothness loss
        """
        mask = mask.permute(0, 2, 1).contiguous()
        dist, idx = knn(self.k, pc, pc)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_mask = grouping_operation(mask, idx.detach())
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            loss = (mask.unsqueeze(3) - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class BallQLoss(nn.Module):
    """
    Ball query based smoothness loss component.
    
    This loss encourages consistent segmentation masks among points within
    a fixed radius ball, complementing the KNN-based approach by providing
    a different neighborhood definition.
    
    Attributes:
        k (int): Maximum number of points in ball
        radius (float): Ball radius for neighbor search
        cross_entropy (bool): Whether to use cross-entropy instead of norm distance
        loss_norm (int): Norm to use for distance calculation (1 for L1, 2 for L2)
    """
    
    def __init__(self, k=16, radius=0.2, cross_entropy=False, loss_norm=1, **kwargs):
        """
        Initialize the Ball Query Loss component.
        
        Args:
            k (int): Maximum points in ball
            radius (float): Ball radius
            cross_entropy (bool): Use cross-entropy loss instead of norm
            loss_norm (int): Which norm to use (1 or 2)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        Compute the ball query based smoothness loss.
        
        Args:
            pc (torch.Tensor): Point cloud coordinates [B, N, 3]
            mask (torch.Tensor): Segmentation mask [B, N, K]
            
        Returns:
            torch.Tensor: Computed ball query smoothness loss
        """
        mask = mask.permute(0, 2, 1).contiguous()
        idx = ball_query(self.radius, self.k, pc, pc)
        nn_mask = grouping_operation(mask, idx.detach())
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            loss = (mask.unsqueeze(3) - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class PointSmoothLoss(nn.Module):
    """
    Combined point smoothness loss for segmentation.
    
    This loss combines KNN and ball query based approaches to encourage
    spatially coherent segmentation masks. The combination provides robust
    smoothness constraints using different neighborhood definitions.
    
    Attributes:
        w_knn (float): Weight for KNN loss component
        w_ball_q (float): Weight for ball query loss component
        knn_loss (KnnLoss): KNN-based smoothness loss
        ball_q_loss (BallQLoss): Ball query based smoothness loss
    """
    
    def __init__(self, w_knn=3, w_ball_q=1):
        """
        Initialize the combined Point Smoothness Loss.
        
        Args:
            w_knn (float): Weight for KNN loss
            w_ball_q (float): Weight for ball query loss
        """
        super().__init__()
        self.knn_loss = KnnLoss()
        self.ball_q_loss = BallQLoss()
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc, mask):
        """
        Compute the combined smoothness loss.
        
        Args:
            pc (list[torch.Tensor]): List of point cloud coordinates, each of shape [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            
        Returns:
            torch.Tensor: Combined smoothness loss averaged across the batch
        """
        # Reshape mask from (B, K, N) to (B, N, K) for compatibility with loss functions
        batch_size = len(pc)
        mask_reshaped = [item.permute(1, 0).unsqueeze(0) for item in mask]
        pc = [item.unsqueeze(0) for item in pc]
        loss = torch.zeros(batch_size).to(pc[0].device)
        for i in range(batch_size):
            loss[i] = (self.w_knn * self.knn_loss(pc[i], mask_reshaped[i])) + (self.w_ball_q * self.ball_q_loss(pc[i], mask_reshaped[i]))
        loss = loss.mean()
        return loss