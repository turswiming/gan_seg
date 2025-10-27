"""
KNN Distance Loss implementation for point cloud comparison using PyTorch3D.

This module implements a K-Nearest Neighbors distance loss that computes distances
between point clouds by finding k nearest neighbors. It provides an alternative to
Chamfer distance with more flexible neighborhood control and supports various
reduction methods and distance thresholding.
"""

import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points
from typing import Optional


class KNNDistanceLoss(nn.Module):
    """
    KNN Distance Loss for point cloud comparison using PyTorch3D.
    
    This implementation uses PyTorch3D's efficient KNN operations to compute
    distances between point clouds. It supports bidirectional distance calculation,
    distance thresholding, and various reduction methods.
    
    Attributes:
        k (int): Number of nearest neighbors to consider
        reduction (str): Method to reduce the loss ('mean', 'sum', or 'none')
        distance_threshold (float): Maximum distance threshold for valid matches
    """

    def __init__(self, k=1, reduction='mean', distance_max_threshold=None, distance_min_threshold=None):
        """
        Initialize the KNN Distance Loss.

        Args:
            k (int): Number of nearest neighbors to consider (default: 1)
            reduction (str): Reduction method for the loss:
                - 'mean': Average across all points and batches
                - 'sum': Sum across all points and batches
                - 'none': Return per-point distances
            distance_threshold (float, optional): Maximum distance for valid matches.
                If None, no thresholding is applied.
        """
        self.k = k
        self.reduction = reduction
        self.distance_max_threshold = distance_max_threshold
        self.distance_min_threshold = distance_min_threshold

    def knn_distance_single_direction(self, x, y, lengths1=None, lengths2=None):
        """
        Compute KNN distance in a single direction (x -> y).
        
        Args:
            x (torch.Tensor): Source point cloud [batch_size, num_points_x, 3]
            y (torch.Tensor): Target point cloud [batch_size, num_points_y, 3]
            lengths1 (torch.Tensor, optional): Valid point counts for x
            lengths2 (torch.Tensor, optional): Valid point counts for y
            
        Returns:
            torch.Tensor: KNN distances from x to y
        """
        # Ensure k doesn't exceed target point cloud size
        effective_k = min(self.k, y.shape[1])
        
        # Find k nearest neighbors using PyTorch3D
        knn_result = knn_points(
            x, y, 
            lengths1=lengths1, 
            lengths2=lengths2, 
            K=effective_k,
            return_nn=False
        )
        
        # Get squared distances to k nearest neighbors
        knn_distances = knn_result.dists  # Shape: [batch_size, num_points_x, k]
        
        # Apply distance thresholding if specified
        if self.distance_max_threshold is not None:
            # Set distances above threshold to 0
            threshold_sq = self.distance_max_threshold ** 2
            knn_distances = torch.where(
                knn_distances <= threshold_sq,
                knn_distances,
                torch.zeros_like(knn_distances)
            )
        
        if self.distance_min_threshold is not None:
            # Set distances below threshold to 0
            threshold_sq = self.distance_min_threshold ** 2
            knn_distances = torch.where(
                knn_distances >= threshold_sq,
                knn_distances,
                torch.zeros_like(knn_distances)
            )
        # Reduce distances within each neighborhood (across k neighbors)
        if self.k > 1:
            # Take mean of k nearest neighbor distances
            point_distances = knn_distances.mean(dim=-1)  # [batch_size, num_points_x]
        else:
            # Single nearest neighbor
            point_distances = knn_distances.squeeze(-1)  # [batch_size, num_points_x]
        
        return point_distances

    def knn_distance_bidirectional(self, x, y, lengths1=None, lengths2=None):
        """
        Compute bidirectional KNN distance between two point clouds.
        
        Args:
            x (torch.Tensor): First point cloud [batch_size, num_points_x, 3]
            y (torch.Tensor): Second point cloud [batch_size, num_points_y, 3]
            lengths1 (torch.Tensor, optional): Valid point counts for x
            lengths2 (torch.Tensor, optional): Valid point counts for y
            
        Returns:
            torch.Tensor: Combined bidirectional KNN distance
        """
        # Forward direction: x -> y
        dist_x_to_y = self.knn_distance_single_direction(x, y, lengths1, lengths2)
        
        # Backward direction: y -> x
        dist_y_to_x = self.knn_distance_single_direction(y, x, lengths2, lengths1)
        
        # Apply reduction across points
        if self.reduction == 'mean':
            forward_loss = dist_x_to_y.mean()
            backward_loss = dist_y_to_x.mean()
        elif self.reduction == 'sum':
            forward_loss = dist_x_to_y.sum()
            backward_loss = dist_y_to_x.sum()
        else:  # 'none'
            return dist_x_to_y, dist_y_to_x
        
        # Combine forward and backward losses
        return forward_loss + backward_loss

    def knn_distance_unidirectional(self, x, y, lengths1=None, lengths2=None):
        """
        Compute unidirectional KNN distance (x -> y only).
        
        Args:
            x (torch.Tensor): Source point cloud [batch_size, num_points_x, 3]
            y (torch.Tensor): Target point cloud [batch_size, num_points_y, 3]
            lengths1 (torch.Tensor, optional): Valid point counts for x
            lengths2 (torch.Tensor, optional): Valid point counts for y
            
        Returns:
            torch.Tensor: Unidirectional KNN distance
        """
        # Forward direction only: x -> y
        dist_x_to_y = self.knn_distance_single_direction(x, y, lengths1, lengths2)
        
        # Apply reduction across points
        if self.reduction == 'mean':
            return dist_x_to_y.mean()
        elif self.reduction == 'sum':
            return dist_x_to_y.sum()
        else:  # 'none'
            return dist_x_to_y

    def __call__(self, x, y, bidirectional=True, lengths1=None, lengths2=None):
        """
        Compute KNN distance between two point clouds.

        Args:
            x (torch.Tensor): First point cloud [batch_size, num_points_x, 3] or [num_points_x, 3]
            y (torch.Tensor): Second point cloud [batch_size, num_points_y, 3] or [num_points_y, 3]
            bidirectional (bool): If True, compute distance in both directions
            lengths1 (torch.Tensor, optional): Valid point counts for x
            lengths2 (torch.Tensor, optional): Valid point counts for y

        Returns:
            torch.Tensor: Computed KNN distance
        """
        # Ensure input tensors are 3D
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        
        # Validate input shapes
        assert x.dim() == 3, f"Expected 3D tensor for x, got {x.dim()}D"
        assert y.dim() == 3, f"Expected 3D tensor for y, got {y.dim()}D"
        assert x.size(0) == y.size(0), "Batch sizes must match"
        assert x.size(2) == 3, f"Expected 3D points, got {x.size(2)}D"
        assert y.size(2) == 3, f"Expected 3D points, got {y.size(2)}D"
        
        # Create default lengths if not provided
        if lengths1 is None:
            lengths1 = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long, device=x.device)
        if lengths2 is None:
            lengths2 = torch.tensor([y.shape[1]] * y.shape[0], dtype=torch.long, device=y.device)
        
        if bidirectional:
            return self.knn_distance_bidirectional(x, y, lengths1, lengths2)
        else:
            return self.knn_distance_unidirectional(x, y, lengths1, lengths2)


class TruncatedKNNDistanceLoss(KNNDistanceLoss):
    """
    Truncated KNN Distance Loss with distance thresholding.
    
    This is a specialized version of KNN Distance Loss that applies distance
    thresholding, similar to the TruncatedChamferLoss concept.
    """
    
    def __init__(self, k=1, distance_max_threshold=2.0, distance_min_threshold=0.02, reduction='mean'):
        """
        Initialize the Truncated KNN Distance Loss.
        
        Args:
            k (int): Number of nearest neighbors to consider
            distance_threshold (float): Distance threshold for truncation
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__(k=k, reduction=reduction, distance_max_threshold=distance_max_threshold, distance_min_threshold=distance_min_threshold)
    
    def __call__(self, warped_pc, target_pc, forward_only=True):
        """
        Compute truncated KNN distance loss.
        
        Args:
            warped_pc (torch.Tensor): Warped point cloud
            target_pc (torch.Tensor): Target point cloud
            forward_only (bool): If True, compute only forward direction
            
        Returns:
            torch.Tensor: Truncated KNN distance loss
        """
        bidirectional = not forward_only
        return super().__call__(warped_pc, target_pc, bidirectional=bidirectional)
