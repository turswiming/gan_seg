"""
Memory-efficient Chamfer Distance Loss implementation for point cloud comparison.

This module implements a memory-efficient version of the Chamfer Distance Loss,
which processes point clouds in chunks to reduce memory usage while maintaining
accuracy. The Chamfer distance measures the bidirectional distance between two
point clouds by finding the nearest neighbor distances in both directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferDistanceLoss:
    """
    Memory-efficient Chamfer Distance Loss for point cloud comparison.
    
    This implementation processes point clouds in chunks to reduce memory usage,
    making it suitable for large point clouds. It supports both unidirectional
    and bidirectional distance calculations with various reduction methods.
    
    Attributes:
        reduction (str): Method to reduce the loss ('mean', 'sum', or 'none')
    """

    def __init__(self, reduction='mean'):
        """
        Initialize the ChamferDistanceLoss.

        Args:
            reduction (str): Reduction method for the loss:
                - 'mean': Average across all points and batches
                - 'sum': Sum across all points and batches
                - 'none': Return per-point distances
        """
        self.reduction = reduction

    def chamfer_distance_memory_efficient(self, x, y, bidirectional=True, reduction='mean', chunk_size=1024):
        """
        Compute the Chamfer distance between two point clouds using chunked processing.
        
        This method splits the computation into chunks to reduce memory usage,
        making it feasible to process large point clouds on limited memory devices.
        
        Args:
            x (torch.Tensor): First point cloud [batch_size, num_points_x, dim]
            y (torch.Tensor): Second point cloud [batch_size, num_points_y, dim]
            bidirectional (bool): If True, compute distance in both directions (x->y and y->x)
            reduction (str): Method to reduce the loss ('mean', 'sum', 'none')
            chunk_size (int): Number of points to process at once (memory vs. speed tradeoff)
            
        Returns:
            torch.Tensor: Computed Chamfer distance with shape determined by reduction method:
                - 'mean': scalar
                - 'sum': scalar
                - 'none': [batch_size, num_points]
                
        Note:
            Smaller chunk_size reduces memory usage but increases computation time.
            The bidirectional option doubles computation time but provides more
            accurate distance measurements.
        """
        assert x.dim() == 3, "Expected 3D tensor for x"
        assert y.dim() == 3, "Expected 3D tensor for y"
        assert x.size(0) == y.size(0), "Batch sizes must match"
        assert x.size(2) == y.size(2), "Point dimensions must match"
        
        batch_size = x.size(0)
        num_points_x = x.size(1)
        num_points_y = y.size(1)
        
        # Forward direction (x -> y)
        mins_x = torch.zeros(batch_size, num_points_x, device=x.device)
        for b in range(batch_size):
            for i in range(0, num_points_x, chunk_size):
                end_i = min(i + chunk_size, num_points_x)
                x_chunk = x[b, i:end_i].unsqueeze(0)  # (1, chunk_size, dim)
                
                min_dists = float('inf') * torch.ones(end_i - i, device=x.device)
                for j in range(0, num_points_y, chunk_size):
                    end_j = min(j + chunk_size, num_points_y)
                    y_chunk = y[b, j:end_j].unsqueeze(0)  # (1, chunk_size, dim)
                    
                    # Compute pairwise distances for this chunk
                    dist_chunk = torch.cdist(x_chunk, y_chunk, p=2.0)**2  # (1, chunk_size_x, chunk_size_y)
                    
                    # Update minimum distances
                    min_dists_chunk, _ = torch.min(dist_chunk.squeeze(0), dim=1)
                    min_dists = torch.minimum(min_dists, min_dists_chunk)
                
                mins_x[b, i:end_i] = min_dists
        
        # Calculate forward Chamfer distance (x -> y)
        if reduction == 'mean':
            forward_chamfer = torch.mean(mins_x, dim=1)
        elif reduction == 'sum':
            forward_chamfer = torch.sum(mins_x, dim=1)
        else:  # 'none'
            forward_chamfer = mins_x
        
        if bidirectional:
            # Backward direction (y -> x)
            mins_y = torch.zeros(batch_size, num_points_y, device=y.device)
            for b in range(batch_size):
                for j in range(0, num_points_y, chunk_size):
                    end_j = min(j + chunk_size, num_points_y)
                    y_chunk = y[b, j:end_j].unsqueeze(0)  # (1, chunk_size, dim)
                    
                    min_dists = float('inf') * torch.ones(end_j - j, device=y.device)
                    for i in range(0, num_points_x, chunk_size):
                        end_i = min(i + chunk_size, num_points_x)
                        x_chunk = x[b, i:end_i].unsqueeze(0)  # (1, chunk_size, dim)
                        
                        # Compute pairwise distances for this chunk
                        dist_chunk = torch.cdist(y_chunk, x_chunk, p=2.0)**2  # (1, chunk_size_y, chunk_size_x)
                        
                        # Update minimum distances
                        min_dists_chunk, _ = torch.min(dist_chunk.squeeze(0), dim=1)
                        min_dists = torch.minimum(min_dists, min_dists_chunk)
                    
                    mins_y[b, j:end_j] = min_dists
            
            # Calculate backward Chamfer distance (y -> x)
            if reduction == 'mean':
                backward_chamfer = torch.mean(mins_y, dim=1)
            elif reduction == 'sum':
                backward_chamfer = torch.sum(mins_y, dim=1)
            else:  # 'none'
                backward_chamfer = mins_y
            
            # Combine forward and backward Chamfer distances
            chamfer_dist = forward_chamfer + backward_chamfer
        else:
            chamfer_dist = forward_chamfer
        
        # Final reduction across batch
        if reduction == 'mean':
            return chamfer_dist.mean()
        elif reduction == 'sum':
            return chamfer_dist.sum()
        else:  # 'none'
            return chamfer_dist

    def __call__(self, x, y):
        """
        Compute the Chamfer distance between two point clouds.

        A convenience wrapper around chamfer_distance_memory_efficient that uses
        the instance's reduction setting.

        Args:
            x (torch.Tensor): First point cloud [batch_size, num_points, dim]
            y (torch.Tensor): Second point cloud [batch_size, num_points, dim]

        Returns:
            torch.Tensor: Computed Chamfer distance
        """
        return self.chamfer_distance_memory_efficient(x, y, bidirectional=True, reduction=self.reduction, chunk_size=1024)