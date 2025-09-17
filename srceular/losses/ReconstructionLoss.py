"""
Reconstruction Loss implementation for point cloud reconstruction.

This module implements a reconstruction loss that measures how well a predicted
point cloud matches the ground truth, taking into account both point positions
and their flow vectors.
"""

import torch
from torch import nn
from torch.nn import functional as F
from losses.loss_chamfer import my_chamfer_fn

class ReconstructionLoss():
    """
    Reconstruction Loss for point cloud and flow prediction.
    
    This loss function evaluates the quality of reconstructed point clouds by
    comparing them with ground truth data, considering both point positions
    and their associated flow vectors.
    
    Attributes:
        device (torch.device): Device to perform computations on
    """
    
    def __init__(self, device):
        """
        Initialize the Reconstruction Loss module.
        
        Args:
            device (torch.device): Device to perform computations on
        """
        self.device = device
        self.chamferDistanceLoss = my_chamfer_fn
        pass

    def fit_motion_svd_batch(self, pc1, pc2, mask=None):
        """
        Fit rigid transformation between two point clouds using SVD.
        
        This method computes the optimal rigid transformation (rotation and translation)
        that aligns two point clouds, optionally weighted by a mask. It uses Singular
        Value Decomposition (SVD) to find the best rotation matrix.
        
        Args:
            pc1 (torch.Tensor): Source point cloud [B, N, 3]
            pc2 (torch.Tensor): Target point cloud [B, N, 3]
            mask (torch.Tensor, optional): Weights for each point [B, N]
            
        Returns:
            tuple: A tuple containing:
                - R (torch.Tensor): Batch of rotation matrices [B, 3, 3]
                - t (torch.Tensor): Batch of translation vectors [B, 3]
        """
        n_batch, n_point, _ = pc1.size()

        if mask is None:
            pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
            pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
        else:
            pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
            pc1_mean.unsqueeze_(1)
            pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
            pc2_mean.unsqueeze_(1)

        pc1_centered = pc1 - pc1_mean
        pc2_centered = pc2 - pc2_mean

        if mask is None:
            S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
        else:
            S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

        # If mask is not well-defined, S will be ill-posed.
        # We just return an identity matrix.
        valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
        R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
        t_base = torch.zeros((n_batch, 3), device=pc1.device)

        if valid_batches.any():
            S = S[valid_batches, ...]
            u, s, v = torch.svd(S, some=False, compute_uv=True)
            R = torch.bmm(v, u.transpose(1, 2))
            det = torch.det(R)

            # Correct reflection matrix to rotation matrix
            diag = torch.ones_like(S[..., 0], requires_grad=False)
            diag[:, 2] = det
            R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

            pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
            t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

            R_base[valid_batches] = R.to(R_base.dtype)
            t_base[valid_batches] = t.to(t_base.dtype)

        return R_base, t_base
    
    def soft_knn(self, query_points, reference_points, reference_values, k=5, temperature=0.1):
        """
        Performs a soft k-nearest neighbors query with distance-based weighting.
        
        Args:
            query_points: Tensor of shape (n_queries, d) - points to query
            reference_points: Tensor of shape (n_ref, d) - database points
            reference_values: Tensor of shape (n_ref, v) - values to interpolate
            k: Number of neighbors to consider
            temperature: Controls the softness of the weighting (lower = harder)
        
        Returns:
            Tensor of shape (n_queries, v) containing the soft kNN interpolated values
        """        
        # Calculate pairwise distances between query and reference points
        # Using efficient batch computation
        n_queries = query_points.shape[0]
        n_refs = reference_points.shape[0]
        
        # Expand dimensions for broadcasting
        query_expanded = query_points.unsqueeze(1)  # (n_queries, 1, d)
        ref_expanded = reference_points.unsqueeze(0)  # (1, n_refs, d)
        
        # Calculate squared Euclidean distances
        squared_distances = torch.sum((query_expanded - ref_expanded) ** 2, dim=2)  # (n_queries, n_refs)
        
        # Get k nearest neighbors
        distances, indices = torch.topk(squared_distances, k=k, dim=1, largest=False)
        
        # Distance-based weighting with temperature control
        weights = torch.exp(-distances / temperature)  # (n_queries, k)
        
        # Normalize weights to sum to 1
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # (n_queries, k)
        
        # Gather the values of the k-nearest neighbors
        neighbor_values = reference_values[indices]  # (n_queries, k, v)
        
        # Compute weighted average
        interpolated_values = torch.sum(weights.unsqueeze(-1) * neighbor_values, dim=1)  # (n_queries, v)
        
        return interpolated_values
    
    def __call__(self, inputs, pred_mask, pred_flow):
        """
        Compute the reconstruction loss.
        
        Args:
            inputs (dict): Input data containing:
                - point_cloud_first (list[torch.Tensor]): List of first frame point clouds [N, 3]
                - point_cloud_second (list[torch.Tensor]): List of second frame point clouds [N, 3]
            pred_mask (list[torch.Tensor]): List of predicted segmentation masks, each of shape [K, N]
            pred_flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]
            
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Computed reconstruction loss averaged across the batch
                - rec_point_cloud (list[torch.Tensor]): List of reconstructed point clouds
        """
        point_cloud_first = [item.to(self.device) for item in inputs["point_cloud_first"]]
        point_cloud_second = [item.to(self.device) for item in inputs["point_cloud_second"]]
        pred_mask = [item.to(self.device) for item in pred_mask]
        pred_flow = [item.to(self.device) for item in pred_flow]
        
        batch_size = len(point_cloud_first)
        loss_summ = 0
        rec_point_cloud = []
        # Process each batch item
        for batch_idx in range(batch_size):
            # Get data for current batch
            current_point_cloud_first = point_cloud_first[batch_idx]  # Keep batch dimension
            current_point_cloud_first = current_point_cloud_first.unsqueeze(0)
            current_point_cloud_second = point_cloud_second[batch_idx]
            current_point_cloud_second = current_point_cloud_second.unsqueeze(0)
            current_pred_mask = pred_mask[batch_idx]  # Shape: [slot_num, N]
            current_pred_flow = pred_flow[batch_idx]  # Shape: [N, 3]
            scene_flow_rec = torch.zeros_like(current_point_cloud_first)
            # Apply softmax to mask
            current_pred_mask = F.softmax(current_pred_mask, dim=0)
            
            # Compute reconstruction for each slot
            for slot_idx in range(current_pred_mask.shape[0]):
                slot_mask = current_pred_mask[slot_idx]  # Shape: [N]
                pred_point_cloud_second = current_point_cloud_first + current_pred_flow.unsqueeze(0)
                rotation, move = self.fit_motion_svd_batch(current_point_cloud_first, pred_point_cloud_second, slot_mask.unsqueeze(0))
                
                # Transform points
                transformed_point = torch.bmm(current_point_cloud_first, rotation) + move.unsqueeze(1)
                
                # Apply mask
                mask_expanded = slot_mask.unsqueeze(0).unsqueeze(-1)  # Shape: [1, N, 1]
                masked_scene_flow = (transformed_point - current_point_cloud_first) * mask_expanded
                
                # Add to reconstruction
                scene_flow_rec[batch_idx:batch_idx+1] += masked_scene_flow
        
            # Compute loss using Chamfer distance
            loss = self.chamferDistanceLoss(scene_flow_rec + current_point_cloud_first, current_point_cloud_second)
            loss_summ += loss
            rec_point_cloud.append(scene_flow_rec + current_point_cloud_first)
        return loss, rec_point_cloud
        pass