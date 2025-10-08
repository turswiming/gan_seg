"""
Memory-optimized Reconstruction Loss implementation for point cloud reconstruction.

This module implements a memory-efficient reconstruction loss that measures how well a predicted
point cloud matches the ground truth, taking into account both point positions
and their flow vectors.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from losses.loss_chamfer import my_chamfer_fn

class ReconstructionLossOptimized():
    """
    Memory-optimized Reconstruction Loss for point cloud and flow prediction.
    
    This loss function evaluates the quality of reconstructed point clouds by
    comparing them with ground truth data, considering both point positions
    and their associated flow vectors. Optimized for memory efficiency.
    
    Attributes:
        device (torch.device): Device to perform computations on
        use_checkpointing (bool): Whether to use gradient checkpointing
        chunk_size (int): Chunk size for memory-efficient operations
    """
    
    def __init__(self, device, use_checkpointing=True, chunk_size=1024):
        """
        Initialize the Memory-optimized Reconstruction Loss module.
        
        Args:
            device (torch.device): Device to perform computations on
            use_checkpointing (bool): Whether to use gradient checkpointing to save memory
            chunk_size (int): Chunk size for memory-efficient operations
        """
        self.device = device
        self.chamferDistanceLoss = my_chamfer_fn
        self.use_checkpointing = use_checkpointing
        self.chunk_size = chunk_size

    def _fit_motion_svd_chunk(self, pc1_chunk, pc2_chunk, mask_chunk=None):
        """
        Memory-efficient SVD computation for a chunk of data.
        """
        n_batch, n_point, _ = pc1_chunk.size()

        if mask_chunk is None:
            pc1_mean = torch.mean(pc1_chunk, dim=1, keepdim=True)
            pc2_mean = torch.mean(pc2_chunk, dim=1, keepdim=True)
        else:
            pc1_mean = torch.einsum('bnd,bn->bd', pc1_chunk, mask_chunk) / torch.sum(mask_chunk, dim=1, keepdim=True)
            pc1_mean.unsqueeze_(1)
            pc2_mean = torch.einsum('bnd,bn->bd', pc2_chunk, mask_chunk) / torch.sum(mask_chunk, dim=1, keepdim=True)
            pc2_mean.unsqueeze_(1)

        pc1_centered = pc1_chunk - pc1_mean
        pc2_centered = pc2_chunk - pc2_mean

        if mask_chunk is None:
            S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
        else:
            S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask_chunk).bmm(pc2_centered))

        valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
        R_base = torch.eye(3, device=pc1_chunk.device).unsqueeze(0).repeat(n_batch, 1, 1)
        t_base = torch.zeros((n_batch, 3), device=pc1_chunk.device)

        if valid_batches.any():
            S_valid = S[valid_batches, ...]
            u, s, v = torch.svd(S_valid, some=False, compute_uv=True)
            R = torch.bmm(v, u.transpose(1, 2))
            det = torch.det(R)

            diag = torch.ones_like(S_valid[..., 0], requires_grad=False)
            diag[:, 2] = det
            R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

            pc1_mean_valid, pc2_mean_valid = pc1_mean[valid_batches], pc2_mean[valid_batches]
            t = pc2_mean_valid.squeeze(1) - torch.bmm(R, pc1_mean_valid.transpose(1, 2)).squeeze(2)

            R_base[valid_batches] = R.to(R_base.dtype)
            t_base[valid_batches] = t.to(t_base.dtype)

        return R_base, t_base

    def fit_motion_svd_batch(self, pc1, pc2, mask=None):
        """
        Memory-efficient batch SVD computation with chunking.
        """
        if self.use_checkpointing:
            return checkpoint(self._fit_motion_svd_chunk, pc1, pc2, mask)
        else:
            return self._fit_motion_svd_chunk(pc1, pc2, mask)
    
    def soft_knn_chunked(self, query_points, reference_points, reference_values, k=5, temperature=0.1):
        """
        Memory-efficient soft k-nearest neighbors with chunking.
        """
        n_queries = query_points.shape[0]
        n_refs = reference_points.shape[0]
        
        # Process in chunks to avoid memory explosion
        chunk_size = min(self.chunk_size, n_queries)
        interpolated_values = torch.zeros((n_queries, reference_values.shape[1]), 
                                        device=query_points.device, dtype=reference_values.dtype)
        
        for i in range(0, n_queries, chunk_size):
            end_idx = min(i + chunk_size, n_queries)
            query_chunk = query_points[i:end_idx]
            
            # Calculate distances for this chunk
            query_expanded = query_chunk.unsqueeze(1)  # (chunk_size, 1, d)
            ref_expanded = reference_points.unsqueeze(0)  # (1, n_refs, d)
            
            squared_distances = torch.sum((query_expanded - ref_expanded) ** 2, dim=2)
            
            # Get k nearest neighbors
            distances, indices = torch.topk(squared_distances, k=k, dim=1, largest=False)
            
            # Distance-based weighting
            weights = torch.exp(-distances / temperature)
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
            
            # Gather values and compute weighted average
            neighbor_values = reference_values[indices]
            interpolated_values[i:end_idx] = torch.sum(weights.unsqueeze(-1) * neighbor_values, dim=1)
        
        return interpolated_values
    
    def __call__(self, point_cloud_first, point_cloud_second, pred_mask, pred_flow):
        """
        Memory-efficient reconstruction loss computation.
        """
        point_cloud_first = [item.to(self.device) for item in point_cloud_first]
        point_cloud_second = [item.to(self.device) for item in point_cloud_second]
        pred_mask = [item.to(self.device) for item in pred_mask]
        pred_flow = [item.to(self.device) for item in pred_flow]
        
        batch_size = len(point_cloud_first)
        loss_summ = 0
        rec_point_cloud = []
        
        # Process each batch item with memory optimization
        for batch_idx in range(batch_size):
            current_point_cloud_first = point_cloud_first[batch_idx].unsqueeze(0)
            current_point_cloud_second = point_cloud_second[batch_idx].unsqueeze(0)
            current_pred_mask = pred_mask[batch_idx]
            current_pred_flow = pred_flow[batch_idx]
            
            # Initialize reconstruction with zeros
            scene_flow_rec = torch.zeros_like(current_point_cloud_first)
            
            # Apply softmax to mask
            current_pred_mask = F.softmax(current_pred_mask, dim=0)
            
            # Process slots with memory optimization
            for slot_idx in range(current_pred_mask.shape[0]):
                slot_mask = current_pred_mask[slot_idx]
                pred_point_cloud_second = current_point_cloud_first + current_pred_flow.unsqueeze(0)
                
                # Use memory-efficient SVD
                rotation, move = self.fit_motion_svd_batch(
                    current_point_cloud_first, 
                    pred_point_cloud_second, 
                    slot_mask.unsqueeze(0)
                )
                
                # Transform points
                transformed_point = torch.bmm(current_point_cloud_first, rotation) + move.unsqueeze(1)
                
                # Apply mask and accumulate in-place
                mask_expanded = slot_mask.unsqueeze(0).unsqueeze(-1)
                masked_scene_flow = (transformed_point - current_point_cloud_first) * mask_expanded
                scene_flow_rec += masked_scene_flow  # In-place accumulation
        
            # Compute loss
            final_point_cloud = scene_flow_rec + current_point_cloud_first
            loss = self.chamferDistanceLoss(final_point_cloud, current_point_cloud_second)
            loss_summ += loss
            rec_point_cloud.append(final_point_cloud)
            
        return loss_summ / batch_size, rec_point_cloud
