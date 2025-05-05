import torch
import open3d as o3d
from torch import nn
from torch.nn import functional as F
from losses.ChamferDistanceLoss import ChamferDistanceLoss
class ReconstructionLoss():
    def __init__(self,device):
        self.device = device
        self.chamferDistanceLoss = ChamferDistanceLoss()
        pass

    def fit_motion_svd_batch(self, pc1, pc2, mask=None):
        """
        :param pc1: (B, N, 3) torch.Tensor.
        :param pc2: (B, N, 3) torch.Tensor. pc1 and pc2 should be the same point cloud only with disturbed.
        :param mask: (B, N) torch.Tensor.
        :return:
            R_base: (B, 3, 3) torch.Tensor.
            t_base: (B, 3) torch.Tensor.
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
    
    def __call__(self, inputs,pred_mask, pred_flow):
        point_cloud_first = inputs["point_cloud_first"].to(self.device)
        point_cloud_second = inputs["point_cloud_second"].to(self.device)
        pred_mask = pred_mask.to(self.device)
        pred_flow = pred_flow.to(self.device)
        pred_mask = F.softmax(pred_mask, dim=0)
        # print(pred_mask.dtype)
        # #convert to one hot
        # pred_mask = torch.argmax(pred_mask, dim=0)
        # pred_mask = F.one_hot(pred_mask)
        # print("pred_mask", pred_mask.shape)
        # pred_mask = pred_mask.permute(1,0)
        # pred_mask = pred_mask.to(self.device)
        # pred_mask = pred_mask.to(torch.float64)
        # pred_mask+= 0.00000000001
        # pred_mask = pred_mask / torch.sum(pred_mask, dim=1, keepdim=True)
        '''
        loss2
        point_cloud_first torch.Size([1, 4121, 3])
        point_cloud_second torch.Size([1, 4121, 3])
        pred_mask torch.Size([8, 4121])
        pred_flow torch.Size([4121, 3])
        '''
        scene_flow_rec = torch.zeros_like(point_cloud_first)
        for b in range(pred_mask.shape[0]):
            pred_mask_b = pred_mask[b]
            pred_point_cloud_second = point_cloud_first + pred_flow.unsqueeze(0)
            rotation, move = self.fit_motion_svd_batch(point_cloud_first, pred_point_cloud_second, pred_mask_b.unsqueeze(0))
            rotation = rotation.to(torch.float64)
            move = move.to(torch.float64)
            transformed_point = torch.bmm(point_cloud_first, rotation) + move

            # Add an extra dimension to the mask tensor to make it [1, 4121, 1]
            mask_expanded = pred_mask_b.unsqueeze(0).unsqueeze(-1)  # Shape: [1, 4121, 1]

            # Now the broadcasting will work correctly
            masked_scene_flow = (transformed_point - point_cloud_first) * mask_expanded
            scene_flow_rec += masked_scene_flow
        loss = self.chamferDistanceLoss(scene_flow_rec+point_cloud_first, point_cloud_second)
        return loss, scene_flow_rec+point_cloud_first
        pass