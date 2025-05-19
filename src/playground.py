import torch
import torch.nn as nn
import torch.nn.functional as F
def fit_motion_svd_batch(pc1, pc2, mask=None):
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



point_cloud_first = torch.randn((1, 1000, 3))

rotation_matrix = torch.tensor([[0.0, -1.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0]])
translation_vector = torch.tensor([0.0, 0.0, 0.0])
point_cloud_second = torch.bmm(point_cloud_first, rotation_matrix.unsqueeze(0)) + translation_vector.unsqueeze(0)

rot_reconstructed, translation_reconstructed = fit_motion_svd_batch(point_cloud_first, point_cloud_second)
print("Rotation Matrix Reconstructed:\n", rot_reconstructed)
print("Translation Vector Reconstructed:\n", translation_reconstructed)