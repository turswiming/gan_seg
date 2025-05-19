import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .ChamferDistanceLoss import ChamferDistanceLoss
class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input  # 前向传播返回原始值

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时调整梯度幅度
        return grad_output * ctx.scale, None  # 第二个 None 是因为 scale 不需要梯度

def normalize_global(x):
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x

def normalize_useing_other(x,points):
    with torch.no_grad():
        std = points.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    return x/std

class FlowSmoothLoss():
    """
    Reproduces the parametric (quadratic) flow approximation loss described in Section 3.1:
      Lf(M|F) = sum_k || Fk - F̂k ||^2_F ,  where F̂k = Ek θ̂k  and θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
    """

    def __init__(self,device):
        """Initialize with config and model/device references."""
        self.device=device
        self.criterion = nn.MSELoss(reduction="mean").to(self.device)
        self.chamferDistanceLoss = ChamferDistanceLoss()
        pass

    def __call__(self, sample, mask, flow):
        """
        flow: shape (B, N, 3) containing flow vectors for each point
        mask: shape (B, K, N) with K segments and B batches
        """
        return self.loss(sample, mask, flow)
        
    def loss(self, sample, mask, flow):
        batch_size = sample["point_cloud_first"].shape[0]
        point_position = sample["point_cloud_first"].to(self.device)
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
                
                # Solve for parameters
                theta_k = torch.linalg.lstsq(Ek, Fk).solution
                
                # Reconstruct flow
                Fk_hat = Ek @ theta_k
                flow_reconstruction += Fk_hat  # (N, 3)
                
            # Compute reconstruction loss
            reconstruction_loss = self.criterion(flow_reconstruction, scene_flow_b)
            total_loss += reconstruction_loss
        
        # Return average loss
        return total_loss / batch_size

    @torch.no_grad()
    def construct_embedding(self, point_position):
        """
        Construct the point coordinate embedding [x, y, z, 1, 1]
        """
        x = point_position[..., 0].view(-1)
        y = point_position[..., 1].view(-1)
        z = point_position[..., 2].view(-1)
        # shape (N, 5)
        emb = torch.stack([x, y, z, torch.ones_like(x), torch.ones_like(x)], dim=1)
        return emb