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
        print(f"std  {std}")
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x

def normalize_global(x):
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        print(f"std  {std}")
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x
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

    def __call__(self, sample, flow, mask):
        """
        flow: shape (B, 2, H, W) containing optical flow vectors for each pixel
        mask: shape (K, H, W) with K segments
        """
        return self.loss(sample, flow, mask)
    def loss(self, sample, mask, flow):
        B = 1
        K = mask.shape[0]
        mask = ScaleGradient.apply(mask, 1)
        point_position = sample["point_cloud_first"].to(self.device)
        scene_flows = flow
        
        # 添加幅度约束
        flow_magnitude = scene_flows.norm(dim=-1).mean()
        
        total_loss = 0.0
        for b in range(B):
            coords = self.construct_embedding(point_position)  # (L, 4)
            scene_flow_b = normalize_global(scene_flows)  # (L, 3)
            scene_flow_b = scene_flow_b*0.2
            mask_binary_b = F.softmax(mask, dim=0)  # (K, L)
            flow_reconstruction = torch.zeros_like(scene_flow_b)  # (L, 3)
            flow_reconstruction_mk = torch.zeros_like(scene_flow_b)
            reconstruction_loss = 0
            distande_loss = 0
            for k in range(K):
                mk = mask_binary_b[k].unsqueeze(-1)  # (L,1)
                
                # 加权最小二乘
                Ek = coords * mk  # sqrt加权
                Fk = scene_flow_b * mk
                
                # 正则化最小二乘
                theta_k = torch.linalg.lstsq(Ek, Fk).solution  # 更稳定的求解
                
                # 重建误差
                Fk_hat = coords @ theta_k
                flow_reconstruction += Fk_hat * mk  # (L, 3)
            reconstruction_loss += self.criterion(flow_reconstruction, scene_flow_b)
            
            # 组合损失：重建项 + 幅度约束项
            total_loss += reconstruction_loss
        
        return total_loss / (K * B)


    @torch.no_grad()
    def construct_embedding(self,point_position):
        """
        Construct the pixel coordinate embedding [x, y, z, 1]
        in flattened (HW, 4) form.
        """
        x = point_position[...,0].view(-1)
        y = point_position[...,1].view(-1)
        z = point_position[...,2].view(-1)
        # shape (L, 4)
        emb = torch.stack([x, y, z, torch.ones_like(x),torch.ones_like(x)], dim=1)
        return emb