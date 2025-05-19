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
    Part of the smooth loss by KNN.
    """
    def __init__(self, k=8, radius=0.1, cross_entropy=False, loss_norm=1, **kwargs):
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
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
    Part of the smooth loss by ball query.
    """
    def __init__(self, k = 16, radius = 0.2, cross_entropy=False, loss_norm=1, **kwargs):
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :return:
            loss: () torch.Tensor.
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
    Enforce local smoothness of object mask.
    """
    def __init__(self, w_knn = 3, w_ball_q =1):
        super().__init__()
        self.knn_loss = KnnLoss()
        self.ball_q_loss = BallQLoss()
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc, mask):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, K, N) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        # Reshape mask from (B, K, N) to (B, N, K) for compatibility with loss functions
        batch_size = mask.shape[0]
        mask_reshaped = mask.permute(0, 2, 1)  # Change to (B, N, K)
        
        loss = (self.w_knn * self.knn_loss(pc, mask_reshaped)) + (self.w_ball_q * self.ball_q_loss(pc, mask_reshaped))
        return loss