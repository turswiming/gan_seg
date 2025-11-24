"""
Flow Smoothness Loss implementation for scene flow prediction.

This module implements a parametric flow smoothness loss that encourages spatially
coherent flow predictions within segmented regions. It uses a quadratic flow
approximation approach to ensure smooth transitions in the predicted flow field.

Key Optimizations:
- Parallel matrix operations for k-dimensional slots processing
- Batch subdivision strategy: automatically selects optimal batch sizes (5, 4, or 3)
  based on K to maximize GPU utilization and memory efficiency
- Vectorized loss computation to eliminate sequential loops
"""

import torch
import functools
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BinaryMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        probs = F.softmax(input, dim=0)
        argmax_index = torch.argmax(input, dim=0)
        binary = F.one_hot(argmax_index, num_classes=input.shape[0]).float().permute(1, 0)
        binary = F.softmax(binary * input.shape[0], dim=0)
        ctx.save_for_backward(binary, probs)
        return binary

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ScaleGradient(torch.autograd.Function):
    """
    Custom autograd function for gradient scaling during backpropagation.

    This function allows for controlled gradient flow by scaling gradients
    during the backward pass while leaving the forward pass unchanged.
    """

    @staticmethod
    def forward(ctx, input, scale):
        """
        Forward pass: store scale and return input unchanged.

        Args:
            ctx: Context object to store information for backward pass
            input (torch.Tensor): Input tensor
            scale (float): Scale factor for gradients

        Returns:
            torch.Tensor: Input tensor unchanged
        """
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: scale the gradients.

        Args:
            ctx: Context object containing the scale factor
            grad_output (torch.Tensor): Incoming gradient

        Returns:
            tuple: (Scaled gradient, None for scale parameter)
        """
        return grad_output * ctx.scale, None


def normalize_global(x):
    """
    Normalize tensor globally using standard deviation.

    Args:
        x (torch.Tensor): Input tensor to normalize

    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x / std  # (HW, 2)
    return x


def normalize_useing_other(x, points):
    """
    Normalize tensor using standard deviation from another tensor.

    Args:
        x (torch.Tensor): Tensor to normalize
        points (torch.Tensor): Tensor to compute normalization statistics from

    Returns:
        torch.Tensor: Normalized tensor
    """
    with torch.no_grad():
        std = points.clone().reshape(-1).std(dim=0)
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    return x / std


class FlowSmoothLoss:
    """
    Flow smoothness loss using parametric quadratic flow approximation.

    This loss encourages smooth flow fields within segmented regions by fitting
    a quadratic model to the flow in each segment and penalizing deviations
    from this model.

    The loss is computed as:
    Lf(M|F) = sum_k || Fk - F̂k ||^2_F
    where F̂k = Ek θ̂k and θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk

    Attributes:
        device (torch.device): Device to perform computations on
        criterion (nn.Module): MSE loss function for reconstruction error
    """

    def __init__(self, device, flow_smooth_loss_config):
        """
        Initialize the Flow Smoothness Loss.

        Args:
            device (torch.device): Device to perform computations on
        """
        self.device = device
        each_mask_item_gradient = flow_smooth_loss_config.each_mask_item.relative_gradient
        sum_mask_item_gradient = flow_smooth_loss_config.sum_mask_item.relative_gradient
        self.each_mask_item_gradient = each_mask_item_gradient / (each_mask_item_gradient + sum_mask_item_gradient)
        self.sum_mask_item_gradient = sum_mask_item_gradient / (each_mask_item_gradient + sum_mask_item_gradient)
        self.each_mask_item_loss = flow_smooth_loss_config.each_mask_item.criterion
        self.sum_mask_item_loss = flow_smooth_loss_config.sum_mask_item.criterion
        self.scale_flow_grad = flow_smooth_loss_config.scale_flow_grad
        self.square_mask = flow_smooth_loss_config.square_mask
        # 是否标准化flow使其对尺度不敏感（loss不监督flow的幅度）
        # 如果为True，会在计算loss前将flow标准化到单位标准差
        self.normalize_flow = getattr(flow_smooth_loss_config, "normalize_flow", True)
        if self.each_mask_item_loss in ["L1", "l1"]:
            self.each_mask_criterion = nn.L1Loss(reduction="none").to(self.device)
        elif self.each_mask_item_loss in ["L2", "l2"]:
            self.each_mask_criterion = nn.MSELoss(reduction="none").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.each_mask_item_loss}")
        if self.sum_mask_item_loss in ["L1", "l1"]:
            self.sum_mask_criterion = nn.L1Loss(reduction="none").to(self.device)
        elif self.sum_mask_item_loss in ["L2", "l2"]:
            self.sum_mask_criterion = nn.MSELoss(reduction="none").to(self.device)
        else:
            raise ValueError(f"Invalid loss criterion: {self.sum_mask_item_loss}")
        self.sparce_filter_ratio = 0.0
        self.singular_value_loss_gradient = flow_smooth_loss_config.singular_value_loss_gradient
        pass

    def __call__(self, point_position, mask, flow, singular_value_loss=False):
        """
        Compute the flow smoothness loss.

        Args:
            point_position (list[torch.Tensor]): List of point positions, each of shape [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]

        Returns:
            torch.Tensor: Computed smoothness loss averaged across the batch
        """
        return self.loss(point_position, mask, flow, singular_value_loss)

    def get_optimal_batch_size(self, K):
        """根据K选择最优的batch size (5, 4, 3的倍数)"""
        # 优先选择能整除K的最大batch size
        # divisors = [5, 4, 3]
        # for batch_size in divisors:
        #     if K % batch_size == 0:
        #         return batch_size

        return 1  # 对于很小的K值，直接全部处理

    def loss(self, point_position, mask, flow, singular_value_loss=False):
        """
        Core loss computation function.

        For each segment in the mask, fits a quadratic model to the flow
        and penalizes deviations from this model.

        Args:
            sample (dict): Input data containing:
                - point_cloud_first (list[torch.Tensor]): List of first frame point clouds [N, 3]
            mask (list[torch.Tensor]): List of segmentation masks, each of shape [K, N]
            flow (list[torch.Tensor]): List of predicted flow vectors, each of shape [N, 3]

        Returns:
            torch.Tensor: Average reconstruction loss across all batches
        """
        batch_size = len(point_position)
        point_position = [item.to(self.device) for item in point_position]
        scene_flows = flow

        total_loss = 0.0
        robust_loss = 0.0
        no_scale_loss = 0.0
        for b in range(batch_size):
            one_batch_loss = 0.0
            N = point_position[b].shape[0]
            # Get batch data
            point_position_b = point_position[b]  # (N, 3)
            scene_flow_b = scene_flows[b]  # (N, 3)
            mask_b = mask[b]  # (K, N)
            K = mask_b.shape[0]
            # scene_flow_b = normalize_useing_other(scene_flow_b, point_position_b)
            # mask_b_logits = torch.log(mask_b + 1e-4)
            # mask_b_logits = mask_b_logits * (10 / math.exp(1))
            # mask_binary_b = F.softmax(mask_b_logits, dim=0)  # (K, N)
            # I already know that this loss works well when the number of slots is 2~4
            # to follow the original paper, we need to normalize the original logits, that make the softmaxed value to "looks like" origin paper
            # We need to select a number between 2 and 4, I don't know how to select it, so I use the math.exp(1)
            # We find it works well,We hope we can explain it in the future
            # mask_binary_b = BinaryMask.apply(mask_b)
            mask_binary_b = mask_b
            if self.normalize_flow:
                flow_std = scene_flow_b.std(dim=0).detach().max()
                scene_flow_b = scene_flow_b / (flow_std + 1e-1) * 10

            scene_flow_b = ScaleGradient.apply(scene_flow_b.clone(), self.scale_flow_grad)
            # Construct embedding
            coords = self.construct_embedding(point_position_b)  # (N, 4)

            # Initialize flow reconstruction
            flow_reconstruction = torch.zeros_like(scene_flow_b)  # (N, 3)
            # Per-slot reconstruction using parallel matrix operations
            reconstruction_loss = 0

            # Batch processing with K subdivision for acceleration
            # 根据K的大小选择最佳的batch_size进行分批计算

            batch_size_k = 1  # always 1 for now
            num_batches = (K + batch_size_k - 1) // batch_size_k  # 向上取整
            flow_magnitude = torch.norm(scene_flow_b, dim=-1)
            # 分批处理K个slots
            ignored_slots = 0
            for k in range(K):
                # 获取当前批次的mask
                mask_bk = mask_binary_b[k]  # (N,)
                # Ensure mask_bk is 1D with shape (N,)
                if mask_bk.dim() > 1:
                    mask_bk = mask_bk.squeeze()
                if self.square_mask:
                    mask_bk = torch.sqrt(mask_bk)

                # 批量应用masks到embeddings和flows
                # coords: (N, 4), mask_bk: (N,), need to unsqueeze for broadcasting
                Ek = coords * mask_bk.unsqueeze(-1)  # (N, 4)
                Fk = scene_flow_b * mask_bk.unsqueeze(-1)  # (N, 3)

                # add a small noise to the Fk
                # Fk = Fk + torch.randn_like(Fk) * 1e-6
                # 线性最小二乘求解
                theta = torch.linalg.lstsq(Ek, Fk, driver="gels").solution  # (4, 3)
                if singular_value_loss and self.singular_value_loss_gradient > 0:
                    M = theta[:3, :].T + torch.eye(3).to(self.device)
                    U, S, V_h = torch.linalg.svd(M,full_matrices=False)
                    no_scale_loss += F.mse_loss(S, torch.ones_like(S).to(self.device)) * self.singular_value_loss_gradient
                # print the invalid slots
                if torch.isnan(theta).any():
                    print(f"theta is nan")
                    flow_reconstruction += Fk
                    continue
                Fk_hat = Ek @ theta  # (N, 3)

                # 只对有效的slots进行累加
                flow_reconstruction += Fk_hat

                # 向量化损失计算
                if self.each_mask_item_gradient > 0:
                    batch_reconstruction_loss = self.each_mask_criterion(Fk_hat, Fk).mean(dim=-1)
                    if self.sparce_filter_ratio > 0:
                        loss_detached = batch_reconstruction_loss.detach()
                        robust_loss += loss_detached.sum() / N * self.each_mask_item_gradient
                        # 使用 topk(largest=False) 获取最小的 k 个值，相当于 bottomk
                        useless_loss_mask = torch.topk(
                            loss_detached,
                            k=int(loss_detached.shape[1] * self.sparce_filter_ratio),
                            dim=1,
                            largest=False,
                        ).indices
                        batch_reconstruction_loss[:, useless_loss_mask] = 0
                    else:
                        robust_loss += batch_reconstruction_loss.sum() / N * self.each_mask_item_gradient
                    batch_reconstruction_loss = batch_reconstruction_loss.sum()

                    one_batch_loss += batch_reconstruction_loss * self.each_mask_item_gradient / N
                pass
            if self.sum_mask_item_gradient > 0:
                reconstruction_loss = self.sum_mask_criterion(scene_flow_b, flow_reconstruction)  # (N, 3)
                # 对每个点的loss进行聚合（对3个维度求和），得到每个点的总loss
                point_loss = reconstruction_loss.sum(dim=-1)  # (N,)
                if self.sparce_filter_ratio > 0:
                    loss_detached = point_loss.detach()
                    robust_loss += loss_detached.sum() / N * self.sum_mask_item_gradient
                    # 使用 topk(largest=False) 获取最小的 k 个值，相当于 bottomk
                    # 确保 k 不超过点的数量
                    useless_loss_mask = torch.topk(
                        loss_detached, k=int(loss_detached.shape[0] * self.sparce_filter_ratio), dim=0, largest=False
                    ).indices
                    # 将选中的点的loss设为0
                    reconstruction_loss[useless_loss_mask, :] = 0
                else:
                    robust_loss += point_loss.sum() / N * self.sum_mask_item_gradient
                reconstruction_loss = reconstruction_loss.sum()
                one_batch_loss += reconstruction_loss * self.sum_mask_item_gradient / N
            total_loss += one_batch_loss
            # Compute reconstruction loss
            # with torch.no_grad():
            #     flow_reconstruction = flow_reconstruction.detach()
            # reconstruction_loss = torch.pow(torch.log((scene_flow_b+1e8)/(flow_reconstruction+1e8)), 2).mean()

        # Return average loss
        return total_loss / batch_size + no_scale_loss / batch_size

    @torch.no_grad()
    def construct_embedding(self, point_position):
        """
        Construct point coordinate embedding [x, y, z, 1, 1].

        Creates an embedding that allows for quadratic flow approximation
        by augmenting the 3D coordinates with constant terms.

        Args:
            point_position (torch.Tensor): Point cloud positions [N, 3]

        Returns:
            torch.Tensor: Embedding vectors [N, 5]
        """
        x = point_position[..., 0].view(-1)
        y = point_position[..., 1].view(-1)
        z = point_position[..., 2].view(-1)
        # shape (N, 4)
        emb = torch.stack(
            [
                x,
                y,
                z,
                # x*x*x,
                # y*y*y,
                # z*z*z,
                # x*x*x*x*x,
                # y*y*y*y*y,
                # z*z*z*z*z,
                # x*y,
                # y*z,
                # x*z,
                torch.ones_like(x),
            ],
            dim=1,
        )
        return emb
