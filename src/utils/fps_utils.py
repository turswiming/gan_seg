"""
Farthest Point Sampling (FPS) utility functions for point cloud downsampling.
"""

import torch
from pointnet2.pointnet2 import furthest_point_sample


def fps_downsample(
    point_cloud: torch.Tensor,
    n_points: int,
    return_indices: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample point cloud using GPU-based Farthest Point Sampling (FPS).
    
    Args:
        point_cloud: Point cloud tensor, shape can be:
            - [N, 3] - Single point cloud
            - [B, N, 3] - Batched point clouds
        n_points: Target number of points to sample
        return_indices: If True, return both sampled points and indices
        
    Returns:
        If return_indices=False:
            sampled_points: Downsampled point cloud [N', 3] or [B, N', 3]
        If return_indices=True:
            (sampled_points, indices): Tuple of (downsampled points, indices)
            - sampled_points: [N', 3] or [B, N', 3]
            - indices: [N'] or [B, N'] - indices of sampled points
    """
    original_shape = point_cloud.shape
    original_device = point_cloud.device
    is_batched = len(original_shape) == 3
    
    # Handle single point cloud case
    if not is_batched:
        point_cloud = point_cloud.unsqueeze(0)  # [1, N, 3]
    
    B, N, C = point_cloud.shape
    
    # Check if downsampling is needed
    if N <= n_points:
        if return_indices:
            indices = torch.arange(N, device=point_cloud.device).unsqueeze(0).expand(B, -1)
            if not is_batched:
                indices = indices.squeeze(0)
            return point_cloud.squeeze(0) if not is_batched else point_cloud, indices
        return point_cloud.squeeze(0) if not is_batched else point_cloud
    
    # Ensure point cloud is on GPU and contiguous
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        point_cloud_gpu = point_cloud.to('cuda').contiguous()
    else:
        point_cloud_gpu = point_cloud.contiguous()
    
    # FPS requires batch dimension: (B, N, 3)
    fps_indices = furthest_point_sample(point_cloud_gpu, n_points)  # [B, n_points] (IntTensor)
    fps_indices = fps_indices.long()  # Convert to LongTensor for indexing
    
    # Move indices back to original device if needed
    if not use_gpu or original_device.type == 'cpu':
        fps_indices = fps_indices.cpu()
    
    # Apply FPS indices to get sampled points
    # Use advanced indexing: point_cloud[batch_idx, fps_indices[batch_idx], :]
    sampled_points = torch.zeros(B, n_points, C, device=point_cloud.device, dtype=point_cloud.dtype)
    for b in range(B):
        sampled_points[b] = point_cloud[b][fps_indices[b]]
    
    # Remove batch dimension if input was single point cloud
    if not is_batched:
        sampled_points = sampled_points.squeeze(0)
        fps_indices = fps_indices.squeeze(0)
    
    if return_indices:
        return sampled_points, fps_indices
    return sampled_points


def fps_downsample_with_attributes(
    point_cloud: torch.Tensor,
    n_points: int,
    *attributes: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Downsample point cloud and corresponding attributes using FPS.
    
    Args:
        point_cloud: Point cloud tensor [N, 3] or [B, N, 3]
        n_points: Target number of points
        *attributes: Variable number of attribute tensors to downsample
            Each attribute should have shape [N, ...] or [B, N, ...]
            matching the point cloud batch dimension
            
    Returns:
        Tuple of (sampled_point_cloud, sampled_attribute1, sampled_attribute2, ...)
        
    Example:
        >>> pc = torch.randn(1000, 3)
        >>> colors = torch.randn(1000, 3)
        >>> normals = torch.randn(1000, 3)
        >>> pc_down, colors_down, normals_down = fps_downsample_with_attributes(
        ...     pc, 512, colors, normals
        ... )
    """
    # Get FPS indices
    _, fps_indices = fps_downsample(point_cloud, n_points, return_indices=True)
    
    # Handle batch dimension
    is_batched = len(point_cloud.shape) == 3
    if not is_batched:
        fps_indices = fps_indices.unsqueeze(0)
        point_cloud = point_cloud.unsqueeze(0)
    
    B = point_cloud.shape[0]
    
    # Downsample point cloud
    sampled_pc = torch.zeros(B, n_points, point_cloud.shape[-1], 
                             device=point_cloud.device, dtype=point_cloud.dtype)
    for b in range(B):
        sampled_pc[b] = point_cloud[b][fps_indices[b]]
    
    # Downsample attributes
    sampled_attrs = []
    for attr in attributes:
        if attr is None:
            sampled_attrs.append(None)
            continue
            
        # Handle different attribute shapes
        if len(attr.shape) == 1:
            # [N] -> [n_points]
            attr_batched = attr.unsqueeze(0) if not is_batched else attr
            sampled_attr = torch.zeros(B, n_points, device=attr.device, dtype=attr.dtype)
            for b in range(B):
                sampled_attr[b] = attr_batched[b][fps_indices[b]]
            sampled_attr = sampled_attr.squeeze(0) if not is_batched else sampled_attr
        elif len(attr.shape) == 2:
            # [N, C] -> [n_points, C]
            attr_batched = attr.unsqueeze(0) if not is_batched else attr
            sampled_attr = torch.zeros(B, n_points, attr.shape[-1], 
                                      device=attr.device, dtype=attr.dtype)
            for b in range(B):
                sampled_attr[b] = attr_batched[b][fps_indices[b]]
            sampled_attr = sampled_attr.squeeze(0) if not is_batched else sampled_attr
        else:
            raise ValueError(f"Unsupported attribute shape: {attr.shape}")
        
        sampled_attrs.append(sampled_attr)
    
    # Remove batch dimension if input was single point cloud
    if not is_batched:
        sampled_pc = sampled_pc.squeeze(0)
    
    return (sampled_pc,) + tuple(sampled_attrs)


def batch_fps_downsample(
    point_clouds: list[torch.Tensor],
    n_points: int,
    return_indices: bool = False
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Downsample multiple point clouds using FPS.
    
    Args:
        point_clouds: List of point cloud tensors, each [N_i, 3]
        n_points: Target number of points for each point cloud
        return_indices: If True, return both sampled points and indices
        
    Returns:
        If return_indices=False:
            List of downsampled point clouds
        If return_indices=True:
            Tuple of (list of sampled points, list of indices)
    """
    sampled_points_list = []
    indices_list = []
    
    for pc in point_clouds:
        if return_indices:
            sampled_pc, indices = fps_downsample(pc, n_points, return_indices=True)
            sampled_points_list.append(sampled_pc)
            indices_list.append(indices)
        else:
            sampled_pc = fps_downsample(pc, n_points, return_indices=False)
            sampled_points_list.append(sampled_pc)
    
    if return_indices:
        return sampled_points_list, indices_list
    return sampled_points_list

