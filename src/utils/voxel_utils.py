"""
Voxel downsampling utility functions for point cloud downsampling.

Voxel downsampling divides the point cloud space into regular voxel grids
and samples one representative point per voxel (typically the centroid).
"""

import torch


def voxel_downsample(
    point_cloud: torch.Tensor,
    voxel_size: float | list[float],
    method: str = "mean",
    return_indices: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample point cloud using voxel grid downsampling.
    
    Args:
        point_cloud: Point cloud tensor, shape can be:
            - [N, 3] - Single point cloud
            - [B, N, 3] - Batched point clouds
        voxel_size: Voxel size, can be:
            - float: Same voxel size for all dimensions
            - list[float]: [vx, vy, vz] - Different voxel size per dimension
        method: Aggregation method for points within each voxel:
            - "mean": Use mean of points in voxel (centroid)
            - "center": Use voxel center
            - "random": Randomly sample one point from voxel
        return_indices: If True, return both sampled points and voxel indices
        
    Returns:
        If return_indices=False:
            sampled_points: Downsampled point cloud [N', 3] or [B, N', 3]
        If return_indices=True:
            (sampled_points, voxel_indices): Tuple of (downsampled points, voxel indices)
            - sampled_points: [N', 3] or [B, N', 3]
            - voxel_indices: [N'] or [B, N'] - voxel index for each sampled point
    """
    original_shape = point_cloud.shape
    original_device = point_cloud.device
    is_batched = len(original_shape) == 3
    
    # Handle single point cloud case
    if not is_batched:
        point_cloud = point_cloud.unsqueeze(0)  # [1, N, 3]
    
    B, N, C = point_cloud.shape
    
    # Normalize voxel_size
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * C
    elif len(voxel_size) != C:
        raise ValueError(f"voxel_size length ({len(voxel_size)}) must match point dimension ({C})")
    
    voxel_size = torch.tensor(voxel_size, device=point_cloud.device, dtype=point_cloud.dtype)
    
    # Compute voxel coordinates for each point
    # voxel_coords = floor((point - min) / voxel_size)
    point_min = point_cloud.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
    voxel_coords = torch.floor((point_cloud - point_min) / voxel_size.unsqueeze(0).unsqueeze(0)).long()  # [B, N, 3]
    
    # Create unique voxel indices by hashing voxel coordinates
    # Use a simple hash: hash = x * max_y * max_z + y * max_z + z
    voxel_max = voxel_coords.max(dim=1)[0]  # [B, 3]
    max_yz = (voxel_max[:, 1].max() + 1) * (voxel_max[:, 2].max() + 1)
    max_z = voxel_max[:, 2].max() + 1
    
    voxel_indices = (
        voxel_coords[:, :, 0] * max_yz +
        voxel_coords[:, :, 1] * max_z +
        voxel_coords[:, :, 2]
    )  # [B, N]
    
    # Sample points per voxel using GPU-accelerated operations
    sampled_points_list = []
    sampled_voxel_indices_list = []
    
    for b in range(B):
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices[b], return_inverse=True, return_counts=True)
        n_voxels = len(unique_voxels)
        
        if method == "mean":
            # GPU-accelerated mean using scatter_add
            # Sum points in each voxel
            sampled_pc = torch.zeros(n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
            for c in range(C):
                sampled_pc[:, c].scatter_add_(0, inverse_indices, point_cloud[b, :, c])
            # Divide by counts to get mean
            counts_float = counts.float().unsqueeze(1)  # [n_voxels, 1]
            sampled_pc = sampled_pc / counts_float
        
        elif method == "center":
            # Use voxel center (GPU-accelerated)
            # Get unique voxel coordinates using first occurrence
            unique_voxel_coords = torch.zeros(n_voxels, 3, device=point_cloud.device, dtype=torch.long)
            # Find first occurrence of each voxel index
            for i in range(n_voxels):
                # Find first point in this voxel
                first_idx = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                unique_voxel_coords[i] = voxel_coords[b, first_idx]
            # Convert to world coordinates (voxel center) - all operations are GPU-accelerated
            sampled_pc = point_min[b, 0].unsqueeze(0) + (unique_voxel_coords.float() + 0.5) * voxel_size.unsqueeze(0)
        
        elif method == "random":
            # Randomly sample one point from each voxel
            # Use first occurrence as a deterministic "random" choice (can be improved with proper RNG)
            sampled_pc = torch.zeros(n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
            for i in range(n_voxels):
                mask = inverse_indices == i
                indices = mask.nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    # Use deterministic selection (first point) for GPU efficiency
                    # For true randomness, would need proper RNG per voxel
                    random_idx = indices[0]  # Use first point (can be replaced with proper random)
                    sampled_pc[i] = point_cloud[b][random_idx]
        
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'mean', 'center', or 'random'")
        
        sampled_points_list.append(sampled_pc)
        sampled_voxel_indices_list.append(unique_voxels)
    
    # Stack results
    max_n_voxels = max(len(sp) for sp in sampled_points_list)
    sampled_points = torch.zeros(B, max_n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
    sampled_voxel_indices = torch.zeros(B, max_n_voxels, device=point_cloud.device, dtype=torch.long)
    
    for b in range(B):
        n_voxels = len(sampled_points_list[b])
        sampled_points[b, :n_voxels] = sampled_points_list[b]
        sampled_voxel_indices[b, :n_voxels] = sampled_voxel_indices_list[b]
    
    # Remove batch dimension if input was single point cloud
    if not is_batched:
        sampled_points = sampled_points.squeeze(0)
        sampled_voxel_indices = sampled_voxel_indices.squeeze(0)
        # Remove padding
        n_voxels = len(sampled_points_list[0])
        sampled_points = sampled_points[:n_voxels]
        sampled_voxel_indices = sampled_voxel_indices[:n_voxels]
    
    if return_indices:
        return sampled_points, sampled_voxel_indices
    return sampled_points


def voxel_downsample_with_attributes(
    point_cloud: torch.Tensor,
    voxel_size: float | list[float],
    *attributes: torch.Tensor,
    method: str = "mean"
) -> tuple[torch.Tensor, ...]:
    """
    Downsample point cloud and corresponding attributes using voxel grid downsampling.
    
    Args:
        point_cloud: Point cloud tensor [N, 3] or [B, N, 3]
        voxel_size: Voxel size (float or list[float])
        *attributes: Variable number of attribute tensors to downsample
            Each attribute should have shape [N, ...] or [B, N, ...]
            matching the point cloud batch dimension
        method: Aggregation method ("mean", "center", or "random")
            
    Returns:
        Tuple of (sampled_point_cloud, sampled_attribute1, sampled_attribute2, ...)
        
    Example:
        >>> pc = torch.randn(1000, 3)
        >>> colors = torch.randn(1000, 3)
        >>> normals = torch.randn(1000, 3)
        >>> pc_down, colors_down, normals_down = voxel_downsample_with_attributes(
        ...     pc, 0.1, colors, normals, method="mean"
        ... )
    """
    # Get voxel downsampling indices
    _, voxel_indices = voxel_downsample(point_cloud, voxel_size, method=method, return_indices=True)
    
    # Handle batch dimension
    is_batched = len(point_cloud.shape) == 3
    if not is_batched:
        voxel_indices = voxel_indices.unsqueeze(0)
        point_cloud = point_cloud.unsqueeze(0)
    
    B = point_cloud.shape[0]
    
    # Normalize voxel_size
    C = point_cloud.shape[-1]
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * C
    voxel_size = torch.tensor(voxel_size, device=point_cloud.device, dtype=point_cloud.dtype)
    
    # Compute voxel coordinates for original points
    point_min = point_cloud.min(dim=1, keepdim=True)[0]
    voxel_coords = torch.floor((point_cloud - point_min) / voxel_size.unsqueeze(0).unsqueeze(0)).long()
    
    # Create voxel indices
    voxel_max = voxel_coords.max(dim=1)[0]
    max_yz = (voxel_max[:, 1].max() + 1) * (voxel_max[:, 2].max() + 1)
    max_z = voxel_max[:, 2].max() + 1
    
    original_voxel_indices = (
        voxel_coords[:, :, 0] * max_yz +
        voxel_coords[:, :, 1] * max_z +
        voxel_coords[:, :, 2]
    )
    
    # Downsample point cloud
    sampled_pc_list = []
    for b in range(B):
        unique_voxels = torch.unique(original_voxel_indices[b])
        n_voxels = len(unique_voxels)
        
        if method == "mean":
            sampled_pc = torch.zeros(n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
            for i, voxel_idx in enumerate(unique_voxels):
                mask = original_voxel_indices[b] == voxel_idx
                sampled_pc[i] = point_cloud[b][mask].mean(dim=0)
        elif method == "center":
            sampled_pc = torch.zeros(n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
            for i, voxel_idx in enumerate(unique_voxels):
                mask = original_voxel_indices[b] == voxel_idx
                voxel_coord = voxel_coords[b][mask][0]
                sampled_pc[i] = point_min[b, 0] + (voxel_coord.float() + 0.5) * voxel_size
        elif method == "random":
            sampled_pc = torch.zeros(n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
            for i, voxel_idx in enumerate(unique_voxels):
                mask = original_voxel_indices[b] == voxel_idx
                indices = torch.where(mask)[0]
                random_idx = indices[torch.randint(0, len(indices), (1,), device=point_cloud.device)]
                sampled_pc[i] = point_cloud[b][random_idx]
        sampled_pc_list.append(sampled_pc)
    
    max_n_voxels = max(len(sp) for sp in sampled_pc_list)
    sampled_pc = torch.zeros(B, max_n_voxels, C, device=point_cloud.device, dtype=point_cloud.dtype)
    for b in range(B):
        n_voxels = len(sampled_pc_list[b])
        sampled_pc[b, :n_voxels] = sampled_pc_list[b]
    
    # Downsample attributes
    sampled_attrs = []
    for attr in attributes:
        if attr is None:
            sampled_attrs.append(None)
            continue
        
        sampled_attr_list = []
        for b in range(B):
            unique_voxels = torch.unique(original_voxel_indices[b])
            n_voxels = len(unique_voxels)
            
            if len(attr.shape) == 1:
                # [N] -> [n_voxels]
                if method == "mean":
                    sampled_attr = torch.zeros(n_voxels, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        sampled_attr[i] = attr_batched[b][mask].mean()
                elif method == "random":
                    sampled_attr = torch.zeros(n_voxels, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        indices = torch.where(mask)[0]
                        random_idx = indices[torch.randint(0, len(indices), (1,), device=point_cloud.device)]
                        sampled_attr[i] = attr_batched[b][random_idx]
                else:  # center - use mean for attributes
                    sampled_attr = torch.zeros(n_voxels, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        sampled_attr[i] = attr_batched[b][mask].mean()
            
            elif len(attr.shape) == 2:
                # [N, C] -> [n_voxels, C]
                attr_dim = attr.shape[-1]
                if method == "mean":
                    sampled_attr = torch.zeros(n_voxels, attr_dim, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        sampled_attr[i] = attr_batched[b][mask].mean(dim=0)
                elif method == "random":
                    sampled_attr = torch.zeros(n_voxels, attr_dim, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        indices = torch.where(mask)[0]
                        random_idx = indices[torch.randint(0, len(indices), (1,), device=point_cloud.device)]
                        sampled_attr[i] = attr_batched[b][random_idx]
                else:  # center - use mean for attributes
                    sampled_attr = torch.zeros(n_voxels, attr_dim, device=attr.device, dtype=attr.dtype)
                    attr_batched = attr.unsqueeze(0) if not is_batched else attr
                    for i, voxel_idx in enumerate(unique_voxels):
                        mask = original_voxel_indices[b] == voxel_idx
                        sampled_attr[i] = attr_batched[b][mask].mean(dim=0)
            else:
                raise ValueError(f"Unsupported attribute shape: {attr.shape}")
            
            sampled_attr_list.append(sampled_attr)
        
        # Stack attributes
        max_n_voxels = max(len(sa) for sa in sampled_attr_list)
        attr_dim = sampled_attr_list[0].shape[-1] if len(sampled_attr_list[0].shape) > 1 else 1
        if len(sampled_attr_list[0].shape) == 1:
            sampled_attr_tensor = torch.zeros(B, max_n_voxels, device=attr.device, dtype=attr.dtype)
            for b in range(B):
                n_voxels = len(sampled_attr_list[b])
                sampled_attr_tensor[b, :n_voxels] = sampled_attr_list[b]
        else:
            sampled_attr_tensor = torch.zeros(B, max_n_voxels, attr_dim, device=attr.device, dtype=attr.dtype)
            for b in range(B):
                n_voxels = len(sampled_attr_list[b])
                sampled_attr_tensor[b, :n_voxels] = sampled_attr_list[b]
        
        if not is_batched:
            n_voxels = len(sampled_attr_list[0])
            sampled_attr_tensor = sampled_attr_tensor.squeeze(0)
            sampled_attr_tensor = sampled_attr_tensor[:n_voxels]
        
        sampled_attrs.append(sampled_attr_tensor)
    
    # Remove batch dimension if input was single point cloud
    if not is_batched:
        n_voxels = len(sampled_pc_list[0])
        sampled_pc = sampled_pc.squeeze(0)
        sampled_pc = sampled_pc[:n_voxels]
    
    return (sampled_pc,) + tuple(sampled_attrs)

