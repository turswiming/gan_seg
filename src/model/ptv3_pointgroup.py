"""
PointGroup module for Point Transformer V3.

PointGroup is an instance segmentation method that uses:
1. Semantic segmentation to identify object classes
2. Offset prediction to predict point-to-instance-center offsets
3. Clustering algorithm to group points into instances

Reference: PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import sys
import os

# Note: We use differentiable alternatives instead of FPS and ball_query
# Original FPS and ball_query are not differentiable, so we implement
# soft versions using distance-based operations

# Add PointTransformerV3 to path
ptv3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PointTransformerV3'))
ptv3_parent = os.path.dirname(ptv3_path)

if ptv3_parent not in sys.path:
    sys.path.insert(0, ptv3_parent)
if ptv3_path not in sys.path:
    sys.path.insert(0, ptv3_path)

try:
    from PointTransformerV3.model import PointTransformerV3
except ImportError:
    import importlib.util
    serialization_path = os.path.join(ptv3_path, "serialization", "__init__.py")
    if os.path.exists(serialization_path):
        serialization_spec = importlib.util.spec_from_file_location("serialization", serialization_path)
        serialization_module = importlib.util.module_from_spec(serialization_spec)
        serialization_spec.loader.exec_module(serialization_module)
    
    model_path = os.path.join(ptv3_path, "model.py")
    spec = importlib.util.spec_from_file_location("ptv3_model", model_path)
    ptv3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ptv3_module)
    PointTransformerV3 = ptv3_module.PointTransformerV3

import torch
import torch.nn.functional as F

class InstanceToSemanticGrad(torch.autograd.Function):
    """
    Fully differentiable instance clustering using PointGroup's Dual-Set method.
    
    PointGroup Algorithm (Differentiable Version):
    1. Original Set: Original point coordinates
    2. Shifted Set: Points shifted by predicted offsets (points + offsets)
    3. Seed Selection: Use soft FPS (distance-based sampling) on shifted set
    4. Soft Ball Query: Use distance-based soft assignment instead of hard ball query
    5. Expansion: Expand clusters in original set with soft weights
    
    - Forward: Performs fully differentiable clustering to compute instance_masks.
    - Backward: Propagates gradients from instance_masks to semantic_logits, offsets, and points.
    """
    @staticmethod
    def forward(ctx, semantic_logits, points, offsets, num_classes, min_points_per_instance, cluster_threshold, max_clusters_per_class, class_prob_threshold):
        # Validate input shapes
        assert points.shape[0] == semantic_logits.shape[0], \
            f"Batch size mismatch: points.shape={points.shape}, semantic_logits.shape={semantic_logits.shape}"
        assert points.shape[0] == offsets.shape[0], \
            f"Batch size mismatch: points.shape={points.shape}, offsets.shape={offsets.shape}"
        assert points.shape[1] == semantic_logits.shape[1], \
            f"Point count mismatch: points.shape={points.shape}, semantic_logits.shape={semantic_logits.shape}"
        assert points.shape[1] == offsets.shape[1], \
            f"Point count mismatch: points.shape={points.shape}, offsets.shape={offsets.shape}"
        
        batch_size = points.shape[0]
        device = points.device
        
        # Compute semantic probabilities (differentiable, but saved for backward)
        semantic_probs = F.softmax(semantic_logits, dim=-1)  # [B, N, num_classes]
        
        # Dual-Set PointGroup method:
        # - Original Set: points (original coordinates)
        # - Shifted Set: points + offsets (shifted coordinates)
        original_points = points  # [B, N, 3] - Original set
        shifted_points = points + offsets  # [B, N, 3] - Shifted set
        
        instance_masks_list = []
        instance_classes_list = []  # List of lists: for each batch, list of class_ids for each instance
        
        for b in range(batch_size):
            pc_original = original_points[b]  # [N, 3] - Original set
            pc_shifted = shifted_points[b]  # [N, 3] - Shifted set
            semantic_b = semantic_probs[b]  # [N, num_classes]
            
            # Collect valid classes and their points
            # 筛选条件：
            # 1. 类别概率 > class_prob_threshold 的点
            # 2. 这些点的数量 >= min_points_per_instance (默认50)
            # 注意：现在包含背景类别（class 0），并且背景也可以分成多个实例
            valid_classes = []
            class_stats = []  # For debugging
            for class_id in range(0, num_classes):  # Include background (class 0)
                class_mask = semantic_b[:, class_id]  # [N] - 每个点属于该类别的概率
                class_points_idx = torch.where(class_mask > class_prob_threshold)[0]  # 找到概率 > threshold 的点
                num_points_in_class = len(class_points_idx)
                
                # 统计信息（用于调试）
                if num_points_in_class > 0:
                    max_prob = class_mask.max().item()
                    mean_prob = class_mask[class_points_idx].mean().item()
                    class_stats.append((class_id, num_points_in_class, max_prob, mean_prob))
                
                # 筛选：点数必须 >= min_points_per_instance
                if num_points_in_class < min_points_per_instance:
                    continue
                
                # Get points from both original and shifted sets
                class_points_original = pc_original[class_points_idx]  # [N_class, 3] - Original set
                class_points_shifted = pc_shifted[class_points_idx]  # [N_class, 3] - Shifted set
                valid_classes.append((class_id, class_points_original, class_points_shifted, class_mask, class_points_idx))
            
            batch_instance_masks = []
            batch_class_ids = []
            
            if len(valid_classes) == 0:
                # No valid instances - 所有类别都被过滤掉了
                if batch_size <= 10:  # Only debug for normal batch sizes
                    print(f"Batch {b}: No valid classes found!")
                    print(f"  Total points: {pc_original.shape[0]}")
                    print(f"  min_points_per_instance: {min_points_per_instance}")
                    print(f"  Class statistics (class_id, num_points>threshold, max_prob, mean_prob):")
                    for stat in class_stats[:5]:  # Show first 5 classes
                        print(f"    Class {stat[0]}: {stat[1]} points, max_prob={stat[2]:.3f}, mean_prob={stat[3]:.3f}")
                    if len(class_stats) > 5:
                        print(f"    ... and {len(class_stats) - 5} more classes")
                instance_mask = torch.zeros((1, pc_original.shape[0]), device=device, dtype=torch.float32)
                batch_class_ids = [-1]  # Invalid
            else:
                for class_id, class_points_orig, class_points_shift, class_mask, class_points_idx in valid_classes:
                    num_class_points = class_points_orig.shape[0]
                    
                    # Determine number of clusters (seeds)
                    max_clusters_calculated = min(max_clusters_per_class, num_class_points // min_points_per_instance)
                    if max_clusters_calculated == 0:
                        continue
                    
                    num_seeds = min(max_clusters_calculated, num_class_points)
                    if num_seeds == 0:
                        continue
                    
                    # Step 1: Use soft FPS (differentiable) to select seed points in SHIFTED set
                    # Instead of hard FPS, we use distance-based soft sampling
                    # This is differentiable and allows gradients to flow through
                    seed_points_shift = InstanceToSemanticGrad._soft_fps(class_points_shift, num_seeds)  # [num_seeds, 3]
                    
                    # Step 2: Soft Ball Query (differentiable) in SHIFTED set to find neighbors
                    # Then expand in ORIGINAL set (Dual-Set method)
                    num_instances = seed_points_shift.shape[0]
                    full_mask = torch.zeros((num_instances, pc_original.shape[0]), device=device, dtype=torch.float32)
                    
                    # Soft ball query: use distance-based soft assignment (fully differentiable)
                    # Compute distances from all class points to seed points in shifted set
                    distances = torch.cdist(class_points_shift.unsqueeze(0), seed_points_shift.unsqueeze(0))[0]  # [N_class, num_seeds]
                    
                    # Soft assignment based on distance (differentiable)
                    # Use Gaussian-like weighting: exp(-d^2 / (2 * sigma^2))
                    # where sigma is related to cluster_threshold
                    sigma = cluster_threshold * 0.5  # Scale factor for soft assignment
                    soft_weights = torch.exp(-distances.pow(2) / (2 * sigma.pow(2)))  # [N_class, num_seeds]
                    
                    # Normalize to get soft assignments (each point can belong to multiple seeds)
                    soft_assignments = soft_weights / (soft_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [N_class, num_seeds]
                    
                    # Expand clusters: assign points in original set based on soft assignments
                    for k in range(num_instances):
                        # Get soft assignment weights for this seed
                        assignment_weights = soft_assignments[:, k]  # [N_class]
                        # Combine with class probability for final weights
                        final_weights = assignment_weights * class_mask[class_points_idx]
                        # Build instance mask in original set
                        full_mask[k, class_points_idx] = final_weights
                    
                    # Append the full_mask for this class (multiple instances)
                    batch_instance_masks.append(full_mask)
                    # Record class_id for each instance in this group
                    batch_class_ids.extend([class_id] * num_instances)
                
                if batch_instance_masks:
                    # Concatenate all instance masks for this batch
                    instance_mask = torch.cat(batch_instance_masks, dim=0)  # [K_total, N]
                    
                    # Sharpen with softmax
                    instance_mask = F.softmax(instance_mask * 10, dim=0)  # [K_total, N]
                else:
                    # Fallback if no instances after filtering
                    if batch_size <= 10:  # Only debug for normal batch sizes
                        print(f"Batch {b}: No instances created after filtering (valid_classes={len(valid_classes)})")
                    instance_mask = torch.zeros((1, pc_original.shape[0]), device=device, dtype=torch.float32)
                    batch_class_ids = [-1]
            
            instance_masks_list.append(instance_mask)
            instance_classes_list.append(batch_class_ids)
        
        # Debug: check list length before padding
        if len(instance_masks_list) != batch_size:
            print(f"ERROR: instance_masks_list length ({len(instance_masks_list)}) != batch_size ({batch_size})")
            print(f"  points.shape={points.shape}, semantic_logits.shape={semantic_logits.shape}, offsets.shape={offsets.shape}")
            print(f"  Loop should have run {batch_size} times, but list has {len(instance_masks_list)} elements")
            # Force correct batch size by taking only first batch_size elements
            instance_masks_list = instance_masks_list[:batch_size]
            instance_classes_list = instance_classes_list[:batch_size]
        
        # Pad instance masks to uniform size
        max_instances = max([m.shape[0] for m in instance_masks_list]) if instance_masks_list else 1
        
        padded_masks = []
        for mask in instance_masks_list:
            if mask.shape[0] < max_instances:
                padding = torch.zeros(
                    (max_instances - mask.shape[0], mask.shape[1]),
                    device=device, dtype=torch.float32
                )
                mask = torch.cat([mask, padding], dim=0)
            padded_masks.append(mask)
        instance_masks = torch.stack(padded_masks, dim=0)  # [B, K, N]
        
        # Final validation
        assert instance_masks.shape[0] == batch_size, \
            f"Final output batch_size mismatch: expected {batch_size}, got {instance_masks.shape[0]}"
        
        # Pad instance classes similarly (use -1 for padding)
        max_k = max_instances  # Same as above
        instance_classes_padded = []
        for b_class_ids in instance_classes_list:
            padded_ids = b_class_ids + [-1] * (max_k - len(b_class_ids))
            instance_classes_padded.append(torch.tensor(padded_ids, device=device, dtype=torch.long))
        instance_class = torch.stack(instance_classes_padded)  # [B, K]
        
        # Save for backward: need to save all intermediate results for gradient computation
        ctx.save_for_backward(
            semantic_probs,  # [B, N, C]
            instance_class,  # [B, K]
            points,  # [B, N, 3] - original points
            offsets,  # [B, N, 3] - offsets
            instance_masks  # [B, K, N] - for computing gradients w.r.t. offsets
        )
        ctx.num_classes = num_classes
        ctx.cluster_threshold = cluster_threshold
        ctx.min_points_per_instance = min_points_per_instance
        ctx.max_clusters_per_class = max_clusters_per_class
        ctx.class_prob_threshold = class_prob_threshold
        
        return instance_masks
    
    @staticmethod
    def backward(ctx, grad_instance_masks):
        semantic_probs, instance_class, points, offsets, instance_masks = ctx.saved_tensors
        B, K, N = grad_instance_masks.shape
        C = semantic_probs.shape[-1]
        cluster_threshold = ctx.cluster_threshold
        
        # Validate shapes
        if B != semantic_probs.shape[0]:
            B = semantic_probs.shape[0]
            grad_instance_masks = grad_instance_masks[:B]
            instance_class = instance_class[:B]
            points = points[:B]
            offsets = offsets[:B]
        
        # Initialize gradients
        grad_semantic_probs = torch.zeros_like(semantic_probs)  # [B, N, C]
        grad_offsets = torch.zeros_like(offsets)  # [B, N, 3]
        grad_points = torch.zeros_like(points)  # [B, N, 3]
        
        # Reconstruct shifted points for gradient computation
        shifted_points = points + offsets  # [B, N, 3]
        
        for b in range(B):
            pc_original = points[b]  # [N, 3]
            pc_shifted = shifted_points[b]  # [N, 3]
            semantic_b = semantic_probs[b]  # [N, C]
            grad_mask_b = grad_instance_masks[b]  # [K, N]
            
            # Process each instance
            for k in range(K):
                if k >= instance_class.shape[1]:
                    continue
                c = instance_class[b, k].item()
                if c < 0:  # Skip padding
                    continue
                
                # Get class mask for this class
                class_mask = semantic_b[:, c]  # [N]
                class_points_idx = torch.where(class_mask > ctx.class_prob_threshold)[0]
                
                if len(class_points_idx) < ctx.min_points_per_instance:
                    continue
                
                class_points_orig = pc_original[class_points_idx]  # [N_class, 3]
                class_points_shift = pc_shifted[class_points_idx]  # [N_class, 3]
                num_class_points = class_points_orig.shape[0]
                
                # Reconstruct soft FPS seeds (simplified approximation for gradient)
                max_clusters = min(ctx.max_clusters_per_class, num_class_points // ctx.min_points_per_instance)
                num_seeds = min(max_clusters, num_class_points)
                
                if num_seeds == 0:
                    continue
                
                # Approximate seed points using soft FPS (same as forward)
                seed_points_shift = InstanceToSemanticGrad._soft_fps(class_points_shift, num_seeds)  # [num_seeds, 3]
                
                # Recompute distances and soft assignments (same as forward)
                distances = torch.cdist(class_points_shift.unsqueeze(0), seed_points_shift.unsqueeze(0))[0]  # [N_class, num_seeds]
                sigma = cluster_threshold * 0.5
                soft_weights = torch.exp(-distances.pow(2) / (2 * sigma.pow(2)))  # [N_class, num_seeds]
                soft_assignments = soft_weights / (soft_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [N_class, num_seeds]
                
                # Get gradient for this instance
                grad_mask_k = grad_mask_b[k, class_points_idx]  # [N_class]
                
                # Determine which seed this instance corresponds to
                seed_idx = k % num_seeds
                
                # Gradient w.r.t. semantic probabilities
                # The mask is: assignment_weights * class_mask
                # So grad w.r.t. class_mask = grad_mask * assignment_weights
                assignment_weights_k = soft_assignments[:, seed_idx]  # [N_class]
                grad_class_mask = grad_mask_k * assignment_weights_k  # [N_class]
                grad_semantic_probs[b, class_points_idx, c] += grad_class_mask
                
                # Gradient w.r.t. offsets (through shifted points)
                # The soft assignment depends on distances in shifted space
                # grad_offset = grad_mask * d(soft_assignment) / d(offset)
                # d(soft_assignment) / d(offset) = d(soft_assignment) / d(distance) * d(distance) / d(shifted_point) * d(shifted_point) / d(offset)
                # d(shifted_point) / d(offset) = 1
                
                # Compute gradient of soft_weights w.r.t. distances
                d_soft_weights_d_dist = -soft_weights * distances / (sigma.pow(2) + 1e-8)  # [N_class, num_seeds]
                
                # Compute gradient of distances w.r.t. shifted points
                # distance = ||class_point_shift - seed_point_shift||
                # d(distance) / d(class_point_shift) = (class_point_shift - seed_point_shift) / distance
                seed_k = seed_points_shift[seed_idx]  # [3]
                diff = class_points_shift - seed_k.unsqueeze(0)  # [N_class, 3]
                distances_k = distances[:, seed_idx]  # [N_class]
                d_dist_d_shift = diff / (distances_k.unsqueeze(-1) + 1e-8)  # [N_class, 3]
                
                # Chain rule: grad_offset = grad_mask * d_soft_weights_d_dist * d_dist_d_shift
                grad_soft_weight_k = d_soft_weights_d_dist[:, seed_idx]  # [N_class]
                grad_shift_k = grad_mask_k.unsqueeze(-1) * grad_soft_weight_k.unsqueeze(-1) * d_dist_d_shift  # [N_class, 3]
                
                # Since shifted_point = point + offset, grad_offset = grad_shift
                grad_offsets[b, class_points_idx] += grad_shift_k
                
                # Gradient w.r.t. points (through original points in expansion)
                # The mask is applied in original space, so grad_points comes from grad_mask
                # Also account for the fact that shifted points affect seed selection
                # For simplicity, we approximate: grad_points ≈ grad_mask * assignment_weights (in original space)
                grad_points[b, class_points_idx] += grad_mask_k.unsqueeze(-1) * assignment_weights_k.unsqueeze(-1) * 0.1  # Small contribution
        
        # Softmax backward: propagate grad_semantic_probs to semantic_logits
        output_grad = grad_semantic_probs  # [B, N, C]
        softmax_grad = semantic_probs     # [B, N, C]
        p_dot_gp = torch.sum(softmax_grad * output_grad, dim=-1, keepdim=True)  # [B, N, 1]
        grad_logits = output_grad * softmax_grad - softmax_grad * p_dot_gp  # [B, N, C]
        
        # Return gradients for all inputs
        return grad_logits, grad_points, grad_offsets, None, None, None, None, None


class OffsetPredictor(nn.Module):
    """
    Predicts offset vectors from each point to its instance center.
    
    The offset is used to shift points towards their instance centers,
    making clustering easier.
    """
    
    def __init__(self, feat_dim=256, hidden_dim=128):
        """
        Initialize offset predictor.
        
        Args:
            feat_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super(OffsetPredictor, self).__init__()
        
        self.offset_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # 3D offset vector
        )
        
    def forward(self, features):
        """
        Predict offset vectors.
        
        Args:
            features (torch.Tensor): Point features [N, feat_dim]
            
        Returns:
            torch.Tensor: Offset vectors [N, 3]
        """
        return self.offset_head(features)


class OverPointsLayerNorm(nn.Module):
    """
    LayerNorm that normalizes across points for each feature dimension.
    
    For input [N, C] or [B, N, C]:
    - Normalizes across the point dimension (N) for each feature (C)
    - This makes each feature dimension have mean=0, std=1 across all points
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # We can't use nn.LayerNorm because the number of points (N) is variable
        # Instead, we'll manually compute normalization
        # num_features is the number of feature dimensions (classes)
        self.num_features = num_features
        self.eps = eps
        # Learnable parameters: gamma and beta for each feature dimension
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Handle different input shapes
        # Normalize across points (N dimension) for each feature (C dimension)
        if x.dim() == 2:
            # x: [N, C] - single batch, N points, C features (classes)
            # Normalize across points (dim 0) for each feature (dim 1)
            # Transpose to [C, N], normalize over N, then transpose back
            x_t = x.transpose(0, 1)  # [C, N]
            # Manual LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
            mean = x_t.mean(dim=-1, keepdim=True)  # [C, 1]
            var = x_t.var(dim=-1, keepdim=True, unbiased=False)  # [C, 1]
            x_norm_t = (x_t - mean) / torch.sqrt(var + self.eps)  # [C, N]
            # Apply learnable scale and shift
            x_norm_t = x_norm_t * self.gamma.unsqueeze(-1) + self.beta.unsqueeze(-1)  # [C, N]
            x_norm = x_norm_t.transpose(0, 1)  # back to [N, C]
            return x_norm
        elif x.dim() == 3:
            # x: [B, N, C] - batched
            # Transpose to [B, C, N], normalize over N, then transpose back
            x_t = x.transpose(1, 2)  # [B, C, N]
            # Manual LayerNorm
            mean = x_t.mean(dim=-1, keepdim=True)  # [B, C, 1]
            var = x_t.var(dim=-1, keepdim=True, unbiased=False)  # [B, C, 1]
            x_norm_t = (x_t - mean) / torch.sqrt(var + self.eps)  # [B, C, N]
            # Apply learnable scale and shift
            x_norm_t = x_norm_t * self.gamma.unsqueeze(0).unsqueeze(-1) + self.beta.unsqueeze(0).unsqueeze(-1)  # [B, C, N]
            x_norm = x_norm_t.transpose(1, 2)  # back to [B, N, C]
            return x_norm
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected 2D [N, C] or 3D [B, N, C]")

class PointGroup(nn.Module):
    """
    PointGroup instance segmentation module for Point Transformer V3.
    
    This module combines:
    1. Semantic segmentation to identify object classes
    2. Offset prediction to predict point-to-instance-center offsets
    3. Clustering to group points into instances
    """
    
    def __init__(
        self,
        num_classes=20,
        in_channels=3,
        feat_dim=256,
        grid_size=0.01,
        enable_flash=True,
        enable_rpe=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        cluster_threshold=0.03,
        min_points_per_instance=50,
        max_clusters_per_class=10,
        class_prob_threshold=0.05,
        semantic_normalize=True,
        semantic_temperature=1.0,
        **kwargs
    ):
        """
        Initialize PointGroup module.
        
        Args:
            num_classes (int): Number of semantic classes
            in_channels (int): Input feature channels (default 3 for xyz)
            feat_dim (int): Feature dimension
            grid_size (float): Grid size for voxelization
            enable_flash (bool): Enable flash attention
            enable_rpe (bool): Enable relative position encoding
            enc_depths (tuple): Encoder depths for each stage
            enc_channels (tuple): Encoder channels for each stage
            dec_depths (tuple): Decoder depths for each stage
            dec_channels (tuple): Decoder channels for each stage
            cluster_threshold (float): Distance threshold for clustering
            min_points_per_instance (int): Minimum points per instance
            max_clusters_per_class (int): Maximum number of clusters per semantic class (default: 10)
            class_prob_threshold (float): Probability threshold for class points (default: 0.05)
                Points with class probability > this threshold are considered for clustering
            semantic_normalize (bool): If True, normalize semantic logits to balance class predictions (default: True)
            semantic_temperature (float): Temperature scaling for semantic logits (default: 1.0)
                Higher temperature makes predictions more uniform, lower makes them more confident
        """
        super(PointGroup, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.grid_size = grid_size
        self.cluster_threshold = cluster_threshold
        self.min_points_per_instance = min_points_per_instance
        self.max_clusters_per_class = max_clusters_per_class
        self.class_prob_threshold = class_prob_threshold
        self.semantic_normalize = semantic_normalize
        self.semantic_temperature = semantic_temperature
        
        # Initialize PTv3 backbone
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            enable_flash=enable_flash,
            enable_rpe=enable_rpe,
            cls_mode=False,
            **kwargs
        )
        
        # Get output feature dimension from decoder
        self.backbone_feat_dim = dec_channels[0] if len(dec_channels) > 0 else enc_channels[-1]
        
        # Semantic segmentation head
        self.semantic_head = nn.Sequential(
            nn.Linear(self.backbone_feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
        )
        
        # LayerNorm for semantic logits to normalize across points for each class
        # This normalizes each class's logits across all points, making predictions more balanced

        self.semantic_logits_norm = None
        
        # Offset prediction head
        self.offset_predictor = OffsetPredictor(
            feat_dim=self.backbone_feat_dim,
            hidden_dim=feat_dim // 2
        )
        
    def forward(self, inputs, differentiable_clustering=True):
        """
        Forward pass for PointGroup instance segmentation.
        
        Args:
            inputs: Can be:
                - torch.Tensor: Point cloud [N, 3] or [B, N, 3]
                - dict: Input dictionary containing:
                    - point_cloud_first (torch.Tensor): Point cloud [B, N, 3] or [N, 3]
                    - points (torch.Tensor): Alternative key for point cloud
                    - feat (torch.Tensor, optional): Point features
                    - offset (torch.Tensor, optional): Batch offsets
                    - batch (torch.Tensor, optional): Batch indices
            differentiable_clustering (bool): If True, use differentiable clustering (default: False)
                - True: Clustering is differentiable, gradients can flow through
                - False: Uses scipy hierarchical clustering (faster, but not differentiable)
        
        Returns:
            torch.Tensor: Instance masks [B, K, N]
                - If differentiable_clustering=True: Soft masks (differentiable)
                - If differentiable_clustering=False: Hard masks (0 or 1)
        """
        # Extract point cloud
        if isinstance(inputs, dict):
            pc = inputs.get('point_cloud_first', inputs.get('points', None))
            if pc is None:
                raise ValueError("Input must contain 'point_cloud_first' or 'points'")
        else:
            pc = inputs
        
        # Handle different input formats
        original_shape = pc.shape
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)  # [1, N, 3]
            batch_size = 1
            N = pc.shape[1]
            is_batched = False
        elif pc.dim() == 3:
            batch_size = pc.shape[0]
            N = pc.shape[1]
            is_batched = True
        else:
            raise ValueError(f"Unexpected point cloud shape: {pc.shape}")
        
        # Flatten for processing
        pc_flat = pc.view(-1, 3)  # [B*N, 3]
        device = pc_flat.device
        
        # Prepare data dict for PTv3
        data_dict = {
            'coord': pc_flat,
            'feat': pc_flat,
            'grid_size': self.grid_size,
        }
        
        # Handle batch information
        if isinstance(inputs, dict):
            if 'offset' in inputs:
                data_dict['offset'] = inputs['offset']
            elif 'batch' in inputs:
                data_dict['batch'] = inputs['batch']
            else:
                if is_batched:
                    offsets = torch.cumsum(
                        torch.tensor([N] * batch_size, device=device, dtype=torch.long), 
                        dim=0
                    )
                    data_dict['offset'] = offsets
                else:
                    data_dict['offset'] = torch.tensor([N], device=device, dtype=torch.long)
        else:
            if is_batched:
                offsets = torch.cumsum(
                    torch.tensor([N] * batch_size, device=device, dtype=torch.long),
                    dim=0
                )
                data_dict['offset'] = offsets
            else:
                data_dict['offset'] = torch.tensor([N], device=device, dtype=torch.long)
        
        # Add features if provided
        if isinstance(inputs, dict) and 'feat' in inputs:
            feat = inputs['feat']
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            feat = feat.view(-1, feat.shape[-1])
            data_dict['feat'] = feat
        
        # Forward through PTv3 backbone
        point = self.backbone(data_dict)
        features = point.feat  # [N_total, C]
        
        # Predict semantic classes
        semantic_logits = self.semantic_head(features)  # [N_total, num_classes]
        
        
        # Predict offsets
        offsets = self.offset_predictor(features)  # [N_total, 3]
        
        # Reshape outputs
        if is_batched:
            semantic_logits = semantic_logits.view(batch_size, N, self.num_classes)
            offsets = offsets.view(batch_size, N, 3)
        else:
            semantic_logits = semantic_logits.unsqueeze(0)  # [1, N, num_classes]
            offsets = offsets.unsqueeze(0)  # [1, N, 3]
        
        result = {
            'semantic_logits': semantic_logits,
            'offsets': offsets
        }
        
        # Debug: check input shapes before clustering
        if pc.shape[0] != semantic_logits.shape[0] or pc.shape[0] != offsets.shape[0]:
            print(f"ERROR before cluster_points: pc.shape={pc.shape}, semantic_logits.shape={semantic_logits.shape}, offsets.shape={offsets.shape}")
        
        clusters, instance_masks = self.cluster_points(
            pc, semantic_logits, offsets, data_dict.get('offset', None),
            differentiable=differentiable_clustering
        )
        result['clusters'] = clusters
        result['instance_masks'] = instance_masks
        
        # Debug: check for unusual shapes
        if instance_masks.shape[1] == 1 and instance_masks.shape[0] > 1:
            print(f"WARNING: All batches have only 1 instance! batch_size={instance_masks.shape[0]}, "
                  f"pc.shape={pc.shape}, semantic_logits.shape={semantic_logits.shape}")
        if instance_masks.shape[0] != batch_size:
            print(f"WARNING: Batch size mismatch! Expected {batch_size}, got {instance_masks.shape[0]}, "
                  f"pc.shape={pc.shape}, semantic_logits.shape={semantic_logits.shape}, offsets.shape={offsets.shape}")
        
        print(f"instance_masks: {instance_masks.shape}")
        return instance_masks
    
    def cluster_points(self, points, semantic_logits, offsets, batch_offsets=None, differentiable=False):
        """
        Cluster points into instances using semantic labels and offsets.
        
        Args:
            points (torch.Tensor): Point coordinates [B, N, 3]
            semantic_logits (torch.Tensor): Semantic logits [B, N, num_classes]
            offsets (torch.Tensor): Offset vectors [B, N, 3]
            batch_offsets (torch.Tensor, optional): Batch offsets [B+1]
            differentiable (bool): If True, use differentiable clustering (default: False)
        
        Returns:
            tuple: (clusters, instance_masks)
                - clusters: List of cluster assignments for each batch (or None if differentiable)
                - instance_masks: Instance masks [B, K, N]
        """
        if differentiable:
            return self._differentiable_clustering(points, semantic_logits, offsets, batch_offsets)
        else:
            return self._non_differentiable_clustering(points, semantic_logits, offsets, batch_offsets)
    
    def _non_differentiable_clustering(self, points, semantic_logits, offsets, batch_offsets=None):
        """
        Non-differentiable clustering using scipy hierarchical clustering.
        """
        batch_size = points.shape[0]
        device = points.device
        
        # Get semantic predictions
        semantic_pred = torch.argmax(semantic_logits, dim=-1)  # [B, N]
        
        clusters = []
        instance_masks_list = []
        
        for b in range(batch_size):
            pc_b = points[b]  # [N, 3]
            offset_b = offsets[b]  # [N, 3]
            semantic_b = semantic_pred[b]  # [N]
            
            # Shift points by predicted offsets
            shifted_points = pc_b + offset_b  # [N, 3]
            
            # Cluster points using hierarchical clustering
            # Detach from computation graph before converting to numpy
            cluster_assignments = self._hierarchical_clustering(
                shifted_points.detach().cpu().numpy(),
                semantic_b.detach().cpu().numpy(),
                threshold=self.cluster_threshold
            )
            
            # Filter small clusters
            unique_clusters = np.unique(cluster_assignments)
            valid_clusters = []
            for cluster_id in unique_clusters:
                mask = cluster_assignments == cluster_id
                if np.sum(mask) >= self.min_points_per_instance:
                    valid_clusters.append(cluster_id)
            
            # Create instance masks
            num_instances = len(valid_clusters)
            if num_instances == 0:
                # No valid instances, return empty mask
                instance_mask = torch.zeros((1, pc_b.shape[0]), device=device, dtype=torch.float32)
                instance_masks_list.append(instance_mask)
                clusters.append(cluster_assignments)
                continue
            
            instance_mask = torch.zeros((num_instances, pc_b.shape[0]), device=device, dtype=torch.float32)
            for i, cluster_id in enumerate(valid_clusters):
                mask = cluster_assignments == cluster_id
                instance_mask[i, mask] = 1.0
            
            instance_masks_list.append(instance_mask)
            clusters.append(cluster_assignments)
        
        # Pad instance masks to same size
        max_instances = max([m.shape[0] for m in instance_masks_list]) if instance_masks_list else 1
        padded_masks = []
        for mask in instance_masks_list:
            if mask.shape[0] < max_instances:
                padding = torch.zeros(
                    (max_instances - mask.shape[0], mask.shape[1]),
                    device=device,
                    dtype=torch.float32
                )
                mask = torch.cat([mask, padding], dim=0)
            padded_masks.append(mask)
        
        instance_masks = torch.stack(padded_masks, dim=0)  # [B, K, N]
        
        return clusters, instance_masks
    
    # Updated wrapper function (integrate into your class method)
    def _differentiable_clustering(self, points, semantic_logits, offsets, batch_offsets=None):
        """
        Wrapper to use the custom autograd function.
        Returns: (None, instance_masks)
        """
        # Debug: check input shapes
        if points.shape[0] != semantic_logits.shape[0]:
            print(f"ERROR in _differentiable_clustering: points.shape={points.shape}, semantic_logits.shape={semantic_logits.shape}, offsets.shape={offsets.shape}")
        
        instance_masks = InstanceToSemanticGrad.apply(
            semantic_logits, points, offsets,
            self.num_classes, self.min_points_per_instance, self.cluster_threshold, self.max_clusters_per_class, self.class_prob_threshold
        )
        return None, instance_masks
    def _hierarchical_clustering(self, points, semantic_labels, threshold=0.03):
        """
        Perform hierarchical clustering on points.
        
        Args:
            points (np.ndarray): Point coordinates [N, 3]
            semantic_labels (np.ndarray): Semantic labels [N]
            threshold (float): Distance threshold for clustering
        
        Returns:
            np.ndarray: Cluster assignments [N]
        """
        N = points.shape[0]
        if N == 0:
            return np.array([])
        if N == 1:
            return np.array([0])
        
        # Only cluster points with same semantic label
        # Group by semantic class first
        unique_semantic = np.unique(semantic_labels)
        cluster_assignments = np.zeros(N, dtype=np.int32)
        current_cluster_id = 0
        
        for sem_id in unique_semantic:
            if sem_id == 0:  # Skip background class
                continue
            
            mask = semantic_labels == sem_id
            if np.sum(mask) < self.min_points_per_instance:
                continue
            
            semantic_points = points[mask]
            
            if semantic_points.shape[0] <= 1:
                continue
            
            # Compute pairwise distances
            try:
                distances = pdist(semantic_points)
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(distances, method='average')
                clusters = fcluster(linkage_matrix, threshold, criterion='distance')
                
                # Map cluster IDs to global IDs
                unique_clusters = np.unique(clusters)
                for local_cluster_id in unique_clusters:
                    local_mask = clusters == local_cluster_id
                    global_indices = np.where(mask)[0][local_mask]
                    cluster_assignments[global_indices] = current_cluster_id
                    current_cluster_id += 1
            except Exception as e:
                # If clustering fails, assign all points to one cluster
                cluster_assignments[mask] = current_cluster_id
                current_cluster_id += 1
        
        return cluster_assignments
    
    def load_pretrained_from_hub(self, pretrained_name=None):
        """
        Load pretrained backbone weights from HuggingFace Hub.
        
        Args:
            pretrained_name (str): Name of pretrained model
        """
        if pretrained_name is None:
            raise ValueError("Pretrained name is required")
        self.backbone.from_pretrained(f"facebook/sonota/{pretrained_name}")
        return self

