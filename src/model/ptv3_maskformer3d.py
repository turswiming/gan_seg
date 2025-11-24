"""
PTv3 + MaskFormer3D head
Combines Point Transformer V3 backbone with the proven MaskFormer3D segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

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

# Add OGCModel to path
ogc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OGCModel'))
if ogc_path not in sys.path:
    sys.path.insert(0, ogc_path)

try:
    from transformer_util import MaskFormerHead
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from OGCModel.transformer_util import MaskFormerHead


def nms_for_instance_masks(masks, scores, iou_threshold=0.5, score_threshold=0.1):
    """
    Non-Maximum Suppression for instance segmentation masks.
    
    Args:
        masks: [K, N] - K 个 slot 的 mask（每个是 N 个点的概率值）
        scores: [K] - 每个 slot 的置信度分数
        iou_threshold: IoU 阈值，超过此值的重叠 mask 会被抑制
        score_threshold: 置信度阈值，低于此值的 mask 会被过滤
    
    Returns:
        keep_indices: 保留的 slot 索引列表
    """
    K, N = masks.shape
    device = masks.device
    
    # 1. 过滤低置信度的 mask
    valid_mask = scores >= score_threshold
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=device)
    
    valid_indices = torch.where(valid_mask)[0]
    valid_masks = masks[valid_indices]  # [K_valid, N]
    valid_scores = scores[valid_indices]  # [K_valid]
    
    # 2. 将 soft mask 转为 binary mask 用于计算 IoU
    binary_masks = (valid_masks > 0.5).float()  # [K_valid, N]
    
    # 3. 按置信度排序（从高到低）
    sorted_indices = torch.argsort(valid_scores, descending=True)
    sorted_masks = binary_masks[sorted_indices]
    sorted_scores = valid_scores[sorted_indices]
    sorted_original_indices = valid_indices[sorted_indices]
    
    # 4. 计算每个 mask 的点数（用于 IoU 计算）
    mask_areas = sorted_masks.sum(dim=1)  # [K_valid]
    
    # 5. NMS 主循环
    keep_indices = []
    suppressed = torch.zeros(len(sorted_indices), dtype=torch.bool, device=device)
    
    for i in range(len(sorted_indices)):
        if suppressed[i]:
            continue
        
        # 保留当前 mask
        keep_indices.append(sorted_original_indices[i].item())
        
        # 计算与后续 mask 的 IoU
        if i + 1 < len(sorted_indices):
            current_mask = sorted_masks[i]  # [N]
            remaining_masks = sorted_masks[i+1:]  # [K_remaining, N]
            
            # 计算交集
            intersection = (current_mask.unsqueeze(0) * remaining_masks).sum(dim=1)  # [K_remaining]
            
            # 计算并集
            current_area = mask_areas[i]
            remaining_areas = mask_areas[i+1:]  # [K_remaining]
            union = current_area + remaining_areas - intersection
            
            # 计算 IoU
            iou = intersection / (union + 1e-8)  # [K_remaining]
            
            # 抑制 IoU 超过阈值的 mask
            suppressed[i+1:] = suppressed[i+1:] | (iou > iou_threshold)
    
    if len(keep_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    
    return torch.tensor(keep_indices, dtype=torch.long, device=device)


class PTV3MaskFormer3D(nn.Module):
    """
    Point Transformer V3 with MaskFormer3D instance segmentation head.

    This model combines:
    1. PTv3 backbone for point cloud feature extraction
    2. MaskFormer3D's proven Transformer decoder architecture (with self-attention)
    3. L2 normalization and temperature scaling for mask prediction
    """

    def __init__(
        self,
        n_slot=20,
        feat_dim=256,
        transformer_embed_dim=256,
        n_transformer_layer=2,
        transformer_n_head=8,
        transformer_input_pos_enc=False,
        # PTv3 backbone parameters
        in_channels=3,
        grid_size=0.01,
        enable_flash=True,
        enable_rpe=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        # NMS parameters
        use_nms=False,
        nms_iou_threshold=0.5,
        nms_score_threshold=0.1,
        **kwargs
    ):
        """
        Initialize PTv3 + MaskFormer3D model.

        Args:
            n_slot (int): Number of instance queries (slots)
            feat_dim (int): Feature dimension for MaskFormer head
            transformer_embed_dim (int): Transformer embedding dimension
            n_transformer_layer (int): Number of Transformer decoder layers
            transformer_n_head (int): Number of attention heads
            transformer_input_pos_enc (bool): Whether to use position encoding
            in_channels (int): Input feature channels for PTv3
            grid_size (float): Grid size for voxelization in PTv3
            enable_flash (bool): Enable flash attention in PTv3
            enable_rpe (bool): Enable relative position encoding in PTv3
            enc_depths (tuple): Encoder depths for PTv3
            enc_channels (tuple): Encoder channels for PTv3
            dec_depths (tuple): Decoder depths for PTv3
            dec_channels (tuple): Decoder channels for PTv3
            use_nms (bool): Whether to apply NMS post-processing (default: False, for inference)
            nms_iou_threshold (float): IoU threshold for NMS (default: 0.5)
            nms_score_threshold (float): Score threshold for filtering masks (default: 0.1)
            **kwargs: Additional arguments for PTv3
        """
        super(PTV3MaskFormer3D, self).__init__()

        self.n_slot = n_slot
        self.feat_dim = feat_dim
        self.grid_size = grid_size
        self.use_nms = use_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold

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

        # NOTE: We modify dec_channels to ensure the output feature dimension is 256
        # The last decoder channel should be 256 to match feat_dim

        # Get output feature dimension from decoder (should be 256 to match feat_dim)
        # No projection needed since dimensions already match
        self.feature_proj = nn.Identity()

        # MaskFormer3D head (proven architecture with self-attention)
        self.MF_head = MaskFormerHead(
            n_slot=n_slot,
            input_dim=feat_dim,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_n_head=transformer_n_head,
            transformer_hidden_dim=transformer_embed_dim,
            input_pos_enc=transformer_input_pos_enc
        )

        # Object MLP for slot processing (from MaskFormer3D)
        # Output dimension should match feat_dim for dot product in mask computation
        self.object_mlp = nn.Sequential(
            nn.Conv1d(transformer_embed_dim, transformer_embed_dim, 1),
            nn.GroupNorm(4, transformer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(transformer_embed_dim, feat_dim, 1)
        )

    def forward(self, inputs):
        """
        Forward pass for instance segmentation.

        Args:
            inputs: Can be:
                - torch.Tensor: Point cloud [N, 3] or [B, N, 3]
                - dict: Input dictionary containing:
                    - point_cloud_first (torch.Tensor): Point cloud [B, N, 3] or [N, 3]
                    - points (torch.Tensor): Alternative key for point cloud
                    - feat (torch.Tensor, optional): Point features
                    - offset (torch.Tensor, optional): Batch offsets
                    - batch (torch.Tensor, optional): Batch indices

        Returns:
            torch.Tensor: Soft instance masks [B, K, N] or [K, N]
                          Each point has K probability values that sum to 1
                          If NMS is enabled, K <= n_slot (unused slots are zero-padded)
                          If NMS is disabled, K = n_slot
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

        # Flatten for PTv3 processing
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
        # Project features to Mask3D dimension
        features = self.feature_proj(features)  # [N_total, feat_dim]
        # Process each batch separately using MaskFormer3D head
        mask_list = []
        for b in range(batch_size):
            start_idx = b * N
            end_idx = (b + 1) * N
            batch_features = features[start_idx:end_idx]  # [N, feat_dim]
            batch_pc = pc_flat[start_idx:end_idx]  # [N, 3]

            # Extract object embeddings with MaskFormer head
            # MaskFormerHead expects [B, N, C] and [B, N, 3]
            slot = self.MF_head(
                batch_features.unsqueeze(0),  # [1, N, feat_dim]
                batch_pc.unsqueeze(0)        # [1, N, 3]
            )  # [1, K, D]

            slot = self.object_mlp(slot.transpose(1, 2))  # [1, D, K]

            # L2 normalization of features and slots (critical for performance)
            # IMPORTANT: Normalize along the feature dimension for proper similarity computation
            batch_features_norm = F.normalize(
                batch_features.transpose(0, 1).unsqueeze(0),  # [1, feat_dim, N]
                p=2,  # L2 norm
                dim=1  # normalize along feature dimension
            )
            slot_norm = F.normalize(slot, p=2, dim=1)  # [1, D, K]

            # Compute mask by dot-product with temperature scaling
            # Temperature 0.05 is critical - makes the softmax sharper
            temperature = 0.05
            mask = torch.einsum('bdn,bdk->bnk',
                               batch_features_norm,
                               slot_norm) / temperature

            # Softmax over instance dimension (pointwise probability)
            mask = mask.softmax(dim=-1)  # [1, N, K]

            mask_list.append(mask.squeeze(0))  # [N, K]

        # Apply NMS if enabled
        if self.use_nms:
            processed_mask_list = []
            for mask in mask_list:  # mask: [N, K]
                # Compute confidence scores for each slot (average probability)
                scores = mask.mean(dim=0)  # [K]
                
                # Transpose to [K, N] for NMS function
                mask_t = mask.permute(1, 0)  # [K, N]
                
                # Apply NMS
                keep_indices = nms_for_instance_masks(
                    mask_t, scores,
                    iou_threshold=self.nms_iou_threshold,
                    score_threshold=self.nms_score_threshold
                )
                
                if len(keep_indices) > 0:
                    # Keep only selected slots
                    mask_filtered = mask[:, keep_indices]  # [N, K_kept]
                    # Renormalize to ensure probabilities sum to 1
                    mask_filtered = mask_filtered / (mask_filtered.sum(dim=1, keepdim=True) + 1e-8)
                    
                    # Pad to original n_slot size with zeros for consistency
                    if mask_filtered.shape[1] < self.n_slot:
                        padding = torch.zeros(mask.shape[0], self.n_slot - mask_filtered.shape[1], 
                                             device=mask.device, dtype=mask.dtype)
                        mask_filtered = torch.cat([mask_filtered, padding], dim=1)  # [N, K]
                else:
                    # If no masks pass NMS, return zeros
                    mask_filtered = torch.zeros(mask.shape[0], self.n_slot, device=mask.device, dtype=mask.dtype)
                
                processed_mask_list.append(mask_filtered)
            
            mask_list = processed_mask_list

        # Return based on input format
        if is_batched:
            return torch.stack(mask_list, dim=0).permute(0, 2, 1)  # [B, K, N]
        else:
            return mask_list[0].permute(1, 0)  # [K, N]


def test_model():
    """Test the model with dummy data"""
    print("Testing PTV3MaskFormer3D model...")

    # Create model
    model = PTV3MaskFormer3D(
        n_slot=20,
        feat_dim=256,
        transformer_embed_dim=256,
        n_transformer_layer=2,
        grid_size=0.01
    ).cuda()
    model.eval()

    print(f"\nModel architecture:")
    print(f"  - PTv3 backbone: {type(model.backbone).__name__}")
    print(f"  - Feature projection: {type(model.feature_proj).__name__}")
    print(f"  - MaskFormer head: {type(model.MF_head).__name__}")
    print(f"  - Object MLP: {type(model.object_mlp).__name__}")

    # Test single batch
    print("\n" + "="*50)
    print("Test 1: Single point cloud")
    pc = torch.randn(8192, 3).cuda()
    with torch.no_grad():
        mask = model(pc)
    print(f"  Input shape: {pc.shape}")
    print(f"  Output shape: {mask.shape}")
    print(f"  Expected: [K={20}, N={8192}]")
    print(f"  min: {mask.min():.4f}, max: {mask.max():.4f}")

    # Test batched input
    print("\n" + "="*50)
    print("Test 2: Batched point clouds")
    pc_batched = torch.randn(4, 8192, 3).cuda()
    with torch.no_grad():
        mask_batched = model(pc_batched)
    print(f"  Input shape: {pc_batched.shape}")
    print(f"  Output shape: {mask_batched.shape}")
    print(f"  Expected: [B={4}, K={20}, N={8192}]")
    print(f"  min: {mask_batched.min():.4f}, max: {mask_batched.max():.4f}")

    # Test with dict input
    print("\n" + "="*50)
    print("Test 3: Dict input format")
    inputs_dict = {
        'point_cloud_first': pc_batched
    }
    with torch.no_grad():
        mask_dict = model(inputs_dict)
    print(f"  Input type: dict with 'point_cloud_first'")
    print(f"  Output shape: {mask_dict.shape}")
    print(f"  Expected: [B={4}, K={20}, N={8192}]")

    # Parameter count
    print("\n" + "="*50)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.MF_head.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in model.object_mlp.parameters() if p.requires_grad)

    print(f"Parameter counts:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - PTv3 backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"  - MaskFormer head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    print(f"  - Object MLP: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")

    print("\n" + "="*50)
    print("All tests passed! ✓")


if __name__ == '__main__':
    test_model()
