"""
PTv3 + SegNet_AV2 segmentation head
Combines Point Transformer V3 backbone with the SegNet_AV2 segmentation architecture
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


class PTV3SegNetAV2(nn.Module):
    """
    Point Transformer V3 with SegNet_AV2 segmentation head.

    This model combines:
    1. PTv3 backbone for point cloud feature extraction
    2. SegNet_AV2's proven segmentation architecture with PointNet++ style processing
    3. MaskFormer head for instance segmentation

    Reference: Uses SegNet_AV2 head from OGCModel which achieved strong performance
    """

    def __init__(
        self,
        n_slot=20,
        feat_dim=64,
        transformer_embed_dim=64,
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
        **kwargs
    ):
        """
        Initialize PTv3 + SegNet_AV2 model.

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
            **kwargs: Additional arguments for PTv3
        """
        super(PTV3SegNetAV2, self).__init__()

        self.n_slot = n_slot
        self.feat_dim = feat_dim
        self.grid_size = grid_size

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

        # No projection needed since dimensions already match
        self.feature_proj = nn.Identity()

        # MaskFormer head from SegNet_AV2
        self.MF_head = MaskFormerHead(
            n_slot=n_slot,
            input_dim=feat_dim,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_n_head=transformer_n_head,
            transformer_hidden_dim=transformer_embed_dim,
            input_pos_enc=transformer_input_pos_enc
        )

        # Object MLP from SegNet_AV2 (output dim 64)
        self.object_mlp = nn.Sequential(
            nn.Conv1d(transformer_embed_dim, transformer_embed_dim, 1),
            nn.GroupNorm(4, transformer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(transformer_embed_dim, 64, 1)
        )

        # Projection layer to match mask computation dimensions
        self.mask_proj = nn.Conv1d(feat_dim, 64, 1)

    def forward(self, pc, point_feats):
        """
        Forward pass for instance segmentation.

        Args:
            pc: (B, N, 3) torch.Tensor - Point cloud coordinates
            point_feats: (B, N, 3) torch.Tensor - Point features (RGB or XYZ)

        Returns:
            torch.Tensor: Soft instance masks [B, N, n_slot]
                          Each point has n_slot probability values that sum to 1
        """
        batch_size = pc.shape[0]
        N = pc.shape[1]

        # Flatten for PTv3 processing
        pc_flat = pc.view(-1, 3)  # [B*N, 3]
        point_feats_flat = point_feats.view(-1, 3)  # [B*N, 3]
        device = pc_flat.device

        # Prepare data dict for PTv3
        data_dict = {
            'coord': pc_flat,
            'feat': point_feats_flat,
            'grid_size': self.grid_size,
        }

        offsets = torch.cumsum(
            torch.tensor([N] * batch_size, device=device, dtype=torch.long),
            dim=0
        )
        data_dict['offset'] = offsets

        # Forward through PTv3 backbone
        point = self.backbone(data_dict)
        deep_features = point.feat  # [N_total, feat_dim]
        deep_features = self.feature_proj(deep_features)  # [N_total, feat_dim]

        # Verify the shape
        assert deep_features.shape[0] == batch_size * N, f"Shape mismatch: {deep_features.shape[0]} vs {batch_size * N}"

        # Store intermediate results (mimicking SegNet_AV2)
        # l_feats[0]: original features [B, 3, N]
        # l_feats[-1]: deep features [B, feat_dim, N]
        l_pc = [pc]
        l_feats = [point_feats.transpose(1, 2).contiguous()]  # [B, 3, N]
        l_feats.append(deep_features.view(batch_size, N, self.feat_dim).transpose(1, 2).contiguous())  # [B, feat_dim, N]

        # Process each batch (SegNet_AV2 style)
        mask_list = []
        for b in range(batch_size):
            # Extract object embeddings with MaskFormer head
            # Use l_feats[-1] (deep features) for slot learning
            deep_feats_batch = l_feats[-1][b:b+1]  # [1, feat_dim, N]
            pc_batch = l_pc[0][b:b+1]  # [1, N, 3]

            slot = self.MF_head(
                deep_feats_batch.transpose(1, 2).contiguous(),  # [1, N, feat_dim]
                pc_batch  # [1, N, 3]
            )  # [1, K, D]

            slot = self.object_mlp(slot.transpose(1, 2))  # [1, 64, K]

            # Compute mask using projected deep features to match 64 channels
            mask_features = self.mask_proj(deep_feats_batch)  # [1, 64, N]

            mask = torch.einsum('bdn,bdk->bnk',
                                F.normalize(mask_features, dim=1),
                                F.normalize(slot, dim=1)) / 0.05

            # Softmax over instance dimension
            mask = mask.softmax(dim=-1)  # [1, N, K]

            mask_list.append(mask.squeeze(0))  # [N, K]

        # Return [B, N, K] like SegNet_AV2
        return torch.stack(mask_list, dim=0)


def test_model():
    """Test the model with dummy data"""
    print("Testing PTV3SegNetAV2 model...")

    # Create model
    model = PTV3SegNetAV2(
        n_slot=20,
        feat_dim=64,
        transformer_embed_dim=64,
        n_transformer_layer=2,
        grid_size=0.01
    ).cuda()
    model.eval()

    print(f"\nModel architecture:")
    print(f"  - PTv3 backbone: {type(model.backbone).__name__}")
    print(f"  - Feature projection: {type(model.feature_proj).__name__}")
    print(f"  - MaskFormer head: {type(model.MF_head).__name__}")
    print(f"  - Object MLP: {type(model.object_mlp).__name__}")

    # Test batched input
    print("\n" + "="*50)
    print("Test: Batched point clouds")
    pc_batched = torch.randn(4, 8192, 3).cuda()
    point_feats = torch.randn_like(pc_batched)  # Use same shape as features

    with torch.no_grad():
        mask_batched = model(pc_batched, point_feats)

    print(f"  Input shape: {pc_batched.shape}")
    print(f"  Features shape: {point_feats.shape}")
    print(f"  Output shape: {mask_batched.shape}")
    print(f"  Expected: [B={4}, N={8192}, K={20}]")
    print(f"  min: {mask_batched.min():.4f}, max: {mask_batched.max():.4f}")

    # Parameter count
    print("\n" + "="*50)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.MF_head.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in model.object_mlp.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in model.mask_proj.parameters() if p.requires_grad)

    print(f"Parameter counts:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - PTv3 backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"  - MaskFormer head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    print(f"  - Object MLP: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
    print(f"  - Mask projection: {proj_params:,} ({proj_params/total_params*100:.1f}%)")

    print("\n" + "="*50)
    print("Test passed! ✓")


if __name__ == '__main__':
    test_model()
