"""
Point Transformer V3 based mask predictor for instance segmentation.

This module implements a mask predictor using PointTransformerV3 as the backbone
for point cloud instance segmentation and panoptic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from OGCModel.nn_util import Seq

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}

from OGCModel.transformer_util import MaskFormerHead

# Add PointTransformerV3 to path
ptv3_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "PointTransformerV3")
)
ptv3_parent = os.path.dirname(ptv3_path)

# Add paths for imports
if ptv3_parent not in sys.path:
    sys.path.insert(0, ptv3_parent)
if ptv3_path not in sys.path:
    sys.path.insert(0, ptv3_path)

# Import PTv3 model
try:
    # Import as package module
    from PointTransformerV3.model import PointTransformerV3
except ImportError:
    # Fallback: direct file import with path manipulation
    import importlib.util
    import importlib.machinery

    # Load serialization module first
    serialization_path = os.path.join(ptv3_path, "serialization", "__init__.py")
    if os.path.exists(serialization_path):
        serialization_spec = importlib.util.spec_from_file_location(
            "serialization", serialization_path
        )
        serialization_module = importlib.util.module_from_spec(serialization_spec)
        serialization_spec.loader.exec_module(serialization_module)

    # Load main model
    model_path = os.path.join(ptv3_path, "model.py")
    spec = importlib.util.spec_from_file_location("ptv3_model", model_path)
    ptv3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ptv3_module)
    PointTransformerV3 = ptv3_module.PointTransformerV3


class SonataBackboneAdapter(nn.Module):
    """
    Adapter to make Sonata model compatible with PTV3MaskPredictor interface.
    Sonata's forward returns only point, while PTV3MaskPredictor expects (point, features, batch, xyz).
    """
    def __init__(self, sonata_model):
        super().__init__()
        self.sonata_model = sonata_model
    
    def forward(self, data_dict):
        """
        Forward pass that adapts Sonata output to match PTV3 interface.
        
        Args:
            data_dict: Input dictionary for point cloud
            
        Returns:
            point: Point object with features
            features: Feature tensor [N, C]
            batch: Batch indices [N]
            xyz: Coordinates [N, 3]
        """
        point = self.sonata_model(data_dict)
        # Extract features, batch, and coordinates from point
        features = point.feat  # [N, C]
        batch = point.batch  # [N]
        xyz = point.coord  # [N, 3]
        
        # Get median features from encoder (before decoder)
        # For Sonata, we need to extract features from encoder output
        # Since Sonata returns decoder output, we use the final features
        median_features = features
        
        return point, median_features, batch, xyz


class PTV3MaskPredictor(nn.Module):
    """
    Point Transformer V3 based mask predictor for instance segmentation.

    This model uses PTv3 as a feature extractor and adds a prediction head
    for instance segmentation masks.

    Attributes:
        backbone (PointTransformerV3): PTv3 backbone for feature extraction
        slot_num (int): Number of instance slots (masks)
        feat_dim (int): Feature dimension from backbone
        mask_head (nn.Module): Prediction head for masks
    """

    def __init__(
        self,
        slot_num=20,
        in_channels=3,
        feat_dim=256,
        grid_size=0.01,
        enable_flash=True,
        enable_rpe=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        **kwargs,
    ):
        """
        Initialize PTv3 mask predictor.

        Args:
            slot_num (int): Number of instance slots
            in_channels (int): Input feature channels (default 3 for xyz)
            feat_dim (int): Feature dimension for mask head
            grid_size (float): Grid size for voxelization
            enable_flash (bool): Enable flash attention
            enable_rpe (bool): Enable relative position encoding
            enc_depths (tuple): Encoder depths for each stage
            enc_channels (tuple): Encoder channels for each stage
            dec_depths (tuple): Decoder depths for each stage
            dec_channels (tuple): Decoder channels for each stage
        """
        super(PTV3MaskPredictor, self).__init__()

        self.slot_num = slot_num
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
            **kwargs,
        )

        # Get output feature dimension from decoder
        self.backbone_feat_dim = enc_channels[-1]

        # Mask prediction head: project features to slot_num masks
        self.mask_head = nn.Sequential(
            nn.Linear(self.backbone_feat_dim, self.backbone_feat_dim),
            nn.LayerNorm(self.backbone_feat_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_feat_dim, self.backbone_feat_dim),
            nn.LayerNorm(self.backbone_feat_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_feat_dim, self.slot_num),
        )
        self.transformer_embed_dim = 256
        self.maskformer_head = MaskFormerHead(
            n_slot=self.slot_num,
            input_dim=self.backbone_feat_dim,
            n_transformer_layer=4,
            transformer_embed_dim=self.transformer_embed_dim,
            transformer_n_head=8,
            transformer_hidden_dim=256,
            input_pos_enc=False,
        )
        self.object_mlp = (
            Seq(self.transformer_embed_dim)
            .conv1d(self.transformer_embed_dim, bn=BN_CONFIG)
            .conv1d(self.backbone_feat_dim, activation=None)
        )
        #freeze the backbone

    def forward(self, inputs):
        """
        Forward pass for instance segmentation.

        Args:
            inputs: Can be:
                - torch.Tensor: Point cloud [N, 3] or [B, N, 3]
                - dict: Input dictionary containing:
                    - point_cloud_first (torch.Tensor): Point cloud [B, N, 3] or [N, 3]
                    - points (torch.Tensor): Alternative key for point cloud
                    - feat (torch.Tensor, optional): Point features [B, N, C] or [N, C]
                    - offset (torch.Tensor, optional): Batch offsets [B+1]
                    - batch (torch.Tensor, optional): Batch indices [N]

        Returns:
            torch.Tensor: Predicted masks [B, K, N] where:
                - B is batch size
                - K is number of slots
                - N is number of points
        """

        pc = inputs

        # Handle different input formats
        original_shape = pc.shape

        batch_size = pc.shape[0]
        N = pc.shape[1]
        is_batched = True

        # Flatten for processing
        pc_flat = pc.view(-1, 3)  # [B*N, 3]
        device = pc_flat.device

        # Prepare data dict for PTv3
        data_dict = {
            "coord": pc_flat,
            "feat": pc_flat,  # Use coordinates as features if no features provided
            "grid_size": self.grid_size,
        }

        batch = torch.ones(N * batch_size, device=device).long()
        for i in range(batch_size):
            batch[i * N : (i + 1) * N] = i
        data_dict["batch"] = batch
        # Forward through PTv3 backbone
        point, median_features, batch,xyz = self.backbone(data_dict)
        # Extract features from point
        features = point.feat  # [N_total, C]
        # slot = self.mask_head(features)
        # slot = slot.reshape(batch_size, N, -1)
        # slot = slot.softmax(dim=-1)
        # slot = slot.transpose(1, 2)
        # return slot
        # print(f"slot.shape: {slot.shape}")
        # exit()
        # print(f"features.shape: {features.shape}")

        slot = []
        for i in range(batch_size):
            median_features_i = median_features[batch == i]
            pc_i = xyz[batch == i]
            median_features_i = median_features_i.unsqueeze(0)
            pc_i = pc_i.unsqueeze(0)
            slot_i = self.maskformer_head(median_features_i, pc_i)  # (B, K, D)
            slot_i = self.object_mlp(slot_i.transpose(1, 2))  # (B, D, K)
            slot.append(slot_i)
        slot = torch.cat(slot, dim=0)
        features = features.reshape(batch_size, N, -1)
        mask = torch.einsum(
            "bdn,bdk->bnk",
            F.normalize(features.transpose(1, 2), dim=1),
            F.normalize(slot, dim=1),
        ) / 0.05
        mask = mask.softmax(dim=-1)

        return mask.transpose(1, 2)

    def load_pretrained_from_hub(self, pretrained_name=None):
        if pretrained_name is None:
            raise ValueError("Pretrained name is required")
        self.backbone.from_pretrained(f"facebook/sonota/{pretrained_name}")
        return self
