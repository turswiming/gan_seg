"""
Sonata (Self-Supervised Point Transformer V3) based mask predictor for instance segmentation.

This module implements a mask predictor using Sonata's pre-trained PointTransformerV3
as the backbone for point cloud instance segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from OGCModel.nn_util import Seq

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}

from OGCModel.transformer_util import MaskFormerHead

# Add sonata to path
sonata_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sonata")
)
if sonata_path not in sys.path:
    sys.path.insert(0, sonata_path)

# Import sonata
try:
    import sonata
    from sonata.model import PointTransformerV3
except ImportError:
    raise ImportError(
        f"Sonata not found. Please ensure sonata is cloned in {os.path.dirname(__file__)}/../sonata"
    )


class SonataMaskPredictor(nn.Module):
    """
    Sonata (Self-Supervised Point Transformer V3) based mask predictor for instance segmentation.

    This model uses Sonata's pre-trained PTv3 as a feature extractor and adds a prediction head
    for instance segmentation masks.

    Attributes:
        backbone (PointTransformerV3): Sonata PTv3 backbone for feature extraction
        slot_num (int): Number of instance slots (masks)
        feat_dim (int): Feature dimension from backbone
        mask_head (nn.Module): Prediction head for masks
    """

    def __init__(
        self,
        slot_num=20,
        in_channels=3,
        feat_dim=256,
        grid_size=0.1,
        sonata_model=None,
        **kwargs,
    ):
        """
        Initialize Sonata mask predictor.

        Args:
            slot_num (int): Number of instance slots
            in_channels (int): Input feature channels (default 3 for xyz)
            feat_dim (int): Feature dimension for mask head
            grid_size (float): Grid size for voxelization
            sonata_model (PointTransformerV3, optional): Pre-loaded Sonata model. 
                If None, will be loaded from config.
        """
        super(SonataMaskPredictor, self).__init__()

        self.slot_num = slot_num
        self.feat_dim = feat_dim
        self.grid_size = grid_size

        # Use provided Sonata model or create placeholder
        if sonata_model is not None:
            self.backbone = sonata_model
        else:
            # Placeholder - will be replaced when loading from checkpoint
            self.backbone = None

        # Get output feature dimension from backbone
        # Sonata default: enc_channels[-1] = 512
        self.backbone_feat_dim = 512

        # Mask prediction head
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
        def grad_hook(module, grad_in, grad_out):
            for g in grad_out:
                if g is not None:
                    print(f"{module.__class__.__name__} grad norm: {g.norm()}")
        self.backbone.register_backward_hook(grad_hook)
        self.maskformer_head.register_backward_hook(grad_hook)

    def get_upsampled_512_features(self, point):
        """
        Extract upsampled 512-dim features from Sonata model output.
        
        This function finds the decoder's 512-dim output and upsamples it to original resolution
        without concatenating skip connections (which would result in 1088-dim).
        
        Args:
            point: Point object from Sonata model forward output
            
        Returns:
            features: torch.Tensor [N, 512] - upsampled 512-dim features at original resolution
            coords: torch.Tensor [N, 3] - coordinates at original resolution
            batch_indices: torch.Tensor [N] - batch indices at original resolution
        """
        # Find decoder's 512-dim output before unpooling concatenation
        decoder_512_feat = None
        decoder_512_batch = None
        decoder_512_xyz = None
        
        # Traverse through decoder to find the 512-dim output
        # The last decoder stage should output 512-dim features
        temp_point = point
        while "pooling_parent" in temp_point.keys():
            # Check if this is a decoder unpooling (has both parent and inverse)
            if "pooling_inverse" in temp_point.keys():
                # This is decoder output before unpooling
                # Check if it's 512-dim (last decoder stage)
                if temp_point.feat.shape[1] == 512:
                    decoder_512_feat = temp_point.feat
                    decoder_512_batch = temp_point.batch
                    decoder_512_xyz = temp_point.coord
                    break
            temp_point = temp_point["pooling_parent"]
        
        # If we didn't find 512-dim decoder output, use the current point's features if they're 512-dim
        if decoder_512_feat is None and point.feat.shape[1] == 512:
            decoder_512_feat = point.feat
            decoder_512_batch = point.batch
            decoder_512_xyz = point.coord
        
        if decoder_512_feat is None:
            # Fallback: return None if 512-dim features not found
            return None, None, None
        
        # Start from decoder 512-dim output and upsample to original resolution
        features = decoder_512_feat
        coords = decoder_512_xyz
        batch_indices = decoder_512_batch
        
        # Upsample through remaining pooling layers to get to original resolution
        # We need to go back through the pooling hierarchy
        # But we want to keep 512-dim, so we only use inverse mapping, not concatenation
        temp_point_for_upsample = point
        # Find the point that has our 512-dim features
        while "pooling_parent" in temp_point_for_upsample.keys():
            if "pooling_inverse" in temp_point_for_upsample.keys():
                if temp_point_for_upsample.feat.shape[1] == 512:
                    break
            temp_point_for_upsample = temp_point_for_upsample["pooling_parent"]
        
        # Upsample by following the inverse mapping
        # Go through all pooling layers to reach original resolution
        while "pooling_parent" in temp_point_for_upsample.keys():
            if "pooling_inverse" in temp_point_for_upsample.keys():
                parent = temp_point_for_upsample["pooling_parent"]
                inverse = temp_point_for_upsample["pooling_inverse"]
                # Upsample features using inverse mapping (keep 512-dim)
                features = features[inverse]
                coords = parent.coord
                batch_indices = parent.batch
                temp_point_for_upsample = parent
            else:
                break
        
        return features, coords, batch_indices

    def forward(self, inputs):
        """
        Forward pass for instance segmentation using Sonata model.

        Args:
            inputs: torch.Tensor: Point cloud [B, N, 3]

        Returns:
            torch.Tensor: Predicted masks [B, K, N] where:
                - B is batch size
                - K is number of slots
                - N is number of points
        """
        if self.backbone is None:
            raise RuntimeError("Sonata backbone not initialized. Please load model first.")

        pc = inputs

        # Handle different input formats
        batch_size = pc.shape[0]
        N = pc.shape[1]

        # Flatten for processing
        pc_flat = pc.view(-1, 3)  # [B*N, 3]
        device = pc_flat.device

        # Check what input channels the model expects
        # Sonata default is 6 (coord+normal) or 9 (coord+color+normal)
        # If we only have coord, we need to pad or use custom_config
        model_in_channels = getattr(self.backbone.embedding, 'in_channels', 6)
        
        # Prepare features based on model's expected input channels
        if model_in_channels == 3:
            # Model expects only coord
            feat = pc_flat
        elif model_in_channels == 6:
            # Model expects coord + normal (3 + 3)
            # Create zero normal if not available
            normal = torch.zeros_like(pc_flat)
            feat = torch.cat([pc_flat, normal], dim=-1)  # [B*N, 6]
        elif model_in_channels == 9:
            # Model expects coord + color + normal (3 + 3 + 3)
            # Create zero color and normal if not available
            color = torch.zeros_like(pc_flat)
            normal = torch.zeros_like(pc_flat)
            feat = torch.cat([pc_flat, color, normal], dim=-1)  # [B*N, 9]
        else:
            # Fallback: use coord and pad to required size
            padding_size = model_in_channels - 3
            if padding_size > 0:
                padding = torch.zeros(pc_flat.shape[0], padding_size, device=device, dtype=pc_flat.dtype)
                feat = torch.cat([pc_flat, padding], dim=-1)
            else:
                feat = pc_flat[:, :model_in_channels]

        # Prepare data dict for Sonata
        data_dict = {
            "coord": pc_flat,
            "feat": feat,  # Features with correct dimension
            "grid_size": self.grid_size,
        }

        # Create batch indices
        batch = torch.zeros(N * batch_size, device=device, dtype=torch.long)
        for i in range(batch_size):
            batch[i * N : (i + 1) * N] = i
        data_dict["batch"] = batch

        point, median_features, batch,xyz = self.backbone(data_dict)
        # Extract encoder features before decoder (for maskformer head)
        # Store encoder output by going back through pooling_parent
        upsampled_512_feat, upsampled_coords, upsampled_batch = self.get_upsampled_512_features(point)
        enc_point = point
        # Find the last encoder stage (before any decoder unpooling)
        if "pooling_parent" in point.keys():
            # Traverse back to find encoder output
            while "pooling_parent" in enc_point.keys():
                enc_point = enc_point["pooling_parent"]
        
        # Map decoder features back to original point cloud scale
        # Sonata uses pooling_parent and pooling_inverse to track hierarchical structure
        # Need to upcast features from decoder back to original resolution
        # First, handle decoder unpooling (concatenate skip connections)
        for _ in range(2):
            if "pooling_parent" in point.keys() and "pooling_inverse" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                # Concatenate decoder features with skip connection
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
        
        # Continue upsampling through remaining pooling layers
        while "pooling_parent" in point.keys():
            if "pooling_inverse" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                # Upsample features using inverse mapping
                parent.feat = point.feat[inverse]
                point = parent

        # Extract features, coordinates, and batch from upsampled point
        features = point.feat  # [N_decoder, C] - features after decoder upsampling
        coords = point.coord  # [N_decoder, 3]
        batch_indices = point.batch  # [N_decoder]

        # Handle inverse mapping if exists (for grid sampling)
        # Grid sampling may reduce the number of points, inverse maps back to original
        if "inverse" in point.keys():
            inverse = point.inverse
            # Map features back to original point cloud size
            original_num_points = inverse.shape[0]
            # Create full-size feature tensor
            features_full = torch.zeros(
                original_num_points, 
                features.shape[1], 
                device=features.device, 
                dtype=features.dtype
            )
            features_full[inverse] = features
            features = features_full
            
            # Also update batch indices and coords to original
            coords = data_dict["coord"]  # Use original coordinates
            batch_indices = data_dict["batch"]  # Use original batch indices

        # Process each batch separately for mask prediction using encoder features
        slot = []
        for i in range(batch_size):
            # Use encoder features for maskformer head (more semantic, fewer points)
            enc_features_i = median_features[batch == i]
            enc_xyz_i = xyz[batch == i]
            
            # Add batch dimension
            enc_features_i = enc_features_i.unsqueeze(0)  # [1, N_i, C]
            enc_xyz_i = enc_xyz_i.unsqueeze(0)  # [1, N_i, 3]
            # Get slots from maskformer head
            slot_i = self.maskformer_head(enc_features_i, enc_xyz_i)  # (1, K, D)
            slot_i = self.object_mlp(slot_i.transpose(1, 2))  # (1, D, K)
            slot.append(slot_i)

        slot = torch.cat(slot, dim=0)  # [B, D, K]

        # Reshape features for batch processing
        # Features are now at original point cloud resolution (after upsampling)
        features_list = []
        for i in range(batch_size):
            batch_mask = upsampled_batch == i
            features_i = upsampled_512_feat[batch_mask]
            features_list.append(features_i)
        
        # Pad features to same length for batch processing
        features_padded = []
        for i in range(batch_size):
            features_i = features_list[i]
            features_padded.append(features_i)
        features_batched = torch.stack(features_padded, dim=0)  # [B, max_N, C]
        mask = torch.einsum(
            "bdn,bdk->bnk",
            F.normalize(features_batched.transpose(1, 2), dim=1),
            F.normalize(slot, dim=1),
        ) / 0.05
        mask = mask.softmax(dim=-1)

        return mask.transpose(1, 2)

