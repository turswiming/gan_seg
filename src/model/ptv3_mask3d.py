"""
Mask3D Instance Segmentation Head for Point Transformer V3.

Mask3D is a Transformer-based instance segmentation method that predicts
instance masks directly from point cloud features using learnable queries
and cross-attention mechanisms.

Reference: Mask3D: Mask Transformer for 3D Instance Segmentation
https://arxiv.org/abs/2210.03105
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

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


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module for Mask3D.
    
    Queries attend to point features (keys and values).
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, point_features):
        """
        Cross-attention: queries attend to point features.
        
        Args:
            queries: [num_queries, d_model] - learnable query embeddings
            point_features: [N, d_model] - point cloud features
        
        Returns:
            attended_queries: [num_queries, d_model] - updated queries
        """
        residual = queries
        num_queries = queries.shape[0]
        N = point_features.shape[0]
        
        # Linear projections
        Q = self.w_q(queries).view(num_queries, self.nhead, self.d_k).transpose(0, 1)  # [nhead, num_queries, d_k]
        K = self.w_k(point_features).view(N, self.nhead, self.d_k).transpose(0, 1)  # [nhead, N, d_k]
        V = self.w_v(point_features).view(N, self.nhead, self.d_k).transpose(0, 1)  # [nhead, N, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [nhead, num_queries, N]
        attn_weights = F.softmax(scores, dim=-1)  # [nhead, num_queries, N]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [nhead, num_queries, d_k]
        attended = attended.transpose(0, 1).contiguous().view(num_queries, self.d_model)  # [num_queries, d_model]
        
        # Output projection
        output = self.w_o(attended)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network for Transformer decoder."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with cross-attention."""
    
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, queries, point_features):
        """
        Args:
            queries: [num_queries, d_model]
            point_features: [N, d_model]
        
        Returns:
            updated_queries: [num_queries, d_model]
        """
        queries = self.cross_attn(queries, point_features)
        queries = self.ffn(queries)
        return queries


class Mask3DHead(nn.Module):
    """
    Mask3D instance segmentation head.
    
    Uses learnable query embeddings and cross-attention to predict
    instance masks directly from point cloud features.
    """
    
    def __init__(
        self,
        d_model=256,
        num_queries=100,
        num_decoder_layers=6,
        nhead=8,
        d_ff=2048,
        dropout=0.1,
        num_classes=20,
    ):
        """
        Initialize Mask3D head.
        
        Args:
            d_model (int): Feature dimension
            num_queries (int): Number of instance queries (masks)
            num_decoder_layers (int): Number of decoder layers
            nhead (int): Number of attention heads
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
            num_classes (int): Number of semantic classes
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Mask prediction head: predict mask logits for each query
        # Uses query features to compute similarity with point features
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Class prediction head: predict semantic class for each query
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize query embeddings
        nn.init.normal_(self.query_embed, std=0.02)
        
    def forward(self, point_features):
        """
        Forward pass for Mask3D head.
        
        Args:
            point_features: [N, d_model] - point cloud features from backbone
        
        Returns:
            dict: Dictionary containing:
                - mask_logits: [num_queries, N] - mask logits for each query
                - class_logits: [num_queries, num_classes] - class logits for each query
                - queries: [num_queries, d_model] - final query embeddings
        """
        N = point_features.shape[0]
        
        # Initialize queries
        queries = self.query_embed  # [num_queries, d_model]
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries, point_features)
        
        # Predict masks: each query predicts a mask over all points
        # Process queries through mask head
        query_features = self.mask_head(queries)  # [num_queries, d_model]
        
        # Compute mask logits by dot product between query features and point features
        # This creates a similarity matrix: [num_queries, N]
        mask_logits = torch.matmul(query_features, point_features.t())  # [num_queries, N]
        
        # Predict classes
        class_logits = self.class_head(queries)  # [num_queries, num_classes]
        
        return {
            'mask_logits': mask_logits,  # [num_queries, N]
            'class_logits': class_logits,  # [num_queries, num_classes]
            'queries': queries,  # [num_queries, d_model]
        }


class PTV3Mask3D(nn.Module):
    """
    Point Transformer V3 with Mask3D instance segmentation head.
    
    This model combines PTv3 backbone with Mask3D head for end-to-end
    instance segmentation.
    """
    
    def __init__(
        self,
        num_queries=20,
        num_classes=5,
        in_channels=3,
        feat_dim=256,
        grid_size=0.01,
        enable_flash=True,
        enable_rpe=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        num_decoder_layers=6,
        nhead=8,
        d_ff=2048,
        dropout=0.1,
        **kwargs
    ):
        """
        Initialize PTv3 + Mask3D model.
        
        Args:
            num_queries (int): Number of instance queries
            num_classes (int): Number of semantic classes
            in_channels (int): Input feature channels
            feat_dim (int): Feature dimension for Mask3D head
            grid_size (float): Grid size for voxelization
            enable_flash (bool): Enable flash attention
            enable_rpe (bool): Enable relative position encoding
            enc_depths (tuple): Encoder depths
            enc_channels (tuple): Encoder channels
            dec_depths (tuple): Decoder depths
            dec_channels (tuple): Decoder channels
            num_decoder_layers (int): Number of Mask3D decoder layers
            nhead (int): Number of attention heads
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
        """
        super(PTV3Mask3D, self).__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
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
        
        # Get output feature dimension from decoder
        self.backbone_feat_dim = dec_channels[0] if len(dec_channels) > 0 else enc_channels[-1]
        
        # Project backbone features to Mask3D feature dimension
        self.feature_proj = nn.Sequential(
            nn.Linear(self.backbone_feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        
        # Mask3D head
        self.mask3d_head = Mask3DHead(
            d_model=feat_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            d_ff=d_ff,
            dropout=dropout,
            num_classes=num_classes,
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
            dict: Dictionary containing:
                - mask_logits: [B, num_queries, N] - mask logits
                - class_logits: [B, num_queries, num_classes] - class logits
                - masks: [B, num_queries, N] - soft masks (after sigmoid)
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
        
        # Project features to Mask3D dimension
        features = self.feature_proj(features)  # [N_total, feat_dim]
        
        # Process each batch separately (Mask3D head expects [N, d_model])
        mask_logits_list = []
        class_logits_list = []
        
        if is_batched:
            for b in range(batch_size):
                start_idx = b * N
                end_idx = (b + 1) * N
                batch_features = features[start_idx:end_idx]  # [N, feat_dim]
                
                # Forward through Mask3D head
                output = self.mask3d_head(batch_features)
                mask_logits_list.append(output['mask_logits'])  # [num_queries, N]
                class_logits_list.append(output['class_logits'])  # [num_queries, num_classes]
            
            # Stack results
            mask_logits = torch.stack(mask_logits_list, dim=0)  # [B, num_queries, N]
            class_logits = torch.stack(class_logits_list, dim=0)  # [B, num_queries, num_classes]
        else:
            # Single batch
            output = self.mask3d_head(features)  # [N, feat_dim]
            mask_logits = output['mask_logits'].unsqueeze(0)  # [1, num_queries, N]
            class_logits = output['class_logits'].unsqueeze(0)  # [1, num_queries, num_classes]
        
        # Apply sigmoid to get soft masks
        masks = torch.sigmoid(mask_logits)  # [B, num_queries, N]
        softmax_masks = torch.nn.functional.softmax(mask_logits, dim=1)  # [B, num_queries, N]
        return softmax_masks  # [B, num_queries, N]

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

