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

# Add PointTransformerV3 to path
ptv3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PointTransformerV3'))
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
        **kwargs
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
            **kwargs
        )
        
        # Get output feature dimension from decoder
        self.backbone_feat_dim = dec_channels[0] if len(dec_channels) > 0 else enc_channels[-1]
        
        # Mask prediction head: project features to slot_num masks
        self.mask_head = nn.Sequential(
            nn.Linear(self.backbone_feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, slot_num)
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
                    - feat (torch.Tensor, optional): Point features [B, N, C] or [N, C]
                    - offset (torch.Tensor, optional): Batch offsets [B+1]
                    - batch (torch.Tensor, optional): Batch indices [N]
                
        Returns:
            torch.Tensor: Predicted masks [B, K, N] where:
                - B is batch size
                - K is number of slots
                - N is number of points
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
            # Single point cloud [N, 3]
            pc = pc.unsqueeze(0)  # [1, N, 3]
            batch_size = 1
            N = pc.shape[1]
            is_batched = False
        elif pc.dim() == 3:
            # Batched point cloud [B, N, 3]
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
            'feat': pc_flat,  # Use coordinates as features if no features provided
            'grid_size': self.grid_size,
        }
        
        # Handle batch information
        if isinstance(inputs, dict):
            if 'offset' in inputs:
                data_dict['offset'] = inputs['offset']
            elif 'batch' in inputs:
                data_dict['batch'] = inputs['batch']
            else:
                # Create batch info from point cloud shape
                if is_batched:
                    offsets = torch.cumsum(
                        torch.tensor([N] * batch_size, device=device, dtype=torch.long), 
                        dim=0
                    )
                    data_dict['offset'] = offsets
                else:
                    data_dict['offset'] = torch.tensor([N], device=device, dtype=torch.long)
        else:
            # Create default batch info
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
        
        # Extract features from point
        features = point.feat  # [N_total, C]
        
        # Project to mask predictions
        mask_logits = self.mask_head(features)  # [N_total, slot_num]
        
        # Reshape to [B, K, N]
        if is_batched:
            mask_logits = mask_logits.view(batch_size, N, self.slot_num)
        else:
            mask_logits = mask_logits.unsqueeze(0)  # [1, N, slot_num]
        
        # Transpose to [B, K, N]
        mask_logits = mask_logits.transpose(1, 2)
        
        # Apply softmax to get probability masks
        masks = F.softmax(mask_logits, dim=1)
        
        return masks
    def load_pretrained_from_hub(self, pretrained_name=None):
        if pretrained_name is None:
            raise ValueError("Pretrained name is required")
        self.backbone.from_pretrained(f"facebook/sonota/{pretrained_name}")
        return self


class PTV3PanopticPredictor(nn.Module):
    """
    Point Transformer V3 based panoptic segmentation predictor.
    
    This model predicts both semantic classes and instance masks.
    """
    
    def __init__(
        self,
        slot_num=20,
        num_classes=20,
        in_channels=3,
        feat_dim=256,
        grid_size=0.01,
        enable_flash=True,
        enable_rpe=False,
        **kwargs
    ):
        """
        Initialize PTv3 panoptic predictor.
        
        Args:
            slot_num (int): Number of instance slots
            num_classes (int): Number of semantic classes
            in_channels (int): Input feature channels
            feat_dim (int): Feature dimension
            grid_size (float): Grid size for voxelization
            enable_flash (bool): Enable flash attention
            enable_rpe (bool): Enable relative position encoding
        """
        super(PTV3PanopticPredictor, self).__init__()
        
        self.slot_num = slot_num
        self.num_classes = num_classes
        
        # Initialize mask predictor
        self.mask_predictor = PTV3MaskPredictor(
            slot_num=slot_num,
            in_channels=in_channels,
            feat_dim=feat_dim,
            grid_size=grid_size,
            enable_flash=enable_flash,
            enable_rpe=enable_rpe,
            **kwargs
        )
        
        # Semantic class prediction head
        self.semantic_head = nn.Sequential(
            nn.Linear(self.mask_predictor.backbone_feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, num_classes)
        )
        
    def forward(self, inputs):
        """
        Forward pass for panoptic segmentation.
        
        Args:
            inputs (dict): Input dictionary
            
        Returns:
            dict: Dictionary containing:
                - masks (torch.Tensor): Instance masks [B, K, N]
                - semantic_logits (torch.Tensor): Semantic class logits [B, N, num_classes]
        """
        # Get features from backbone
        pc = inputs.get('point_cloud_first', inputs.get('points', None))
        if pc is None:
            raise ValueError("Input must contain 'point_cloud_first' or 'points'")
        
        # Prepare data dict
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
            batch_size = 1
            N = pc.shape[1]
        else:
            batch_size = pc.shape[0]
            N = pc.shape[1]
            pc = pc.view(-1, 3)
        
        data_dict = {
            'coord': pc,
            'feat': pc,
            'grid_size': self.mask_predictor.grid_size,
        }
        
        if isinstance(inputs, dict) and 'offset' in inputs:
            data_dict['offset'] = inputs['offset']
        elif isinstance(inputs, dict) and 'batch' in inputs:
            data_dict['batch'] = inputs['batch']
        else:
            data_dict['offset'] = torch.tensor([N], device=pc.device, dtype=torch.long)
        
        # Forward through backbone
        point = self.mask_predictor.backbone(data_dict)
        features = point.feat
        
        # Predict instance masks
        masks = self.mask_predictor(inputs)
        
        # Predict semantic classes
        semantic_logits = self.semantic_head(features)  # [N_total, num_classes]
        
        if batch_size == 1:
            semantic_logits = semantic_logits.unsqueeze(0)
        else:
            semantic_logits = semantic_logits.view(batch_size, N, self.num_classes)
        
        return {
            'masks': masks,
            'semantic_logits': semantic_logits
        }

