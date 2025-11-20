"""
Predictor factory functions for scene flow and mask prediction models.

This module provides factory functions to create and configure the appropriate
predictor models based on the provided configuration. It supports both mask
prediction and scene flow prediction models.
"""

import torch
from omegaconf import OmegaConf

from model.scene_flow_predict_model import OptimizedFLowPredictor, Neural_Prior
from model.mask_predict_model import OptimizedMaskPredictor, Neural_Mask_Prior
from typing import Optional


def get_scene_flow_predictor(flow_model_config, N):
    """
    Create and configure a scene flow predictor based on configuration.

    Args:
        flow_model_config: Configuration object containing model parameters
        N (int): Number of points in the point cloud

    Returns:
        Union[OptimizedFLowPredictor, Neural_Prior, SceneFlowPredictor]:
            Configured scene flow prediction model

    Raises:
        NotImplementedError: If the requested model type is not implemented
    """
    if flow_model_config.name == "NSFP":
        return Neural_Prior(
            input_dim=3,
            output_dim=3,
            filter_size=flow_model_config.NSFP.num_hidden,
            act_fn=flow_model_config.NSFP.activation,
            layer_size=flow_model_config.NSFP.num_layers,
        )
    if flow_model_config.name == "EularFlow":
        model_detail = flow_model_config.NSFP
        return Neural_Prior(
            input_dim=4,
            output_dim=3,
            filter_size=model_detail.num_hidden,
            act_fn=model_detail.activation,
            layer_size=model_detail.num_layers,
        )
    if flow_model_config.name == "EulerFlowMLP":
        from model.eulerflow_raw_mlp import EulerFlowMLP
        from model.nsfp_raw_mlp import ActivationFn

        return EulerFlowMLP(
            output_dim=3,
            latent_dim=128,
            act_fn=ActivationFn.RELU,
            num_layers=18,
            use_normalization=False,
            normalization_type="batch_norm",
        )
    elif flow_model_config.name == "OptimizedFlow":
        return OptimizedFLowPredictor(dim=3, pointSize=N)
    elif flow_model_config.name == "EulerFlowMLPResidual":
        from model.eulerflow_residual_mlp import EulerFlowMLPResidual
        from model.nsfp_raw_mlp import ActivationFn

        return EulerFlowMLPResidual(output_dim=3, latent_dim=128, act_fn=ActivationFn.RELU, num_layers=9)
    elif flow_model_config.name == "FastFlow3D":
        # Lazy import to avoid SceneFlowZoo dependency unless used
        from SceneFlowZoo.models.feed_forward.fast_flow_3d import FastFlow3D

        model_config = flow_model_config.FastFlow3D
        return FastFlow3D(
            VOXEL_SIZE=model_config.VOXEL_SIZE,
            PSEUDO_IMAGE_DIMS=model_config.PSEUDO_IMAGE_DIMS,
            POINT_CLOUD_RANGE=model_config.POINT_CLOUD_RANGE,
            FEATURE_CHANNELS=model_config.FEATURE_CHANNELS,
            SEQUENCE_LENGTH=model_config.SEQUENCE_LENGTH,
        )
    elif flow_model_config.name == "FlowStep3D":
        from OGCModel.flownet_kitti import FlowStep3D

        return FlowStep3D(
            npoint=8192, use_instance_norm=False, loc_flow_nn=16, loc_flow_rad=1.5, k_decay_fact=1.0
        ).cuda()
    else:
        raise NotImplementedError("scene flow predictor not implemented")


def get_mask_predictor(mask_model_config, N):
    """
    Create and configure a mask predictor based on configuration.

    Args:
        mask_model_config: Configuration object containing:
            - name (str): Model type ("OptimizedMask" supported)
            - slot_num (int): Number of segmentation slots
        N (int): Number of points in the point cloud

    Returns:
        OptimizedMaskPredictor: Configured mask prediction model

    Raises:
        NotImplementedError: If the requested model type is not implemented
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mask_model_config.name == "OptimizedMask":
        return OptimizedMaskPredictor(slot_num=mask_model_config.slot_num, point_length=N).to(device)
    elif mask_model_config.name == "NMP":  # short for NeuralMaskPrior
        model_detail = mask_model_config.NMP
        return Neural_Mask_Prior(
            input_dim=3,
            slot_num=mask_model_config.slot_num,
            filter_size=model_detail.num_hidden,
            act_fn=model_detail.activation,
            layer_size=model_detail.num_layers,
        ).to(device)
    elif mask_model_config.name == "EularNMP":
        model_detail = mask_model_config.NMP

        return Neural_Mask_Prior(
            input_dim=4,
            slot_num=mask_model_config.slot_num,
            filter_size=model_detail.num_hidden,
            act_fn=model_detail.activation,
            layer_size=model_detail.num_layers,
        ).to(device)
    elif mask_model_config.name == "EulerMaskMLP":
        from model.mask_predict_model import ActivationFn, EulerMaskMLP

        model_detail = mask_model_config.MLP
        use_norm = getattr(mask_model_config, "use_normalization", True)  # 默认启用
        norm_type = getattr(mask_model_config, "normalization_type", "group_norm")  # 默认group_norm
        return EulerMaskMLP(
            slot_num=mask_model_config.slot_num,
            filter_size=model_detail.num_hidden,
            act_fn=ActivationFn.LEAKYRELU,
            layer_size=model_detail.num_layers,
            use_normalization=use_norm,
            normalization_type=norm_type,
        ).to(device)
    elif mask_model_config.name == "EulerMaskMLPResidual":
        from model.mask_predict_model import ActivationFn, EulerMaskMLPResidual

        model_detail = mask_model_config.MLP
        return EulerMaskMLPResidual(
            slot_num=mask_model_config.slot_num,
            filter_size=model_detail.num_hidden,
            act_fn=ActivationFn.RELU,
            layer_size=model_detail.num_layers,
        ).to(device)
    elif mask_model_config.name == "EulerMaskMLPRoutine":
        from model.mask_predict_model import ActivationFn, EulerMaskMLPRoutine

        model_detail = mask_model_config.MLP
        return EulerMaskMLPRoutine(
            slot_num=mask_model_config.slot_num,
            filter_size=model_detail.num_hidden,
            act_fn=ActivationFn.RELU,
            layer_size=model_detail.num_layers,
        ).to(device)
    elif mask_model_config.name == "PointMaskFormer":
        from model.pmformer import pmformer

        config = dict(
            lims=[[-48, 48], [-48, 48], [-3, 1.8]],
            offset=0.5,
            target_scale=1,
            grid_meters=[0.2, 0.2, 0.1],
            scales=[0.5, 1],
            pooling_scale=[0.5, 1, 2, 4, 6, 8, 12],
            sizes=[480, 480, 48],
            n_class=mask_model_config.slot_num,
            class_weight=1.0,
            dice_weight=20.0,
            mask_weight=50.0,
            match_class_weight=1.0,
            match_dice_weight=2.0,
            match_mask_weight=5.0,
            num_queries=30,
            dec_layers=6,
        )
        return pmformer(config).to(device)

    elif mask_model_config.name == "MaskFormer3D":
        from OGCModel.segnet_av2 import MaskFormer3D

        mask_former = MaskFormer3D(
            n_slot=mask_model_config.slot_num,
            use_xyz=True,
            n_point=mask_model_config.MaskFormer3D.n_point,
            n_transformer_layer=mask_model_config.MaskFormer3D.n_transformer_layer,
            transformer_embed_dim=mask_model_config.MaskFormer3D.transformer_embed_dim,
            transformer_input_pos_enc=mask_model_config.MaskFormer3D.transformer_input_pos_enc,
        ).cuda()
        return mask_former
    elif mask_model_config.name == "PTV3":
        from model.ptv3_mask_predictor import PTV3MaskPredictor
        
        # Use getattr for OmegaConf objects
        ptv3_config = getattr(mask_model_config, 'PTV3', OmegaConf.create({}))
        model = PTV3MaskPredictor(
            slot_num=mask_model_config.slot_num,
            in_channels=getattr(ptv3_config, 'in_channels', 3),
            feat_dim=getattr(ptv3_config, 'feat_dim', 256),
            grid_size=getattr(ptv3_config, 'grid_size', 0.01),
            enable_flash=getattr(ptv3_config, 'enable_flash', True),
            enable_rpe=getattr(ptv3_config, 'enable_rpe', False),
            enc_depths=tuple(getattr(ptv3_config, 'enc_depths', [2, 2, 2, 6, 2])),
            enc_channels=tuple(getattr(ptv3_config, 'enc_channels', [32, 64, 128, 256, 512])),
            dec_depths=tuple(getattr(ptv3_config, 'dec_depths', [2, 2, 2, 2])),
            dec_channels=tuple(getattr(ptv3_config, 'dec_channels', [64, 64, 128, 256])),
        ).to(device)
        
        # Load pretrained weights if specified
        pretrained_path = getattr(ptv3_config, 'pretrained_path', None)
        pretrained_name = getattr(ptv3_config, 'pretrained_name', None)
        if pretrained_path or pretrained_name:
            try:
                from utils.ptv3_utils import load_ptv3_pretrained
                model = load_ptv3_pretrained(model, pretrained_path, pretrained_name)
                model = model.to(device)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        return model
    elif mask_model_config.name == "PTV3Panoptic":
        from model.ptv3_mask_predictor import PTV3PanopticPredictor
        
        # Use getattr for OmegaConf objects
        ptv3_config = getattr(mask_model_config, 'PTV3', OmegaConf.create({}))
        model = PTV3PanopticPredictor(
            slot_num=mask_model_config.slot_num,
            num_classes=getattr(ptv3_config, 'num_classes', 20),
            in_channels=getattr(ptv3_config, 'in_channels', 3),
            feat_dim=getattr(ptv3_config, 'feat_dim', 256),
            grid_size=getattr(ptv3_config, 'grid_size', 0.01),
            enable_flash=getattr(ptv3_config, 'enable_flash', True),
            enable_rpe=getattr(ptv3_config, 'enable_rpe', False),
        ).to(device)
        
        # Load pretrained weights if specified
        pretrained_path = getattr(ptv3_config, 'pretrained_path', None)
        pretrained_name = getattr(ptv3_config, 'pretrained_name', None)
        if pretrained_path or pretrained_name:
            try:
                from utils.ptv3_utils import load_ptv3_pretrained
                model = load_ptv3_pretrained(model, pretrained_path, pretrained_name)
                model = model.to(device)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        return model
    elif mask_model_config.name == "PointGroup":
        from model.ptv3_pointgroup import PointGroup
        return PointGroup(
            num_classes=getattr(mask_model_config, 'num_classes', 10),
            in_channels=3,
            feat_dim=256,
            min_points_per_instance=50,
            grid_size=0.05,
        ).to(device)
    elif mask_model_config.name == "PTV3Mask3D":
        from model.ptv3_mask3d import PTV3Mask3D
        return PTV3Mask3D(
            num_queries=mask_model_config.slot_num,
            num_classes=getattr(mask_model_config, 'num_classes', 10),
            in_channels=3,
            feat_dim=256,
        ).to(device)
    else:
        raise NotImplementedError("Mask predictor type not implemented")
