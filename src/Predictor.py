"""
Predictor factory functions for scene flow and mask prediction models.

This module provides factory functions to create and configure the appropriate
predictor models based on the provided configuration. It supports both mask
prediction and scene flow prediction models.
"""
import torch

from model.scene_flow_predict_model import OptimizedFLowPredictor, Neural_Prior
from model.mask_predict_model import OptimizedMaskPredictor, Neural_Mask_Prior
from typing import Optional

def get_scene_flow_predictor(flow_model_config,N):
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
        return Neural_Prior(input_dim=3,
                            output_dim=3,
                            filter_size=flow_model_config.NSFP.num_hidden,
                            act_fn=flow_model_config.NSFP.activation,
                            layer_size=flow_model_config.NSFP.num_layers)
    if flow_model_config.name == "EularFlow":
        model_detail = flow_model_config.NSFP
        return Neural_Prior(input_dim=4,
                            output_dim=3,
                            filter_size=model_detail.num_hidden,
                            act_fn=model_detail.activation,
                            layer_size=model_detail.num_layers)
    if flow_model_config.name == "EulerFlowMLP":
        from model.eulerflow_raw_mlp import EulerFlowMLP
        from model.nsfp_raw_mlp import ActivationFn
        return EulerFlowMLP(output_dim=3,
                            latent_dim=128,
                            act_fn=ActivationFn.RELU,
                            num_layers=18,
                            use_normalization=False,
                            normalization_type="batch_norm")
    elif flow_model_config.name == "OptimizedFlow":
        return OptimizedFLowPredictor(dim=3,
                             pointSize=N)
    elif flow_model_config.name == "EulerFlowMLPResidual":
        from model.eulerflow_residual_mlp import EulerFlowMLPResidual
        from model.nsfp_raw_mlp import ActivationFn
        return EulerFlowMLPResidual(output_dim=3,
                            latent_dim=128,
                            act_fn=ActivationFn.RELU,
                            num_layers=9)
    elif flow_model_config.name == "FastFlow3D":
        # Lazy import to avoid SceneFlowZoo dependency unless used
        from model.fastflow3d_wrapper import FastFlow3DInference
        scene_flow_zoo_root: str = getattr(flow_model_config, "scene_flow_zoo_root", "/workspace/gan_seg/SceneFlowZoo")
        ckpt_path: Optional[str] = getattr(flow_model_config, "ckpt_path", None)
        device: Optional[str] = getattr(flow_model_config, "device", None)
        class _Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapper = FastFlow3DInference(scene_flow_zoo_root=scene_flow_zoo_root, ckpt_path=ckpt_path, device=device)
            def forward(self, pc0: torch.Tensor, pc1: Optional[torch.Tensor]=None, pose0: Optional[torch.Tensor]=None, pose1: Optional[torch.Tensor]=None):
                if pc1 is None or pose0 is None or pose1 is None:
                    raise ValueError("FastFlow3D requires pc1, pose0, pose1 for inference")
                return self.wrapper.predict(pc0[:, :3], pc1[:, :3], pose0, pose1)
        return _Adapter()
    else:
        raise NotImplementedError("scene flow predictor not implemented")
    
def get_mask_predictor(mask_model_config,N):
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
        return OptimizedMaskPredictor(slot_num=mask_model_config.slot_num,
                             point_length=N).to(device)
    elif mask_model_config.name == "NMP":#short for NeuralMaskPrior
        model_detail = mask_model_config.NMP
        return Neural_Mask_Prior(input_dim=3,
                            slot_num=mask_model_config.slot_num,
                            filter_size=model_detail.num_hidden,
                            act_fn=model_detail.activation,
                            layer_size=model_detail.num_layers).to(device)
    elif mask_model_config.name == "EularNMP":
        model_detail = mask_model_config.NMP

        return Neural_Mask_Prior(input_dim=4,
                            slot_num=mask_model_config.slot_num,
                            filter_size=model_detail.num_hidden,
                            act_fn=model_detail.activation,
                            layer_size=model_detail.num_layers).to(device)
    elif mask_model_config.name == "EulerMaskMLP":
        from model.mask_predict_model import ActivationFn, EulerMaskMLP
        model_detail = mask_model_config.MLP
        use_norm = getattr(mask_model_config, 'use_normalization', True)  # 默认启用
        norm_type = getattr(mask_model_config, 'normalization_type', 'group_norm')  # 默认group_norm
        return EulerMaskMLP(slot_num=mask_model_config.slot_num,
                            filter_size=model_detail.num_hidden,
                            act_fn=ActivationFn.LEAKYRELU,
                            layer_size=model_detail.num_layers,
                            use_normalization=use_norm,
                            normalization_type=norm_type).to(device)
    elif mask_model_config.name == "EulerMaskMLPResidual":
        from model.mask_predict_model import ActivationFn, EulerMaskMLPResidual
        model_detail = mask_model_config.MLP
        return EulerMaskMLPResidual(slot_num=mask_model_config.slot_num,
                            filter_size=model_detail.num_hidden,
                            act_fn=ActivationFn.RELU,
                            layer_size=model_detail.num_layers,
                            ).to(device)
    elif mask_model_config.name == "EulerMaskMLPRoutine":
        from model.mask_predict_model import ActivationFn, EulerMaskMLPRoutine
        model_detail = mask_model_config.MLP
        return EulerMaskMLPRoutine(slot_num=mask_model_config.slot_num,
                            filter_size=model_detail.num_hidden,
                            act_fn=ActivationFn.RELU,
                            layer_size=model_detail.num_layers,).to(device)
    else:
        raise NotImplementedError("Mask predictor type not implemented")