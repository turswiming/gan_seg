"""
Predictor factory functions for scene flow and mask prediction models.

This module provides factory functions to create and configure the appropriate
predictor models based on the provided configuration. It supports both mask
prediction and scene flow prediction models.
"""

from model.scene_flow_predict_model import FLowPredictor, Neural_Prior ,SceneFlowPredictor
from model.mask_predict_model import MaskPredictor

def get_scene_flow_predictor(flow_model_config,N):
    """
    Create and configure a scene flow predictor based on configuration.
    
    Args:
        flow_model_config: Configuration object containing model parameters
        N (int): Number of points in the point cloud
        
    Returns:
        Union[FlowPredictor, Neural_Prior, SceneFlowPredictor]: 
            Configured scene flow prediction model
        
    Raises:
        NotImplementedError: If the requested model type is not implemented
    """
    if flow_model_config.name == "NSFP":
        return Neural_Prior(dim_x=3,
                            filter_size=flow_model_config.NSFP.num_layers,
                            act_fn=flow_model_config.NSFP.activation,
                            layer_size=flow_model_config.NSFP.num_layers)
    elif flow_model_config.name == "OptimizedFlow":
        return FLowPredictor(dim=3,
                             pointSize=N)
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
        MaskPredictor: Configured mask prediction model
        
    Raises:
        NotImplementedError: If the requested model type is not implemented
    """
    if mask_model_config.name == "OptimizedMask":
        return MaskPredictor(slot_num=mask_model_config.slot_num,
                             point_length=N)
    else:
        raise NotImplementedError("Mask predictor type not implemented")