from model.scene_flow_predict_model import FLowPredictor, Neural_Prior ,SceneFlowPredictor
from model.mask_predict_model import MaskPredictor

def get_scene_flow_predictor(flow_model_config,N):
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
    if mask_model_config.name == "OptimizedMask":
        return MaskPredictor(slot_num=mask_model_config.slot_num,
                             point_length=N)
    else:
        raise NotImplementedError("scene flow predictor not implemented")