"""
Forward pass utilities for scene flow and mask prediction.

This module contains forward pass functions that were
previously in main.py, organized for better modularity.
"""

import torch
from model.eulerflow_raw_mlp import QueryDirection


def forward_scene_flow(point_cloud_firsts, point_cloud_nexts, sample, scene_flow_predictor, 
                      config, train_flow, device):
    """Perform forward pass for scene flow prediction.
    
    Args:
        point_cloud_firsts: List of first point clouds
        point_cloud_nexts: List of next point clouds
        sample: Data sample
        scene_flow_predictor: Scene flow model
        config: Configuration object
        train_flow: Whether in training mode
        device: Training device
        
    Returns:
        tuple: (pred_flow, reverse_pred_flow, longterm_pred_flow)
    """
    pred_flow = []
    reverse_pred_flow = []
    longterm_pred_flow = {}
    
    if train_flow:
        for i in range(len(point_cloud_firsts)):
            if config.model.flow.name in config.model.euler_flow_models:
                if sample["k"][i] == 1:
                    pred_flow.append(scene_flow_predictor(point_cloud_firsts[i], sample["idx"][i], 
                                                        sample["total_frames"][i], QueryDirection.FORWARD))
                    reverse_pred_flow.append(scene_flow_predictor(point_cloud_nexts[i], sample["idx"][i]+1, 
                                                                sample["total_frames"][i], QueryDirection.REVERSE))
                else:
                    # Multi-step prediction
                    pred_pc = point_cloud_firsts[i].clone()
                    for k in range(0, sample["k"][i]):
                        pred_flow_temp = scene_flow_predictor(pred_pc, sample["idx"][i]+k, 
                                                            sample["total_frames"][i], QueryDirection.FORWARD)
                        pred_pc = pred_pc + pred_flow_temp
                        longterm_pred_flow[sample["idx"][i]+k+1] = pred_pc.clone()
                        if k == 0:
                            pred_flow.append(pred_flow_temp)
                            
                    # Reverse multi-step prediction
                    pred_pc = point_cloud_nexts[i].clone()
                    for k in range(0, sample["k"][i]):
                        pred_flow_temp = scene_flow_predictor(pred_pc, sample["idx"][i]-k+1, 
                                                            sample["total_frames"][i], QueryDirection.REVERSE)
                        pred_pc = pred_pc + pred_flow_temp
                        longterm_pred_flow[sample["idx"][i]-k] = pred_pc.clone()
                        if k == 0:
                            reverse_pred_flow.append(pred_flow_temp)
            else:
                pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))
    else:
        with torch.no_grad():
            for i in range(len(point_cloud_firsts)):
                if config.model.flow.name in config.model.euler_flow_models:
                    pred_flow.append(scene_flow_predictor(point_cloud_firsts[i], sample["idx"][i], 
                                                        sample["total_frames"][i], QueryDirection.FORWARD))
                else:
                    pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))
                    
    return pred_flow, reverse_pred_flow, longterm_pred_flow


def forward_mask_prediction(point_cloud_firsts, sample, mask_predictor, config, train_mask):
    """Perform forward pass for mask prediction.
    
    Args:
        point_cloud_firsts: List of first point clouds
        sample: Data sample
        mask_predictor: Mask prediction model
        config: Configuration object
        train_mask: Whether in training mode
        
    Returns:
        list: Predicted masks
    """
    pred_mask = []
    
    if train_mask:
        for i in range(len(point_cloud_firsts)):
            if config.model.mask.name in config.model.euler_mask_models:
                mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                mask = mask.permute(1, 0)
                pred_mask.append(mask)
            else:
                pred_mask.append(mask_predictor(point_cloud_firsts[i]))
    else:
        with torch.no_grad():
            for i in range(len(point_cloud_firsts)):
                if config.model.mask.name in config.model.euler_mask_models:
                    mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                    mask = mask.permute(1, 0)
                    pred_mask.append(mask)
                else:
                    pred_mask.append(mask_predictor(point_cloud_firsts[i]))
                    
    return pred_mask
