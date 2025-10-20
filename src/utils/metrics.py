"""
Evaluation metrics for instance segmentation and scene flow prediction.

This module provides metrics for evaluating model performance, including:
- Mean Intersection over Union (mIoU) for instance segmentation
- End Point Error (EPE) for scene flow prediction
"""

import torch
import torch.nn.functional as F

def calculate_miou(pred_mask, gt_mask, min_points=100):
    """
    Calculate Mean Intersection over Union (mIoU) between predicted and ground truth instance masks.
    
    This function computes the mIoU by:
    1. Converting predicted mask to one-hot format
    2. Finding best matching IoU for each ground truth instance
    3. Averaging the best IoUs across all instances
    
    Args:
        pred_mask (torch.Tensor): Predicted instance masks with shape [K, N], where K is number of predicted 
                                 instances and N is number of points
        gt_mask (torch.Tensor): Ground truth instance masks with shape [K, N], where K is number of ground
                               truth instances and N is number of points
        
    Returns:
        torch.Tensor: Mean IoU value across all matched instance pairs, excluding background (instance 0)
    """
    pred_mask = torch.softmax(pred_mask, dim=0)
    pred_mask = torch.argmax(pred_mask, dim=0)
    pred_mask = F.one_hot(pred_mask).permute(1, 0).to(device=gt_mask.device)
    pred_mask = pred_mask>0.49
    pred_mask = pred_mask.to(dtype=torch.float32)
    max_iou_list = []
    gt_mask = gt_mask>0.5
    #pre calculate the size of each mask
    gt_mask_size = torch.sum(gt_mask, dim=1)
    pred_mask_size = torch.sum(pred_mask, dim=1)
    for j in range(gt_mask.shape[0]):
        max_iou = 0
        if gt_mask_size[j] <= min_points:
            continue  # Skip small masks to avoid noise in IoU calculation
        if j==0:
            continue
        for i in range(pred_mask.shape[0]):
        
            intersection = torch.sum(pred_mask[i] * gt_mask[j])
            union = pred_mask_size[i] + gt_mask_size[j] - intersection
            iou = float(intersection) / float(union) if union != 0 else 0
            if iou > max_iou:
                max_iou = iou
        max_iou_list.append(max_iou)
        # print(f"Instance {j}: Max IoU = {max_iou:.4f} Size = {gt_mask_size[j]}")
    mean_iou = torch.mean(torch.tensor(max_iou_list).to(dtype=torch.float32))
    if torch.isnan(mean_iou):
        return None
    return mean_iou

def calculate_epe(pred_flows, gt_flows):
    """
    Calculate End Point Error (EPE) between predicted and ground truth scene flows.
    
    EPE measures the Euclidean distance between predicted and ground truth flow vectors.
    For batched inputs, it computes the mean EPE across all points and all batch items.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
                                        where N is number of points
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
        
    Returns:
        torch.Tensor: Mean EPE value across all points and batch items
    """
    epe_list = []
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        # Calculate Euclidean distance between predicted and ground truth flows
        epe = torch.norm(pred_flow - gt_flow, dim=1, p=2)  # Shape: [N]
        mean_epe = torch.mean(epe)
        if torch.isnan(mean_epe):
            mean_epe = torch.zeros_like(mean_epe)
        epe_list.append(mean_epe)
    
    # Average across batch
    epe_mean = torch.mean(torch.stack(epe_list))
    return epe_mean 