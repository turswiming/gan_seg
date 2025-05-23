"""
Evaluation metrics for instance segmentation and scene flow prediction.

This module provides metrics for evaluating model performance, including:
- Mean Intersection over Union (mIoU) for instance segmentation
- End Point Error (EPE) for scene flow prediction
"""

import torch
import torch.nn.functional as F

def calculate_miou(pred_mask, gt_mask):
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
        if j == 0:
            continue#skip the background
        max_iou = 0
        print(f"gt_mask {j} size {gt_mask_size[j]}")
        for i in range(pred_mask.shape[0]):
        
            intersection = torch.sum(pred_mask[i] * gt_mask[j])
            union = pred_mask_size[i] + gt_mask_size[j] - intersection
            iou = float(intersection) / float(union) if union != 0 else 0
            if iou > max_iou:
                max_iou = iou
        print(f"max_iou {max_iou}")
        max_iou_list.append(max_iou)
    print(f"max_iou_list {max_iou_list}")
    mean_iou = torch.mean(torch.tensor(max_iou_list).to(dtype=torch.float32))
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
        epe_list.append(torch.mean(epe))
    
    # Average across batch
    epe_mean = torch.mean(torch.stack(epe_list))
    return epe_mean 