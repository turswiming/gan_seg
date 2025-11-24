"""
Evaluation metrics for instance segmentation and scene flow prediction.

This module provides metrics for evaluating model performance, including:
- Mean Intersection over Union (mIoU) for instance segmentation
- End Point Error (EPE) for scene flow prediction
- Average Precision (AP) for instance segmentation
- Panoptic Quality (PQ), F1-score, Precision, Recall
"""

import torch
import math
import torch.nn.functional as F
import numpy as np


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
    pred_mask = pred_mask > 0.49
    pred_mask = pred_mask.to(dtype=torch.float32)
    max_iou_list = []
    gt_mask = gt_mask > 0.5
    # pre calculate the size of each mask
    gt_mask_size = torch.sum(gt_mask, dim=1)
    pred_mask_size = torch.sum(pred_mask, dim=1)
    iou_recorder = torch.zeros(gt_mask.shape[0], pred_mask.shape[0])
    for j in range(gt_mask.shape[0]):
        max_iou = 0
        if gt_mask_size[j] <= min_points:
            continue  # Skip small masks to avoid noise in IoU calculation

        for i in range(pred_mask.shape[0]):

            intersection = torch.sum(pred_mask[i] * gt_mask[j])
            union = pred_mask_size[i] + gt_mask_size[j] - intersection
            iou = float(intersection) / float(union) if union != 0 else 0

            iou_recorder[j, i] = iou
            if math.isnan(iou):
                continue
            if iou > max_iou:
                max_iou = iou
        # print(f"Instance {j}: Max IoU = {max_iou:.4f} Size = {gt_mask_size[j]}")
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


def calculate_recall_at_threshold(pred_flows, gt_flows, threshold=0.01):
    """
    Calculate recall at a given threshold (e.g., 1cm, 5cm).
    
    Recall at threshold is the percentage of points where the flow error
    (Euclidean distance) is less than the threshold.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
        threshold (float): Error threshold in meters (default: 0.01 for 1cm)
    
    Returns:
        float: Recall at threshold (percentage of points with error < threshold)
    """
    all_errors = []
    total_points = 0
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        # Calculate Euclidean distance between predicted and ground truth flows
        epe = torch.norm(pred_flow - gt_flow, dim=1, p=2)  # Shape: [N]
        all_errors.append(epe)
        total_points += epe.shape[0]
    
    # Concatenate all errors
    all_errors = torch.cat(all_errors)
    
    # Count points with error < threshold
    correct_points = (all_errors < threshold).sum().float()
    
    # Calculate recall
    recall = correct_points / total_points if total_points > 0 else 0.0
    return recall.item()


def calculate_angular_error(pred_flows, gt_flows):
    """
    Calculate angular error between predicted and ground truth flow vectors.
    
    Angular error measures the angle between predicted and ground truth flow vectors.
    This is useful for evaluating direction accuracy regardless of magnitude.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
    
    Returns:
        float: Mean angular error in degrees
    """
    all_angles = []
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        # Normalize vectors
        pred_norm = pred_flow / (torch.norm(pred_flow, dim=1, keepdim=True) + 1e-8)
        gt_norm = gt_flow / (torch.norm(gt_flow, dim=1, keepdim=True) + 1e-8)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.sum(pred_norm * gt_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Ensure valid range for acos
        
        # Calculate angle in radians, then convert to degrees
        angles = torch.acos(cos_sim) * 180.0 / math.pi
        all_angles.append(angles)
    
    # Concatenate and calculate mean
    all_angles = torch.cat(all_angles)
    mean_angle = torch.mean(all_angles)
    return mean_angle.item()


def calculate_outlier_ratio(pred_flows, gt_flows, threshold=0.3):
    """
    Calculate outlier ratio: percentage of points with error > threshold.
    
    Outliers are points where the flow error exceeds a certain threshold.
    Common thresholds: 0.1m, 0.2m, 0.3m.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
        threshold (float): Error threshold in meters (default: 0.3 for 30cm)
    
    Returns:
        float: Outlier ratio (percentage of points with error > threshold)
    """
    all_errors = []
    total_points = 0
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        epe = torch.norm(pred_flow - gt_flow, dim=1, p=2)
        all_errors.append(epe)
        total_points += epe.shape[0]
    
    all_errors = torch.cat(all_errors)
    outliers = (all_errors > threshold).sum().float()
    outlier_ratio = outliers / total_points if total_points > 0 else 0.0
    return outlier_ratio.item()


def calculate_mse(pred_flows, gt_flows):
    """
    Calculate Mean Squared Error (MSE) between predicted and ground truth flows.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
    
    Returns:
        float: Mean squared error
    """
    all_squared_errors = []
    total_points = 0
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        squared_error = torch.sum((pred_flow - gt_flow) ** 2, dim=1)
        all_squared_errors.append(squared_error)
        total_points += squared_error.shape[0]
    
    all_squared_errors = torch.cat(all_squared_errors)
    mse = torch.mean(all_squared_errors)
    return mse.item()


def calculate_mae(pred_flows, gt_flows):
    """
    Calculate Mean Absolute Error (MAE) between predicted and ground truth flows.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
    
    Returns:
        float: Mean absolute error
    """
    all_abs_errors = []
    total_points = 0
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        abs_error = torch.sum(torch.abs(pred_flow - gt_flow), dim=1)
        all_abs_errors.append(abs_error)
        total_points += abs_error.shape[0]
    
    all_abs_errors = torch.cat(all_abs_errors)
    mae = torch.mean(all_abs_errors)
    return mae.item()


def calculate_flow_magnitude_error(pred_flows, gt_flows):
    """
    Calculate error in flow magnitude (speed) prediction.
    
    This measures how well the model predicts the magnitude of flow vectors,
    regardless of direction accuracy.
    
    Args:
        pred_flows (list[torch.Tensor]): List of predicted flow tensors, each with shape [N, 3]
        gt_flows (list[torch.Tensor]): List of ground truth flow tensors, each with shape [N, 3]
    
    Returns:
        float: Mean absolute error in flow magnitude
    """
    all_mag_errors = []
    total_points = 0
    
    for pred_flow, gt_flow in zip(pred_flows, gt_flows):
        pred_mag = torch.norm(pred_flow, dim=1, p=2)
        gt_mag = torch.norm(gt_flow, dim=1, p=2)
        mag_error = torch.abs(pred_mag - gt_mag)
        all_mag_errors.append(mag_error)
        total_points += mag_error.shape[0]
    
    all_mag_errors = torch.cat(all_mag_errors)
    mean_mag_error = torch.mean(all_mag_errors)
    return mean_mag_error.item()


def eval_segm(segm, mask, ignore_npoint_thresh=0):
    """
    Evaluate segmentation on a single sample.
    
    :param segm: (N,) numpy array, ground truth instance labels
    :param mask: (N, K) numpy array, predicted mask logits or probabilities
    :param ignore_npoint_thresh: threshold to ignore GT objects with too few points
    :return:
        pred_iou: (N_pred,) numpy array, IoU for each prediction
        pred_matched: (N_pred,) numpy array, whether each prediction is matched (IoU >= 0.5)
        confidence: (N_pred,) numpy array, confidence score for each prediction
        n_gt_inst: integer, number of ground truth instances
    """
    segm_pred = np.argmax(mask, axis=1)
    _, segm, gt_sizes = np.unique(segm, return_inverse=True, return_counts=True)
    pred_ids, segm_pred, pred_sizes = np.unique(segm_pred, return_inverse=True, return_counts=True)
    n_gt_inst = gt_sizes.shape[0]
    n_pred_inst = pred_sizes.shape[0]
    mask = mask[:, pred_ids]

    # Compute Intersection
    intersection = np.zeros((n_gt_inst, n_pred_inst))
    for i in range(n_gt_inst):
        for j in range(n_pred_inst):
            intersection[i, j] = np.sum(np.logical_and(segm == i, segm_pred == j))

    # Ignore too small GT objects
    ignore_gt_ids = np.where(gt_sizes < ignore_npoint_thresh)[0]

    # An FP is not penalized, if mostly overlapped with ignored GT
    pred_ignore_ratio = np.sum(intersection[ignore_gt_ids], axis=0) / (pred_sizes + 1e-10)
    invalid_pred = (pred_ignore_ratio > 0.5)

    # Kick out predictions' area intersectioned with ignored GT
    pred_sizes = pred_sizes - np.sum(intersection[ignore_gt_ids], axis=0)
    valid_pred = np.logical_and(pred_sizes > 0, np.logical_not(invalid_pred))

    intersection = np.delete(intersection, ignore_gt_ids, axis=0)
    gt_sizes = np.delete(gt_sizes, ignore_gt_ids, axis=0)
    n_gt_inst = gt_sizes.shape[0]

    intersection = intersection[:, valid_pred]
    pred_sizes = pred_sizes[valid_pred]
    mask = mask[:, valid_pred]
    n_pred_inst = pred_sizes.shape[0]

    # Compute confidence scores for predictions
    confidence = np.zeros((n_pred_inst))
    for j in range(n_pred_inst):
        inst_mask = mask[segm_pred == j, j]
        confidence[j] = np.mean(inst_mask)

    # Find matched predictions
    union = np.expand_dims(gt_sizes, 1) + np.expand_dims(pred_sizes, 0) - intersection
    iou = intersection / (union + 1e-10)
    pred_iou = iou.max(axis=0)
    # In panoptic segmentation, Greedy gives the same result as Hungarian
    pred_matched = (pred_iou >= 0.5).astype(float)
    return pred_iou, pred_matched, confidence, n_gt_inst


def accumulate_eval_results(segm, mask, ignore_npoint_thresh=0):
    """
    Accumulate evaluation results on a batch of samples.
    
    :param segm: (B, N) torch tensor, ground truth instance labels
    :param mask: (B, K, N) or (B, N, K) torch tensor, predicted mask logits
    :param ignore_npoint_thresh: threshold to ignore GT objects with too few points
    :return:
        Pred_IoU: (N') numpy array, IoU for each prediction across all batches
        Pred_Matched: (N') numpy array, whether each prediction is matched
        Confidence: (N') numpy array, confidence score for each prediction
        N_GT_Inst: integer, total number of ground truth instances
    """
    segm = segm.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    # Handle different mask formats: (B, K, N) or (B, N, K)
    if mask.shape[1] != segm.shape[1]:
        # Assume (B, K, N) format, transpose to (B, N, K)
        mask = np.transpose(mask, (0, 2, 1))

    Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = [], [], [], []
    n_batch = segm.shape[0]
    for b in range(n_batch):
        pred_iou, pred_matched, confidence, n_gt_inst = eval_segm(segm[b], mask[b], ignore_npoint_thresh=ignore_npoint_thresh)
        Pred_IoU.append(pred_iou)
        Pred_Matched.append(pred_matched)
        Confidence.append(confidence)
        N_GT_Inst.append(n_gt_inst)
    Pred_IoU = np.concatenate(Pred_IoU)
    Pred_Matched = np.concatenate(Pred_Matched)
    Confidence = np.concatenate(Confidence)
    N_GT_Inst = np.sum(N_GT_Inst)
    return Pred_IoU, Pred_Matched, Confidence, N_GT_Inst


def calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=False, eps=1e-10):
    """
    Calculate Average Precision (AP) with MS-COCO standards.
    
    :param Pred_Matched: (N) numpy array, whether each prediction is matched
    :param Confidence: (N) numpy array, confidence score for each prediction
    :param N_GT_Inst: integer, total number of ground truth instances
    :param plot: bool, whether to plot precision-recall curve
    :param eps: float, small epsilon to avoid division by zero
    :return: float, Average Precision value
    """
    inds = np.argsort(-Confidence, kind='mergesort')
    Pred_Matched = Pred_Matched[inds]
    TP = np.cumsum(Pred_Matched)
    FP = np.cumsum(1 - Pred_Matched)
    precisions = TP / np.maximum(TP + FP, eps)
    recalls = TP / np.maximum(N_GT_Inst, eps)
    precisions, recalls = precisions.tolist(), recalls.tolist()

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Query 101-point
    recall_thresholds = np.linspace(0, 1, int(np.round((1 - 0) / 0.01)) + 1, endpoint=True)
    inds = np.searchsorted(recalls, recall_thresholds, side='left').tolist()
    precisions_queried = np.zeros(len(recall_thresholds))
    for rid, pid in enumerate(inds):
        if pid < len(precisions):
            precisions_queried[rid] = precisions[pid]
    precisions, recalls = precisions_queried.tolist(), recall_thresholds.tolist()
    AP = np.mean(precisions)
    
    # Plot P-R curve if needed
    if plot:
        try:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            Pre, Rec = precisions[:2], recalls[:2]
            for i in range(1, len(precisions) - 1):
                Pre.extend([precisions[i+1], precisions[i+1]])
                Rec.extend([recalls[i], recalls[i+1]])
            Pre.append(precisions[-1])
            Rec.append(recalls[-1])

            plt.plot(Rec, Pre)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping plot")
    return AP


def calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst, eps=1e-10):
    """
    Calculate Panoptic Quality (PQ), F1-score, Precision, and Recall.
    
    :param Pred_IoU: (N) numpy array, IoU for each prediction
    :param Pred_Matched: (N) numpy array, whether each prediction is matched
    :param N_GT_Inst: integer, total number of ground truth instances
    :param eps: float, small epsilon to avoid division by zero
    :return: tuple of (PQ, F1, Pre, Rec)
    """
    TP = Pred_Matched.sum()
    TP_IoU = Pred_IoU[Pred_Matched > 0].sum()
    FP = Pred_Matched.shape[0] - TP
    FN = N_GT_Inst - TP

    PQ = TP_IoU / max(TP + 0.5*FP + 0.5*FN, eps)
    Pre = TP / max(TP + FP, eps)
    Rec = TP / max(TP + FN, eps)
    F1 = (2 * Pre * Rec) / max(Pre + Rec, eps)
    return PQ, F1, Pre, Rec
