"""
Loss functions for scene flow and mask prediction training.

This module contains all the individual loss computation functions
that were previously in main.py, organized for better modularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.eulerflow_raw_mlp import QueryDirection


def compute_reconstruction_loss(
    config, loss_functions, point_cloud_firsts, point_cloud_nexts, pred_mask, pred_flow, train_mask, device
):
    """Compute reconstruction related losses.

    Returns:
        tuple: (rec_loss, rec_flow_loss, reconstructed_points)
    """
    if (config.lr_multi.rec_loss > 0 or config.lr_multi.rec_flow_loss > 0) and train_mask:
        pred_flow_detach = [flow.detach() for flow in pred_flow]
        rec_loss, reconstructed_points = loss_functions["reconstruction"](
            point_cloud_firsts, point_cloud_nexts, pred_mask, pred_flow_detach
        )
        rec_loss = rec_loss * config.lr_multi.rec_loss
    else:
        rec_loss = torch.tensor(0.0, device=device, requires_grad=True)
        reconstructed_points = None

    # Reconstruction flow loss
    if config.lr_multi.rec_flow_loss > 0 and train_mask and reconstructed_points is not None:
        rec_flow_loss = 0
        for i in range(len(point_cloud_firsts)):
            pred_second_point = point_cloud_firsts[i][:, :3] + pred_flow[i]
            rec_flow_loss += loss_functions["flow_rec"](pred_second_point, reconstructed_points[i])
        rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
    else:
        rec_flow_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return rec_loss, rec_flow_loss, reconstructed_points


def compute_flow_losses(
    config,
    loss_functions,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    reverse_pred_flow,
    longterm_pred_flow,
    sample,
    device,
):
    """Compute flow related losses."""
    # Chamfer distance loss
    if config.lr_multi.flow_loss > 0:
        flow_loss = 0
        for i in range(len(point_cloud_firsts)):
            pred_second_points = point_cloud_firsts[i][:, :3] + pred_flow[i]
            flow_loss += loss_functions["chamfer"](
                pred_second_points.unsqueeze(0), point_cloud_nexts[i][:, :3].to(device).unsqueeze(0)
            )
            if len(reverse_pred_flow) > 0:
                pred_first_point = point_cloud_nexts[i][:, :3] + reverse_pred_flow[i]
                flow_loss += loss_functions["chamfer"](
                    pred_first_point.unsqueeze(0), point_cloud_firsts[i][:, :3].to(device).unsqueeze(0)
                )
                flow_loss = flow_loss / 2
        if len(longterm_pred_flow) > 0:
            for idx in longterm_pred_flow:
                pred_points = longterm_pred_flow[idx][:, :3]
                real_points = sample["self"][0].get_item(idx)["point_cloud_first"][:, :3].to(device)
                flow_loss += loss_functions["chamfer"](pred_points.unsqueeze(0), real_points.to(device).unsqueeze(0))
        flow_loss = flow_loss * config.lr_multi.flow_loss
    else:
        flow_loss = torch.tensor(0.0, device=device)

    return flow_loss


def compute_scene_flow_smoothness_loss(
    config, loss_functions, point_cloud_firsts, pred_mask, pred_flow, step, scene_flow_smoothness_scheduler, device
):
    """Compute scene flow smoothness loss."""
    if config.lr_multi.scene_flow_smoothness > 0 and step > config.training.begin_train_smooth:
        scene_flow_smooth_loss = loss_functions["flow_smooth"](point_cloud_firsts, pred_mask, pred_flow)
        scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
        scene_flow_smooth_loss = scene_flow_smooth_loss * scene_flow_smoothness_scheduler(step)
    else:
        scene_flow_smooth_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return scene_flow_smooth_loss


def compute_point_smoothness_loss(config, loss_functions, point_cloud_firsts, pred_mask, step, device):
    """Compute point smoothness loss."""
    if config.lr_multi.point_smooth_loss > 0 and step > config.training.begin_train_point_smooth:
        point_smooth_loss = loss_functions["point_smooth"](point_cloud_firsts, pred_mask)
        point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
    else:
        point_smooth_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return point_smooth_loss


def compute_euler_flow_loss(
    config,
    flow_predictor,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    reverse_pred_flow,
    sample,
    train_flow,
    device,
):
    """Compute Euler flow consistency loss."""
    if (
        config.lr_multi.eular_flow_loss > 0
        and config.model.flow.name in ["EulerFlowMLP", "EulerFlowMLPResidual"]
        and train_flow
    ):
        eular_flow_loss = 0
        for i in range(len(point_cloud_firsts)):
            point_cloud_first_forward = point_cloud_firsts[i][:, :3] + pred_flow[i]
            forward_reverse = flow_predictor(
                point_cloud_first_forward, sample["idx"][i] + 1, sample["total_frames"][i], QueryDirection.REVERSE
            )
            l2_error = torch.norm(pred_flow[i] + forward_reverse, dim=1)
            eular_flow_loss += torch.sigmoid(l2_error).mean()

            if len(reverse_pred_flow) > 0:
                point_cloud_next_reverse = point_cloud_nexts[i][:, :3] + reverse_pred_flow[i]
                reverse_forward = flow_predictor(
                    point_cloud_next_reverse, sample["idx"][i], sample["total_frames"][i], QueryDirection.FORWARD
                )
                l2_error = torch.norm(reverse_pred_flow[i] + reverse_forward, dim=1)
                eular_flow_loss += torch.sigmoid(l2_error).mean()
        eular_flow_loss = eular_flow_loss * config.lr_multi.eular_flow_loss
    else:
        eular_flow_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return eular_flow_loss


def compute_euler_mask_loss(
    config, mask_predictor, point_cloud_firsts, pred_flow, pred_mask, sample, train_mask, device
):
    """Compute Euler mask consistency loss."""
    if (
        config.lr_multi.eular_mask_loss > 0
        and config.model.mask.name in ["EulerMaskMLP", "EulerMaskMLPResidual"]
        and train_mask
    ):
        eular_mask_loss = 0
        for i in range(len(point_cloud_firsts)):
            point_cloud_first_forward = point_cloud_firsts[i][:, :3] + pred_flow[i]
            forward_mask = mask_predictor(point_cloud_first_forward, sample["idx"][i] + 1, sample["total_frames"][i])
            forward_mask = F.log_softmax(forward_mask, dim=1)
            target_mask = F.log_softmax(pred_mask[i].clone().permute(1, 0), dim=1)
            relative_entropy = torch.nn.functional.kl_div(
                forward_mask, target_mask, reduction="batchmean", log_target=True
            )
            eular_mask_loss += relative_entropy.mean()
        eular_mask_loss = eular_mask_loss * config.lr_multi.eular_mask_loss
    else:
        eular_mask_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return eular_mask_loss


def compute_kdtree_loss(
    config,
    loss_functions,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    longterm_pred_flow,
    sample,
    train_flow,
    device,
):
    """Compute KDTree distance loss."""
    if config.lr_multi.KDTree_loss > 0 and train_flow:
        kdtree_dist_loss = 0
        for i in range(len(point_cloud_firsts)):
            pred_second_point = point_cloud_firsts[i][:, :3] + pred_flow[i]
            kdtree_dist_loss += loss_functions["kdtree"](
                pred_second_point, sample["idx"][i] + 1, point_cloud_nexts[i][:, :3].to(device)
            )
        if len(longterm_pred_flow) > 0:
            for idx in longterm_pred_flow:
                pred_points = longterm_pred_flow[idx][:, :3]
                data_object = sample["self"][0].get_item(idx)
                real_points = data_object["point_cloud_first"][:, :3].to(device)
                kdtree_dist_loss += loss_functions["kdtree"](pred_points, data_object["idx"], real_points)
        kdtree_dist_loss = kdtree_dist_loss * config.lr_multi.KDTree_loss
    else:
        kdtree_dist_loss = torch.tensor(0.0, device=device)

    return kdtree_dist_loss


def compute_knn_loss(
    config,
    loss_functions,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    longterm_pred_flow,
    sample,
    cascade_flow_outs,
    train_flow,
    device,
):
    """Compute KNN distance loss."""
    if config.lr_multi.KNN_loss > 0 and train_flow:
        knn_dist_loss = 0
        for i in range(len(point_cloud_firsts)):
            ground_mask_first = None
            if config.dataset.name == "KITTISF_new":
                ground_mask_first = point_cloud_firsts[i][:, 1] < -1.4
                ground_mask_next = point_cloud_nexts[i][:, 1] < -1.4
                point_cloud_firsts_filtered = point_cloud_firsts[i][~ground_mask_first]
                point_cloud_nexts_filtered = point_cloud_nexts[i][~ground_mask_next]
                pred_flow_filtered = pred_flow[i][~ground_mask_first]
            else:
                point_cloud_firsts_filtered = point_cloud_firsts[i]
                point_cloud_nexts_filtered = point_cloud_nexts[i]
                pred_flow_filtered = pred_flow[i]
            pred_second_point_filtered = point_cloud_firsts_filtered[:, :3] + pred_flow_filtered
            l = loss_functions["knn"](
                pred_second_point_filtered, point_cloud_nexts_filtered[:, :3].to(device), forward_only=True
            )
            knn_dist_loss += l
            if cascade_flow_outs is not None:
                for cascade_flow in cascade_flow_outs:
                    if config.dataset.name == "KITTISF_new":
                        cascade_flow_filtered = cascade_flow[i][~ground_mask_first]
                        pred_second_point_cascade = (
                            point_cloud_firsts_filtered[:, :3].to(device) + cascade_flow_filtered
                        )
                        target_points = point_cloud_nexts_filtered[:, :3].to(device)
                    else:
                        pred_second_point_cascade = point_cloud_firsts[i][:, :3].to(device) + cascade_flow[i]
                        target_points = point_cloud_nexts[i][:, :3].to(device)
                    l = loss_functions["knn"](pred_second_point_cascade, target_points , forward_only=True)
                    knn_dist_loss += l
        if longterm_pred_flow is not None and len(longterm_pred_flow) > 0:
            for idx in longterm_pred_flow:
                pred_points = longterm_pred_flow[idx][:, :3]
                real_points = sample["self"][0].get_item(idx)["point_cloud_first"][:, :3].to(device)
                knn_dist_loss += loss_functions["knn"](real_points, pred_points, forward_only=True)

        knn_dist_loss = knn_dist_loss * config.lr_multi.KNN_loss
    else:
        knn_dist_loss = torch.tensor(0.0, device=device)

    return knn_dist_loss


def compute_regularization_loss(config, pred_flow, reverse_pred_flow, longterm_pred_flow, train_flow, device):
    """Compute L1 regularization loss."""
    if config.lr_multi.l1_regularization > 0 and train_flow:
        l1_regularization_loss = 0
        thres = config.loss.l1_regularization.threshold

        for flow in pred_flow:
            dist = torch.norm(flow, dim=1)
            dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
            l1_regularization_loss += dist.mean()

        for flow in reverse_pred_flow:
            dist = torch.norm(flow, dim=1)
            dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
            l1_regularization_loss += dist.mean()

        for idx in longterm_pred_flow:
            dist = torch.norm(longterm_pred_flow[idx], dim=1)
            dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
            l1_regularization_loss += dist.mean()

        l1_regularization_loss = l1_regularization_loss * config.lr_multi.l1_regularization
    else:
        l1_regularization_loss = torch.tensor(0.0, device=device)

    return l1_regularization_loss


def compute_all_losses(
    config,
    loss_functions,
    flow_predictor,
    mask_predictor,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    reverse_pred_flow,
    longterm_pred_flow,
    pred_mask,
    sample,
    step,
    scene_flow_smoothness_scheduler,
    train_flow,
    train_mask,
    device,
):
    """Compute all losses and return loss dictionary and total loss."""
    # Reconstruction losses
    rec_loss, rec_flow_loss, reconstructed_points = compute_reconstruction_loss(
        config, loss_functions, point_cloud_firsts, point_cloud_nexts, pred_mask, pred_flow, train_mask, device
    )

    # Flow losses
    flow_loss = compute_flow_losses(
        config,
        loss_functions,
        point_cloud_firsts,
        point_cloud_nexts,
        pred_flow,
        reverse_pred_flow,
        longterm_pred_flow,
        sample,
        device,
    )

    # Scene flow smoothness loss
    scene_flow_smooth_loss = compute_scene_flow_smoothness_loss(
        config, loss_functions, point_cloud_firsts, pred_mask, pred_flow, step, scene_flow_smoothness_scheduler, device
    )

    # Point smoothness loss
    point_smooth_loss = compute_point_smoothness_loss(
        config, loss_functions, point_cloud_firsts, pred_mask, step, device
    )

    # Euler flow loss
    eular_flow_loss = compute_euler_flow_loss(
        config,
        flow_predictor,
        point_cloud_firsts,
        point_cloud_nexts,
        pred_flow,
        reverse_pred_flow,
        sample,
        train_flow,
        device,
    )

    # Euler mask loss
    eular_mask_loss = compute_euler_mask_loss(
        config, mask_predictor, point_cloud_firsts, pred_flow, pred_mask, sample, train_mask, device
    )

    # KDTree loss
    kdtree_dist_loss = compute_kdtree_loss(
        config,
        loss_functions,
        point_cloud_firsts,
        point_cloud_nexts,
        pred_flow,
        longterm_pred_flow,
        sample,
        train_flow,
        device,
    )

    # KNN loss
    knn_dist_loss = compute_knn_loss(
        config,
        loss_functions,
        point_cloud_firsts,
        point_cloud_nexts,
        pred_flow,
        longterm_pred_flow,
        sample,
        train_flow,
        device,
    )

    # Regularization loss
    l1_regularization_loss = compute_regularization_loss(
        config, pred_flow, reverse_pred_flow, longterm_pred_flow, train_flow, device
    )

    # Create loss dictionary
    loss_dict = {
        "rec_loss": rec_loss,
        "flow_loss": flow_loss,
        "scene_flow_smooth_loss": scene_flow_smooth_loss,
        "rec_flow_loss": rec_flow_loss,
        "point_smooth_loss": point_smooth_loss,
        "eular_flow_loss": eular_flow_loss,
        "kdtree_dist_loss": kdtree_dist_loss,
        "knn_dist_loss": knn_dist_loss,
        "l1_regularization_loss": l1_regularization_loss,
        "eular_mask_loss": eular_mask_loss,
    }

    # Total loss
    total_loss = sum(loss_dict.values())

    return loss_dict, total_loss, reconstructed_points


def compute_invariance_loss(config, loss_functions, point_cloud_firsts, pred_mask, mask_predictor, step, device):
    """Compute invariance loss."""
    if config.lr_multi.invariance_loss > 0 and step > config.training.begin_train_invariance:
        from utils.forward_utils import augment_transform, forward_mask_prediction_general
        pc_aug_list = []
        for i in range(len(point_cloud_firsts)):
            pc_aug, _, _, _ = augment_transform(
                point_cloud_firsts[i],
                point_cloud_firsts[i],#should be point_cloud_nexts, i just input some meaningless data
                point_cloud_firsts[i],#should be flow, i just input some meaningless data, which will be ignored
                None,
                config.training.augment_params,
            )
            
            pc_aug_list.append(pc_aug)
        pred_mask_aug = forward_mask_prediction_general(pc_aug_list, mask_predictor)
        pred_mask_aug = torch.stack([mask.permute(1, 0) for mask in pred_mask_aug])
        pred_mask = torch.stack([mask.permute(1, 0) for mask in pred_mask])
        invariance_loss = loss_functions["invariance"](pred_mask, pred_mask_aug)
        invariance_loss = invariance_loss * config.lr_multi.invariance_loss
    else:
        invariance_loss = torch.tensor(0.0, device=device, requires_grad=True)
    return invariance_loss


def compute_all_losses_general(
    config,
    loss_functions,
    flow_predictor,
    mask_predictor,
    point_cloud_firsts,
    point_cloud_nexts,
    pred_flow,
    pred_mask,
    step,
    scene_flow_smoothness_scheduler,
    train_flow,
    train_mask,
    device,
    cascade_flow_outs=None,
):
    """
    Compute all losses for general training with new data structure.

    Args:
        config: Configuration object
        loss_functions: Dictionary of loss functions
        flow_predictor: Scene flow prediction model
        mask_predictor: Mask prediction model
        point_cloud_firsts: List of first frame point clouds
        point_cloud_nexts: List of next frame point clouds
        pred_flow: Predicted scene flows
        pred_mask: Predicted masks
        step: Current training step
        scene_flow_smoothness_scheduler: Scheduler for flow smoothness
        train_flow: Whether to train flow model
        train_mask: Whether to train mask model
        device: Device to run computations on
        cascade_flow_outs: Cascade flow outputs from flow predictor
    Returns:
        tuple: (loss_dict, total_loss, reconstructed_points)
            - loss_dict: Dictionary of individual losses
            - total_loss: Total combined loss
            - reconstructed_points: Reconstructed point clouds (if available)
    """
    scaled_pred_flow = [flow * config.loss.scale_flow_magnitude for flow in pred_flow]
    # Reconstruction losses
    rec_loss, rec_flow_loss, reconstructed_points = compute_reconstruction_loss(
        config, loss_functions, point_cloud_firsts, point_cloud_nexts, pred_mask.copy(), scaled_pred_flow, train_mask, device
    )

    # Scene flow smoothness loss
    scene_flow_smooth_loss = compute_scene_flow_smoothness_loss(
        config, loss_functions, point_cloud_firsts, pred_mask.copy(), scaled_pred_flow, step, scene_flow_smoothness_scheduler, device
    )

    # Point smoothness loss
    point_smooth_loss = compute_point_smoothness_loss(
        config, loss_functions, point_cloud_firsts, pred_mask.copy(), step, device
    )

    # KNN loss
    knn_dist_loss = compute_knn_loss(
        config=config,
        loss_functions=loss_functions,
        point_cloud_firsts=point_cloud_firsts,
        point_cloud_nexts=point_cloud_nexts,
        pred_flow=pred_flow,
        longterm_pred_flow=None,
        sample=None,
        cascade_flow_outs=cascade_flow_outs,
        train_flow=train_flow,
        device=device,
    )
    invariance_loss = compute_invariance_loss(
        config, loss_functions, point_cloud_firsts, pred_mask.copy(), mask_predictor, step, device
    )
    # Create loss dictionary
    loss_dict = {
        "rec_loss": rec_loss,
        "scene_flow_smooth_loss": scene_flow_smooth_loss,
        "point_smooth_loss": point_smooth_loss,
        "knn_dist_loss": knn_dist_loss,
        "invariance_loss": invariance_loss,
    }

    # Total loss
    total_loss = sum(loss_dict.values())

    return loss_dict, total_loss, reconstructed_points
