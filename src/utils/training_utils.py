"""
Training utility functions for scene flow and mask prediction.

This module contains training-related utility functions that were
previously in main.py, organized for better modularity.
"""

import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn import functional as F

from eval import eval_model
from utils.visualization_utils import remap_instance_labels, color_mask
from visualize.open3d_func import visualize_vectors, update_vector_visualization
from eval import eval_model_general

def determine_training_modes(step, config, alter_scheduler):
    """Determine which models should be trained at current step.
    
    Args:
        step: Current training step
        config: Configuration object
        alter_scheduler: Alternating scheduler
        
    Returns:
        tuple: (train_flow, train_mask)
    """
    train_flow = alter_scheduler.flow_train()
    train_mask = alter_scheduler.mask_train()
    
    # Override mask training if before begin_train_mask step
    if step < config.training.begin_train_mask:
        train_mask = False
        train_flow = True
        
    return train_flow, train_mask


def set_model_training_modes(scene_flow_predictor, mask_predictor, train_flow, train_mask):
    """Set training/evaluation modes for models.
    
    Args:
        scene_flow_predictor: Scene flow model
        mask_predictor: Mask prediction model
        train_flow: Whether to train flow model
        train_mask: Whether to train mask model
    """
    if train_flow:
        scene_flow_predictor.train()
    else:
        scene_flow_predictor.eval()
        
    if train_mask:
        mask_predictor.train()
    else:
        mask_predictor.eval()


def compute_individual_gradients(loss_dict, model, retain_graph=False):
    """Compute individual gradient contributions for debugging.
    
    Args:
        loss_dict: Dictionary of losses
        model: Model to compute gradients for
        retain_graph: Whether to retain computation graph
        
    Returns:
        dict: Gradient contributions by loss name
    """
    grad_contributions = {}
    
    for name, loss in loss_dict.items():
        # Ensure loss is part of computation graph
        if not loss.requires_grad:
            loss = loss.clone().requires_grad_(True)
            
        model.zero_grad()
        
        # Add gradient checking
        try:
            loss.backward(retain_graph=retain_graph)
        except RuntimeError as e:
            print(f"Error computing gradient for {name}: {str(e)}")
            print(f"Loss value: {loss.item()}")
            continue
            
        grad_norms = {}
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[param_name] = torch.norm(param.grad).item()
                param.grad = None  # Clear gradients immediately
        
        grad_contributions[name] = grad_norms
    
    return grad_contributions


def log_gradient_debug_info(config, writer, loss_dict, scene_flow_predictor, mask_predictor, step):
    """Log gradient debugging information to TensorBoard."""
    if not config.vis.debug_grad:
        return
        
    with torch.no_grad():  # Avoid affecting actual gradient computation
        flow_grads = compute_individual_gradients(
            loss_dict, scene_flow_predictor, retain_graph=True)
        mask_grads = compute_individual_gradients(
            loss_dict, mask_predictor, retain_graph=True)

    # Log to TensorBoard
    for loss_name in loss_dict:
        if loss_dict[loss_name].item() == 0:
            continue
        writer.add_scalar(
            f"grad_norm/flow_{loss_name}", 
            sum(flow_grads[loss_name].values()),  # Sum of all layer gradient norms
            step)
        writer.add_scalar(
            f"grad_norm/mask_{loss_name}", 
            sum(mask_grads[loss_name].values()), 
            step)


def perform_optimization_step(config, total_loss, optimizer_flow, optimizer_mask, 
                             scene_flow_predictor, mask_predictor, train_flow, train_mask):
    """Perform optimization step with gradient clipping and error handling.
    
    Args:
        config: Configuration object
        total_loss: Total loss tensor
        optimizer_flow: Flow model optimizer
        optimizer_mask: Mask model optimizer
        scene_flow_predictor: Scene flow model
        mask_predictor: Mask model
        train_flow: Whether to train flow model
        train_mask: Whether to train mask model
        
    Returns:
        bool: True if optimization succeeded, False otherwise
    """
    # Zero gradients
    optimizer_flow.zero_grad()
    optimizer_mask.zero_grad()
    
    # Backward pass
    try:
        total_loss.backward()
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print("Error computing gradient for total loss")
        print(f"Total loss value: {total_loss.item()}")
        import traceback
        traceback.print_exc()
        return False
    
    # Clear gradients for models not being trained
    if not train_flow:
        optimizer_flow.zero_grad()
    if not train_mask:
        optimizer_mask.zero_grad()
    
    # Gradient clipping
    if hasattr(config.training, "flow_model_grad_clip"):
        torch.nn.utils.clip_grad_norm_(scene_flow_predictor.parameters(), 
                                     config.training.flow_model_grad_clip)
    if hasattr(config.training, "mask_model_grad_clip"):
        torch.nn.utils.clip_grad_norm_(mask_predictor.parameters(), 
                                     config.training.mask_model_grad_clip)
    
    # Optimizer steps
    if train_flow:
        optimizer_flow.step()
    if train_mask:
        optimizer_mask.step()
        
    return True


def handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint):
    """Handle periodic checkpoint saving."""
    if save_every_iters > 0 and step % save_every_iters == 0:
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        try:
            save_checkpoint(latest_path, step)
            tqdm.write(f"Saved checkpoint at step {step} -> {latest_path}")
        except Exception as e:
            tqdm.write(f"Failed to save checkpoint: {e}")


def handle_evaluation(config, step, scene_flow_predictor, mask_predictor, dataloader, device, writer):
    """Handle model evaluation and logging."""
    if step % config.training.eval_loop == 1:
        epe, miou, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = eval_model(
            scene_flow_predictor, mask_predictor, dataloader, config, device, writer, step)
        
        # Log evaluation metrics
        writer.add_scalar("val_epe", epe.mean().item(), step)
        writer.add_scalar("val_miou", miou.item(), step)
        
        if bg_epe is not None:
            writer.add_scalar("val_bg_epe", bg_epe.mean().item(), step)
        if fg_static_epe is not None:
            writer.add_scalar("val_fg_static_epe", fg_static_epe.mean().item(), step)
        if fg_dynamic_epe is not None:
            writer.add_scalar("val_fg_dynamic_epe", fg_dynamic_epe.mean().item(), step)
        if threeway_mean is not None:
            writer.add_scalar("val_threeway_mean", threeway_mean.mean().item(), step)

def handle_evaluation_general(config, step, scene_flow_predictor, mask_predictor, val_flow_dataloader, val_mask_dataloader, device, writer, downsample_factor):
    """
    Handle model evaluation and logging with general data structure.
    
    Args:
        config: Configuration object
        step: Current training step
        scene_flow_predictor: Scene flow prediction model
        mask_predictor: Mask prediction model
        val_flow_dataloader: Validation dataloader for flow evaluation
        val_mask_dataloader: Validation dataloader for mask evaluation
        device: Device to run computations on
        writer: TensorBoard writer for logging
        downsample_factor: Factor to downsample point clouds
    """
    if step % config.training.eval_loop == 1:
        epe, miou, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = eval_model_general(
            scene_flow_predictor, 
            mask_predictor, 
            val_flow_dataloader, 
            val_mask_dataloader, 
            config, 
            device, 
            writer, 
            step, 
            downsample_factor
            )
        
        # Log evaluation metrics
        writer.add_scalar("val_epe", epe.mean().item(), step)
        writer.add_scalar("val_miou", miou.item(), step)
        
        if bg_epe is not None:
            writer.add_scalar("val_bg_epe", bg_epe.mean().item(), step)
        if fg_static_epe is not None:
            writer.add_scalar("val_fg_static_epe", fg_static_epe.mean().item(), step)
        if fg_dynamic_epe is not None:
            writer.add_scalar("val_fg_dynamic_epe", fg_dynamic_epe.mean().item(), step)

        if threeway_mean is not None:
            writer.add_scalar("val_threeway_mean", threeway_mean.mean().item(), step)

def log_prediction_histograms(config, writer, pred_flow, pred_mask, step):
    """Log prediction histograms to TensorBoard."""
    if not config.vis.log_histogram:
        return
        
    # Get first prediction for logging
    flow_log = pred_flow[0].detach() if pred_flow[0].requires_grad else pred_flow[0]
    mask_log = pred_mask[0].detach() if pred_mask[0].requires_grad else pred_mask[0]
    
    flow_log = flow_log.cpu().numpy()
    mask_log = mask_log.cpu().numpy()
    
    # Process mask for different representations
    processed_mask = mask_log.copy()
    processed_mask = processed_mask / pow(processed_mask.std(), 0.5)
    
    # Numpy softmax
    min_value = np.min(processed_mask) * config.vis.histogram.softmax_scale
    softmaxed_mask = np.exp(processed_mask * config.vis.histogram.softmax_scale - min_value) / np.sum(
        np.exp(processed_mask * config.vis.histogram.softmax_scale - min_value), axis=0)
    
    # Log histograms
    writer.add_histogram("prediction_flow", flow_log, step)
    writer.add_histogram("prediction_mask", mask_log, step)
    writer.add_histogram("prediction_mask_processed", processed_mask, step)
    writer.add_histogram("prediction_mask_softmaxed", softmaxed_mask, step)
    
    # Log statistics
    writer.add_scalar("prediction_flow_mean", np.mean(np.linalg.norm(flow_log, axis=1)), step)
    writer.add_scalar("prediction_mask_mean", np.mean(np.linalg.norm(mask_log, axis=1)), step)
    writer.add_scalar("prediction_flow_std", np.std(np.linalg.norm(flow_log, axis=1)), step)
    writer.add_scalar("prediction_mask_processed_mean", np.mean(np.linalg.norm(processed_mask, axis=1)), step)
    writer.add_scalar("prediction_mask_softmaxed_mean", np.mean(np.linalg.norm(softmaxed_mask, axis=1)), step)
    writer.add_scalar("prediction_mask_std", np.std(np.linalg.norm(mask_log, axis=1)), step)
    writer.add_scalar("prediction_mask_processed_std", np.std(np.linalg.norm(processed_mask, axis=1)), step)
    writer.add_scalar("prediction_mask_softmaxed_std", np.std(np.linalg.norm(softmaxed_mask, axis=1)), step)


def handle_visualization(config, vis, pcd, gt_pcd, point_cloud_firsts, point_cloud_nexts, 
                        pred_flow, pred_mask, sample, first_iteration):
    """Handle Open3D visualization."""
    if not config.vis.show_window or point_cloud_firsts[0].shape[0] == 0:
        return first_iteration, None, None
        
    import open3d as o3d
    batch_idx = 0
    
    # Get data for visualization
    point_cloud_first = point_cloud_firsts[batch_idx].cpu().numpy()
    point_cloud_second = point_cloud_nexts[batch_idx].cpu().numpy()
    current_pred_flow = pred_flow[batch_idx].cpu().detach().numpy()
    current_pred_mask = pred_mask[batch_idx].cpu().detach()
    gt_flow = sample["flow"][batch_idx].cpu().detach().numpy()
    
    if config.vis.debug_flow_magnitude:
        print("mean length of pred_flow", np.mean(np.linalg.norm(current_pred_flow)))
    
    # Color processing
    pred_color = color_mask(current_pred_mask)
    gt_mask = sample["dynamic_instance_mask"][0]
    gt_mask = remap_instance_labels(gt_mask)
    gt_color = color_mask(F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(torch.float32))
    
    # Update point clouds
    pcd.points = o3d.utility.Vector3dVector(point_cloud_first)
    pcd.colors = o3d.utility.Vector3dVector(gt_color.numpy())
    
    gt_pcd.points = o3d.utility.Vector3dVector(point_cloud_second)
    gt_pcd.paint_uniform_color(config.vis.gt_point_color)
    
    if first_iteration:
        vis, lineset = visualize_vectors(
            point_cloud_first, current_pred_flow, vis=vis, color=pred_color.numpy())
        vis, lineset_gt = visualize_vectors(
            point_cloud_first, gt_flow, vis=vis, color=gt_color.numpy())
        first_iteration = False
    else:
        lineset = update_vector_visualization(
            lineset, point_cloud_first, current_pred_flow, color=pred_color.numpy())
        lineset_gt = update_vector_visualization(
            lineset_gt, point_cloud_first, gt_flow, color=gt_color.numpy())
        vis.update_geometry(lineset)
        vis.update_geometry(lineset_gt)
    
    vis.poll_events()
    vis.update_renderer()
    
    return first_iteration, lineset, lineset_gt


def cleanup_memory():
    """Clear CUDA cache and collect garbage."""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
