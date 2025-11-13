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
    assert not ((config.training.begin_train_flow !=0) and (config.training.begin_train_mask !=0)), f"your training is idle for the first {min(config.training.begin_train_flow, config.training.begin_train_mask)} steps?"
    train_flow = alter_scheduler.flow_train()
    train_mask = alter_scheduler.mask_train()
    
    # Override mask training if before begin_train_mask step
    if step < config.training.begin_train_mask:
        train_mask = False
    if step < config.training.begin_train_flow:
        train_flow = False
    
    return train_flow, train_mask


def set_model_training_modes(flow_predictor, mask_predictor, train_flow, train_mask):
    """Set training/evaluation modes for models.
    
    Args:
        flow_predictor: Scene flow model
        mask_predictor: Mask prediction model
        train_flow: Whether to train flow model
        train_mask: Whether to train mask model
    """
    if train_flow:
        flow_predictor.train()
    else:
        flow_predictor.eval()
        
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
    grad_ori = {}
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
        grad = {}
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[param_name] = torch.norm(param.grad).item()
                grad[param_name] = param.grad
                param.grad = None  # Clear gradients immediately
        
        grad_contributions[name] = grad_norms
        grad_ori[name] = grad
    return grad_contributions, grad_ori


def log_gradient_debug_info(config, writer, loss_dict, flow_predictor, mask_predictor, step):
    """Log gradient debugging information to TensorBoard."""
    if not config.vis.debug_grad:
        return
        
    with torch.no_grad():  # Avoid affecting actual gradient computation
        flow_grads, flow_grads_ori = compute_individual_gradients(
            loss_dict, flow_predictor, retain_graph=True)
        mask_grads, mask_grads_ori = compute_individual_gradients(
            loss_dict, mask_predictor, retain_graph=True)

    # Log to TensorBoard
    flow_grad_mean_dict = {}
    mask_grad_mean_dict = {}
    for loss_name in loss_dict:
        if loss_dict[loss_name].item() == 0:
            continue
        if len(flow_grads[loss_name].values()) != 0:
            
            flow_grad_mean_dict[loss_name] = sum(flow_grads[loss_name].values())/len(flow_grads[loss_name].values())
            writer.add_scalar(
                f"grad_norm/flow_{loss_name}", 
                flow_grad_mean_dict[loss_name],  # mean of all layer gradient norms
                step)

        if len(mask_grads[loss_name].values()) != 0:
            mask_grad_mean_dict[loss_name] = sum(mask_grads[loss_name].values())/len(mask_grads[loss_name].values())
    
            writer.add_scalar(
                f"grad_norm/mask_{loss_name}", 
                mask_grad_mean_dict[loss_name], 
                step)
    #print mask sum 
    writer.add_scalar(
        f"grad_norm/flow_sum", 
        sum(flow_grad_mean_dict.values()), 
        step)
    writer.add_scalar(
        f"grad_norm/mask_sum", 
        sum(mask_grad_mean_dict.values()), 
        step)

    #print the cosine similarity between different grad
    print("keys of flow_grads_ori: ", flow_grads_ori.keys())
    print("keys of mask_grads_ori: ", mask_grads_ori.keys())
    
    # Calculate flow gradient cosine similarities (only upper triangle to avoid duplicates)
    flow_grad_names = list(flow_grads_ori.keys())
    for i, flow_grad_name1 in enumerate(flow_grad_names):
        for j in range(i, len(flow_grad_names)):  # Start from i to include diagonal
            flow_grad_name2 = flow_grad_names[j]
            cosine_similarity_list = []
            for key in flow_grads_ori[flow_grad_name1]:
                if key not in flow_grads_ori[flow_grad_name2]:
                    continue
                cosine_similarity = torch.cosine_similarity(
                    flow_grads_ori[flow_grad_name1][key].flatten(), 
                    flow_grads_ori[flow_grad_name2][key].flatten(), 
                    dim=0
                )
                cosine_similarity_list.append(cosine_similarity)
            if len(cosine_similarity_list) == 0:
                continue
            cosine_similarity_list = torch.stack(cosine_similarity_list)
            mean_cosine_similarity = cosine_similarity_list.mean()
            writer.add_scalar(f"cosine_similarity/flow_{flow_grad_name1}_and_flow_{flow_grad_name2}", mean_cosine_similarity, step)
    
    # Calculate mask gradient cosine similarities (only upper triangle to avoid duplicates)
    mask_grad_names = list(mask_grads_ori.keys())
    for i, mask_grad_name1 in enumerate(mask_grad_names):
        for j in range(i, len(mask_grad_names)):  # Start from i to include diagonal
            mask_grad_name2 = mask_grad_names[j]
            cosine_similarity_list = []
            for key in mask_grads_ori[mask_grad_name1]:
                if key not in mask_grads_ori[mask_grad_name2]:
                    continue
                cosine_similarity = torch.cosine_similarity(
                    mask_grads_ori[mask_grad_name1][key].flatten(), 
                    mask_grads_ori[mask_grad_name2][key].flatten(), 
                    dim=0
                )
                cosine_similarity_list.append(cosine_similarity)
            if len(cosine_similarity_list) == 0:
                continue
            cosine_similarity_list = torch.stack(cosine_similarity_list)
            mean_cosine_similarity = cosine_similarity_list.mean()
            writer.add_scalar(f"cosine_similarity/mask_{mask_grad_name1}_and_mask_{mask_grad_name2}", mean_cosine_similarity, step)

def perform_optimization_step(config, total_loss, optimizer_flow, optimizer_mask, 
                             flow_predictor, mask_predictor, train_flow, train_mask,step):
    """Perform optimization step with gradient clipping and error handling.
    
    Args:
        config: Configuration object
        total_loss: Total loss tensor
        optimizer_flow: Flow model optimizer
        optimizer_mask: Mask model optimizer
        flow_predictor: Scene flow model
        mask_predictor: Mask model
        train_flow: Whether to train flow model
        train_mask: Whether to train mask model
        step: Current training step
    Returns:
        bool: True if optimization succeeded, False otherwise
    """
    # Backward pass
    try:
        total_loss.div(config.training.grad_accumulation_steps).backward()
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
        torch.nn.utils.clip_grad_norm_(flow_predictor.parameters(), 
                                     config.training.flow_model_grad_clip)
    if hasattr(config.training, "mask_model_grad_clip"):
        torch.nn.utils.clip_grad_norm_(mask_predictor.parameters(), 
                                     config.training.mask_model_grad_clip)
    
    # Optimizer steps
    if step % config.training.grad_accumulation_steps != 0:
        return True
    if train_flow:
        optimizer_flow.step()
        optimizer_flow.zero_grad()
    if train_mask:
        optimizer_mask.step()
        optimizer_mask.zero_grad()
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


def handle_evaluation(config, step, flow_predictor, mask_predictor, dataloader, device, writer):
    """Handle model evaluation and logging."""
    if step % config.training.eval_loop == 1:
        epe, miou, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = eval_model(
            flow_predictor, mask_predictor, dataloader, config, device, writer, step)
        
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

def handle_evaluation_general(config, step, flow_predictor, mask_predictor, val_flow_dataloader, val_mask_dataloader, train_dataloader,device, writer):
    """
    Handle model evaluation and logging with general data structure.
    
    Args:
        config: Configuration object
        step: Current training step
        flow_predictor: Scene flow prediction model
        mask_predictor: Mask prediction model
        val_flow_dataloader: Validation dataloader for flow evaluation
        val_mask_dataloader: Validation dataloader for mask evaluation
        device: Device to run computations on
        writer: TensorBoard writer for logging
    """
    if step % config.training.eval_loop == 1:
        print("Evaluating model at step ", step)
        eval_model_general(
            flow_predictor, 
            mask_predictor, 
            val_flow_dataloader, 
            val_mask_dataloader, 
            config, 
            device, 
            writer, 
            step,type="val",save_sample=True)
        # eval_model_general(
        #     flow_predictor, 
        #     mask_predictor, 
        #     train_dataloader, 
        #     train_dataloader,
        #     config, 
        #     device, 
        #     writer, 
        #     step,type="train",save_sample=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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
