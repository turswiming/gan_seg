"""
Main training script for scene flow and mask prediction.

This script handles the complete training pipeline including:
- Configuration loading and setup
- Dataset and model initialization
- Training loop with loss computation
- Visualization and logging
"""

# Standard library imports
import datetime
import argparse
import os
import shutil
import gc
import time
# Third party imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm
from pathlib import Path

# Local imports
from eval import evaluate_predictions, eval_model, eval_model_general
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders
from config.config import correct_datatype

# Import refactored modules
from utils.model_utils import (
    setup_device_and_training, initialize_models_and_optimizers, 
    initialize_loss_functions, initialize_visualization,
    setup_checkpointing, load_checkpoint, create_checkpoint_saver
)
from utils.forward_utils import forward_scene_flow, forward_mask_prediction
from utils.training_utils import (
    determine_training_modes, set_model_training_modes,
    log_gradient_debug_info, perform_optimization_step,
    handle_checkpoint_saving, handle_evaluation_general,
    log_prediction_histograms, handle_visualization, cleanup_memory
)
from losses.loss_functions import compute_all_losses, compute_all_losses_general

def create_dataloaders_general(config):
    """
    Create dataloaders for general training with AV2 SceneFlowZoo dataset.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        tuple: (dataloader, val_flow_dataloader, val_mask_dataloader)
            - dataloader: Main training dataloader
            - val_flow_dataloader: Validation dataloader for flow evaluation
            - val_mask_dataloader: Validation dataloader for mask evaluation
    """
    if config.dataset.name == "AV2_SceneFlowZoo":
        from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2CausalSceneFlow
        
        dataset = Argoverse2CausalSceneFlow(
            root_dir=Path(config.dataset.AV2_SceneFlowZoo.root_dir),
            subsequence_length=config.dataset.AV2_SceneFlowZoo.subsequence_length,
            sliding_window_step_size=config.dataset.AV2_SceneFlowZoo.sliding_window_step_size,
            with_ground=config.dataset.AV2_SceneFlowZoo.with_ground,
            use_gt_flow=config.dataset.AV2_SceneFlowZoo.use_gt_flow,
            eval_type=config.dataset.AV2_SceneFlowZoo.eval_type,
            expected_camera_shape=config.dataset.AV2_SceneFlowZoo.expected_camera_shape,
            eval_args=dict(),
            with_rgb=config.dataset.AV2_SceneFlowZoo.with_rgb,
            flow_data_path=Path(config.dataset.AV2_SceneFlowZoo.flow_data_path),
            range_crop_type="ego",
        )

    if config.dataset.val_name == "AV2_SceneFlowZoo_val":
        from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2CausalSceneFlow
        val_flow_config = config.dataset.AV2_SceneFlowZoo_val_flow
        val_dataset = Argoverse2CausalSceneFlow(
            root_dir=Path(val_flow_config.root_dir),
            with_ground=val_flow_config.with_ground,
            use_gt_flow=val_flow_config.use_gt_flow,
            eval_type=val_flow_config.eval_type,
            expected_camera_shape=val_flow_config.expected_camera_shape,
            eval_args=dict(output_path=val_flow_config.eval_args_output_path),
            with_rgb=val_flow_config.with_rgb,
            flow_data_path=Path(val_flow_config.flow_data_path),
            range_crop_type="ego",
            load_flow=True,
            load_boxes=False,
        )
        val_mask_config = config.dataset.AV2_SceneFlowZoo_val_mask
        val_mask_dataset = Argoverse2CausalSceneFlow(
            root_dir=Path(val_mask_config.root_dir),
            with_ground=val_mask_config.with_ground,
            use_gt_flow=val_mask_config.use_gt_flow,
            eval_type=val_mask_config.eval_type,
            expected_camera_shape=val_mask_config.expected_camera_shape,
            eval_args=dict(output_path=val_mask_config.eval_args_output_path),
            with_rgb=val_mask_config.with_rgb,
            flow_data_path=Path(val_mask_config.flow_data_path),
            range_crop_type="ego",
            load_flow=False,
            load_boxes=True,
        )
    collect_fn = lambda x: x
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=True, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    val_flow_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    val_mask_dataloader = torch.utils.data.DataLoader(
        val_mask_dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    return dataloader, val_flow_dataloader, val_mask_dataloader
    pass
def forward_scene_flow_general(sample_firsts, sample_nexts, scene_flow_predictor, downsample_factor):
    """
    Perform forward pass for scene flow prediction with general data structure.
    
    Args:
        sample_firsts: List of first frame samples
        sample_nexts: List of next frame samples  
        scene_flow_predictor: Scene flow prediction model
        downsample_factor: Factor to downsample point clouds
        
    Returns:
        pred_flow: Predicted scene flow vectors
    """
    device = next(scene_flow_predictor.parameters()).device
    visible_valid_mask_first = [item.pc.mask for item in sample_firsts] 
    visible_valid_mask_next = [item.pc.mask for item in sample_nexts]
    pc0 = [item.pc.full_global_pc.points[mask,:][::downsample_factor,:] for item, mask in zip(sample_firsts, visible_valid_mask_first)]
    pc1 = [item.pc.full_global_pc.points[mask,:][::downsample_factor,:] for item, mask in zip(sample_nexts, visible_valid_mask_next)]
    pc0s = [(torch.from_numpy(item).to(device).float(), torch.ones(item.shape[0], device=device).bool()) for item in pc0]
    pc1s = [(torch.from_numpy(item).to(device).float(), torch.ones(item.shape[0], device=device).bool()) for item in pc1]
    pc0_transforms = [(torch.from_numpy(item.pc.pose.sensor_to_ego.to_array()).to(device).float(), torch.from_numpy(item.pc.pose.ego_to_global.to_array()).to(device).float()) for item in sample_firsts] #(pc0_sensor_to_ego, pc0_ego_to_global)
    pred_flow = scene_flow_predictor._model_forward(pc0s, pc1s, pc0_transforms)
    return pred_flow

def forward_mask_prediction_general(sample_firsts, mask_predictor, downsample_factor):
    """
    Perform forward pass for mask prediction with general data structure.
    
    Args:
        sample_firsts: List of first frame samples
        mask_predictor: Mask prediction model
        downsample_factor: Factor to downsample point clouds
        
    Returns:
        list: Predicted mask tensors
    """
    device = next(mask_predictor.parameters()).device
    visible_valid_mask_first = [item.pc.mask for item in sample_firsts]
    pc0 = [item.pc.full_global_pc.points[mask,:][::downsample_factor,:] for item, mask in zip(sample_firsts, visible_valid_mask_first)]
    data = {'points': [torch.from_numpy(item).to(device).float() for item in pc0]}
    pred_mask = mask_predictor.forward_train(data)
    
    logits = pred_mask["pred_logits"]
    pred_masks = pred_mask["pred_masks"]
    logits = [item.permute(1, 0) for item in logits]
    return [pred_masks.to(device).float() ]

def main(config, writer):
    """
    Main training function.
    
    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    # Setup device and basic configurations
    device = setup_device_and_training()
    
    # Create dataloaders
    dataloader, val_flow_dataloader, val_mask_dataloader = create_dataloaders_general(config)
    # step = 0
    # for i in range(config.training.num_epochs):
    #     for sample in dataloader:
    #         step += 1
    #         if step > config.training.max_iter:
    #             break
    #         print(sample)
    #         pass
    #     pass
    #     for sample in val_dataloader:
    #         print(sample)
    #         pass
    # pass
    # # Initialize models, optimizers and schedulers
    (mask_predictor, scene_flow_predictor, optimizer_flow, optimizer_mask, 
     alter_scheduler, scene_flow_smoothness_scheduler) = initialize_models_and_optimizers(config,None,device)
    
    # # Initialize loss functions
    loss_functions = initialize_loss_functions(config, device)
    
    # # Initialize visualization
    # if config.vis.show_window:
    #     vis, pcd, gt_pcd, reconstructed_pcd = initialize_visualization(config)
    
    # # Setup checkpointing
    checkpoint_dir, save_every_iters, step, resume, resume_path = setup_checkpointing(config, device)
    
    # # Load checkpoint if resuming
    step = load_checkpoint(resume, resume_path, checkpoint_dir, device, scene_flow_predictor, 
                          mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler)
    
    # Create checkpoint saver
    save_checkpoint = create_checkpoint_saver(checkpoint_dir, scene_flow_predictor, mask_predictor,
                                            optimizer_flow, optimizer_mask, alter_scheduler, config)
    
    first_iteration = True

    # Main training loop
    with tqdm(dataloader, desc="Training", total=config.training.max_iter-step) as dataloader:
        tqdm.write("Starting training...")
        for sample in dataloader:
            step += 1


            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            # print(sample[0][0].keys())
            sample_firsts = [item[0] for item in sample]
            sample_nexts = [item[1] for item in sample]
            visible_valid_mask_first = [(item.pc.mask).reshape(-1) for item in sample_firsts]
            downsample_factor = config.dataset.AV2_SceneFlowZoo.downsample_factor
            point_cloud_firsts = [item.pc.full_global_pc.points[mask,:][::downsample_factor,:] for item, mask in zip(sample_firsts, visible_valid_mask_first)]
            visible_valid_mask_next = [(item.pc.mask).reshape(-1) for item in sample_nexts]
            point_cloud_nexts = [item.pc.full_global_pc.points[mask,:][::downsample_factor,:] for item, mask in zip(sample_nexts, visible_valid_mask_next)]
            point_cloud_firsts = [torch.from_numpy(item).to(device).float() for item in point_cloud_firsts]
            point_cloud_nexts = [torch.from_numpy(item).to(device).float() for item in point_cloud_nexts]

            # Determine training modes and set model states
            train_flow, train_mask = determine_training_modes(step, config, alter_scheduler)
            set_model_training_modes(scene_flow_predictor, mask_predictor, train_flow, train_mask)  
            scene_flow_predictor.to(device)
            mask_predictor.to(device)
            # Forward pass for scene flow
            pred_flow_object = forward_scene_flow_general(sample_firsts, sample_nexts, scene_flow_predictor,downsample_factor)
            pred_flow = [item.ego_flows.squeeze(0) for item in pred_flow_object]
            # Forward pass for mask prediction
            pred_mask = forward_mask_prediction_general(sample_firsts, mask_predictor,downsample_factor )
            
            # Compute all losses
            loss_dict, total_loss, reconstructed_points = compute_all_losses_general(
                config, loss_functions, scene_flow_predictor, mask_predictor,
                point_cloud_firsts, point_cloud_nexts, pred_flow, pred_mask, step, scene_flow_smoothness_scheduler,
                train_flow, train_mask, device)
                #def compute_all_losses_general(config, loss_functions, scene_flow_predictor, mask_predictor,
                    #   point_cloud_firsts, point_cloud_nexts, pred_flow, pred_mask, step, scene_flow_smoothness_scheduler,
                    #   train_flow, train_mask, device):

                
            # Log to tensorboard
            if step % config.log.tensorboard_log_interval == 0:
                loss_log_dict = {name: loss.item() for name, loss in loss_dict.items()}
                loss_log_dict["total_loss"] = total_loss.item()
                writer.add_scalars("losses", loss_log_dict, step)
            
            # Log gradient debugging information
            log_gradient_debug_info(config, writer, loss_dict, scene_flow_predictor, mask_predictor, step)
            
            # Perform optimization step
            optimization_success = perform_optimization_step(
                config, total_loss, optimizer_flow, optimizer_mask,
                scene_flow_predictor, mask_predictor, train_flow, train_mask)
            
            if not optimization_success:
                continue


            alter_scheduler.step()
            
            # Handle checkpoint saving
            handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint)
            
            # Handle evaluation
            handle_evaluation_general(config, step, scene_flow_predictor, mask_predictor, val_flow_dataloader, val_mask_dataloader, device, writer,downsample_factor)
            
            # Clear memory cache
            if step % config.hardware.clear_cache_interval == 0:
                cleanup_memory()
            
            # Log prediction histograms
            log_prediction_histograms(config, writer, pred_flow, pred_mask, step)
            
            # Handle visualization
            # first_iteration, lineset, lineset_gt = handle_visualization(
            #     config, vis, pcd, gt_pcd, point_cloud_firsts, point_cloud_nexts,
            #     pred_flow, pred_mask, sample, first_iteration)
            
            pass  # end loop 

    # Cleanup
    if config.vis.show_window:
        vis.destroy_window()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", 
                       help="Path to the config file")
    
    args, unknown = parser.parse_known_args()
    config_obj = load_config_with_inheritance(args.config)

    # print_config(config_obj)
    
    cli_opts = OmegaConf.from_cli()
    
    # Merge configs
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)
    print_config(config)

    # Setup logging directory
    if config.log.dir == "":
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.log.dir = f"../outputs/exp/{time_str}"
    
    writer = SummaryWriter(log_dir=config.log.dir)
    
    # Save config and code
    save_config_and_code(config, config.log.dir)

    # Start training
    main(config, writer)