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
from typing import List
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

from SceneFlowZoo.dataloaders import (
    TorchFullFrameInputSequence,
    TorchFullFrameOutputSequence,
    TorchFullFrameOutputSequenceWithDistance,
)

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
        val_flow_dataset = Argoverse2CausalSceneFlow(
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
        val_flow_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    val_mask_dataloader = torch.utils.data.DataLoader(
        val_mask_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    return (dataset, dataloader, 
            val_flow_dataset, val_flow_dataloader, 
            val_mask_dataset, val_mask_dataloader)
    pass
def forward_scene_flow_general(sequences, flow_predictor)-> List[TorchFullFrameOutputSequence]:
    flow_out_seqs = flow_predictor.forward(sequences)
    return flow_out_seqs

def forward_mask_prediction_general(pc_tensors, mask_predictor):
    pred_masks = []
    for pc_tensor in pc_tensors:
        pc_tensor = pc_tensor.unsqueeze(0).contiguous()
        pred_mask = mask_predictor.forward(pc_tensor,pc_tensor)
        pred_mask = pred_mask.squeeze(0)
        pred_mask = pred_mask.permute(1,0)
        pred_masks.append(pred_mask)
    return pred_masks



def inference_models(flow_predictor,mask_predictor, sequences,downsample_factor):
    flow_out_seqs = forward_scene_flow_general(sequences, flow_predictor)
    pred_flow_full = [item.ego_flows.squeeze(0) for item in flow_out_seqs]
    # Forward pass for mask prediction
    #print the shape of pred_mask
    pred_flows_ego = [flow[seq.get_full_pc_gt_flow_mask(0)] for flow,seq in zip(pred_flow_full,sequences)]
    pred_flows_global = [flow_predictor.ego_to_global_flow(seq.get_full_ego_pc(0)[seq.get_full_pc_mask(0)],flow,seq.get_pc_transform_matrices(0)[1]) for flow,seq in zip(pred_flows_ego,sequences)]
    pc_tensors = [sequence.get_full_global_pc(0) for sequence in sequences]

    pc_tensors = [pc[seq.get_full_pc_gt_flow_mask(0),:] for pc,seq in zip(pc_tensors,sequences)]
    # Build first/next point clouds in ego frame (masked valid points)
    _, _, pc_tensors, _ = downsample_point_clouds(None, None, pc_tensors, None, downsample_factor)
    pred_masks = forward_mask_prediction_general(pc_tensors, mask_predictor)

    point_cloud_firsts = [seq.get_full_global_pc(0)[seq.get_full_pc_mask(0)] for seq in sequences]
    point_cloud_nexts = [seq.get_full_global_pc(1)[seq.get_full_pc_mask(1)] for seq in sequences]
    pred_flows_global,_,point_cloud_firsts,point_cloud_nexts = downsample_point_clouds(pred_flows_global,None,point_cloud_firsts,point_cloud_nexts,downsample_factor)

    return pred_flows_global,pred_masks,point_cloud_firsts,point_cloud_nexts

def downsample_point_clouds(pred_flow,pred_mask,point_cloud_firsts,point_cloud_nexts,downsample_factor):
    if pred_flow is not None:
        pred_flow = [item[::downsample_factor,:] for item in pred_flow]
    if pred_mask is not None:
        pred_mask = [item[:,::downsample_factor] for item in pred_mask]
    if point_cloud_firsts is not None:
        point_cloud_firsts = [item[::downsample_factor,:] for item in point_cloud_firsts]
    if point_cloud_nexts is not None:
        point_cloud_nexts = [item[::downsample_factor,:] for item in point_cloud_nexts]
    return pred_flow,pred_mask,point_cloud_firsts,point_cloud_nexts

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
    (dataset, dataloader,
    val_flow_dataset, val_flow_dataloader,
    val_mask_dataset, val_mask_dataloader) = create_dataloaders_general(config)
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
    (mask_predictor, flow_predictor, optimizer_flow, optimizer_mask, 
     alter_scheduler, scene_flow_smoothness_scheduler) = initialize_models_and_optimizers(config,None,device)
    
    # # Initialize loss functions
    loss_functions = initialize_loss_functions(config, device)
    
    # # Initialize visualization
    # if config.vis.show_window:
    #     vis, pcd, gt_pcd, reconstructed_pcd = initialize_visualization(config)
    
    # # Setup checkpointing
    checkpoint_dir, save_every_iters, step, resume, resume_path = setup_checkpointing(config, device)
    
    # # Load checkpoint if resuming
    step = load_checkpoint(config, flow_predictor, 
                          mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler)
    
    # Create checkpoint saver
    save_checkpoint = create_checkpoint_saver(checkpoint_dir, flow_predictor, mask_predictor,
                                            optimizer_flow, optimizer_mask, alter_scheduler, config)
    
    first_iteration = True

    # Main training loop
    with tqdm(dataloader, desc="Training", total=config.training.max_iter-step) as dataloader:
        tqdm.write("Starting training...")
        for framelists in dataloader:
            step += 1
            
            handle_evaluation_general(config, step, flow_predictor, mask_predictor, val_flow_dataloader, val_mask_dataloader, device, writer)


            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            #
            sequences = [TorchFullFrameInputSequence.from_frame_list(
                    idx=0,
                    frame_list=framelist,
                    pc_max_len=120000,  # Maximum point cloud points
                    loader_type=dataset.loader_type(),
                    allow_pc_slicing=False
                ).to(device) for framelist in framelists]
            # Determine training modes and set model states
            train_flow, train_mask = determine_training_modes(step, config, alter_scheduler)
            set_model_training_modes(flow_predictor, mask_predictor, train_flow, train_mask)  
            flow_predictor.to(device)
            mask_predictor.to(device)
            # Forward pamss for scene flow
            ego_gt_flows = [torch.from_numpy(framelist[0].flow.full_flow[framelist[0].pc.mask,:]).to(device).float() for framelist in framelists]
            global_gt_flows = [
                flow_predictor.global_to_ego_flow(
                    sequence.get_full_ego_pc(0)[framelist[0].pc.mask,:], 
                    ego_gt_flow, 
                    sequence.get_pc_transform_matrices(0)[1]
                ) for ego_gt_flow,sequence,framelist in zip(ego_gt_flows,sequences,framelists)]

            pred_flow,pred_mask,point_cloud_firsts,point_cloud_nexts = inference_models(flow_predictor,mask_predictor,sequences,config.dataset.AV2_SceneFlowZoo.downsample_factor)

            # Compute all losses
            loss_dict, total_loss, reconstructed_points = compute_all_losses_general(
                config=config, loss_functions=loss_functions, flow_predictor=flow_predictor, mask_predictor=mask_predictor,
                point_cloud_firsts=point_cloud_firsts, point_cloud_nexts=point_cloud_nexts, pred_flow=pred_flow, pred_mask=pred_mask, step=step, scene_flow_smoothness_scheduler=scene_flow_smoothness_scheduler,
                train_flow=train_flow, train_mask=train_mask, device=device)
                #def compute_all_losses_general(config, loss_functions, flow_predictor, mask_predictor,
                    #   point_cloud_firsts, point_cloud_nexts, pred_flow, pred_mask, step, scene_flow_smoothness_scheduler,
                    #   train_flow, train_mask, device):

                
            # Log to tensorboard
            if step % config.log.tensorboard_log_interval == 0:
                loss_log_dict = {name: loss.item() for name, loss in loss_dict.items()}
                loss_log_dict["total_loss"] = total_loss.item()
                writer.add_scalars("losses", loss_log_dict, step)
            
            # Log gradient debugging information
            log_gradient_debug_info(config, writer, loss_dict, flow_predictor, mask_predictor, step)
            
            # Perform optimization step
            optimization_success = perform_optimization_step(
                config, total_loss, optimizer_flow, optimizer_mask,
                flow_predictor, mask_predictor, train_flow, train_mask)
            
            if not optimization_success:
                continue


            alter_scheduler.step()
            
            # Handle checkpoint saving
            handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint)
            
            # Handle evaluation
            
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